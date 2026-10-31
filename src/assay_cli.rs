//! Explicit supplied-observation and native-model Assay commands.
//! Complete bounded reports use a separate protocol from the frozen robot schema.
use std::{io::{Read, Write}, path::{Path, PathBuf}, process::ExitCode};
use clap::{Args, ArgMatches, FromArgMatches};
use serde::{Deserialize, Serialize};
use crate::{assay::{self, AssayError, BootstrapOptions, Dataset, EvalOptions, Limits},
    calibration::{ShiftAssessment, ShiftPolicy}, canonjson, error::ErrorCode,
    execution_identity::Sha256Digest, local_io::{self, Destination}, receipt::CommitmentKey};

mod schema;
mod native;
mod ner;
#[cfg(test)] mod tests;

const PROTOCOL: &str = "fnlp.assay.scored";
const SCHEMA_VERSION: u32 = 1;

#[derive(Args)]
struct Common {
    /// Describe this command's exact data and report protocol, without reading files.
    #[arg(long, exclusive = true)]
    schema: bool,
    /// Strict suite manifest with execution identity, byte pin and split membership.
    #[arg(long, required_unless_present = "schema")]
    suite: Option<PathBuf>,
    /// Versioned binary-event observation NDJSON; '-' reads stdin.
    #[arg(long, default_value = "-")]
    observations: PathBuf,
    /// Existing protected file containing exactly 32 secret bytes; never an argv secret.
    #[arg(long, required_unless_present = "schema")]
    key_file: Option<PathBuf>,
    /// Nonsecret identifier for the supplied commitment key.
    #[arg(long, required_unless_present = "schema")]
    key_id: Option<String>,
    /// Nonsecret namespace binding every private commitment in this local suite.
    #[arg(long, required_unless_present = "schema")]
    namespace: Option<String>,
    /// New protected report file; '-' or omission writes the complete report to stdout.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 1024 * 1024)]
    max_suite_bytes: usize,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    max_input_bytes: usize,
    #[arg(long, default_value_t = 64 * 1024)]
    max_line_bytes: usize,
    #[arg(long, default_value_t = 100_000)]
    max_rows: usize,
    /// Complete report, including protocol envelope and newline.
    #[arg(long, default_value_t = 4 * 1024 * 1024)]
    max_output_bytes: usize,
    #[arg(long, default_value_t = 1000)]
    max_bootstrap_resamples: usize,
    #[arg(long, default_value_t = 10_000_000)]
    max_bootstrap_draws: u64,
}

#[derive(Args)]
struct Metrics {
    /// Minimum eligible split size; smaller populations yield insufficient_data.
    #[arg(long, default_value_t = 20)]
    minimum_samples: usize,
    #[arg(long, default_value_t = 0.8)]
    acceptance_threshold: f64,
    #[arg(long, default_value_t = 200)]
    bootstrap_resamples: usize,
    #[arg(long, default_value_t = 0.95)]
    confidence_level: f64,
    /// Explicit deterministic bootstrap seed; unrelated to model sampling.
    #[arg(long, default_value_t = 0)]
    bootstrap_seed: u64,
}
impl Metrics {
    fn options(&self) -> EvalOptions {
        EvalOptions { minimum_samples: self.minimum_samples, acceptance_threshold: self.acceptance_threshold,
            bootstrap: BootstrapOptions { resamples: self.bootstrap_resamples,
                confidence_level: self.confidence_level, seed: self.bootstrap_seed } }
    }
}

#[derive(Args)]
pub(crate) struct Eval {
    #[command(flatten)] common: Common,
    #[command(flatten)] metrics: Metrics,
    /// Complete report previously produced by `calibrate scored`.
    #[arg(long, requires_all = ["expected_calibration_sha256", "as_of", "shift_assessment"])]
    calibration: Option<PathBuf>,
    /// Independent exact-byte pin for the complete calibration report file.
    #[arg(long, requires = "calibration")]
    expected_calibration_sha256: Option<String>,
    #[arg(long, requires = "calibration", value_name = "YYYY-MM-DD")]
    as_of: Option<String>,
    #[arg(long, requires = "calibration", value_parser = ["in-distribution", "detected"])]
    shift_assessment: Option<String>,
    /// Required only with --shift-assessment detected; never inferred from the scores.
    #[arg(long, requires = "calibration")]
    shift_indicator: Option<String>,
}

#[derive(Args)]
pub(crate) struct Calibrate {
    #[command(flatten)] common: Common,
    #[command(flatten)] metrics: Metrics,
    #[arg(long, required_unless_present = "schema", value_parser = ["temperature", "isotonic"])]
    method: Option<String>,
    #[arg(long, required_unless_present = "schema", value_name = "YYYY-MM-DD")]
    valid_from: Option<String>,
    #[arg(long, required_unless_present = "schema", value_name = "YYYY-MM-DD")]
    valid_through: Option<String>,
    #[arg(long, required_unless_present = "schema", value_parser = ["raw-scores-uncalibrated", "conservative-abstain"])]
    shift_policy: Option<String>,
}

#[derive(Args)]
pub(crate) struct Qualify {
    #[command(flatten)] common: Common,
    #[command(flatten)] metrics: Metrics,
    /// Baseline suite over the same frozen evaluation units and human labels.
    #[arg(long, required_unless_present = "schema")]
    baseline_suite: Option<PathBuf>,
    /// Baseline observation NDJSON; its exact bytes must match the baseline suite pin.
    #[arg(long, required_unless_present = "schema")]
    baseline_observations: Option<PathBuf>,
    /// Strict caller-declared comparison gates; no inferred acceptance policy.
    #[arg(long, required_unless_present = "schema")]
    gates: Option<PathBuf>,
    /// Caller-declared host scope; this command does not observe native execution.
    #[arg(long, required_unless_present = "schema")]
    host_scope: Option<String>,
}

pub(crate) enum AssayCommand { Scored(ScoredCommand), Native(native::Command), NativeNer(ner::Command) }
pub(crate) enum ScoredCommand { Eval(Eval), Calibrate(Calibrate), Qualify(Qualify) }

pub(crate) fn definitions() -> Vec<clap::Command> {
    [("eval", "Evaluate explicitly supplied binary-event observations; no model replay or activation",
        Eval::augment_args(clap::Command::new("scored"))),
     ("calibrate", "Fit and evaluate a scoped calibration artifact from disjoint supplied observations",
        Calibrate::augment_args(clap::Command::new("scored"))),
     ("qualify", "Compare paired supplied observations under explicit local gates; no activation authority",
        Qualify::augment_args(clap::Command::new("scored")))]
        .into_iter().map(|(name, about, scored)| {
            let command = clap::Command::new(name).about(about).subcommand_required(true)
                .subcommand(scored.about(about).after_help(
                    "All output is one complete canonical JSON report under fnlp.assay.scored v1. Scores and human labels are supplied by the caller; this command does not run a model or grant activation authority. Use --schema for the strict wire contract."));
            match name {
                "eval" => command.about("Evaluate supplied observations or replay pinned native INT4 tasks")
                    .subcommand(native::definition("eval")).subcommand(ner::definition("eval")),
                "calibrate" => command.about("Fit supplied observations or actual native INT4 reliability on calibration rows")
                    .subcommand(native::definition("calibrate")),
                "qualify" => command.about("Compare paired supplied observations or native INT4 models under explicit local gates")
                    .subcommand(native::definition("qualify")).subcommand(ner::definition("qualify")),
                _ => command,
            }
        }).collect()
}

impl AssayCommand {
    pub(crate) fn recognizes(name: &str) -> bool { matches!(name, "eval" | "calibrate" | "qualify") }
    pub(crate) fn from_matches(name: &str, matches: &ArgMatches) -> Result<Self, clap::Error> {
        match matches.subcommand() {
            Some(("native-int4", matches)) if Self::recognizes(name) =>
                native::Command::from_matches(name, matches).map(Self::Native),
            Some(("native-ner-int4", matches)) if matches!(name, "eval" | "qualify") =>
                ner::Command::from_matches(name, matches).map(Self::NativeNer),
            Some(("scored", _)) => ScoredCommand::from_matches(name, matches).map(Self::Scored),
            _ => Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand,
                "an explicit supported Assay backend is required")),
        }
    }
    pub(crate) fn run(self, input: &mut impl Read, output: &mut impl Write, errors: &mut impl Write) -> ExitCode {
        match self {
            Self::Scored(command) => command.run(input, output, errors),
            Self::Native(command) => command.run(input, output, errors),
            Self::NativeNer(command) => command.run(input, output, errors),
        }
    }
}

impl ScoredCommand {
    pub(crate) fn from_matches(name: &str, matches: &ArgMatches) -> Result<Self, clap::Error> {
        let Some(("scored", matches)) = matches.subcommand() else {
            return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, "explicit scored backend required"));
        };
        match name {
            "eval" => Eval::from_arg_matches(matches).map(Self::Eval),
            "calibrate" => Calibrate::from_arg_matches(matches).map(Self::Calibrate),
            "qualify" => Qualify::from_arg_matches(matches).map(Self::Qualify),
            _ => Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, "unknown Assay command")),
        }
    }
    fn common(&self) -> &Common {
        match self { Self::Eval(c) => &c.common, Self::Calibrate(c) => &c.common, Self::Qualify(c) => &c.common }
    }
    fn operation(&self) -> &'static str {
        match self { Self::Eval(_) => "eval", Self::Calibrate(_) => "calibrate", Self::Qualify(_) => "qualify" }
    }
    fn options(&self) -> EvalOptions {
        match self { Self::Eval(c) => c.metrics.options(), Self::Calibrate(c) => c.metrics.options(),
            Self::Qualify(c) => c.metrics.options() }
    }
    fn validate(&self) -> Result<Limits, Failure> {
        let limits = self.common().limits()?;
        self.options().validate(limits)?;
        match self {
            Self::Eval(c) => { let _ = c.apply_options()?; },
            Self::Calibrate(c) => { let _ = c.fit_options()?; },
            Self::Qualify(c) => c.validate()?,
        }
        Ok(limits)
    }
    fn evaluate(&self, dataset: &Dataset, key: &CommitmentKey, namespace: &str, limits: Limits) -> Result<Vec<u8>, Failure> {
        match self {
            Self::Eval(c) => match c.apply_options()? {
                None => encode(self.operation(), &assay::evaluate(dataset, key, namespace, self.options(), limits)?, limits),
                Some((pin, options)) => {
                    let path = c.calibration.as_deref().ok_or_else(invalid_options)?;
                    let bytes = read_file(path, limits.max_output_bytes)?;
                    if Sha256Digest::of_bytes(&bytes) != pin { return Err(AssayError::InputDigest.into()); }
                    let stored: StoredCalibration = strict_json(&bytes).map_err(|_| AssayError::InvalidCalibration)?;
                    if stored.schema_version != SCHEMA_VERSION || stored.protocol != PROTOCOL || stored.operation != "calibrate"
                        || stored.evidence != "provided_observations" || stored.activation_authority {
                        return Err(AssayError::InvalidCalibration.into());
                    }
                    let report = assay::render(&stored.report, limits)?;
                    let result = assay::calibrate::apply(&report, dataset, key, namespace, options, self.options(), limits)?;
                    encode(self.operation(), &result, limits)
                }
            },
            Self::Calibrate(c) => encode(self.operation(), &assay::calibrate::fit(dataset, key, namespace,
                c.fit_options()?, self.options(), limits)?, limits),
            Self::Qualify(c) => {
                let gates = assay::qualify::parse_gates(&read_file(c.gates.as_deref()
                    .ok_or_else(invalid_options)?, limits.max_manifest_bytes)?, limits)?;
                let manifest = read_file(c.baseline_suite.as_deref().ok_or_else(invalid_options)?, limits.max_manifest_bytes)?;
                let observations = read_file(c.baseline_observations.as_deref().ok_or_else(invalid_options)?, limits.max_input_bytes)?;
                let baseline = Dataset::parse(&manifest, &observations, limits)?;
                let report = assay::qualify::compare(dataset, &baseline, key, namespace,
                    c.host_scope.as_deref().ok_or_else(invalid_options)?, gates, self.options(), limits)?;
                encode(self.operation(), &report, limits)
            }
        }
    }
    fn emit(&self, input: &mut impl Read, output: &mut impl Write) -> Result<(), Failure> {
        if self.common().schema {
            let bytes = assay::render(&schema::document(self.operation()), Limits::default())?;
            return output.write_all(&bytes).and_then(|()| output.flush()).map_err(|_| output_failure());
        }
        let limits = self.validate()?;
        if !local_io::PROFILE_SUPPORTED { return Err(Failure::new(ErrorCode::AdmissionOrResourceLimit, "protected_file_profile_unavailable")); }
        let common = self.common();
        let destination = common.output.as_deref().filter(|p| p.as_os_str() != "-")
            .map(Destination::prepare).transpose().map_err(file_failure)?;
        let key = read_key(common.key_file.as_deref().ok_or_else(invalid_options)?,
            common.key_id.as_deref().ok_or_else(invalid_options)?)?;
        let manifest = read_file(common.suite.as_deref().ok_or_else(invalid_options)?, limits.max_manifest_bytes)?;
        let observations = if common.observations.as_os_str() == "-" { read_bytes(input, limits.max_input_bytes)? }
            else { read_file(&common.observations, limits.max_input_bytes)? };
        let dataset = Dataset::parse(&manifest, &observations, limits)?;
        let bytes = self.evaluate(&dataset, &key, common.namespace.as_deref().ok_or_else(invalid_options)?, limits)?;
        match destination {
            Some(destination) => destination.stage(&bytes).and_then(|stage| stage.publish()).map_err(file_failure),
            None => output.write_all(&bytes).and_then(|()| output.flush()).map_err(|_| output_failure()),
        }
    }
    pub(crate) fn run(self, input: &mut impl Read, output: &mut impl Write, errors: &mut impl Write) -> ExitCode {
        match self.emit(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                if let Ok(mut bytes) = canonjson::canonical_bytes(&failure) {
                    bytes.push(b'\n'); let _ = errors.write_all(&bytes).and_then(|()| errors.flush());
                }
                failure.code.as_process_exit()
            }
        }
    }
}

impl Common {
    fn limits(&self) -> Result<Limits, Failure> {
        let limits = Limits { max_manifest_bytes: self.max_suite_bytes, max_input_bytes: self.max_input_bytes,
            max_line_bytes: self.max_line_bytes, max_rows: self.max_rows, max_output_bytes: self.max_output_bytes,
            max_bootstrap_resamples: self.max_bootstrap_resamples, max_bootstrap_draws: self.max_bootstrap_draws };
        limits.validate()?;
        for path in [self.suite.as_deref(), self.key_file.as_deref()] {
            if path.is_none_or(|p| p.as_os_str().is_empty() || p.as_os_str() == "-") { return Err(invalid_options()); }
        }
        if self.observations.as_os_str().is_empty() || self.output.as_ref().is_some_and(|p| p.as_os_str().is_empty()) {
            return Err(invalid_options());
        }
        for value in [self.key_id.as_deref(), self.namespace.as_deref()] {
            if value.is_none_or(|s| !identifier(s)) { return Err(invalid_options()); }
        }
        Ok(limits)
    }
}
impl Eval {
    fn apply_options(&self) -> Result<Option<(Sha256Digest, assay::calibrate::ApplyOptions)>, Failure> {
        if self.calibration.is_none() {
            if self.expected_calibration_sha256.is_some() || self.as_of.is_some() || self.shift_assessment.is_some() || self.shift_indicator.is_some() {
                return Err(invalid_options());
            }
            return Ok(None);
        }
        if self.calibration.as_ref().is_some_and(|p| p.as_os_str().is_empty() || p.as_os_str() == "-") { return Err(invalid_options()); }
        let pin = Sha256Digest::from_hex(self.expected_calibration_sha256.as_deref().ok_or_else(invalid_options)?)
            .map_err(|_| invalid_options())?;
        let observed_date = assay::calibrate::parse_date(self.as_of.as_deref().ok_or_else(invalid_options)?)?;
        let shift = match self.shift_assessment.as_deref() {
            Some("in-distribution") if self.shift_indicator.is_none() => ShiftAssessment::InDistribution,
            Some("detected") => {
                let indicator = self.shift_indicator.as_deref().ok_or_else(invalid_options)?;
                if indicator.trim().is_empty() || indicator.len() > 1024 || indicator.chars().any(|c| c.is_control() || matches!(c, '\u{2028}' | '\u{2029}')) {
                    return Err(invalid_options());
                }
                ShiftAssessment::Detected { indicator: indicator.to_owned() }
            }
            _ => return Err(invalid_options()),
        };
        Ok(Some((pin, assay::calibrate::ApplyOptions { observed_date, shift })))
    }
}
impl Calibrate {
    fn fit_options(&self) -> Result<assay::calibrate::FitOptions, Failure> {
        use assay::calibrate::{FitMethod, FitOptions, parse_date};
        let method = match self.method.as_deref() { Some("temperature") => FitMethod::Temperature,
            Some("isotonic") => FitMethod::Isotonic, _ => return Err(invalid_options()) };
        let valid_from = parse_date(self.valid_from.as_deref().ok_or_else(invalid_options)?)?;
        let valid_through = parse_date(self.valid_through.as_deref().ok_or_else(invalid_options)?)?;
        let validity = crate::calibration::ValidityWindow::new(valid_from, valid_through).map_err(|_| invalid_options())?;
        let shift_policy = match self.shift_policy.as_deref() {
            Some("raw-scores-uncalibrated") => ShiftPolicy::RawScoresUncalibrated,
            Some("conservative-abstain") => ShiftPolicy::ConservativeAbstain,
            _ => return Err(invalid_options()),
        };
        Ok(FitOptions { method, validity, shift_policy })
    }
}

impl Qualify {
    fn validate(&self) -> Result<(), Failure> {
        for path in [self.baseline_suite.as_deref(), self.baseline_observations.as_deref(), self.gates.as_deref()] {
            if path.is_none_or(|p| p.as_os_str().is_empty() || p.as_os_str() == "-") {
                return Err(invalid_options());
            }
        }
        if self.host_scope.as_deref().is_none_or(|scope| !identifier(scope)) { return Err(invalid_options()); }
        Ok(())
    }
}

#[derive(Serialize)]
struct Envelope<'a, T: Serialize> {
    schema_version: u32, protocol: &'static str, operation: &'static str,
    evidence: &'static str, activation_authority: bool, report: &'a T,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredCalibration {
    schema_version: u32, protocol: String, operation: String, evidence: String,
    activation_authority: bool, report: serde_json::Value,
}
fn encode(operation: &'static str, report: &impl Serialize, limits: Limits) -> Result<Vec<u8>, Failure> {
    assay::render(&Envelope { schema_version: SCHEMA_VERSION, protocol: PROTOCOL, operation,
        evidence: "provided_observations", activation_authority: false, report }, limits).map_err(Into::into)
}
fn strict_json<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, Failure> {
    let text = std::str::from_utf8(bytes).map_err(|_| AssayError::Json)?;
    let value = canonjson::parse_str_with_limits(text, canonjson::ParseLimits { max_depth: 32, max_string_bytes: 4096 })
        .map_err(|_| AssayError::Json)?;
    serde_json::from_value(value).map_err(|_| AssayError::Json.into())
}
fn identifier(value: &str) -> bool {
    value.len() <= 256 && value.as_bytes().first().is_some_and(u8::is_ascii_alphanumeric)
        && value.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'/' | b'-'))
}
fn read_file(path: &Path, cap: usize) -> Result<Vec<u8>, Failure> {
    let mut file = local_io::open_document(path).map_err(|_| Failure::new(ErrorCode::InputDecodeOrParse, "regular_input_open_failed"))?;
    read_bytes(&mut file, cap)
}
fn read_bytes(input: &mut impl Read, cap: usize) -> Result<Vec<u8>, Failure> {
    let mut bytes = Vec::new(); let mut buffer = [0_u8; 8192];
    loop {
        let next = cap.checked_sub(bytes.len()).and_then(|n| n.checked_add(1)).ok_or(AssayError::InputLimit)?.min(buffer.len());
        let read = input.read(&mut buffer[..next]).map_err(|_| Failure::new(ErrorCode::InputDecodeOrParse, "input_read_failed"))?;
        if read == 0 { return Ok(bytes); }
        if read > cap - bytes.len() { return Err(AssayError::InputLimit.into()); }
        bytes.try_reserve(read).map_err(|_| AssayError::Allocation)?; bytes.extend_from_slice(&buffer[..read]);
    }
}
fn read_key(path: &Path, key_id: &str) -> Result<CommitmentKey, Failure> {
    let mut file = local_io::open_key(path).map_err(|_| Failure::new(ErrorCode::AdmissionOrResourceLimit, "protected_key_refused"))?;
    let mut secret = [0_u8; 32];
    let result = (|| {
        file.read_exact(&mut secret).map_err(|_| Failure::new(ErrorCode::InputDecodeOrParse, "key_requires_exactly_32_bytes"))?;
        let mut extra = [0_u8; 1];
        if file.read(&mut extra).map_err(|_| Failure::new(ErrorCode::InputDecodeOrParse, "key_read_failed"))? != 0 {
            return Err(Failure::new(ErrorCode::InputDecodeOrParse, "key_requires_exactly_32_bytes"));
        }
        CommitmentKey::new(key_id, secret).map_err(|_| invalid_options())
    })();
    secret.fill(0); result
}

#[derive(Debug, Serialize)]
struct Failure { schema_version: u32, protocol: &'static str, event: &'static str, code: ErrorCode, reason: &'static str }
impl Failure {
    fn new(code: ErrorCode, reason: &'static str) -> Self {
        Self { schema_version: SCHEMA_VERSION, protocol: PROTOCOL, event: "assay_error", code, reason }
    }
}
impl From<AssayError> for Failure { fn from(error: AssayError) -> Self { Self::new(error.exit_code(), error.code()) } }
fn invalid_options() -> Failure { Failure::new(ErrorCode::Usage, "invalid_assay_options") }
fn output_failure() -> Failure { Failure::new(ErrorCode::Generic, "output_write_failed") }
fn file_failure(error: local_io::LocalIoError) -> Failure {
    Failure::new(ErrorCode::AdmissionOrResourceLimit, match error {
        local_io::LocalIoError::PublicationUncertain => "output_may_be_published_but_sync_or_cleanup_failed",
        local_io::LocalIoError::AlreadyExists => "destination_already_exists",
        _ => "protected_output_refused",
    })
}
