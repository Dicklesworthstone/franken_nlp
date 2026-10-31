//! Explicit native classification replay over a frozen labeled text suite.
use super::*;
use crate::{native_engine::{int4::PackedInt4Kernel, simd::Int8KernelKind,
    portable_int8::ProjectionWork, strict_int4::{Int4MemoryRequirement, Int4Work,
        weights::snapshot::{Int4SnapshotPin, Int4SnapshotRequirements}}},
    tasks::ir::TaskBudget};

mod schema;
#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)] mod tests;

const NATIVE_PROTOCOL: &str = "fnlp.assay.native-int4";
const MIB: u64 = 1024 * 1024;
const ALLOCATOR_BYTES: u64 = 64 * MIB;
const CLEANUP_BYTES: u64 = 65_536;

#[derive(Args)]
struct Arguments {
    #[command(flatten)] common: Common,
    #[command(flatten)] metrics: Metrics,
    /// Explicit candidate snapshot; no active-model lookup or source fallback.
    #[arg(long, required_unless_present = "schema")]
    snapshot: Option<PathBuf>,
    /// Independently retained model-content SHA-256 from a trusted conversion.
    #[arg(long, required_unless_present = "schema")]
    expected_model_sha256: Option<String>,
    #[arg(long, default_value_t = 128)]
    group_width: usize,
    /// Process memory-ledger ceiling, including both models for qualification.
    #[arg(long, required_unless_present = "schema")]
    memory_mib: Option<u64>,
    /// Exact integer kernel; unsupported requests refuse before input IO.
    #[arg(long, default_value_t = Int8KernelKind::Scalar)]
    kernel: Int8KernelKind,
    #[arg(long, default_value_t = 2048)]
    context_tokens: usize,
    /// Complete candidate continuation depth, including its scored EOS.
    #[arg(long, default_value_t = 16)]
    max_candidate_tokens: usize,
    /// Admission cap for the complete label set; multilabel uses one independent head per label.
    #[arg(long, default_value_t = 2)]
    max_labels: usize,
    /// Complete native task result for one document, separate from the report.
    #[arg(long, default_value_t = 1_048_576)]
    max_result_bytes: usize,
    /// Compiler/request headroom; dataset, native state and reports are additional.
    #[arg(long, default_value_t = 128)]
    preparation_mib: u64,
    /// One cooperative deadline across reads, both loads, replay and publication.
    #[arg(long, default_value_t = 3600)]
    timeout_seconds: u64,
    /// Checkpoint ceiling for each bounded snapshot load, separate from replay.
    #[arg(long, default_value_t = 1_000_000_000)]
    snapshot_checkpoints: u64,
    /// One replay checkpoint budget, shared by both qualification arms.
    #[arg(long, default_value_t = 1_000_000_000)]
    max_checkpoints: u64,
    /// Whole replay ceilings; neither documents nor the baseline renew them.
    #[arg(long, default_value_t = 131_072)]
    max_forward_positions: u64,
    #[arg(long, default_value_t = 100_000_000)]
    max_projected_logits: u64,
    #[arg(long, default_value_t = 100_000_000_000)]
    max_attention_pairs: u64,
    #[arg(long, default_value_t = 10_000_000_000_000)]
    max_dot_products: u64,
    #[arg(long, default_value_t = 1_000_000_000_000_000)]
    max_multiply_accumulates: u64,
}

#[derive(Args)]
struct Comparison {
    /// Independently pinned baseline snapshot over exactly the same frozen cases.
    #[arg(long, required_unless_present = "schema")]
    baseline_snapshot: Option<PathBuf>,
    #[arg(long, required_unless_present = "schema")]
    baseline_model_sha256: Option<String>,
    #[arg(long, default_value_t = 128)]
    baseline_group_width: usize,
    /// Strict native quality and reliability gate document; no inferred policy.
    #[arg(long, required_unless_present = "schema")]
    gates: Option<PathBuf>,
    /// Caller-declared nonsecret scope, distinct from observed model execution.
    #[arg(long, required_unless_present = "schema")]
    host_scope: Option<String>,
    /// Complete independently pinned native calibration report for the baseline.
    #[arg(long, requires_all = ["expected_baseline_calibration_sha256", "baseline_as_of", "baseline_shift_assessment"])]
    baseline_calibration: Option<PathBuf>,
    #[arg(long, requires = "baseline_calibration")]
    expected_baseline_calibration_sha256: Option<String>,
    #[arg(long, requires = "baseline_calibration", value_name = "YYYY-MM-DD")]
    baseline_as_of: Option<String>,
    #[arg(long, requires = "baseline_calibration", value_parser = ["in-distribution", "detected"])]
    baseline_shift_assessment: Option<String>,
    #[arg(long, requires = "baseline_calibration")]
    baseline_shift_indicator: Option<String>,
}

#[derive(Args, Default)]
struct Application {
    /// Complete authenticated report from `calibrate native-int4`, over this exact frozen suite.
    #[arg(long, requires_all = ["expected_calibration_sha256", "as_of", "shift_assessment"])]
    calibration: Option<PathBuf>,
    #[arg(long, requires = "calibration")]
    expected_calibration_sha256: Option<String>,
    #[arg(long, requires = "calibration", value_name = "YYYY-MM-DD")]
    as_of: Option<String>,
    #[arg(long, requires = "calibration", value_parser = ["in-distribution", "detected"])]
    shift_assessment: Option<String>,
    #[arg(long, requires = "calibration")]
    shift_indicator: Option<String>,
}

#[derive(Args)]
struct Fit {
    /// Isotonic supports endpoint weights; temperature refuses 0/1 without clipping.
    #[arg(long, default_value = "isotonic", value_parser = ["temperature", "isotonic"])]
    method: String,
    #[arg(long, required_unless_present = "schema", value_name = "YYYY-MM-DD")]
    valid_from: Option<String>,
    #[arg(long, required_unless_present = "schema", value_name = "YYYY-MM-DD")]
    valid_through: Option<String>,
    #[arg(long, required_unless_present = "schema", value_parser = ["raw-scores-uncalibrated", "conservative-abstain"])]
    shift_policy: Option<String>,
}
struct ApplicationInput { path: PathBuf, pin: Sha256Digest, options: assay::calibrate::ApplyOptions }

pub(crate) struct Command {
    args: Arguments, comparison: Option<Comparison>, fit: Option<Fit>, application: Option<Application>,
}

pub(super) fn definition(operation: &str) -> clap::Command {
    let command = Arguments::augment_args(clap::Command::new("native-int4")
        .about(match operation {
            "qualify" => "Replay pinned candidate and baseline classifiers on identical labeled text",
            "calibrate" => "Fit authenticated reliability calibration from actual native replay",
            _ => "Run a pinned native INT4 classifier on a frozen labeled text suite",
        }))
        .mut_arg("suite", |arg| arg.help("Strict binary, multiclass or multilabel suite with exact case-file pin, labels and split membership"))
        .mut_arg("observations", |arg| arg.long("cases")
            .help("Frozen labeled-text NDJSON; '-' reads stdin; caller probabilities are rejected"))
        .mut_arg("max_output_bytes", |arg| arg.help("Complete output report cap, also applied independently to each imported calibration report"))
        .after_help("Output is one complete canonical JSON report under fnlp.assay.native-int4 v1. Native execution is observed; human gold labels and host scope remain caller declarations. Reports grant no model activation authority. Use --schema for exact data contracts.");
    match operation {
        "calibrate" => Fit::augment_args(command),
        "qualify" => Comparison::augment_args(Application::augment_args(command)),
        _ => Application::augment_args(command),
    }
}
impl Command {
    pub(super) fn from_matches(operation: &str, matches: &ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { args: Arguments::from_arg_matches(matches)?,
            comparison: if operation == "qualify" { Some(Comparison::from_arg_matches(matches)?) } else { None },
            fit: if operation == "calibrate" { Some(Fit::from_arg_matches(matches)?) } else { None },
            application: if operation == "calibrate" { None } else { Some(Application::from_arg_matches(matches)?) } })
    }
    fn operation(&self) -> &'static str {
        if self.fit.is_some() { "calibrate" } else if self.comparison.is_some() { "qualify" } else { "eval" }
    }
    fn artifact_count(&self) -> usize {
        usize::from(self.application.as_ref().is_some_and(|options| options.calibration.is_some()))
            + usize::from(self.comparison.as_ref().is_some_and(|options| options.baseline_calibration.is_some()))
    }
    fn validate(&self) -> Result<Limits, Failure> {
        let limits = self.args.common.limits()?;
        self.args.metrics.options().validate(limits)?;
        if let Some(fit) = &self.fit {
            if self.application.is_some() || self.comparison.is_some() { return Err(invalid_options()); }
            assay::native_classify::calibrate::validate_fit_options(fit.options()?, limits)?;
        }
        if let Some(application) = &self.application { let _ = application.input()?; }
        let args = &self.args;
        model_input(args.snapshot.as_deref(), args.expected_model_sha256.as_deref(), args.group_width)?;
        mib(args.memory_mib.ok_or_else(invalid_options)?)?; mib(args.preparation_mib)?;
        if !(2..=64).contains(&args.max_candidate_tokens) || args.max_candidate_tokens >= args.context_tokens
            || !(1..=assay::native_classify::MAX_LABELS).contains(&args.max_labels)
            || !(1..=64 * 1024 * 1024).contains(&args.max_result_bytes)
            || !(1..=86_400).contains(&args.timeout_seconds)
            || !(2..=1_000_000_000_000).contains(&args.snapshot_checkpoints)
            || !(2..=1_000_000_000_000).contains(&args.max_checkpoints)
            || !(1..=16 * 1024 * 1024).contains(&args.max_forward_positions)
            || args.max_projected_logits == 0 || args.max_attention_pairs == 0
            || args.max_dot_products == 0 || args.max_multiply_accumulates == 0 {
            return Err(invalid_options());
        }
        Int4MemoryRequirement::for_context(args.context_tokens, args.group_width).map_err(|_| invalid_options())?;
        PackedInt4Kernel::detect(args.kernel).map_err(|_| invalid_options())?;
        if let Some(comparison) = &self.comparison {
            let _ = comparison.application_input()?;
            model_input(comparison.baseline_snapshot.as_deref(), comparison.baseline_model_sha256.as_deref(),
                comparison.baseline_group_width)?;
            Int4MemoryRequirement::for_context(args.context_tokens, comparison.baseline_group_width)
                .map_err(|_| invalid_options())?;
            if comparison.gates.as_deref().is_none_or(|path| !file_path(path))
                || comparison.host_scope.as_deref().is_none_or(|scope| !identifier(scope)) {
                return Err(invalid_options());
            }
        }
        Ok(limits)
    }
    fn emit(&self, input: &mut impl Read, output: &mut impl Write) -> Result<(), Failure> {
        if self.args.common.schema {
            let bytes = assay::render(&schema::document(self.operation()), Limits::default())?;
            return output.write_all(&bytes).and_then(|()| output.flush()).map_err(|_| output_failure());
        }
        let limits = self.validate()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (limits, input, output);
          Err(Failure::new(ErrorCode::Generic, "native_runtime_unavailable")) }
    }
    pub(super) fn run(self, input: &mut impl Read, output: &mut impl Write, errors: &mut impl Write) -> ExitCode {
        match self.emit(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(mut failure) => {
                failure.protocol = NATIVE_PROTOCOL;
                if let Ok(mut bytes) = canonjson::canonical_bytes(&failure) {
                    bytes.push(b'\n'); let _ = errors.write_all(&bytes).and_then(|()| errors.flush());
                }
                failure.code.as_process_exit()
            }
        }
    }
}
impl Arguments {
    fn work(&self) -> Int4Work {
        Int4Work { forward_positions: self.max_forward_positions, projected_logits: self.max_projected_logits,
            attention_pairs: self.max_attention_pairs, projections: ProjectionWork {
                dot_products: self.max_dot_products, multiply_accumulates: self.max_multiply_accumulates } }
    }
    fn task(&self) -> Result<TaskBudget, Failure> {
        let native = Int4MemoryRequirement::for_context(self.context_tokens, self.group_width)
            .map_err(|_| invalid_options())?;
        let prompt = self.context_tokens.checked_sub(self.max_candidate_tokens)
            .and_then(|count| count.checked_add(1)).ok_or_else(invalid_options)?;
        Ok(TaskBudget { max_input_tokens: u32::try_from(prompt)
                .map_err(|_| invalid_options())?,
            max_output_tokens: self.max_candidate_tokens as u32,
            max_output_bytes: self.max_result_bytes as u64, max_grammar_states: 4096,
            max_kv_bytes: native.kv_bytes })
    }
}
impl Fit {
    fn options(&self) -> Result<assay::calibrate::FitOptions, Failure> {
        use assay::calibrate::{FitMethod, FitOptions, parse_date};
        let method = match self.method.as_str() {
            "temperature" => FitMethod::Temperature, "isotonic" => FitMethod::Isotonic,
            _ => return Err(invalid_options()),
        };
        let valid_from = parse_date(self.valid_from.as_deref().ok_or_else(invalid_options)?)?;
        let valid_through = parse_date(self.valid_through.as_deref().ok_or_else(invalid_options)?)?;
        let validity = crate::calibration::ValidityWindow::new(valid_from, valid_through).map_err(|_| invalid_options())?;
        let shift_policy = match self.shift_policy.as_deref() {
            Some("raw-scores-uncalibrated") => crate::calibration::ShiftPolicy::RawScoresUncalibrated,
            Some("conservative-abstain") => crate::calibration::ShiftPolicy::ConservativeAbstain,
            _ => return Err(invalid_options()),
        };
        Ok(FitOptions { method, validity, shift_policy })
    }
}
impl Application {
    fn input(&self) -> Result<Option<ApplicationInput>, Failure> {
        application_input(self.calibration.as_deref(), self.expected_calibration_sha256.as_deref(),
            self.as_of.as_deref(), self.shift_assessment.as_deref(), self.shift_indicator.as_deref())
    }
}
impl Comparison {
    fn application_input(&self) -> Result<Option<ApplicationInput>, Failure> {
        application_input(self.baseline_calibration.as_deref(), self.expected_baseline_calibration_sha256.as_deref(),
            self.baseline_as_of.as_deref(), self.baseline_shift_assessment.as_deref(), self.baseline_shift_indicator.as_deref())
    }
}
fn application_input(path: Option<&Path>, digest: Option<&str>, date: Option<&str>,
    assessment: Option<&str>, indicator: Option<&str>) -> Result<Option<ApplicationInput>, Failure> {
    let Some(path) = path else {
        if [digest, date, assessment, indicator].iter().any(Option::is_some) { return Err(invalid_options()); }
        return Ok(None);
    };
    if !file_path(path) { return Err(invalid_options()); }
    let pin = Sha256Digest::from_hex(digest.ok_or_else(invalid_options)?).map_err(|_| invalid_options())?;
    let observed_date = assay::calibrate::parse_date(date.ok_or_else(invalid_options)?)?;
    let shift = match assessment {
        Some("in-distribution") if indicator.is_none() => crate::calibration::ShiftAssessment::InDistribution,
        Some("detected") => {
            let indicator = indicator.ok_or_else(invalid_options)?;
            if indicator.trim().is_empty() || indicator.len() > 1024
                || indicator.chars().any(|ch| ch.is_control() || matches!(ch, '\u{2028}' | '\u{2029}')) {
                return Err(invalid_options());
            }
            crate::calibration::ShiftAssessment::Detected { indicator: indicator.to_owned() }
        }
        _ => return Err(invalid_options()),
    };
    Ok(Some(ApplicationInput { path: path.to_owned(), pin,
        options: assay::calibrate::ApplyOptions { observed_date, shift } }))
}
fn file_path(path: &Path) -> bool { !path.as_os_str().is_empty() && path.as_os_str() != "-" }
fn model_input(path: Option<&Path>, digest: Option<&str>, width: usize)
    -> Result<(Int4SnapshotPin, Int4SnapshotRequirements), Failure> {
    if path.is_none_or(|path| !file_path(path)) { return Err(invalid_options()); }
    let digest = digest.ok_or_else(invalid_options)?;
    Sha256Digest::from_hex(digest).map_err(|_| invalid_options())?;
    let mut model_content_sha256 = [0_u8; 32];
    for (index, byte) in model_content_sha256.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&digest[index * 2..index * 2 + 2], 16).map_err(|_| invalid_options())?;
    }
    let need = Int4SnapshotRequirements::for_group_width(width).map_err(|_| invalid_options())?;
    Ok((Int4SnapshotPin { group_width: width, model_content_sha256 }, need))
}
fn mib(value: u64) -> Result<u64, Failure> {
    value.checked_mul(MIB).filter(|&bytes| bytes != 0).ok_or_else(invalid_options)
}
fn sum(values: &[u64]) -> Result<u64, Failure> {
    values.iter().try_fold(0_u64, |total, &value| total.checked_add(value).ok_or_else(invalid_options))
}
