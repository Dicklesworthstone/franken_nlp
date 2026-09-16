//! Model-free, bounded redaction CLI. Rules-only scope is explicit; this module
//! does not load weights or substitute rules for a requested native model pass.
use std::{collections::BTreeSet, convert::Infallible, io::{Read, Write}, path::PathBuf, process::ExitCode};
mod local_io;
use local_io::{Destination, LocalIoError};
use clap::{Args, ValueEnum};
use serde::Serialize;
use crate::{canonjson, error::ErrorCode, tasks::redact::{
    PiiKind, RedactError, RedactionRequest, redact_rules,
    actions::{RedactionAction, RedactionResult},
    pipeline::{LeakReport, PipelineError},
    pseudonym::{PseudonymBudget, PseudonymKey, Pseudonyms},
    union::DetectedDocument,
}};

#[derive(Clone, Copy, Eq, PartialEq, ValueEnum)]
enum Action { Mask, Placeholder, Pseudonymize }
#[derive(Clone, Copy, ValueEnum)]
enum Rule { Email, Phone, Url, IpAddress, CreditCard, Date }
impl Rule {
    fn kind(self) -> PiiKind { match self {
        Self::Email => PiiKind::Email, Self::Phone => PiiKind::Phone, Self::Url => PiiKind::Url,
        Self::IpAddress => PiiKind::IpAddress, Self::CreditCard => PiiKind::CreditCard, Self::Date => PiiKind::Date,
    } }
}

#[derive(Args)]
pub(crate) struct RedactCommand {
    /// UTF-8 document file, or '-' for stdin. A single invocation is one document/job.
    #[arg(default_value = "-", value_name = "DOCUMENT")]
    document: PathBuf,
    /// Explicitly select deterministic rules only; no person/org/location NER is run.
    #[arg(long, required = true)]
    rules_only: bool,
    #[arg(long, value_enum, default_value = "placeholder")]
    action: Action,
    /// Replace the default rule set. Calendar dates are opt-in.
    #[arg(long, value_enum, value_delimiter = ',', default_value = "email,phone,url,ip-address,credit-card")]
    rules: Vec<Rule>,
    /// Emit the canonical result envelope rather than the exact transformed text.
    #[arg(long)]
    json: bool,
    /// Rerun the selected rules on transformed text (enabled by default).
    #[arg(long, conflicts_with = "no_verify")]
    verify: bool,
    /// Explicitly opt out of the residual scan; result metadata remains unverified.
    #[arg(long)]
    no_verify: bool,
    /// Read raw 32..4096-byte secret key material from stdin, never argv.
    #[arg(long, conflicts_with = "key_file")]
    key_stdin: bool,
    /// Raw private key file (Linux x86_64/aarch64, owner-only file in a 0700 parent).
    #[arg(long, conflicts_with = "key_stdin")]
    key_file: Option<PathBuf>,
    /// New protected output file; '-' keeps stdout. No overwrite is permitted.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
    /// New protected coordinate-map file, never stdout; requires a private 0700 parent.
    #[arg(long)]
    map_out: Option<PathBuf>,
    /// Nonsecret rotation identifier; required for pseudonymization.
    #[arg(long)]
    key_id: Option<String>,
    /// Nonempty pseudonym domain; required for pseudonymization.
    #[arg(long)]
    namespace: Option<String>,
    /// Saved nonsecret HMAC key commitment; mismatch fails before document input.
    #[arg(long)]
    expected_key_commitment: Option<String>,
    /// Use 256-bit pseudonyms from the start, without a truncated-value preflight.
    #[arg(long)]
    full_digest: bool,
    #[arg(long, default_value_t = 1024 * 1024)]
    max_input_bytes: usize,
    /// Bounds the complete result envelope, not just its transformed text.
    #[arg(long, default_value_t = 4 * 1024 * 1024)]
    max_output_bytes: usize,
    #[arg(long, default_value_t = 4096)]
    max_detections: usize,
    /// Conservative work units available to each complete rule scan.
    #[arg(long, default_value_t = 128 * 1024 * 1024)]
    max_rule_work: u64,
    #[arg(long, default_value_t = 16_384)]
    max_preflight_values: usize,
    #[arg(long, default_value_t = 8 * 1024 * 1024)]
    max_preflight_bytes: usize,
}

pub(crate) fn definition() -> clap::Command {
    RedactCommand::augment_args(clap::Command::new("redact"))
        .about("Redact one document with explicitly selected rules; no model required")
        .after_help("Verification checks only the declared rule set, not all PII. Unicode obfuscations can be missed. Pseudonyms are not anonymization.\nExamples:\n  fnlp redact --rules-only --verify document.txt\n  fnlp redact --rules-only --action mask --json -\n  fnlp redact --rules-only --action pseudonymize --key-stdin --key-id rotation-1 --namespace job-1 document.txt < secret.key\nFile keys, -o and --map-out currently require Linux x86_64/aarch64, procfs and a private 0700 parent. Final paths must not exist. Maps contain coordinates, not original values.\n128-bit mode preflights this complete document before output. Use --full-digest when a larger job cannot be preflighted as a whole.")
}

#[derive(Serialize)]
struct Failure {
    schema_version: u32,
    event: &'static str,
    code: ErrorCode,
    reason: &'static str,
    map_published: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    leak_report: Option<LeakReport>,
}
impl Failure {
    fn new(code: ErrorCode, reason: &'static str) -> Self {
        Self { schema_version: 1, event: "redact_error", code, reason, map_published: false, leak_report: None }
    }
}
impl From<LocalIoError> for Failure {
    fn from(error: LocalIoError) -> Self {
        let reason = match error {
            LocalIoError::UnsupportedProfile => "private_file_profile_unavailable",
            LocalIoError::InvalidPath => "private_file_path_refused",
            LocalIoError::PrivateParentRequired => "owner_only_0700_parent_required",
            LocalIoError::UnsafeFile => "private_file_authority_refused",
            LocalIoError::AlreadyExists => "destination_already_exists",
            LocalIoError::Io => "local_file_io_failed",
            LocalIoError::StageExhausted => "private_stage_names_exhausted",
            LocalIoError::PublicationUncertain => "output_may_be_published_but_sync_or_cleanup_failed",
        };
        Self::new(ErrorCode::AdmissionOrResourceLimit, reason)
    }
}
impl From<RedactError> for Failure {
    fn from(error: RedactError) -> Self {
        use RedactError::*;
        let (code, reason) = match error {
            InvalidOptions | MissingPolicy | MissingKey => (ErrorCode::Usage, "invalid_redaction_options_or_key"),
            WorkBudget => (ErrorCode::BudgetOrTimeout, "rule_work_budget"),
            InputBudget | DetectionBudget | CandidateBudget | OutputBudget | AllocationRefused =>
                (ErrorCode::AdmissionOrResourceLimit, "redaction_resource_budget"),
            KeyMismatch => (ErrorCode::StructuredTaskNoResult, "key_commitment_mismatch"),
            Collision => (ErrorCode::StructuredTaskNoResult, "pseudonym_collision"),
            InvalidSpan | InvalidNerEvidence | VerificationResidual { .. } =>
                (ErrorCode::StructuredTaskNoResult, "redaction_no_valid_result"),
            Serialization => (ErrorCode::Generic, "result_serialization"),
        };
        Self::new(code, reason)
    }
}
impl From<PipelineError<Infallible>> for Failure {
    fn from(error: PipelineError<Infallible>) -> Self {
        match error {
            PipelineError::Redaction(error) => error.into(),
            PipelineError::Model(never) => match never {},
            PipelineError::Residual(report) => Self { leak_report: Some(report),
                ..Self::new(ErrorCode::StructuredTaskNoResult, "residual_rule_findings") },
        }
    }
}

impl RedactCommand {
    fn validate(&self) -> Result<(), Failure> {
        let usage = |reason| Failure::new(ErrorCode::Usage, reason);
        if !self.rules_only || self.rules.is_empty() { return Err(usage("explicit_rules_only_scope_required")); }
        if self.key_stdin && self.document.as_os_str() == "-" { return Err(usage("key_stdin_conflicts_with_document_stdin")); }
        if self.map_out.as_ref().is_some_and(|p| p.as_os_str() == "-")
            || self.key_file.as_ref().is_some_and(|p| p.as_os_str() == "-") {
            return Err(usage("key_and_map_paths_cannot_be_stdout_or_stdin"));
        }
        if !local_io::PROFILE_SUPPORTED && (self.key_file.is_some() || self.map_out.is_some() || self.file_output().is_some()) {
            return Err(LocalIoError::UnsupportedProfile.into());
        }
        let needs_key = self.action == Action::Pseudonymize;
        if needs_key {
            if (self.key_stdin == self.key_file.is_some()) || self.key_id.as_ref().is_none_or(|s| s.is_empty() || s.len() > 128
                || !s.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)))
                || self.namespace.as_ref().is_none_or(|s| s.is_empty() || s.len() > 256) {
                return Err(usage("pseudonymization_requires_one_key_source_key_id_and_namespace"));
            }
        } else if self.key_stdin || self.key_file.is_some() || self.key_id.is_some() || self.namespace.is_some()
            || self.expected_key_commitment.is_some() || self.full_digest {
            return Err(usage("key_options_require_pseudonymize_action"));
        }
        if self.expected_key_commitment.as_ref().is_some_and(|s| s.len() != 64
            || !s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))) {
            return Err(usage("invalid_key_commitment_encoding"));
        }
        const BYTE_CAP: usize = 64 * 1024 * 1024;
        if self.max_input_bytes > BYTE_CAP || self.max_output_bytes == 0 || self.max_output_bytes > BYTE_CAP
            || !(1..=16_384).contains(&self.max_detections) || self.max_rule_work == 0
            || self.max_preflight_values > 1_000_000 || self.max_preflight_bytes > BYTE_CAP {
            return Err(usage("invalid_resource_limits"));
        }
        Ok(())
    }
    fn file_output(&self) -> Option<&PathBuf> {
        self.output.as_ref().filter(|p| p.as_os_str() != "-")
    }
    fn request(&self) -> RedactionRequest {
        let mut request = RedactionRequest::default();
        request.rules.enabled = self.rules.iter().map(|r| r.kind()).collect::<BTreeSet<_>>();
        request.actions.default_action = match self.action {
            Action::Mask => RedactionAction::Mask, Action::Placeholder => RedactionAction::Placeholder,
            Action::Pseudonymize => RedactionAction::Pseudonymize,
        };
        request.actions.include_map = self.map_out.is_some();
        request.actions.expected_key_commitment = self.expected_key_commitment.clone();
        // Input admission remains separately bounded below. Verification may
        // need to scan an expanded transformed document, up to its output cap.
        request.rule_budget.max_input_bytes = self.max_input_bytes.max(self.max_output_bytes);
        request.rule_budget.max_detections = self.max_detections;
        request.rule_budget.max_work = self.max_rule_work;
        request.edit_budget.max_regions = self.max_detections;
        request.edit_budget.max_output_bytes = self.max_output_bytes;
        request.verify = !self.no_verify;
        request
    }
}

fn read_document(input: &mut impl Read, cap: usize) -> Result<String, Failure> {
    // Never read the entire stream and check afterwards. At most cap+1 bytes
    // are consumed; the extra byte proves overflow without a partial result.
    let mut data = Vec::new();
    let mut scratch = [0_u8; 8192];
    loop {
        let width = scratch.len().min(cap.saturating_sub(data.len()) + 1);
        let count = match input.read(&mut scratch[..width]) {
            Ok(0) => break,
            Ok(n) if n <= width => n,
            Ok(_) => return Err(Failure::new(ErrorCode::Generic, "invalid_input_reader")),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(Failure::new(ErrorCode::InputDecodeOrParse, "document_read_failed")),
        };
        if count > cap - data.len() { return Err(Failure::new(ErrorCode::AdmissionOrResourceLimit, "document_byte_budget")); }
        data.try_reserve(count).map_err(|_| Failure::from(RedactError::AllocationRefused))?;
        data.extend_from_slice(&scratch[..count]);
    }
    String::from_utf8(data).map_err(|_| Failure::new(ErrorCode::InputDecodeOrParse, "document_not_utf8"))
}

fn execute(options: &RedactCommand, input: &mut impl Read) -> Result<RedactionResult, Failure> {
    options.validate()?; // Before any key/document read or side effect.
    let mut key_file = options.key_file.as_ref().map(|p| local_io::open_key(p)).transpose()?;
    let key = if options.key_stdin || key_file.is_some() {
        let id = options.key_id.as_deref().unwrap_or("");
        let key = match key_file.as_mut() {
            Some(file) => PseudonymKey::from_reader(file, id)?,
            None => PseudonymKey::from_reader(input, id)?,
        };
        if let Some(expected) = &options.expected_key_commitment { key.require_commitment(expected)?; }
        Some(key)
    } else { None };
    let source = if options.document.as_os_str() == "-" {
        read_document(input, options.max_input_bytes)?
    } else {
        let mut file = local_io::open_document(&options.document)
            .map_err(|_| Failure::new(ErrorCode::InputDecodeOrParse, "regular_document_open_failed"))?;
        if let Some(key_file) = &key_file {
            if local_io::same_file(key_file, &file)? {
                return Err(Failure::new(ErrorCode::Usage, "key_file_is_also_document"));
            }
        }
        read_document(&mut file, options.max_input_bytes)?
    };
    let request = options.request();
    let namespace = options.namespace.as_deref().unwrap_or("");
    let context = match &key {
        None => None,
        Some(key) if options.full_digest => Some(Pseudonyms::full256(key, namespace, options.expected_key_commitment.as_deref())?),
        Some(key) => {
            let document = DetectedDocument::rules_only(&source, &request.rules, request.rule_budget)?;
            let mut values = Vec::new();
            values.try_reserve_exact(document.regions().len()).map_err(|_| Failure::from(RedactError::AllocationRefused))?;
            // Mixed connected components are masked, not assigned an invented
            // entity type. Only actual pseudonym replacements enter the set.
            for region in document.regions() {
                if region.kinds.len() == 1 {
                    let value = source.get(region.span.byte_start..region.span.byte_end).ok_or(RedactError::InvalidSpan)?;
                    values.push((*region.kinds.iter().next().ok_or(RedactError::InvalidSpan)?, value));
                }
            }
            Some(Pseudonyms::preflight128(key, namespace, &values, PseudonymBudget {
                max_values: options.max_preflight_values, max_value_bytes: options.max_preflight_bytes,
            }, options.expected_key_commitment.as_deref())?)
        }
    };
    redact_rules(&source, &request, context.as_ref()).map_err(Into::into)
}

#[derive(Serialize)]
struct CoordinateMap<'a> {
    schema_version: u32,
    task_spec_version: &'static str,
    policy_digest: crate::execution_identity::Sha256Digest,
    verification: crate::tasks::redact::actions::VerificationStatus,
    pseudonym_identity: Option<&'a crate::tasks::redact::pseudonym::PseudonymIdentity>,
    edits: &'a [crate::tasks::redact::actions::RedactionEdit],
}

fn emit(options: &RedactCommand, input: &mut impl Read, out: &mut impl Write) -> Result<(), Failure> {
    options.validate()?;
    // Bind destination capabilities before reading private input, but do not
    // create staging files until the entire result and optional map are ready.
    let output = options.file_output().map(|p| Destination::prepare(p)).transpose()?;
    let map = options.map_out.as_ref().map(|p| Destination::prepare(p)).transpose()?;
    if let (Some(output), Some(map)) = (&output, &map) {
        if output.same_target(map)? { return Err(Failure::new(ErrorCode::Usage, "output_and_map_are_same_target")); }
    }
    let mut result = execute(options, input)?;
    let map_bytes = if map.is_some() {
        let edits = std::mem::take(&mut result.edits);
        let document = CoordinateMap { schema_version: 1, task_spec_version: "redact-coordinate-map-v1",
            policy_digest: result.policy_digest(), verification: result.verification(),
            pseudonym_identity: result.pseudonym_identity.as_ref(), edits: &edits };
        let mut bytes = canonjson::canonical_bytes(&document).map_err(|_| Failure::from(RedactError::Serialization))?;
        bytes.push(b'\n'); Some(bytes)
    } else { None };
    // Map coordinates are emitted ONLY to the requested protected file. They
    // are removed from the result before JSON serialization to stdout/-o.
    let mut bytes = if options.json { canonjson::canonical_bytes(&result).map_err(|_| Failure::from(RedactError::Serialization))? }
        else { result.text().as_bytes().to_vec() };
    if options.json { bytes.push(b'\n'); }
    let total = bytes.len().checked_add(map_bytes.as_ref().map_or(0, Vec::len)).ok_or(RedactError::OutputBudget)?;
    if total > options.max_output_bytes { return Err(RedactError::OutputBudget.into()); }
    let map_stage = match (map, map_bytes) { (Some(path), Some(bytes)) => Some(path.stage(&bytes)?), _ => None };
    let output_stage = output.map(|path| path.stage(&bytes)).transpose()?;
    // Both files are fully staged+synced before publishing either. Two paths
    // are NOT an atomic pair: the map is published first, the main document
    // last. A later error reports map_published; no rollback deletes finals.
    let mut map_published = false;
    if let Some(stage) = map_stage { stage.publish()?; map_published = true; }
    let result = match output_stage {
        Some(stage) => stage.publish().map_err(Failure::from),
        None => out.write_all(&bytes).and_then(|()| out.flush())
            .map_err(|_| Failure::new(ErrorCode::Generic, "output_write_failed")),
    };
    result.map_err(|mut failure| { failure.map_published = map_published; failure })
}

pub(crate) fn run(options: RedactCommand, input: &mut impl Read, out: &mut impl Write, err: &mut impl Write) -> ExitCode {
    match emit(&options, input, out) {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            if let Ok(mut row) = canonjson::canonical_bytes(&failure) { row.push(b'\n'); let _ = err.write_all(&row).and_then(|()| err.flush()); }
            failure.code.as_process_exit()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::FromArgMatches;
    fn options(args: &[&str]) -> RedactCommand {
        let m = definition().try_get_matches_from(args).unwrap();
        RedactCommand::from_arg_matches(&m).unwrap()
    }
    #[test]
    fn default_redaction_is_verified_rules_only_without_model() {
        let result = execute(&options(&["redact", "--rules-only"]), &mut "é a@example.org".as_bytes()).unwrap_or_else(|_| panic!("redaction failed"));
        assert_eq!(result.text(), "é [redacted:email]");
        assert!(result.model_types().is_empty());
        assert_eq!(result.verification(), crate::tasks::redact::actions::VerificationStatus::CleanDeclaredUnion);
    }
    #[test]
    fn stdin_conflict_and_unused_keys_refuse_before_read() {
        struct NoRead;
        impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> { panic!("must not read"); } }
        for args in [vec!["redact", "--rules-only", "--key-stdin"],
            vec!["redact", "--rules-only", "--action", "pseudonymize", "--key-stdin", "--key-id", "v1", "--namespace", "job"]] {
            assert!(execute(&options(&args), &mut NoRead).is_err());
        }
    }
    #[test]
    fn byte_limit_consumes_only_one_overflow_byte() {
        let mut input = std::io::Cursor::new(b"123456789");
        assert!(read_document(&mut input, 4).is_err()); assert_eq!(input.position(), 5);
        assert_eq!(read_document(&mut &b""[..], 0).ok().as_deref(), Some(""));
        assert!(read_document(&mut &b"\xff"[..], 4).is_err());
    }
    #[test]
    fn plaintext_preserves_no_final_newline_and_json_is_one_record() {
        for json in [false, true] {
            let mut args = vec!["redact", "--rules-only"]; if json { args.push("--json"); }
            let mut out = Vec::new(); let mut err = Vec::new();
            assert_eq!(run(options(&args), &mut "a@example.org".as_bytes(), &mut out, &mut err), ExitCode::SUCCESS);
            assert!(err.is_empty());
            if json { let v: serde_json::Value = serde_json::from_slice(&out).unwrap(); assert_eq!(v["text"], "[redacted:email]"); assert_eq!(out.iter().filter(|b| **b == b'\n').count(), 1); }
            else { assert_eq!(out, b"[redacted:email]"); }
        }
    }
    #[test]
    fn invalid_utf8_and_output_budget_have_no_stdout_or_raw_input() {
        for (data, args) in [(b"private\xff".as_slice(), vec!["redact", "--rules-only"]),
            (b"a@example.org".as_slice(), vec!["redact", "--rules-only", "--max-output-bytes", "1"])] {
            let mut out = Vec::new(); let mut err = Vec::new();
            assert_ne!(run(options(&args), &mut &*data, &mut out, &mut err), ExitCode::SUCCESS);
            assert!(out.is_empty()); let log = String::from_utf8(err).unwrap();
            assert!(!log.contains("private")); assert!(!log.contains("a@example.org"));
        }
    }
    #[test]
    fn rules_scope_and_opt_out_are_visible_in_result() {
        let args = options(&["redact", "--rules-only", "--rules", "date", "--no-verify"]);
        let result = execute(&args, &mut "2024-02-29 a@example.org".as_bytes()).unwrap_or_else(|_| panic!("failed"));
        assert_eq!(result.text(), "[redacted:date] a@example.org");
        assert_eq!(result.verification(), crate::tasks::redact::actions::VerificationStatus::NotRequested);
    }
}
