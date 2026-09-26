//! Explicit private result retention and authenticated native inference resume.
//! No implicit key generation, input persistence, directory creation or retry.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]
use super::*;
use clap::Subcommand;
use crate::jobs::{JobId, JobLimits};
use super::source::SourceHostArgs;
mod config;
pub(super) use config::{Defaults, load_defaults, native_limits, parse_limits};
#[cfg(test)] pub(super) mod tests;

pub(super) const LIMIT_BYTES: usize = 16 * 1024;
pub(super) const DEFAULT_BYTES: usize = 1024 * 1024;
pub(super) const REPORT_BYTES: usize = 16 * 1024;

#[derive(Args)]
pub(crate) struct JobCommand {
    #[command(subcommand)] operation: Operation,
}
#[derive(Subcommand)]
enum Operation {
    /// Start a NEW retained-result job in an existing protected directory.
    Start(JobArgs),
    /// Authenticate the original inputs/configuration and run only pending items.
    Resume(ResumeArgs),
}
#[derive(Args)]
struct ResumeArgs {
    #[command(flatten)] common: JobArgs,
    /// Explicitly truncate uncommitted spool bytes and discard reserved output
    /// stages ONLY after the original contract and committed frames authenticate.
    /// Without this flag, uncommitted tails/stages are reported without repair.
    #[arg(long)] discard_uncommitted_tail: bool,
}
#[derive(Args)]
pub(super) struct JobArgs {
    /// COMPLETE original NDJSON population, also on resume. '-' reads stdin.
    /// Input is admitted in memory, not saved for you; preserve it separately.
    #[arg(default_value = "-")] pub input: PathBuf,
    #[arg(long, value_parser = ["ner", "keyphrases", "summarize", "answer", "extract"])]
    pub task: String,
    #[command(flatten)] pub host: SourceHostArgs,
    /// Required consent: retain private native results in the owner-only spool.
    #[arg(long)] pub store_results: bool,
    /// Existing owner-only directory; never created or silently adopted.
    #[arg(long)] pub job_dir: PathBuf,
    /// Expected random 128-bit job ID, as exactly 32 lowercase hex characters.
    #[arg(long, value_parser = parse_job_id)] pub job_id: JobId,
    /// Protected regular file with exactly 32 raw secret bytes, NOT a hex key.
    #[arg(long)] pub key_file: PathBuf,
    /// Exact original JobLimits JSON, including lifetime work and retry ceilings.
    #[arg(long = "limits")] pub limits_file: PathBuf,
    /// Complete SourceBatchArgs or ExtractionBatchArgs defaults; no hidden input.
    #[arg(long, conflicts_with = "schema")] pub defaults: Option<PathBuf>,
    /// Shared exact local schema for --task extract only; alternatively use task_args.
    #[arg(long, conflicts_with = "defaults")] pub schema: Option<PathBuf>,
    #[arg(long, requires = "schema")] pub source_membership: bool,
    /// Publish verified materialized.ndjson after ALL items commit; never replace.
    #[arg(long)] pub materialize: bool,
    /// Per-invocation transport cap; immutable item/snapshot limits are additional.
    #[arg(long, default_value_t = 64)] pub max_stream_mib: u64,
    /// Includes blank lines; flush commands are NOT accepted in durable populations.
    #[arg(long, default_value_t = 100_000)] pub max_input_lines: u64,
    /// Explicit modeled database RAM, separate from the database file-size limit.
    #[arg(long, default_value_t = 64)] pub journal_memory_mib: u64,
    #[arg(long, default_value_t = 16)] pub serialization_memory_mib: u64,
}
#[derive(Clone, Copy)]
pub(super) enum RunMode { Start, Resume { discard_uncommitted: bool } }
impl RunMode {
    pub(super) fn name(self) -> &'static str { match self { Self::Start => "start", Self::Resume { .. } => "resume" } }
}

pub(super) fn definition() -> clap::Command {
    JobCommand::augment_args(clap::Command::new("job")
        .about("Start or resume explicitly retained, non-certified native corpus jobs")
        .long_about("Requires metadata-store plus asupersync-runtime on Linux x86-64/AArch64. Store private results only with --store-results, a protected existing directory, original key, random job ID and immutable limits. Resume requires the complete original input and unchanged task/model recipe. Stdout contains metadata only; errors may follow durable progress. No automatic retry, input spooling, directory/key creation or release activation."))
}
impl JobArgs {
    pub(super) fn validate(&self) -> Result<(CandidateArgs, Limits), Failure> {
        let local = |p: &std::path::Path| !p.as_os_str().is_empty() && p.as_os_str() != "-";
        if !self.store_results || !local(&self.job_dir) || !local(&self.key_file) || !local(&self.limits_file)
            || self.defaults.as_ref().is_some_and(|p| !local(p)) || self.schema.as_ref().is_some_and(|p| !local(p))
            || self.defaults.is_some() && self.schema.is_some() || self.source_membership && self.schema.is_none()
            || self.task != "extract" && (self.schema.is_some() || self.source_membership)
            || !(1..=1024 * 1024).contains(&self.max_stream_mib)
            || !(1..=1_000_000_000).contains(&self.max_input_lines)
            || self.journal_memory_mib == 0 || self.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_storage_or_job_options"));
        }
        self.task_kind()?;
        let (common, limits) = self.host.common(self.input.clone()).map_err(Failure::from)?;
        for mib in [self.journal_memory_mib, self.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > limits.memory_bytes) {
                return Err(Failure::usage("job_memory_options"));
            }
        }
        Ok((common, limits))
    }
    pub(super) fn task_kind(&self) -> Result<crate::tasks::BuiltInTask, Failure> {
        use crate::tasks::BuiltInTask;
        match self.task.as_str() {
            "ner" => Ok(BuiltInTask::Ner), "keyphrases" => Ok(BuiltInTask::Keyphrases),
            "summarize" => Ok(BuiltInTask::Summarize), "answer" => Ok(BuiltInTask::Answer),
            "extract" => Ok(BuiltInTask::Extract), _ => Err(Failure::usage("job_task")),
        }
    }
}
impl JobCommand {
    fn mode(&self) -> RunMode {
        match &self.operation { Operation::Start(_) => RunMode::Start,
            Operation::Resume(r) => RunMode::Resume { discard_uncommitted: r.discard_uncommitted_tail } }
    }
    fn into_parts(self) -> (RunMode, JobArgs) {
        match self.operation { Operation::Start(args) => (RunMode::Start, args),
            Operation::Resume(r) => (RunMode::Resume { discard_uncommitted: r.discard_uncommitted_tail }, r.common) }
    }
    pub(super) fn run_owned<R: Read + Send + 'static>(self, input: R,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode().name();
        match self.execute_owned(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let error = ErrorReport { schema_version: 1, operation, status: "error", code: failure.code,
                    exit_code: failure.exit, durable_progress_may_exist: true,
                    recovery: "authenticate stored job status or explicitly resume; do not assume rollback" };
                if publish(&error, REPORT_BYTES, diagnostics).is_err() { return ErrorCode::Generic.as_process_exit(); }
                failure.exit.as_process_exit()
            }
        }
    }
    fn execute_owned<R: Read + Send + 'static>(self, input: R, output: &mut impl Write) -> Result<(), Failure> {
        let (mode, args) = self.into_parts();
        let (common, limits) = args.validate()?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::owned_jobs::execute(mode, args, common, limits, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        {
            let _ = (mode, args, common, limits, input, output);
            // BEFORE reading key/config/input or creating any process runtime.
            Err(Failure { exit: ErrorCode::Usage, code: "owned_job_profile_unavailable" })
        }
    }
}
fn parse_job_id(text: &str) -> Result<JobId, String> {
    if text.len() != 32 || !text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("expected 32 lowercase hexadecimal characters".to_owned());
    }
    let digit = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
    let mut bytes = [0_u8; 16];
    for (out, pair) in bytes.iter_mut().zip(text.as_bytes().chunks_exact(2)) { *out = (digit(pair[0]) << 4) | digit(pair[1]); }
    Ok(JobId(bytes))
}
pub(super) fn job_id_hex(id: JobId) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(32);
    for b in id.0 { text.push(HEX[(b >> 4) as usize] as char); text.push(HEX[(b & 15) as usize] as char); }
    text
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Failure { pub exit: ErrorCode, pub code: &'static str }
impl Failure {
    pub(super) fn usage(code: &'static str) -> Self { Self { exit: ErrorCode::Usage, code } }
}
impl From<CandidateError> for Failure {
    fn from(error: CandidateError) -> Self {
        let (exit, code) = match error {
            CandidateError::Arguments => (ErrorCode::Usage, "invalid_job_options"),
            CandidateError::Input => (ErrorCode::InputDecodeOrParse, "job_configuration_input"),
            CandidateError::Planning => (ErrorCode::Usage, "job_configuration_refused"),
            CandidateError::Memory | CandidateError::Runtime => (ErrorCode::AdmissionOrResourceLimit, "process_admission"),
            CandidateError::Timeout => (ErrorCode::BudgetOrTimeout, "invocation_deadline"),
            CandidateError::Identity => (ErrorCode::ArtifactIntegrityOrFormatOrVersion, "model_identity"),
            CandidateError::Model => (ErrorCode::Generic, "model_loading"),
            CandidateError::Output => (ErrorCode::Generic, "report_delivery"),
            _ => (ErrorCode::Generic, "job_setup_failed"),
        };
        Self { exit, code }
    }
}
#[derive(Serialize)]
struct ErrorReport {
    schema_version: u32, operation: &'static str, status: &'static str, code: &'static str,
    exit_code: ErrorCode, durable_progress_may_exist: bool, recovery: &'static str,
}
