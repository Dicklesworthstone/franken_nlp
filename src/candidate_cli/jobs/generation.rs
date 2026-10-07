//! Explicit retained generation/chat; the live text-batch protocol is unchanged.
use super::*;
use crate::candidate_cli::text_batch::TextTask;

#[derive(Args)]
pub(crate) struct TextJobCommand { #[command(subcommand)] operation: TextOperation }
#[derive(Subcommand)]
enum TextOperation {
    /// Create a NEW retained generation/chat job in protected local storage.
    Start(TextJobArgs),
    /// Authenticate original inputs/options and run only uncommitted items.
    Resume(TextResumeArgs),
}
#[derive(Args)]
struct TextResumeArgs {
    #[command(flatten)] common: TextJobArgs,
    /// Repair only uncommitted tails, after authenticating the original contract.
    #[arg(long)] discard_uncommitted_tail: bool,
}
#[derive(Args)]
pub(in crate::candidate_cli) struct TextJobArgs {
    #[arg(long, value_enum, default_value = "generate")] pub task: TextTask,
    /// Fixed per-item generation options. Input is the COMPLETE original NDJSON population.
    #[command(flatten)] pub common: CandidateArgs,
    /// Consent to retain private generated content, tokens and optional scores.
    #[arg(long)] pub store_results: bool,
    #[arg(long)] pub job_dir: PathBuf,
    #[arg(long, value_parser = parse_job_id)] pub job_id: JobId,
    /// Protected file containing exactly 32 raw job-authentication bytes; not a sampling seed.
    #[arg(long)] pub key_file: PathBuf,
    /// Immutable JobLimits JSON, including ALL five native counters and attempts.
    #[arg(long = "limits")] pub limits_file: PathBuf,
    #[arg(long)] pub materialize: bool,
    #[arg(long, default_value_t = 64)] pub max_stream_mib: u64,
    #[arg(long, default_value_t = 100_000)] pub max_input_lines: u64,
    #[arg(long, default_value_t = 64)] pub journal_memory_mib: u64,
    #[arg(long, default_value_t = 16)] pub serialization_memory_mib: u64,
}
impl TextJobArgs {
    pub(in crate::candidate_cli) fn task_name(&self) -> &'static str {
        match self.task { TextTask::Generate => "generate", TextTask::Chat => "chat" }
    }
    pub(in crate::candidate_cli) fn validate(&self) -> Result<Limits, Failure> {
        let limits = self.common.validate()?;
        // The retained adapter is serial. Never accept a scheduling option and
        // quietly ignore it or let it change across resumed executions.
        if self.common.policy.prefill()?.is_some() { return Err(Failure::usage("text_job_serial_schedule")); }
        let local = |p: &std::path::Path| !p.as_os_str().is_empty() && p.as_os_str() != "-";
        if !self.store_results || !local(&self.job_dir) || !local(&self.key_file) || !local(&self.limits_file)
            || !(1..=1024 * 1024).contains(&self.max_stream_mib)
            || !(1..=1_000_000_000).contains(&self.max_input_lines)
            || self.journal_memory_mib == 0 || self.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_text_job_storage"));
        }
        for mib in [self.journal_memory_mib, self.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|n| n > limits.memory_bytes) {
                return Err(Failure::usage("text_job_memory"));
            }
        }
        Ok(limits)
    }
    pub(in crate::candidate_cli) fn check_lifetime(&self, job: JobLimits, limits: Limits) -> Result<(), Failure> {
        job.validate().map_err(|_| Failure::usage("text_job_lifetime"))?;
        if job.max_input_bytes_per_item > self.common.max_input_bytes || job.max_result_bytes < limits.result_bytes
            || job.max_snapshot_bytes > self.max_stream_mib.checked_mul(MIB).ok_or_else(|| Failure::usage("text_job_transport"))? {
            return Err(Failure::usage("text_job_input_or_result_ceiling"));
        }
        let w = job.max_work.model;
        if w.forward_positions == 0 || w.projected_logits == 0 || w.attention_pairs == 0
            || w.projections.dot_products == 0 || w.projections.multiply_accumulates == 0 {
            return Err(Failure::usage("text_job_native_work"));
        }
        Ok(())
    }
}
pub(in crate::candidate_cli) fn definition() -> clap::Command {
    TextJobCommand::augment_args(clap::Command::new("text-job")
        .about("Start or resume retained native generation/chat jobs")
        .long_about("Requires metadata-store plus asupersync-runtime on Linux x86-64/AArch64. Input is the COMPLETE original {id,text,task_args?} NDJSON population on both start and resume, not the text-batch prompt/messages protocol. Optional task_args contains sample_index and chat history only; text becomes the final user turn. Shared generation flags, explicit seed, stops and penalties are frozen in the authenticated recipe. No seed means greedy, never random seed generation. Results commit only after whole native completion; no partial token stream is retained. Failed attempts retain all native work charges. --store-results, protected existing storage, original job key and immutable limits are required. Stdout is completion metadata only. No automatic retry, input persistence, model download, tools or thinking mode. This retained route is serial; --prefill-rows/cohort-rows/active-rows are unsupported."))
}
impl TextJobCommand {
    fn into_parts(self) -> (RunMode, TextJobArgs) {
        match self.operation { TextOperation::Start(args) => (RunMode::Start, args),
            TextOperation::Resume(args) => (RunMode::Resume { discard_uncommitted: args.discard_uncommitted_tail }, args.common) }
    }
    pub(in crate::candidate_cli) fn run_owned<R: Read + Send + 'static>(self, input: R,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let (mode, args) = self.into_parts();
        match execute_owned(mode, args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { schema_version: 1, operation: mode.name(), status: "error", code: failure.code,
                    exit_code: failure.exit, durable_progress_may_exist: true,
                    recovery: "authenticate stored status or explicitly resume with original inputs and options; do not assume rollback" };
                if publish(&report, REPORT_BYTES, diagnostics).is_err() { return ErrorCode::Generic.as_process_exit(); }
                failure.exit.as_process_exit()
            }
        }
    }
}
fn execute_owned<R: Read + Send + 'static>(mode: RunMode, args: TextJobArgs, input: R, output: &mut impl Write)
    -> Result<(), Failure> {
    let limits = args.validate()?;
    #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")))]
    { crate::candidate_cli::runtime::owned_jobs::generation::execute(mode, args, limits, input, output) }
    #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64"))))]
    {
        let _ = (mode, args, limits, input, output);
        Err(Failure::usage("owned_text_job_profile_unavailable"))
    }
}
#[cfg(test)] pub(in crate::candidate_cli) mod tests;
