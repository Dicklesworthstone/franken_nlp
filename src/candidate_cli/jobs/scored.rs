//! Explicit retained finite-score jobs; separate from the live score-batch pipe.
use super::*;
use crate::candidate_cli::{scored::{Kind, ScoredArgs}, scored_batch};

#[derive(Args)]
pub(crate) struct ScoreJobCommand {
    #[command(subcommand)] operation: ScoreOperation,
}
#[derive(Subcommand)]
enum ScoreOperation {
    /// Start a NEW native classification/sentiment job; retain private results.
    Start(ScoreJobArgs),
    /// Authenticate the COMPLETE original population and run only pending items.
    Resume(ScoreResumeArgs),
}
#[derive(Args)]
struct ScoreResumeArgs {
    #[command(flatten)] common: ScoreJobArgs,
    /// Repair uncommitted tails/stages only AFTER original-contract authentication.
    #[arg(long)] discard_uncommitted_tail: bool,
}
#[derive(Args)]
pub(in crate::candidate_cli) struct ScoreJobArgs {
    #[arg(long, value_parser = ["classify", "sentiment"])]
    pub task: String,
    #[command(flatten)] pub host: ScoredArgs,
    /// Required consent to retain private native output in the protected spool.
    #[arg(long)] pub store_results: bool,
    /// Existing owner-only directory. Never created or adopted automatically.
    #[arg(long)] pub job_dir: PathBuf,
    /// Expected random 128-bit job ID: exactly 32 lowercase hex characters.
    #[arg(long, value_parser = parse_job_id)] pub job_id: JobId,
    /// Protected regular file containing exactly 32 RAW secret bytes, not hex.
    #[arg(long)] pub key_file: PathBuf,
    /// Immutable JobLimits JSON, including lifetime work and attempt ceilings.
    #[arg(long = "limits")] pub limits_file: PathBuf,
    /// Same settings JSON as score-batch; no document, budget or identity fields.
    #[arg(long)] pub defaults: Option<PathBuf>,
    /// Publish verified materialized.ndjson only after all items commit.
    #[arg(long)] pub materialize: bool,
    #[arg(long, default_value_t = 64)] pub max_stream_mib: u64,
    /// Includes blank lines. Flush records are forbidden in durable populations.
    #[arg(long, default_value_t = 100_000)] pub max_input_lines: u64,
    #[arg(long, default_value_t = 64)] pub journal_memory_mib: u64,
    #[arg(long, default_value_t = 16)] pub serialization_memory_mib: u64,
}
impl ScoreJobArgs {
    pub(in crate::candidate_cli) fn kind(&self) -> Result<Kind, Failure> {
        Kind::named(&self.task).ok_or_else(|| Failure::usage("score_job_task"))
    }
    pub(in crate::candidate_cli) fn validate(&self) -> Result<(CandidateArgs, Limits), Failure> {
        self.kind()?;
        let local = |p: &std::path::Path| !p.as_os_str().is_empty() && p.as_os_str() != "-";
        if !self.store_results || !local(&self.job_dir) || !local(&self.key_file) || !local(&self.limits_file)
            || self.defaults.as_ref().is_some_and(|p| !local(p))
            || !(1..=1024 * 1024).contains(&self.max_stream_mib)
            || !(1..=1_000_000_000).contains(&self.max_input_lines)
            || self.journal_memory_mib == 0 || self.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_scored_job_storage"));
        }
        let (common, limits) = self.host.common()?;
        for mib in [self.journal_memory_mib, self.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > limits.memory_bytes) {
                return Err(Failure::usage("score_job_memory"));
            }
        }
        Ok((common, limits))
    }
    pub(in crate::candidate_cli) fn parse_defaults(&self, json: Option<&str>, budget: crate::tasks::ir::TaskBudget)
        -> Result<scored_batch::Defaults, Failure> {
        scored_batch::parse_defaults(self.kind()?, &self.host, json, budget).map_err(Into::into)
    }
}

pub(in crate::candidate_cli) fn definition() -> clap::Command {
    ScoreJobCommand::augment_args(clap::Command::new("score-job")
        .about("Start or resume retained native classification/sentiment corpus jobs")
        .long_about("Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64. The input is the COMPLETE original NDJSON {id,text,task_args?} population, including on resume. Uses the same full-vocabulary scorers and defaults as score-batch; no generated-label shortcut or calibrated-confidence claim. All scoring flags and defaults are frozen for resume. Lifetime work and failed attempts remain debited. Explicit --store-results, protected directory, original secret and immutable JobLimits are mandatory. Stdout is metadata only, not private document scores. No automatic retry or input spooling."))
}
impl ScoreJobCommand {
    fn into_parts(self) -> (RunMode, ScoreJobArgs) {
        match self.operation { ScoreOperation::Start(args) => (RunMode::Start, args),
            ScoreOperation::Resume(args) => (RunMode::Resume { discard_uncommitted: args.discard_uncommitted_tail }, args.common) }
    }
    pub(in crate::candidate_cli) fn run_owned<R: Read + Send + 'static>(self, input: R,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let (mode, args) = self.into_parts();
        match execute_owned(mode, args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { schema_version: 1, operation: mode.name(), status: "error",
                    code: failure.code, exit_code: failure.exit, durable_progress_may_exist: true,
                    recovery: "authenticate stored job status or explicitly resume; do not assume rollback" };
                if publish(&report, REPORT_BYTES, diagnostics).is_err() { return ErrorCode::Generic.as_process_exit(); }
                failure.exit.as_process_exit()
            }
        }
    }
}
fn execute_owned<R: Read + Send + 'static>(mode: RunMode, args: ScoreJobArgs, input: R, output: &mut impl Write)
    -> Result<(), Failure> {
    let (common, limits) = args.validate()?;
    #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")))]
    { crate::candidate_cli::runtime::owned_jobs::scored::execute(mode, args, common, limits, input, output) }
    #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64"))))]
    {
        let _ = (mode, args, common, limits, input, output);
        // BEFORE key, defaults, population, model or runtime IO.
        Err(Failure::usage("owned_score_job_profile_unavailable"))
    }
}

#[cfg(test)] pub(in crate::candidate_cli) mod tests;
