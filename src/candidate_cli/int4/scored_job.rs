//! Explicit retained scoring on the same resident grouped-INT4 task planners.
//! Immutable JobLimits own lifetime work; CLI scoring caps bound each item.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use crate::{
    batch::scored_int4::Int4ScoredBatchLimits,
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, Failure, RunMode},
        scored_batch::{self as shared, Defaults, ScoreTask}},
    jobs::{JobLimits, population::PopulationReadLimits},
    tasks::ir::TaskBudget,
};
use super::extract_job::StorageArgs;

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)]
mod tests;

pub(super) struct JobCommand { mode: RunMode, args: JobArgs }
pub(super) struct JobArgs {
    scoring: classify::ScoringArgs,
    score: ScoreArgs,
    storage: StorageArgs,
}
#[derive(Args)]
struct ScoreArgs {
    /// One fixed scoring task for the complete frozen population.
    #[arg(long, value_parser = ["classify", "sentiment", "judge"])]
    task: String,
    /// Same bounded private task settings as score-batch; no document or identity.
    #[arg(long, value_name = "FILE")]
    defaults: Option<PathBuf>,
}

pub(super) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        StorageArgs::augment_args(ScoreArgs::augment_args(classify::definition().name(name).about(about)
            .long_about("Retain and resume complete classification, sentiment or judgment results with one resident source-converted or independently pinned snapshot INT4 model. Requires --store-results, an existing protected --job-dir, original --job-id and --key-file, and immutable --limits. Supply the COMPLETE original NDJSON {id,text,task_args?} population on every invocation. --task is fixed; --defaults has the same typed settings as score-batch. Supplied task_args replaces the complete shared arguments. Sentiment scoring mode, EOS and policy stay fixed for the run. Every native head preflights before execution. Resume authenticates original input bytes, defaults, planner limits and model/group identity before any repair or native forward. Five scoring flags bound EACH ITEM; JobLimits.max_work is the sole lifetime authority, including failed attempts, and its mask_node_visits must be zero. Neither population size nor resume renews that work. Flush controls are refused. Stdout is metadata only; private scores remain in the explicitly retained spool. --materialize publishes only after all items commit. Requires metadata-store plus asupersync-runtime on Linux x86-64/AArch64. No automatic retry, input spooling, key/directory creation, release activation or semantic-accuracy claim.")
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population, also on resume; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.help("Outer cap on each complete NDJSON record; immutable JobLimits.max_input_bytes_per_item must fit"))
            .mut_arg("max_result_bytes", |arg| arg.help("Complete per-item scored result ceiling; must fit immutable JobLimits.max_result_bytes"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoints for this invocation; persisted work and attempts do not renew"))
            .mut_arg("max_forward_positions", |arg| arg.help("Per-item forward ceiling across all scoring heads; lifetime ceiling comes from --limits"))
            .mut_arg("max_projected_logits", |arg| arg.help("Per-item full-vocabulary projection ceiling; lifetime ceiling comes from --limits"))
            .mut_arg("max_attention_pairs", |arg| arg.help("Per-item attention pair ceiling; lifetime ceiling comes from --limits"))
            .mut_arg("max_dot_products", |arg| arg.help("Per-item actual grouped dot-product ceiling; lifetime ceiling comes from --limits"))
            .mut_arg("max_multiply_accumulates", |arg| arg.help("Per-item multiply-accumulate ceiling; lifetime ceiling comes from --limits"))))
            .mut_arg("store_results", |arg| arg.help("Required consent to retain private scored results in the owner-only spool"))
            .mut_arg("limits_file", |arg| arg.help("Exact immutable JobLimits JSON; lifetime five-axis native work and zero mask work"))
    };
    let start = operation("start", "Create an explicitly retained INT4 scoring job");
    let resume = operation("resume", "Authenticate the original population and run pending INT4 scoring items")
        .arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
            .action(clap::ArgAction::SetTrue)
            .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"));
    clap::Command::new("score-job")
        .about("Start or resume retained INT4 classification, sentiment or judgment")
        .subcommand_required(true).subcommand(start).subcommand(resume)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct JobExecutionLimits {
    common: Int4Limits,
    transport: PopulationReadLimits,
    native: Int4ScoredBatchLimits,
    buffer_bytes: u64,
    preparation_reserve_bytes: u64,
    io_reserve_bytes: u64,
    journal_reserve_bytes: u64,
    serialization_reserve_bytes: u64,
}
impl JobArgs {
    fn kind(&self) -> Result<ScoreTask, CandidateError> {
        ScoreTask::named(&self.score.task).ok_or(CandidateError::Arguments)
    }
    fn validate(&self, args: &Int4Args) -> Result<Int4Limits, Failure> {
        self.kind()?;
        let s = &self.storage;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || self.score.defaults.as_ref().is_some_and(|path| !local(path))
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_scored_job_storage"));
        }
        let common = self.scoring.validate(args)?;
        for mib in [s.journal_memory_mib, s.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > common.memory_bytes) {
                return Err(Failure::usage("score_job_memory"));
            }
        }
        Ok(common)
    }
    fn load_defaults(&self, text: Option<&str>, budget: TaskBudget, args: &Int4Args)
        -> Result<Defaults, CandidateError> {
        shared::parse_defaults_with_limits(self.kind()?, self.scoring.classification_limits(args),
            args.max_input_bytes, text, budget)
    }
    fn lifetime_limits(&self, args: &Int4Args, mut common: Int4Limits, job: JobLimits)
        -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("job_limits"))?;
        if job.max_input_bytes_per_item > args.max_input_bytes {
            return Err(Failure::usage("stored_input_ceiling"));
        }
        if common.result_bytes > job.max_result_bytes {
            return Err(Failure::usage("stored_result_ceiling"));
        }
        let native = Int4ScoredBatchLimits { max_item_work: self.scoring.work_ceiling(),
            max_model_work: job.max_work.int4_model_work() };
        if job.max_work.mask_node_visits != 0 || native.validate().is_err() {
            return Err(Failure::usage("lifetime_scored_work_limits"));
        }
        let transport = PopulationReadLimits { max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines };
        transport.validate().map_err(|_| Failure::usage("job_transport"))?;
        let preparation_reserve_bytes = common.preparation_bytes;
        let io_reserve_bytes = product(IO_BUFFER_BYTES as u64, 2)?;
        let journal_reserve_bytes = product(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = product(self.storage.serialization_memory_mib, MIB)?;
        // Match JobHostLimits::required_buffer_bytes. The CLI retains its
        // preparation owner while the host separately owns the full snapshot.
        let buffer_bytes = sum(&[
            product(job.max_snapshot_bytes, 2)?, product(job.max_items, 1024)?,
            product(job.max_input_bytes_per_item as u64, 8)?,
            product(job.max_result_bytes as u64, 4)?, 2 * MIB,
            preparation_reserve_bytes, io_reserve_bytes,
            journal_reserve_bytes, serialization_reserve_bytes,
        ]).map_err(|_| Failure::usage("job_memory_arithmetic"))?;
        common.minimum_memory_bytes = common.minimum_memory_bytes.checked_add(buffer_bytes)
            .ok_or_else(|| Failure::usage("job_memory_arithmetic"))?;
        if common.minimum_memory_bytes > common.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit, code: "job_population_memory" });
        }
        Ok(JobExecutionLimits { common, transport, native, buffer_bytes, preparation_reserve_bytes,
            io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes })
    }
}
impl JobCommand {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<(Self, Int4Args), clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "score-job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, "score-job operation refused")),
        };
        let args = JobArgs { scoring: classify::ScoringArgs::from_matches(matches)?,
            score: ScoreArgs::from_arg_matches(matches)?, storage: StorageArgs::from_arg_matches(matches)? };
        Ok((Self { mode, args }, Int4Args::from_arg_matches(matches)?))
    }
    pub(super) fn run_owned<R: Read + Send + 'static>(self, args: Int4Args, input: R,
        output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode.name();
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { schema_version: 1, operation, status: "error", code: failure.code,
                    exit_code: failure.exit, durable_progress_may_exist: true,
                    recovery: "authenticate stored job status or explicitly resume; do not assume rollback" };
                if publish(&report, job_cli::REPORT_BYTES, diagnostics).is_err() { return ErrorCode::Generic.as_process_exit(); }
                failure.exit.as_process_exit()
            }
        }
    }
    fn execute_owned<R: Read + Send + 'static>(self, args: Int4Args, input: R, output: &mut impl Write)
        -> Result<(), Failure> {
        let common = self.args.validate(&args)?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::execute(self.mode, args, self.args, common, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        { let _ = (self, args, common, input, output); Err(Failure::usage("owned_score_job_profile_unavailable")) }
    }
}
fn product(value: u64, count: u64) -> Result<u64, Failure> {
    value.checked_mul(count).ok_or_else(|| Failure::usage("job_memory_arithmetic"))
}
#[derive(Serialize)]
struct ErrorReport {
    schema_version: u32, operation: &'static str, status: &'static str,
    code: &'static str, exit_code: ErrorCode, durable_progress_may_exist: bool, recovery: &'static str,
}
