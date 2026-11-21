//! Explicit retained summaries and passage answers with complete source review.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use crate::{
    batch::{extract::ExtractionMaskBudget, source::semantic::Int4SemanticSourceBatchLimits},
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, RunMode}, stream::Failure},
    jobs::{JobLimits, population::PopulationReadLimits},
    native_engine::decode::DecodeCancellationKind,
};
use super::super::extract_job::StorageArgs;

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)]
mod tests;

pub(in crate::candidate_cli::int4) struct JobCommand { mode: RunMode, args: JobArgs }
struct JobArgs {
    kind: Kind,
    semantic: SemanticSourceArgs,
    storage: StorageArgs,
}

pub(in crate::candidate_cli::int4) fn definitions() -> Vec<clap::Command> {
    [(Kind::Summarize, "summarize-semantic-job"), (Kind::Answer, "answer-semantic-job")]
        .into_iter().zip(super::definitions()).map(|((kind, name), command)| {
            let operation = |operation: &'static str, about: &'static str| {
                StorageArgs::augment_args(command.clone().name(operation).about(about)
                    .long_about(concat!(
                        "Retain and resume cited summaries or passage answers followed by complete, explicitly configured same-model review. Each committed private result preserves the original source result and every configured statement review together; a failed review commits no partial source result. ",
                        "Requires --store-results, an existing protected --job-dir, original --job-id and --key-file, immutable --limits, and exact --semantic-options plus any fixed --options. ",
                        "Summaries accept {id,text} and refuse non-null task_args. Answers accept {id,text:question,task_args:{passages:[{id,text}]}}; task, options, budgets, review policy, token IDs and execution authority cannot be supplied per record. Duplicate IDs anywhere in the complete population and flush controls reject. ",
                        "Resume authenticates the complete original population, model/group identity, both compilers, exact source/review/tokenizer policy and lifetime limits before repair or inference. Only pending records run. ",
                        "The five native axes and generation-mask work in --limits cover all records and attempts, including failed attempts and earlier invocations. Every admitted attempt spends its full declared source-plus-review allowance before its first forward. More items, attempts or threads grant no additional work. ",
                        "Supply the complete original NDJSON population on every invocation, or explicitly retain inputs at start and resume with --use-retained-input. --max-input-bytes and the stored item cap bound the complete source input; for QA that includes the question, joined passages and canonical passage manifest. Every complete source and review prompt must fit context; no chunking or retrieval is performed. ",
                        "--input-encoding safe-bpe-v1 gives each attempt one bounded tokenizer ledger shared by source/question/passages and all review prompts; the exact policy is authenticated. --compute-threads 2..64 keeps one admitted numerical team across all pending records and reviews. Physical width may change on resume without renewing lifetime work. ",
                        "The CLI quotes complete preparation, population, native, combined output, journal, serializer and team memory before key/source-options/population/model IO after bounded semantic-options and limits reads. Source conversion, independently pinned snapshots and optional snapshot saving share the route. Record-specific semantic admission occurs during preparation; a refused record never becomes a successful abstention. ",
                        "Stdout is metadata only. --materialize publishes verified materialized.ndjson after all items commit. Failure after durable acknowledgement does not imply rollback. Review remains an experimental uncalibrated same-model diagnostic; original semantic_support stays not_assessed. No repair, acceptance authority, automatic retry, selected rows, prefill, cohorts, sampling, thinking or tools. Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64."))
                    .after_help(job_cli::INPUT_RETENTION_HELP)
                    .mut_arg("store_results", |arg| arg.help("Required consent to retain private source results and complete reviews in the protected spool"))
                    .mut_arg("options", |arg| arg.help("Fixed complete source-task options for every item; omitted uses the typed task default"))
                    .mut_arg("input", |arg| arg.help(match kind {
                        Kind::Summarize => "COMPLETE original {id,text} NDJSON; omit with --use-retained-input; '-' reads stdin",
                        _ => "COMPLETE original {id,text:question,task_args:{passages:[{id,text}]}} NDJSON; omit with --use-retained-input",
                    }))
                    .mut_arg("max_input_bytes", |arg| arg.help("Outer record and complete decoded source-input cap; immutable JobLimits item cap must fit"))
                    .mut_arg("max_checkpoints", |arg| arg.help("Checkpoints for this invocation; persisted work and attempts never renew")))
            };
            let start = operation("start", "Create an explicitly retained source-result and complete-review population");
            let resume = operation("resume", "Authenticate the frozen source-review job and run pending records")
                .arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
                    .action(clap::ArgAction::SetTrue)
                    .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"));
            clap::Command::new(name).about("Start or resume retained source tasks with complete correlated review")
                .subcommand_required(true).subcommand(start).subcommand(resume)
        }).collect()
}

#[derive(Clone, Copy, Debug)]
struct JobExecutionLimits {
    common: Int4Limits,
    task: TaskBudget,
    planning: SourcePlanningLimits,
    native: Int4SemanticSourceBatchLimits,
    transport: PopulationReadLimits,
    io_reserve_bytes: u64,
    journal_reserve_bytes: u64,
    serialization_reserve_bytes: u64,
}
impl JobArgs {
    fn validate(&self, args: &Int4Args) -> Result<Int4Limits, Failure> {
        let s = &self.storage;
        s.input_retention.limits().map_err(storage_failure)?;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !matches!(self.kind, Kind::Summarize | Kind::Answer)
            || !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0 {
            return Err(Failure::usage("semantic_source_job_storage_options"));
        }
        let common = self.semantic.validate(args)?;
        for mib in [s.journal_memory_mib, s.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > common.memory_bytes) {
                return Err(Failure::usage("semantic_source_job_memory_options"));
            }
        }
        Ok(common)
    }
    fn lifetime_limits(&self, args: &Int4Args, initial: Int4Limits,
        options: &Int4SourceSemanticOptions, job: JobLimits) -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("semantic_source_job_limits"))?;
        let common = self.semantic.admit_options(args, initial, options)?;
        if job.max_input_bytes_per_item > args.max_input_bytes || job.max_id_bytes > 256
            || options.limits.max_output_bytes > job.max_result_bytes as u64 {
            return Err(Failure::usage("semantic_source_job_input_or_combined_result_limit"));
        }
        let native = Int4SemanticSourceBatchLimits { max_model_work: job.max_work.int4_model_work(),
            masks: ExtractionMaskBudget { per_mask: self.semantic.source.masks(),
                max_visits_per_item: self.semantic.source.mask_visits(),
                max_visits_per_run: job.max_work.mask_node_visits } };
        native.validate_item(options.max_model_work)
            .map_err(|_| Failure::usage("semantic_source_job_lifetime_work"))?;
        let transport = PopulationReadLimits { max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines };
        transport.validate().map_err(|_| Failure::usage("semantic_source_job_transport"))?;
        let mut planning = self.semantic.source.planning(args);
        // One bound covers the complete decoded QA aggregate, not only the
        // envelope's text/question. Store the same ceiling in the job recipe.
        planning.max_input_bytes = planning.max_input_bytes.min(job.max_input_bytes_per_item);
        Ok(JobExecutionLimits {
            task: self.semantic.source.budget(args, common), common, planning, native, transport,
            io_reserve_bytes: product(IO_BUFFER_BYTES as u64, 2)?,
            journal_reserve_bytes: product(self.storage.journal_memory_mib, MIB)?,
            serialization_reserve_bytes: product(self.storage.serialization_memory_mib, MIB)?,
        })
    }
    fn input_ceiling(&self, limits: JobExecutionLimits, options: &Int4SourceSemanticOptions)
        -> SourceSemanticInputFacts {
        let answer = self.kind == Kind::Answer;
        SourceSemanticInputFacts {
            source_bytes: limits.planning.max_input_bytes.min(options.limits.max_source_bytes),
            question_bytes: if answer { limits.planning.max_input_bytes.min(options.limits.max_claim_bytes) } else { 0 },
            passage_count: if answer { limits.planning.max_passages } else { 0 },
        }
    }
}

impl JobCommand {
    pub(in crate::candidate_cli::int4) fn from_matches(kind: Kind, matches: &clap::ArgMatches)
        -> Result<(Self, Int4Args), clap::Error> {
        if !matches!(kind, Kind::Summarize | Kind::Answer) {
            return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand,
                "semantic source job task refused"));
        }
        let (name, matches) = matches.subcommand().ok_or_else(|| clap::Error::raw(
            clap::error::ErrorKind::MissingSubcommand, "semantic source job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand,
                "semantic source job operation refused")),
        };
        let args = JobArgs { kind, semantic: SemanticSourceArgs::from_matches(matches)?,
            storage: StorageArgs::from_arg_matches(matches)? };
        let mut common = Int4Args::from_arg_matches(matches)?;
        common.compute = compute::Args::from_matches(matches)?;
        // SemanticSourceArgs owns the source-plus-review tokenizer allowance.
        Ok((Self { mode, args }, common))
    }
    pub(in crate::candidate_cli::int4) fn run_owned<R: Read + Send + 'static>(self,
        args: Int4Args, input: R, output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode.name(); let task = task_name(self.args.kind);
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { schema_version: 1, operation, task, status: "error", code: failure.code,
                    exit_code: failure.exit, cancellation: failure.cancellation, durable_progress_may_exist: true,
                    recovery: "authenticate stored job status or explicitly resume; do not assume rollback" };
                if publish(&report, job_cli::REPORT_BYTES, diagnostics).is_err() {
                    return ErrorCode::Generic.as_process_exit();
                }
                failure.exit.as_process_exit()
            }
        }
    }
    fn execute_owned<R: Read + Send + 'static>(self, args: Int4Args,
        input: R, output: &mut impl Write) -> Result<(), Failure> {
        self.args.storage.input_retention.validate(self.mode, &args.input).map_err(storage_failure)?;
        let common = self.args.validate(&args)?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::execute(self.mode, args, self.args, common, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        { let _ = (self, args, common, input, output); Err(Failure::usage("owned_job_profile_unavailable")) }
    }
}
fn task_name(kind: Kind) -> &'static str {
    match kind { Kind::Summarize => "summarize-semantic", Kind::Answer => "answer-semantic",
        _ => "unsupported-semantic-source" }
}
fn storage_failure(failure: job_cli::Failure) -> Failure {
    Failure { exit: failure.exit, code: failure.code, cancellation: None }
}
fn product(value: u64, count: u64) -> Result<u64, Failure> {
    value.checked_mul(count).ok_or_else(|| Failure::usage("semantic_source_job_memory_arithmetic"))
}
#[derive(Serialize)]
struct ErrorReport {
    schema_version: u32, operation: &'static str, task: &'static str, status: &'static str,
    code: &'static str, exit_code: ErrorCode,
    #[serde(skip_serializing_if = "Option::is_none")]
    cancellation: Option<DecodeCancellationKind>,
    durable_progress_may_exist: bool, recovery: &'static str,
}
