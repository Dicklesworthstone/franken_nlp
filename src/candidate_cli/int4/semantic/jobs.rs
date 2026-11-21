//! Explicit retained extraction and complete correlated field review.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use crate::{
    batch::extract::{ExtractionBatchArgs, ExtractionBatchGrounding, ExtractionMaskBudget,
        semantic::Int4SemanticExtractionBatchLimits},
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, RunMode}, stream::Failure},
    grammar::runtime::JsonValidationSchema,
    jobs::{JobLimits, population::PopulationReadLimits},
    native_engine::decode::DecodeCancellationKind,
};
use super::super::extract_job::StorageArgs;

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)]
mod tests;

pub(in crate::candidate_cli::int4) struct JobCommand {
    mode: RunMode,
    args: JobArgs,
}
struct JobArgs {
    semantic: SemanticArgs,
    storage: StorageArgs,
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        StorageArgs::augment_args(super::definition().name(name).about(about)
            .long_about(concat!(
                "Retain and resume bounded schema extraction followed by complete, explicitly configured same-model field review. ",
                "Each private committed result contains the unchanged independently validated extraction and every configured field's entailed/contradicted/unsupported/not_checked diagnostics. A failed review commits no partial extraction. ",
                "Requires --store-results, an existing protected --job-dir, original --job-id and --key-file, immutable --limits, exact --schema and --semantic-options. ",
                "Resume authenticates the complete original population, model/group identity, exact schema and grounding, claim rules, judge policy, tokenizer policy and all lifetime limits before repair or inference. Only pending items run. ",
                "The five native axes and extraction-mask work in --limits are lifetime ceilings across all records and attempts, including failed attempts and earlier invocations. Every admitted attempt spends its full declared extraction-plus-review allowance before its first forward. ",
                "Supply the complete original NDJSON population on each invocation, or explicitly retain inputs at start and resume with --use-retained-input. Records are {id,text}; non-null task_args and flush controls are refused. ",
                "--input-encoding safe-bpe-v1 gives each complete attempt one bounded tokenizer ledger shared by exact schema/source encoding and every review prompt. Its policy is authenticated on resume. ",
                "Serial compute is the default; --compute-threads 2..64 keeps one admitted numerical team across extraction and review. Physical team width may change on resume without renewing lifetime work. ",
                "The host admits complete semantic preparation, input ownership, native memory, combined output, journal and serializer storage before opening the population or job. The CLI checks the full peak before key/schema/population/model IO after bounded options and limits reads. ",
                "Stdout is completion metadata only. --materialize publishes verified materialized.ndjson after every item commits. Failure after durable acknowledgement never implies rollback. ",
                "This remains an experimental, uncalibrated same-model diagnostic; it does not repair fields or authorize acceptance. No automatic retry, selected-row projection, prefill, cohorts, thinking or tools. ",
                "Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64. Source conversion and independently pinned snapshots share the native implementation."))
            .after_help(job_cli::INPUT_RETENTION_HELP)
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population; omit with --use-retained-input; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.help("Outer byte cap for each complete NDJSON record; immutable JobLimits item cap must fit"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoints for this invocation; persisted work and attempts never renew")))
    };
    let start = operation("start", "Create an explicitly retained extraction-and-review population");
    let resume = operation("resume", "Authenticate the frozen semantic job and run pending records")
        .arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
            .action(clap::ArgAction::SetTrue)
            .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"));
    clap::Command::new("extract-semantic-job")
        .about("Start or resume retained INT4 extraction with complete correlated field review")
        .subcommand_required(true).subcommand(start).subcommand(resume)
}

#[derive(Clone, Copy, Debug)]
struct JobExecutionLimits {
    common: Int4Limits,
    task: TaskBudget,
    native: Int4SemanticExtractionBatchLimits,
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
        if !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0 {
            return Err(Failure::usage("semantic_job_storage_options"));
        }
        let common = self.semantic.validate(args)?;
        for mib in [s.journal_memory_mib, s.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > common.memory_bytes) {
                return Err(Failure::usage("semantic_job_memory_options"));
            }
        }
        Ok(common)
    }

    fn lifetime_limits(&self, args: &Int4Args, initial: Int4Limits,
        options: &Int4SemanticOptions, job: JobLimits) -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("semantic_job_limits"))?;
        let common = self.semantic.admit_options(args, initial, options)?;
        if job.max_input_bytes_per_item > args.max_input_bytes
            || options.limits.max_output_bytes > job.max_result_bytes as u64 {
            return Err(Failure::usage("semantic_job_input_or_combined_result_limit"));
        }
        let native = Int4SemanticExtractionBatchLimits {
            max_model_work: job.max_work.int4_model_work(),
            masks: ExtractionMaskBudget { per_mask: self.semantic.extraction.masks(),
                max_visits_per_item: self.semantic.extraction.mask_visits(),
                max_visits_per_run: job.max_work.mask_node_visits },
        };
        native.validate_item(options.max_model_work)
            .map_err(|_| Failure::usage("semantic_job_lifetime_work"))?;
        let transport = PopulationReadLimits {
            max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines,
        };
        transport.validate().map_err(|_| Failure::usage("semantic_job_transport"))?;
        Ok(JobExecutionLimits {
            task: self.semantic.extraction.budget(args, common), common, native, transport,
            io_reserve_bytes: product(IO_BUFFER_BYTES as u64, 2)?,
            journal_reserve_bytes: product(self.storage.journal_memory_mib, MIB)?,
            serialization_reserve_bytes: product(self.storage.serialization_memory_mib, MIB)?,
        })
    }

    fn defaults(&self, schema: String, task: TaskBudget, options: &Int4SemanticOptions)
        -> Result<ExtractionBatchArgs, Failure> {
        let defaults = self.semantic.extraction.shared_defaults(schema, task)?;
        let declaration = JsonValidationSchema::compile(&defaults.schema, self.semantic.extraction.compiler())
            .map_err(|_| CandidateError::Planning)?;
        if declaration.requires_verbatim_source()
            != (defaults.grounding == ExtractionBatchGrounding::SourceMembership) {
            return Err(CandidateError::Planning.into());
        }
        options.validate_schema(declaration.root()).map_err(|_| CandidateError::Planning)?;
        Ok(defaults)
    }
}

impl JobCommand {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches)
        -> Result<(Self, Int4Args), clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "semantic job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand, "semantic job operation refused")),
        };
        let args = JobArgs { semantic: SemanticArgs::from_matches(matches)?,
            storage: StorageArgs::from_arg_matches(matches)? };
        let mut common = Int4Args::from_arg_matches(matches)?;
        common.compute = compute::Args::from_matches(matches)?;
        // SemanticArgs owns its aggregate source/schema/review tokenizer policy.
        // The unrelated text-task policy remains at its byte-only default.
        Ok((Self { mode, args }, common))
    }

    pub(in crate::candidate_cli::int4) fn run_owned<R: Read + Send + 'static>(self,
        args: Int4Args, input: R, output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode.name();
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { schema_version: 1, operation, status: "error", code: failure.code,
                    exit_code: failure.exit, cancellation: failure.cancellation,
                    durable_progress_may_exist: true,
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
        {
            let _ = (self, args, common, input, output);
            Err(Failure::usage("owned_job_profile_unavailable"))
        }
    }
}
fn storage_failure(failure: job_cli::Failure) -> Failure {
    Failure { exit: failure.exit, code: failure.code, cancellation: None }
}
fn product(value: u64, count: u64) -> Result<u64, Failure> {
    value.checked_mul(count).ok_or_else(|| Failure::usage("semantic_job_memory_arithmetic"))
}
#[derive(Serialize)]
struct ErrorReport {
    schema_version: u32,
    operation: &'static str,
    status: &'static str,
    code: &'static str,
    exit_code: ErrorCode,
    #[serde(skip_serializing_if = "Option::is_none")]
    cancellation: Option<DecodeCancellationKind>,
    durable_progress_may_exist: bool,
    recovery: &'static str,
}
