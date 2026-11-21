//! Retained complete-document root-object extraction and full-source review.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use crate::{
    batch::extract::semantic::document::Int4SemanticObjectDocumentBatchLimits,
    candidate_cli::{batch::IO_BUFFER_BYTES, jobs::{self as job_cli, RunMode}},
    jobs::{JobLimits, population::PopulationReadLimits},
    native_engine::decode::DecodeCancellationKind,
    validation::grounded_fields::GroundingBudget,
};
use crate::candidate_cli::int4::extract_job::StorageArgs;

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
    semantic: SemanticObjectArgs,
    storage: StorageArgs,
}

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        StorageArgs::augment_args(super::definition().name(name).about(about)
            .long_about(concat!(
                "Retain and resume complete-document root-object extraction with complete-source correlated scalar reviews. ",
                "Every record preserves its original source, extracts all lossless chunks, reconciles complementary root properties against the exact ORIGINAL schema, then reviews each configured scalar claim across the complete source. ",
                "Only the complete combined result commits; conflicts, missing required fields, incomplete execution or a failed review retain no partial extraction. ",
                "Requires --store-results, an existing protected --job-dir, original --job-id and --key-file, immutable --limits, exact --schema and --semantic-options. ",
                "Resume authenticates the original population, model/group identity, schema derivation, reconciliation limits, extraction and review chunk geometry, claims, policies, compact encoding and lifetime work before repair or inference. Only pending records run. ",
                "All five native axes and extraction-mask work in --limits are lifetime ceilings across records and attempts. Each attempt spends its complete declaration, including failed or interrupted attempts. ",
                "The declared per-document grounding field/match/scan budgets are funded for every permitted lifetime attempt with checked arithmetic. Resume cannot renew them. ",
                "Provide the COMPLETE original NDJSON population on each invocation, or explicitly retain input at start and resume with --use-retained-input. Records are {id,text}; non-null task_args and flush controls refuse. ",
                "--input-encoding safe-bpe-v1 shares one nonrenewable per-attempt ledger across original preparation, exact reconstruction, extraction probes and all review heads. ",
                "Serial compute is the default; --compute-threads 2..64 keeps one admitted numerical team across the full population. Physical team width may change on resume without renewing authenticated work. ",
                "The complete memory peak is admitted before key, schema, population or model IO after bounded fixed options and limits reads. Source conversion and digest-pinned snapshots use the same native path. ",
                "Stdout contains bounded completion metadata only. --materialize publishes verified materialized.ndjson after all records commit. Failure after durable acknowledgement never implies rollback. ",
                "These are experimental uncalibrated same-model diagnostics, with no automatic acceptance, repair or retry. Selected-row projection, cohorts, packed prefill, sampling, thinking and tools are unavailable. ",
                "Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64."))
            .after_help(job_cli::INPUT_RETENTION_HELP)
            .after_long_help("Repeat every source/schema/review/encoding declaration exactly on resume. Each extraction chunk has its original output bound; semantic limits.max_output_bytes and immutable JobLimits.max_result_bytes cover the complete map, reconciled object and all full-source reviews.")
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population; omit with --use-retained-input; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.help("Outer cap for each original NDJSON record and complete source; immutable JobLimits item cap must fit"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoints for this invocation; authenticated attempts and work never renew")))
    };
    let start = operation("start", "Create a retained population of complete reviewed objects");
    let resume = operation("resume", "Authenticate the original whole-document job and run pending records")
        .arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
            .action(clap::ArgAction::SetTrue)
            .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse"));
    clap::Command::new("extract-semantic-document-job")
        .about("Start or resume retained INT4 whole-document object extraction and full-source review")
        .subcommand_required(true).subcommand(start).subcommand(resume)
}

#[derive(Clone, Copy, Debug)]
struct JobExecutionLimits {
    object: extract_map::object::ObjectMapLimits,
    task: TaskBudget,
    review: Int4FaithfulnessMapLimits,
    native: Int4SemanticObjectDocumentBatchLimits,
    transport: PopulationReadLimits,
    io_reserve_bytes: u64,
    journal_reserve_bytes: u64,
    serialization_reserve_bytes: u64,
}
impl JobArgs {
    fn validate(&self, args: &Int4Args) -> Result<extract_map::object::ObjectMapLimits, Failure> {
        let s = &self.storage;
        s.input_retention.limits().map_err(storage_failure)?;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !s.store_results || !local(&s.job_dir) || !local(&s.key_file) || !local(&s.limits_file)
            || !(1..=1024 * 1024).contains(&s.max_stream_mib)
            || !(1..=1_000_000_000).contains(&s.max_input_lines)
            || s.journal_memory_mib == 0 || s.serialization_memory_mib == 0 {
            return Err(Failure::usage("semantic_object_job_storage_options"));
        }
        let object = self.semantic.validate(args)?;
        for mib in [s.journal_memory_mib, s.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > object.map.common.memory_bytes) {
                return Err(Failure::usage("semantic_object_job_memory_options"));
            }
        }
        Ok(object)
    }

    fn lifetime_limits(&self, args: &Int4Args, object: extract_map::object::ObjectMapLimits,
        options: &Int4SemanticOptions, job: JobLimits) -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("semantic_object_job_limits"))?;
        let common = object.map.common;
        let review = self.semantic.review_limits(args, common, options)?;
        if job.max_input_bytes_per_item > args.max_input_bytes
            || options.limits.max_output_bytes > job.max_result_bytes as u64 {
            return Err(Failure::usage("semantic_object_job_input_or_combined_result_limit"));
        }
        // Native/mask authority is copied exactly from the authenticated job,
        // never multiplied by population size or permitted retries.
        let per = object.mapping().mapping.verification;
        let native = Int4SemanticObjectDocumentBatchLimits {
            max_model_work: job.max_work.int4_model_work(),
            max_mask_node_visits: job.max_work.mask_node_visits,
            max_verification: lifetime_grounding(per, job.max_attempts)?,
        };
        native.validate_item(options.max_model_work,
            object.mapping().mapping.mapping.max_mask_visits, per)
            .map_err(|_| Failure::usage("semantic_object_job_lifetime_work"))?;
        let transport = PopulationReadLimits {
            max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines,
        };
        transport.validate().map_err(|_| Failure::usage("semantic_object_job_transport"))?;
        Ok(JobExecutionLimits { object,
            task: self.semantic.object.map.extraction.budget(args, common),
            review, native, transport, io_reserve_bytes: product(IO_BUFFER_BYTES as u64, 2)?,
            journal_reserve_bytes: product(self.storage.journal_memory_mib, MIB)?,
            serialization_reserve_bytes: product(self.storage.serialization_memory_mib, MIB)? })
    }
}

fn lifetime_grounding(per: GroundingBudget, attempts: u64) -> Result<GroundingBudget, Failure> {
    let invalid = || Failure::usage("semantic_object_job_grounding_lifetime");
    if attempts == 0 { return Err(invalid()); }
    let count = |value: usize| u64::try_from(value).ok().and_then(|n| n.checked_mul(attempts))
        .and_then(|n| usize::try_from(n).ok()).ok_or_else(invalid);
    Ok(GroundingBudget { max_fields: count(per.max_fields)?, max_matches: count(per.max_matches)?,
        max_scan_steps: per.max_scan_steps.checked_mul(attempts).ok_or_else(invalid)? })
}

impl JobCommand {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches)
        -> Result<(Self, Int4Args), clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "semantic document job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand,
                "semantic document job operation refused")),
        };
        let args = JobArgs { semantic: SemanticObjectArgs::from_matches(matches)?,
            storage: StorageArgs::from_arg_matches(matches)? };
        let mut common = Int4Args::from_arg_matches(matches)?;
        common.compute = compute::Args::from_matches(matches)?;
        // The object-map options own the complete-document tokenizer policy.
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
        let object = self.args.validate(&args)?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::execute(self.mode, args, self.args, object, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        {
            let _ = (self, args, object, input, output);
            Err(Failure::usage("owned_job_profile_unavailable"))
        }
    }
}
fn storage_failure(failure: job_cli::Failure) -> Failure {
    Failure { exit: failure.exit, code: failure.code, cancellation: None }
}
fn product(value: u64, count: u64) -> Result<u64, Failure> {
    value.checked_mul(count).ok_or_else(|| Failure::usage("semantic_object_job_memory_arithmetic"))
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
