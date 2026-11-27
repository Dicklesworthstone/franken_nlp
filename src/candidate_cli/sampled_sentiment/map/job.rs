//! Retained complete-document INT8 samples with immutable lifetime work.
//! Storage and replay use the existing authenticated owned-job contract.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use crate::{
    batch::{extract::ExtractionMaskBudget, sentiment::sampled::{
        SAMPLED_SENTIMENT_JOB_RECIPE_BYTES, quantized::Int8SampledSentimentBatchLimits}},
    candidate_cli::{batch::IO_BUFFER_BYTES, int4::extract_job::StorageArgs,
        jobs::{self as job_cli, RunMode}},
    jobs::{JobLimits, population::PopulationReadLimits},
    tasks::sentiment::sampled::quantized::long::memory::MIN_INT8_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES,
};

pub(in crate::candidate_cli) struct SampledSentimentDocumentJobCommand {
    pub(in crate::candidate_cli) mode: RunMode,
    pub(in crate::candidate_cli) args: JobArgs,
}
pub(in crate::candidate_cli) type JobCommand = SampledSentimentDocumentJobCommand;

pub(in crate::candidate_cli) struct JobArgs {
    pub(in crate::candidate_cli) document: SampledSentimentMapCommand,
    pub(in crate::candidate_cli) storage: StorageArgs,
}

pub(in crate::candidate_cli) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        let command = StorageArgs::augment_args(super::definition().name(name).about(about)
            .long_about("Retain and resume complete sampled sentiment documents from an exact {id,text,task_args?} NDJSON population on one resident INT8 candidate. Every original source is losslessly partitioned and all chunk/axis/sample generations must validate before its complete result can commit. Omitted task_args selects all four axes; records may choose axes and a complete tighter task budget. The seed, sampler, chunking, encoding, result limits and optional prefill schedule are authenticated and must match on resume. Requires --store-results, an existing protected --job-dir, --key-file, --job-id and immutable --limits. JobLimits.max_work owns all five native lifetime counters and mask_node_visits, including failed attempts. Chunks, samples, item count and restart never replenish that work. CLI work and mask flags cap one complete document. Pending documents keep their original item/chunk/axis/sample addresses. Repeat the COMPLETE original population unless explicit retained-input creation and replay are selected. Stdout contains metadata only. --materialize publishes complete verified results after all items commit. --compute-threads retains one team across every chunk and item and composes with bounded --prefill-rows; worker width may change on resume, while the declared prefill policy stays fixed. Compact safe-bpe-v1 shares one tokenizer-work ledger across all probes, chunk plans, records and refused preparations in this invocation. That finite policy is authenticated; its invocation allowance does not replenish persisted native work or attempts. Complete planner, durable population, sampler, result and physical-schedule memory are admitted before key/model/population IO. No flush controls, automatic retry, partial-document commits, truncation or calibrated-confidence claim. Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64.")
            .after_help(job_cli::INPUT_RETENTION_HELP)
            .after_long_help("Repeat all semantic options on resume. Original source bytes and the actual private sampler recipe authenticate before tail repair or inference. A pending document restarts as one durable unit. Diagnostics never contain source, samples, seed or private recipe hashes. Reporting failure may occur after committed progress; authenticate status or explicitly resume before deciding what remains.")
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population; omit with --use-retained-input; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.help("Outer cap on each complete NDJSON envelope; immutable stored input ceiling must fit"))
            .mut_arg("max_result_bytes", |arg| arg.help("Complete sample bundle per chunk; stored document output has its own cap"))
            .mut_arg("max_map_result_bytes", |arg| arg.help("Complete committed document result; must fit JobLimits.max_result_bytes"))
            .mut_arg("max_forward_positions", |arg| arg.help("Per-document cold forward ceiling across every chunk, axis and sample"))
            .mut_arg("max_projected_logits", |arg| arg.help("Per-document projected output-logit ceiling"))
            .mut_arg("max_attention_pairs", |arg| arg.help("Per-document attention-pair ceiling"))
            .mut_arg("max_dot_products", |arg| arg.help("Per-document integer dot-product ceiling"))
            .mut_arg("max_multiply_accumulates", |arg| arg.help("Per-document multiply-accumulate ceiling"))
            .mut_arg("max_mask_node_visits", |arg| arg.help("Per-document complete mask allowance; lifetime ceiling comes from --limits"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoints for this invocation; durable work and attempts never renew")))
            .mut_arg("store_results", |arg| arg.help("Required consent to retain private complete-document scores and explanations"))
            .mut_arg("materialize", |arg| arg.help("Publish verified materialized.ndjson after ALL items commit; never replace it"));
        if name == "resume" {
            command.arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
                .action(clap::ArgAction::SetTrue)
                .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"))
        } else { command }
    };
    clap::Command::new("sentiment-sampled-document-job")
        .about("Start or resume retained complete-document INT8 sentiment samples")
        .subcommand_required(true)
        .subcommand(operation("start", "Create a retained complete-document sampled population"))
        .subcommand(operation("resume", "Authenticate and sample every pending complete document"))
}

pub(in crate::candidate_cli) struct JobExecutionLimits {
    pub(in crate::candidate_cli) document: MapLimits,
    pub(in crate::candidate_cli) planning: SampledSentimentLimits,
    pub(in crate::candidate_cli) native: Int8SampledSentimentBatchLimits,
    pub(in crate::candidate_cli) transport: PopulationReadLimits,
    pub(in crate::candidate_cli) buffer_bytes: u64,
    pub(in crate::candidate_cli) execution_bytes: u64,
    pub(in crate::candidate_cli) preparation_reserve_bytes: u64,
    pub(in crate::candidate_cli) io_reserve_bytes: u64,
    pub(in crate::candidate_cli) journal_reserve_bytes: u64,
    pub(in crate::candidate_cli) serialization_reserve_bytes: u64,
}
impl JobArgs {
    /// Pure option validation precedes even the bounded immutable limits file.
    pub(in crate::candidate_cli) fn validate(&self) -> Result<MapLimits, Failure> {
        let storage = &self.storage;
        storage.input_retention.limits().map_err(storage_failure)?;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !storage.store_results || !local(&storage.job_dir) || !local(&storage.key_file)
            || !local(&storage.limits_file) || !(1..=1024 * 1024).contains(&storage.max_stream_mib)
            || !(1..=1_000_000_000).contains(&storage.max_input_lines)
            || storage.journal_memory_mib == 0 || storage.serialization_memory_mib == 0 {
            return Err(Failure::usage("int8_sampled_document_job_storage"));
        }
        let document = self.document.validate()
            .map_err(|_| Failure::usage("int8_sampled_document_job_options"))?;
        for mib in [storage.journal_memory_mib, storage.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > document.shared.memory_bytes) {
                return Err(Failure::usage("int8_sampled_document_job_memory_options"));
            }
        }
        Ok(document)
    }

    /// JobLimits is the sole authority for lifetime work. Transport, item,
    /// sample and attempt counts can never multiply those immutable ceilings.
    pub(in crate::candidate_cli) fn lifetime_limits(&self, mut document: MapLimits, job: JobLimits)
        -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("job_limits"))?;
        if job.max_input_bytes_per_item > document.common.max_input_bytes || job.max_id_bytes > 256 {
            return Err(Failure::usage("int8_sampled_document_job_input_ceiling"));
        }
        if document.mapping.max_result_bytes > job.max_result_bytes as u64
            || document.task.max_output_bytes > job.max_result_bytes as u64 {
            return Err(Failure::usage("int8_sampled_document_job_result_ceiling"));
        }
        let sampled = &self.document.sampled;
        let mut planning = sampled.task_limits();
        planning.max_input_bytes = planning.max_input_bytes.min(job.max_input_bytes_per_item);
        document.mapping.chunks.max_input_bytes = job.max_input_bytes_per_item;
        let native = Int8SampledSentimentBatchLimits {
            max_item_work: sampled.work_ceiling(), max_model_work: job.max_work.model,
            masks: ExtractionMaskBudget { per_mask: sampled.masks(),
                max_visits_per_item: sampled.max_mask_node_visits,
                max_visits_per_run: job.max_work.mask_node_visits },
            max_sampler_bytes: json_sampling_workspace_bytes().map_err(|_| CandidateError::Arguments)?,
        };
        native.validate().map_err(|_| Failure::usage("int8_sampled_document_job_lifetime_work"))?;
        let transport = PopulationReadLimits {
            max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines,
        };
        transport.validate().map_err(|_| Failure::usage("job_transport"))?;
        let mut memory = Int8SampledSentimentMapMemoryLimits {
            context_tokens: document.common.context_tokens,
            preparation_reserve_bytes: MIN_INT8_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES,
            allocator_reserve_bytes: 64 * MIB, max_axes: SentimentAxis::ALL.len(),
            max_sampler_bytes: native.max_sampler_bytes,
        }.requirements(document.task, planning, document.mapping, &sampled.options()?, &document.encoding)
            .map_err(|_| Failure::usage("int8_sampled_document_job_preparation"))?;
        let preparation_reserve_bytes = sum(&[memory.input_bytes, SAMPLED_SENTIMENT_JOB_RECIPE_BYTES as u64])?;
        let io_reserve_bytes = product(IO_BUFFER_BYTES as u64, 2)?;
        let journal_reserve_bytes = product(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = product(self.storage.serialization_memory_mib, MIB)?;
        let buffer_bytes = sum(&[self.storage.input_retention.buffer_bytes(job).map_err(storage_failure)?,
            preparation_reserve_bytes, io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes])?;
        // Host input ownership already includes the complete map compiler and
        // recipe. Retain the CLI's separate preparation charge alongside it.
        memory.input_bytes = buffer_bytes;
        let prefill_bytes = document.prefill.map(Int8PrefillLimits::validate).transpose()
            .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
        memory.workspace_bytes = sum(&[memory.workspace_bytes, prefill_bytes])?;
        let execution_bytes = sum(&[memory.input_bytes, memory.native.kv_bytes,
            memory.workspace_bytes, memory.output_reservation_bytes, document.shared.compute.workspace_bytes])?;
        let execution_peak = sum(&[execution_bytes, document.shared.weight_bytes, 192 * MIB, 65_536])?;
        let loading_peak = sum(&[document.shared.weight_bytes, 192 * MIB, 64 * 1024, 65_536])?;
        document.minimum_memory_bytes = sum(&[document.shared.preparation_bytes,
            execution_peak.max(loading_peak)])?;
        if document.minimum_memory_bytes > document.shared.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit,
                code: "int8_sampled_document_job_population_memory", cancellation: None });
        }
        document.memory = memory;
        Ok(JobExecutionLimits { document, planning, native, transport, buffer_bytes, execution_bytes,
            preparation_reserve_bytes, io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes })
    }
}

impl SampledSentimentDocumentJobCommand {
    pub(in crate::candidate_cli) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "sampled document job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand,
                "sampled document job operation refused")),
        };
        Ok(Self { mode, args: JobArgs { document: SampledSentimentMapCommand::from_matches(matches)?,
            storage: StorageArgs::from_arg_matches(matches)? } })
    }

    pub(in crate::candidate_cli) fn run_owned<R: Read + Send + 'static>(self,
        input: R, output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode.name();
        match self.execute_owned(input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => report_job_failure(operation, failure, diagnostics),
        }
    }

    pub(in crate::candidate_cli) fn execute_owned<R: Read + Send + 'static>(self,
        input: R, output: &mut impl Write) -> Result<(), Failure> {
        self.args.storage.input_retention.validate(self.mode, &self.args.document.sampled.input)
            .map_err(storage_failure)?;
        let admitted = self.args.validate()?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::map::job::execute(self.mode, self.args, admitted, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        { let _ = (self, admitted, input, output);
            Err(Failure::usage("int8_sampled_document_job_profile_unavailable")) }
    }
}

fn report_job_failure(operation: &'static str, failure: Failure, diagnostics: &mut impl Write) -> ExitCode {
    let report = ErrorReport { protocol: "fnlp-candidate-int8-sentiment-sampled-document-job-v1",
        schema_version: 1, operation, status: "error", code: failure.code,
        exit_code: failure.exit, cancellation: failure.cancellation,
        durable_progress_may_exist: true,
        recovery: "authenticate stored job status or explicitly resume; do not assume rollback" };
    // Diagnostic IO happens after a terminal cause is already known. A broken
    // stderr must not turn cancellation or deadline exhaustion into Generic.
    if publish(&report, job_cli::REPORT_BYTES, diagnostics).is_err() && failure.cancellation.is_none() {
        return ErrorCode::Generic.as_process_exit();
    }
    failure.exit.as_process_exit()
}

pub(in crate::candidate_cli) fn storage_failure(error: job_cli::Failure) -> Failure {
    Failure { exit: error.exit, code: error.code, cancellation: None }
}
fn product(value: u64, count: u64) -> Result<u64, Failure> {
    value.checked_mul(count).ok_or_else(|| Failure::usage("int8_sampled_document_job_memory_arithmetic"))
}
#[derive(Serialize)]
struct ErrorReport {
    protocol: &'static str, schema_version: u32, operation: &'static str, status: &'static str,
    code: &'static str, exit_code: ErrorCode,
    cancellation: Option<crate::native_engine::decode::DecodeCancellationKind>,
    durable_progress_may_exist: bool, recovery: &'static str,
}

#[cfg(test)] mod tests;
