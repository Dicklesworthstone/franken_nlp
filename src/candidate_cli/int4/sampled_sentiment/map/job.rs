//! Complete sampled documents with authenticated durable result retention.
//! The immutable journal owns lifetime work across chunks, samples and restarts.
#![cfg_attr(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"))), allow(dead_code))]

use super::*;
use crate::{
    batch::{extract::ExtractionMaskBudget, sentiment::sampled::{SampledSentimentBatchLimits,
        SAMPLED_SENTIMENT_JOB_RECIPE_BYTES}},
    candidate_cli::{batch::IO_BUFFER_BYTES, int4::extract_job::StorageArgs,
        jobs::{self as job_cli, RunMode}},
    jobs::{JobLimits, population::PopulationReadLimits},
    tasks::sentiment::sampled::long::memory::MIN_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES,
};

#[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")))]
mod runtime;
#[cfg(test)] mod tests;

pub(in crate::candidate_cli::int4) struct JobCommand { mode: RunMode, args: JobArgs }
struct JobArgs { document: MapArgs, storage: StorageArgs }

pub(in crate::candidate_cli::int4) fn definition() -> clap::Command {
    let operation = |name: &'static str, about: &'static str| {
        // The complete-map command already owns encoding and compute flags.
        let command = StorageArgs::augment_args(super::definition().name(name).about(about)
            .long_about("Retain and resume complete sampled sentiment documents from a frozen {id,text,task_args?} NDJSON population. Losslessly partition every source, preflight all chunks and commit only the complete independent sample bundles plus exact scalar-weighted document replicates. Omitted task_args selects all four axes; axes and a complete tighter task budget may vary per record. Seed, sampler, chunking, encoding, result limits, prefix reuse and physical prefill policy are authenticated and must match on resume. Requires --store-results, an existing protected --job-dir, --key-file, --job-id and immutable --limits. All five native counters and mask_node_visits in JobLimits.max_work are lifetime allowances, including failed attempts; chunks, samples, item count and restart never replenish them. CLI work and mask flags cap ONE complete document. Pending documents restart with their original item/chunk/axis/sample addresses. Repeat the COMPLETE original population unless --spool-input and --use-retained-input explicitly enable authenticated private input retention. Stdout contains metadata only. --materialize publishes verified complete results only after all records commit. One resident model, vocabulary and native allocation serve the invocation. --compute-threads 2..64 retains one numerical team; alternatively --prefill-rows 1..64 uses serial layer-major prompts. --reuse-prefill keeps only the current chunk/axis prefix. Compact safe-bpe-v1 has one per-document tokenizer ledger shared by probes and plans. Complete planner, durable input, sampler, result and schedule memory are admitted before key/model/population IO. No snapshot saving, flush controls, automatic retry, partial-document commits, truncation, calibrated confidence or default sentiment promotion. Requires metadata-store and asupersync-runtime on Linux x86-64/AArch64.")
            .after_help(format!("Repeat every semantic option on resume. The authenticated recipe refuses changed partition, reducer, sampler or encoding policies before tail repair or inference. Reused prefixes are never persisted and do not reduce conservative lifetime cold-work debits.\n\n{}", job_cli::INPUT_RETENTION_HELP))
            .mut_arg("input", |arg| arg.help("COMPLETE original NDJSON population; omit with --use-retained-input; '-' reads stdin"))
            .mut_arg("max_input_bytes", |arg| arg.help("Outer ceiling for each complete NDJSON envelope; immutable stored input ceiling must fit"))
            .mut_arg("max_result_bytes", |arg| arg.help("Complete sample bundle per chunk; --max-map-result-bytes caps the committed document"))
            .mut_arg("max_map_result_bytes", |arg| arg.help("Complete committed document result; must fit JobLimits.max_result_bytes"))
            .mut_arg("max_forward_positions", |arg| arg.help("Per-document cold forward ceiling across every chunk, axis and sample"))
                .mut_arg("max_projected_logits", |arg| arg.help("Per-document projected output-row ceiling"))
            .mut_arg("max_attention_pairs", |arg| arg.help("Per-document attention-pair ceiling"))
            .mut_arg("max_dot_products", |arg| arg.help("Per-document grouped dot-product ceiling"))
            .mut_arg("max_multiply_accumulates", |arg| arg.help("Per-document multiply-accumulate ceiling"))
            .mut_arg("max_mask_node_visits", |arg| arg.help("Per-document complete mask allowance; lifetime ceiling comes from --limits"))
            .mut_arg("max_checkpoints", |arg| arg.help("Cooperative checkpoints for this invocation; durable work and attempts never renew")))
            .mut_arg("store_results", |arg| arg.help("Required consent to retain private complete-document scores and explanations"));
        if name == "resume" {
            command.arg(clap::Arg::new("discard_uncommitted_tail").long("discard-uncommitted-tail")
                .action(clap::ArgAction::SetTrue)
                .help("After authentication, explicitly discard uncommitted spool bytes/stages; otherwise refuse them"))
        } else { command }
    };
    clap::Command::new("sentiment-sampled-document-job")
        .about("Start or resume retained complete-document sentiment samples")
        .subcommand_required(true)
        .subcommand(operation("start", "Create a retained complete-document sampled population"))
        .subcommand(operation("resume", "Authenticate and sample every pending complete document"))
}

#[derive(Clone, Copy, Debug)]
struct JobExecutionLimits {
    common: Int4Limits,
    task: TaskBudget,
    planning: SampledSentimentLimits,
    mapping: SampledSentimentMapLimits,
    native: SampledSentimentBatchLimits,
    memory: SampledSentimentMapMemoryRequirement,
    transport: PopulationReadLimits,
    buffer_bytes: u64,
    execution_bytes: u64,
    preparation_reserve_bytes: u64,
    io_reserve_bytes: u64,
    journal_reserve_bytes: u64,
    serialization_reserve_bytes: u64,
}

impl JobArgs {
    /// Check options before opening even the bounded immutable limits file.
    fn validate(&self, args: &Int4Args) -> Result<MapLimits, Failure> {
        let storage = &self.storage;
        storage.input_retention.limits().map_err(storage_failure)?;
        let local = |path: &std::path::Path| !path.as_os_str().is_empty() && path.as_os_str() != "-";
        if !storage.store_results || !local(&storage.job_dir) || !local(&storage.key_file)
            || !local(&storage.limits_file) || !(1..=1024 * 1024).contains(&storage.max_stream_mib)
            || !(1..=1_000_000_000).contains(&storage.max_input_lines)
            || storage.journal_memory_mib == 0 || storage.serialization_memory_mib == 0 {
            return Err(Failure::usage("explicit_sampled_document_job_storage"));
        }
        let document = self.document.validate(args)?;
        for mib in [storage.journal_memory_mib, storage.serialization_memory_mib] {
            if mib.checked_mul(MIB).is_none_or(|bytes| bytes > document.common.memory_bytes) {
                return Err(Failure::usage("sampled_document_job_memory_options"));
            }
        }
        Ok(document)
    }

    fn lifetime_limits(&self, args: &Int4Args, document: MapLimits, job: JobLimits)
        -> Result<JobExecutionLimits, Failure> {
        job.validate().map_err(|_| Failure::usage("job_limits"))?;
        if job.max_input_bytes_per_item > args.max_input_bytes || job.max_id_bytes > 256 {
            return Err(Failure::usage("sampled_document_job_input_ceiling"));
        }
        if document.mapping.max_result_bytes > job.max_result_bytes as u64
            || document.common.result_bytes > job.max_result_bytes {
            return Err(Failure::usage("sampled_document_job_result_ceiling"));
        }
        let mut common = document.common;
        let sampled = self.document.sampled;
        let task = sampled.budget(args, common, SentimentAxis::ALL.len())?;
        let mut planning = document.planning;
        planning.max_input_bytes = planning.max_input_bytes.min(job.max_input_bytes_per_item);
        let mut mapping = document.mapping;
        mapping.chunks.max_input_bytes = job.max_input_bytes_per_item;
        let native = SampledSentimentBatchLimits {
            max_item_work: sampled.work_ceiling(), max_model_work: job.max_work.int4_model_work(),
            masks: ExtractionMaskBudget { per_mask: sampled.masks(),
                max_visits_per_item: sampled.max_mask_node_visits,
                max_visits_per_run: job.max_work.mask_node_visits },
            max_sampler_bytes: json_sampling_workspace_bytes().map_err(|_| CandidateError::Arguments)?,
        };
        native.validate().map_err(|_| Failure::usage("sampled_document_job_lifetime_work"))?;
        let transport = PopulationReadLimits { max_stream_bytes: product(self.storage.max_stream_mib, MIB)?,
            max_lines: self.storage.max_input_lines };
        transport.validate().map_err(|_| Failure::usage("job_transport"))?;
        // All retained chunk plans and results belong to one host-owned
        // document preparation charge; a leaf-task compiler quote is too small.
        let encoding = self.document.input_encoding(args)?;
        let mut memory = SampledSentimentMapMemoryLimits {
            context_tokens: args.context_tokens,
            preparation_reserve_bytes: MIN_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES,
            allocator_reserve_bytes: ALLOCATOR_BYTES, max_axes: SentimentAxis::ALL.len(),
            max_sampler_bytes: native.max_sampler_bytes,
        }.requirements(args.group_width, task, planning, mapping, &sampled.options(args)?, &encoding)
            .map_err(|_| Failure::usage("sampled_document_job_preparation"))?;
        let preparation_reserve_bytes = sum(&[memory.input_bytes, SAMPLED_SENTIMENT_JOB_RECIPE_BYTES as u64])?;
        let io_reserve_bytes = product(IO_BUFFER_BYTES as u64, 2)?;
        let journal_reserve_bytes = product(self.storage.journal_memory_mib, MIB)?;
        let serialization_reserve_bytes = product(self.storage.serialization_memory_mib, MIB)?;
        let buffer_bytes = sum(&[self.storage.input_retention.buffer_bytes(job).map_err(storage_failure)?,
            preparation_reserve_bytes, io_reserve_bytes, journal_reserve_bytes, serialization_reserve_bytes])?;
        memory.input_bytes = buffer_bytes;
        let prefill = common.prefill.map(Int4PrefillLimits::validate).transpose()
            .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
        memory.workspace_bytes = sum(&[memory.workspace_bytes, prefill])?;
        let execution_bytes = sum(&[memory.input_bytes, memory.native.kv_bytes, memory.workspace_bytes,
            memory.output_reservation_bytes, common.compute.workspace_bytes])?;
        let load_stage = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
            else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
        common.minimum_memory_bytes = sum(&[common.preparation_bytes, common.load.weight_payload_bytes,
            ALLOCATOR_BYTES, load_stage.max(execution_bytes), CLEANUP_BYTES])?;
        if common.minimum_memory_bytes > common.memory_bytes {
            return Err(Failure { exit: ErrorCode::AdmissionOrResourceLimit,
                code: "sampled_document_job_population_memory", cancellation: None });
        }
        Ok(JobExecutionLimits { common, task, planning, mapping, native, memory, transport,
            buffer_bytes, execution_bytes, preparation_reserve_bytes, io_reserve_bytes,
            journal_reserve_bytes, serialization_reserve_bytes })
    }
}

impl JobCommand {
    pub(in crate::candidate_cli::int4) fn from_matches(matches: &clap::ArgMatches)
        -> Result<(Self, Int4Args), clap::Error> {
        let (name, matches) = matches.subcommand().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingSubcommand, "sampled document job operation required"))?;
        let mode = match name {
            "start" => RunMode::Start,
            "resume" => RunMode::Resume { discard_uncommitted: matches.get_flag("discard_uncommitted_tail") },
            _ => return Err(clap::Error::raw(clap::error::ErrorKind::InvalidSubcommand,
                "sampled document job operation refused")),
        };
        let job = JobArgs { document: MapArgs::from_matches(matches)?,
            storage: StorageArgs::from_arg_matches(matches)? };
        let mut args = Int4Args::from_arg_matches(matches)?;
        args.input_encoding = input_encoding::InputEncodingArgs::from_matches(matches)?;
        args.compute = compute::Args::from_matches(matches)?;
        Ok((Self { mode, args: job }, args))
    }
    pub(in crate::candidate_cli::int4) fn run_owned<R: Read + Send + 'static>(self, args: Int4Args,
        input: R, output: &mut impl Write, diagnostics: &mut impl Write) -> ExitCode {
        let operation = self.mode.name();
        match self.execute_owned(args, input, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(failure) => {
                let report = ErrorReport { protocol: "fnlp-candidate-int4-sentiment-sampled-document-job-v1",
                    schema_version: 1, operation, status: "error", code: failure.code,
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
    fn execute_owned<R: Read + Send + 'static>(self, args: Int4Args, input: R, output: &mut impl Write)
        -> Result<(), Failure> {
        self.args.storage.input_retention.validate(self.mode, &args.input).map_err(storage_failure)?;
        let admitted = self.args.validate(&args)?;
        #[cfg(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")))]
        { runtime::execute(self.mode, args, self.args, admitted, input, output) }
        #[cfg(not(all(feature = "metadata-store", feature = "asupersync-runtime", target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64"))))]
        { let _ = (self, args, admitted, input, output);
            Err(Failure::usage("sampled_document_job_profile_unavailable")) }
    }
}

fn storage_failure(error: job_cli::Failure) -> Failure {
    Failure { exit: error.exit, code: error.code, cancellation: None }
}
fn product(value: u64, count: u64) -> Result<u64, Failure> {
    value.checked_mul(count).ok_or_else(|| Failure::usage("sampled_document_job_memory_arithmetic"))
}
#[derive(Serialize)]
struct ErrorReport {
    protocol: &'static str, schema_version: u32, operation: &'static str, status: &'static str,
    code: &'static str, exit_code: ErrorCode,
    cancellation: Option<crate::native_engine::decode::DecodeCancellationKind>,
    durable_progress_may_exist: bool, recovery: &'static str,
}
