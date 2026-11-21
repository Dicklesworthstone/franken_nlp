//! One charged resident run, complete document frames and terminal drain.
use super::*;
use std::{io::{BufRead, BufReader}, time::{Duration, Instant}};
use crate::{
    NlpEngine, MemoryClass,
    batch::{BatchCode, BatchFault},
    candidate_cli::batch::CandidateWriter,
    hosted::{CancellationToken, HostedError, NativeLimits, RunLimits,
        SampledSentimentDocumentBatchHostConfig, SampledSentimentMapHostConfig},
    native_engine::{decode::DecodeCancellationKind,
        generation::grouped_int4::INT4_SOURCE_FORMAT,
        strict_int4::{prefill::INT4_LAYER_MAJOR_PREFILL_VERSION,
            weights::snapshot::INT4_SNAPSHOT_VERSION}},
};
use super::super::super::runtime::cancelled;

fn batch_failure(fault: BatchFault) -> Failure {
    if let Some(cause) = fault.cancellation { return cancelled(cause); }
    use BatchCode::*;
    let (exit, code) = match fault.code {
        InvalidLimits => (ErrorCode::Usage, "sampled_sentiment_corpus_limits"),
        InvalidUtf8 | InvalidJson | InvalidEnvelope | InvalidId | DuplicateId | Planning | InputIo =>
            (ErrorCode::InputDecodeOrParse, "sampled_sentiment_corpus_input"),
        LineLimit | DocumentLimit | EpochIdLimit | InputLimit | RequestLimit
            | OutputLineLimit | OutputLimit | WorkLimit =>
            (ErrorCode::BudgetOrTimeout, "sampled_sentiment_corpus_budget"),
        Admission | Allocation => (ErrorCode::AdmissionOrResourceLimit, "sampled_sentiment_corpus_admission"),
        Execution => (ErrorCode::StructuredTaskNoResult, "sampled_sentiment_corpus_no_result"),
        Cancelled => (ErrorCode::Generic, "sampled_sentiment_corpus_missing_cancellation"),
        SequenceOverflow | InvalidExecution | Serialization | OutputIo =>
            (ErrorCode::Generic, "sampled_sentiment_corpus_runtime_or_delivery"),
    };
    Failure { exit, code, cancellation: None }
}

fn execution_failure(error: HostedError) -> Failure {
    match error {
        HostedError::BatchSetup(fault) => batch_failure(fault),
        HostedError::Batch(error) => batch_failure(error.fault),
        HostedError::Stopped { stop, task_error } => {
            if matches!(stop.kind, DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit) {
                if let Some(error) = task_error {
                    let mut failure = execution_failure(*error);
                    failure.cancellation = Some(stop.kind);
                    return failure;
                }
            }
            cancelled(stop.kind)
        }
        error => super::super::runtime::host_failure(error),
    }
}

fn host_config(args: &Int4Args, options: CorpusArgs, limits: CorpusLimits, remaining: Duration)
    -> Result<SampledSentimentDocumentBatchHostConfig, CandidateError> {
    let sampled = options.document.sampled;
    Ok(SampledSentimentDocumentBatchHostConfig {
        document: SampledSentimentMapHostConfig {
            native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
                run: RunLimits { max_elapsed: remaining, max_checkpoints: args.max_checkpoints,
                    cleanup_reserve_bytes: CLEANUP_BYTES } },
            planning: limits.document.planning, mapping: limits.document.mapping,
            options: sampled.options(args)?, preparation_reserve_bytes: limits.common.preparation_bytes,
            max_sampler_bytes: limits.native.max_sampler_bytes,
            mask_limits: limits.native.masks.per_mask,
            max_mask_node_visits: limits.native.masks.max_visits_per_item,
            prefill: limits.common.prefill, reuse_prefill: sampled.reuse_prefill,
            selected_rows: sampled.selected_rows()?,
        },
        task: sampled.budget(args, limits.common, SentimentAxis::ALL.len())?,
        encoding: options.document.input_encoding(args)?,
        transport: limits.transport, io_reserve_bytes: limits.io_bytes,
        max_model_work: limits.native.max_model_work,
        max_mask_node_visits: limits.native.masks.max_visits_per_run,
    })
}

fn check_host_quote(args: &Int4Args, limits: CorpusLimits,
    config: &SampledSentimentDocumentBatchHostConfig) -> Result<(), Failure> {
    let required = if limits.common.compute.threads > 1 {
        config.requirements_with_threads(args.group_width, limits.common.compute.threads)
    } else { config.requirements(args.group_width) }.map_err(execution_failure)?;
    if required != limits.memory {
        return Err(Failure::usage("sampled_sentiment_corpus_memory_quote"));
    }
    Ok(())
}

pub(super) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    args: Int4Args, options: CorpusArgs, _admitted: CorpusLimits, input: R, output: W)
    -> Result<(), Failure> {
    let started = Instant::now();
    let limits = options.validate(&args)?;
    let common = limits.common;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or_else(|| cancelled(DecodeCancellationKind::Deadline));
    let mut config = host_config(&args, options, limits, remaining()?)?;
    check_host_quote(&args, limits, &config)?;
    let engine = NlpEngine::builder().resource_config(compute::resource_config(common))
        .build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    let _preparation = lease.reserve(MemoryClass::JobBuffers, limits.cli_preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let reader: Box<dyn BufRead + Send> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            crate::local_io::open_document(&args.input).map_err(|_| CandidateError::Input)?))
    };
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = crate::candidate_cli::int4::runtime::load_model_with_failure(&engine, &args, common,
        remaining()?, cancellation.clone(), execution_failure)?;
    remaining()?;
    let identity = model.model_identity();
    let content: String = identity.content_sha256().iter().map(|byte| format!("{byte:02x}")).collect();
    let recipe = format!("{}:group={}", identity.matrix_recipe(), identity.group_width());
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-sentiment-sampled-corpus-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        task: "sentiment-sampled-map", model_id: "Nanbeige4.2-3B",
        source_revision: identity.revision(), source_root_sha256: identity.source_root_sha256(),
        model_content_sha256: &content, candidate_format: INT4_SOURCE_FORMAT, quant_recipe: &recipe,
        group_width: args.group_width, integer_kernel: common.kernel.kind().id(),
        snapshot: common.snapshot_pin.map(|_| SnapshotRecord {
            operation: "loaded", format: INT4_SNAPSHOT_VERSION, file_bytes: common.snapshot_need.file_bytes }),
        compute_schedule: compute::Schedule::for_threads(common.compute.threads),
        prefill_schedule: common.prefill.map(PrefillSchedule::new).transpose()?,
        prefill_reuse: options.document.sampled.prefill_reuse_execution(),
        max_selected_rows: options.document.sampled.max_selected_rows,
        input_encoding: input_encoding::reported_policy(&config.encoding),
    };
    let writer = CandidateWriter::new(output, &provenance, limits.transport.max_output_line_bytes,
        limits.output_bytes)?;
    config.document.native.run.max_elapsed = remaining()?;
    let summary = if common.compute.threads > 1 {
        engine.batch_int4_sampled_sentiment_documents_with_threads(&model, config, common.kernel,
            common.compute.threads, reader, writer, cancellation)
    } else {
        engine.batch_int4_sampled_sentiment_documents(&model, config, common.kernel,
            reader, writer, cancellation)
    }.map_err(execution_failure)?;
    completed(summary)
}

#[derive(Serialize)]
struct SnapshotRecord { operation: &'static str, format: &'static str, file_bytes: u64 }
#[derive(Serialize)]
struct PrefillSchedule { execution: &'static str, max_batch_rows: usize, admitted_workspace_bytes: u64 }
impl PrefillSchedule {
    fn new(limits: Int4PrefillLimits) -> Result<Self, CandidateError> {
        Ok(Self { execution: INT4_LAYER_MAJOR_PREFILL_VERSION, max_batch_rows: limits.max_batch_rows,
            admitted_workspace_bytes: limits.validate().map_err(|_| CandidateError::Arguments)? })
    }
}
#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    task: &'static str, model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str,
    model_content_sha256: &'a str, candidate_format: &'static str, quant_recipe: &'a str,
    group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_schedule: Option<compute::Schedule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_schedule: Option<PrefillSchedule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_reuse: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_selected_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
}

#[cfg(test)] mod tests;
