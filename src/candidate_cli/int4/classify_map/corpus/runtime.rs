//! One owned stream, one resident native engine and one lifetime work ledger.
use super::*;
use std::{io::{BufRead, BufReader}, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    batch::classify::grouped_int4::Int4ClassificationDocumentBatchPlanner,
    candidate_cli::batch::{CandidateWriter, completed},
    hosted::{CancellationToken, HostedError, Int4HostError, NativeLimits, RunLimits,
        corpus::CorpusLimits as HostCorpusLimits},
    native_engine::{decode::DecodeCancellationKind,
        strict_int4::{Int4ModelError, StrictInt4Error,
            weights::snapshot::{Int4SnapshotError, INT4_SNAPSHOT_VERSION}}},
    tasks::classify::grouped_int4::Int4ClassificationPlanner,
};

fn stopped(cause: DecodeCancellationKind) -> CandidateError {
    match cause {
        DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution,
    }
}
fn host_error(error: HostedError, fallback: CandidateError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => stopped(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => host_error(*error, fallback),
        HostedError::BatchSetup(fault) => fault.cancellation.map(stopped).unwrap_or(fallback),
        HostedError::Batch(error) => error.fault.cancellation.map(stopped).unwrap_or(fallback),
        HostedError::Int4(Int4HostError::ClassificationMap(error)) =>
            error.cancellation().map(stopped).unwrap_or(fallback),
        HostedError::Int4(Int4HostError::Classification(error)) =>
            error.cancellation().map(stopped).unwrap_or(fallback),
        HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))) => stopped(cause),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => fallback,
    }
}

pub(super) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    args: Int4Args, command: CorpusArgs, _admitted: CorpusLimits, input: R, output: W,
) -> Result<(), CandidateError> {
    let started = Instant::now();
    let limits = command.validate(&args)?;
    let encoding = command.document.input_encoding(&args)?;
    let common = limits.common;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: common.memory_bytes,
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    // Covers defaults parsing and transport allocation before the host takes
    // ownership, and stays live through its physical cleanup and final flush.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, limits.cli_preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let text = command.stream.defaults.as_ref().map(|path| {
        read_input(&mut crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?, shared::DEFAULTS_BYTES)
    }).transpose()?;
    let defaults = command.load_defaults(text.as_deref(), limits)?;
    drop(text);
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = crate::candidate_cli::int4::runtime::load_model_with_failure(&engine, &args, common,
        remaining()?, cancellation.clone(), |error| host_error(error, CandidateError::Model))?;
    remaining()?;
    let budget = command.document.budget(common, limits.document.planning.max_labels)?;
    let planner = Int4ClassificationPlanner::for_model(model.model_identity(), budget, limits.document.planning)
        .and_then(|planner| planner.with_document_encoding(encoding.clone()))
        .map_err(|error| error.cancellation().map(stopped).unwrap_or(CandidateError::Planning))?;
    let planner = Int4ClassificationDocumentBatchPlanner::new(planner, defaults,
        limits.document.mapping, limits.document.mapping.max_model_work)
        .map_err(|fault| fault.cancellation.map(stopped).unwrap_or(CandidateError::Planning))?;
    let cohort_chunks = command.document.cohort_chunks();
    let planner = if let Some(chunks) = cohort_chunks {
        planner.with_chunk_cohorts(chunks, Int4ClassificationMapMemoryLimits {
            context_tokens: args.context_tokens, preparation_reserve_bytes: common.preparation_bytes,
            allocator_reserve_bytes: ALLOCATOR_BYTES,
        }).map_err(|fault| fault.cancellation.map(stopped).unwrap_or(CandidateError::Planning))?
    } else { planner };
    if planner.group_width() != model.model_identity().group_width() { return Err(CandidateError::Identity); }
    let mut host = host_limits(&args, limits, remaining()?);
    // Validate the complete fixed host envelope before an optional snapshot
    // save. Future NDJSON documents are still admitted incrementally.
    let quote = memory_quote(args.group_width, budget, limits, host, cohort_chunks, &encoding)?;
    if quote.native != limits.document.memory.native || quote.workspace_bytes != limits.document.memory.workspace_bytes
        || quote.output_reservation_bytes != limits.document.memory.output_reservation_bytes
        || quote.max_output_bytes != limits.document.memory.max_output_bytes
        || quote.input_bytes != sum(&[limits.document.memory.input_bytes,
            corpus_buffer_bytes(limits.transport, common.preparation_bytes, limits.io_bytes)?,
            retained_metadata_bytes(limits.document.mapping.chunks.max_chunks)?])? {
        return Err(CandidateError::Identity);
    }
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone())
            .map_err(|error| host_error(error, CandidateError::Model))?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    remaining()?;
    let identity = planner.base_execution_identity();
    let content = identity.logical_model_digest.to_hex();
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-classify-corpus-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "classify",
        classification_mode: "complete_document", chunk_schedule: chunk_schedule(cohort_chunks), cohort_chunks,
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(), snapshot,
        input_encoding: input_encoding::reported_policy(&encoding),
    };
    let writer = CandidateWriter::new(output, &provenance, limits.transport.max_output_line_bytes, limits.output_bytes)?;
    let reader: Box<dyn BufRead + Send> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            crate::local_io::open_document(&args.input).map_err(|_| CandidateError::Input)?))
    };
    host.native.run.max_elapsed = remaining()?;
    let summary = engine.batch_int4_classification_documents(&model, planner, host, common.kernel,
        reader, writer, cancellation).map_err(|error| host_error(error, CandidateError::Batch))?;
    remaining()?;
    // The native host emitted the sole footer. Failed records still require a
    // nonzero process exit; no extra line or failed document body is invented.
    completed(summary)
}

fn host_limits(args: &Int4Args, limits: CorpusLimits, remaining: Duration) -> HostCorpusLimits {
    HostCorpusLimits {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        transport: limits.transport, preparation_reserve_bytes: limits.common.preparation_bytes,
        io_reserve_bytes: limits.io_bytes,
    }
}

fn memory_quote(width: usize, task: TaskBudget, limits: CorpusLimits, host: HostCorpusLimits,
    cohort_chunks: Option<usize>, encoding: &UntrustedEncoding)
    -> Result<Int4ClassificationMapMemoryRequirement, CandidateError> {
    let quote = if let Some(chunks) = cohort_chunks {
        crate::hosted::int4_classification_document_corpus_cohort_requirements(width, task,
            limits.document.planning, limits.document.mapping, host, chunks)
            .map(|quote| quote.memory).map_err(|error| host_error(error, CandidateError::Planning))
    } else {
        crate::hosted::int4_classification_document_corpus_requirements(width, task,
            limits.document.planning, limits.document.mapping, host)
            .map_err(|error| host_error(error, CandidateError::Planning))
    }?;
    quote.with_input_encoding(encoding).map_err(|_| CandidateError::Arguments)
}

fn chunk_schedule(cohort_chunks: Option<usize>) -> &'static str {
    if cohort_chunks.is_some() { "native_chunk_cohorts" } else { "serial" }
}

#[derive(Serialize)]
struct SnapshotRecord { operation: &'static str, format: &'static str, file_bytes: u64 }
impl SnapshotRecord {
    fn new(operation: &'static str, need: Int4SnapshotRequirements) -> Self {
        Self { operation, format: INT4_SNAPSHOT_VERSION, file_bytes: need.file_bytes }
    }
}
#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str, task: &'static str,
    classification_mode: &'static str, chunk_schedule: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    cohort_chunks: Option<usize>,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a UntrustedEncoding>,
}

#[cfg(test)]
mod tests;
