//! One owned stream, exact fixed schema and one resident native document mapper.
use super::*;
use std::{io::{BufRead, BufReader}, sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    batch::extract::grouped_int4::long::stream::{Int4ExtractionDocumentBatchConfig,
        Int4ExtractionDocumentBatchPlanner},
    candidate_cli::{batch::{CandidateWriter, completed}, extract::SCHEMA_BYTES,
        int4::extract_map::object::RootObjectSelection},
    hosted::{CancellationToken, HostedError, Int4HostError, NativeLimits, RunLimits,
        corpus::CorpusLimits as HostCorpusLimits},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int4::{Int4ModelError, StrictInt4Error,
            weights::snapshot::{Int4SnapshotError, INT4_SNAPSHOT_VERSION}}},
    tasks::extract::ExtractionVocabulary,
};

struct DeadlineControl { started: Instant, deadline: Duration }
impl DecodeStepControl for DeadlineControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.started.elapsed() >= self.deadline).then_some(DecodeCancellationKind::Deadline)
    }
}

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
        HostedError::Int4(Int4HostError::ExtractionMap(error)) => error.cancellation().map(stopped).unwrap_or(fallback),
        HostedError::Int4(Int4HostError::ObjectExtractionMap(error)) => error.cancellation().map(stopped).unwrap_or(fallback),
        HostedError::Int4(Int4HostError::Extraction(error)) => error.cancellation().map(stopped).unwrap_or(fallback),
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
    // Fixed schema, tokenizer/vocabulary preparation and candidate frame
    // staging stay charged until the owned hosted invocation physically drains.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, limits.cli_preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let object_preparation = if limits.object_preparation_bytes == 0 { None } else {
        Some(lease.reserve(MemoryClass::JobBuffers, limits.object_preparation_bytes)
            .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?)
    };
    remaining()?;
    let schema = read_input(&mut crate::local_io::open_document(command.document.extraction.schema_path())
        .map_err(|_| CandidateError::Input)?, SCHEMA_BYTES)?;
    let budget = command.document.extraction.budget(&args, common);
    let request = command.load_defaults_with_control(schema, budget, &mut DeadlineControl { started, deadline })?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = crate::candidate_cli::int4::runtime::load_model_with_failure(&engine, &args, common,
        remaining()?, cancellation.clone(), |error| host_error(error, CandidateError::Model))?;
    remaining()?;
    let planner = Int4ExtractionDocumentBatchPlanner::for_model(model.model_identity(), budget,
        command.document.extraction.compiler(), command.document.extraction.source_limits(),
        Int4ExtractionDocumentBatchConfig { request, limits: limits.document.mapping,
            max_model_work: limits.document.mapping.mapping.max_model_work,
            max_mask_visits: limits.max_run_mask_visits, max_verification: limits.max_run_verification })
        .map_err(|fault| fault.cancellation.map(stopped).unwrap_or(CandidateError::Planning))?;
    let planner = if let Some(object) = limits.root_object {
        planner.with_root_object_reconciliation(object)
            .map_err(|fault| fault.cancellation.map(stopped).unwrap_or(CandidateError::Planning))?
    } else { planner };
    let encoding = command.document.encoding.policy()?;
    let planner = planner.with_document_encoding(encoding.clone())
        .map_err(|fault| fault.cancellation.map(stopped).unwrap_or(CandidateError::Planning))?;
    let planner = match command.document.extraction.selected_rows()? {
        Some(rows) => planner.with_selected_rows(rows)
            .map_err(|fault| fault.cancellation.map(stopped).unwrap_or(CandidateError::Planning))?,
        None => planner,
    };
    let planner = if limits.document.cohort_chunks.is_some() {
        planner.with_chunk_cohorts()
            .map_err(|fault| fault.cancellation.map(stopped).unwrap_or(CandidateError::Planning))?
    } else { planner };
    if planner.group_width() != model.model_identity().group_width()
        || planner.max_result_bytes() != limits.memory.max_output_bytes {
        return Err(CandidateError::Identity);
    }
    let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let vocabulary = ExtractionVocabulary::pinned(controls.template_controls())
        .map(Arc::new).map_err(|_| CandidateError::Planning)?;
    let mut host = host_limits(&args, limits, remaining()?);
    let quote = crate::hosted::int4_extraction_document_object_corpus_requirements_with_encoding(args.group_width, budget,
        limits.document.mapping, host, limits.document.reduction_reserve_bytes, limits.document.cohort_chunks,
        command.document.extraction.compiler(), limits.root_object, &encoding)
        .map_err(|error| host_error(error, CandidateError::Planning))?;
    if quote != limits.memory { return Err(CandidateError::Identity); }
    remaining()?;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // Fixed schema and maximum complete-document envelope are admitted;
        // actual source-bound schemas are compiled only for incoming documents.
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
        protocol: "fnlp-candidate-int4-extract-corpus-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "extract",
        source_mode: "complete_document", chunk_schedule: chunk_schedule(limits.document.cohort_chunks),
        cohort_chunks: limits.document.cohort_chunks,
        max_selected_rows: planner.selected_rows().map(|rows| rows.max_rows_per_step),
        root_array_union: limits.document.mapping.root_array_union.map(array_union::ArrayUnionSelection::new),
        root_object_reconciliation: limits.root_object.map(RootObjectSelection::new),
        input_encoding: input_encoding::reported_policy(&encoding),
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(), snapshot,
    };
    let writer = CandidateWriter::new(output, &provenance, limits.transport.max_output_line_bytes, limits.output_bytes)?;
    let reader: Box<dyn BufRead + Send> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            crate::local_io::open_document(&args.input).map_err(|_| CandidateError::Input)?))
    };
    host.native.run.max_elapsed = remaining()?;
    // The hosted quote now retains the same schema/reconciliation workspace
    // through original-document validation, serialization and writer flush.
    drop(object_preparation);
    let summary = engine.batch_int4_extract_documents(&model, planner, vocabulary, host,
        limits.document.reduction_reserve_bytes, common.kernel, reader, writer, cancellation)
        .map_err(|error| host_error(error, CandidateError::Batch))?;
    remaining()?;
    // A drained stream can contain rejected documents. The host owns its only
    // completion footer, and a later error never becomes a second stdout frame.
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
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    task: &'static str, source_mode: &'static str, chunk_schedule: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    cohort_chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_selected_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    root_array_union: Option<array_union::ArrayUnionSelection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    root_object_reconciliation: Option<RootObjectSelection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<&'a crate::tokenizer::untrusted::UntrustedEncoding>,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
}

#[cfg(test)]
mod tests;
