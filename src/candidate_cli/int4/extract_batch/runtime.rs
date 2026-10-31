//! One owned ordered stream, one resident INT4 model and one hosted invocation.
use super::*;
use std::{fs::File, io::{BufRead, BufReader}, sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    batch::extract::grouped_int4::Int4ExtractionBatchPlanner,
    candidate_cli::{batch::{CandidateWriter, completed}, extract::SCHEMA_BYTES},
    hosted::{CancellationToken, NativeLimits, RunLimits, corpus::CorpusLimits},
    native_engine::strict_int4::weights::snapshot::INT4_SNAPSHOT_VERSION,
    tasks::extract::ExtractionVocabulary,
};

pub(super) fn execute<R: Read + Send + 'static, W: Write + Send + 'static>(
    args: Int4Args, batch: BatchArgs, _admitted: BatchLimits, input: R, output: W,
) -> Result<(), CandidateError> {
    let started = Instant::now();
    // Recompute the finite quote at the owned boundary before opening paths
    // or allocating the runtime, even for an internal programmatic caller.
    let limits = batch.validate(&args)?;
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
    // Live before schema/tokenizer/planner/IO allocations and until the
    // blocking host has drained all output and physically released its owners.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let schema = {
        let mut file = File::open(batch.extraction.schema_path()).map_err(|_| CandidateError::Input)?;
        read_input(&mut file, SCHEMA_BYTES)?
    };
    let budget = batch.extraction.budget(&args, common);
    let defaults = batch.load_defaults(schema, budget)?;
    remaining()?;
    let reader: Box<dyn BufRead + Send> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            File::open(&args.input).map_err(|_| CandidateError::Input)?))
    };
    let cancellation = CancellationToken::default();
    let model = super::super::runtime::load_model(&engine, &args, common, remaining()?, cancellation.clone())?;
    remaining()?;
    let planner = Int4ExtractionBatchPlanner::for_model(model.model_identity(), budget,
        batch.extraction.compiler(), batch.extraction.source_limits(), Some(defaults))
        .and_then(Int4ExtractionBatchPlanner::require_shared_defaults)
        .map_err(|_| CandidateError::Planning)?;
    remaining()?;
    let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let vocabulary = ExtractionVocabulary::pinned(controls.template_controls())
        .map(Arc::new).map_err(|_| CandidateError::Planning)?;
    remaining()?;
    let identity = planner.base_execution_identity().clone();
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // The shared declaration and policy are ready. Individual source-bound
        // schemas are compiled later against each real document in the stream.
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    remaining()?;
    let content = identity.logical_model_digest.to_hex();
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-extract-batch-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(),
        model_content_sha256: &content, candidate_format: &identity.artifact_format,
        quant_recipe: &identity.quant_recipe, group_width: model.model_identity().group_width(),
        integer_kernel: common.kernel.kind().id(), task: "extract", snapshot,
        cohort_schedule: limits.cohort_documents.map(CohortSchedule::new),
    };
    // This writer copies the canonical inner event without parsing its JSON
    // numbers. It stages one frame and permanently stops after partial IO.
    let writer = CandidateWriter::new(output, &provenance, limits.transport.max_output_line_bytes,
        limits.output_bytes)?;
    let corpus = CorpusLimits {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        transport: limits.transport, preparation_reserve_bytes: limits.preparation_reserve_bytes,
        io_reserve_bytes: limits.io_bytes,
    };
    let summary = if let Some(max_documents) = limits.cohort_documents {
        engine.batch_int4_extract_cohorts(&model, planner, vocabulary, limits.native,
            corpus, max_documents, common.kernel, reader, writer, cancellation)
            .map_err(|_| CandidateError::Batch)?.summary
    } else {
        engine.batch_int4_extract(&model, planner, vocabulary, limits.native,
            corpus, common.kernel, reader, writer, cancellation).map_err(|_| CandidateError::Batch)?
    };
    // Draining the protocol can still report failed documents. Never append
    // another stdout completion/error after the host's terminal frame.
    completed(summary)
}

#[derive(Serialize)]
struct SnapshotRecord { operation: &'static str, format: &'static str, file_bytes: u64 }
impl SnapshotRecord {
    fn new(operation: &'static str, need: Int4SnapshotRequirements) -> Self {
        Self { operation, format: INT4_SNAPSHOT_VERSION, file_bytes: need.file_bytes }
    }
}

/// Only public model provenance is repeated. Per-document execution identities,
/// private source text/schema content and local paths are not outer metadata.
#[derive(Serialize)]
struct Provenance<'a> {
    protocol: &'static str,
    schema_version: u32,
    scope: &'static str,
    evidence: &'static str,
    model_id: &'static str,
    source_revision: &'a str,
    source_root_sha256: &'a str,
    model_content_sha256: &'a str,
    candidate_format: &'a str,
    quant_recipe: &'a str,
    group_width: usize,
    integer_kernel: &'static str,
    task: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cohort_schedule: Option<CohortSchedule>,
}

#[derive(Serialize)]
struct CohortSchedule { execution: &'static str, declared_max_documents: usize }
impl CohortSchedule {
    fn new(declared_max_documents: usize) -> Self {
        Self { execution: "cross_document_constrained_cohorts", declared_max_documents }
    }
}

#[cfg(test)]
mod tests;
