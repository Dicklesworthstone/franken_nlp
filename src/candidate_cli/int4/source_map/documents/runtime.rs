//! Fixed typed task, exact memory quote and one owned resident source stream.
use super::*;
use std::{io::{BufRead, BufReader}, sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    batch::source::grouped_int4::document::Int4SourceDocumentBatchPlanner,
    candidate_cli::batch::{CandidateWriter, completed},
    hosted::{CancellationToken, HostedError, Int4HostError, NativeLimits, RunLimits,
        Int4SourceDocumentCorpusRequirement, int4_source_document_corpus_requirements,
        corpus::CorpusLimits as HostCorpusLimits},
    native_engine::{decode::DecodeCancellationKind,
        strict_int4::{Int4ModelError, StrictInt4Error,
            weights::snapshot::{Int4SnapshotError, INT4_SNAPSHOT_VERSION}}},
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
        HostedError::Int4(Int4HostError::SourceMap(error)) => error.cancellation().map(stopped).unwrap_or(fallback),
        HostedError::Int4(Int4HostError::SourceSummary(error)) => error.cancellation().map(stopped).unwrap_or(fallback),
        HostedError::Int4(Int4HostError::SourceKeyphrases(error)) => error.cancellation().map(stopped).unwrap_or(fallback),
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
    let _preparation = lease.reserve(MemoryClass::JobBuffers, limits.cli_preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    let options = command.document.source.options_path().map(|path|
        read_config(path, shared::OPTIONS_BYTES, &remaining)).transpose()?;
    let question = command.document.question_path().map(|path|
        read_config(path, question::QUESTION_BYTES, &remaining)).transpose()?;
    let task = command.document.document_task(&args, limits.document, options.as_deref(), question,
        limits.run.max_keyphrase_scan_work)?;
    drop(options);
    if task.retained_bytes().map_err(|_| CandidateError::Memory)? > limits.retained_defaults_bytes {
        return Err(CandidateError::Memory);
    }
    let budget = command.document.source.budget(&args, common);
    let planning = command.document.planning(&args);
    let quote = int4_source_document_corpus_requirements(args.group_width, &task, budget, planning,
        limits.document.mapping, limits.run, host_limits(&args, limits, remaining()?),
        limits.document.reduction_reserve_bytes, limits.document.cohort_chunks.is_some(),
        limits.document.hierarchy_cohorts).map_err(|error| host_error(error, CandidateError::Planning))?;
    check_quote(limits, quote)?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = crate::candidate_cli::int4::runtime::load_model_with_failure(&engine, &args, common,
        remaining()?, cancellation.clone(), |error| host_error(error, CandidateError::Model))?;
    remaining()?;
    let (compiler, vocabulary) = source::planner()?;
    let planner = Int4SourceDocumentBatchPlanner::for_model(model.model_identity(), Arc::new(compiler),
        task, budget, planning, limits.document.mapping, limits.run)
        .map_err(|fault| fault.cancellation.map(stopped).unwrap_or(CandidateError::Planning))?;
    let planner = if limits.document.cohort_chunks.is_some() {
        planner.with_chunk_cohorts()
            .map_err(|fault| fault.cancellation.map(stopped).unwrap_or(CandidateError::Planning))?
    } else { planner };
    let planner = if let Some(schedule) = limits.document.hierarchy_cohorts {
        planner.with_hierarchy_cohorts(schedule.max_groups)
            .map_err(|fault| fault.cancellation.map(stopped).unwrap_or(CandidateError::Planning))?
    } else { planner };
    if planner.group_width() != model.model_identity().group_width()
        || planner.hierarchy_cohort_schedule() != limits.document.hierarchy_cohorts
        || planner.max_result_bytes() != limits.document.max_result_bytes as u64 {
        return Err(CandidateError::Identity);
    }
    let cohort = planner.uses_native_cohorts().then(|| planner.cohort_requirements())
        .transpose().map_err(|_| CandidateError::Memory)?;
    if cohort != quote.cohort { return Err(CandidateError::Identity); }
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // The immutable task and all declared simultaneous phase owners have
        // been admitted. Each incoming source is planned by the owned host.
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone())
            .map_err(|error| host_error(error, CandidateError::Model))?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    remaining()?;
    let identity = planner.execution_identity().clone();
    let content = identity.logical_model_digest.to_hex();
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-source-corpus-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        task: command.document.kind.name(), source_mode: "complete_document",
        document_mode: command.mode_name(limits.document),
        chunk_schedule: if limits.document.cohort_chunks.is_some() { "native_chunk_cohorts" } else { "serial" },
        cohort_chunks: limits.document.cohort_chunks,
        hierarchy_schedule: limits.document.hierarchy_cohorts.map(|_| "native_hierarchy_group_cohorts"),
        cohort_hierarchy_groups: limits.document.hierarchy_cohorts.map(|schedule| schedule.max_groups),
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
    let summary = engine.batch_int4_source_documents(&model, planner, Arc::new(vocabulary),
        host_limits(&args, limits, remaining()?), limits.document.reduction_reserve_bytes,
        common.kernel, reader, writer, cancellation)
        .map_err(|error| host_error(error, CandidateError::Batch))?;
    remaining()?;
    // A drained stream may contain rejected rows. Its host emits the sole
    // run_complete footer; any later error is diagnostic-only and exits nonzero.
    completed(summary)
}

fn check_quote(limits: CorpusLimits, quote: Int4SourceDocumentCorpusRequirement) -> Result<(), CandidateError> {
    if quote.cohort != limits.document.cohort
        || quote.native != limits.document.cohort.map_or(limits.common.native, |cohort| cohort.native)
        || quote.document_workspace_bytes != limits.document.workspace_bytes {
        return Err(CandidateError::Identity);
    }
    if quote.total_bytes > limits.execution_bytes { return Err(CandidateError::Memory); }
    Ok(())
}
fn read_config(path: &std::path::Path, cap: usize,
    remaining: &impl Fn() -> Result<Duration, CandidateError>) -> Result<String, CandidateError> {
    remaining()?;
    let mut file = crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?;
    let text = read_input(&mut file, cap)?;
    remaining()?;
    Ok(text)
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
    task: &'static str, source_mode: &'static str, document_mode: &'static str, chunk_schedule: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    cohort_chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hierarchy_schedule: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cohort_hierarchy_groups: Option<usize>,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
}

#[cfg(test)]
mod tests;
