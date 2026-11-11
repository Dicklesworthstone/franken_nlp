//! One actual model and source compiler for ordered per-record complete-source QA.
use super::*;
use std::{io::{BufRead, BufReader}, sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, MemoryClass,
    batch::source::grouped_int4::question::Int4QuestionDocumentBatchPlanner,
    candidate_cli::batch::{CandidateWriter, completed},
    hosted::{CancellationToken, HostedError, Int4HostError, NativeLimits, RunLimits,
        corpus::CorpusLimits as HostCorpusLimits},
    native_engine::{decode::DecodeCancellationKind,
        strict_int4::{Int4ModelError, StrictInt4Error,
            weights::snapshot::{Int4SnapshotError, INT4_SNAPSHOT_VERSION}}},
    tasks::BuiltInTask,
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
        HostedError::Int4(Int4HostError::SourceTask(error)) => error.cancellation().map(stopped).unwrap_or(fallback),
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
    let declared = command.validate(&args)?;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(compute::resource_config(declared.common))
        .build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    let _preparation = lease.reserve(MemoryClass::JobBuffers, declared.cli_preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let raw = command.source.options_path().map(|path| {
        read_input(&mut crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?, shared::OPTIONS_BYTES)
    }).transpose()?;
    let options = question::options(raw.as_deref())?;
    drop(raw);
    let limits = command.validate_options(&args, options)?;
    if limits.memory != declared.memory || limits.cli_preparation_bytes != declared.cli_preparation_bytes {
        return Err(CandidateError::Memory);
    }
    let common = limits.common;
    let (planner, vocabulary) = source::planner()?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = crate::candidate_cli::int4::runtime::load_model_with_failure(&engine, &args, common,
        remaining()?, cancellation.clone(), |error| host_error(error, CandidateError::Model))?;
    remaining()?;
    let identity = source::identity(model.model_identity(), &planner, BuiltInTask::Answer)?;
    let factory = Int4QuestionDocumentBatchPlanner::for_model(model.model_identity(), Arc::new(planner),
        identity, limits.config).map_err(|fault| fault.cancellation.map(stopped).unwrap_or(CandidateError::Planning))?;
    let factory = if let Some(encoding) = limits.encoding {
        factory.with_map_document_encoding(encoding.policy(), encoding.complete)
            .map_err(|fault| fault.cancellation.map(stopped).unwrap_or(CandidateError::Planning))?
    } else { factory };
    let factory = if command.stream.cohort_chunks.is_some() {
        factory.with_chunk_cohorts()
            .map_err(|fault| fault.cancellation.map(stopped).unwrap_or(CandidateError::Planning))?
    } else { factory };
    let factory = if let Some(groups) = command.stream.cohort_hierarchy_groups {
        factory.with_hierarchy_cohorts(groups)
            .map_err(|fault| fault.cancellation.map(stopped).unwrap_or(CandidateError::Planning))?
    } else { factory };
    if factory.group_width() != model.model_identity().group_width()
        || factory.max_result_bytes() != limits.memory.max_output_bytes
        || factory.max_output_tokens() != limits.memory.max_output_tokens {
        return Err(CandidateError::Identity);
    }
    let mut host = host_limits(&args, limits, remaining()?);
    let quote = crate::hosted::int4_question_document_corpus_requirements(args.group_width, *factory.config(),
        host, limits.reduction_reserve_bytes, command.stream.cohort_chunks, command.stream.cohort_hierarchy_groups)
        .map_err(|error| host_error(error, CandidateError::Planning))?;
    if quote != limits.memory { return Err(CandidateError::Memory); }
    remaining()?;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // Fixed options/model/maximum envelopes are admitted. Real question
        // and source bindings are compiled only when each record arrives.
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone())
            .map_err(|error| host_error(error, CandidateError::Model))?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    remaining()?;
    let identity = factory.base_execution_identity();
    let content = identity.logical_model_digest.to_hex();
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-answer-corpus-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "answer",
        source_mode: "complete_document", question_source: "per_record",
        answer_mode: answer_mode(limits.config.synthesis.is_some(), limits.config.hierarchy.is_some()),
        chunk_schedule: chunk_schedule(command.stream.cohort_chunks), cohort_chunks: command.stream.cohort_chunks,
        hierarchy_schedule: limits.config.hierarchy.map(|_| hierarchy_schedule(command.stream.cohort_hierarchy_groups)),
        cohort_hierarchy_groups: command.stream.cohort_hierarchy_groups,
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(), snapshot,
        compute_schedule: compute::Schedule::for_threads(common.compute.threads),
    };
    let provenance = encoding::population_response(provenance, limits.encoding);
    let writer = CandidateWriter::new(output, &provenance, limits.transport.max_output_line_bytes, limits.output_bytes)?;
    let reader: Box<dyn BufRead + Send> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))
    } else {
        Box::new(BufReader::with_capacity(IO_BUFFER_BYTES,
            crate::local_io::open_document(&args.input).map_err(|_| CandidateError::Input)?))
    };
    host.native.run.max_elapsed = remaining()?;
    let summary = if common.compute.threads == 1 {
        engine.batch_int4_question_documents(&model, factory, Arc::new(vocabulary), host,
            limits.reduction_reserve_bytes, common.kernel, reader, writer, cancellation)
    } else {
        engine.batch_int4_question_documents_with_threads(&model, factory, Arc::new(vocabulary), host,
            limits.reduction_reserve_bytes, common.kernel, common.compute.threads, reader, writer, cancellation)
    }.map_err(|error| host_error(error, CandidateError::Batch))?;
    remaining()?;
    completed(summary)
}

fn host_limits(args: &Int4Args, limits: CorpusLimits, remaining: Duration) -> HostCorpusLimits {
    HostCorpusLimits { native: NativeLimits { context_tokens: args.context_tokens,
        allocator_reserve_bytes: ALLOCATOR_BYTES,
        run: RunLimits { max_elapsed: remaining, max_checkpoints: args.max_checkpoints,
            cleanup_reserve_bytes: CLEANUP_BYTES } },
        transport: limits.transport, preparation_reserve_bytes: limits.common.preparation_bytes,
        io_reserve_bytes: limits.io_bytes }
}
fn chunk_schedule(cohort_chunks: Option<usize>) -> &'static str {
    if cohort_chunks.is_some() { "native_chunk_cohorts" } else { "serial" }
}
fn hierarchy_schedule(groups: Option<usize>) -> &'static str {
    if groups.is_some() { "native_group_cohorts" } else { "serial" }
}
fn answer_mode(synthesis: bool, hierarchy: bool) -> &'static str {
    if hierarchy { "hierarchical_quote_selection" }
    else if synthesis { "verified_evidence_synthesis" } else { "independent_passage_answers" }
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
    task: &'static str, source_mode: &'static str, question_source: &'static str, answer_mode: &'static str,
    chunk_schedule: &'static str,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_schedule: Option<compute::Schedule>,
}

#[cfg(test)] mod tests;
