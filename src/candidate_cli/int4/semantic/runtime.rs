//! One real native extraction and complete correlated field-review invocation.
use super::*;
use std::{sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    batch::{BatchDocument, extract::{ExtractionBatchArgs, ExtractionBatchGrounding,
        grouped_int4::Int4ExtractionBatchPlanner}},
    grammar::runtime::JsonProgram,
    hosted::{CancellationToken, HostedError, Int4HostError, Int4SemanticHostLimits,
        Int4SemanticHostRequirements, NativeLimits, RunLimits},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int4::{Int4Work, STRICT_INT4_PROFILE, weights::snapshot::INT4_SNAPSHOT_VERSION}},
    tasks::{extract::{ExtractionVocabulary, semantic::{
        grouped_int4::{Int4SemanticError, Int4SemanticExtractionRun,
            PreparedInt4SemanticExtraction, INT4_SEMANTIC_EXECUTION}}},
        judge::grouped_int4::Int4JudgePlanner},
};

struct DeadlineControl { started: Instant, deadline: Duration }
impl DecodeStepControl for DeadlineControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.started.elapsed() >= self.deadline).then_some(DecodeCancellationKind::Deadline)
    }
}
fn cancelled(cause: DecodeCancellationKind) -> CandidateError {
    match cause {
        DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution,
    }
}
fn planning_error(error: Int4SemanticError) -> CandidateError {
    error.cancellation().map(cancelled).unwrap_or(CandidateError::Planning)
}
fn execution_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => cancelled(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => execution_error(*error),
        HostedError::Int4(Int4HostError::SemanticExtraction(error)) =>
            error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Execution,
    }
}

pub(super) fn execute(args: Int4Args, command: SemanticArgs, _admitted: Int4Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let started = Instant::now();
    let common = command.validate(&args)?;
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
    // This charge owns all CLI inputs/planners and remains through write/flush.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let options = parse_options(&read_local(&command.semantic_options, OPTIONS_BYTES)?)?;
    let common = command.admit_options(&args, common, &options)?;
    remaining()?;
    let document = if args.input.as_os_str() == "-" { read_input(input, args.max_input_bytes)? }
        else { read_local(&args.input, args.max_input_bytes)? };
    remaining()?;
    let schema = read_local(command.extraction.schema_path(), crate::candidate_cli::extract::SCHEMA_BYTES)?;
    let budget = command.extraction.budget(&args, common);
    let request = source_schema(&command, &options, schema, &document, budget)?;
    let source_bytes = document.len();
    let planning = judge_limits(&options);
    let native_work = options.max_model_work;
    let semantic_limits = options.limits;
    let judge_budget = options.specification.judge_budget;
    let host = host_limits(&args, &command, common, remaining()?);
    let quote = host.requirements(args.group_width, budget, judge_budget, semantic_limits, planning, source_bytes)
        .map_err(|_| CandidateError::Memory)?;
    let common = admit_memory(&args, common, quote)?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::runtime::load_model(&engine, &args, common, remaining()?, cancellation.clone())?;
    remaining()?;
    let mut control = DeadlineControl { started, deadline };
    let compiler = Int4ExtractionBatchPlanner::for_model(model.model_identity(), budget,
        command.extraction.compiler(), command.extraction.source_limits(), None)
        .map_err(|_| CandidateError::Planning)?;
    let extraction = compiler.prepare_with_control(BatchDocument {
        id: "cli".to_owned(), text: document, task_args: Some(request),
    }, &mut control).map_err(|error| error.fault.cancellation.map(cancelled).unwrap_or(CandidateError::Planning))?;
    if extraction.planned_work().forward_positions > args.context_tokens as u64 {
        return Err(CandidateError::Planning);
    }
    extraction.verify_identity(extraction.execution_identity()).map_err(|_| CandidateError::Identity)?;
    drop(compiler);
    let judge = Int4JudgePlanner::for_model(model.model_identity(), judge_budget, planning)
        .map_err(|error| error.cancellation().map(cancelled).unwrap_or(CandidateError::Planning))?;
    let prepared = PreparedInt4SemanticExtraction::new(extraction, judge, options, &mut control)
        .map_err(planning_error)?;
    let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls.template_controls())
        .map_err(|_| CandidateError::Planning)?);
    remaining()?;
    // Actual sealed factory facts must reproduce the pre-model quote before
    // saving a snapshot. Dynamic field prompts only exist after extraction.
    let actual = host.requirements(prepared.group_width(), prepared.extraction_budget(), prepared.judge_budget(),
        prepared.semantic_limits(), prepared.judge_planning_limits(), prepared.source_bytes())
        .map_err(|_| CandidateError::Memory)?;
    if prepared.group_width() != args.group_width || prepared.extraction_budget() != budget
        || prepared.judge_budget() != judge_budget || prepared.semantic_limits() != semantic_limits
        || prepared.source_bytes() != source_bytes || prepared.max_model_work() != native_work
        || prepared.max_result_bytes() != common.result_bytes as u64
        || actual.native != quote.native || actual.native_workspace_bytes != quote.native_workspace_bytes
        || actual.preparation_bytes != quote.preparation_bytes || actual.output_bytes != quote.output_bytes {
        return Err(CandidateError::Identity);
    }
    admit_memory(&args, common, actual)?;
    let identity = prepared.execution_identity().clone();
    let extraction_reserved = prepared.extraction_reserved_work();
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    let result = engine.execute_int4_semantic_extract(&model, prepared, vocabulary,
        host_limits(&args, &command, common, remaining()?), common.kernel, cancellation)
        .map_err(execution_error)?;
    check_result(result.result(), args.group_width, extraction_reserved, native_work)?;
    remaining()?;
    let content = identity.logical_model_digest.to_hex();
    let response = Response {
        protocol: "fnlp-candidate-int4-semantic-extraction-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(), snapshot,
        output: result.result(),
    };
    // The guarded combined result stays borrowed through serialization and
    // delivery. No per-field prefix or error-shaped success is published.
    publish(&response, common.result_bytes.checked_add(4096).ok_or(CandidateError::Memory)?, output)
}

fn read_local(path: &std::path::Path, cap: usize) -> Result<String, CandidateError> {
    read_input(&mut crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?, cap)
}
fn source_schema(command: &SemanticArgs, options: &Int4SemanticOptions, schema: String,
    source: &str, budget: TaskBudget) -> Result<ExtractionBatchArgs, CandidateError> {
    let request = command.extraction.check_schema(schema, source, budget)?;
    let program = match request.grounding {
        ExtractionBatchGrounding::Structural => JsonProgram::compile(&request.schema, command.extraction.compiler()),
        ExtractionBatchGrounding::SourceMembership => JsonProgram::compile_with_source(
            &request.schema, source, command.extraction.compiler(), command.extraction.source_limits()),
    }.map_err(|_| CandidateError::Planning)?;
    options.validate_source_schema(program.declarative_schema(), source).map_err(planning_error)?;
    Ok(request)
}
fn host_limits(args: &Int4Args, command: &SemanticArgs, common: Int4Limits, remaining: Duration)
    -> Int4SemanticHostLimits {
    Int4SemanticHostLimits {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        masks: command.extraction.masks(), max_mask_node_visits: command.extraction.mask_visits(),
        preparation_reserve_bytes: common.preparation_bytes,
    }
}
fn admit_memory(args: &Int4Args, mut common: Int4Limits, quote: Int4SemanticHostRequirements)
    -> Result<Int4Limits, CandidateError> {
    if quote.native != common.native { return Err(CandidateError::Identity); }
    let load = if common.snapshot_pin.is_some() { common.snapshot_need.scratch_bytes }
        else { sum(&[common.source_preparation_bytes, common.load.scratch_payload_bytes])? };
    let save = if args.save_snapshot.is_some() { common.snapshot_need.scratch_bytes } else { 0 };
    let execute = quote.total_bytes().map_err(|_| CandidateError::Memory)?;
    common.minimum_memory_bytes = sum(&[common.preparation_bytes, common.load.weight_payload_bytes,
        ALLOCATOR_BYTES, load.max(save).max(execute), CLEANUP_BYTES])?;
    if common.minimum_memory_bytes > common.memory_bytes { return Err(CandidateError::Memory); }
    Ok(common)
}
fn fits(work: Int4Work, cap: Int4Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn check_result(run: &Int4SemanticExtractionRun, width: usize, extraction: Int4Work, cap: Int4Work)
    -> Result<(), CandidateError> {
    if run.schema_version != 1 || run.execution != INT4_SEMANTIC_EXECUTION
        || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != width
        || run.extraction_reserved_work != extraction || !fits(run.reserved_model_work, cap)
        || !fits(run.model_work, run.reserved_model_work) || !run.semantic.same_model_correlated
        || run.semantic.calibration != "experimental_uncalibrated_diagnostic_only" {
        return Err(CandidateError::Execution);
    }
    Ok(())
}

#[derive(Serialize)]
struct SnapshotRecord { operation: &'static str, format: &'static str, file_bytes: u64 }
impl SnapshotRecord {
    fn new(operation: &'static str, need: Int4SnapshotRequirements) -> Self {
        Self { operation, format: INT4_SNAPSHOT_VERSION, file_bytes: need.file_bytes }
    }
}
#[derive(Serialize)]
struct Response<'a, T: Serialize> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str,
    model_content_sha256: &'a str, candidate_format: &'a str, quant_recipe: &'a str,
    group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    output: &'a T,
}

#[cfg(test)]
mod tests;
