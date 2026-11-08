//! One complete generation/review invocation with a single elapsed deadline.
use super::*;
use std::{sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    hosted::{CancellationToken, HostedError, Int4HostError, Int4SourceSemanticHostLimits,
        Int4SourceSemanticHostRequirements, NativeLimits, RunLimits},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int4::{Int4Work, STRICT_INT4_PROFILE, weights::snapshot::INT4_SNAPSHOT_VERSION}},
    tasks::{ir::PlanContext, source_planning::{grouped_int4::{semantic::{Int4SourceSemanticError, Int4SourceSemanticRun,
            PreparedInt4SourceSemantic, INT4_SOURCE_SEMANTIC_EXECUTION, SOURCE_SEMANTIC_SCOPE,
            SOURCE_SEMANTIC_CALIBRATION}}}, judge::grouped_int4::Int4JudgePlanner},
};

struct DeadlineControl { started: Instant, deadline: Duration }
impl DecodeStepControl for DeadlineControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.started.elapsed() >= self.deadline).then_some(DecodeCancellationKind::Deadline)
    }
}
fn cancelled(cause: DecodeCancellationKind) -> CandidateError {
    match cause { DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution }
}
fn planning_error(error: Int4SourceSemanticError) -> CandidateError {
    error.cancellation().map(cancelled).unwrap_or(CandidateError::Planning)
}
fn execution_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => cancelled(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => execution_error(*error),
        HostedError::Int4(Int4HostError::SourceSemantic(error)) =>
            error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Execution,
    }
}

pub(super) fn execute(kind: Kind, args: Int4Args, command: SemanticSourceArgs, _admitted: Int4Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    if !matches!(kind, Kind::Summarize | Kind::Answer) { return Err(CandidateError::Arguments); }
    let started = Instant::now(); let common = command.validate(&args)?;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero()).ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: common.memory_bytes,
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    // Prices bounded option/input parsing and pinned compiler construction;
    // the separate host quote accounts every retained dynamic review plan.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let options = parse_options(&read_local(&command.semantic_options, OPTIONS_BYTES)?)?;
    let common = command.admit_options(&args, common, &options)?;
    let source_budget = command.source.budget(&args, common);
    let planning = judge_limits(&options); let judge_budget = options.judge_budget;
    let semantic_limits = options.limits; let native_work = options.max_model_work;
    let policy_digest = options.policy_digest().map_err(planning_error)?;
    let host = host_limits(&args, &command, common, remaining()?);
    // The declared input ceiling is quoted BEFORE private source/options IO,
    // and before source conversion or snapshot access. Actual inputs can only
    // reduce this quote; they cannot extend the admitted population or budget.
    let ceiling = input_ceiling(kind, &args, &options)?;
    let upper = host.requirements(args.group_width, source_budget, judge_budget, semantic_limits, planning, ceiling)
        .map_err(|_| CandidateError::Memory)?;
    let common = admit_memory(&args, common, upper)?;
    remaining()?;
    let option_json = command.source.options_path().map(|path| read_local(path, crate::candidate_cli::source::OPTIONS_BYTES)).transpose()?;
    remaining()?;
    let text = if args.input.as_os_str() == "-" { read_input(input, args.max_input_bytes)? }
        else { read_local(&args.input, args.max_input_bytes)? };
    remaining()?;
    let request = crate::candidate_cli::source::request(kind, text, option_json.as_deref(), source_budget, args.max_input_bytes)?;
    let facts = options.input_facts(&request).map_err(planning_error)?;
    if !input_fits(facts, ceiling) { return Err(CandidateError::Identity); }
    let quote = host.requirements(args.group_width, source_budget, judge_budget, semantic_limits, planning, facts)
        .map_err(|_| CandidateError::Memory)?;
    if !quote_fits(quote, upper) { return Err(CandidateError::Identity); }
    let common = admit_memory(&args, common, quote)?;
    // Prepare source and evidence shape before the model is opened. The
    // sealed composed constructor repeats these checks against its exact plan.
    options.validate_evidence_with_control(&request, &mut DeadlineControl { started, deadline }).map_err(planning_error)?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::runtime::load_model(&engine, &args, common, remaining()?, cancellation.clone())?;
    remaining()?;
    let mut control = DeadlineControl { started, deadline };
    let (compiler, vocabulary) = source::planner()?;
    let identity = source::identity(model.model_identity(), &compiler, source::task(kind))?;
    let context = PlanContext::new(&identity, source_budget).map_err(|_| CandidateError::Planning)?;
    let prepared_source = compiler.plan_int4_with_control(&request, &context, command.source.planning(&args),
        model.model_identity().group_width(), &mut control)
        .map_err(|e| e.cancellation().map(cancelled).unwrap_or(CandidateError::Planning))?;
    drop(compiler);
    let judge = Int4JudgePlanner::for_model(model.model_identity(), judge_budget, planning)
        .map_err(|e| e.cancellation().map(cancelled).unwrap_or(CandidateError::Planning))?;
    let prepared = PreparedInt4SourceSemantic::new(prepared_source, request, judge, options, &mut control)
        .map_err(planning_error)?;
    let actual = host.requirements(prepared.group_width(), prepared.source_budget(), prepared.judge_budget(),
        prepared.semantic_limits(), prepared.judge_planning_limits(), prepared.input_facts()).map_err(|_| CandidateError::Memory)?;
    if prepared.group_width() != args.group_width || prepared.source_budget() != source_budget
        || prepared.judge_budget() != judge_budget || prepared.semantic_limits() != semantic_limits
        || prepared.input_facts() != facts || prepared.policy_digest() != policy_digest
        || prepared.max_model_work() != native_work || prepared.max_result_bytes() != common.result_bytes as u64
        || actual != quote {
        return Err(CandidateError::Identity);
    }
    admit_memory(&args, common, actual)?;
    let identity = prepared.execution_identity().clone(); let source_reserved = prepared.source_reserved_work();
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    remaining()?;
    let result = engine.execute_int4_semantic_source(&model, prepared, Arc::new(vocabulary),
        host_limits(&args, &command, common, remaining()?), common.kernel, cancellation).map_err(execution_error)?;
    check_result(result.result(), args.group_width, source_reserved, native_work, policy_digest)?;
    remaining()?;
    let content = identity.logical_model_digest.to_hex();
    let response = Response { protocol: "fnlp-candidate-int4-semantic-source-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: kind.name(),
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(), snapshot,
        output: result.result() };
    // Guarded original+review output remains live through serialization and
    // delivery; a deadline crossed by serialization publishes no bytes.
    publish_checked(&response, common.result_bytes.checked_add(4096).ok_or(CandidateError::Memory)?,
        output, &|| remaining().map(|_| ()))
}
fn read_local(path: &std::path::Path, cap: usize) -> Result<String, CandidateError> {
    read_input(&mut crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?, cap)
}
fn host_limits(args: &Int4Args, command: &SemanticSourceArgs, common: Int4Limits, remaining: Duration)
    -> Int4SourceSemanticHostLimits {
    Int4SourceSemanticHostLimits { native: NativeLimits { context_tokens: args.context_tokens,
        allocator_reserve_bytes: ALLOCATOR_BYTES, run: RunLimits { max_elapsed: remaining,
            max_checkpoints: args.max_checkpoints, cleanup_reserve_bytes: CLEANUP_BYTES } },
        masks: command.source.masks(), max_mask_node_visits: command.source.mask_visits(),
        preparation_reserve_bytes: common.preparation_bytes }
}
fn admit_memory(args: &Int4Args, mut common: Int4Limits, quote: Int4SourceSemanticHostRequirements)
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
fn input_fits(actual: SourceSemanticInputFacts, ceiling: SourceSemanticInputFacts) -> bool {
    actual.source_bytes <= ceiling.source_bytes && actual.question_bytes <= ceiling.question_bytes
        && actual.passage_count <= ceiling.passage_count
}
fn quote_fits(actual: Int4SourceSemanticHostRequirements, ceiling: Int4SourceSemanticHostRequirements) -> bool {
    actual.native == ceiling.native && actual.native_workspace_bytes <= ceiling.native_workspace_bytes
        && actual.preparation_bytes <= ceiling.preparation_bytes && actual.output_bytes <= ceiling.output_bytes
}
fn fits(work: Int4Work, cap: Int4Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn check_result(run: &Int4SourceSemanticRun, width: usize, source: Int4Work, cap: Int4Work,
    policy: crate::execution_identity::Sha256Digest) -> Result<(), CandidateError> {
    if run.schema_version != 1 || run.execution != INT4_SOURCE_SEMANTIC_EXECUTION
        || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != width
        || run.source_reserved_work != source || !fits(run.reserved_model_work, cap)
        || !fits(run.model_work, run.reserved_model_work) || !run.semantic.same_model_correlated
        || run.semantic.policy_digest != policy || run.semantic.evidence_scope != SOURCE_SEMANTIC_SCOPE
        || run.semantic.calibration != SOURCE_SEMANTIC_CALIBRATION {
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
    task: &'static str, model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str,
    model_content_sha256: &'a str, candidate_format: &'a str, quant_recipe: &'a str,
    group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>, output: &'a T,
}

#[cfg(test)] mod tests;
