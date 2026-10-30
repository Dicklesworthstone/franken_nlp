//! One complete admitted native judgment, retained through output flush.
use super::*;
use std::time::{Duration, Instant};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    hosted::{CancellationToken, HostedError, Int4HostError, Int4JudgeHostLimits, NativeLimits, RunLimits},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int4::{STRICT_INT4_PROFILE, weights::snapshot::INT4_SNAPSHOT_VERSION}},
    tasks::judge::{JudgeResult, grouped_int4::{Int4JudgeError, Int4JudgePlanner, INT4_JUDGE_EXECUTION}},
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
fn planning_error(error: Int4JudgeError) -> CandidateError {
    if let Some(cause) = error.cancellation() { return cancelled(cause); }
    match error { Int4JudgeError::Identity => CandidateError::Identity, _ => CandidateError::Planning }
}
fn execution_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => cancelled(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => execution_error(*error),
        HostedError::Int4(Int4HostError::Judge(error)) =>
            error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Execution,
    }
}
fn result_mode(result: &JudgeResult) -> &'static str {
    match result { JudgeResult::Pairwise(_) => "pairwise", JudgeResult::Rubric(_) => "rubric",
        JudgeResult::Faithfulness(_) => "faithfulness" }
}

pub(super) fn execute(args: Int4Args, command: JudgeArgs, _admitted: Int4Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let started = Instant::now();
    let limits = command.validate(&args)?;
    let deadline = Duration::from_secs(args.timeout_seconds);
    let remaining = || deadline.checked_sub(started.elapsed()).filter(|time| !time.is_zero())
        .ok_or(CandidateError::Timeout);
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: limits.memory_bytes,
        leak_response_policy: LeakResponsePolicy::RecordAndEscalate,
    }).build().map_err(|_| CandidateError::Runtime)?;
    let lease = engine.resources().acquire_lease();
    let _preparation = lease.reserve(MemoryClass::JobBuffers, limits.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let text = if args.input.as_os_str() == "-" { read_input(input, args.max_input_bytes)? }
        else { read_input(&mut crate::local_io::open_document(&args.input)
            .map_err(|_| CandidateError::Input)?, args.max_input_bytes)? };
    remaining()?;
    let budget = command.scoring.budget(limits);
    let request = parse_request(&text, budget, args.max_input_bytes)?;
    let (mode, expected_heads) = request_shape(&request)?;
    drop(text);
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::runtime::load_model(&engine, &args, limits, remaining()?, cancellation.clone())?;
    remaining()?;
    // The model-bound planner owns the pinned EOS and full-vocabulary scoring
    // policy. Neither the request nor CLI flags can replace those semantics.
    let planner = Int4JudgePlanner::for_model(model.model_identity(), budget,
        command.scoring.judge_limits(budget)).map_err(planning_error)?;
    remaining()?;
    let plan = planner.plan_with_control(&request, &mut DeadlineControl { started, deadline })
        .map_err(planning_error)?;
    command.scoring.admit_plan(plan.required_context(), args.context_tokens, plan.planned_work())?;
    if plan.group_width() != model.model_identity().group_width() || plan.head_count() != expected_heads {
        return Err(CandidateError::Identity);
    }
    let expected_work = plan.planned_work();
    let identity = plan.execution_identity().clone();
    drop(planner);
    drop(request);
    remaining()?;
    let mut snapshot = limits.snapshot_pin.map(|_| SnapshotRecord::new("loaded", limits.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // Save only after all orders, criteria or evidence heads preflight.
        let pin = engine.save_int4_snapshot(&model, path.clone(), limits.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", limits.snapshot_need));
    }
    let host = Int4JudgeHostLimits {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        preparation_reserve_bytes: limits.preparation_bytes, max_model_work: command.scoring.work_ceiling(),
    };
    let result = match limits.prefill {
        None => engine.execute_int4_judge(&model, plan, host, limits.kernel, cancellation),
        Some(prefill) => engine.execute_int4_judge_layer_major(&model, plan, host,
            limits.kernel, prefill, cancellation),
    }.map_err(execution_error)?;
    if result.result().schema_version != 1 || result.result().execution != INT4_JUDGE_EXECUTION
        || result.result().numerics_profile != STRICT_INT4_PROFILE || result_mode(&result.result().result) != mode
        || result.result().model_work != expected_work || result.result().head_count != expected_heads
        || result.result().group_width != model.model_identity().group_width() {
        return Err(CandidateError::Execution);
    }
    remaining()?;
    let content = identity.logical_model_digest.to_hex();
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-judge-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "judge", mode,
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: limits.kernel.kind().id(), snapshot,
    };
    // Both guards survive bounded staging, writing and flushing. Inference
    // failure never emits a prefix; a writer error remains an output failure.
    publish(&Response { provenance: &provenance, output: result.result() }, limits.result_bytes + 4096, output)
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
    mode: &'static str, model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str,
    model_content_sha256: &'a str, candidate_format: &'a str, quant_recipe: &'a str,
    group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
}
#[derive(Serialize)]
struct Response<'a, 'b, T: Serialize> {
    #[serde(flatten)]
    provenance: &'a Provenance<'b>,
    output: &'a T,
}

#[cfg(test)]
mod tests;
