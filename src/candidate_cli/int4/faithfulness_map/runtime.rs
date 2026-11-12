//! Complete preflight, one owned document invocation and guarded publication.
use super::*;
use std::time::{Duration, Instant};
use crate::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    hosted::{CancellationToken, HostedError, Int4HostError, Int4FaithfulnessMapHostConfig,
        NativeLimits, RunLimits},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        generation::grouped_int4::INT4_SOURCE_FORMAT,
        strict_int4::{Int4MemoryBudget, Int4ModelError, Int4RunBudget, StrictInt4Error,
            weights::snapshot::{Int4SnapshotError, INT4_SNAPSHOT_VERSION}}},
    tasks::judge::grouped_int4::{Int4JudgeError, Int4JudgePlanner,
        faithfulness_map::{Int4FaithfulnessMapError, Int4FaithfulnessMapCohortBudget}}};

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
fn planning_error(error: Int4FaithfulnessMapError) -> CandidateError {
    error.cancellation().map(stopped).unwrap_or(CandidateError::Planning)
}
fn completion_error(error: Int4FaithfulnessMapError) -> CandidateError {
    error.cancellation().map(stopped).unwrap_or(CandidateError::Execution)
}
fn judge_error(error: Int4JudgeError) -> CandidateError {
    error.cancellation().map(stopped).unwrap_or(CandidateError::Planning)
}
fn host_error(error: HostedError, fallback: CandidateError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => stopped(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => host_error(*error, fallback),
        HostedError::Int4(Int4HostError::FaithfulnessMap(error)) =>
            error.cancellation().map(stopped).unwrap_or(fallback),
        HostedError::Int4(Int4HostError::Judge(error)) =>
            error.cancellation().map(stopped).unwrap_or(fallback),
        HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause)))
            | HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))) => stopped(cause),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => fallback,
    }
}

pub(super) fn execute(args: Int4Args, options: MapArgs, _admitted: MapLimits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let started = Instant::now();
    let limits = options.validate(&args)?;
    let encoding = options.input_encoding(&args)?;
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
    // Source/parser and final JSON staging remain charged from before input
    // IO through physical write/flush, including cancellation and errors.
    let _preparation = lease.reserve(MemoryClass::JobBuffers, limits.cli_preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    let text = if args.input.as_os_str() == "-" { read_input(input, args.max_input_bytes)? }
        else { read_input(&mut crate::local_io::open_document(&args.input)
            .map_err(|_| CandidateError::Input)?, args.max_input_bytes)? };
    remaining()?;
    let budget = options.budget(common)?;
    let request = parse_request(&text, budget, args.max_input_bytes)?;
    drop(text);
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::runtime::load_model_with_failure(&engine, &args, common,
        remaining()?, cancellation.clone(), |error| host_error(error, CandidateError::Model))?;
    remaining()?;
    let mut config = Int4FaithfulnessMapHostConfig {
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        planning: limits.planning, mapping: limits.mapping,
        preparation_reserve_bytes: common.preparation_bytes,
    };
    // Snapshot creation is an explicit extra effect, so all chunks and every
    // actual ragged window must fit first. Keep only the small sealed metadata
    // after this temporary plan, then compile once inside the owned host.
    let expected = if args.save_snapshot.is_some() {
        let preflight_bytes = sum(&[limits.memory.input_bytes, limits.memory.workspace_bytes])?;
        let _preflight = lease.reserve(MemoryClass::JobBuffers, preflight_bytes)
            .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
        let planner = Int4JudgePlanner::for_model(model.model_identity(), budget, config.planning)
            .map_err(judge_error)?.with_document_encoding(encoding.clone()).map_err(judge_error)?;
        let mut control = DeadlineControl { started, deadline };
        let prepared = planner.plan_faithfulness_document_with_control(&request, config.mapping,
            &mut control).map_err(planning_error)?;
        if prepared.required_context() > args.context_tokens
            || prepared.group_width() != model.model_identity().group_width()
            || prepared.task_budget() != budget || prepared.max_result_bytes() != config.mapping.max_result_bytes
            || prepared.chunk_count() == 0 || prepared.chunk_count() > config.mapping.chunks.max_chunks
            || prepared.head_count() < prepared.chunk_count() || !(1..=32).contains(&prepared.max_head_count()) {
            return Err(CandidateError::Identity);
        }
        options.scoring.admit_plan(prepared.required_context(), args.context_tokens, prepared.planned_work())?;
        let admitted = prepared.identities().map_err(planning_error)?;
        if admitted.len() != prepared.chunk_count() { return Err(CandidateError::Identity); }
        for identity in &admitted {
            if let Some(cause) = control.checkpoint(0) { return Err(stopped(cause)); }
            crate::tasks::judge::grouped_int4::check_model(identity, model.model_identity()).map_err(judge_error)?;
            if identity.task_spec != planner.execution_identity().task_spec
                || identity.tokenizer_digest != planner.execution_identity().tokenizer_digest
                || identity.template_digest != planner.execution_identity().template_digest {
                return Err(CandidateError::Identity);
            }
        }
        if let Some(chunks) = options.mapping.cohort_chunks {
            let quoted = config.cohort_requirements_with_encoding(model.model_identity().group_width(), budget, chunks, &encoding)
                .map_err(|error| host_error(error, CandidateError::Planning))?;
            if quoted.memory != limits.memory { return Err(CandidateError::Identity); }
            prepared.preflight_cohorts_with_control(&admitted, model.model_identity(), chunks,
                Int4FaithfulnessMapCohortBudget {
                    native: Int4RunBudget::exact(prepared.planned_work()),
                    memory: Int4MemoryBudget { max_kv_bytes: quoted.memory.native.kv_bytes,
                        max_rope_bytes: quoted.memory.native.rope_bytes,
                        max_scratch_payload_bytes: quoted.memory.native.scratch_payload_bound },
                    max_task_workspace_bytes: quoted.task_workspace_bytes,
                    max_output_bytes: quoted.max_cohort_output_bytes,
                }, &mut control).map_err(planning_error)?;
        } else if limits.memory.native.kv_bytes > budget.max_kv_bytes {
            return Err(CandidateError::Planning);
        }
        Some(prepared.preflight_metadata())
    } else { None };
    remaining()?;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone())
            .map_err(|error| host_error(error, CandidateError::Model))?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    config.native.run.max_elapsed = remaining()?;
    let model_identity = model.model_identity();
    let content: String = model_identity.content_sha256().iter().map(|byte| format!("{byte:02x}")).collect();
    let recipe = format!("{}:group={}", model_identity.matrix_recipe(), model_identity.group_width());
    let provenance = Provenance { protocol: "fnlp-candidate-int4-faithfulness-map-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "faithfulness-map",
        model_id: "Nanbeige4.2-3B", source_revision: model_identity.revision(),
        source_root_sha256: model_identity.source_root_sha256(), model_content_sha256: &content,
        candidate_format: INT4_SOURCE_FORMAT, quant_recipe: &recipe,
        group_width: model_identity.group_width(), integer_kernel: common.kernel.kind().id(), snapshot,
        input_encoding: match &encoding { UntrustedEncoding::ByteFallback => None,
            UntrustedEncoding::SafeBpeV1(_) => Some(encoding.clone()) } };
    if let Some(chunks) = options.mapping.cohort_chunks {
        let result = engine.map_int4_faithfulness_cohorts_with_encoding(&model, request, encoding, config, chunks,
            common.kernel, cancellation).map_err(|error| host_error(error, CandidateError::Execution))?;
        if let Some(expected) = expected {
            expected.verify_completed_cohorts_with_control(result.result(), chunks,
                &mut DeadlineControl { started, deadline }).map_err(completion_error)?;
        }
        let cohort = Cohort { schema_version: result.result().schema_version,
            execution: &result.result().execution, max_chunks: chunks, chunk_cohorts: result.result().chunk_cohorts,
            group_steps: result.result().group_steps, projection_groups: result.result().projection_groups };
        publish_complete(&Response { provenance: &provenance, output: &result.result().output, cohort: Some(cohort) },
            limits.envelope_bytes, output, &remaining)
    } else {
        let result = engine.map_int4_faithfulness_with_encoding(&model, request, encoding, config, common.kernel, cancellation)
            .map_err(|error| host_error(error, CandidateError::Execution))?;
        if let Some(expected) = expected {
            expected.verify_completed_with_control(result.result(),
                &mut DeadlineControl { started, deadline }).map_err(completion_error)?;
        }
        publish_complete(&Response { provenance: &provenance, output: result.result(), cohort: None },
            limits.envelope_bytes, output, &remaining)
    }
}

/// Serialize the whole response before any external write, and check the
/// same invocation deadline before serialization, before write and after
/// physical flush. The guarded native result remains owned by the caller.
fn publish_complete<T: Serialize>(value: &T, cap: usize, output: &mut impl Write,
    remaining: &impl Fn() -> Result<Duration, CandidateError>) -> Result<(), CandidateError> {
    remaining()?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(cap).map_err(|_| CandidateError::Memory)?;
    let mut stage = BoundedOutput { bytes, cap };
    serde_json::to_writer(&mut stage, value).map_err(|_| CandidateError::Output)?;
    stage.write_all(b"\n").map_err(|_| CandidateError::Output)?;
    remaining()?;
    output.write_all(&stage.bytes).and_then(|()| output.flush()).map_err(|_| CandidateError::Output)?;
    remaining()?;
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
struct Provenance<'a> {
    protocol: &'static str, schema_version: u32, scope: &'static str, evidence: &'static str, task: &'static str,
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot: Option<SnapshotRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_encoding: Option<UntrustedEncoding>,
}
#[derive(Serialize)]
struct Cohort<'a> {
    schema_version: u32, execution: &'a str,
    max_chunks: usize, chunk_cohorts: usize, group_steps: u64, projection_groups: u64,
}
#[derive(Serialize)]
struct Response<'a, 'b, T: Serialize> {
    #[serde(flatten)]
    provenance: &'a Provenance<'b>,
    output: &'a T,
    #[serde(skip_serializing_if = "Option::is_none")]
    cohort: Option<Cohort<'a>>,
}

#[cfg(test)] mod tests;
