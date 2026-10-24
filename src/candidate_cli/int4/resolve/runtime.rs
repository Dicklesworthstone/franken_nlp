//! One original snapshot, one native invocation and one guarded publication.
use super::*;
use std::{sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    corpus::{resolve::{ResolutionPlan, ResolveError, RESOLVE_VERSION},
        native_resolve::{ResolutionPlanner, grouped_int4::{Int4ResolveError, Int4ResolutionRun, INT4_RESOLUTION_EXECUTION}}},
    hosted::{CancellationToken, HostedError, Int4HostError, Int4ResolveConfig, NativeLimits, RunLimits},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        generation::grouped_int4::bind_candidate_model,
        strict_int4::{Int4ModelIdentity, Int4Work, STRICT_INT4_EXECUTION, STRICT_INT4_PROFILE,
            scoring::INT4_SCORING_EXECUTION, weights::snapshot::INT4_SNAPSHOT_VERSION}},
};
mod discovery;

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
fn graph_error(error: ResolveError) -> CandidateError {
    match error {
        ResolveError::Cancelled(cause) => cancelled(cause),
        ResolveError::InvalidInput | ResolveError::InvalidAnchor | ResolveError::InputBudget => CandidateError::Input,
        _ => CandidateError::Planning,
    }
}
fn planning_error(error: Int4ResolveError) -> CandidateError {
    if let Some(cause) = error.cancellation() { return cancelled(cause); }
    match error { Int4ResolveError::Identity => CandidateError::Identity, _ => CandidateError::Planning }
}
fn execution_error(error: HostedError) -> CandidateError {
    match error {
        // A linked/fail-fast wrapper can reflect a completed physical failure.
        // Keep that typed cause instead of replacing it with generic cancellation.
        HostedError::Stopped { stop, task_error: Some(error) }
            if matches!(stop.kind, DecodeCancellationKind::FailFast | DecodeCancellationKind::LinkedExit) =>
                execution_error(*error),
        HostedError::Stopped { stop, .. } => cancelled(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => execution_error(*error),
        HostedError::Int4(Int4HostError::Resolution(error)) => {
            if let Some(cause) = error.cancellation() { return cancelled(cause); }
            match error { Int4ResolveError::Identity => CandidateError::Identity, _ => CandidateError::Execution }
        }
        HostedError::Int4(Int4HostError::Entities(error)) => {
            if let Some(cause) = error.cancellation() { return cancelled(cause); }
            match error { crate::corpus::entities_int4::Int4EntityError::Identity => CandidateError::Identity,
                _ => CandidateError::Execution }
        }
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Execution,
    }
}
fn planner(model: &Int4ModelIdentity) -> Result<(ResolutionPlanner, ExecutionIdentity), CandidateError> {
    let registry = crate::tokenizer::pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let mut ends = controls.entries().iter().filter(|entry| entry.special && entry.surface == crate::template::IM_END);
    let eos = ends.next().ok_or(CandidateError::Identity)?.id;
    if ends.next().is_some() { return Err(CandidateError::Identity); }
    let planner = ResolutionPlanner::pinned(controls, eos).map_err(|_| CandidateError::Planning)?;
    let absent = Sha256Digest::of_bytes(b"null");
    // The binder replaces every unbound model field from authenticated loaded
    // weights. The shared compiler binds both prompt orders and policy per pair.
    let identity = bind_candidate_model(ExecutionIdentity {
        schema_version: 1, source_revision: "unbound".to_owned(), logical_model_digest: absent,
        artifact_format: "unbound".to_owned(), quant_recipe: "unbound".to_owned(), packing_set_digest: absent,
        tokenizer_digest: planner.tokenizer_digest(), template_digest: planner.template_digest(),
        task_spec: RESOLVE_VERSION.to_owned(), taskir_digest: absent, prompt_digest: absent,
        grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None,
    }, model).map_err(|_| CandidateError::Identity)?;
    Ok((planner, identity))
}

#[derive(Clone, Copy)]
struct Expected { documents: usize, mentions: usize, pairs: usize, group_width: usize, work: Int4Work }
fn check_result(result: &Int4ResolutionRun, expected: Expected) -> Result<(), CandidateError> {
    if result.schema_version != 1 || result.execution != INT4_RESOLUTION_EXECUTION
        || result.numerics_profile != STRICT_INT4_PROFILE || result.scoring_execution != INT4_SCORING_EXECUTION
        || result.group_width != expected.group_width || result.planned_model_work != expected.work
        || result.model_work != expected.work
        || result.head_count != expected.pairs.checked_mul(2).ok_or(CandidateError::Execution)?
        || result.model_evaluated != (expected.pairs != 0)
        || result.result.document_count != expected.documents || result.result.mentions.len() != expected.mentions
        || result.result.judgments.len() != expected.pairs {
        return Err(CandidateError::Execution);
    }
    Ok(())
}

pub(super) fn execute(args: Int4Args, command: ResolveArgs, _admitted: Int4Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    if command.discovery.discover_entities { return discovery::execute(args, command, input, output); }
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
    let population = command.parse(&text, args.max_input_bytes)?;
    drop(text);
    let graph_charge = lease.reserve(MemoryClass::JobBuffers, command.graph_bytes()?)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    // Reject a bad final anchor, duplicate ID or incomplete graph before source
    // conversion/snapshot reads. No lexical decision substitutes for inference.
    let graph = ResolutionPlan::prepare(&population.documents, population.options,
        command.graph_limits(&args, limits), &mut DeadlineControl { started, deadline }).map_err(graph_error)?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::runtime::load_model(&engine, &args, limits, remaining()?, cancellation.clone())?;
    remaining()?;
    let (planner, identity) = planner(model.model_identity())?;
    let scoring = command.scoring_limits(&args, limits);
    let prepared = planner.prepare_int4(&graph, &identity, scoring, model.model_identity(),
        &mut DeadlineControl { started, deadline }).map_err(planning_error)?;
    let expected = Expected { documents: population.documents.len(), mentions: graph.mentions().len(),
        pairs: prepared.pair_count(), group_width: model.model_identity().group_width(), work: prepared.planned_work() };
    if expected.pairs != 0 {
        command.scoring.admit_plan(prepared.required_context_tokens(), args.context_tokens, expected.work)?;
    }
    drop(prepared);
    drop(graph);
    drop(graph_charge);
    remaining()?;
    let mut snapshot = limits.snapshot_pin.map(|_| SnapshotRecord::new("loaded", limits.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // Save only after complete source/graph/model/work admission. A saved
        // snapshot remains if later execution or output delivery fails.
        let pin = engine.save_int4_snapshot(&model, path.clone(), limits.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", limits.snapshot_need));
    }
    let config = Int4ResolveConfig { identity: identity.clone(), options: population.options,
        graph: command.graph_limits(&args, limits), scoring,
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        preparation_reserve_bytes: limits.preparation_bytes, graph_reserve_bytes: command.graph_bytes()? };
    let result = engine.resolve_int4(&model, population.documents, Arc::new(planner), config,
        limits.kernel, cancellation).map_err(execution_error)?;
    check_result(result.result(), expected)?;
    remaining()?;
    let content = identity.logical_model_digest.to_hex();
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-resolve-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "resolve",
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: limits.kernel.kind().id(), snapshot,
    };
    // Keep both the host result guard and CLI staging owner through write AND
    // flush. The full result is staged before the first external byte.
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
    model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str, model_content_sha256: &'a str,
    candidate_format: &'a str, quant_recipe: &'a str, group_width: usize, integer_kernel: &'static str,
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
