//! One admitted native detector union and optional fresh transformed-text pass.
use super::*;
use std::{sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    execution_identity::ExecutionIdentity,
    hosted::{CancellationToken, HostedError, Int4HostError, NativeLimits, RunLimits,
        RedactConfig, RedactionPseudonyms},
    native_engine::{constrained_int4, decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int4::{Int4ModelIdentity, Int4Work, STRICT_INT4_PROFILE,
            weights::snapshot::INT4_SNAPSHOT_VERSION}},
    tasks::{BuiltInTask, ir::PlanContext, ner::NerOptions,
        redact::{actions::VerificationStatus, detectors, pseudonym::Pseudonyms,
            grouped_int4::{Int4RedactionConfig, Int4Redactor, Int4RedactionError, INT4_REDACTION_EXECUTION}},
        source_planning::{SourceTaskPlanner, SourceTaskRequest, grouped_int4::Int4SourceError}},
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
fn source_error(error: Int4SourceError) -> CandidateError {
    error.cancellation().map(cancelled).unwrap_or(CandidateError::Planning)
}
fn planning_error(error: Int4RedactionError) -> CandidateError {
    if let Some(cause) = error.cancellation() { return cancelled(cause); }
    match error { Int4RedactionError::Identity => CandidateError::Identity, _ => CandidateError::Planning }
}
fn execution_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => cancelled(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => execution_error(*error),
        HostedError::Int4(Int4HostError::Redaction(error)) =>
            error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Execution,
    }
}
fn within(work: Int4Work, cap: Int4Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn work_ceiling(first: Int4Work, verify: bool, width: usize, context: usize, output: usize)
    -> Result<Int4Work, CandidateError> {
    if output == 0 || output >= context || first.forward_positions == 0 || first.forward_positions >= context as u64 {
        return Err(CandidateError::Planning);
    }
    if !verify { return Ok(first); }
    // The second source does not exist yet. Bound its complete independent
    // context, summing attention triangles rather than concatenating contexts.
    let second = constrained_int4::planned_work(width, context - output, output)
        .map_err(|_| CandidateError::Planning)?;
    first.checked_add(second).map_err(|_| CandidateError::Planning)
}
#[allow(clippy::too_many_arguments)]
fn detector(source: &str, ner: NerOptions, request: &RedactionRequest, identity: &ExecutionIdentity,
    planner: &SourceTaskPlanner, command: &RedactArgs, args: &Int4Args, limits: RedactLimits,
    model: &Int4ModelIdentity, control: &mut impl DecodeStepControl)
    -> Result<Int4RedactionConfig, CandidateError> {
    let budget = command.source.budget(args, limits.common);
    let context = PlanContext::new(identity, budget).map_err(|_| CandidateError::Planning)?;
    let first = planner.plan_int4_with_control(&SourceTaskRequest::Ner {
        document: source.to_owned(), options: ner.clone(), budget,
    }, &context, command.source.planning(args), model.group_width(), control).map_err(source_error)?;
    let work = work_ceiling(first.planned_work(), request.verify, model.group_width(),
        args.context_tokens, args.max_new_tokens)?;
    let mut planning = command.source.planning(args);
    planning.max_input_bytes = args.max_input_bytes.max(limits.common.result_bytes);
    let masks = command.source.mask_visits().checked_mul(1 + u64::from(request.verify))
        .ok_or(CandidateError::Arguments)?;
    let config = Int4RedactionConfig { ner, per_pass: budget, planning, max_model_work: work,
        mask_limits: command.source.masks(), mask_visits_per_pass: command.source.mask_visits(),
        max_mask_visits: masks, max_result_bytes: limits.common.result_bytes as u64 };
    Int4Redactor::new(planner, identity.clone(), config.clone(), model).map_err(planning_error)?;
    Ok(config)
}

pub(super) fn execute(args: Int4Args, command: RedactArgs, _admitted: RedactLimits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
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
    let _preparation = lease.reserve(MemoryClass::JobBuffers, common.preparation_bytes)
        .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
    remaining()?;
    // Binary key ownership and commitment validation precede document IO.
    // No raw secret becomes a String, argv option, diagnostic or provenance.
    let secret = command.key(input)?.map(|(key, namespace)| RedactionPseudonyms { key: Arc::new(key), namespace });
    let request = command.request(&args, limits);
    {
        let pseudonyms = secret.as_ref().map(|s| Pseudonyms::full256(&s.key, &s.namespace,
            request.actions.expected_key_commitment.as_deref())).transpose().map_err(|_| CandidateError::Identity)?;
        request.actions.check_key(pseudonyms.as_ref()).map_err(|_| CandidateError::Arguments)?;
    }
    remaining()?;
    let text = if args.input.as_os_str() == "-" { read_input(input, args.max_input_bytes)? }
        else { read_input(&mut crate::local_io::open_document(&args.input)
            .map_err(|_| CandidateError::Input)?, args.max_input_bytes)? };
    remaining()?;
    let options = command.source.options_path().map(|path| {
        read_input(&mut crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?,
            crate::candidate_cli::source::OPTIONS_BYTES)
    }).transpose()?;
    let ner = shared::ner_options(options.as_deref())?;
    drop(options);
    // Refuse bounded rule failures before loading weights. These detections
    // are discarded and never stand in for the subsequent native NER pass.
    drop(detectors::detect(&text, &request.rules, request.rule_budget).map_err(|_| CandidateError::Planning)?);
    remaining()?;
    let (planner, vocabulary) = source::planner()?;
    remaining()?;
    let cancellation = CancellationToken::default();
    let model = super::super::runtime::load_model(&engine, &args, common, remaining()?, cancellation.clone())?;
    remaining()?;
    let identity = source::identity(model.model_identity(), &planner, BuiltInTask::Ner)?;
    let detector = detector(&text, ner, &request, &identity, &planner, &command, &args, limits,
        model.model_identity(), &mut DeadlineControl { started, deadline })?;
    let expected_passes = 1 + usize::from(request.verify);
    let work_cap = detector.max_model_work;
    let mask_cap = detector.max_mask_visits;
    let expected_verification = if request.verify { VerificationStatus::CleanDeclaredUnion }
        else { VerificationStatus::NotRequested };
    let include_map = request.actions.include_map;
    remaining()?;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // Original planning and conservative complete-operation admission
        // precede snapshot publication; transformed planning stays fresh.
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    let result = engine.redact_int4(&model, text, Arc::new(planner), Arc::new(vocabulary), RedactConfig {
        ner_identity: identity.clone(), request, detector,
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        preparation_reserve_bytes: common.preparation_bytes, edit_reserve_bytes: limits.edit_bytes,
    }, secret, common.kernel, cancellation).map_err(execution_error)?;
    let run = result.result();
    if run.schema_version != 1 || run.execution != INT4_REDACTION_EXECUTION || run.numerics_profile != STRICT_INT4_PROFILE
        || run.group_width != model.model_identity().group_width() || run.ner_passes != expected_passes
        || !within(run.reserved_model_work, work_cap) || !within(run.model_work, run.reserved_model_work)
        || run.reserved_mask_node_visits != mask_cap || run.mask_node_visit_charge > run.reserved_mask_node_visits
        || run.result.verification() != expected_verification || (!include_map && !run.result.edits().is_empty()) {
        return Err(CandidateError::Execution);
    }
    remaining()?;
    let content = identity.logical_model_digest.to_hex();
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-redact-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "redact",
        model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(), snapshot,
    };
    // Both guards survive staging, writing AND flushing. A rejected native,
    // editing or verification result never reaches the writer.
    publish(&Response { provenance: &provenance, output: run }, common.result_bytes + 4096, output)
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
