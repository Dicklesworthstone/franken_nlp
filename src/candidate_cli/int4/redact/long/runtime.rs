//! Exact original-map preflight and one guarded complete-document native run.
use super::*;
use std::{sync::Arc, time::{Duration, Instant}};
use crate::{
    NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy, MemoryClass,
    hosted::{CancellationToken, HostedError, Int4HostError, NativeLimits, RunLimits,
        RedactConfig, RedactionPseudonyms},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        strict_int4::{STRICT_INT4_PROFILE, weights::snapshot::INT4_SNAPSHOT_VERSION}},
    tasks::{BuiltInTask, redact::{actions::VerificationStatus, detectors, pseudonym::Pseudonyms,
        grouped_int4::{Int4RedactionError, long::{Int4DocumentRedactor, Int4LongRedactionError,
            Int4LongRedactionPreflight, Int4LongRedactionRun, Int4LongRedactionStage,
            INT4_LONG_REDACTION_EXECUTION}}},
        source_planning::grouped_int4::long::SourceMapVerification},
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
fn planning_error(error: Int4LongRedactionError) -> CandidateError {
    if let Some(cause) = error.cancellation() { return cancelled(cause); }
    match error {
        Int4LongRedactionError::Redaction(Int4RedactionError::Identity) => CandidateError::Identity,
        _ => CandidateError::Planning,
    }
}
fn execution_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => cancelled(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => execution_error(*error),
        HostedError::Int4(Int4HostError::SourceMap(error)) =>
            error.cancellation().map(cancelled).unwrap_or(CandidateError::Execution),
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
fn verified_within(used: SourceMapVerification, cap: GroundingBudget) -> bool {
    used.fields <= cap.max_fields && used.matches <= cap.max_matches && used.scan_steps <= cap.max_scan_steps
}
fn add_usage(a: SourceMapVerification, b: SourceMapVerification) -> Result<SourceMapVerification, CandidateError> {
    Ok(SourceMapVerification {
        fields: a.fields.checked_add(b.fields).ok_or(CandidateError::Execution)?,
        matches: a.matches.checked_add(b.matches).ok_or(CandidateError::Execution)?,
        scan_steps: a.scan_steps.checked_add(b.scan_steps).ok_or(CandidateError::Execution)?,
    })
}
fn check_stage(stage: &Int4LongRedactionStage, width: usize, request: &RedactionRequest,
    detector: &Int4LongRedactionConfig) -> Result<(), CandidateError> {
    let p = stage.preflight;
    if p.group_width != width || p.source_bytes == 0 || p.source_scalars == 0 || p.source_scalars > p.source_bytes
        || p.source_bytes > detector.mapping.chunks.max_input_bytes
        || p.chunks == 0 || p.chunks > detector.mapping.chunks.max_chunks
        || p.planned_model_work.forward_positions == 0 || p.planned_model_work.projected_logits == 0
        || p.planned_model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0
        || stage.model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64 != 0
        || !within(p.planned_model_work, detector.mapping.max_model_work)
        || !within(stage.model_work, p.planned_model_work)
        || detector.mapping.mask_visits_per_chunk.checked_mul(p.chunks as u64) != Some(p.reserved_mask_node_visits)
        || stage.mask_node_visit_charge > p.reserved_mask_node_visits
        || !verified_within(stage.map_verification_used, detector.mapping.verification)
        || !verified_within(stage.detection_verification_used, request.grounding_budget) {
        return Err(CandidateError::Execution);
    }
    Ok(())
}
fn check_completed(expected: Int4LongRedactionPreflight, run: &Int4LongRedactionRun,
    request: &RedactionRequest, detector: &Int4LongRedactionConfig) -> Result<(), CandidateError> {
    let verification = if request.verify { VerificationStatus::CleanDeclaredUnion } else { VerificationStatus::NotRequested };
    if run.schema_version != 1 || run.execution != INT4_LONG_REDACTION_EXECUTION
        || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != expected.group_width
        || run.detector_scope != "whole-source-rules-independent-ner-chunks-v1"
        || run.original.preflight != expected || run.verification.is_some() != request.verify
        || run.result.verification() != verification || (!request.actions.include_map && !run.result.edits().is_empty())
        || !within(run.reserved_model_work, detector.mapping.max_model_work)
        || !within(run.model_work, run.reserved_model_work)
        || run.reserved_mask_node_visits > detector.mapping.max_mask_visits
        || run.mask_node_visit_charge > run.reserved_mask_node_visits
        || !verified_within(run.map_verification_used, detector.mapping.verification)
        || !verified_within(run.detection_verification_used, request.grounding_budget)
        || run.verification_scan_steps != run.detection_verification_used.scan_steps {
        return Err(CandidateError::Execution);
    }
    if let Some(second) = &run.verification {
        if second.preflight.source_bytes != run.result.text().len()
            || second.preflight.source_scalars != run.result.text().chars().count() {
            return Err(CandidateError::Execution);
        }
    }
    let mut planned = Int4Work::default(); let mut actual = Int4Work::default();
    let mut masks = 0_u64; let mut charged = 0_u64;
    let mut map_used = SourceMapVerification { fields: 0, matches: 0, scan_steps: 0 };
    let mut detection_used = map_used;
    for stage in std::iter::once(&run.original).chain(run.verification.iter()) {
        check_stage(stage, expected.group_width, request, detector)?;
        planned = planned.checked_add(stage.preflight.planned_model_work).map_err(|_| CandidateError::Execution)?;
        actual = actual.checked_add(stage.model_work).map_err(|_| CandidateError::Execution)?;
        masks = masks.checked_add(stage.preflight.reserved_mask_node_visits).ok_or(CandidateError::Execution)?;
        charged = charged.checked_add(stage.mask_node_visit_charge).ok_or(CandidateError::Execution)?;
        map_used = add_usage(map_used, stage.map_verification_used)?;
        detection_used = add_usage(detection_used, stage.detection_verification_used)?;
    }
    if planned != run.reserved_model_work || actual != run.model_work || masks != run.reserved_mask_node_visits
        || charged != run.mask_node_visit_charge || map_used != run.map_verification_used
        || detection_used != run.detection_verification_used {
        return Err(CandidateError::Execution);
    }
    Ok(())
}

pub(super) fn execute(args: Int4Args, command: DocumentArgs, _admitted: DocumentLimits,
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
    // One inherited binary key, checked before original document IO. Reuse the
    // same full256 scope for both stages; no per-chunk key or policy override.
    let secret = command.redaction.key(input)?.map(|(key, namespace)| RedactionPseudonyms { key: Arc::new(key), namespace });
    let request = command.request(&args, limits);
    {
        let context = secret.as_ref().map(|s| Pseudonyms::full256(&s.key, &s.namespace,
            request.actions.expected_key_commitment.as_deref())).transpose().map_err(|_| CandidateError::Identity)?;
        request.actions.check_key(context.as_ref()).map_err(|_| CandidateError::Arguments)?;
    }
    remaining()?;
    let text = if args.input.as_os_str() == "-" { read_input(input, args.max_input_bytes)? }
        else { read_input(&mut crate::local_io::open_document(&args.input)
            .map_err(|_| CandidateError::Input)?, args.max_input_bytes)? };
    if text.is_empty() { return Err(CandidateError::Input); }
    remaining()?;
    let options = command.redaction.source.options_path().map(|path| {
        read_input(&mut crate::local_io::open_document(path).map_err(|_| CandidateError::Input)?,
            crate::candidate_cli::source::OPTIONS_BYTES)
    }).transpose()?;
    let ner = shared::ner_options(options.as_deref())?;
    drop(options);
    // Rules preflight over the WHOLE source, before loading weights. Actual
    // native and rule detections are still performed inside the owned host.
    drop(detectors::detect(&text, &request.rules, request.rule_budget).map_err(|_| CandidateError::Planning)?);
    remaining()?;
    let (planner, vocabulary) = source::planner()?;
    let cancellation = CancellationToken::default();
    let model = super::super::super::runtime::load_model(&engine, &args, common, remaining()?, cancellation.clone())?;
    remaining()?;
    let identity = source::identity(model.model_identity(), &planner, BuiltInTask::Ner)?;
    let detector = command.detector(&args, limits, ner);
    let expected = Int4DocumentRedactor::new(&planner, identity.clone(), detector.clone(), model.model_identity())
        .map_err(planning_error)?.preflight(&text, &mut DeadlineControl { started, deadline }).map_err(planning_error)?;
    if expected.group_width != model.model_identity().group_width() || expected.source_bytes != text.len()
        || expected.source_scalars != text.chars().count() {
        return Err(CandidateError::Planning);
    }
    remaining()?;
    let mut snapshot = common.snapshot_pin.map(|_| SnapshotRecord::new("loaded", common.snapshot_need));
    if let Some(path) = &args.save_snapshot {
        // Every original chunk fits before snapshot publication. Verification
        // is freshly partitioned later from actual transformed bytes and must
        // consume the same complete-operation ceilings without renewal.
        let pin = engine.save_int4_snapshot(&model, path.clone(), common.snapshot_budget(false)?,
            RunLimits { max_elapsed: remaining()?, max_checkpoints: args.snapshot_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES }, cancellation.clone()).map_err(|_| CandidateError::Model)?;
        if pin != model.model_identity().snapshot_pin() { return Err(CandidateError::Identity); }
        snapshot = Some(SnapshotRecord::new("saved", common.snapshot_need));
    }
    let result = engine.redact_int4_document(&model, text, Arc::new(planner), Arc::new(vocabulary), RedactConfig {
        ner_identity: identity.clone(), request: request.clone(), detector: detector.clone(),
        native: NativeLimits { context_tokens: args.context_tokens, allocator_reserve_bytes: ALLOCATOR_BYTES,
            run: RunLimits { max_elapsed: remaining()?, max_checkpoints: args.max_checkpoints,
                cleanup_reserve_bytes: CLEANUP_BYTES } },
        preparation_reserve_bytes: common.preparation_bytes, edit_reserve_bytes: limits.edit_bytes,
    }, secret, common.kernel, cancellation).map_err(execution_error)?;
    check_completed(expected, result.result(), &request, &detector)?;
    remaining()?;
    let content = identity.logical_model_digest.to_hex();
    let provenance = Provenance {
        protocol: "fnlp-candidate-int4-redact-document-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "redact",
        redaction_mode: "complete_document", model_id: "Nanbeige4.2-3B", source_revision: &identity.source_revision,
        source_root_sha256: model.model_identity().source_root_sha256(), model_content_sha256: &content,
        candidate_format: &identity.artifact_format, quant_recipe: &identity.quant_recipe,
        group_width: model.model_identity().group_width(), integer_kernel: common.kernel.kind().id(), snapshot,
    };
    // Both output ownership and CLI preparation survive staging/write/flush.
    // No chunk NER transcript, provisional edit or failed verification escapes.
    publish(&Response { provenance: &provenance, output: result.result() }, common.result_bytes + 4096, output)
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
    redaction_mode: &'static str, model_id: &'static str, source_revision: &'a str, source_root_sha256: &'a str,
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
