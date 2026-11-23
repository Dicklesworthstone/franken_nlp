//! Complete INT8 classification: preflight, owned execution, sealed publication.
use super::*;
use super::source_tasks::Session;
use crate::{
    candidate_cli::classify_map::{self as command, MapLimits},
    hosted::{HostedError, Int8ClassificationMapHostConfig, Int8ClassificationMapMemoryRequirement},
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl},
    tasks::{classify::{ClassificationPlanner, ClassificationRequest,
        quantized::long::{Int8ClassificationMapError, Int8ClassificationMapPreflight}},
        ir::PlanContext},
};

fn config(admitted: &MapLimits, elapsed: Duration) -> Int8ClassificationMapHostConfig {
    Int8ClassificationMapHostConfig {
        native: NativeLimits { context_tokens: admitted.common.context_tokens,
            allocator_reserve_bytes: 64 * MIB, run: RunLimits {
                max_elapsed: elapsed, max_checkpoints: admitted.common.max_checkpoints,
                cleanup_reserve_bytes: 65_536 } },
        planning: admitted.planning, mapping: admitted.mapping,
        preparation_reserve_bytes: admitted.limits.preparation_bytes,
    }
}

/// Reject an impossible declared process envelope before opening private input.
/// Load and execution peaks are alternatives; CLI preparation spans both.
pub(in crate::candidate_cli) fn admit(admitted: &MapLimits)
    -> Result<Int8ClassificationMapMemoryRequirement, CandidateError> {
    let config = config(admitted, Duration::from_secs(admitted.common.timeout_seconds));
    let quote = config.requirements_with_encoding(admitted.task, &admitted.encoding)
        .map_err(|_| CandidateError::Arguments)?;
    let sum = |values: &[u64]| values.iter().try_fold(0_u64,
        |total, &bytes| total.checked_add(bytes).ok_or(CandidateError::Arguments));
    // Mirrors the existing checked candidate loader: weights, tokenizer/
    // metadata, allocator headroom, and one bounded streaming payload window.
    let load = sum(&[admitted.limits.weight_bytes, 192 * MIB, 64 * 1024, 65_536])?;
    let execution = sum(&[admitted.limits.weight_bytes, 192 * MIB, quote.input_bytes,
        quote.native.kv_bytes, quote.workspace_bytes, quote.output_reservation_bytes, 65_536])?;
    let peak = sum(&[admitted.limits.preparation_bytes, load.max(execution).max(quote.input_bytes)])?;
    if peak > admitted.limits.memory_bytes { return Err(CandidateError::Arguments); }
    Ok(quote)
}

fn stopped(cause: DecodeCancellationKind) -> CandidateError {
    match cause {
        DecodeCancellationKind::Deadline | DecodeCancellationKind::Timeout => CandidateError::Timeout,
        _ => CandidateError::Execution,
    }
}
fn planning_error(error: Int8ClassificationMapError) -> CandidateError {
    error.cancellation().map(stopped).unwrap_or(CandidateError::Planning)
}
fn completion_error(error: Int8ClassificationMapError) -> CandidateError {
    error.cancellation().map(stopped).unwrap_or(CandidateError::Execution)
}
fn host_error(error: HostedError) -> CandidateError {
    match error {
        HostedError::Stopped { stop, .. } => stopped(stop.kind),
        HostedError::Join { physical_error: Some(error), .. }
            | HostedError::Scope { physical_error: Some(error), .. } => host_error(*error),
        HostedError::ClassificationMap(error) => completion_error(error),
        HostedError::Classification(error) => error.cancellation().map(stopped).unwrap_or(CandidateError::Execution),
        HostedError::Reservation(_) => CandidateError::Memory,
        HostedError::ModelIdentity | HostedError::ResourceDomain => CandidateError::Identity,
        _ => CandidateError::Execution,
    }
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), CandidateError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(stopped(cause)), None => Ok(()) }
}

/// Uses the same pinned planner and policy as hosted preparation. Only the
/// content-free sealed authority survives; all prompt/tokenizer owners drop
/// before weights load or the host reconstructs its owned request.
fn prepare(request: &ClassificationRequest, facts: &ArtifactIdentity, admitted: &MapLimits,
    control: &mut impl DecodeStepControl)
    -> Result<(ExecutionIdentity, Int8ClassificationMapPreflight), CandidateError> {
    poll(control)?;
    let mut identity = candidate_identity(facts)?;
    let registry = pinned_controls::pinned().map_err(|_| CandidateError::Identity)?;
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry|
        entry.special && entry.surface == crate::template::IM_END)
        .map(|entry| entry.id).ok_or(CandidateError::Identity)?;
    let planner = ClassificationPlanner::pinned(controls, eos)
        .and_then(|planner| planner.with_document_encoding(admitted.encoding.clone()))
        .map_err(|_| CandidateError::Planning)?;
    identity.task_spec = "classify-v1".to_owned();
    identity.template_digest = *planner.template_digest();
    identity.tokenizer_digest = planner.tokenizer_digest();
    let context = PlanContext::new(&identity, request.budget).map_err(|_| CandidateError::Planning)?;
    let plan = planner.plan_int8_document_with_control(request, &context,
        admitted.planning, admitted.mapping, control).map_err(planning_error)?;
    let expected = plan.preflight_metadata();
    drop(plan);
    poll(control)?;
    Ok((identity, expected))
}

pub(in crate::candidate_cli) fn execute(admitted: MapLimits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let quote = admit(&admitted)?;
    let session = Session::new(&admitted.common, admitted.limits)?;
    let text = session.read(&admitted.common.input, input, admitted.common.max_input_bytes)?;
    let request = command::request(&text, &admitted)?;
    drop(text);
    session.remaining()?;
    let facts = session.facts(&admitted.common)?;
    let (identity, expected) = {
        let lease = session.engine.resources().acquire_lease();
        let _preparation = lease.reserve(MemoryClass::JobBuffers, quote.input_bytes)
            .map_err(|_| CandidateError::Memory)?.commit().map_err(|_| CandidateError::Memory)?;
        prepare(&request, &facts, &admitted, &mut session.control())?
    };
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&admitted.common, admitted.limits, &facts, cancellation.clone())?;
    let config = config(&admitted, session.remaining()?);
    let result = session.engine.map_int8_classify_with_encoding(&model, request,
        identity, config, admitted.encoding.clone(), cancellation).map_err(host_error)?;
    expected.verify_completed_with_control(result.result(), &mut session.control())
        .map_err(completion_error)?;
    session.remaining()?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &result };
    let response = input_encoding::Response { response: &response,
        input_encoding: input_encoding::reported_policy(&admitted.encoding) };
    // Both result memory and CLI staging remain charged until write/flush end.
    // The original deadline is checked after serialization and before delivery.
    publish_checked(&response, admitted.envelope_bytes, output, &|| session.remaining().map(|_| ()))
}

#[cfg(test)] mod tests;
