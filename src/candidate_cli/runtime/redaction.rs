//! One charged native redactor; original and verification passes share a ceiling.
use super::*;
use std::sync::Arc;
use super::source_tasks::{Session, planner, source_identity};
use crate::{candidate_cli::redact::{self as command, RedactCommand},
    hosted::{RedactConfig, RedactionPseudonyms},
    native_engine::{constrained_int8, decode::DecodeStepControl, strict_int8::Int8Work},
    tasks::{BuiltInTask, ir::PlanContext, ner::NerOptions,
        redact::{RedactionRequest, detectors, pseudonym::Pseudonyms,
            quantized::{Int8RedactionConfig, Int8Redactor}},
        source_planning::{SourceTaskPlanner, SourceTaskRequest}},
};

#[allow(clippy::too_many_arguments)]
fn detector(source: &str, ner: NerOptions, request: &RedactionRequest,
    identity: &ExecutionIdentity, planner: &SourceTaskPlanner, command: &RedactCommand,
    limits: Limits, control: &mut impl DecodeStepControl) -> Result<Int8RedactionConfig, CandidateError> {
    let host = &command.host;
    let budget = host.task_budget(limits);
    let context = PlanContext::new(identity, budget).map_err(|_| CandidateError::Planning)?;
    let first = planner.plan_int8_with_control(&SourceTaskRequest::Ner {
        document: source.to_owned(), options: ner.clone(), budget,
    }, &context, host.planning(), control).map_err(|_| CandidateError::Planning)?;
    let work = work_ceiling(first.planned_work(), request.verify, host.context_tokens, host.max_new_tokens)?;
    // Only the transformed pass can use the larger byte envelope. Each actual
    // prompt still passes the exact native context check; no truncation/retry.
    let mut planning = host.planning();
    planning.max_input_bytes = host.max_input_bytes.max(host.max_result_bytes);
    let masks = host.max_mask_node_visits.checked_mul(1 + u64::from(request.verify)).ok_or(CandidateError::Arguments)?;
    let config = Int8RedactionConfig { ner, per_pass: budget, planning, max_model_work: work,
        mask_limits: host.masks(), mask_visits_per_pass: host.max_mask_node_visits,
        max_mask_visits: masks, max_result_bytes: host.max_result_bytes as u64 };
    Int8Redactor::new(planner, identity.clone(), config.clone()).map_err(|_| CandidateError::Planning)?;
    Ok(config)
}
fn work_ceiling(first: Int8Work, verify: bool, context: usize, output: usize) -> Result<Int8Work, CandidateError> {
    if output == 0 || output >= context || first.forward_positions >= context as u64 {
        return Err(CandidateError::Planning);
    }
    if !verify { return Ok(first); }
    // Independent reset contexts: summing work, not concatenating attention
    // triangles. The second pass's actual source does not exist yet.
    let second = constrained_int8::planned_work(context - output, output).map_err(|_| CandidateError::Planning)?;
    first.checked_add(second).map_err(|_| CandidateError::Planning)
}

pub(in crate::candidate_cli) fn execute(command: RedactCommand, common: CandidateArgs, limits: Limits,
    input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
    let session = Session::new(&common, limits)?;
    // Key material never reaches argv, String, serialization or diagnostics.
    // A key read consumes stdin only with an explicitly separate document file.
    let secret = command.key(input)?.map(|(key, namespace)| RedactionPseudonyms { key: Arc::new(key), namespace });
    session.remaining()?;
    let request = command.request();
    {
        let context = secret.as_ref().map(|s| Pseudonyms::full256(&s.key, &s.namespace,
            request.actions.expected_key_commitment.as_deref())).transpose().map_err(|_| CandidateError::Identity)?;
        request.actions.check_key(context.as_ref()).map_err(|_| CandidateError::Arguments)?;
    }
    let text = session.read(&command.input, input, common.max_input_bytes)?;
    let options = command.ner_options.as_ref().map(|p| session.read(p, input, source::OPTIONS_BYTES)).transpose()?;
    let ner = command::ner_options(options.as_deref())?;
    // Refuse bounded rule failures before metadata or weights. These private
    // detections are discarded, never printed or accepted as neural evidence.
    drop(detectors::detect(&text, &request.rules, request.rule_budget).map_err(|_| CandidateError::Planning)?);
    session.remaining()?;
    let facts = session.facts(&common)?;
    let (planner, vocabulary) = planner()?;
    let identity = source_identity(&facts, &planner, BuiltInTask::Ner)?;
    let detector = detector(&text, ner, &request, &identity, &planner, &command, limits, &mut session.control())?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(&common, limits, &facts, cancellation.clone())?;
    let result = session.engine.redact_int8(&model, text, Arc::new(planner), Arc::new(vocabulary), RedactConfig {
        ner_identity: identity, request, detector, native: session.native(&common)?,
        preparation_reserve_bytes: limits.preparation_bytes, edit_reserve_bytes: command.edit_reserve_mib * MIB,
    }, secret, cancellation).map_err(|_| CandidateError::Execution)?;
    session.remaining()?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &result };
    // Native result contains only final edited text and opted-in coordinates.
    // Keep both guards alive through the complete write AND flush.
    publish(&response, command.host.max_result_bytes + 4096, output)
}

#[cfg(test)] mod tests;
