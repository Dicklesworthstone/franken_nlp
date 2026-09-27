//! Original-stage exact preflight, one resident model, complete redaction output.
use super::*;
use crate::tasks::{extract::ExtractionVocabulary, redact::{actions::VerificationStatus,
    long::{Int8DocumentRedactor, LongRedactionConfig, LongRedactionPreflight, LongRedactionRun, LONG_REDACTION_EXECUTION}}};
use crate::native_engine::strict_int8::STRICT_INT8_PROFILE;

pub(super) struct Input {
    pub source: String,
    pub ner: NerOptions,
    pub request: RedactionRequest,
    pub secret: Option<RedactionPseudonyms>,
    pub facts: ArtifactIdentity,
    pub planner: SourceTaskPlanner,
    pub vocabulary: ExtractionVocabulary,
    pub identity: ExecutionIdentity,
}
pub(super) fn execute(command: &RedactCommand, common: &CandidateArgs, limits: Limits,
    session: &Session, input: Input, output: &mut impl Write) -> Result<(), CandidateError> {
    let Input { source, ner, request, secret, facts, planner, vocabulary, identity } = input;
    let detector = command.long.config(&command.host, limits, ner)?;
    // Compile EVERY original NER request before weights. Only compact geometry
    // survives; the temporary prepared grammars drain before the host rebuild.
    // Verification cannot be preplanned until the transformed bytes exist.
    let expected = Int8DocumentRedactor::new(&planner, identity.clone(), detector.clone())
        .map_err(|_| CandidateError::Planning)?.preflight(&source, &mut session.control())
        .map_err(|_| CandidateError::Planning)?;
    session.remaining()?;
    let cancellation = CancellationToken::default();
    let model = session.load(common, limits, &facts, cancellation.clone())?;
    let result = session.engine.redact_int8_document(&model, source, Arc::new(planner), Arc::new(vocabulary), RedactConfig {
        ner_identity: identity, request: request.clone(), detector: detector.clone(), native: session.native(common)?,
        preparation_reserve_bytes: limits.preparation_bytes, edit_reserve_bytes: command.edit_reserve_mib * MIB,
    }, secret, cancellation).map_err(|_| CandidateError::Execution)?;
    session.remaining()?;
    check_completed(expected, result.result(), &request, &detector)?;
    let response = CandidateResponse { schema_version: 1, scope: "real-artifact-current-candidate",
        evidence: "non_authoritative", model_id: &facts.model_id, source_revision: &facts.revision,
        source_root_sha256: &facts.source_root_sha256, logical_model_sha256: &facts.logical_model_sha256,
        quant_recipe: &facts.recipe_id, output: &result };
    // Only complete edited text and opt-in coordinates escape. The existing
    // result guard and caller Session survive canonical staging, write and flush.
    publish(&response, command.host.max_result_bytes + 4096, output)
}
fn fits(work: Int8Work, ceiling: Int8Work) -> bool {
    work.forward_positions <= ceiling.forward_positions && work.projected_logits <= ceiling.projected_logits
        && work.attention_pairs <= ceiling.attention_pairs && work.projections.fits(ceiling.projections)
}
fn check_completed(expected: LongRedactionPreflight, result: &LongRedactionRun,
    request: &RedactionRequest, config: &LongRedactionConfig) -> Result<(), CandidateError> {
    if result.schema_version != 1 || result.execution != LONG_REDACTION_EXECUTION
        || result.numerics_profile != STRICT_INT8_PROFILE
        || result.detector_scope != "whole-source-rules-independent-ner-chunks-v1"
        || result.original.preflight != expected || result.verification.is_some() != request.verify
        || result.result.verification() != if request.verify { VerificationStatus::CleanDeclaredUnion } else { VerificationStatus::NotRequested }
        || !fits(result.reserved_model_work, config.mapping.max_model_work)
        || !fits(result.model_work, result.reserved_model_work)
        || result.reserved_mask_node_visits > config.mapping.max_mask_visits
        || result.mask_node_visit_charge > result.reserved_mask_node_visits
        || result.verification_scan_steps > request.grounding_budget.max_scan_steps {
        return Err(CandidateError::Execution);
    }
    if let Some(verification) = &result.verification {
        if verification.preflight.source_bytes != result.result.text().len()
            || verification.preflight.source_scalars != result.result.text().chars().count() {
            return Err(CandidateError::Execution);
        }
    }
    let mut planned = Int8Work::default(); let mut actual = Int8Work::default(); let mut masks = 0_u64; let mut charged = 0_u64;
    for stage in std::iter::once(&result.original).chain(result.verification.iter()) {
        if stage.preflight.chunks == 0 || stage.preflight.chunks > config.mapping.chunks.max_chunks
            || !fits(stage.model_work, stage.preflight.planned_model_work)
            || stage.mask_node_visit_charge > stage.preflight.reserved_mask_node_visits
            || config.mapping.mask_visits_per_chunk.checked_mul(stage.preflight.chunks as u64)
                != Some(stage.preflight.reserved_mask_node_visits) {
            return Err(CandidateError::Execution);
        }
        planned = planned.checked_add(stage.preflight.planned_model_work).map_err(|_| CandidateError::Execution)?;
        actual = actual.checked_add(stage.model_work).map_err(|_| CandidateError::Execution)?;
        masks = masks.checked_add(stage.preflight.reserved_mask_node_visits).ok_or(CandidateError::Execution)?;
        charged = charged.checked_add(stage.mask_node_visit_charge).ok_or(CandidateError::Execution)?;
    }
    if planned != result.reserved_model_work || actual != result.model_work || masks != result.reserved_mask_node_visits
        || charged != result.mask_node_visit_charge { return Err(CandidateError::Execution); }
    Ok(())
}

#[cfg(test)] mod tests;
