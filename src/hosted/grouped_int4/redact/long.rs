//! Complete-document INT4 redaction on one physically owned invocation.
use super::*;
use crate::{
    tasks::{redact::{actions::VerificationStatus, grouped_int4::long::{
        Int4DocumentRedactor, Int4LongRedactionConfig, Int4LongRedactionError,
        Int4LongRedactionRun, Int4LongRedactionStage, INT4_LONG_REDACTION_EXECUTION}},
        source_planning::grouped_int4::long::SourceMapVerification},
    validation::grounded_fields::{GroundingBudget, SourceFieldEvidence, VerifiedSourceSpan},
};

impl NlpEngine {
    /// Scan rules over each complete document while native NER uses bounded
    /// source-aligned chunks. Verification plans the actual edited text afresh.
    /// Both stages share the model, key scope, deadline and nonrenewable limits.
    #[allow(clippy::too_many_arguments)]
    pub fn redact_int4_document(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactConfig<Int4LongRedactionConfig>, pseudonyms: Option<RedactionPseudonyms>,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4LongRedactionRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        redact_task::check_model(&config.ner_identity, model.model_identity()).map_err(redaction_error)?;
        let width = model.model_identity().group_width();
        let required = document_requirements(&config, source.len(), width,
            planner.tokenizer_digest(), *planner.template_digest())?;
        let captured = input_bytes(&source, &config, pseudonyms.as_ref())?;
        let temporary_bytes = document_temporary_bytes(&config)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, captured)?,
            || Ok(RedactInput { source, planner, vocabulary, config, pseudonyms }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Own the complete capture through every native pass and physical
            // drain, including cancellation while a wrapper is already done.
            let input = input;
            let config = &input.value.config;
            let temporary = Pending::reserve(&lease, MemoryClass::JobBuffers, temporary_bytes)?;
            let redactor = Int4DocumentRedactor::new(&input.value.planner, config.ner_identity.clone(),
                config.detector.clone(), model.model_identity()).map_err(document_error)?;
            let pseudonyms = pseudonym_context(input.value.pseudonyms.as_ref(), &config.request)?;
            let output = output_claim(&lease, config.detector.max_result_bytes, 0)?;
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, config.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                    max_scratch_payload_bytes: required.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let result = redactor.redact(&input.value.source, &config.request, pseudonyms.as_ref(),
                    &mut engine.value, &input.value.vocabulary, control).map_err(document_error)?;
                validate_completed(&result, config, &input.value.source, width)?;
                if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(invalid_result());
                }
                checkpoint(control)?;
                Ok(result)
            })?;
            drop(engine);
            drop(pseudonyms);
            drop(redactor);
            drop(temporary);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn document_error(error: Int4LongRedactionError) -> HostedError {
    match error {
        Int4LongRedactionError::Map(error) => HostedError::Int4(Int4HostError::SourceMap(error)),
        // Drop residual coordinate vectors inside the failing output allocation.
        Int4LongRedactionError::Redaction(error) => execution_error(error),
    }
}
fn invalid_result() -> HostedError { redaction_error(Int4RedactionError::InvalidResult) }

fn document_requirements(config: &RedactConfig<Int4LongRedactionConfig>, source_bytes: usize,
    width: usize, tokenizer: Sha256Digest, template: Sha256Digest)
    -> Result<Int4MemoryRequirement, HostedError> {
    config.native.run.validate()?;
    config.ner_identity.validate().map_err(|_| HostedError::ModelIdentity)?;
    constrained_int4::check_profile(&config.ner_identity, width).map_err(|_| HostedError::ModelIdentity)?;
    if config.ner_identity.task_spec != NER_TASK_VERSION || config.ner_identity.tokenizer_digest != tokenizer
        || config.ner_identity.template_digest != template { return Err(HostedError::ModelIdentity); }
    let detector = &config.detector;
    detector.ner.validate().map_err(|_| detector_error(RedactError::InvalidOptions))?;
    detector.per_chunk.validate().map_err(|_| detector_error(RedactError::InvalidOptions))?;
    config.request.rules.validate().map_err(detector_error)?;
    detector.mapping.chunks.effective_token_limit()
        .map_err(|_| HostedError::Limits("INT4 document redaction chunk capacity"))?;
    if config.native.allocator_reserve_bytes == 0 || config.preparation_reserve_bytes == 0
        || config.edit_reserve_bytes == 0 || source_bytes == 0
        || source_bytes > detector.mapping.chunks.max_input_bytes
        || source_bytes > config.request.rule_budget.max_input_bytes
        || detector.planning.max_context_tokens > config.native.context_tokens
        || detector.mapping.chunks.context_tokens > detector.planning.max_context_tokens
        || detector.mapping.chunks.max_chunk_bytes > detector.planning.max_input_bytes
        || detector.mapping.chunks.reserved_tokens < detector.per_chunk.max_output_tokens as usize
        || detector.per_chunk.max_output_bytes > detector.mapping.reduction.max_value_bytes as u64
        || config.request.edit_budget.max_output_bytes as u64 > detector.max_result_bytes
        || !(1..=64 * 1024 * 1024).contains(&detector.max_result_bytes)
        || detector.mapping.mask_limits.max_trie_node_visits == 0
        || detector.mapping.mask_limits.checkpoint_interval_nodes == 0 {
        return Err(HostedError::Limits("INT4 document redaction source, context, preparation or output"));
    }
    let work = detector.mapping.max_model_work;
    let verification = detector.mapping.verification;
    if work.forward_positions == 0 || work.projected_logits == 0 || work.attention_pairs == 0
        || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0
        || !(1..=1_000_000).contains(&verification.max_fields)
        || !(1..=1_000_000).contains(&verification.max_matches) || verification.max_scan_steps == 0 {
        return Err(HostedError::Limits("INT4 document redaction whole-operation allowance"));
    }
    let required = Int4MemoryRequirement::for_context(config.native.context_tokens, width).map_err(native_error)?;
    if required.kv_bytes > detector.per_chunk.max_kv_bytes {
        return Err(HostedError::Limits("INT4 document redaction complete resident KV capacity"));
    }
    let stages = 1 + u64::from(config.request.verify);
    if detector.mapping.mask_visits_per_chunk == 0
        || detector.mapping.mask_visits_per_chunk.checked_mul(stages)
            .is_none_or(|n| n > detector.mapping.max_mask_visits) {
        return Err(redaction_error(Int4RedactionError::WorkBudget));
    }
    Ok(required)
}

fn document_temporary_bytes(config: &RedactConfig<Int4LongRedactionConfig>) -> Result<u64, HostedError> {
    let detector = &config.detector;
    let count = detector.mapping.chunks.max_chunks;
    if !(1..=256).contains(&count) || detector.mapping.reduction.max_live_value_bytes == 0 {
        return Err(HostedError::Limits("INT4 document redaction finite map frontier"));
    }
    let multiply = |a: u64, b: u64| a.checked_mul(b)
        .ok_or(HostedError::Limits("INT4 document redaction workspace arithmetic"));
    // At peak retain one complete native map, its reduction/serialization
    // frontier and local/original coordinate evidence. The second stage starts
    // only after the first map dies; edit workspace remains live across both.
    let native = multiply(detector.per_chunk.max_output_bytes, count as u64)?;
    let staged = multiply(sum(&[native, detector.mapping.reduction.max_live_value_bytes as u64])?, 4)?;
    let tokens = multiply(multiply(u64::from(detector.per_chunk.max_output_tokens), count as u64)?, 8)?;
    let fields = multiply(detector.mapping.verification.max_fields as u64,
        std::mem::size_of::<SourceFieldEvidence>() as u64)?;
    let spans = multiply(multiply(detector.mapping.verification.max_matches as u64,
        std::mem::size_of::<VerifiedSourceSpan>() as u64)?, 2)?;
    sum(&[staged, tokens, fields, spans, config.edit_reserve_bytes])
}

fn validate_completed(run: &Int4LongRedactionRun, config: &RedactConfig<Int4LongRedactionConfig>,
    source: &str, width: usize) -> Result<(), HostedError> {
    let expected_verification = if config.request.verify { VerificationStatus::CleanDeclaredUnion }
        else { VerificationStatus::NotRequested };
    if run.schema_version != 1 || run.execution != INT4_LONG_REDACTION_EXECUTION
        || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != width
        || run.detector_scope != "whole-source-rules-independent-ner-chunks-v1"
        || run.verification.is_some() != config.request.verify
        || run.result.verification() != expected_verification
        || run.result.model_types().len() != config.detector.ner.types.len()
        || config.detector.ner.types.iter().any(|kind| !run.result.model_types().contains(kind))
        || run.result.rules != config.request.rules
        || (!config.request.actions.include_map && !run.result.edits().is_empty())
        || run.warnings != ["ner_chunk_boundaries_may_split_entities", "detector_recall_not_established",
            "clean_scan_and_pseudonyms_are_not_anonymization"] {
        return Err(invalid_result());
    }
    validate_stage(&run.original, source, config, width)?;
    let mut reserved = run.original.preflight.planned_model_work;
    let mut actual = run.original.model_work;
    let mut reserved_masks = run.original.preflight.reserved_mask_node_visits;
    let mut actual_masks = run.original.mask_node_visit_charge;
    let mut map_used = run.original.map_verification_used;
    let mut detection_used = run.original.detection_verification_used;
    if let Some(stage) = &run.verification {
        validate_stage(stage, run.result.text(), config, width)?;
        reserved = reserved.checked_add(stage.preflight.planned_model_work).map_err(|_| invalid_result())?;
        actual = actual.checked_add(stage.model_work).map_err(|_| invalid_result())?;
        reserved_masks = reserved_masks.checked_add(stage.preflight.reserved_mask_node_visits).ok_or_else(invalid_result)?;
        actual_masks = actual_masks.checked_add(stage.mask_node_visit_charge).ok_or_else(invalid_result)?;
        map_used = add_verification(map_used, stage.map_verification_used)?;
        detection_used = add_verification(detection_used, stage.detection_verification_used)?;
    }
    if run.reserved_model_work != reserved || run.model_work != actual
        || !within(actual, reserved) || !within(reserved, config.detector.mapping.max_model_work)
        || run.reserved_mask_node_visits != reserved_masks || run.mask_node_visit_charge != actual_masks
        || actual_masks > reserved_masks || reserved_masks > config.detector.mapping.max_mask_visits
        || run.map_verification_used != map_used || run.detection_verification_used != detection_used
        || !verification_fits(map_used, config.detector.mapping.verification)
        || !verification_fits(detection_used, config.request.grounding_budget)
        || run.verification_scan_steps != detection_used.scan_steps {
        return Err(invalid_result());
    }
    Ok(())
}
fn validate_stage(stage: &Int4LongRedactionStage, source: &str,
    config: &RedactConfig<Int4LongRedactionConfig>, width: usize) -> Result<(), HostedError> {
    let preflight = stage.preflight;
    if preflight.group_width != width || preflight.source_bytes != source.len()
        || preflight.source_scalars != source.chars().count()
        || preflight.chunks == 0 || preflight.chunks > config.detector.mapping.chunks.max_chunks
        || config.detector.mapping.mask_visits_per_chunk.checked_mul(preflight.chunks as u64)
            != Some(preflight.reserved_mask_node_visits)
        || !within(stage.model_work, preflight.planned_model_work)
        || stage.mask_node_visit_charge > preflight.reserved_mask_node_visits
        || !verification_fits(stage.map_verification_used, config.detector.mapping.verification)
        || !verification_fits(stage.detection_verification_used, config.request.grounding_budget) {
        return Err(invalid_result());
    }
    Ok(())
}
fn verification_fits(used: SourceMapVerification, budget: GroundingBudget) -> bool {
    used.fields <= budget.max_fields && used.matches <= budget.max_matches && used.scan_steps <= budget.max_scan_steps
}
fn add_verification(a: SourceMapVerification, b: SourceMapVerification) -> Result<SourceMapVerification, HostedError> {
    Ok(SourceMapVerification {
        fields: a.fields.checked_add(b.fields).ok_or_else(invalid_result)?,
        matches: a.matches.checked_add(b.matches).ok_or_else(invalid_result)?,
        scan_steps: a.scan_steps.checked_add(b.scan_steps).ok_or_else(invalid_result)?,
    })
}

#[cfg(test)]
mod tests;

