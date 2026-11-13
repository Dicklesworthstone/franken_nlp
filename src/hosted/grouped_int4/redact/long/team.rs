//! One fixed numerical team through original and edited-source NER maps.
use super::*;

impl NlpEngine {
    /// Keep one persistent output-row team across complete-document detection,
    /// transactional editing and the fresh verification map. The same native
    /// engine, rule policy, key scope and nonrenewable native/mask/grounding
    /// accounting serve both stages. Independent chunk cohorts remain a
    /// separate schedule. Workers join before the guarded result can return.
    #[allow(clippy::too_many_arguments)]
    pub fn redact_int4_document_with_threads(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactConfig<Int4LongRedactionConfig>, pseudonyms: Option<RedactionPseudonyms>,
        kernel: PackedInt4Kernel, compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt4TeamResult<Int4LongRedactionRun>, HostedError> {
        dispatch::preflight_int4_team(self, config.native.run, compute_threads)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        redact_task::check_model(&config.ner_identity, model.model_identity()).map_err(redaction_error)?;
        let width = model.model_identity().group_width();
        let required = document_requirements(&config, source.len(), width,
            planner.tokenizer_digest(), *planner.template_digest())?;
        team_requirements(&config, width, compute_threads)?;
        let captured = input_bytes(&source, &config, pseudonyms.as_ref())?;
        let temporary_bytes = sum(&[document_temporary_bytes(&config)?, document_encoding_workspace(&planner)?])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, captured)?,
            || Ok(RedactInput { source, planner, vocabulary, config, pseudonyms }))?;
        dispatch::run_int4_team(self, run, cancellation, model.clone(), compute_threads,
            kernel, move |weights, control, team| {
                let input = input;
                checkpoint(control)?;
                let config = &input.value.config;
                let temporary = Pending::reserve(&lease, MemoryClass::JobBuffers, temporary_bytes)?;
                let redactor = Int4DocumentRedactor::new(&input.value.planner, config.ner_identity.clone(),
                    config.detector.clone(), weights.identity()).map_err(document_error)?;
                let pseudonyms = pseudonym_context(input.value.pseudonyms.as_ref(), &config.request)?;
                let output = output_claim(&lease, config.detector.max_result_bytes, 0)?;
                let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
                let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                    sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
                let mut engine = allocate_native(kv, scratch, || {
                    StrictInt4Engine::new(weights, config.native.context_tokens, Int4MemoryBudget {
                        max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                        max_scratch_payload_bytes: required.scratch_payload_bound,
                    }).map_err(native_error)?.with_kernel(kernel)
                        .with_projection_executor(team).map_err(native_error)
                })?;
                let result = allocate(output, || {
                    // The task's single driver holds this same engine through
                    // both maps; edited text never creates a second team or
                    // gets a fresh native, mask or grounding allowance.
                    let result = redactor.redact(&input.value.source, &config.request, pseudonyms.as_ref(),
                        &mut engine.value, &input.value.vocabulary, control).map_err(document_error)?;
                    redactor.verify_completed_with_control(&input.value.source, &config.request,
                        pseudonyms.as_ref().map(Pseudonyms::identity), &result, control).map_err(document_error)?;
                    validate_completed(&result, config, &input.value.source, width)?;
                    if engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                        return Err(invalid_result());
                    }
                    checkpoint(control)?;
                    Ok(result)
                })?;
                drop(engine);
                // The engine has released its executor borrow. Only actual
                // successful physical projections appear in this receipt.
                let compute = team.stats().into();
                drop(pseudonyms); drop(redactor); drop(temporary); drop(input); drop(lease);
                Ok(HostedInt4TeamResult::new(GuardedOutput::new(result.value, result._memory), compute))
            })
    }
}

/// One reusable worker workspace regardless of the number of original or
/// verification chunks. The dispatcher owns its charge until physical join.
fn team_requirements(config: &RedactConfig<Int4LongRedactionConfig>,
    group_width: usize, compute_threads: usize) -> Result<u64, HostedError> {
    if config.detector.mapping.reduction.map_batch_chunks != 1 {
        return Err(HostedError::Limits("INT4 document redaction teams require serial chunk scheduling"));
    }
    let extra = int4_task_team_workspace_bytes(compute_threads, group_width)?;
    let native = Int4MemoryRequirement::for_context(config.native.context_tokens, group_width).map_err(native_error)?;
    sum(&[native.rope_bytes, native.scratch_payload_bound, config.native.allocator_reserve_bytes, extra])?;
    Ok(extra)
}

#[cfg(test)]
mod tests;
