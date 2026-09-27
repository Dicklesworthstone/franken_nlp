//! Whole-document redaction on the existing process-owned native invocation.
use super::*;
use crate::tasks::redact::long::{Int8DocumentRedactor, LongRedactionConfig, LongRedactionError, LongRedactionRun};

impl NlpEngine {
    /// Whole-source rules plus source-aligned native NER chunks, followed by one
    /// edit and optional fresh detection on the transformed document. Existing
    /// RedactConfig keeps its short-document default; this concrete detector
    /// specialization has independent complete-map and whole-run ceilings.
    #[allow(clippy::too_many_arguments)]
    pub fn redact_int8_document(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactConfig<LongRedactionConfig>, pseudonyms: Option<RedactionPseudonyms>, cancellation: CancellationToken)
        -> Result<HostedOutput<LongRedactionRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.ner_identity)?;
        let required = requirements(config.native)?;
        validate_document(&config, source.len(), planner.tokenizer_digest(), *planner.template_digest(), required.kv_bytes)?;
        let bytes = input_bytes(&source, &config, pseudonyms.as_ref())?;
        let temporary_bytes = document_temporary_bytes(&config)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(RedactInput { source, planner, vocabulary, config, pseudonyms }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Force capture of the WHOLE package. Source, private key/namespace
            // and configuration outlive all native and verification borrowers.
            let input = input;
            let config = &input.value.config;
            let temporary = Pending::reserve(&lease, MemoryClass::JobBuffers, temporary_bytes)?;
            let redactor = Int8DocumentRedactor::new(&input.value.planner, config.ner_identity.clone(), config.detector.clone())
                .map_err(document_error)?;
            let pseudonyms = pseudonym_context(input.value.pseudonyms.as_ref(), &config.request)?;
            let output = output_claim(&lease, config.detector.max_result_bytes, 0)?;
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(config.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let result = allocate(output, || redactor.redact(&input.value.source, &config.request,
                pseudonyms.as_ref(), &mut engine.value, &input.value.vocabulary, control).map_err(document_error))?;
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
fn document_error(error: LongRedactionError) -> HostedError {
    match error {
        LongRedactionError::Map(error) => HostedError::SourceMap(error),
        // Drop residual coordinate vectors while the result/temporary charge
        // is alive. Hosted errors retain only the existing typed count/cause.
        LongRedactionError::Redaction(error) => execution_error(error),
    }
}
fn validate_document(config: &RedactConfig<LongRedactionConfig>, source_bytes: usize,
    tokenizer: Sha256Digest, template: Sha256Digest, kv_bytes: u64) -> Result<(), HostedError> {
    config.native.run.validate()?;
    config.ner_identity.validate().map_err(|_| HostedError::ModelIdentity)?;
    constrained_int8::check_profile(&config.ner_identity).map_err(|_| HostedError::ModelIdentity)?;
    if config.ner_identity.task_spec != NER_TASK_VERSION || config.ner_identity.tokenizer_digest != tokenizer
        || config.ner_identity.template_digest != template { return Err(HostedError::ModelIdentity); }
    config.request.rules.validate().map_err(redaction_error)?;
    let detector = &config.detector;
    if config.preparation_reserve_bytes == 0 || config.edit_reserve_bytes == 0
        || source_bytes == 0 || source_bytes > detector.mapping.chunks.max_input_bytes
        || source_bytes > config.request.rule_budget.max_input_bytes
        || detector.planning.max_context_tokens > config.native.context_tokens
        || detector.mapping.chunks.context_tokens > config.native.context_tokens
        || detector.mapping.chunks.max_chunk_bytes > detector.planning.max_input_bytes
        || kv_bytes > detector.per_chunk.max_kv_bytes
        || detector.per_chunk.max_output_bytes > detector.mapping.reduction.max_value_bytes as u64
        || config.request.edit_budget.max_output_bytes as u64 > detector.max_result_bytes
        || !(1..=64 * 1024 * 1024).contains(&detector.max_result_bytes) {
        return Err(HostedError::Limits("document redaction source, context, preparation or output"));
    }
    let stages = 1 + u64::from(config.request.verify);
    if detector.mapping.mask_visits_per_chunk == 0
        || detector.mapping.mask_visits_per_chunk.checked_mul(stages).is_none_or(|n| n > detector.mapping.max_mask_visits) {
        return Err(HostedError::Redaction(Int8RedactionError::WorkBudget));
    }
    Ok(())
}
fn document_temporary_bytes(config: &RedactConfig<LongRedactionConfig>) -> Result<u64, HostedError> {
    let detector = &config.detector;
    let count = detector.mapping.chunks.max_chunks;
    if !(1..=256).contains(&count) || detector.mapping.reduction.max_live_value_bytes == 0 {
        return Err(HostedError::Limits("document redaction finite map frontier"));
    }
    // One stage at peak, never two simultaneous maps or resident engines.
    // Retain every native chunk result, coordinate lifts, reduction frontier
    // and serializer staging before the complete envelope can be validated.
    let native = detector.per_chunk.max_output_bytes.checked_mul(count as u64)
        .ok_or(HostedError::Limits("document redaction native intermediates"))?;
    let staged = sum(&[native, detector.mapping.reduction.max_live_value_bytes as u64])?
        .checked_mul(4).ok_or(HostedError::Limits("document redaction map staging"))?;
    let tokens = u64::from(detector.per_chunk.max_output_tokens).checked_mul(count as u64)
        .and_then(|n| n.checked_mul(8)).ok_or(HostedError::Limits("document redaction token storage"))?;
    sum(&[staged, tokens, config.edit_reserve_bytes])
}

#[cfg(test)] mod tests;
