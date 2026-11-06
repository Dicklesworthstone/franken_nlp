//! One owned redaction invocation with native chunk cohorts in both NER stages.
use super::*;
use crate::{
    tasks::redact::grouped_int4::long::{Int4LongRedactionCohortRun,
        INT4_LONG_REDACTION_COHORT_ENVELOPE_BYTES},
    tasks::source_planning::grouped_int4::long::cohort::Int4SourceMapCohortBudget,
};

impl NlpEngine {
    /// Execute each complete NER map through the explicitly selected native
    /// chunk windows. `mapping.reduction.map_batch_chunks` is the window width;
    /// it is never silently clamped or substituted with serial execution.
    /// Original and transformed maps share one full declared peak, resource
    /// domain, key scope, deadline and nonrenewable work/evidence allowances.
    #[allow(clippy::too_many_arguments)]
    pub fn redact_int4_document_cohorts(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: RedactConfig<Int4LongRedactionConfig>, pseudonyms: Option<RedactionPseudonyms>,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4LongRedactionCohortRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        redact_task::check_model(&config.ner_identity, model.model_identity()).map_err(redaction_error)?;
        let width = model.model_identity().group_width();
        document_requirements(&config, source.len(), width, planner.tokenizer_digest(), *planner.template_digest())?;
        let required = config.detector.cohort_requirements(config.native.context_tokens, width).map_err(document_error)?;
        let captured = input_bytes(&source, &config, pseudonyms.as_ref())?;
        // The native quote already includes transient decoder/cursor/result
        // storage. Keep the complete map, grounding and edit frontier separate;
        // the actual edited source can need a larger window than the original.
        let temporary_bytes = sum(&[document_temporary_bytes(&config)?, required.workspace_bytes])?;
        let output_bytes = sum(&[config.detector.max_result_bytes, INT4_LONG_REDACTION_COHORT_ENVELOPE_BYTES])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, captured)?,
            || Ok(RedactInput { source, planner, vocabulary, config, pseudonyms }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            let input = input;
            let config = &input.value.config;
            let temporary = Pending::reserve(&lease, MemoryClass::JobBuffers, temporary_bytes)?;
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.native.rope_bytes, required.native.scratch_payload_bound,
                    config.native.allocator_reserve_bytes])?)?;
            let output = output_claim(&lease, output_bytes, 0)?;
            let redactor = Int4DocumentRedactor::new(&input.value.planner, config.ner_identity.clone(),
                config.detector.clone(), model.model_identity()).map_err(document_error)?;
            let pseudonyms = pseudonym_context(input.value.pseudonyms.as_ref(), &config.request)?;
            let budget = Int4SourceMapCohortBudget {
                context_limit: config.native.context_tokens,
                native_memory: Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes,
                    max_rope_bytes: required.native.rope_bytes,
                    max_scratch_payload_bytes: required.native.scratch_payload_bound },
                max_workspace_bytes: required.workspace_bytes, max_result_bytes: required.max_result_bytes,
            };
            let result = allocate(output, || {
                let result = redactor.redact_cohorts(&input.value.source, &config.request, pseudonyms.as_ref(),
                    &model.inner.loaded.value, &input.value.vocabulary, budget, kernel, control).map_err(document_error)?;
                validate_completed(&result.output, config, &input.value.source, width)?;
                if result.cohort_chunks != config.detector.mapping.reduction.map_batch_chunks {
                    return Err(invalid_result());
                }
                result.validate_schedule().map_err(document_error)?;
                checkpoint(control)?;
                Ok(result)
            })?;
            // Every exact-sized native window is physically drained before
            // redact_cohorts returns, including cancellation/error unwinding.
            drop(pseudonyms);
            drop(redactor);
            drop(input);
            drop(scratch);
            drop(kv);
            drop(temporary);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
