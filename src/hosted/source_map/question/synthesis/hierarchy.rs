//! One owned INT8 invocation across discovery and all quotation-selection levels.
use super::*;
use crate::tasks::source_planning::quantized::long::question::synthesis::hierarchy::{
    Int8QuestionHierarchyRun, QuestionHierarchyLimits, memory::workspace_bytes};

impl NlpEngine {
    /// Explicit lossy reduction when complete collected evidence exceeds one
    /// answer context. Every pass receives the original question and verified
    /// original-source quotes; generated answer prose never becomes evidence.
    /// Retain all native/lifted passes and all guards until physical completion.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int8_document_answer_hierarchical(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig<SourceQuestionSynthesis>, hierarchy: QuestionHierarchyLimits,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8QuestionHierarchyRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        let required = requirements(config.native)?;
        validate_synthesis(&config, source.len(), required.kv_bytes, planner.document_encoding())?;
        hierarchy.validate().map_err(HostedError::SourceMap)?;
        check_assets(&config.identity, planner.tokenizer_digest(), *planner.template_digest())?;
        // Check complete configured arithmetic before transferring input. The
        // actual source partition determines the subsequent committed reserve.
        hierarchy_bytes(&config, config.mapping.chunks.max_chunks, hierarchy)?;
        let bytes = input_bytes(&config, source.capacity(), &planner)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            let input = input;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("question hierarchy task context"))?;
            let prepared = input.value.planner.plan_int8_question_hierarchy_with_control(&input.value.source,
                &config.task, hierarchy, config.budget, &context, config.planning, config.mapping, control)
                .map_err(HostedError::SourceMap)?;
            let expected = prepared.preflight_metadata();
            let discovery = expected.discovery();
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                hierarchy_bytes(config, discovery.chunk_count(), hierarchy)?)?;
            let tokens = transcript_tokens(config.budget.max_output_tokens, discovery.native_chunks(), hierarchy)?;
            let output = output_claim(&lease, prepared.max_result_bytes(), tokens)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(discovery.native_chunks())
                .map_err(|_| HostedError::SourceMap(Int8SourceMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                if let Some(cause) = control.prefill_checkpoint(0) {
                    return Err(HostedError::SourceMap(Int8SourceError::Cancelled(cause).into()));
                }
                check_model_identity(model.artifact_identity(), identity)?;
                constrained_int8::check_profile(identity).map_err(|_| HostedError::ModelIdentity)?;
                admitted.push(identity.clone());
            }
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(config.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let result = allocate(output, || {
                let result = prepared.execute_with_control(&admitted, &input.value.planner,
                    &mut engine.value, &input.value.vocabulary, control).map_err(HostedError::SourceMap)?;
                expected.verify_completed_with_control(&result, control).map_err(HostedError::SourceMap)?;
                Ok(result)
            })?;
            drop(engine);
            drop(admitted);
            drop(input);
            drop(reduction);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn hierarchy_bytes(config: &SourceMapConfig<SourceQuestionSynthesis>, chunks: usize,
    hierarchy: QuestionHierarchyLimits) -> Result<u64, HostedError> {
    workspace_bytes(config.budget, config.mapping, chunks, config.reduction_reserve_bytes,
        config.task.limits, config.task.question.verification, hierarchy).map_err(HostedError::SourceMap)
}
fn transcript_tokens(per_pass: u32, discovery_passes: usize, hierarchy: QuestionHierarchyLimits)
    -> Result<u64, HostedError> {
    hierarchy.validate().map_err(HostedError::SourceMap)?;
    if per_pass == 0 { return Err(HostedError::Limits("question hierarchy output token allowance")); }
    (discovery_passes as u64).checked_add(hierarchy.max_passes as u64)
        .and_then(|n| n.checked_mul(u64::from(per_pass)))
        .ok_or(HostedError::Limits("question hierarchy output token arithmetic"))
}

#[cfg(test)] mod tests;
