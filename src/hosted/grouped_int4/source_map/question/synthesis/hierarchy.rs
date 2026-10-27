//! Physically owned question answering across bounded quotation frontiers.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::question::synthesis::hierarchy::{
    Int4QuestionHierarchyRun, QuestionHierarchyLimits,
};

impl<T: std::borrow::Borrow<SourceQuestionSynthesis>> Int4SourceMapConfig<T> {
    /// Reserve every permitted native answer transcript and complete-level
    /// planning witness. Evidence and verification limits remain shared by the
    /// entire invocation; these checked reserves are not measured RSS claims.
    pub fn question_hierarchy_workspace_bytes(&self, chunks: usize,
        hierarchy: QuestionHierarchyLimits) -> Result<u64, HostedError> {
        hierarchy.validate().map_err(map_error)?;
        let mul = |a: u64, b: u64| a.checked_mul(b)
            .ok_or(HostedError::Limits("INT4 answer hierarchy workspace arithmetic"));
        let per_pass = sum(&[mul(self.budget.max_output_bytes, 4)?,
            mul(u64::from(self.budget.max_output_tokens), 8)?])?;
        // The base quote includes discovery, one native result, source/question
        // copies, verification fanout and simultaneous evidence frontiers.
        // Witnesses bind every group before any forward in that level, while
        // only one complete compiled grammar is retained at a time.
        sum(&[self.question_synthesis_workspace_bytes(chunks)?,
            mul(per_pass, (hierarchy.max_passes - 1) as u64)?,
            mul(hierarchy.max_passes as u64, 8192 + 256)?,
            mul(hierarchy.max_levels as u64, 128)?])
    }
}

impl NlpEngine {
    /// Explicit lossy answer reduction. Every pass receives the original
    /// question and only independently verified original-source quotations.
    /// Intermediate answers and abstentions stay inspectable; generated answer
    /// prose never enters the next evidence frontier. No partial hierarchy is
    /// published on cancellation, refusal, or nonprogress.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_document_answer_hierarchy(&self, model: &ResidentInt4,
        source: String, planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceQuestionSynthesis>, hierarchy: QuestionHierarchyLimits,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4QuestionHierarchyRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(&config.identity, model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let width = model.model_identity().group_width();
        let required = hierarchy_requirements(&config, hierarchy, source.len(), width, &planner)?;
        let bytes = synthesis_input_bytes(source.capacity(), &config)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Both private strings, the input planners and their complete
            // charge survive every pass and the native engine's physical drain.
            let input = input;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("INT4 answer hierarchy task context"))?;
            let prepared = input.value.planner.plan_int4_question_hierarchy_with_control(
                &input.value.source, &config.task, hierarchy, config.budget, &context,
                config.planning, config.mapping, width, control).map_err(map_error)?;
            let expected = prepared.preflight_metadata();
            let discovery = expected.discovery();
            if prepared.group_width() != width || expected.group_width() != width
                || discovery.group_width() != width || discovery.native_chunks() == 0
                || prepared.execution_identities().len() != discovery.native_chunks()
                || required.kv_bytes > prepared.max_kv_bytes() {
                return Err(HostedError::ModelIdentity);
            }
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                config.question_hierarchy_workspace_bytes(discovery.chunk_count(), hierarchy)?)?;
            // Discovery keeps answers but no token arrays. Every additional
            // native pass retains its transcript until the caller delivers it.
            let output = output_claim(&lease, prepared.max_result_bytes(),
                hierarchy_output_tokens(config.budget.max_output_tokens, hierarchy)?)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(discovery.native_chunks())
                .map_err(|_| map_error(Int4SourceMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                checkpoint(control)?;
                constrained_int4::check_model(identity, model.model_identity())
                    .map_err(|_| HostedError::ModelIdentity)?;
                admitted.push(identity.clone());
            }
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound,
                    config.native.allocator_reserve_bytes])?)?;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, config.native.context_tokens,
                    Int4MemoryBudget { max_kv_bytes: required.kv_bytes,
                        max_rope_bytes: required.rope_bytes,
                        max_scratch_payload_bytes: required.scratch_payload_bound })
                    .map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let result = prepared.execute_with_control(&admitted, &input.value.planner,
                    &mut engine.value, &input.value.vocabulary, control).map_err(map_error)?;
                expected.verify_completed(&result).map_err(map_error)?;
                check_clean(&engine.value)?;
                checkpoint(control)?;
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

fn hierarchy_requirements(config: &Int4SourceMapConfig<SourceQuestionSynthesis>,
    hierarchy: QuestionHierarchyLimits, source_bytes: usize, width: usize,
    planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError> {
    let required = synthesis_requirements(config, source_bytes, width, planner)?;
    hierarchy.validate().map_err(map_error)?;
    config.question_hierarchy_workspace_bytes(config.mapping.chunks.max_chunks, hierarchy)?;
    Ok(required)
}

fn hierarchy_output_tokens(per_pass: u32, hierarchy: QuestionHierarchyLimits) -> Result<u64, HostedError> {
    hierarchy.validate().map_err(map_error)?;
    if per_pass == 0 { return Err(HostedError::Limits("INT4 answer hierarchy transcript count")); }
    (hierarchy.max_passes as u64).checked_mul(u64::from(per_pass))
        .ok_or(HostedError::Limits("INT4 answer hierarchy output token arithmetic"))
}

#[cfg(test)]
mod tests;
