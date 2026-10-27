//! Physically owned discovery and a final evidence-only INT4 answer.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::question::synthesis::{
    Int4QuestionSynthesisRun, SourceQuestionSynthesis,
};
mod hierarchy;

impl Int4SourceMapConfig<SourceQuestionSynthesis> {
    /// Charge all source/question maps, verbatim evidence and its origin fanout,
    /// independent occurrence scans and the temporary final native transcript.
    /// The final result keeps text/citations, so native token storage is temporary.
    pub fn question_synthesis_workspace_bytes(&self, chunks: usize) -> Result<u64, HostedError> {
        let mul = |a: u64, b: u64| a.checked_mul(b)
            .ok_or(HostedError::Limits("INT4 answer synthesis workspace arithmetic"));
        let verification = self.task.question.verification;
        let limits = self.task.limits;
        let fields = mul(verification.max_fields as u64,
            (std::mem::size_of::<PassageCitation>() + std::mem::size_of::<SourceCitation>()) as u64)?;
        let spans = mul(verification.max_matches as u64,
            (std::mem::size_of::<PassageOccurrence>() + 2 * std::mem::size_of::<VerifiedSourceSpan>()) as u64)?;
        sum(&[
            self.workspace_bytes(chunks)?, fields, spans,
            mul(limits.max_evidence_bytes as u64, 64)?,
            mul(limits.max_evidence_passages as u64, 1024)?,
            mul(verification.max_matches as u64, 128)?,
            mul(verification.max_fields as u64, 128)?,
            mul(self.budget.max_output_bytes, 4)?,
            mul(u64::from(self.budget.max_output_tokens), 8)?,
        ])
    }
}

impl NlpEngine {
    /// Answer a single private question using every collected verbatim citation
    /// from a complete source-document map. Map answers remain inspectable;
    /// they are never presented to the final model as factual source material.
    /// All work, masks and independent verification share nonrenewable limits.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_int4_document_answer(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceQuestionSynthesis>, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4QuestionSynthesisRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(&config.identity, model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let width = model.model_identity().group_width();
        let required = synthesis_requirements(&config, source.len(), width, &planner)?;
        let bytes = synthesis_input_bytes(source.capacity(), &config)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Retain both private strings and their complete charge until the
            // native engine and all evidence owners have physically drained.
            let input = input;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("INT4 answer synthesis task context"))?;
            let prepared = input.value.planner.plan_int4_question_synthesis_with_control(
                &input.value.source, &config.task, config.budget, &context,
                config.planning, config.mapping, width, control).map_err(map_error)?;
            let expected = prepared.preflight_metadata();
            let discovery = expected.discovery();
            if prepared.group_width() != width || discovery.group_width() != width
                || discovery.native_chunks() == 0
                || prepared.execution_identities().len() != discovery.native_chunks()
                || required.kv_bytes > prepared.max_kv_bytes() {
                return Err(HostedError::ModelIdentity);
            }
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                config.question_synthesis_workspace_bytes(discovery.chunk_count())?)?;
            // Discovery and final answers contain text/citations; no native
            // token-id transcripts outlive this charged temporary workspace.
            let output = output_claim(&lease, prepared.max_result_bytes(), 0)?;
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

fn synthesis_requirements(config: &Int4SourceMapConfig<SourceQuestionSynthesis>,
    source_bytes: usize, width: usize, planner: &SourceTaskPlanner)
    -> Result<Int4MemoryRequirement, HostedError> {
    let required = common_requirements(config, source_bytes, width, planner)?;
    if config.identity.task_spec != ANSWER_TASK_VERSION { return Err(HostedError::ModelIdentity); }
    config.task.question.validate().map_err(map_error)?;
    let limits = config.task.limits;
    if config.task.question.question.len() > config.planning.max_input_bytes
        || config.task.question.question.len() > config.budget.max_input_tokens as usize
        || !(1..=1024).contains(&limits.max_evidence_passages)
        || limits.max_evidence_passages > config.planning.max_passages
        || limits.max_evidence_bytes == 0 || limits.max_evidence_bytes > config.planning.max_input_bytes {
        return Err(HostedError::Limits("INT4 answer synthesis question or evidence limits"));
    }
    config.question_synthesis_workspace_bytes(config.mapping.chunks.max_chunks)?;
    Ok(required)
}

fn synthesis_input_bytes(source_capacity: usize, config: &Int4SourceMapConfig<SourceQuestionSynthesis>)
    -> Result<u64, HostedError> {
    sum(&[source_capacity as u64, config.task.question.question.capacity() as u64,
        config.preparation_reserve_bytes])
}

#[cfg(test)]
mod tests;
