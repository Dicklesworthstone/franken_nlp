//! One immutable question across every source chunk on one charged INT4 engine.
//! Passage answers and abstentions are retained without global answer selection.
use super::*;
use crate::tasks::{
    answer::{ANSWER_TASK_VERSION, PassageCitation, PassageOccurrence},
    source_planning::grouped_int4::long::question::{Int4QuestionRun, SourceQuestion},
    summarize::SourceCitation,
};

impl Int4SourceMapConfig<SourceQuestion> {
    /// Complete frontier plus native local, independent scan and lifted
    /// citation storage. This is a modeled reservation, not measured RSS.
    pub fn question_workspace_bytes(&self, chunks: usize) -> Result<u64, HostedError> {
        let multiply = |n: u64, m: u64| n.checked_mul(m)
            .ok_or(HostedError::Limits("INT4 question workspace arithmetic"));
        let citations = multiply(self.task.verification.max_fields as u64,
            (std::mem::size_of::<PassageCitation>() + std::mem::size_of::<SourceCitation>()) as u64)?;
        let spans = multiply(self.task.verification.max_matches as u64,
            (std::mem::size_of::<PassageOccurrence>() + 2 * std::mem::size_of::<VerifiedSourceSpan>()) as u64)?;
        sum(&[self.workspace_bytes(chunks)?, citations, spans])
    }
}

impl NlpEngine {
    /// Apply the same private question to every nonblank document chunk.
    /// Preserve all answers, abstentions and interior whitespace in source
    /// order with original-coordinate citations. Distinct texts are not votes
    /// or semantic contradiction labels. Empty/wholly blank documents refuse.
    #[allow(clippy::too_many_arguments)]
    pub fn answer_int4_document(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig<SourceQuestion>, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4QuestionRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(&config.identity, model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let width = model.model_identity().group_width();
        let required = question_requirements(&config, source.len(), width, &planner)?;
        let bytes = input_bytes(source.capacity(), &config)?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Keep the whole charged source/question capture alive, avoiding
            // field-disjoint capture releasing its charge before its storage.
            let input = input;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("INT4 document question context"))?;
            let prepared = input.value.planner.plan_int4_question_with_control(&input.value.source,
                &config.task, config.budget, &context, config.planning, config.mapping, width, control)
                .map_err(map_error)?;
            let expected = prepared.preflight_metadata();
            if prepared.group_width() != width || expected.group_width() != width
                || expected.native_chunks() == 0
                || prepared.execution_identities().len() != expected.native_chunks()
                || required.kv_bytes > prepared.max_kv_bytes() {
                return Err(HostedError::ModelIdentity);
            }
            // Whitespace ranges occupy the result even without native plans.
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers,
                config.question_workspace_bytes(expected.chunk_count())?)?;
            let output = output_claim(&lease, prepared.max_result_bytes(), 0)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(expected.native_chunks())
                .map_err(|_| map_error(Int4SourceMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                checkpoint(control)?;
                constrained_int4::check_model(identity, model.model_identity())
                    .map_err(|_| HostedError::ModelIdentity)?;
                admitted.push(identity.clone());
            }
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
                let result = prepared.execute_with_control(&admitted,
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
fn question_requirements(config: &Int4SourceMapConfig<SourceQuestion>, source_bytes: usize,
    width: usize, planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError> {
    let required = common_requirements(config, source_bytes, width, planner)?;
    if config.identity.task_spec != ANSWER_TASK_VERSION { return Err(HostedError::ModelIdentity); }
    config.task.validate().map_err(map_error)?;
    if config.task.question.len() > config.planning.max_input_bytes
        || config.task.question.len() > config.budget.max_input_tokens as usize {
        return Err(HostedError::Limits("INT4 document question input budget"));
    }
    config.question_workspace_bytes(config.mapping.chunks.max_chunks)?;
    Ok(required)
}
fn input_bytes(source_capacity: usize, config: &Int4SourceMapConfig<SourceQuestion>) -> Result<u64, HostedError> {
    sum(&[source_capacity as u64, config.task.question.capacity() as u64, config.preparation_reserve_bytes])
}

#[cfg(test)]
mod tests;
