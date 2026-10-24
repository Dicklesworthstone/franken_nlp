//! Whole-document source tasks on one charged grouped-INT4 engine.
//! Every chunk is planned and admitted before the first native forward.
use super::*;
use crate::{
    native_engine::{constrained_int4, lmhead::NANBEIGE_VOCAB_SIZE},
    tasks::{ir::{PlanContext, TaskBudget},
        source_planning::{SourcePlanningLimits, SourceTaskPlanner,
            grouped_int4::long::{Int4SourceMapError, Int4SourceMapLimits, Int4SourceMapRun,
                SourceMapTask}}},
    validation::grounded_fields::{SourceFieldEvidence, VerifiedSourceSpan},
};
mod summary;
mod question;

/// One original document, fixed task/model and nonrenewable work envelope.
/// Preparation prices all retained prompts/grammars, tokenizer/vocabulary
/// owners and configuration. Reduction reserve prices metadata/allocator
/// overhead beyond the explicit result/frontier/evidence payload floor.
pub struct Int4SourceMapConfig<T = SourceMapTask> {
    pub identity: ExecutionIdentity,
    pub task: T,
    pub budget: TaskBudget,
    pub planning: SourcePlanningLimits,
    pub mapping: Int4SourceMapLimits,
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub reduction_reserve_bytes: u64,
}
impl<T> Int4SourceMapConfig<T> {
    /// Checked complete-frontier quote shared with CLI preflight. This is not
    /// a memory permit or an observed RSS limit; the host reserves it below.
    pub fn workspace_bytes(&self, chunks: usize) -> Result<u64, HostedError> {
        if chunks == 0 || chunks > self.mapping.chunks.max_chunks || chunks > 256
            || self.reduction_reserve_bytes == 0 {
            return Err(HostedError::Limits("INT4 source map partition or reduction reserve"));
        }
        let multiply = |n: u64, m: u64| n.checked_mul(m)
            .ok_or(HostedError::Limits("INT4 source map workspace arithmetic"));
        let native = multiply(self.budget.max_output_bytes, chunks as u64)?;
        let staged = multiply(sum(&[native, self.mapping.reduction.max_live_value_bytes as u64])?, 4)?;
        let tokens = multiply(multiply(u64::from(self.budget.max_output_tokens), chunks as u64)?, 8)?;
        // Both local and original-coordinate occurrence arrays can coexist
        // with a temporary independent verification scan.
        let fields = multiply(self.mapping.verification.max_fields as u64,
            std::mem::size_of::<SourceFieldEvidence>() as u64)?;
        let spans = multiply(multiply(self.mapping.verification.max_matches as u64,
            std::mem::size_of::<VerifiedSourceSpan>() as u64)?, 2)?;
        sum(&[staged, tokens, fields, spans, self.reduction_reserve_bytes])
    }
}
struct SourceMapInput<T = SourceMapTask> {
    source: String,
    planner: Arc<SourceTaskPlanner>,
    vocabulary: Arc<ExtractionVocabulary>,
    config: Int4SourceMapConfig<T>,
}

impl NlpEngine {
    /// Execute lossless document-wide NER, keyphrase or cited-summary maps.
    /// Output keeps independent chunk results plus original-document evidence;
    /// no global ranking or neural reconciliation is implied by this method.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int4_source(&self, model: &ResidentInt4, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: Int4SourceMapConfig, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4SourceMapRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(&config.identity, model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        let width = model.model_identity().group_width();
        let required = map_requirements(&config, source.len(), width, &planner)?;
        let input_bytes = sum(&[source.capacity() as u64, config.preparation_reserve_bytes])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            let input = input;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("INT4 source map task context"))?;
            let prepared = input.value.planner.plan_int4_map_with_control(&input.value.source,
                &config.task, config.budget, &context, config.planning, config.mapping, width, control)
                .map_err(map_error)?;
            let expected = prepared.preflight_metadata();
            let count = prepared.chunk_count();
            if prepared.group_width() != width || expected.group_width() != width
                || prepared.execution_identities().len() != count || required.kv_bytes > prepared.max_kv_bytes() {
                return Err(HostedError::ModelIdentity);
            }
            let tokens = (count as u64).checked_mul(u64::from(config.budget.max_output_tokens))
                .ok_or(HostedError::Limits("INT4 source map output tokens"))?;
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers, config.workspace_bytes(count)?)?;
            let output = output_claim(&lease, prepared.max_result_bytes(), tokens)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(count).map_err(|_| map_error(Int4SourceMapError::Allocation))?;
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

fn map_requirements(config: &Int4SourceMapConfig, source_bytes: usize, width: usize,
    planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError> {
    let required = common_requirements(config, source_bytes, width, planner)?;
    let task = match &config.task {
        SourceMapTask::Ner(_) => "ner-v1", SourceMapTask::Keyphrases(_) => "keyphrases-v1",
        SourceMapTask::Summarize(_) => "summarize-v1",
    };
    if config.identity.task_spec != task {
        return Err(HostedError::ModelIdentity);
    }
    Ok(required)
}
fn common_requirements<T>(config: &Int4SourceMapConfig<T>, source_bytes: usize, width: usize,
    planner: &SourceTaskPlanner) -> Result<Int4MemoryRequirement, HostedError> {
    config.native.run.validate()?;
    config.budget.validate().map_err(|_| HostedError::Limits("INT4 source map task budget"))?;
    config.identity.validate().map_err(|_| HostedError::ModelIdentity)?;
    constrained_int4::check_profile(&config.identity, width).map_err(|_| HostedError::ModelIdentity)?;
    if config.identity.tokenizer_digest != planner.tokenizer_digest()
        || config.identity.template_digest != *planner.template_digest() {
        return Err(HostedError::ModelIdentity);
    }
    let required = Int4MemoryRequirement::for_context(config.native.context_tokens, width).map_err(native_error)?;
    if source_bytes == 0 || source_bytes > config.mapping.chunks.max_input_bytes
        || config.preparation_reserve_bytes == 0 || config.reduction_reserve_bytes == 0
        || config.native.allocator_reserve_bytes == 0
        || config.planning.max_context_tokens > config.native.context_tokens
        || required.kv_bytes > config.budget.max_kv_bytes
        || config.mapping.max_model_work.projected_logits < NANBEIGE_VOCAB_SIZE as u64 {
        return Err(HostedError::Limits("INT4 source map input, preparation or resident capacity"));
    }
    Ok(required)
}
fn map_error(error: Int4SourceMapError) -> HostedError { HostedError::Int4(Int4HostError::SourceMap(error)) }
fn check_clean(engine: &StrictInt4Engine<'_>) -> Result<(), HostedError> {
    if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
        return Err(HostedError::Int4(Int4HostError::SourceTask(
            crate::tasks::source_planning::grouped_int4::Int4SourceError::InvalidResult)));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
