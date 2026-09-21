//! Owned long-document planning and native map/merge on the existing host.
use super::*;
use crate::{
    native_engine::{constrained_int8, lmhead::NANBEIGE_VOCAB_SIZE},
    tasks::{ir::{PlanContext, TaskBudget}, source_planning::{SourcePlanningLimits, SourceTaskPlanner,
        quantized::long::{Int8SourceMapError, Int8SourceMapLimits, Int8SourceMapRun, SourceMapTask}}},
};

/// One fixed task, one whole-document commitment and one physical invocation.
/// Preparation must price all retained prompts/grammars, tokenizer/vocabulary
/// Arcs and configuration metadata. Reduction headroom prices non-payload
/// objects/allocator overhead in addition to the modeled payload reservation.
/// Neither reservation is an allocator interceptor or an observed RSS bound.
pub struct SourceMapConfig {
    pub identity: ExecutionIdentity,
    pub task: SourceMapTask,
    pub budget: TaskBudget,
    pub planning: SourcePlanningLimits,
    pub mapping: Int8SourceMapLimits,
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub reduction_reserve_bytes: u64,
}
struct SourceMapInput {
    source: String,
    planner: Arc<SourceTaskPlanner>,
    vocabulary: Arc<ExtractionVocabulary>,
    config: SourceMapConfig,
}

impl NlpEngine {
    /// Plan and execute a complete original document using real process-owned
    /// weights, memory and cancellation. NER/keyphrases/summaries remain typed
    /// independent chunk results, with original-source citation coordinates.
    /// This neither loads a model nor creates a runtime, admission provider,
    /// continuous batch, global synthesis pass, durable job or public CLI route.
    #[allow(clippy::too_many_arguments)]
    pub fn map_int8_source(&self, model: &ResidentInt8, source: String,
        planner: Arc<SourceTaskPlanner>, vocabulary: Arc<ExtractionVocabulary>,
        config: SourceMapConfig, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8SourceMapRun>, HostedError> {
        dispatch::preflight(self, config.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), &config.identity)?;
        let required = requirements(config.native)?;
        validate(&config, source.len(), required.kv_bytes)?;
        check_assets(&config.identity, planner.tokenizer_digest(), *planner.template_digest())?;
        let input_bytes = sum(&[source.capacity() as u64, config.preparation_reserve_bytes])?;
        let run = config.native.run;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, input_bytes)?,
            || Ok(SourceMapInput { source, planner, vocabulary, config }))?;
        let model = model.clone();
        dispatch::run(self, run, cancellation, move |control| {
            // Retain the WHOLE aggregate: source and owned preparation storage
            // must drop before the charge, including queued/error/unwind paths.
            let input = input;
            let config = &input.value.config;
            let context = PlanContext::new(&config.identity, config.budget)
                .map_err(|_| HostedError::Limits("source map task context"))?;
            let prepared = input.value.planner.plan_int8_map_with_control(&input.value.source,
                &config.task, config.budget, &context, config.planning, config.mapping, control)
                .map_err(HostedError::SourceMap)?;
            let count = prepared.chunk_count();
            let tokens = (count as u64).checked_mul(u64::from(config.budget.max_output_tokens))
                .ok_or(HostedError::Limits("source map token arithmetic"))?;
            // Price the actual complete partition, not max_chunks independent
            // engines. Keep transient output authority through final admission.
            let reduction = Pending::reserve(&lease, MemoryClass::JobBuffers, reduction_bytes(config, count)?)?;
            let output = output_claim(&lease, prepared.max_result_bytes(), tokens)?;
            let mut admitted = Vec::new();
            admitted.try_reserve_exact(count).map_err(|_| HostedError::SourceMap(Int8SourceMapError::Allocation))?;
            for identity in prepared.execution_identities() {
                check_model_identity(model.artifact_identity(), identity)?;
                constrained_int8::check_profile(identity).map_err(|_| HostedError::ModelIdentity)?;
                admitted.push(identity.clone());
            }
            let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
            let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
                sum(&[required.rope_bytes, required.scratch_payload_bound, config.native.allocator_reserve_bytes])?)?;
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(config.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            // The consumed plan has no borrowing output. All source validation,
            // merge/error cleanup and final envelope checks precede commit.
            let result = allocate(output, || prepared.execute_with_control(&admitted,
                &mut engine.value, &input.value.vocabulary, control).map_err(HostedError::SourceMap))?;
            drop(engine);
            drop(admitted);
            drop(input);
            drop(reduction);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

fn validate(config: &SourceMapConfig, source_bytes: usize, kv_bytes: u64) -> Result<(), HostedError> {
    config.native.run.validate()?;
    config.budget.validate().map_err(|_| HostedError::Limits("source map task budget"))?;
    config.identity.validate().map_err(|_| HostedError::ModelIdentity)?;
    constrained_int8::check_profile(&config.identity).map_err(|_| HostedError::ModelIdentity)?;
    let task = match &config.task {
        SourceMapTask::Ner(_) => "ner-v1", SourceMapTask::Keyphrases(_) => "keyphrases-v1",
        SourceMapTask::Summarize(_) => "summarize-v1",
    };
    if config.identity.task_spec != task { return Err(HostedError::ModelIdentity); }
    if source_bytes == 0 || source_bytes > config.mapping.chunks.max_input_bytes
        || config.preparation_reserve_bytes == 0 || config.reduction_reserve_bytes == 0
        || config.planning.max_context_tokens > config.native.context_tokens
        || kv_bytes > config.budget.max_kv_bytes {
        return Err(HostedError::Limits("source map input, preparation or complete native capacity"));
    }
    // A full projection per possible output token is required by this backend.
    if config.mapping.max_model_work.projected_logits < NANBEIGE_VOCAB_SIZE as u64 {
        return Err(HostedError::Limits("source map full-vocabulary work"));
    }
    Ok(())
}
fn check_assets(identity: &ExecutionIdentity, tokenizer: Sha256Digest, template: Sha256Digest) -> Result<(), HostedError> {
    if identity.tokenizer_digest != tokenizer || identity.template_digest != template {
        return Err(HostedError::ModelIdentity);
    }
    Ok(())
}
fn reduction_bytes(config: &SourceMapConfig, chunks: usize) -> Result<u64, HostedError> {
    if chunks == 0 || chunks > config.mapping.chunks.max_chunks || chunks > 256 {
        return Err(HostedError::Limits("source map partition cardinality"));
    }
    // Native chunk results may coexist with a retained reduction frontier
    // BEFORE their serialized limits are checked. Include all chunk envelopes,
    // duplicated coordinate payload and live serialized-value staging. The
    // final returned output is separately charged and remains charged later.
    let native = config.budget.max_output_bytes.checked_mul(chunks as u64)
        .ok_or(HostedError::Limits("source map native result arithmetic"))?;
    let staged = sum(&[native, config.mapping.reduction.max_live_value_bytes as u64])?
        .checked_mul(4).ok_or(HostedError::Limits("source map reduction staging arithmetic"))?;
    let tokens = u64::from(config.budget.max_output_tokens).checked_mul(chunks as u64)
        .and_then(|n| n.checked_mul(8)).ok_or(HostedError::Limits("source map native token arithmetic"))?;
    sum(&[staged, tokens, config.reduction_reserve_bytes])
}

#[cfg(test)] mod tests;
