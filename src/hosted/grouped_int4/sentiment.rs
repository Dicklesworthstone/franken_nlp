//! Complete independent sentiment dimensions under one charged INT4 invocation.
use super::*;
use crate::{
    native_engine::strict_int4::{Int4Work, scoring::Int4ScoringBudget},
    tasks::{ir::TaskBudget, sentiment::grouped_int4::{self, Int4SentimentError,
        Int4SentimentRun, PreparedInt4Sentiment}},
};

/// Complete-bundle work and input ownership. Before transfer, callers admit
/// the compiled plan's storage. The host retains its own explicit preparation
/// charge until physical execution completes. Reserves are not RSS measures.
#[derive(Clone, Copy, Debug)]
pub struct Int4SentimentHostLimits {
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub max_model_work: Int4Work,
}
impl NlpEngine {
    /// Every independent dimension completes under one allocation and one
    /// blocking invocation on the actual model/group partition. Logical KV
    /// clears between heads. Failure never publishes partial axis results.
    pub fn execute_int4_sentiment(&self, model: &ResidentInt4, prepared: PreparedInt4Sentiment,
        limits: Int4SentimentHostLimits, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4SentimentRun>, HostedError> {
        dispatch::preflight(self, limits.native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        grouped_int4::check_model(prepared.execution_identity(), model.model_identity()).map_err(sentiment_error)?;
        if prepared.group_width() != model.model_identity().group_width() { return Err(HostedError::ModelIdentity); }
        let work = prepared.planned_work();
        let (required, scratch_bytes) = requirements(limits, prepared.task_budget(),
            prepared.required_context(), prepared.group_width(), work)?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, limits.preparation_reserve_bytes)?,
            || Ok(prepared))?;
        let output = output_claim(&lease, input.value.max_result_bytes(),
            u64::from(input.value.task_budget().max_output_tokens))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let input = input;
            let mut engine = allocate_native(kv, scratch, || {
                StrictInt4Engine::new(&model.inner.loaded.value, limits.native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                    max_scratch_payload_bytes: required.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let result = allocate(output, || {
                let result = input.value.execute_with_control(input.value.execution_identity(), &mut engine.value,
                    Int4ScoringBudget { native: Int4RunBudget::exact(work), max_kv_bytes: required.kv_bytes }, control)
                    .map_err(sentiment_error)?;
                if result.model_work != work || result.head_count != input.value.head_count()
                    || result.group_width != input.value.group_width()
                    || engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(sentiment_error(Int4SentimentError::Accounting));
                }
                checkpoint(control)?;
                Ok(result)
            })?;
            drop(engine);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
fn sentiment_error(error: Int4SentimentError) -> HostedError { HostedError::Int4(Int4HostError::Sentiment(error)) }
fn requirements(limits: Int4SentimentHostLimits, task: TaskBudget, context: usize,
    width: usize, work: Int4Work) -> Result<(Int4MemoryRequirement, u64), HostedError> {
    limits.native.run.validate()?;
    task.validate().map_err(|_| HostedError::Limits("INT4 sentiment task budget"))?;
    if limits.preparation_reserve_bytes == 0 || limits.native.allocator_reserve_bytes == 0
        || context == 0 || context > limits.native.context_tokens {
        return Err(HostedError::Limits("INT4 sentiment context or preparation admission"));
    }
    let required = Int4MemoryRequirement::for_context(limits.native.context_tokens, width).map_err(native_error)?;
    if required.kv_bytes > task.max_kv_bytes { return Err(HostedError::Limits("INT4 sentiment complete resident KV capacity")); }
    let cap = limits.max_model_work;
    if work.forward_positions > cap.forward_positions || work.projected_logits > cap.projected_logits
        || work.attention_pairs > cap.attention_pairs || !work.projections.fits(cap.projections) {
        return Err(HostedError::Limits("INT4 sentiment complete model work"));
    }
    let scratch = sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?;
    Ok((required, scratch))
}

#[cfg(test)]
mod tests;
