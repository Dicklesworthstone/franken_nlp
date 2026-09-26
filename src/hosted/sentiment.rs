//! Independent affect heads on the existing charged resident model/runtime.
use super::*;
use crate::{native_engine::strict_int8::{Int8Work, scoring::Int8ScoringBudget},
    tasks::{ir::TaskBudget, sentiment::quantized::{PreparedInt8Sentiment, Int8SentimentRun, Int8SentimentError}}};

/// Whole-bundle compute and caller-priced input ownership. Preparation is a
/// modeled reservation, not measured RSS. Before transfer the caller remains
/// responsible for admitting the already-compiled plan's allocations.
#[derive(Clone, Copy, Debug)]
pub struct SentimentHostLimits {
    pub native: NativeLimits,
    pub preparation_reserve_bytes: u64,
    pub max_model_work: Int8Work,
}
impl NlpEngine {
    /// All sentiment axes share one native allocation and blocking invocation.
    /// No partial axis response escapes; input and native storage drain before
    /// return, while the output guard survives caller serialization/delivery.
    pub fn execute_int8_sentiment(&self, model: &ResidentInt8, prepared: PreparedInt8Sentiment,
        limits: SentimentHostLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8SentimentRun>, HostedError> {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        let required = requirements(limits.native)?;
        let work = prepared.planned_work();
        validate(limits, prepared.task_budget(), prepared.required_context(), required.kv_bytes, work)?;
        let lease = self.resources().acquire_lease();
        let output = output_claim(&lease, prepared.max_result_bytes(), u64::from(prepared.task_budget().max_output_tokens))?;
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, limits.preparation_reserve_bytes)?,
            || Ok(prepared))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, limits.native.allocator_reserve_bytes])?)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let input = input; // capture storage and its charge as one ordered aggregate
            let mut engine = allocate_native(kv, scratch, || model.inner.loaded.value
                .engine(limits.native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let result = allocate(output, || {
                let result = input.value.execute_with_control(input.value.execution_identity(), &mut engine.value,
                    Int8ScoringBudget { native: Int8RunBudget::exact(work), max_kv_bytes: required.kv_bytes }, control)
                    .map_err(HostedError::Sentiment)?;
                if result.model_work != work || result.head_count != input.value.head_count()
                    || engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(HostedError::Sentiment(Int8SentimentError::Accounting));
                }
                Ok(result)
            })?;
            drop(engine);
            drop(input);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
fn validate(limits: SentimentHostLimits, task: TaskBudget, context: usize, kv: u64, work: Int8Work)
    -> Result<(), HostedError> {
    task.validate().map_err(|_| HostedError::Limits("sentiment task budget"))?;
    let cap = limits.max_model_work;
    if limits.preparation_reserve_bytes == 0 || context == 0 || context > limits.native.context_tokens
        || kv == 0 || kv > task.max_kv_bytes {
        return Err(HostedError::Limits("sentiment context or preparation admission"));
    }
    if work.forward_positions > cap.forward_positions || work.projected_logits > cap.projected_logits
        || work.attention_pairs > cap.attention_pairs || !work.projections.fits(cap.projections) {
        return Err(HostedError::Limits("sentiment complete model work"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn case() -> (SentimentHostLimits, TaskBudget, Int8Work) {
        let work = Int8Work::for_sequence(0, 64, 1000).unwrap();
        (SentimentHostLimits { native: NativeLimits { context_tokens: 64, allocator_reserve_bytes: 1 << 20,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 1000, cleanup_reserve_bytes: 65536 } },
            preparation_reserve_bytes: 1 << 20, max_model_work: work },
            TaskBudget { max_input_tokens: 64, max_output_tokens: 16, max_output_bytes: 65536,
                max_grammar_states: 4096, max_kv_bytes: 1024 }, work)
    }
    #[test]
    fn each_resource_axis_is_enforced_before_dispatch() {
        let (limits, task, work) = case();
        validate(limits, task, 64, 1024, work).unwrap();
        for axis in 0..6 {
            let mut limits = limits;
            match axis { 0 => limits.preparation_reserve_bytes = 0, 1 => limits.max_model_work.forward_positions -= 1,
                2 => limits.max_model_work.projected_logits -= 1, 3 => limits.max_model_work.attention_pairs -= 1,
                4 => limits.max_model_work.projections.dot_products -= 1,
                _ => limits.max_model_work.projections.multiply_accumulates -= 1 }
            assert!(validate(limits, task, 64, 1024, work).is_err());
        }
    }
    #[test]
    fn whole_kv_and_largest_head_not_aggregate_forward_count_determine_capacity() {
        let (mut limits, task, work) = case(); let work = work.checked_add(work).unwrap();
        limits.max_model_work = work;
        validate(limits, task, 64, 1024, work).unwrap();
        assert!(validate(limits, task, 65, 1024, work).is_err());
        assert!(validate(limits, task, 64, 1025, work).is_err());
        assert!(validate(limits, task, 0, 1024, work).is_err());
    }
    #[test]
    fn typed_task_cause_survives_without_private_default_diagnostics() {
        let cause = crate::native_engine::decode::DecodeCancellationKind::Deadline;
        let error = HostedError::Sentiment(Int8SentimentError::Cancelled(cause));
        assert_eq!(error.to_string(), "hosted native sentiment failed");
        assert_eq!(format!("{error:?}"), error.to_string());
        assert!(error.source().is_some());
        let HostedError::Sentiment(inner) = error else { unreachable!() };
        assert_eq!(inner.cancellation(), Some(cause));
    }
    #[test]
    fn plan_and_output_support_owned_blocking_transfer() {
        fn send<T: Send + 'static>() {}
        send::<PreparedInt8Sentiment>(); send::<Int8SentimentRun>();
    }
}
