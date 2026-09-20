//! Complete finite-head tasks on the existing process-owned INT8 invocation.
//! Context capacity is the largest live head, NOT aggregate forward work.

use super::*;
use crate::{
    native_engine::strict_int8::scoring::Int8ScoringBudget,
    tasks::{classify::quantized::{Int8ClassificationError, Int8ClassificationRun, PreparedInt8Classification},
        ir::TaskBudget},
};

impl NlpEngine {
    /// Execute exclusive or independent multi-label classification. The sealed
    /// planner supplies exact full-vocabulary/EOS scores, not sampled labels.
    /// Every head is preflighted and charged before the first native forward.
    /// The returned guard owns result memory through the caller's delivery.
    pub fn execute_int8_classify(&self, model: &ResidentInt8,
        prepared: PreparedInt8Classification, native: NativeLimits,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8ClassificationRun>, HostedError> {
        dispatch::preflight(self, native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        let required = requirements(native)?;
        let task = prepared.task_budget();
        check_capacity(prepared.required_context(), task, native.context_tokens, required.kv_bytes)?;
        let work = prepared.planned_work();
        let lease = self.resources().acquire_lease();
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, native.allocator_reserve_bytes])?)?;
        let output = output_claim(&lease, task.max_output_bytes, u64::from(task.max_output_tokens))?;
        let model = model.clone();
        dispatch::run(self, native.run, cancellation, move |control| {
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let result = prepared.execute_with_control(prepared.execution_identity(), &mut engine.value,
                Int8ScoringBudget { native: Int8RunBudget::exact(work), max_kv_bytes: required.kv_bytes }, control)
                .map_err(HostedError::Classification)?;
            if result.model_work != work || engine.value.is_poisoned()
                || !engine.value.kv_cache().all_slots_have_len(0) {
                return Err(HostedError::Classification(Int8ClassificationError::Accounting));
            }
            drop(engine);
            let committed = output.commit()?;
            drop(lease);
            Ok(GuardedOutput::new(result, committed))
        })
    }
}

// The engine allocates its FULL admitted KV capacity. Checking only the
// largest prompt's payload would permit an underfunded allocation. Conversely,
// adding heads' forward counts would reject valid sequential reuse of that KV.
fn check_capacity(required_context: usize, task: TaskBudget,
    context_capacity: usize, allocated_kv_bytes: u64) -> Result<(), HostedError> {
    task.validate().map_err(|_| HostedError::Limits("scored task budget"))?;
    if required_context == 0 || required_context > context_capacity
        || allocated_kv_bytes == 0 || allocated_kv_bytes > task.max_kv_bytes {
        return Err(HostedError::Limits("scored context or whole KV allocation"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn budget() -> TaskBudget {
        TaskBudget { max_input_tokens: 4096, max_output_tokens: 64,
            max_output_bytes: 8192, max_grammar_states: 4096, max_kv_bytes: 65536 }
    }
    #[test]
    fn exact_context_and_full_kv_boundary_fit() {
        check_capacity(128, budget(), 128, 65536).unwrap();
    }
    #[test]
    fn larger_reserved_engine_is_not_priced_as_just_the_prompt() {
        assert!(check_capacity(16, budget(), 128, 65537).is_err());
    }
    #[test]
    fn empty_or_oversized_head_refuses() {
        for context in [0, 129, usize::MAX] {
            assert!(check_capacity(context, budget(), 128, 65536).is_err());
        }
    }
    #[test]
    fn missing_kv_reservation_refuses() {
        assert!(check_capacity(16, budget(), 128, 0).is_err());
    }
    #[test]
    fn independent_heads_reuse_one_context_but_sum_work() {
        use crate::native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int8::Int8Work};
        let head = Int8Work::for_sequence(0, 100, 3 * NANBEIGE_VOCAB_SIZE).unwrap();
        let complete = head.checked_add(head).unwrap();
        assert!(complete.forward_positions > 128);
        check_capacity(100, budget(), 128, 65536).unwrap();
        assert_eq!(complete.attention_pairs, 2 * head.attention_pairs);
        assert_eq!(complete.projections.multiply_accumulates, 2 * head.projections.multiply_accumulates);
    }
    #[test]
    fn classified_errors_keep_typed_causes_without_private_debug_strings() {
        let error = HostedError::Classification(Int8ClassificationError::Identity);
        assert!(error.source().is_some());
        assert_eq!(error.to_string(), format!("{error:?}"));
        assert_eq!(error.to_string(), "hosted native classification failed");
    }
    #[test]
    fn owned_plans_cross_the_real_blocking_boundary() {
        fn send<T: Send + 'static>() {}
        send::<PreparedInt8Classification>();
        send::<Int8ClassificationRun>();
        send::<Arc<crate::tasks::classify::ClassificationPlanner>>();
    }
}
