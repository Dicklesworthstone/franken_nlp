//! Complete finite-head tasks on the existing process-owned INT8 invocation.
//! Serial context capacity is the largest live head, NOT aggregate forward work.

use super::*;
use crate::{
    native_engine::strict_int8::{scoring::Int8ScoringBudget, prefill::Int8PrefillLimits},
    tasks::{classify::quantized::{Int8ClassificationError, Int8ClassificationRun, PreparedInt8Classification},
        ir::TaskBudget,
        judge::quantized::{Int8JudgeError, Int8JudgeRun, PreparedInt8Judge}},
};
mod cohort;

impl NlpEngine {
    /// Execute exclusive or independent multi-label classification. The sealed
    /// planner supplies exact full-vocabulary/EOS scores, not sampled labels.
    /// Every head is preflighted and charged before the first native forward.
    /// The returned guard owns result memory through the caller's delivery.
    pub fn execute_int8_classify(&self, model: &ResidentInt8,
        prepared: PreparedInt8Classification, native: NativeLimits,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8ClassificationRun>, HostedError> {
        self.execute_int8_classify_scheduled(model, prepared, native, None, cancellation)
    }
    /// Explicit grouped-prompt classification. Additional workspace is charged
    /// once to this process, reused by sequential heads, and retained with the
    /// native engine until all head/result checks finish. No new runtime.
    pub fn execute_int8_classify_layer_major(&self, model: &ResidentInt8,
        prepared: PreparedInt8Classification, native: NativeLimits, prefill: Int8PrefillLimits,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8ClassificationRun>, HostedError> {
        self.execute_int8_classify_scheduled(model, prepared, native, Some(prefill), cancellation)
    }
    fn execute_int8_classify_scheduled(&self, model: &ResidentInt8,
        prepared: PreparedInt8Classification, native: NativeLimits, prefill: Option<Int8PrefillLimits>,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8ClassificationRun>, HostedError> {
        dispatch::preflight(self, native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        let required = requirements(native)?;
        let task = prepared.task_budget();
        check_capacity(prepared.required_context(), task, native.context_tokens, required.kv_bytes)?;
        let work = prepared.planned_work();
        let scratch_bytes = scoring_scratch(sum(&[required.rope_bytes, required.scratch_payload_bound,
            native.allocator_reserve_bytes])?, prefill)?;
        let lease = self.resources().acquire_lease();
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
        let output = output_claim(&lease, task.max_output_bytes, u64::from(task.max_output_tokens))?;
        let model = model.clone();
        dispatch::run(self, native.run, cancellation, move |control| {
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            // allocate() drops a result before its reservation on commit failure.
            let result = allocate(output, || {
                let budget = Int8ScoringBudget { native: Int8RunBudget::exact(work), max_kv_bytes: required.kv_bytes };
                let result = match prefill {
                    Some(prefill) => prepared.execute_layer_major_with_control(prepared.execution_identity(),
                        &mut engine.value, budget, prefill, control),
                    None => prepared.execute_with_control(prepared.execution_identity(), &mut engine.value, budget, control),
                }.map_err(HostedError::Classification)?;
                if result.model_work != work || result.head_count != prepared.head_count()
                    || engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(HostedError::Classification(Int8ClassificationError::Accounting));
                }
                Ok(result)
            })?;
            drop(engine);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }

    /// Execute both pairwise orders, every rubric criterion or the complete
    /// full-source/evidence bundle on the shared resident INT8 model. A failed
    /// head never becomes a partial judgment, and scores are not truth claims.
    pub fn execute_int8_judge(&self, model: &ResidentInt8,
        prepared: PreparedInt8Judge, native: NativeLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8JudgeRun>, HostedError> {
        self.execute_int8_judge_scheduled(model, prepared, native, None, cancellation)
    }
    /// Group prompts while retaining the same full-bundle admission, finalizer,
    /// cancellation region and guarded output. This is not head/document batching.
    pub fn execute_int8_judge_layer_major(&self, model: &ResidentInt8,
        prepared: PreparedInt8Judge, native: NativeLimits, prefill: Int8PrefillLimits,
        cancellation: CancellationToken) -> Result<HostedOutput<Int8JudgeRun>, HostedError> {
        self.execute_int8_judge_scheduled(model, prepared, native, Some(prefill), cancellation)
    }
    fn execute_int8_judge_scheduled(&self, model: &ResidentInt8,
        prepared: PreparedInt8Judge, native: NativeLimits, prefill: Option<Int8PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8JudgeRun>, HostedError> {
        dispatch::preflight(self, native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        let required = requirements(native)?;
        let task = prepared.task_budget();
        check_capacity(prepared.required_context(), task, native.context_tokens, required.kv_bytes)?;
        let work = prepared.planned_work();
        let scratch_bytes = scoring_scratch(sum(&[required.rope_bytes, required.scratch_payload_bound,
            native.allocator_reserve_bytes])?, prefill)?;
        let lease = self.resources().acquire_lease();
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch_bytes)?;
        let output = output_claim(&lease, prepared.max_result_bytes(), u64::from(task.max_output_tokens))?;
        let model = model.clone();
        dispatch::run(self, native.run, cancellation, move |control| {
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let result = allocate(output, || {
                let budget = Int8ScoringBudget { native: Int8RunBudget::exact(work), max_kv_bytes: required.kv_bytes };
                let result = match prefill {
                    Some(prefill) => prepared.execute_layer_major_with_control(prepared.execution_identity(),
                        &mut engine.value, budget, prefill, control),
                    None => prepared.execute_with_control(prepared.execution_identity(), &mut engine.value, budget, control),
                }.map_err(HostedError::Judge)?;
                if result.model_work != work || result.head_count != prepared.head_count()
                    || engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(HostedError::Judge(Int8JudgeError::Accounting));
                }
                Ok(result)
            })?;
            drop(engine);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}

/// Shared by scored task hosts. The native row geometry derives the payload;
/// callers cannot discount existing scratch or underprice the extra workspace.
/// This validates arithmetic only; each host must hold its actual reservation.
pub(super) fn scoring_scratch(ordinary: u64, prefill: Option<Int8PrefillLimits>) -> Result<u64, HostedError> {
    let extra = prefill.map(Int8PrefillLimits::validate).transpose().map_err(HostedError::Native)?.unwrap_or(0);
    sum(&[ordinary, extra])
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
    #[test]
    fn judge_errors_preserve_typed_causes_and_safe_display() {
        let error = HostedError::Judge(Int8JudgeError::Identity);
        assert!(error.source().is_some());
        assert_eq!(error.to_string(), format!("{error:?}"));
        assert_eq!(error.to_string(), "hosted native judgment failed");
    }
    #[test]
    fn judge_plans_and_results_can_be_owned_by_the_blocking_closure() {
        fn send<T: Send + 'static>() {}
        send::<PreparedInt8Judge>();
        send::<Int8JudgeRun>();
        send::<Arc<crate::tasks::judge::JudgePlanner>>();
    }
    #[test]
    fn grouped_scoring_preserves_all_existing_scratch_and_adds_derived_payload() {
        let ordinary = 12_345_u64;
        assert_eq!(scoring_scratch(ordinary, None).unwrap(), ordinary);
        for rows in [1, 4, 64] {
            let extra = Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
            let limits = Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: extra };
            assert_eq!(scoring_scratch(ordinary, Some(limits)).unwrap(), ordinary + extra);
        }
    }
    #[test]
    fn scoring_workspace_cannot_be_underpriced_or_overflow_the_process_ledger() {
        let extra = Int8PrefillLimits::required_extra_scratch_bytes(4).unwrap();
        let limits = Int8PrefillLimits { max_batch_rows: 4, max_extra_scratch_bytes: extra };
        assert!(scoring_scratch(1, Some(Int8PrefillLimits { max_extra_scratch_bytes: extra - 1, ..limits })).is_err());
        assert_eq!(scoring_scratch(u64::MAX - extra, Some(limits)).unwrap(), u64::MAX);
        assert!(scoring_scratch(u64::MAX - extra + 1, Some(limits)).is_err());
    }
    #[test]
    fn scored_hosts_refuse_invalid_prefill_geometry_before_reserving() {
        for rows in [0, 65, usize::MAX] {
            assert!(scoring_scratch(1024, Some(Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX })).is_err());
        }
    }
    #[test]
    fn an_oversized_allowance_does_not_replace_the_geometry_derived_charge() {
        let limits = Int8PrefillLimits { max_batch_rows: 4, max_extra_scratch_bytes: u64::MAX };
        let extra = Int8PrefillLimits::required_extra_scratch_bytes(4).unwrap();
        assert_eq!(scoring_scratch(8192, Some(limits)).unwrap(), 8192 + extra);
    }
}
