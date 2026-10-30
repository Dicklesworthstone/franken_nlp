//! Complete exclusive/multi-label tasks on one charged resident INT4 model.
//! No sampler, uncharged model import, second runtime or cross-head KV fork.
use super::*;
use crate::{native_engine::strict_int4::{prefill::Int4PrefillLimits, scoring::Int4ScoringBudget},
    tasks::{classify::grouped_int4::{self, Int4ClassificationError, Int4ClassificationRun, PreparedInt4Classification},
        ir::TaskBudget}};

impl NlpEngine {
    /// Execute all classification heads under one existing host call. A
    /// source-converted or independently pinned snapshot-loaded ResidentInt4
    /// follows the same path. The caller separately admits planner/request
    /// storage; this host charges full native KV/scratch and complete output.
    /// The returned result guard remains live through the caller's delivery.
    pub fn execute_int4_classify(&self, model: &ResidentInt4, prepared: PreparedInt4Classification,
        native: NativeLimits, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ClassificationRun>, HostedError> {
        self.execute_int4_classify_scheduled(model, prepared, native, kernel, None, cancellation)
    }
    /// Opt in to bounded layer-major prompts for every classification head.
    /// Extra workspace is admitted once alongside full native/output storage;
    /// all work and result ownership remain within the same physical call.
    pub fn execute_int4_classify_layer_major(&self, model: &ResidentInt4, prepared: PreparedInt4Classification,
        native: NativeLimits, kernel: PackedInt4Kernel, prefill: Int4PrefillLimits,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4ClassificationRun>, HostedError> {
        self.execute_int4_classify_scheduled(model, prepared, native, kernel, Some(prefill), cancellation)
    }
    fn execute_int4_classify_scheduled(&self, model: &ResidentInt4, prepared: PreparedInt4Classification,
        native: NativeLimits, kernel: PackedInt4Kernel, prefill: Option<Int4PrefillLimits>,
        cancellation: CancellationToken) -> Result<HostedOutput<Int4ClassificationRun>, HostedError> {
        dispatch::preflight(self, native.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        grouped_int4::check_model(prepared.execution_identity(), model.model_identity()).map_err(classification_error)?;
        if prepared.group_width() != model.model_identity().group_width() { return Err(HostedError::ModelIdentity); }
        let (required, scratch) = requirements(prepared.required_context(), prepared.task_budget(), native, prepared.group_width())?;
        let scratch = super::prefill::with_extra_workspace(scratch, prefill)?;
        let work = prepared.planned_work();
        let task = prepared.task_budget();
        let lease = self.resources().acquire_lease();
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch)?;
        let output = output_claim(&lease, task.max_output_bytes, u64::from(task.max_output_tokens))?;
        let model = model.clone();
        dispatch::run(self, native.run, cancellation, move |control| {
            let mut engine = allocate_native(kv, workspace, || {
                StrictInt4Engine::new(&model.inner.loaded.value, native.context_tokens, Int4MemoryBudget {
                    max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                    max_scratch_payload_bytes: required.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            // Commit failure drops the actual result before releasing its
            // output reservation. All earlier head results are held privately.
            let result = allocate(output, || {
                let budget = Int4ScoringBudget { native: Int4RunBudget::exact(work), max_kv_bytes: required.kv_bytes };
                let result = match prefill {
                    None => prepared.execute_with_control(prepared.execution_identity(), &mut engine.value, budget, control),
                    Some(limits) => prepared.execute_layer_major_with_control(prepared.execution_identity(),
                        &mut engine.value, budget, limits, control),
                }.map_err(classification_error)?;
                if result.model_work != work || result.head_count != prepared.head_count()
                    || result.group_width != prepared.group_width()
                    || engine.value.is_poisoned() || !engine.value.kv_cache().all_slots_have_len(0) {
                    return Err(classification_error(Int4ClassificationError::Accounting));
                }
                Ok(result)
            })?;
            drop(engine);
            drop(lease);
            Ok(GuardedOutput::new(result.value, result._memory))
        })
    }
}
fn classification_error(error: Int4ClassificationError) -> HostedError {
    HostedError::Int4(Int4HostError::Classification(error))
}
fn requirements(context: usize, task: TaskBudget, native: NativeLimits, width: usize)
    -> Result<(Int4MemoryRequirement, u64), HostedError> {
    native.run.validate()?;
    task.validate().map_err(|_| HostedError::Limits("INT4 classification task budget"))?;
    if context == 0 || context > native.context_tokens || native.allocator_reserve_bytes == 0 {
        return Err(HostedError::Limits("INT4 classification context or allocator reserve"));
    }
    // Sequential heads share capacity, not compute allowances. Charge the
    // actual full allocation, not merely the longest prompt's live prefix.
    let required = Int4MemoryRequirement::for_context(native.context_tokens, width).map_err(native_error)?;
    if required.kv_bytes > task.max_kv_bytes { return Err(HostedError::Limits("INT4 classification complete KV allocation")); }
    let scratch = sum(&[required.rope_bytes, required.scratch_payload_bound, native.allocator_reserve_bytes])?;
    Ok((required, scratch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use crate::tasks::classify::grouped_int4::Int4ClassificationPlanner;
    fn native() -> NativeLimits {
        NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65536,
            run: RunLimits { max_elapsed: Duration::from_secs(1), max_checkpoints: 1000, cleanup_reserve_bytes: 65536 } }
    }
    fn task() -> TaskBudget {
        TaskBudget { max_input_tokens: 128, max_output_tokens: 8, max_output_bytes: 65536,
            max_grammar_states: 128, max_kv_bytes: Int4MemoryRequirement::for_context(128, 32).unwrap().kv_bytes }
    }
    #[test]
    fn all_allocated_native_storage_is_priced_without_a_sampler() {
        for width in [1, 3, 16, 32, 128, 10752] {
            let n = native(); let (required, scratch) = requirements(100, task(), n, width).unwrap();
            assert_eq!(required, Int4MemoryRequirement::for_context(n.context_tokens, width).unwrap());
            assert_eq!(scratch, required.rope_bytes + required.scratch_payload_bound + n.allocator_reserve_bytes);
        }
    }
    #[test]
    fn full_context_not_only_live_prefix_must_fit_the_task_budget() {
        let mut under = task(); under.max_kv_bytes -= 1;
        assert!(requirements(16, under, native(), 32).is_err());
        assert!(requirements(0, task(), native(), 32).is_err());
        assert!(requirements(129, task(), native(), 32).is_err());
        let mut n = native(); n.allocator_reserve_bytes = 0;
        assert!(requirements(16, task(), n, 32).is_err());
    }
    #[test]
    fn partition_and_overflow_fail_before_any_model_execution() {
        for width in [0, 10753, usize::MAX] { assert!(requirements(16, task(), native(), width).is_err()); }
        let mut n = native(); n.allocator_reserve_bytes = u64::MAX;
        assert!(requirements(16, task(), n, 32).is_err());
    }
    #[test]
    fn independent_heads_reuse_capacity_without_combining_causal_contexts() {
        use crate::native_engine::strict_int4::Int4Work;
        let head = Int4Work::for_sequence(32, 0, 100, 3 * 166144).unwrap();
        let complete = head.checked_add(head).unwrap();
        assert!(complete.forward_positions > native().context_tokens as u64);
        requirements(100, task(), native(), 32).unwrap();
        assert_eq!(complete.attention_pairs, 2 * head.attention_pairs);
    }
    #[test]
    fn errors_keep_typed_causes_without_private_debug_strings() {
        let cause = Int4ClassificationError::Native(StrictInt4Error::Cancelled(
            crate::native_engine::decode::DecodeCancellationKind::Deadline));
        let error = Int4HostError::Classification(cause);
        assert!(error.source().is_some()); assert_eq!(error.to_string(), format!("{error:?}"));
        assert_eq!(error.to_string(), "INT4 classification execution or finalization refused");
        let hosted = classification_error(Int4ClassificationError::Identity);
        assert!(hosted.source().is_some());
    }
    #[test]
    fn plans_and_results_cross_the_existing_blocking_boundary() {
        fn send<T: Send + 'static>() {}
        send::<PreparedInt4Classification>(); send::<Int4ClassificationRun>();
        send::<Int4ClassificationPlanner>();
        let _grouped = NlpEngine::execute_int4_classify_layer_major;
    }
    #[test]
    fn grouped_prompts_add_one_workspace_without_changing_full_kv_capacity() {
        let (required, base) = requirements(100, task(), native(), 32).unwrap();
        for rows in [1, 4, 64] {
            let extra = Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
            let limits = Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: extra };
            assert_eq!(super::super::prefill::with_extra_workspace(base, Some(limits)).unwrap(), base + extra);
            assert_eq!(required.kv_bytes, task().max_kv_bytes);
        }
    }
}
