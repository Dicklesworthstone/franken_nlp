//! Complete exclusive/multi-label scoring on one persistent projection team.
use super::*;

impl NlpEngine {
    /// Keep one fixed output-row team across every classification head. Each
    /// head still has its own causal session and full-vocabulary denominator;
    /// the exact complete-task work allowance is never copied per worker/head.
    pub fn execute_int4_classify_with_threads(&self, model: &ResidentInt4, prepared: PreparedInt4Classification,
        native: NativeLimits, kernel: PackedInt4Kernel, compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt4TeamResult<Int4ClassificationRun>, HostedError> {
        dispatch::preflight_int4_team(self, native.run, compute_threads)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        grouped_int4::check_model(prepared.execution_identity(), model.model_identity()).map_err(classification_error)?;
        if prepared.group_width() != model.model_identity().group_width() { return Err(HostedError::ModelIdentity); }
        let (required, scratch_bytes) = requirements(prepared.required_context(), prepared.task_budget(),
            native, prepared.group_width())?;
        let work = prepared.planned_work();
        let task = prepared.task_budget();
        let memory = parallel::TaskMemory { required, scratch_bytes,
            output_bytes: task.max_output_bytes, output_tokens: u64::from(task.max_output_tokens),
            preparation_bytes: None };
        parallel::execute_task(self, model, prepared, native, kernel, compute_threads, memory,
            cancellation, move |prepared, engine, control| {
                let budget = Int4ScoringBudget { native: Int4RunBudget::exact(work), max_kv_bytes: required.kv_bytes };
                let result = prepared.execute_with_control(prepared.execution_identity(), engine, budget, control)
                    .map_err(classification_error)?;
                if result.model_work != work || result.head_count != prepared.head_count()
                    || result.group_width != prepared.group_width()
                    || engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                    return Err(classification_error(Int4ClassificationError::Accounting));
                }
                Ok(result)
            })
    }
}
