//! Complete rubric, order-debiased pairwise or faithfulness team execution.
use super::*;

impl NlpEngine {
    /// Keep one persistent native output-row team across every criterion,
    /// comparison order and evidence window. The sealed judgment and complete
    /// work budget are unchanged; failures publish no partial judgment.
    pub fn execute_int4_judge_with_threads(&self, model: &ResidentInt4, prepared: PreparedInt4Judge,
        limits: Int4JudgeHostLimits, kernel: PackedInt4Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt4TeamResult<Int4JudgeRun>, HostedError> {
        dispatch::preflight_int4_team(self, limits.native.run, compute_threads)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        judge_task::check_model(prepared.execution_identity(), model.model_identity()).map_err(judge_error)?;
        if prepared.group_width() != model.model_identity().group_width() { return Err(HostedError::ModelIdentity); }
        let work = prepared.planned_work();
        let (required, scratch_bytes) = requirements(limits, prepared.task_budget(),
            prepared.required_context(), prepared.group_width(), work)?;
        let memory = parallel::TaskMemory { required, scratch_bytes, output_bytes: prepared.max_result_bytes(),
            output_tokens: u64::from(prepared.task_budget().max_output_tokens),
            preparation_bytes: Some(limits.preparation_reserve_bytes) };
        parallel::execute_task(self, model, prepared, limits.native, kernel, compute_threads, memory,
            cancellation, move |prepared, engine, control| {
                let budget = Int4ScoringBudget { native: Int4RunBudget::exact(work), max_kv_bytes: required.kv_bytes };
                let result = prepared.execute_with_control(prepared.execution_identity(), engine, budget, control)
                    .map_err(judge_error)?;
                if result.schema_version != 1 || result.execution != INT4_JUDGE_EXECUTION
                    || result.numerics_profile != STRICT_INT4_PROFILE || result.model_work != work
                    || result.head_count != prepared.head_count() || result.group_width != prepared.group_width()
                    || engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                    return Err(judge_error(Int4JudgeError::Accounting));
                }
                Ok(result)
            })
    }
}
