//! All independent dimensions on one admitted fixed-width native team.
use super::*;

impl NlpEngine {
    /// Reuse one persistent output-row team for all independent axes, with
    /// unchanged candidate distributions, captured-mass audits and abstention.
    /// Preparation, complete result and one full native capacity are charged;
    /// all five work axes remain bounded across the complete bundle.
    pub fn execute_int4_sentiment_with_threads(&self, model: &ResidentInt4, prepared: PreparedInt4Sentiment,
        limits: Int4SentimentHostLimits, kernel: PackedInt4Kernel, compute_threads: usize,
        cancellation: CancellationToken) -> Result<HostedInt4TeamResult<Int4SentimentRun>, HostedError> {
        dispatch::preflight_int4_team(self, limits.native.run, compute_threads)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        grouped_int4::check_model(prepared.execution_identity(), model.model_identity()).map_err(sentiment_error)?;
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
                    .map_err(sentiment_error)?;
                if result.model_work != work || result.head_count != prepared.head_count()
                    || result.group_width != prepared.group_width()
                    || engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                    return Err(sentiment_error(Int4SentimentError::Accounting));
                }
                Ok(result)
            })
    }
}
