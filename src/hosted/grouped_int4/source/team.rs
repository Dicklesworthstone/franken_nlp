//! Original source-task drivers/finalizers with fixed native row ownership.
use super::*;

impl NlpEngine {
    /// Execute NER, keyphrases, cited summaries or passage QA on one persistent
    /// output-row team. Every legal selected row remains eligible. Source
    /// coordinates, citations, mask caps and one-request identity are unchanged.
    /// Packed prefill and independent-sequence cohorts use their separate APIs.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_source_task_with_threads(&self, model: &ResidentInt4, prepared: PreparedInt4SourceTask,
        vocabulary: Arc<ExtractionVocabulary>, native: NativeLimits,
        mask_limits: MaskWorkLimits, max_mask_node_visits: u64, kernel: PackedInt4Kernel,
        compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt4TeamResult<Int4SourceTaskRun>, HostedError> {
        dispatch::preflight_int4_team(self, native.run, compute_threads)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        constrained_int4::check_model(prepared.execution_identity(), model.model_identity())
            .map_err(Int4SourceError::from).map_err(task_error)?;
        if prepared.group_width() != model.model_identity().group_width() { return Err(HostedError::ModelIdentity); }
        let work = prepared.planned_work();
        let (required, scratch_bytes) = extract::scheduled_requirements(work, prepared.group_width(),
            prepared.max_kv_bytes(), native, mask_limits, max_mask_node_visits, None)?;
        let memory = parallel::TaskMemory { required, scratch_bytes,
            output_bytes: prepared.max_result_bytes(), output_tokens: prepared.task_budget().max_output_tokens as u64,
            preparation_bytes: None };
        parallel::execute_task(self, model, prepared, native, kernel, compute_threads, memory,
            cancellation, move |prepared, engine, control| {
                let budget = Int4JsonBudget { native: Int4RunBudget::exact(work), json: JsonWorkBudget {
                    max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
                    max_kv_bytes: required.kv_bytes, max_total_mask_node_visits: max_mask_node_visits, mask_limits,
                } };
                let run = prepared.execute_with_control(prepared.execution_identity(), engine,
                    &vocabulary, budget, control).map_err(task_error)?;
                if run.group_width != prepared.group_width() || engine.is_poisoned()
                    || !engine.kv_cache().all_slots_have_len(0) {
                    return Err(task_error(Int4SourceError::InvalidResult));
                }
                Ok(run)
            })
    }
}
