//! One fixed numerical row team through source decoding and finalization.
use super::*;
use crate::native_engine::simd::Int8Kernel;
use crate::tasks::source_planning::quantized::Int8SourceError;

/// Own preparation and every reservation outside the scoped callback. Its
/// early error or unwind drops native storage before these charges, and the
/// dispatcher keeps this aggregate until all persistent workers have joined.
struct SourceTeamAdmission {
    input: Charged<SourceInput>,
    kv: Option<Pending>, scratch: Option<Pending>, output: Option<Pending>,
    _kv: Option<CommittedMemory>, _scratch: Option<CommittedMemory>, _lease: EngineLease,
}
impl SourceTeamAdmission {
    fn commit_native(&mut self) -> Result<(), HostedError> {
        self._kv = Some(self.kv.take().ok_or(HostedError::Limits("INT8 source KV admission consumed"))?.commit()?);
        self._scratch = Some(self.scratch.take().ok_or(HostedError::Limits("INT8 source scratch admission consumed"))?.commit()?);
        Ok(())
    }
    fn finish(&mut self, result: Int8SourceTaskRun, compute: Int8ComputeReceipt)
        -> Result<HostedInt8TeamResult<Int8SourceTaskRun>, HostedError> {
        let output = self.output.take().ok_or(HostedError::Limits("INT8 source output admission consumed"))?;
        match output.commit() {
            Ok(memory) => Ok(HostedInt8TeamResult::from_guarded(GuardedOutput::new(result, memory), compute)),
            Err(error) => { drop(result); Err(error) }
        }
    }
}

struct TeamRequirements { native: Int8MemoryRequirement, scratch_bytes: u64, max_batch_rows: usize }
fn team_requirements(limits: SourceLimits, task: TaskBudget, work: Int8Work, width: usize,
    prefill: Option<Int8PrefillLimits>)
    -> Result<TeamRequirements, HostedError> {
    let native = requirements(limits.native)?;
    validate(limits, task, work, native.kv_bytes)?;
    let scratch_bytes = source_scratch(sum(&[native.rope_bytes, native.scratch_payload_bound,
        limits.native.allocator_reserve_bytes])?, prefill)?;
    let max_batch_rows = prefill.map_or(1, |limits| limits.max_batch_rows);
    // Children have independent fixed payloads, not a discount against the
    // existing coordinator scratch or caller/host preparation reservations.
    // Their batch capacity is fixed before any source-task work starts.
    sum(&[scratch_bytes, dispatch::int8_prefill_team_workspace_bytes(width, max_batch_rows)?])?;
    Ok(TeamRequirements { native, scratch_bytes, max_batch_rows })
}
fn native_budget(limits: SourceLimits, work: Int8Work, kv_bytes: u64) -> Int8JsonBudget {
    Int8JsonBudget { native: Int8RunBudget::exact(work), json: JsonWorkBudget {
        max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
        max_kv_bytes: kv_bytes, max_total_mask_node_visits: limits.max_mask_node_visits,
        mask_limits: limits.mask_limits,
    } }
}

impl NlpEngine {
    /// Execute NER, keyphrases, cited summarization or passage QA with one
    /// persistent output-row team. Width includes the coordinator and must be
    /// admitted by the process host before execution. Every original source
    /// byte, citation, sealed head policy and complete task finalizer remains
    /// governed by the prepared request; worker width is physical metadata.
    ///
    /// One team serves the complete sequential prompt and constrained decode.
    /// Every legal selected head row stays eligible. The explicit layer-major
    /// variant admits prompt morsels before forming the same fixed team.
    /// Input, KV, scratch and output authority survive physical join, including
    /// cancellation and panic. The returned guard owns result memory through
    /// serialization and delivery; no partial source result is published.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_source_with_threads(&self, model: &ResidentInt8, prepared: PreparedInt8SourceTask,
        vocabulary: Arc<ExtractionVocabulary>, limits: SourceLimits, kernel: Int8Kernel,
        compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SourceTaskRun>, HostedError> {
        self.execute_int8_source_team(model, prepared, vocabulary, limits, kernel,
            compute_threads, None, cancellation)
    }

    /// Use one persistent output-row team for bounded layer-major prompt
    /// morsels and constrained source decoding. Coordinator prefill scratch
    /// and fixed worker row capacity are both admitted before execution; the
    /// same team, native work ledger and cancellation allowance span the
    /// complete task. This scheduling choice preserves the prepared input,
    /// every legal selected row, source finalization and guarded delivery.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_source_layer_major_with_threads(&self, model: &ResidentInt8,
        prepared: PreparedInt8SourceTask, vocabulary: Arc<ExtractionVocabulary>,
        limits: SourceLimits, kernel: Int8Kernel, compute_threads: usize,
        prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SourceTaskRun>, HostedError> {
        self.execute_int8_source_team(model, prepared, vocabulary, limits, kernel,
            compute_threads, Some(prefill), cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_int8_source_team(&self, model: &ResidentInt8, prepared: PreparedInt8SourceTask,
        vocabulary: Arc<ExtractionVocabulary>, limits: SourceLimits, kernel: Int8Kernel,
        compute_threads: usize, prefill: Option<Int8PrefillLimits>, cancellation: CancellationToken)
        -> Result<HostedInt8TeamResult<Int8SourceTaskRun>, HostedError> {
        dispatch::preflight_int8_team(self, limits.native.run, compute_threads)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        let work = prepared.planned_work();
        let required = team_requirements(limits, prepared.task_budget(), work, compute_threads, prefill)?;
        let lease = self.resources().acquire_lease();
        let output = output_claim(&lease, prepared.max_result_bytes(),
            u64::from(prepared.task_budget().max_output_tokens))?;
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers,
            limits.preparation_reserve_bytes)?, || Ok(SourceInput { prepared, vocabulary }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.scratch_bytes)?;
        let admission = SourceTeamAdmission { input, kv: Some(kv), scratch: Some(scratch), output: Some(output),
            _kv: None, _scratch: None, _lease: lease };
        dispatch::run_int8_prefill_team(self, limits.native.run, cancellation, model.clone(), compute_threads,
            kernel, required.max_batch_rows, admission, move |model, admission, control, team| {
                let mut engine = model.engine(limits.native.context_tokens, memory_budget(required.native))
                    .map_err(HostedError::Model)?.with_kernel(kernel)
                    .with_projection_executor(team).map_err(HostedError::Native)?;
                admission.commit_native()?;
                let input = &admission.input.value;
                let budget = native_budget(limits, work, required.native.kv_bytes);
                let result = match prefill {
                    Some(prefill) => input.prepared.execute_layer_major_with_control(
                        input.prepared.execution_identity(), &mut engine, &input.vocabulary,
                        budget, prefill, control),
                    None => input.prepared.execute_with_control(input.prepared.execution_identity(),
                        &mut engine, &input.vocabulary, budget, control),
                }.map_err(HostedError::Source)?;
                if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                    return Err(HostedError::Source(Int8SourceError::InvalidResult));
                }
                drop(engine);
                let compute = team.stats().into();
                admission.finish(result, compute)
            })
    }
}

#[cfg(test)]
mod tests;
