//! One owned runtime invocation for a pre-admitted slot-refilling text epoch.
use super::*;
use crate::native_engine::kv::KV_BYTES_PER_TOKEN;

impl NlpEngine {
    /// Keep at most active_sequences native KV/sampler slots while refilling
    /// from up to 64 immutable prepared requests. A slot may hold ANY queued
    /// request, so each reserves the longest planned context. This deliberate
    /// capacity floor may refuse heterogeneous tasks with smaller KV ceilings.
    ///
    /// All plans and all completed results remain admitted for the whole epoch.
    /// One run/deadline/checkpoint budget spans every refill. No runtime, model,
    /// worker team or allowance is recreated per request. Results are returned
    /// in INPUT order only after every task validates and native state drains.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_chat_refilling(&self, model: &ResidentInt8, prepared: Vec<PreparedInt8Chat>,
        first_request_seq: u64, limits: ChatCohortLimits, active_sequences: usize,
        prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ChatCohortResult>, HostedError> {
        dispatch::preflight(self, limits.native.run)?; self.check_resident_domain(model)?;
        validate_limits(limits, prepared.len(), first_request_seq)?;
        validate_slots(active_sequences, prepared.len())?;
        let extra = packed_scratch(Some(prefill))?;
        for plan in &prepared { check_model_identity(model.artifact_identity(), plan.execution_identity())?; }
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, limits.preparation_reserve_bytes)?,
            || build_input(prepared, limits, active_sequences))?;
        let required = Int8MemoryRequirement::for_cohort(&input.value.contexts).map_err(HostedError::Native)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            workspace_bytes(required, input.value.sampler_bytes, limits.native.allocator_reserve_bytes, extra)?)?;
        let output = output_claim(&lease, limits.max_result_bytes, input.value.output_tokens)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            let input = input;
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .cohort_engine(&input.value.contexts, memory_budget(required)).map_err(HostedError::Model))?;
            let mut requests = Vec::new(); requests.try_reserve_exact(input.value.prepared.len())
                .map_err(|_| HostedError::Native(StrictInt8Error::Allocation))?;
            let slot_kv = input.value.contexts[0] as u64 * KV_BYTES_PER_TOKEN as u64;
            for (index, plan) in input.value.prepared.iter().enumerate() {
                requests.push(Int8ChatCohortRequest { prepared: plan, admitted_identity: plan.execution_identity(),
                    request_seq: first_request_seq + index as u64,
                    budget: Int8GenerationBudget { native: Int8RunBudget::exact(plan.planned_work()),
                        max_kv_bytes: slot_kv, max_sampler_bytes: plan.native_plan().sampler_bytes() } });
            }
            let budget = Int8ChatCohortBudget { generation: Int8CohortBudget {
                native: Int8RunBudget::exact(input.value.work), max_kv_bytes: required.kv_bytes,
                max_sampler_bytes: input.value.sampler_bytes }, max_result_bytes: limits.max_result_bytes };
            let result = task::packed::refill::execute(&mut engine.value, &requests, budget, prefill, control)
                .map_err(HostedError::Chat)?;
            drop(requests); drop(engine); drop(input);
            let committed = output.commit()?; drop(lease);
            Ok(GuardedOutput::new(result, committed))
        })
    }
}
fn validate_slots(active: usize, requests: usize) -> Result<(), HostedError> {
    if active == 0 || active > requests || requests > MAX_BATCH_ROWS {
        return Err(HostedError::Limits("refilling epoch live slots"));
    }
    Ok(())
}
fn build_input(prepared: Vec<PreparedInt8Chat>, limits: ChatCohortLimits, active: usize) -> Result<Input, HostedError> {
    validate_slots(active, prepared.len())?;
    let mut longest = 0_usize; let mut largest_sampler = 0_u64;
    let mut work = Int8Work::default(); let mut output_tokens = 0_u64; let mut output_bytes = 4096_u64;
    for plan in &prepared {
        let row = plan.planned_work();
        let context = usize::try_from(row.forward_positions).map_err(|_| HostedError::Limits("refilling epoch context"))?;
        if context == 0 || context > limits.native.context_tokens { return Err(HostedError::Limits("refilling epoch context")); }
        longest = longest.max(context); largest_sampler = largest_sampler.max(plan.native_plan().sampler_bytes());
        work = work.checked_add(row).map_err(HostedError::Native)?;
        output_tokens = sum(&[output_tokens, u64::from(plan.task_plan().ir().budget().max_output_tokens)])?;
        output_bytes = sum(&[output_bytes, plan.task_plan().ir().budget().max_output_bytes])?;
    }
    let slot = Int8MemoryRequirement::for_context(longest).map_err(HostedError::Native)?;
    if prepared.iter().any(|plan| slot.kv_bytes > plan.task_plan().ir().budget().max_kv_bytes) {
        return Err(HostedError::Limits("refill slot capacity exceeds a queued task KV ceiling"));
    }
    let sampler_bytes = largest_sampler.checked_mul(active as u64).ok_or(HostedError::Limits("refill sampler arithmetic"))?;
    if sampler_bytes > limits.max_sampler_bytes || output_bytes > limits.max_result_bytes {
        return Err(HostedError::Limits("refill live samplers or whole-epoch retained results"));
    }
    let mut contexts = Vec::new(); contexts.try_reserve_exact(active)
        .map_err(|_| HostedError::Native(StrictInt8Error::Allocation))?;
    contexts.resize(active, longest);
    Ok(Input { prepared, contexts, work, sampler_bytes, output_tokens })
}
#[cfg(test)] mod tests;
