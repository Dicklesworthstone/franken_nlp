//! Owned resource admission and physical drain for cross-document INT8 chat.
use super::*;
use crate::{native_engine::{portable_int8::batch::MAX_BATCH_ROWS, strict_int8::Int8Work,
    generation::quantized::cohort::Int8CohortBudget},
    tasks::chat::quantized::cohort::{self as task, Int8ChatCohortRequest, Int8ChatCohortBudget, Int8ChatCohortResult}};

#[derive(Clone, Copy, Debug)]
pub struct ChatCohortLimits {
    /// context_tokens is a PER-ROW ceiling. Actual row capacities are derived
    /// from each immutable plan, rather than padding all KV to this maximum.
    pub native: NativeLimits,
    /// Aggregate simultaneous sampler workspaces, not one row's allowance.
    pub max_sampler_bytes: u64,
    /// Entire transferred plans/tokenizer graphs and cohort bookkeeping.
    pub preparation_reserve_bytes: u64,
    /// Complete retained canonical cohort output, including metadata.
    pub max_result_bytes: u64,
}
struct Input {
    prepared: Vec<PreparedInt8Chat>, contexts: Vec<usize>, work: Int8Work,
    sampler_bytes: u64, output_tokens: u64,
}
impl NlpEngine {
    /// Execute a bounded set of prepared chat/generate requests together on the
    /// existing single blocking coordinator. All rows use one resident binding,
    /// one native scope and one deadline/checkpoint budget. No extra worker team,
    /// hidden retry, catalog activation, or numerical/performance award.
    ///
    /// Returns input-order results only after EVERY task validates and the
    /// physical invocation drains. The real output reservation remains owned
    /// by the returned guard through serialization, delivery and flush.
    pub fn execute_int8_chat_cohort(&self, model: &ResidentInt8, prepared: Vec<PreparedInt8Chat>,
        first_request_seq: u64, limits: ChatCohortLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ChatCohortResult>, HostedError> {
        dispatch::preflight(self, limits.native.run)?;
        self.check_resident_domain(model)?;
        validate_limits(limits, prepared.len(), first_request_seq)?;
        for plan in &prepared { check_model_identity(model.artifact_identity(), plan.execution_identity())?; }
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, limits.preparation_reserve_bytes)?,
            || build_input(prepared, limits))?;
        let required = Int8MemoryRequirement::for_cohort(&input.value.contexts).map_err(HostedError::Native)?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, input.value.sampler_bytes,
                limits.native.allocator_reserve_bytes])?)?;
        let output = output_claim(&lease, limits.max_result_bytes, input.value.output_tokens)?;
        let model = model.clone();
        dispatch::run(self, limits.native.run, cancellation, move |control| {
            // Capture the aggregate, not uncharged disjoint fields. Input and
            // native storage both drain before their owning memory guards.
            let input = input;
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .cohort_engine(&input.value.contexts, memory_budget(required)).map_err(HostedError::Model))?;
            let mut requests = Vec::new(); requests.try_reserve_exact(input.value.prepared.len())
                .map_err(|_| HostedError::Native(StrictInt8Error::Allocation))?;
            for (index, plan) in input.value.prepared.iter().enumerate() {
                requests.push(Int8ChatCohortRequest { prepared: plan, admitted_identity: plan.execution_identity(),
                    request_seq: first_request_seq + index as u64,
                    budget: Int8GenerationBudget { native: Int8RunBudget::exact(plan.planned_work()),
                        max_kv_bytes: input.value.contexts[index] as u64 * crate::native_engine::kv::KV_BYTES_PER_TOKEN as u64,
                        max_sampler_bytes: plan.native_plan().sampler_bytes() } });
            }
            let budget = Int8ChatCohortBudget { generation: Int8CohortBudget {
                native: Int8RunBudget::exact(input.value.work), max_kv_bytes: required.kv_bytes,
                max_sampler_bytes: input.value.sampler_bytes }, max_result_bytes: limits.max_result_bytes };
            let result = task::execute(&mut engine.value, &requests, budget, control).map_err(HostedError::Chat)?;
            drop(requests); drop(engine); drop(input);
            let committed = output.commit()?; drop(lease);
            Ok(GuardedOutput::new(result, committed))
        })
    }
}
fn validate_limits(limits: ChatCohortLimits, count: usize, first: u64) -> Result<(), HostedError> {
    limits.native.run.validate()?;
    if count == 0 || count > MAX_BATCH_ROWS || first == 0 || first.checked_add((count - 1) as u64).is_none()
        || limits.native.allocator_reserve_bytes == 0 || limits.max_sampler_bytes == 0
        || limits.preparation_reserve_bytes == 0 || limits.max_result_bytes == 0 {
        return Err(HostedError::Limits("cohort count, delivery range or memory admission"));
    }
    Int8MemoryRequirement::for_context(limits.native.context_tokens).map_err(HostedError::Native)?; Ok(())
}
fn build_input(prepared: Vec<PreparedInt8Chat>, limits: ChatCohortLimits) -> Result<Input, HostedError> {
    let mut contexts = Vec::new(); contexts.try_reserve_exact(prepared.len())
        .map_err(|_| HostedError::Native(StrictInt8Error::Allocation))?;
    let mut work = Int8Work::default(); let mut sampler_bytes = 0_u64; let mut output_tokens = 0_u64;
    let mut output_bytes = 4096_u64; // bounded wrapper/array punctuation and work metadata
    for plan in &prepared {
        let row = plan.planned_work();
        let context = usize::try_from(row.forward_positions).map_err(|_| HostedError::Limits("cohort context arithmetic"))?;
        if context == 0 || context > limits.native.context_tokens { return Err(HostedError::Limits("cohort row context")); }
        let required = Int8MemoryRequirement::for_context(context).map_err(HostedError::Native)?;
        if required.kv_bytes > plan.task_plan().ir().budget().max_kv_bytes { return Err(HostedError::Limits("cohort task KV")); }
        contexts.push(context); work = work.checked_add(row).map_err(HostedError::Native)?;
        sampler_bytes = sum(&[sampler_bytes, plan.native_plan().sampler_bytes()])?;
        output_tokens = sum(&[output_tokens, u64::from(plan.task_plan().ir().budget().max_output_tokens)])?;
        output_bytes = sum(&[output_bytes, plan.task_plan().ir().budget().max_output_bytes])?;
    }
    // Reserve enough for all rows to exist simultaneously, not just one frame
    // or an optimistic early-EOS estimate. Refuse before native allocations.
    if sampler_bytes > limits.max_sampler_bytes || output_bytes > limits.max_result_bytes {
        return Err(HostedError::Limits("aggregate cohort sampler or retained result"));
    }
    Ok(Input { prepared, contexts, work, sampler_bytes, output_tokens })
}
#[cfg(test)] mod tests;
