//! One physically supervised blocking call for an admitted INT4 chat cohort.
//! The resident model and all transient charges remain owned until full drain.
use super::*;
use crate::{
    native_engine::{generation::grouped_int4::batch::Int4BatchGenerationRequirements,
        strict_int4::cohort::{Int4CohortEngine, MAX_INT4_SEQUENCES}},
    tasks::chat::grouped_int4::batch::{self as task, Int4ChatBatchRequest, Int4ChatBatchLimits, Int4ChatBatchResult},
};

/// Caller-owned preparation is moved into the physical invocation. Context
/// capacity is explicit per row; no padded longest-context charge is assumed.
pub struct Int4ChatBatchMember {
    pub prepared: PreparedInt4Chat,
    pub request_seq: u64,
    pub context_tokens: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct Int4BatchHostLimits {
    pub task: Int4ChatBatchLimits,
    pub allocator_reserve_bytes: u64,
    pub run: RunLimits,
}
#[derive(Clone, Copy, Debug)]
pub struct Int4BatchHostRequirements {
    pub generation: Int4BatchGenerationRequirements,
    pub workspace_bytes: u64,
    pub max_output_tokens: u64,
}
impl Int4BatchHostLimits {
    /// Pure quotation; no model access or reservations. Model/domain checks
    /// still precede allocation in execute_int4_chat_batch.
    pub fn requirements(self, members: &[Int4ChatBatchMember]) -> Result<Int4BatchHostRequirements, HostedError> {
        self.run.validate()?;
        if self.allocator_reserve_bytes == 0 { return Err(HostedError::Limits("allocator reserve")); }
        let requests = bindings(members)?;
        let mut contexts = [0_usize; MAX_INT4_SEQUENCES];
        let mut tokens = 0_u64;
        for (index, member) in members.iter().enumerate() {
            contexts[index] = member.context_tokens;
            tokens = sum(&[tokens, u64::from(member.prepared.task_plan().ir().budget().max_output_tokens)])?;
        }
        let generation = task::requirements(&requests, &contexts[..members.len()], self.task).map_err(chat_error)?;
        let workspace_bytes = workspace_bytes(generation, self.allocator_reserve_bytes)?;
        Ok(Int4BatchHostRequirements { generation, workspace_bytes, max_output_tokens: tokens })
    }
}

impl NlpEngine {
    /// Execute independently prepared text/chat rows on ONE shared model and
    /// native cohort, rather than a loop of separately allocated engines.
    /// Returns no successful prefix: all task finalizers must pass. Streaming
    /// remains available at the lower task layer under its provisional contract.
    pub fn execute_int4_chat_batch(&self, model: &ResidentInt4, members: Vec<Int4ChatBatchMember>,
        limits: Int4BatchHostLimits, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ChatBatchResult>, HostedError> {
        dispatch::preflight(self, limits.run)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let required = limits.requirements(&members)?;
        for member in &members {
            let rebound = bind_candidate_model(member.prepared.execution_identity().clone(), model.model_identity())
                .map_err(|_| HostedError::ModelIdentity)?;
            if &rebound != member.prepared.execution_identity()
                || member.prepared.native_plan().group_width() != model.model_identity().group_width() {
                return Err(HostedError::ModelIdentity);
            }
        }
        let lease = self.resources().acquire_lease();
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.generation.native.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let output = output_claim(&lease, limits.task.max_result_bytes, required.max_output_tokens)?;
        let model = model.clone();
        dispatch::run(self, limits.run, cancellation, move |control| {
            checkpoint(control)?;
            let mut contexts = [0_usize; MAX_INT4_SEQUENCES];
            for (index, member) in members.iter().enumerate() { contexts[index] = member.context_tokens; }
            let need = required.generation.native;
            let mut engine = allocate_native(kv, workspace, || {
                Int4CohortEngine::new(&model.inner.loaded.value, &contexts[..members.len()], Int4MemoryBudget {
                    max_kv_bytes: need.kv_bytes, max_rope_bytes: need.rope_bytes,
                    max_scratch_payload_bytes: need.scratch_payload_bound,
                }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
            })?;
            let requests = bindings(&members)?;
            let result = task::execute(&mut engine.value, &requests, limits.task, control).map_err(chat_error)?;
            // Drop the entire cohort before publishing its aggregate output.
            // The closure still owns the prepared prompts, model and lease.
            drop(engine);
            let committed = output.commit()?;
            drop(lease);
            Ok(GuardedOutput::new(result, committed))
        })
    }
}

fn bindings(members: &[Int4ChatBatchMember]) -> Result<Vec<Int4ChatBatchRequest<'_>>, HostedError> {
    if members.is_empty() || members.len() > MAX_INT4_SEQUENCES { return Err(HostedError::Limits("INT4 cohort width")); }
    let mut requests = Vec::new();
    requests.try_reserve_exact(members.len()).map_err(|_| HostedError::Limits("INT4 cohort allocation"))?;
    for member in members {
        let prepared = &member.prepared;
        requests.push(Int4ChatBatchRequest { prepared, admitted_identity: prepared.execution_identity(),
            request_seq: member.request_seq, budget: Int4GenerationBudget {
                native: Int4RunBudget::exact(prepared.planned_work()),
                max_kv_bytes: prepared.task_plan().ir().budget().max_kv_bytes,
                max_sampler_bytes: prepared.native_plan().sampler_bytes(),
            } });
    }
    Ok(requests)
}
fn workspace_bytes(need: Int4BatchGenerationRequirements, allocator: u64) -> Result<u64, HostedError> {
    // Raw generation payload can coexist with finalized chat results; it has
    // its own charge, in addition to output_claim's serialization allowance.
    sum(&[need.native.rope_bytes, need.native.scratch_payload_bound,
        need.sampler_payload_bytes, need.output_payload_upper_bytes, allocator])
}

#[cfg(test)] mod tests;
