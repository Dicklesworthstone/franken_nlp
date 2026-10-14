//! One charged execution path for sequential and explicit INT4 prompt prefill.
//! The scheduler choice cannot bypass identity/domain checks, task limits,
//! physical-completion ownership or final output guards.
use super::*;
use crate::native_engine::strict_int4::prefill::Int4PrefillLimits;

impl NlpEngine {
    /// Opt in to bounded layer-major prompt execution. Reserves the exact
    /// additional workspace in this resource domain before native allocation.
    /// Decode remains sequential; neither model identity nor sampling changes.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_chat_layer_major(&self, model: &ResidentInt4, prepared: PreparedInt4Chat,
        request_seq: u64, native: NativeLimits, max_sampler_bytes: u64, kernel: PackedInt4Kernel,
        prefill: Int4PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int4ChatResult>, HostedError> {
        execute(self, model, prepared, request_seq, native, max_sampler_bytes, kernel, Some(prefill), cancellation)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute(host: &NlpEngine, model: &ResidentInt4, prepared: PreparedInt4Chat,
    request_seq: u64, native: NativeLimits, max_sampler_bytes: u64, kernel: PackedInt4Kernel,
    prefill: Option<Int4PrefillLimits>, cancellation: CancellationToken)
    -> Result<HostedOutput<Int4ChatResult>, HostedError> {
    dispatch::preflight(host, native.run)?;
    if !Arc::ptr_eq(host.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
    let width = model.model_identity().group_width();
    let rebound = bind_candidate_model(prepared.execution_identity().clone(), model.model_identity())
        .map_err(|_| HostedError::ModelIdentity)?;
    if prepared.native_plan().group_width() != width || &rebound != prepared.execution_identity() {
        return Err(HostedError::ModelIdentity);
    }
    let required = native_requirements(&prepared, request_seq, native, max_sampler_bytes)?;
    let scratch = workspace_bytes(required, max_sampler_bytes, native.allocator_reserve_bytes, prefill)?;
    let work = prepared.planned_work();
    let lease = host.resources().acquire_lease();
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
    let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch)?;
    let output = output_claim(&lease, prepared.task_plan().ir().budget().max_output_bytes,
        u64::from(prepared.task_plan().ir().budget().max_output_tokens))?;
    let model = model.clone();
    dispatch::run(host, native.run, cancellation, move |control| {
        let mut engine = allocate_native(kv, workspace, || {
            StrictInt4Engine::new(&model.inner.loaded.value, native.context_tokens, Int4MemoryBudget {
                max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                max_scratch_payload_bytes: required.scratch_payload_bound,
            }).map(|engine| engine.with_kernel(kernel)).map_err(native_error)
        })?;
        let budget = Int4GenerationBudget { native: Int4RunBudget::exact(work),
            max_kv_bytes: required.kv_bytes, max_sampler_bytes };
        // The same preparation, native/result guards and control are used by
        // both routes. No nested runtime, separate cancellation budget or retry.
        let result = match prefill {
            None => prepared.execute(prepared.execution_identity(), &mut engine.value, request_seq, budget, control),
            Some(limits) => prepared.execute_layer_major(prepared.execution_identity(), &mut engine.value,
                request_seq, budget, limits, control),
        }.map_err(chat_error)?;
        drop(engine);
        let committed = output.commit()?;
        drop(lease);
        Ok(GuardedOutput::new(result, committed))
    })
}
fn workspace_bytes(required: Int4MemoryRequirement, sampler: u64, allocator: u64,
    prefill: Option<Int4PrefillLimits>) -> Result<u64, HostedError> {
    let extra = prefill.map(Int4PrefillLimits::validate).transpose().map_err(native_error)?.unwrap_or(0);
    sum(&[required.rope_bytes, required.scratch_payload_bound, sampler, allocator, extra])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_prices_extra_prompt_storage_without_changing_base_kv_or_group_scratch() {
        for group in [1, 3, 128, MAX_MODEL_K] { for context in [1, 128, 2048] {
            let need = Int4MemoryRequirement::for_context(context, group).unwrap();
            let sampler = 32 * 1024 * 1024; let allocator = 64 * 1024 * 1024;
            let sequential = workspace_bytes(need, sampler, allocator, None).unwrap();
            assert_eq!(sequential, need.rope_bytes + need.scratch_payload_bound + sampler + allocator);
            for rows in [1, 3, 4, 64] {
                let extra = Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
                let prefill = Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: extra };
                assert_eq!(workspace_bytes(need, sampler, allocator, Some(prefill)).unwrap(), sequential + extra);
            }
        } }
    }
    #[test]
    fn invalid_extra_workspace_and_aggregate_overflow_refuse_before_reservation() {
        let need = Int4MemoryRequirement::for_context(128, 128).unwrap();
        for rows in [0, 65, usize::MAX] {
            assert!(workspace_bytes(need, 1, 1, Some(Int4PrefillLimits {
                max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX })).is_err());
        }
        let exact = Int4PrefillLimits::required_extra_scratch_bytes(4).unwrap();
        assert!(workspace_bytes(need, 1, 1, Some(Int4PrefillLimits {
            max_batch_rows: 4, max_extra_scratch_bytes: exact - 1 })).is_err());
        assert!(workspace_bytes(need, u64::MAX, 1, None).is_err());
        assert!(workspace_bytes(need, 1, u64::MAX, None).is_err());
    }
}
