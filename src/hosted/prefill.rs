//! Process-admitted opt-in layer-major generation/chat, not a second runtime.
use super::*;
use crate::native_engine::strict_int8::prefill::Int8PrefillLimits;

impl NlpEngine {
    /// Execute the same prepared generate/chat task with bounded layer-major
    /// prompt morsels. The ordinary execute_int8_chat remains sequential.
    /// Additional scratch has a REAL process-ledger reservation owned by the
    /// blocking closure through native drain. This is not an RSS/perf claim.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_chat_layer_major(&self, model: &ResidentInt8, prepared: PreparedInt8Chat,
        request_seq: u64, native: NativeLimits, max_sampler_bytes: u64,
        prefill: Int8PrefillLimits, cancellation: CancellationToken)
        -> Result<HostedOutput<Int8ChatResult>, HostedError> {
        dispatch::preflight(self, native.run)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        if request_seq == 0 || max_sampler_bytes == 0 { return Err(HostedError::Limits("sequence/sampler")); }
        let required = requirements(native)?;
        let extra = prefill.validate().map_err(HostedError::Native)?;
        let work = prepared.planned_work();
        if work.forward_positions > native.context_tokens as u64
            || required.kv_bytes > prepared.task_plan().ir().budget().max_kv_bytes
            || prepared.native_plan().sampler_bytes() > max_sampler_bytes {
            return Err(HostedError::Limits("native task preflight"));
        }
        let lease = self.resources().acquire_lease();
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound, extra,
                max_sampler_bytes, native.allocator_reserve_bytes])?)?;
        let output = output_claim(&lease, prepared.task_plan().ir().budget().max_output_bytes,
            u64::from(prepared.task_plan().ir().budget().max_output_tokens))?;
        let model = model.clone();
        dispatch::run(self, native.run, cancellation, move |control| {
            let mut engine = allocate_native(kv, workspace, || model.inner.loaded.value
                .engine(native.context_tokens, memory_budget(required)).map_err(HostedError::Model))?;
            let budget = Int8GenerationBudget { native: Int8RunBudget::exact(work),
                max_kv_bytes: required.kv_bytes, max_sampler_bytes };
            let result = prepared.execute_layer_major(prepared.execution_identity(), &mut engine.value,
                request_seq, budget, prefill, control).map_err(HostedError::Chat)?;
            drop(engine);
            let committed = output.commit()?;
            drop(lease);
            Ok(GuardedOutput::new(result, committed))
        })
    }
}
