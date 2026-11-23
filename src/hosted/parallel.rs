//! Explicit persistent output-row teams for buffered INT8 generation and chat.
use super::*;
use crate::native_engine::{simd::Int8Kernel, strict_int8::projection::Int8ProjectionStats};

/// Physical completion evidence, separate from prompt, model and sampler
/// identity. These counters do not certify speed or model fidelity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Int8ComputeReceipt {
    pub schedule: &'static str,
    /// Coordinator plus persistent children; runtime threads are additional.
    pub compute_threads: usize,
    pub completed_projections: u64,
    pub completed_rows: u64,
    pub completed_worker_rows: u64,
}
impl From<Int8ProjectionStats> for Int8ComputeReceipt {
    fn from(stats: Int8ProjectionStats) -> Self {
        Self { schedule: "int8-output-row-team-v1", compute_threads: stats.team_width,
            completed_projections: stats.completed_projections, completed_rows: stats.completed_rows,
            completed_worker_rows: stats.completed_worker_rows }
    }
}

/// The complete semantic result keeps its ordinary output reservation until
/// the caller finishes serialization and delivery. No result escapes a live
/// team: the dispatcher physically joins every child before returning it.
pub struct HostedInt8TeamResult<T> { output: HostedOutput<T>, compute: Int8ComputeReceipt }
impl<T> HostedInt8TeamResult<T> {
    pub fn result(&self) -> &T { self.output.result() }
    pub fn compute(&self) -> &Int8ComputeReceipt { &self.compute }
}
pub type HostedInt8ChatTeamResult = HostedInt8TeamResult<Int8ChatResult>;

/// Additional fixed worker storage, exclusive of ordinary native KV, scratch,
/// sampler, allocator, result and cleanup reservations.
pub fn int8_chat_team_workspace_bytes(compute_threads: usize) -> Result<u64, HostedError> {
    dispatch::int8_team_workspace_bytes(compute_threads)
}

/// Pending and committed reservations stay outside the scoped callback, so
/// native allocation failure, cancellation and panic all retain admission
/// until the dispatcher has joined every child. Storage is still freed before
/// its corresponding charge, without keeping an engine borrow across join.
struct ChatAdmission {
    kv: Option<Pending>, scratch: Option<Pending>, output: Option<Pending>,
    _kv: Option<CommittedMemory>, _scratch: Option<CommittedMemory>, _lease: EngineLease,
}
impl ChatAdmission {
    fn commit_native(&mut self) -> Result<(), HostedError> {
        self._kv = Some(self.kv.take().ok_or(HostedError::Limits("INT8 KV admission consumed"))?.commit()?);
        self._scratch = Some(self.scratch.take().ok_or(HostedError::Limits("INT8 scratch admission consumed"))?.commit()?);
        Ok(())
    }
}

impl NlpEngine {
    /// Execute one already prepared INT8 request with one fixed team of at
    /// least two compute threads. All decoder and head projections partition
    /// complete output rows, preserving each row's serial arithmetic order.
    /// The coordinator owns quantization, the sole native work ledger, KV,
    /// sequence order and sampling. Child formation occurs once per request.
    ///
    /// The resource host must admit this width before execution. This initial
    /// route accepts sequential prompt prefill only; it does not run packed
    /// prompt rows, streams, cohorts or resident batches on the team.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int8_chat_with_threads(&self, model: &ResidentInt8, prepared: PreparedInt8Chat,
        request_seq: u64, native: NativeLimits, max_sampler_bytes: u64, kernel: Int8Kernel,
        compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt8ChatTeamResult, HostedError> {
        dispatch::preflight_int8_team(self, native.run, compute_threads)?;
        self.check_resident_domain(model)?;
        check_model_identity(model.artifact_identity(), prepared.execution_identity())?;
        if request_seq == 0 || max_sampler_bytes == 0 { return Err(HostedError::Limits("sequence/sampler")); }
        let required = requirements(native)?;
        let work = prepared.planned_work();
        if work.forward_positions > native.context_tokens as u64
            || required.kv_bytes > prepared.task_plan().ir().budget().max_kv_bytes
            || prepared.native_plan().sampler_bytes() > max_sampler_bytes {
            return Err(HostedError::Limits("native task preflight"));
        }
        int8_chat_team_workspace_bytes(compute_threads)?;
        let lease = self.resources().acquire_lease();
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch,
            sum(&[required.rope_bytes, required.scratch_payload_bound,
                max_sampler_bytes, native.allocator_reserve_bytes])?)?;
        let output = output_claim(&lease, prepared.task_plan().ir().budget().max_output_bytes,
            u64::from(prepared.task_plan().ir().budget().max_output_tokens))?;
        let admission = ChatAdmission { kv: Some(kv), scratch: Some(workspace), output: Some(output),
            _kv: None, _scratch: None, _lease: lease };
        dispatch::run_int8_team(self, native.run, cancellation, model.clone(), compute_threads, kernel,
            admission, move |model, admission, control, team| {
                let mut engine = model.engine(native.context_tokens, memory_budget(required)).map_err(HostedError::Model)?
                    .with_kernel(kernel).with_projection_executor(team).map_err(HostedError::Native)?;
                admission.commit_native()?;
                let budget = Int8GenerationBudget { native: Int8RunBudget::exact(work),
                    max_kv_bytes: required.kv_bytes, max_sampler_bytes };
                let result = prepared.execute(prepared.execution_identity(), &mut engine,
                    request_seq, budget, control).map_err(HostedError::Chat)?;
                drop(engine);
                let compute = team.stats().into();
                let committed = admission.output.take().ok_or(HostedError::Limits("INT8 output admission consumed"))?.commit()?;
                Ok(HostedInt8TeamResult { output: GuardedOutput::new(result, committed), compute })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn receipt_contains_only_completed_physical_work() {
        let receipt = Int8ComputeReceipt::from(Int8ProjectionStats { team_width: 4,
            completed_projections: 31, completed_rows: 1000, completed_worker_rows: 750 });
        assert_eq!(serde_json::to_value(receipt).unwrap(), serde_json::json!({
            "schedule": "int8-output-row-team-v1", "compute_threads": 4,
            "completed_projections": 31, "completed_rows": 1000, "completed_worker_rows": 750,
        }));
    }
    #[test]
    fn quote_is_fixed_extra_storage_and_rejects_invalid_team_widths() {
        for width in [0, 1, 65, usize::MAX] { assert!(int8_chat_team_workspace_bytes(width).is_err()); }
        let two = int8_chat_team_workspace_bytes(2).unwrap();
        let four = int8_chat_team_workspace_bytes(4).unwrap();
        assert!(two > 0 && four > two);
    }
}
