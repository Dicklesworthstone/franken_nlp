//! Explicit whole-request output-row teams for single-request INT4 tasks.
use super::*;
use crate::native_engine::strict_int4::projection::Int4ProjectionStats;
mod task;
pub(super) use task::{TaskMemory, execute_task};

/// Physical execution evidence, separate from model, prompt and sampler identity.
/// Counts describe successful projections; they do not certify speed or parity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Int4ComputeReceipt {
    pub schedule: &'static str,
    /// Coordinator plus persistent scoped children. Runtime threads are extra.
    pub compute_threads: usize,
    pub completed_projections: u64,
    pub completed_rows: u64,
    pub completed_worker_rows: u64,
}
impl From<Int4ProjectionStats> for Int4ComputeReceipt {
    fn from(stats: Int4ProjectionStats) -> Self {
        Self { schedule: "int4-output-row-team-v1", compute_threads: stats.team_width,
            completed_projections: stats.completed_projections, completed_rows: stats.completed_rows,
            completed_worker_rows: stats.completed_worker_rows }
    }
}

/// A complete task and its physical counters retain the ordinary output
/// charge through serialization and publication by their caller. The receipt
/// does not alter the task's model, prompt, grammar or scoring identity.
pub struct HostedInt4TeamResult<T> {
    output: HostedOutput<T>,
    compute: Int4ComputeReceipt,
}
impl<T> HostedInt4TeamResult<T> {
    pub fn result(&self) -> &T { self.output.result() }
    pub fn compute(&self) -> &Int4ComputeReceipt { &self.compute }
}
/// Backward-compatible name for the buffered chat/generation team result.
pub type HostedInt4ChatTeamResult = HostedInt4TeamResult<Int4ChatResult>;

/// Additional modeled scratch for one fixed team, exclusive of the ordinary
/// native KV, activation, sampler, allocator, result and cleanup reservations.
pub fn int4_chat_team_workspace_bytes(compute_threads: usize, group_width: usize)
    -> Result<u64, HostedError> {
    dispatch::int4_team_workspace_bytes(compute_threads, group_width)
}

/// Additional modeled scratch for a fixed structured/scored task team. This
/// is the same quote used by the dispatcher; it excludes ordinary native KV,
/// task preparation, masks, result storage, allocator and cleanup reserves.
pub fn int4_task_team_workspace_bytes(compute_threads: usize, group_width: usize)
    -> Result<u64, HostedError> {
    dispatch::int4_team_workspace_bytes(compute_threads, group_width)
}

impl NlpEngine {
    /// Run one ordinary INT4 text request on a fixed team of at least two
    /// compute threads. Each projection partitions complete output rows; no
    /// row's reduction order changes. The team is formed once for the request,
    /// sealed before math starts, and joined before physical completion.
    ///
    /// The configured resource host must admit the requested children and
    /// process thread ceiling. This route does not enable packed prefill or
    /// independent-sequence cohorts, and never falls back to another width.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4_chat_with_threads(&self, model: &ResidentInt4, prepared: PreparedInt4Chat,
        request_seq: u64, native: NativeLimits, max_sampler_bytes: u64, kernel: PackedInt4Kernel,
        compute_threads: usize, cancellation: CancellationToken)
        -> Result<HostedInt4ChatTeamResult, HostedError> {
        dispatch::preflight_int4_team(self, native.run, compute_threads)?;
        if !Arc::ptr_eq(self.resources(), model.resources()) { return Err(HostedError::ResourceDomain); }
        let width = model.model_identity().group_width();
        let rebound = bind_candidate_model(prepared.execution_identity().clone(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        if prepared.native_plan().group_width() != width || &rebound != prepared.execution_identity() {
            return Err(HostedError::ModelIdentity);
        }
        let required = native_requirements(&prepared, request_seq, native, max_sampler_bytes)?;
        // The dispatcher reserves the same team quote before its single
        // blocking crossing; no native worker buffer exists before admission.
        int4_chat_team_workspace_bytes(compute_threads, width)?;
        let scratch = sum(&[required.rope_bytes, required.scratch_payload_bound,
            max_sampler_bytes, native.allocator_reserve_bytes])?;
        let work = prepared.planned_work();
        let lease = self.resources().acquire_lease();
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.kv_bytes)?;
        let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, scratch)?;
        let output = output_claim(&lease, prepared.task_plan().ir().budget().max_output_bytes,
            u64::from(prepared.task_plan().ir().budget().max_output_tokens))?;
        dispatch::run_int4_team(self, native.run, cancellation, model.clone(), compute_threads,
            kernel, move |weights, control, team| {
                let mut engine = allocate_native(kv, workspace, || {
                    StrictInt4Engine::new(weights, native.context_tokens, Int4MemoryBudget {
                        max_kv_bytes: required.kv_bytes, max_rope_bytes: required.rope_bytes,
                        max_scratch_payload_bytes: required.scratch_payload_bound,
                    }).map_err(native_error)?.with_kernel(kernel)
                        .with_projection_executor(team).map_err(native_error)
                })?;
                let budget = Int4GenerationBudget { native: Int4RunBudget::exact(work),
                    max_kv_bytes: required.kv_bytes, max_sampler_bytes };
                let result = prepared.execute(prepared.execution_identity(), &mut engine.value,
                    request_seq, budget, control).map_err(chat_error)?;
                drop(engine);
                let compute = team.stats().into();
                let committed = output.commit()?;
                drop(lease);
                Ok(HostedInt4ChatTeamResult { output: GuardedOutput::new(result, committed), compute })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compute_receipt_keeps_physical_work_separate_and_contains_no_text_identity() {
        let receipt = Int4ComputeReceipt::from(Int4ProjectionStats { team_width: 4,
            completed_projections: 31, completed_rows: 1000, completed_worker_rows: 750 });
        assert_eq!(serde_json::to_value(receipt).unwrap(), serde_json::json!({
            "schedule": "int4-output-row-team-v1", "compute_threads": 4,
            "completed_projections": 31, "completed_rows": 1000, "completed_worker_rows": 750,
        }));
    }
    #[test]
    fn team_workspace_is_extra_and_refuses_invalid_widths() {
        for threads in [0, 1, usize::MAX] {
            assert!(int4_chat_team_workspace_bytes(threads, 128).is_err());
        }
        let two = int4_chat_team_workspace_bytes(2, 128).unwrap();
        let four = int4_chat_team_workspace_bytes(4, 128).unwrap();
        assert!(two > 0 && four > two);
        assert!(int4_chat_team_workspace_bytes(2, 0).is_err());
    }
}
