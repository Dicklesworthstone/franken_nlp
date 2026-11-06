//! A fixed native schedule; the caller's five-axis document ledger remains
//! authoritative across every chunk, window, failed attempt and flush.
use super::*;
use crate::{native_engine::strict_int4::{Int4ModelIdentity, cohort::MAX_INT4_SEQUENCES},
    tasks::classify::grouped_int4::long::{Int4ClassificationMapCohortRun, PreparedInt4ClassificationMap}};

/// Preserve the original native result without an extra stream-only wrapper.
/// Serial encoding is unchanged; explicit cohorts retain their real counters.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(untagged)]
pub enum Int4ClassificationDocumentRun {
    Serial(Int4ClassificationMapRun),
    Cohorts(Int4ClassificationMapCohortRun),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct CohortSchedule {
    pub chunks: usize,
    pub max_heads: usize,
    pub kv_per_head: u64,
    pub required: Int4ClassificationMapCohortMemoryRequirement,
}
impl CohortSchedule {
    pub(super) fn new(width: usize, task: TaskBudget, planning: ClassificationLimits,
        mapping: Int4ClassificationMapLimits, chunks: usize, memory: Int4ClassificationMapMemoryLimits)
        -> Result<Self, BatchFault> {
        if memory.context_tokens != mapping.chunks.context_tokens {
            return Err(BatchCode::InvalidLimits.into());
        }
        let max_heads = cohort_head_limit(planning.max_labels, chunks)?;
        let required = memory.cohort_requirements(width, task, planning, mapping, max_heads, chunks)
            .map_err(|error| map_failure(error, false).fault)?;
        let kv_per_head = Int4MemoryRequirement::for_context(memory.context_tokens, width)
            .map_err(|_| BatchCode::InvalidLimits)?.kv_bytes;
        Ok(Self { chunks, max_heads, kv_per_head, required })
    }
    pub(super) fn check_request(self, mode: ClassificationMode, labels: usize, task: TaskBudget)
        -> Result<(), BatchItemFailure> {
        let heads = if mode == ClassificationMode::Exclusive { 1 } else { labels };
        if labels == 0 || heads == 0 || heads > self.max_heads
            || self.chunks.checked_mul(heads).is_none_or(|rows| rows > MAX_INT4_SEQUENCES) {
            return Err(BatchItemFailure::reject(BatchCode::Planning));
        }
        // One chunk's task authority must cover all of its independent heads;
        // the whole window's KV reservation is a separate host obligation.
        if self.kv_per_head.checked_mul(heads as u64).is_none_or(|bytes| bytes > task.max_kv_bytes) {
            return Err(BatchItemFailure::reject(BatchCode::Admission));
        }
        Ok(())
    }
}

pub(crate) fn cohort_head_limit(max_labels: usize, chunks: usize) -> Result<usize, BatchFault> {
    if max_labels == 0 || !(1..=MAX_INT4_SEQUENCES).contains(&chunks) {
        return Err(BatchCode::InvalidLimits.into());
    }
    Ok(max_labels.min(MAX_INT4_SEQUENCES / chunks))
}

pub(super) enum Backend<'e, 'w> {
    Serial(&'e mut StrictInt4Engine<'w>),
    Cohorts { weights: &'w Int4ModelWeights, budget: Int4ClassificationMapCohortBudget,
        kernel: PackedInt4Kernel },
}
impl Backend<'_, '_> {
    fn model_identity(&self) -> &Int4ModelIdentity {
        match self { Self::Serial(engine) => engine.model_identity(), Self::Cohorts { weights, .. } => weights.identity() }
    }
    pub(super) fn is_clean(&self) -> bool {
        match self {
            Self::Serial(engine) => clean(engine),
            // The existing map executor owns and drops every native window
            // before returning. A failed attempt still poisons the run ledger.
            Self::Cohorts { .. } => true,
        }
    }
    pub(super) fn check_capacity(&self, compiler: &Int4ClassificationDocumentBatchPlanner, task: TaskBudget)
        -> Result<u64, BatchItemFailure> {
        constrained_int4::check_model(compiler.base_execution_identity(), self.model_identity())
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        match self {
            Self::Serial(engine) if compiler.cohorts.is_none() => capacity(compiler, task, engine),
            Self::Cohorts { budget, .. } => {
                let schedule = compiler.cohorts.ok_or_else(|| BatchItemFailure::fatal(BatchCode::Admission))?;
                check_cohort_budget(schedule.required, compiler.mapping.max_model_work, *budget)?;
                if task.max_kv_bytes < schedule.kv_per_head {
                    return Err(BatchItemFailure::reject(BatchCode::Admission));
                }
                Ok(schedule.required.memory.native.kv_bytes)
            }
            _ => Err(BatchItemFailure::fatal(BatchCode::Admission)),
        }
    }
    pub(super) fn preflight<C: DecodeStepControl>(&self, compiler: &Int4ClassificationDocumentBatchPlanner,
        plan: &PreparedInt4ClassificationMap<'_>, admitted: &[ExecutionIdentity], kv: u64, control: &mut C)
        -> Result<(), Int4ClassificationMapError> {
        match self {
            Self::Serial(engine) => plan.preflight_with_control(admitted, engine, budget(plan.planned_work(), kv), control),
            Self::Cohorts { weights, budget, .. } => {
                let chunks = compiler.cohort_chunks().ok_or(Int4ClassificationMapError::Admission)?;
                plan.preflight_cohorts_with_control(admitted, weights.identity(), chunks,
                    Int4ClassificationMapCohortBudget { native: Int4RunBudget::exact(plan.planned_work()), ..*budget }, control)
            }
        }
    }
    pub(super) fn execute<C: DecodeStepControl>(&mut self, compiler: &Int4ClassificationDocumentBatchPlanner,
        plan: &PreparedInt4ClassificationMap<'_>, admitted: &[ExecutionIdentity], kv: u64, control: &mut C)
        -> Result<Int4ClassificationDocumentRun, Int4ClassificationMapError> {
        match self {
            Self::Serial(engine) => plan.execute_with_control(admitted, engine, budget(plan.planned_work(), kv), control)
                .map(Int4ClassificationDocumentRun::Serial),
            Self::Cohorts { weights, budget, kernel } => {
                let chunks = compiler.cohort_chunks().ok_or(Int4ClassificationMapError::Admission)?;
                plan.execute_cohorts_with_control(admitted, weights, *kernel, chunks,
                    Int4ClassificationMapCohortBudget { native: Int4RunBudget::exact(plan.planned_work()), ..*budget }, control)
                    .map(Int4ClassificationDocumentRun::Cohorts)
            }
        }
    }
}

fn check_cohort_budget(required: Int4ClassificationMapCohortMemoryRequirement, work: Int4Work,
    budget: Int4ClassificationMapCohortBudget) -> Result<(), BatchItemFailure> {
    let native = required.memory.native;
    if native.kv_bytes > budget.memory.max_kv_bytes || native.rope_bytes > budget.memory.max_rope_bytes
        || native.scratch_payload_bound > budget.memory.max_scratch_payload_bytes
        || required.task_workspace_bytes > budget.max_task_workspace_bytes
        || required.max_cohort_output_bytes != budget.max_output_bytes
        || work.forward_positions > budget.native.max_forward_positions
        || work.attention_pairs > budget.native.max_attention_pairs
        || !work.projections.fits(budget.native.max_projection_work) {
        return Err(BatchItemFailure::fatal(BatchCode::Admission));
    }
    Ok(())
}

#[cfg(test)] mod tests;
