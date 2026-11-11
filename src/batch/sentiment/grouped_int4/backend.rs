//! Native execution only. All cohort windows are preflighted before the first
//! engine allocation; the outer lifetime ledger is shared by both schedules.
use super::*;
use crate::{native_engine::strict_int4::Int4MemoryBudget,
    tasks::sentiment::grouped_int4::long::{PreparedInt4SentimentMap,
        Int4SentimentMapCohortBudget,
        memory::Int4SentimentMapMemoryRequirement}};

pub(super) enum Backend<'e, 'w> {
    Serial(&'e mut StrictInt4Engine<'w>),
    Cohorts { weights: &'w Int4ModelWeights, kernel: PackedInt4Kernel,
        chunks: usize, context: usize, max_axes: usize, required: Int4SentimentMapMemoryRequirement },
}
impl<'e, 'w> Backend<'e, 'w> {
    pub(super) fn cohorts(compiler: &Int4SentimentDocumentBatchPlanner, weights: &'w Int4ModelWeights,
        kernel: PackedInt4Kernel, memory: Int4SentimentMapMemoryLimits) -> Result<Self, BatchFault> {
        let chunks = compiler.cohort_chunks.ok_or(BatchCode::InvalidLimits)?;
        if memory.cohort_chunks != Some(chunks) || memory.max_axes != compiler.max_axes {
            return Err(BatchCode::InvalidLimits.into());
        }
        validate_geometry(memory.max_axes, Some(chunks)).map_err(|e| e.fault)?;
        constrained_int4::check_model(compiler.base_execution_identity(), weights.identity())
            .map_err(|_| BatchCode::Admission)?;
        let required = memory.requirements(compiler.group_width(), compiler.task_budget(),
            compiler.planning_limits(), compiler.mapping).map_err(|e| map_failure(e, false).fault)?;
        Ok(Self::Cohorts { weights, kernel, chunks, context: memory.context_tokens,
            max_axes: memory.max_axes, required })
    }
    pub(super) fn clean(&self) -> bool {
        match self { Self::Serial(engine) => !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0),
            // Cohort engines are scoped to exact windows in the sealed map
            // executor and are physically destroyed before it returns.
            Self::Cohorts { .. } => true }
    }
    pub(super) fn check(&self, compiler: &Int4SentimentDocumentBatchPlanner, task: TaskBudget,
        axes: usize) -> Result<u64, BatchItemFailure> {
        if !self.clean() || axes > compiler.max_axes {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        validate_geometry(axes, compiler.cohort_chunks)?;
        let (model, context, kv) = match self {
            Self::Serial(engine) => {
                if compiler.cohort_chunks.is_some() { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
                let context = engine.kv_cache().capacity_positions();
                let required = Int4MemoryRequirement::for_context(context, compiler.group_width())
                    .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
                (engine.model_identity(), context, required.kv_bytes)
            }
            Self::Cohorts { weights, chunks, context, max_axes, required, .. } => {
                if compiler.cohort_chunks != Some(*chunks) || compiler.max_axes != *max_axes {
                    return Err(BatchItemFailure::fatal(BatchCode::Admission));
                }
                (weights.identity(), *context, required.native.kv_bytes)
            }
        };
        constrained_int4::check_model(compiler.base_execution_identity(), model)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        if model.group_width() != compiler.group_width() || context < compiler.mapping.chunks.context_tokens {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        check_task_kv(context, compiler.group_width(), axes, task)?;
        Ok(kv)
    }
    pub(super) fn preflight<C: DecodeStepControl>(&self, plan: &PreparedInt4SentimentMap<'_>,
        identities: &[ExecutionIdentity], control: &mut C) -> Result<(), Int4SentimentMapError> {
        match self {
            Self::Serial(engine) => {
                let kv = Int4MemoryRequirement::for_context(engine.kv_cache().capacity_positions(), plan.group_width())?.kv_bytes;
                plan.preflight_with_control(identities, engine, serial_budget(plan.planned_work(), kv), control)
            }
            Self::Cohorts { weights, chunks, required, .. } => plan.preflight_cohorts_with_control(identities,
                weights, *chunks, cohort_budget(plan.planned_work(), *required), control),
        }
    }
    pub(super) fn execute<C: DecodeStepControl>(&mut self, plan: &PreparedInt4SentimentMap<'_>,
        identities: &[ExecutionIdentity], control: &mut C) -> Result<Int4SentimentDocumentRun, Int4SentimentMapError> {
        match self {
            Self::Serial(engine) => {
                let kv = Int4MemoryRequirement::for_context(engine.kv_cache().capacity_positions(), plan.group_width())?.kv_bytes;
                plan.execute_with_control(identities, engine, serial_budget(plan.planned_work(), kv), control)
                    .map(Int4SentimentDocumentRun::Serial)
            }
            Self::Cohorts { weights, kernel, chunks, required, .. } => plan.execute_cohorts_with_control(identities,
                weights, *kernel, *chunks, cohort_budget(plan.planned_work(), *required), control)
                .map(Int4SentimentDocumentRun::ChunkCohorts),
        }
    }
}

pub(super) fn check_task_kv(context: usize, width: usize, axes: usize, task: TaskBudget)
    -> Result<(), BatchItemFailure> {
    let row = Int4MemoryRequirement::for_context(context, width)
        .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
    // TaskBudget is one complete chunk's independent axes. The physical
    // cohort owns several chunks simultaneously and has a separate KV charge.
    if task.validate().is_err() || row.kv_bytes.checked_mul(axes as u64).is_none_or(|n| n > task.max_kv_bytes) {
        return Err(BatchItemFailure::reject(BatchCode::Admission));
    }
    Ok(())
}
fn serial_budget(work: Int4Work, max_kv_bytes: u64) -> Int4ScoringBudget {
    Int4ScoringBudget { native: Int4RunBudget::exact(work), max_kv_bytes }
}
fn cohort_budget(work: Int4Work, required: Int4SentimentMapMemoryRequirement) -> Int4SentimentMapCohortBudget {
    Int4SentimentMapCohortBudget { native: Int4RunBudget::exact(work),
        memory: Int4MemoryBudget { max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
            max_scratch_payload_bytes: required.native.scratch_payload_bound },
        max_task_workspace_bytes: required.task_workspace_bytes,
        max_output_bytes: required.max_cohort_output_bytes }
}

pub(super) fn verify_completed<C: DecodeStepControl>(expected: &Int4SentimentMapPreflight, chunks: Option<usize>,
    result: &Int4SentimentDocumentRun, control: &mut C) -> Result<(), Int4SentimentMapError> {
    match (chunks, result) {
        (None, Int4SentimentDocumentRun::Serial(run)) => expected.verify_completed_with_control(run, control),
        (Some(chunks), Int4SentimentDocumentRun::ChunkCohorts(run)) => {
            verify_schedule(expected.chunk_count(), chunks, run.chunk_cohorts,
                run.group_steps, run.projection_groups, expected.planned_work())?;
            expected.verify_completed_cohorts_with_control(run, chunks, control)
        }
        _ => Err(Int4SentimentMapError::Admission),
    }
}
pub(super) fn verify_schedule(doc_chunks: usize, window: usize, windows: usize,
    steps: u64, groups: u64, work: Int4Work) -> Result<(), Int4SentimentMapError> {
    if doc_chunks == 0 || window == 0 || windows != doc_chunks.div_ceil(window)
        || steps == 0 || steps > work.forward_positions || groups == 0 || groups > work.projected_logits {
        return Err(Int4SentimentMapError::Admission);
    }
    Ok(())
}
