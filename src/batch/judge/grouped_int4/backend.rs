//! The same sealed native map executes both schedules. No alternate scorer or
//! caller-supplied chunk result can enter the complete-source path.
use super::*;
use crate::{native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4MemoryBudget},
    tasks::judge::grouped_int4::faithfulness_map::{Int4FaithfulnessMapCohortBudget,
        INT4_FAITHFULNESS_MAP_COHORT_EXECUTION, memory::Int4FaithfulnessMapCohortMemoryRequirement}};

pub(super) enum Backend<'e, 'w> {
    Serial(&'e mut StrictInt4Engine<'w>),
    Cohorts { weights: &'w Int4ModelWeights, kernel: PackedInt4Kernel,
        chunks: usize, context: usize, max_heads: usize, required: Int4FaithfulnessMapCohortMemoryRequirement },
}
impl<'e, 'w> Backend<'e, 'w> {
    pub(super) fn cohorts(compiler: &Int4FaithfulnessDocumentBatchPlanner, weights: &'w Int4ModelWeights,
        kernel: PackedInt4Kernel, memory: Int4FaithfulnessMapMemoryLimits) -> Result<Self, BatchFault> {
        let chunks = compiler.cohort_chunks.ok_or(BatchCode::InvalidLimits)?;
        validate_cohort_chunks(chunks, compiler.mapping)?;
        constrained_int4::check_model(compiler.execution_identity(), weights.identity()).map_err(|_| BatchCode::Admission)?;
        let required = memory.cohort_requirements(compiler.group_width(), compiler.task_budget(),
            compiler.planning_limits(), compiler.mapping, compiler.max_heads, chunks)
            .map_err(|error| map_failure(error, false).fault)?;
        Ok(Self::Cohorts { weights, kernel, chunks, context: memory.context_tokens,
            max_heads: compiler.max_heads, required })
    }
    pub(super) fn clean(&self) -> bool {
        match self { Self::Serial(engine) => !engine.is_poisoned() && engine.kv_cache().all_slots_have_len(0),
            // Every exact window engine is scoped and destroyed by the sealed
            // map executor before it returns or propagates cancellation.
            Self::Cohorts { .. } => true }
    }
    pub(super) fn check(&self, compiler: &Int4FaithfulnessDocumentBatchPlanner) -> Result<u64, BatchItemFailure> {
        if !self.clean() { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
        let (model, context, kv) = match self {
            Self::Serial(engine) => {
                if compiler.cohort_chunks.is_some() { return Err(BatchItemFailure::fatal(BatchCode::Admission)); }
                let context = engine.kv_cache().capacity_positions();
                let row = Int4MemoryRequirement::for_context(context, compiler.group_width())
                    .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
                if row.kv_bytes > compiler.task_budget().max_kv_bytes {
                    return Err(BatchItemFailure::fatal(BatchCode::Admission));
                }
                (engine.model_identity(), context, row.kv_bytes)
            }
            Self::Cohorts { weights, chunks, context, max_heads, required, .. } => {
                if compiler.cohort_chunks != Some(*chunks) || compiler.max_heads != *max_heads
                    || required.memory.max_output_bytes != compiler.max_result_bytes() {
                    return Err(BatchItemFailure::fatal(BatchCode::Admission));
                }
                (weights.identity(), *context, required.memory.native.kv_bytes)
            }
        };
        constrained_int4::check_model(compiler.execution_identity(), model)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        if model.group_width() != compiler.group_width() || context < compiler.mapping.chunks.context_tokens {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Ok(kv)
    }
    pub(super) fn preflight<C: DecodeStepControl>(&self, plan: &PreparedInt4FaithfulnessMap<'_>,
        identities: &[ExecutionIdentity], control: &mut C) -> Result<(), Int4FaithfulnessMapError> {
        match self {
            Self::Serial(engine) => {
                let kv = Int4MemoryRequirement::for_context(engine.kv_cache().capacity_positions(), plan.group_width())?.kv_bytes;
                plan.preflight_with_control(identities, engine, serial_budget(plan.planned_work(), kv), control)
            }
            Self::Cohorts { weights, chunks, required, .. } => plan.preflight_cohorts_with_control(identities,
                weights.identity(), *chunks, cohort_budget(plan.planned_work(), *required), control),
        }
    }
    pub(super) fn execute<C: DecodeStepControl>(&mut self, plan: &PreparedInt4FaithfulnessMap<'_>,
        identities: &[ExecutionIdentity], control: &mut C) -> Result<Int4FaithfulnessDocumentRun, Int4FaithfulnessMapError> {
        match self {
            Self::Serial(engine) => {
                let kv = Int4MemoryRequirement::for_context(engine.kv_cache().capacity_positions(), plan.group_width())?.kv_bytes;
                plan.execute_with_control(identities, engine, serial_budget(plan.planned_work(), kv), control)
                    .map(Int4FaithfulnessDocumentRun::Serial)
            }
            Self::Cohorts { weights, kernel, chunks, required, .. } => plan.execute_cohorts_with_control(identities,
                weights, *kernel, *chunks, cohort_budget(plan.planned_work(), *required), control)
                .map(Int4FaithfulnessDocumentRun::ChunkCohorts),
        }
    }
}

fn serial_budget(work: Int4Work, max_kv_bytes: u64) -> Int4ScoringBudget {
    Int4ScoringBudget { native: Int4RunBudget::exact(work), max_kv_bytes }
}
fn cohort_budget(work: Int4Work, required: Int4FaithfulnessMapCohortMemoryRequirement) -> Int4FaithfulnessMapCohortBudget {
    Int4FaithfulnessMapCohortBudget { native: Int4RunBudget::exact(work),
        memory: Int4MemoryBudget { max_kv_bytes: required.memory.native.kv_bytes,
            max_rope_bytes: required.memory.native.rope_bytes,
            max_scratch_payload_bytes: required.memory.native.scratch_payload_bound },
        max_task_workspace_bytes: required.task_workspace_bytes, max_output_bytes: required.max_cohort_output_bytes }
}
pub(super) fn verify_completed<C: DecodeStepControl>(expected: &Int4FaithfulnessMapPreflight,
    chunks: Option<usize>, result: &Int4FaithfulnessDocumentRun, control: &mut C)
    -> Result<(), Int4FaithfulnessMapError> {
    match (chunks, result) {
        (None, Int4FaithfulnessDocumentRun::Serial(run)) => expected.verify_completed_with_control(run, control),
        (Some(chunks), Int4FaithfulnessDocumentRun::ChunkCohorts(run)) => {
            if run.schema_version != 1 || run.execution != INT4_FAITHFULNESS_MAP_COHORT_EXECUTION {
                return Err(Int4FaithfulnessMapError::Admission);
            }
            verify_schedule(expected.chunk_count(), chunks, run.chunk_cohorts,
                run.group_steps, run.projection_groups, expected.planned_work())?;
            expected.verify_completed_cohorts_with_control(run, chunks, control)
        }
        _ => Err(Int4FaithfulnessMapError::Admission),
    }
}
pub(super) fn verify_schedule(chunks: usize, width: usize, windows: usize,
    steps: u64, projections: u64, work: Int4Work) -> Result<(), Int4FaithfulnessMapError> {
    let vocab = NANBEIGE_VOCAB_SIZE as u64;
    if chunks == 0 || width == 0 || windows != chunks.div_ceil(width)
        || steps == 0 || steps > work.forward_positions || work.projected_logits % vocab != 0
        || projections == 0 || projections > work.projected_logits / vocab {
        return Err(Int4FaithfulnessMapError::Admission);
    }
    Ok(())
}
