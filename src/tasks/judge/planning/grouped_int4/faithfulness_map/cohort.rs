//! Real native chunk/evidence-head cohorts. Ragged windows retain each
//! original judgment's complete heads and finalizer without serial fallback.
use super::*;
use crate::native_engine::{int4::PackedInt4Kernel,
    strict_int4::{Int4MemoryBudget, Int4MemoryRequirement, Int4ModelWeights,
        cohort::{Int4CohortEngine, MAX_INT4_SEQUENCES},
        scoring::cohort::{Int4ScoredCohort, Int4ScoringCohortBudget,
            INT4_SCORING_COHORT_EXECUTION, required_task_workspace_bytes,
            task::document_layout}}};

pub const INT4_FAITHFULNESS_MAP_COHORT_EXECUTION: &str = "grouped-int4-faithfulness-chunk-cohorts-v1";
pub(super) const COHORT_WRAPPER_BYTES: u64 = 4096;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Int4FaithfulnessMapCohortRun {
    pub schema_version: u32,
    pub execution: String,
    pub chunk_cohorts: usize,
    pub group_steps: u64,
    pub projection_groups: u64,
    pub output: Int4FaithfulnessMapRun,
}
#[derive(Clone, Copy, Debug)]
pub struct Int4FaithfulnessMapCohortBudget {
    pub native: Int4RunBudget,
    pub memory: Int4MemoryBudget,
    pub max_task_workspace_bytes: u64,
    pub max_output_bytes: u64,
}

impl PreparedInt4FaithfulnessMap<'_> {
    /// Full model-bound admission for every future window without allocating
    /// an engine. Call before external persistence; execution repeats it.
    pub fn preflight_cohorts_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        model: &Int4ModelIdentity, cohort_chunks: usize, budget: Int4FaithfulnessMapCohortBudget,
        control: &mut C) -> Result<(), Int4FaithfulnessMapError> {
        self.admitted_cohort_windows(admitted, model, cohort_chunks, budget, control).map(|_| ())
    }
    fn admitted_cohort_windows<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        model: &Int4ModelIdentity, cohort_chunks: usize, budget: Int4FaithfulnessMapCohortBudget,
        control: &mut C) -> Result<Vec<Window>, Int4FaithfulnessMapError> {
        checkpoint(control)?;
        if model.group_width() != self.group_width() || admitted.len() != self.plans.len() {
            return Err(Int4FaithfulnessMapError::Admission);
        }
        for identity in admitted { checkpoint(control)?; check_model(identity, model)?; }
        self.cohort_windows(admitted, cohort_chunks, budget, control)
    }

    /// Check every actual future window, including a longer later chunk's
    /// additional evidence heads. Never resize or silently serialize a window.
    pub fn validate_cohort_chunks(&self, cohort_chunks: usize) -> Result<(), Int4FaithfulnessMapError> {
        if cohort_chunks == 0 || cohort_chunks > self.limits.chunks.max_chunks
            || cohort_chunks > MAX_INT4_SEQUENCES || self.plans.is_empty() {
            return Err(Int4FaithfulnessMapError::Limits);
        }
        for window in self.plans.chunks(cohort_chunks) {
            let mut heads = 0_usize;
            for plan in window {
                if plan.head_count() == 0 || plan.head_count() > MAX_FAITHFULNESS_MAP_HEADS {
                    return Err(Int4FaithfulnessMapError::Admission);
                }
                heads = heads.checked_add(plan.head_count()).ok_or(Int4FaithfulnessMapError::Limits)?;
            }
            if heads > MAX_INT4_SEQUENCES { return Err(Int4FaithfulnessMapError::Limits); }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_cohorts_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        weights: &Int4ModelWeights, kernel: PackedInt4Kernel, cohort_chunks: usize,
        budget: Int4FaithfulnessMapCohortBudget, control: &mut C)
        -> Result<Int4FaithfulnessMapCohortRun, Int4FaithfulnessMapError> {
        let windows = self.admitted_cohort_windows(admitted, weights.identity(), cohort_chunks, budget, control)?;
        self.collect_cohorts_with_control(&windows, control, |window, control| {
            checkpoint(control)?;
            let mut documents = reserved(window.end - window.start)?;
            for index in window.start..window.end { documents.push((&self.plans[index], &admitted[index])); }
            let mut engine = Int4CohortEngine::new(weights, &window.contexts, budget.memory)?.with_kernel(kernel);
            let run = PreparedInt4Judge::execute_document_cohort_with_control(&documents, &mut engine,
                Int4ScoringCohortBudget { native: Int4RunBudget::exact(window.work),
                    max_kv_bytes: budget.memory.max_kv_bytes, max_output_bytes: budget.max_output_bytes }, control)?;
            engine.check_idle()?;
            drop(engine);
            checkpoint(control)?;
            Ok(run)
        })
    }

    pub(super) fn cohort_windows<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity], chunks: usize,
        budget: Int4FaithfulnessMapCohortBudget, control: &mut C)
        -> Result<Vec<Window>, Int4FaithfulnessMapError> {
        checkpoint(control)?;
        self.validate_cohort_chunks(chunks)?;
        if !(1..=MAX_BYTES).contains(&budget.max_output_bytes) || budget.max_task_workspace_bytes == 0 {
            return Err(Int4FaithfulnessMapError::Limits);
        }
        if admitted.len() != self.plans.len() || self.plans.len() != self.expected.chunks
            || self.plans.len() != self.chunks.chunks().len() {
            return Err(Int4FaithfulnessMapError::Admission);
        }
        check_work(self.planned_work(), budget.native)?;
        if !fits_work(self.planned_work(), self.limits.max_model_work) { return Err(Int4FaithfulnessMapError::Work); }
        let mut windows = reserved(self.plans.len().div_ceil(chunks))?;
        let mut total = Int4Work::default();
        for start in (0..self.plans.len()).step_by(chunks) {
            checkpoint(control)?;
            let end = start + chunks.min(self.plans.len() - start);
            let head_count = self.plans[start..end].iter().try_fold(0_usize, |n, plan|
                n.checked_add(plan.head_count()).ok_or(Int4FaithfulnessMapError::Limits))?;
            let mut contexts = reserved(head_count)?;
            let mut documents = reserved(end - start)?;
            let mut expected = Int4Work::default();
            let mut projection_steps = 0_u64;
            let mut minimum_projection_steps = 0_u64;
            for index in start..end {
                checkpoint(control)?;
                let plan = &self.plans[index];
                plan.verify_identity(&admitted[index])?;
                if plan.group_width() != self.group_width() || plan.task_budget() != self.task_budget() {
                    return Err(Int4FaithfulnessMapError::Admission);
                }
                contexts.extend(plan.cohort_contexts()?);
                expected = expected.checked_add(plan.planned_work())?;
                for schedule in &plan.schedules {
                    projection_steps = projection_steps.checked_add(schedule.scoring.prefix_evaluations as u64)
                        .ok_or(Int4FaithfulnessMapError::Work)?;
                    minimum_projection_steps = minimum_projection_steps.max(schedule.scoring.prefix_evaluations as u64);
                }
                documents.push((plan, &admitted[index]));
            }
            let required = Int4MemoryRequirement::for_cohort(&contexts, self.group_width())?;
            if required.kv_bytes > budget.memory.max_kv_bytes || required.rope_bytes > budget.memory.max_rope_bytes
                || required.scratch_payload_bound > budget.memory.max_scratch_payload_bytes {
                return Err(Int4FaithfulnessMapError::Admission);
            }
            let (_, work) = document_layout(&documents, &contexts, Int4ScoringCohortBudget {
                native: Int4RunBudget::exact(expected), max_kv_bytes: budget.memory.max_kv_bytes,
                max_output_bytes: budget.max_output_bytes })?;
            if work != expected || projection_steps.checked_mul(NANBEIGE_VOCAB_SIZE as u64) != Some(work.projected_logits)
                || required_task_workspace_bytes(work, contexts.len(), budget.max_output_bytes)?
                > budget.max_task_workspace_bytes {
                return Err(Int4FaithfulnessMapError::Admission);
            }
            total = total.checked_add(work)?;
            windows.push(Window { start, end, contexts, work, projection_steps, minimum_projection_steps });
        }
        if total != self.planned_work() { return Err(Int4FaithfulnessMapError::Admission); }
        checkpoint(control)?; Ok(windows)
    }

    // Only the internal execution handoff and private fixtures can supply
    // native completions. Both schedules share the exact complete collector.
    pub(super) fn collect_cohorts_with_control<C: DecodeStepControl, F>(&self, windows: &[Window],
        control: &mut C, mut execute: F) -> Result<Int4FaithfulnessMapCohortRun, Int4FaithfulnessMapError>
    where F: FnMut(&Window, &mut C) -> Result<Int4ScoredCohort<Vec<Int4JudgeRun>>, Int4FaithfulnessMapError> {
        let mut next_window = 0;
        let mut results = Vec::new().into_iter();
        let mut group_steps = 0_u64; let mut projection_groups = 0_u64;
        let output = self.collect_with_control(control, |index, _, control| {
            if results.len() == 0 {
                let window = windows.get(next_window).ok_or(Int4FaithfulnessMapError::Admission)?;
                if window.start != index || window.end <= index || window.end > self.plans.len() {
                    return Err(Int4FaithfulnessMapError::Admission);
                }
                let run = execute(window, control)?;
                checkpoint(control)?;
                if run.schema_version != 1 || run.execution != INT4_SCORING_COHORT_EXECUTION
                    || run.output.len() != window.end - window.start
                    || run.group_steps < window.contexts.iter().copied().max().unwrap_or(1) as u64
                    || run.group_steps > window.work.forward_positions
                    || run.projection_groups < window.minimum_projection_steps
                    || run.projection_groups > window.projection_steps {
                    return Err(Int4FaithfulnessMapError::Admission);
                }
                group_steps = group_steps.checked_add(run.group_steps).ok_or(Int4FaithfulnessMapError::Work)?;
                projection_groups = projection_groups.checked_add(run.projection_groups).ok_or(Int4FaithfulnessMapError::Work)?;
                results = run.output.into_iter(); next_window += 1;
            }
            results.next().ok_or(Int4FaithfulnessMapError::Admission)
        })?;
        if results.len() != 0 || next_window != windows.len() { return Err(Int4FaithfulnessMapError::Admission); }
        let run = Int4FaithfulnessMapCohortRun { schema_version: 1,
            execution: INT4_FAITHFULNESS_MAP_COHORT_EXECUTION.to_owned(),
            chunk_cohorts: windows.len(), group_steps, projection_groups, output };
        value_bytes(&run, self.limits.max_result_bytes.checked_add(COHORT_WRAPPER_BYTES)
            .ok_or(Int4FaithfulnessMapError::Limits)?)?;
        checkpoint(control)?; Ok(run)
    }
}

pub(super) struct Window {
    pub start: usize, pub end: usize, pub contexts: Vec<usize>, pub work: Int4Work,
    projection_steps: u64, minimum_projection_steps: u64,
}
