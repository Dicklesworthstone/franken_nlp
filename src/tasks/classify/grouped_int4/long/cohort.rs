//! Shared native projections across independent source-chunk classification
//! heads. The original complete-document collector owns order and semantics.
use super::*;
use crate::native_engine::{int4::PackedInt4Kernel,
    strict_int4::{Int4MemoryBudget, Int4MemoryRequirement, Int4ModelWeights,
        cohort::{Int4CohortEngine, MAX_INT4_SEQUENCES},
        scoring::cohort::{Int4ScoredCohort, Int4ScoringCohortBudget,
            INT4_SCORING_COHORT_EXECUTION, required_task_workspace_bytes,
            task::document_layout}}};

pub const INT4_CLASSIFICATION_MAP_COHORT_EXECUTION: &str = "grouped-int4-classification-chunk-cohorts-v1";
pub(super) const COHORT_WRAPPER_BYTES: u64 = 4096;

/// Physical grouping statistics stay separate from the unchanged per-chunk
/// scores and the explicitly named whole-document aggregate estimator.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Int4ClassificationMapCohortRun {
    pub schema_version: u32,
    pub execution: String,
    pub chunk_cohorts: usize,
    pub group_steps: u64,
    pub projection_groups: u64,
    pub output: Int4ClassificationMapRun,
}

#[derive(Clone, Copy, Debug)]
pub struct Int4ClassificationMapCohortBudget {
    /// One allowance for the whole source, never replenished per window.
    pub native: Int4RunBudget,
    /// Peak simultaneous allocation across every chunk/head in a window.
    pub memory: Int4MemoryBudget,
    pub max_task_workspace_bytes: u64,
    /// Complete native window result, independently of the document cap.
    pub max_output_bytes: u64,
}

impl PreparedInt4ClassificationMap<'_> {
    /// Execute real native chunk × head cohorts, retaining independent KV
    /// and each original exact classifier finalizer. Every future window's
    /// identities, work, KV and workspace are checked before the first engine
    /// is allocated. An oversized requested window is refused, never shrunk
    /// into a hidden serial fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cohorts_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        weights: &Int4ModelWeights, kernel: PackedInt4Kernel, cohort_chunks: usize,
        budget: Int4ClassificationMapCohortBudget, control: &mut C)
        -> Result<Int4ClassificationMapCohortRun, Int4ClassificationMapError> {
        planning::checkpoint(control)?;
        if admitted.len() != self.plans.len() || weights.identity().group_width() != self.group_width() {
            return Err(Int4ClassificationMapError::Admission);
        }
        for identity in admitted {
            planning::checkpoint(control)?;
            check_model(identity, weights.identity())?;
        }
        let windows = self.cohort_windows(admitted, cohort_chunks, budget, control)?;
        self.collect_cohorts_with_control(&windows, control, |window, control| {
            planning::checkpoint(control)?;
            let mut documents = planning::reserved(window.end - window.start)?;
            for index in window.start..window.end { documents.push((&self.plans[index], &admitted[index])); }
            let mut engine = Int4CohortEngine::new(weights, &window.contexts, budget.memory)?.with_kernel(kernel);
            let run = PreparedInt4Classification::execute_document_cohort_with_control(&documents, &mut engine,
                Int4ScoringCohortBudget { native: Int4RunBudget::exact(window.work),
                    max_kv_bytes: budget.memory.max_kv_bytes, max_output_bytes: budget.max_output_bytes }, control)?;
            engine.check_idle()?;
            drop(engine);
            planning::checkpoint(control)?;
            Ok(run)
        })
    }

    pub(super) fn cohort_windows<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity], chunks: usize,
        budget: Int4ClassificationMapCohortBudget, control: &mut C)
        -> Result<Vec<Window>, Int4ClassificationMapError> {
        planning::checkpoint(control)?;
        if chunks == 0 || chunks > self.limits.chunks.max_chunks || self.head_count() == 0
            || chunks.checked_mul(self.head_count()).is_none_or(|heads| heads > MAX_INT4_SEQUENCES)
            || !(1..=MAX_BYTES).contains(&budget.max_output_bytes) || budget.max_task_workspace_bytes == 0 {
            return Err(Int4ClassificationMapError::Limits);
        }
        if self.plans.is_empty() || admitted.len() != self.plans.len()
            || self.plans.len() != self.expected.chunks || self.plans.len() != self.chunks.chunks().len() {
            return Err(Int4ClassificationMapError::Admission);
        }
        check_work_ceiling(self.planned_work(), budget.native)?;
        if !fits(self.planned_work(), self.limits.max_model_work) { return Err(Int4ClassificationMapError::Work); }
        let mut windows = planning::reserved(self.plans.len().div_ceil(chunks))?;
        let mut total = Int4Work::default();
        for start in (0..self.plans.len()).step_by(chunks) {
            planning::checkpoint(control)?;
            let end = start + chunks.min(self.plans.len() - start);
            let mut contexts = planning::reserved((end - start) * self.head_count())?;
            let mut documents = planning::reserved(end - start)?;
            let mut expected = Int4Work::default();
            for index in start..end {
                planning::checkpoint(control)?;
                let plan = &self.plans[index];
                plan.verify_identity(&admitted[index])?;
                if plan.group_width() != self.group_width() || plan.task_budget() != self.task_budget()
                    || plan.head_count() != self.head_count() {
                    return Err(Int4ClassificationMapError::Admission);
                }
                contexts.extend(plan.cohort_contexts()?);
                expected = expected.checked_add(plan.planned_work())?;
                documents.push((plan, &admitted[index]));
            }
            let required = Int4MemoryRequirement::for_cohort(&contexts, self.group_width())?;
            if required.kv_bytes > budget.memory.max_kv_bytes || required.rope_bytes > budget.memory.max_rope_bytes
                || required.scratch_payload_bound > budget.memory.max_scratch_payload_bytes {
                return Err(Int4ClassificationMapError::Admission);
            }
            let (_, work) = document_layout(&documents, &contexts, Int4ScoringCohortBudget {
                native: Int4RunBudget::exact(expected), max_kv_bytes: budget.memory.max_kv_bytes,
                max_output_bytes: budget.max_output_bytes })?;
            if work != expected || required_task_workspace_bytes(work, contexts.len(), budget.max_output_bytes)?
                > budget.max_task_workspace_bytes {
                return Err(Int4ClassificationMapError::Admission);
            }
            total = total.checked_add(work)?;
            windows.push(Window { start, end, contexts, work });
        }
        if total != self.planned_work() { return Err(Int4ClassificationMapError::Admission); }
        planning::checkpoint(control)?;
        Ok(windows)
    }

    // Private handoff: callers cannot supply externally fabricated native
    // scores. Both execution schedules use the same complete map collector.
    pub(super) fn collect_cohorts_with_control<C: DecodeStepControl, F>(&self, windows: &[Window], control: &mut C,
        mut execute: F) -> Result<Int4ClassificationMapCohortRun, Int4ClassificationMapError>
    where F: FnMut(&Window, &mut C) -> Result<Int4ScoredCohort<Vec<Int4ClassificationRun>>, Int4ClassificationMapError> {
        let mut next_window = 0;
        let mut results = Vec::new().into_iter();
        let mut group_steps = 0_u64;
        let mut projection_groups = 0_u64;
        let output = self.collect_with_control(control, |index, _, control| {
            if results.len() == 0 {
                let window = windows.get(next_window).ok_or(Int4ClassificationMapError::Admission)?;
                if window.start != index || window.end <= index || window.end > self.plans.len() {
                    return Err(Int4ClassificationMapError::Admission);
                }
                let run = execute(window, control)?;
                planning::checkpoint(control)?;
                if run.schema_version != 1 || run.execution != INT4_SCORING_COHORT_EXECUTION
                    || run.output.len() != window.end - window.start || run.group_steps == 0
                    || run.group_steps > window.work.forward_positions || run.projection_groups == 0
                    || run.projection_groups > window.work.projected_logits {
                    return Err(Int4ClassificationMapError::Admission);
                }
                group_steps = group_steps.checked_add(run.group_steps).ok_or(Int4ClassificationMapError::Work)?;
                projection_groups = projection_groups.checked_add(run.projection_groups).ok_or(Int4ClassificationMapError::Work)?;
                results = run.output.into_iter();
                next_window += 1;
            }
            results.next().ok_or(Int4ClassificationMapError::Admission)
        })?;
        if results.len() != 0 || next_window != windows.len() { return Err(Int4ClassificationMapError::Admission); }
        let run = Int4ClassificationMapCohortRun { schema_version: 1,
            execution: INT4_CLASSIFICATION_MAP_COHORT_EXECUTION.to_owned(),
            chunk_cohorts: windows.len(), group_steps, projection_groups, output };
        value_bytes(&run, self.limits.max_result_bytes.checked_add(COHORT_WRAPPER_BYTES)
            .ok_or(Int4ClassificationMapError::Limits)?)?;
        planning::checkpoint(control)?;
        Ok(run)
    }
}

pub(super) struct Window {
    pub start: usize,
    pub end: usize,
    pub contexts: Vec<usize>,
    pub work: Int4Work,
}
