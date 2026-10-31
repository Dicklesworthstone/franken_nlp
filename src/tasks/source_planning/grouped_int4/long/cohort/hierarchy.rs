//! Bounded native windows within a fully witnessed evidence hierarchy level.
//! Semantic limits remain task-owned; this schedule only changes the physical
//! execution of independent groups. Every engine drains before ordered lifting.
use super::*;
use crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE;

pub const INT4_HIERARCHY_COHORT_EXECUTION: &str = "grouped-int4-evidence-hierarchy-cohorts-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Int4HierarchyCohortSchedule {
    pub max_groups: usize,
    /// Independent choice: discovery may keep its original serial execution.
    pub discovery_cohorts: bool,
}
impl Int4HierarchyCohortSchedule {
    pub fn validate(self, max_passes: usize) -> Result<(), Int4SourceMapError> {
        if !(1..=256).contains(&max_passes)
            || !(1..=MAX_INT4_SEQUENCES.min(max_passes)).contains(&self.max_groups) {
            return Err(Int4SourceMapError::InvalidLimits);
        }
        Ok(())
    }

    /// Static full-context peak over sequential physical phases. No later
    /// level may enlarge the task's original per-row KV or result authority.
    pub fn requirements(self, discovery: Option<Int4SourceMapCohortRequirement>,
        context_limit: usize, group_width: usize, task_budget: TaskBudget)
        -> Result<Int4SourceMapCohortRequirement, Int4SourceMapError> {
        self.validate(256)?;
        task_budget.validate().map_err(|_| Int4SourceMapError::InvalidLimits)?;
        if discovery.is_some() != self.discovery_cohorts { return Err(Int4SourceMapError::InvalidLimits); }
        let groups = Int4SourceMapCohortMemoryLimits { max_chunks: self.max_groups,
            context_tokens: context_limit,
            max_output_tokens_per_chunk: u64::from(task_budget.max_output_tokens),
            max_result_bytes_per_chunk: task_budget.max_output_bytes }.requirements(group_width)?
            .with_synthesis_context(context_limit, group_width, task_budget.max_kv_bytes)?;
        Ok(match discovery { Some(discovery) => maximum(discovery, groups), None => groups })
    }

    /// The existing owner already prices one complete compiled plan. Each
    /// additional simultaneous plan receives another full declared preparation
    /// owner. This conservative admission reserve is not a measured RSS claim.
    pub fn additional_preparation_bytes(self, existing_full_owner_reserve: u64)
        -> Result<u64, Int4SourceMapError> {
        self.validate(256)?;
        if existing_full_owner_reserve == 0 { return Err(Int4SourceMapError::InvalidLimits); }
        existing_full_owner_reserve.checked_mul((self.max_groups - 1) as u64)
            .ok_or(Int4SourceMapError::WorkLimit)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Int4HierarchyCohortStats {
    pub cohorts: u64,
    pub passes: u64,
    pub group_steps: u64,
    pub projection_groups: u64,
    pub planned_work: Int4Work,
    pub model_work: Int4Work,
}

#[derive(Serialize)]
pub struct Int4HierarchyCohortRun<T> {
    pub schema_version: u32,
    pub execution: &'static str,
    pub document: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub discovery_cohorts: Option<Int4SourceMapCohortStats>,
    pub hierarchy_cohorts: Int4HierarchyCohortStats,
}

/// Admission only: no native state is allocated while witnessing a level.
pub(in crate::tasks::source_planning::grouped_int4::long)
fn preflight_plan<C: DecodeStepControl>(plan: &PreparedInt4SourceTask, weights: &Int4ModelWeights,
    vocabulary: &ExtractionVocabulary, budget: Int4SourceMapCohortBudget,
    mapping: Int4SourceMapLimits, control: &mut C) -> Result<(), Int4SourceMapError> {
    checkpoint(control)?;
    let identity = plan.execution_identity();
    plan.verify_identity(identity)?;
    constrained_int4::check_model(identity, weights.identity()).map_err(Int4SourceError::from)?;
    let actual = window(std::slice::from_ref(plan), weights.group_width(), budget.context_limit)?;
    check_budget(actual.requirement, budget)?;
    source_group::preflight(&Int4SourceCohortRequest { prepared: plan, admitted_identity: identity,
        budget: row_budget(plan, actual.contexts[0], mapping)? }, vocabulary)?;
    checkpoint(control)?;
    Ok(())
}

/// Recheck all rows before allocating the cohort. Original task finalizers run
/// while its KV is exclusively owned; a failure discards the complete window.
#[allow(clippy::too_many_arguments)]
pub(in crate::tasks::source_planning::grouped_int4::long)
fn execute_window<C: DecodeStepControl>(plans: &[PreparedInt4SourceTask], weights: &Int4ModelWeights,
    vocabulary: &ExtractionVocabulary, budget: Int4SourceMapCohortBudget, kernel: PackedInt4Kernel,
    mapping: Int4SourceMapLimits, control: &mut C) -> Result<Int4SourceCohortRun, Int4SourceMapError> {
    checkpoint(control)?;
    check_budget(window(plans, weights.group_width(), budget.context_limit)?.requirement, budget)?;
    let mut admitted = Vec::new();
    admitted.try_reserve_exact(plans.len()).map_err(|_| Int4SourceMapError::Allocation)?;
    for plan in plans {
        preflight_plan(plan, weights, vocabulary, budget, mapping, control)?;
        admitted.push(plan.execution_identity().clone());
    }
    let run = Native { weights, vocabulary, budget, kernel }.execute(plans, &admitted, mapping, control)?;
    validate_group(plans, &run)?;
    checkpoint(control)?;
    Ok(run)
}

pub(in crate::tasks::source_planning::grouped_int4::long)
fn accumulate(previous: Int4HierarchyCohortStats, run: &Int4SourceCohortRun)
    -> Result<Int4HierarchyCohortStats, Int4SourceMapError> {
    let add = |a: u64, b: u64| a.checked_add(b).ok_or(Int4SourceError::InvalidResult);
    Ok(Int4HierarchyCohortStats { cohorts: add(previous.cohorts, 1)?,
        passes: add(previous.passes, run.sequences.len() as u64)?,
        group_steps: add(previous.group_steps, run.group_steps)?,
        projection_groups: add(previous.projection_groups, run.projection_groups)?,
        planned_work: add_work(previous.planned_work, run.planned_work).ok_or(Int4SourceError::InvalidResult)?,
        model_work: add_work(previous.model_work, run.model_work).ok_or(Int4SourceError::InvalidResult)? })
}

pub(in crate::tasks::source_planning::grouped_int4::long)
fn check_stats(stats: Int4HierarchyCohortStats, cohorts: usize, passes: usize,
    planned: Int4Work, actual: Int4Work) -> Result<(), Int4SourceMapError> {
    if stats.cohorts != cohorts as u64 || stats.passes != passes as u64
        || stats.planned_work != planned || stats.model_work != actual || !fits(actual, planned)
        || (passes == 0 && stats != Int4HierarchyCohortStats::default())
        || (passes != 0 && (stats.cohorts == 0 || stats.cohorts > stats.passes
            || stats.group_steps == 0 || stats.group_steps > actual.forward_positions
            || stats.projection_groups == 0 || stats.projection_groups > stats.group_steps
            || stats.projection_groups > actual.projected_logits / NANBEIGE_VOCAB_SIZE as u64)) {
        return Err(Int4SourceError::InvalidResult.into());
    }
    Ok(())
}

pub(in crate::tasks::source_planning::grouped_int4::long)
fn finish<T: Serialize, C: DecodeStepControl>(document: T, discovery: Option<Int4SourceMapCohortStats>,
    stats: Int4HierarchyCohortStats, cap: u64, control: &mut C)
    -> Result<Int4HierarchyCohortRun<T>, Int4SourceMapError> {
    checkpoint(control)?;
    let run = Int4HierarchyCohortRun { schema_version: 1, execution: INT4_HIERARCHY_COHORT_EXECUTION,
        document, discovery_cohorts: discovery, hierarchy_cohorts: stats };
    let cap = cap.checked_add(COHORT_ENVELOPE_BYTES).ok_or(Int4SourceMapError::WorkLimit)?;
    check_size(&run, cap).map_err(Int4SourceError::from)?;
    checkpoint(control)?;
    Ok(run)
}

#[cfg(test)] mod tests;
