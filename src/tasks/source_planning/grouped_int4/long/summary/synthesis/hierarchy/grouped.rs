//! Native windows of independent groups within a fully authenticated level.
//! Discovery scheduling is explicit and physically precedes every reduction
//! window; both paths retain the original complete hierarchy document.
use super::*;
use crate::{
    native_engine::{int4::PackedInt4Kernel, strict_int4::Int4ModelWeights},
    tasks::source_planning::grouped_int4::long::cohort::{self as shared,
        Int4SourceMapCohortBudget, Int4SourceMapCohortRequirement,
        hierarchy::{self as physical, Int4HierarchyCohortSchedule, Int4HierarchyCohortStats,
            Int4HierarchyCohortRun}, synthesis::{check_discovery_stats, scalar_engine}},
};

pub type Int4SummaryHierarchyGroupedRun = Int4HierarchyCohortRun<Int4SummaryHierarchyRun>;

impl PreparedInt4SummaryHierarchy<'_> {
    /// Reserve the complete declared reduction window before discovery, even
    /// when evidence later fits one group or no evidence survives discovery.
    pub fn hierarchy_cohort_requirements(&self, context_limit: usize, schedule: Int4HierarchyCohortSchedule)
        -> Result<Int4SourceMapCohortRequirement, Int4SourceMapError> {
        schedule.validate(self.expected.limits.max_passes)?;
        if context_limit < self.base.planning.max_context_tokens { return Err(Int4SourceMapError::Admission); }
        let discovery = if schedule.discovery_cohorts {
            Some(self.base.map.cohort_requirements(context_limit)?)
        } else { None };
        schedule.requirements(discovery, context_limit, self.group_width(), self.base.budget)
    }

    /// Execute whole levels in bounded windows after all of that level's
    /// groups have been compiled and authenticated. A group's original task,
    /// causal KV, pass allowance and source verifier never become shared state.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_hierarchy_cohorts_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary,
        schedule: Int4HierarchyCohortSchedule, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, control: &mut C) -> Result<Int4SummaryHierarchyGroupedRun, Int4SourceMapError> {
        checkpoint(control)?;
        self.base.check_planner(planner)?;
        shared::check_budget(self.hierarchy_cohort_requirements(budget.context_limit, schedule)?, budget)?;
        if weights.group_width() != self.group_width() || admitted.len() != self.execution_identities().len() {
            return Err(Int4SourceMapError::Admission);
        }
        for (expected, identity) in self.execution_identities().zip(admitted) {
            checkpoint(control)?;
            if expected != identity { return Err(Int4SourceMapError::Admission); }
            constrained_int4::check_model(identity, weights.identity()).map_err(Int4SourceError::from)?;
        }
        let cap = self.max_result_bytes();
        let native_chunks = self.expected.chunk_count();
        let (map, tail) = self.into_parts();
        let (discovery, discovery_cohorts) = if schedule.discovery_cohorts {
            let run = map.execute_cohorts_with_control(admitted, weights, vocabulary, budget, kernel, control)?;
            (run.document, Some(run.cohorts))
        } else {
            let mut engine = scalar_engine(weights, budget, kernel, control)?;
            let run = map.execute_with_control(admitted, &mut engine, vocabulary, control)?;
            // No discovery engine or session can overlap a reduction window.
            drop(engine);
            (run, None)
        };
        if let Some(stats) = discovery_cohorts {
            check_discovery_stats(stats, native_chunks, discovery.mapped.map_batches(),
                discovery.planned_model_work, discovery.model_work)?;
        }
        let mut hierarchy_cohorts = Int4HierarchyCohortStats::default();
        let document = tail.execute(discovery, |tail, collection, remaining, control| {
            tail.reduce_with_execution(collection, remaining, planner, vocabulary,
                execution::Execution::Cohorts { weights, budget, kernel, max_groups: schedule.max_groups,
                    stats: &mut hierarchy_cohorts }, control)
        }, control)?;
        check_completed(&document, schedule, hierarchy_cohorts)?;
        physical::finish(document, discovery_cohorts, hierarchy_cohorts, cap, control)
    }
}

fn check_completed(document: &Int4SummaryHierarchyRun, schedule: Int4HierarchyCohortSchedule,
    stats: Int4HierarchyCohortStats) -> Result<(), Int4SourceMapError> {
    schedule.validate(document.limits.max_passes)?;
    let mut cohorts = 0_usize;
    for level in &document.levels {
        cohorts = cohorts.checked_add(level.passes.len().div_ceil(schedule.max_groups))
            .ok_or(Int4SourceMapError::WorkLimit)?;
    }
    let mut planned = Int4Work::default(); let mut actual = Int4Work::default();
    for pass in &document.passes {
        planned = add_work(planned, pass.planned_model_work).ok_or(Int4SourceMapError::WorkLimit)?;
        actual = add_work(actual, pass.native.model_work).ok_or(Int4SourceMapError::WorkLimit)?;
    }
    physical::check_stats(stats, cohorts, document.passes.len(), planned, actual)
}

#[cfg(test)] mod tests;
