//! Native windows within fully admitted question hierarchy levels. Only
//! verified original quotations cross a level boundary; answers never do.
use super::*;
use crate::{
    native_engine::{int4::PackedInt4Kernel, strict_int4::Int4ModelWeights},
    tasks::source_planning::grouped_int4::long::cohort::{self as shared,
        Int4SourceMapCohortBudget, Int4SourceMapCohortRequirement, Int4SourceMapCohortStats,
        hierarchy::{self as groups, Int4HierarchyCohortSchedule, Int4HierarchyCohortStats},
        synthesis as staged},
};

pub type Int4QuestionHierarchyGroupedRun = groups::Int4HierarchyCohortRun<Int4QuestionHierarchyRun>;

impl PreparedInt4QuestionHierarchy<'_> {
    /// Complete physical peak before discovery: its selected schedule plus a
    /// full declared window of future hierarchy contexts. Per-row task KV and
    /// the original all-pass work/mask reserves are never enlarged.
    pub fn hierarchy_cohort_requirements(&self, context_limit: usize, schedule: Int4HierarchyCohortSchedule)
        -> Result<Int4SourceMapCohortRequirement, Int4SourceMapError> {
        schedule.validate(self.expected.limits.max_passes)?;
        if context_limit < self.base.planning.max_context_tokens { return Err(Int4SourceMapError::Admission); }
        let discovery = if schedule.discovery_cohorts { Some(self.base.map.cohort_requirements(context_limit)?) } else { None };
        schedule.requirements(discovery, context_limit, self.group_width(), self.base.budget)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_hierarchy_cohorts_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary,
        schedule: Int4HierarchyCohortSchedule, budget: Int4SourceMapCohortBudget,
        kernel: PackedInt4Kernel, control: &mut C) -> Result<Int4QuestionHierarchyGroupedRun, Int4SourceMapError> {
        checkpoint(control)?;
        self.base.check_planner(planner)?;
        shared::check_budget(self.hierarchy_cohort_requirements(budget.context_limit, schedule)?, budget)?;
        if weights.group_width() != self.group_width() || self.execution_identities().len() != admitted.len() {
            return Err(Int4SourceMapError::Admission);
        }
        for (expected, actual) in self.execution_identities().zip(admitted) {
            checkpoint(control)?;
            if expected != actual { return Err(Int4SourceMapError::Admission); }
            constrained_int4::check_model(actual, weights.identity()).map_err(Int4SourceError::from)?;
        }
        let cap = self.max_result_bytes();
        let (map, continuation) = self.into_phases();
        let (discovery, discovery_cohorts) = if schedule.discovery_cohorts {
            let run = map.execute_cohorts_with_control(admitted, weights, vocabulary, budget, kernel, control)?;
            (run.document, Some(run.cohorts))
        } else {
            // This allocation is scoped to discovery and drains before any
            // compiled hierarchy window or cohort engine is constructed.
            let mut engine = staged::scalar_engine(weights, budget, kernel, control)?;
            (map.execute_with_control(admitted, &mut engine, vocabulary, control)?, None)
        };
        let mut hierarchy_cohorts = Int4HierarchyCohortStats::default();
        let document = continuation.execute(discovery, control, |continuation, collection, remaining, control| {
            let base = &continuation.base;
            let mut driver = execution::Native { planner, identity: &base.identity, question: &base.question,
                budget: base.budget, planning: base.planning, options: base.options, mapping: continuation.mapping,
                engine: execution::Engine::Cohorts { weights, budget, kernel, max_groups: schedule.max_groups,
                    stats: &mut hierarchy_cohorts }, vocabulary, expected: continuation.expected };
            execution::reduce(base.source, collection, continuation.expected, cap, remaining, &mut driver, control)
        })?;
        finish(document, discovery_cohorts, hierarchy_cohorts, schedule, cap, control)
    }
}

pub(super) fn finish<C: DecodeStepControl>(document: Int4QuestionHierarchyRun, discovery: Option<Int4SourceMapCohortStats>,
    hierarchy: Int4HierarchyCohortStats, schedule: Int4HierarchyCohortSchedule, cap: u64, control: &mut C)
    -> Result<Int4QuestionHierarchyGroupedRun, Int4SourceMapError> {
    schedule.validate(document.limits.max_passes)?;
    if discovery.is_some() != schedule.discovery_cohorts { return Err(Int4SourceMapError::Admission); }
    if let Some(stats) = discovery {
        staged::check_discovery_stats(stats, document.discovery.native_chunks, document.discovery.mapped.map_batches(),
            document.discovery.planned_model_work, document.discovery.model_work)?;
    }
    let mut cohorts = 0_usize;
    for level in &document.levels {
        cohorts = cohorts.checked_add(level.passes.len().div_ceil(schedule.max_groups))
            .ok_or(Int4SourceMapError::WorkLimit)?;
    }
    let mut planned = Int4Work::default(); let mut actual = Int4Work::default();
    for pass in &document.passes {
        checkpoint(control)?;
        planned = add_work(planned, pass.planned_model_work).ok_or(Int4SourceMapError::WorkLimit)?;
        actual = add_work(actual, pass.native.model_work).ok_or(Int4SourceMapError::WorkLimit)?;
    }
    groups::check_stats(hierarchy, cohorts, document.passes.len(), planned, actual)?;
    groups::finish(document, discovery, hierarchy, cap, control)
}
