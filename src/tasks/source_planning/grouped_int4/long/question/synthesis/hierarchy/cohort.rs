//! Native question-discovery cohorts followed by bounded scalar quote selection.
//! The original hierarchy retains every pass, transcript and source citation;
//! its whole-request reservations are never renewed at a smaller frontier.
use super::*;
use super::super::super::super::cohort::{self as shared, synthesis as staged,
    Int4SourceMapCohortBudget, Int4SourceMapCohortRequirement, Int4SourceMapCohortStats};
use crate::native_engine::{int4::PackedInt4Kernel, strict_int4::Int4ModelWeights};

pub type Int4QuestionHierarchyCohortRun = staged::Int4SynthesisCohortRun<Int4QuestionHierarchyRun>;

impl PreparedInt4QuestionHierarchy<'_> {
    /// Discovery cohorts and the scalar hierarchy execute in distinct phases.
    /// Quote their componentwise peak, including a complete scalar context
    /// even when each discovery row has a much shorter actual prompt.
    pub fn cohort_requirements(&self, context_limit: usize)
        -> Result<Int4SourceMapCohortRequirement, Int4SourceMapError> {
        if context_limit < self.base.planning.max_context_tokens { return Err(Int4SourceMapError::Admission); }
        self.base.map.cohort_requirements(context_limit)?
            .with_synthesis_context(context_limit, self.group_width(), self.max_kv_bytes())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_cohorts_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary,
        budget: Int4SourceMapCohortBudget, kernel: PackedInt4Kernel, control: &mut C)
        -> Result<Int4QuestionHierarchyCohortRun, Int4SourceMapError> {
        self.preflight_cohorts(planner, budget, control)?;
        let cap = self.max_result_bytes();
        let (map, continuation) = self.into_phases();
        let discovery = map.execute_cohorts_with_control(admitted, weights, vocabulary, budget, kernel, control)?;
        let document = continuation.execute(discovery.document, control,
            |continuation, collection, remaining, control| {
                // Allocate once for all hierarchy levels, after discovery has
                // drained. No-evidence results never enter this closure.
                let mut engine = staged::scalar_engine(weights, budget, kernel, control)?;
                continuation.reduce(collection, remaining, planner, &mut engine, vocabulary, control)
            })?;
        finish(document, discovery.cohorts, cap, control)
    }

    fn preflight_cohorts<C: DecodeStepControl>(&self, planner: &SourceTaskPlanner,
        budget: Int4SourceMapCohortBudget, control: &mut C) -> Result<(), Int4SourceMapError> {
        checkpoint(control)?;
        self.base.check_planner(planner)?;
        shared::check_budget(self.cohort_requirements(budget.context_limit)?, budget)?;
        checkpoint(control)?;
        Ok(())
    }
}

fn finish<C: DecodeStepControl>(document: Int4QuestionHierarchyRun, stats: Int4SourceMapCohortStats,
    cap: u64, control: &mut C) -> Result<Int4QuestionHierarchyCohortRun, Int4SourceMapError> {
    let discovery = &document.discovery;
    let (native_chunks, source_windows, planned, actual) = (discovery.native_chunks,
        discovery.mapped.map_batches(), discovery.planned_model_work, discovery.model_work);
    staged::finish(document, stats, native_chunks, source_windows, planned, actual, cap, control)
}

#[cfg(test)]
mod tests;
