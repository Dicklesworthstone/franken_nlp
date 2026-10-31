//! Native discovery windows feeding the unchanged bounded quote hierarchy.
use super::*;
use super::super::super::super::cohort::{self as shared, Int4SourceMapCohortBudget,
    Int4SourceMapCohortRequirement, synthesis::{self as phases, Int4SynthesisCohortRun}};
use crate::native_engine::{int4::PackedInt4Kernel, strict_int4::Int4ModelWeights};

pub type Int4SummaryHierarchyCohortRun = Int4SynthesisCohortRun<Int4SummaryHierarchyRun>;

impl PreparedInt4SummaryHierarchy<'_> {
    /// The sealed discovery map has already reserved every hierarchy pass.
    /// Native admission also covers one full scalar context before discovery.
    pub fn cohort_requirements(&self, context_limit: usize)
        -> Result<Int4SourceMapCohortRequirement, Int4SourceMapError> {
        self.base.cohort_requirements(context_limit)
    }

    /// Discovery is the only batched phase. All later quote levels retain their
    /// original complete-level witnesses, scalar execution and shared ceilings.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cohorts_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary,
        budget: Int4SourceMapCohortBudget, kernel: PackedInt4Kernel, control: &mut C)
        -> Result<Int4SummaryHierarchyCohortRun, Int4SourceMapError> {
        checkpoint(control)?;
        self.base.check_planner(planner)?;
        shared::check_budget(self.cohort_requirements(budget.context_limit)?, budget)?;
        let cap = self.max_result_bytes();
        let native_chunks = self.expected.chunk_count();
        let (map, tail) = self.into_parts();
        let mapped = map.execute_cohorts_with_control(admitted, weights, vocabulary, budget, kernel, control)?;
        let discovery = mapped.document;
        let source_windows = discovery.mapped.map_batches();
        let planned = discovery.planned_model_work;
        let actual = discovery.model_work;
        let document = tail.execute(discovery, |tail, collection, remaining, control| {
            // The full discovery driver has drained. A single scalar engine is
            // reused across every level and omitted entirely for empty evidence.
            let mut engine = phases::scalar_engine(weights, budget, kernel, control)?;
            tail.reduce(collection, remaining, planner, &mut engine, vocabulary, control)
        }, control)?;
        phases::finish(document, mapped.cohorts, native_chunks, source_windows, planned, actual, cap, control)
    }
}

#[cfg(test)] mod tests;
