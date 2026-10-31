//! Native discovery cohorts followed by one scalar answer over verified quotes.
//! Discovery engines drain before synthesis, so admission retains the larger
//! native peak together with the original whole-document evidence allowance.
use super::*;
use super::super::super::cohort::{self as shared, synthesis as staged,
    Int4SourceMapCohortBudget, Int4SourceMapCohortRequirement, Int4SourceMapCohortStats};
use crate::native_engine::{int4::PackedInt4Kernel, strict_int4::Int4ModelWeights};

pub type Int4QuestionSynthesisCohortRun = staged::Int4SynthesisCohortRun<Int4QuestionSynthesisRun>;

impl PreparedInt4QuestionSynthesis<'_> {
    /// Exact discovery peak combined with one full synthesis context. The
    /// final pass remains reserved even when discovery later yields no quotes.
    pub fn cohort_requirements(&self, context_limit: usize)
        -> Result<Int4SourceMapCohortRequirement, Int4SourceMapError> {
        if context_limit < self.planning.max_context_tokens { return Err(Int4SourceMapError::Admission); }
        self.map.cohort_requirements(context_limit)?
            .with_synthesis_context(context_limit, self.group_width(), self.max_kv_bytes())
    }

    /// Every discovery row and the full scalar continuation are admitted
    /// before native allocation. Only discovery shares decoder layers; the
    /// final answer uses the existing evidence-only source and receipt checks.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cohorts_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary,
        budget: Int4SourceMapCohortBudget, kernel: PackedInt4Kernel, control: &mut C)
        -> Result<Int4QuestionSynthesisCohortRun, Int4SourceMapError> {
        self.preflight_cohorts(planner, budget, control)?;
        let cap = self.max_result_bytes();
        let mapping = self.mapping;
        let (map, continuation) = self.into_phases();
        let discovery = map.execute_cohorts_with_control(admitted, weights, vocabulary, budget, kernel, control)?;
        let document = continuation.execute(discovery.document, planner, control, |plan, control| {
            // No evidence never invokes this closure or allocates scalar KV.
            let mut engine = staged::scalar_engine(weights, budget, kernel, control)?;
            synthesis_pass(plan, &mut engine, vocabulary, mapping, control)
        })?;
        finish(document, discovery.cohorts, cap, control)
    }

    // Model-free phase admission. The sealed discovery executor separately
    // checks every actual identity against the immutable native weights.
    fn preflight_cohorts<C: DecodeStepControl>(&self, planner: &SourceTaskPlanner,
        budget: Int4SourceMapCohortBudget, control: &mut C) -> Result<(), Int4SourceMapError> {
        checkpoint(control)?;
        self.check_planner(planner)?;
        shared::check_budget(self.cohort_requirements(budget.context_limit)?, budget)?;
        checkpoint(control)?;
        Ok(())
    }
}

fn finish<C: DecodeStepControl>(document: Int4QuestionSynthesisRun, stats: Int4SourceMapCohortStats,
    cap: u64, control: &mut C) -> Result<Int4QuestionSynthesisCohortRun, Int4SourceMapError> {
    let discovery = &document.discovery;
    let (native_chunks, source_windows, planned, actual) = (discovery.native_chunks,
        discovery.mapped.map_batches(), discovery.planned_model_work, discovery.model_work);
    staged::finish(document, stats, native_chunks, source_windows, planned, actual, cap, control)
}

#[cfg(test)]
mod tests;
