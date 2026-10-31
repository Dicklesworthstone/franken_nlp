//! Native discovery windows with one phase-owned quote-only summary pass.
use super::*;
use super::super::super::cohort::{self as shared, Int4SourceMapCohortBudget,
    Int4SourceMapCohortRequirement, synthesis::{self as phases, Int4SynthesisCohortRun}};
use crate::native_engine::{int4::PackedInt4Kernel, strict_int4::Int4ModelWeights};

pub type Int4SummarySynthesisCohortRun = Int4SynthesisCohortRun<Int4SummarySynthesisRun>;

impl PreparedInt4SummarySynthesis<'_> {
    /// Discovery cohorts and scalar synthesis never coexist physically. The
    /// shared peak nevertheless admits the final full context before any map.
    pub fn cohort_requirements(&self, context_limit: usize)
        -> Result<Int4SourceMapCohortRequirement, Int4SourceMapError> {
        if context_limit < self.planning.max_context_tokens { return Err(Int4SourceMapError::Admission); }
        self.map.cohort_requirements(context_limit)?.with_synthesis_context(
            context_limit, self.group_width(), self.budget.max_kv_bytes)
    }

    /// Batch only independent discovery chunks. Every collected verbatim quote
    /// still reaches the unchanged one-context synthesis and citation verifier;
    /// neither early EOS nor a cohort boundary renews any phase allowance.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cohorts_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        planner: &SourceTaskPlanner, weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary,
        budget: Int4SourceMapCohortBudget, kernel: PackedInt4Kernel, control: &mut C)
        -> Result<Int4SummarySynthesisCohortRun, Int4SourceMapError> {
        checkpoint(control)?;
        self.check_planner(planner)?;
        shared::check_budget(self.cohort_requirements(budget.context_limit)?, budget)?;
        let cap = self.max_result_bytes();
        let native_chunks = self.expected.chunk_count();
        let (map, tail) = self.into_parts();
        let mapped = map.execute_cohorts_with_control(admitted, weights, vocabulary, budget, kernel, control)?;
        let discovery = mapped.document;
        let source_windows = discovery.mapped.map_batches();
        let planned = discovery.planned_model_work;
        let actual = discovery.model_work;
        let mapping = tail.mapping;
        let document = tail.execute(discovery, planner, |plan, control| {
            // No scalar KV is allocated for empty evidence. Discovery engines
            // have already physically drained before this phase can allocate.
            let mut engine = phases::scalar_engine(weights, budget, kernel, control)?;
            run_final(plan, &mut engine, vocabulary, mapping, control)
        }, control)?;
        phases::finish(document, mapped.cohorts, native_chunks, source_windows, planned, actual, cap, control)
    }
}

#[cfg(test)] mod tests;
