//! Exact native cohorts for independent question-aware passage answers.
//!
//! Source windows include whitespace chunks. Only their real native plans
//! share decoder layers; the original question executor independently lifts
//! citations and retains every answer, abstention and whitespace source range.
use super::*;
use super::super::cohort::{self as shared, AggregateDriver, Int4SourceMapCohortBudget,
    Int4SourceMapCohortRequirement, Int4SourceMapCohortStats, COHORT_ENVELOPE_BYTES};
use crate::{
    native_engine::{int4::PackedInt4Kernel,
        strict_int4::{Int4ModelWeights, cohort::MAX_INT4_SEQUENCES}},
    tasks::source_planning::grouped_int4::cohort::{self as source_group, Int4SourceCohortRequest},
};

pub const INT4_QUESTION_COHORT_EXECUTION: &str = "grouped-int4-question-independent-passage-cohorts-v1";

#[derive(Serialize)]
pub struct Int4QuestionCohortRun {
    pub schema_version: u32,
    pub execution: &'static str,
    /// Unchanged answer semantics and complete original source partition.
    pub document: Int4QuestionRun,
    /// Native rows only: chunks equals document.native_chunks, and cohorts
    /// excludes all-whitespace windows counted by document.mapped.map_batches.
    pub cohorts: Int4SourceMapCohortStats,
}

#[derive(Clone, Copy)]
struct Schedule { source_windows: usize, native_windows: usize }

impl PreparedInt4Question<'_> {
    /// Exact componentwise peak over the nonblank native rows in each source
    /// window. Whitespace occupies reduction/output memory but no native KV.
    /// The shared static source-cohort quote remains a conservative ceiling.
    pub fn cohort_requirements(&self, context_limit: usize)
        -> Result<Int4SourceMapCohortRequirement, Int4SourceMapError> {
        self.cohort_schedule()?;
        Int4MemoryRequirement::for_context(context_limit, self.group_width())
            .map_err(Int4JsonError::from).map_err(Int4SourceError::from)?;
        let mut peak = None;
        let mut work = Int4Work::default();
        let mut next = 0_usize;
        for indices in self.plan_indices.chunks(self.mapping.reduction.map_batch_chunks) {
            let count = indices.iter().flatten().count();
            let end = next.checked_add(count).ok_or(Int4SourceMapError::WorkLimit)?;
            if count != 0 {
                let plans = self.plans.get(next..end).ok_or(Int4SourceMapError::Admission)?;
                let window = shared::window(plans, self.group_width(), context_limit)?;
                work = add_work(work, window.work).ok_or(Int4SourceMapError::WorkLimit)?;
                peak = Some(match peak {
                    None => window.requirement,
                    Some(previous) => shared::maximum(previous, window.requirement),
                });
            }
            next = end;
        }
        let masks = self.mapping.mask_visits_per_chunk.checked_mul(self.plans.len() as u64)
            .ok_or(Int4SourceMapError::WorkLimit)?;
        if next != self.plans.len() || work != self.expected.work || !fits(work, self.mapping.max_model_work)
            || masks != self.expected.masks || masks > self.mapping.max_mask_visits {
            return Err(Int4SourceMapError::WorkLimit);
        }
        peak.ok_or(Int4SourceMapError::EmptySource)
    }

    /// All sealed passage identities, options, grammar controls, vocabulary,
    /// work and masks are admitted before the first native engine allocation.
    /// The existing executor also checks every identity against these actual
    /// immutable weights before invoking any cohort. The caller retains the
    /// quoted native peak plus complete question/frontier/output reservations.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cohorts_with_control<C: DecodeStepControl>(self, admitted: &[ExecutionIdentity],
        weights: &Int4ModelWeights, vocabulary: &ExtractionVocabulary,
        budget: Int4SourceMapCohortBudget, kernel: PackedInt4Kernel, control: &mut C)
        -> Result<Int4QuestionCohortRun, Int4SourceMapError> {
        self.preflight_cohorts(admitted, vocabulary, budget, control)?;
        let schedule = self.cohort_schedule()?;
        let cap = self.max_result_bytes();
        let mut stats = Int4SourceMapCohortStats::default();
        let driver = AggregateDriver::new(weights, vocabulary, budget, kernel,
            self.mapping, &mut stats, control);
        let document = self.execute_with_driver(admitted, driver)?;
        finish(document, stats, schedule, cap, control)
    }

    fn cohort_schedule(&self) -> Result<Schedule, Int4SourceMapError> {
        let width = self.mapping.reduction.map_batch_chunks;
        if !(1..=MAX_INT4_SEQUENCES).contains(&width) || self.plans.is_empty()
            || self.plans.len() != self.expected.native_chunks
            || self.plan_indices.len() != self.chunks.chunks().len()
            || self.plan_indices.len() != self.expected.chunks
            || !self.plan_indices.iter().filter_map(|index| *index).eq(0..self.plans.len())
            || self.chunks.chunks().iter().zip(&self.plan_indices)
                .any(|(chunk, index)| has_text(chunk.text()) != index.is_some()) {
            return Err(Int4SourceMapError::Admission);
        }
        Ok(Schedule {
            source_windows: self.plan_indices.chunks(width).len(),
            native_windows: self.plan_indices.chunks(width).filter(|indices|
                indices.iter().any(Option::is_some)).count(),
        })
    }

    // Private model-free admission seam. A real immutable-weight driver still
    // validates every actual model identity before public execution can run.
    fn preflight_cohorts<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        vocabulary: &ExtractionVocabulary, budget: Int4SourceMapCohortBudget, control: &mut C)
        -> Result<(), Int4SourceMapError> {
        checkpoint(control)?;
        self.check_plans(admitted)?;
        shared::check_budget(self.cohort_requirements(budget.context_limit)?, budget)?;
        for (plan, identity) in self.plans.iter().zip(admitted) {
            checkpoint(control)?;
            let context = usize::try_from(plan.planned_work().forward_positions)
                .map_err(|_| Int4SourceMapError::Admission)?;
            source_group::preflight(&Int4SourceCohortRequest { prepared: plan, admitted_identity: identity,
                budget: shared::row_budget(plan, context, self.mapping)? }, vocabulary)?;
        }
        checkpoint(control)?;
        Ok(())
    }
}

fn finish<C: DecodeStepControl>(document: Int4QuestionRun, cohorts: Int4SourceMapCohortStats,
    schedule: Schedule, max_result_bytes: u64, control: &mut C)
    -> Result<Int4QuestionCohortRun, Int4SourceMapError> {
    if document.mapped.map_batches() != schedule.source_windows
        || schedule.native_windows > schedule.source_windows {
        return Err(Int4SourceError::InvalidResult.into());
    }
    shared::check_aggregate_stats(cohorts, document.native_chunks, schedule.native_windows,
        document.planned_model_work, document.model_work)?;
    let run = Int4QuestionCohortRun { schema_version: 1,
        execution: INT4_QUESTION_COHORT_EXECUTION, document, cohorts };
    let cap = max_result_bytes.checked_add(COHORT_ENVELOPE_BYTES).ok_or(Int4SourceMapError::WorkLimit)?;
    check_size(&run, cap).map_err(Int4SourceError::from)?;
    checkpoint(control)?;
    Ok(run)
}

#[cfg(test)]
mod tests;
