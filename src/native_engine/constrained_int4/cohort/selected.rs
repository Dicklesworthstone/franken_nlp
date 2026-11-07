//! Explicit all-legal-row heads over the common structured cohort scheduler.
//! Limits align with input order and retain each row's own declared ceiling.
//! A complete legal set above its ceiling refuses before that row's first
//! projection; no truncation, forced-token shortcut or dense retry is allowed.
use super::*;
use super::super::sparse::Int4JsonSparseLimits;

pub fn decode_json_int4_cohort_selected<C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4JsonCohortRequest<'_>], vocabulary: &VocabMaskOracle,
    budget: Int4JsonCohortBudget, limits: &[Int4JsonSparseLimits], control: &mut C)
    -> Result<Int4JsonCohortRun, Int4JsonError> {
    execute_with_selected(engine, requests, vocabulary, budget, limits, control, Ok)
}

/// The caller holds the same cohort/workspace reservations as full heads plus
/// exactly the packed scratch from `prefill.validate()`. Selected projection
/// uses one live mask and bounded head chunks inside that native envelope.
pub fn decode_json_int4_cohort_selected_packed<C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4JsonCohortRequest<'_>], vocabulary: &VocabMaskOracle,
    budget: Int4JsonCohortBudget, limits: &[Int4JsonSparseLimits], prefill: Int4PrefillLimits,
    control: &mut C) -> Result<Int4JsonCohortRun, Int4JsonError> {
    execute_with_selected_packed(engine, requests, vocabulary, budget, limits, prefill, control, Ok)
}

pub(crate) fn execute_with_selected<C, T, E, F>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4JsonCohortRequest<'_>], vocabulary: &VocabMaskOracle,
    budget: Int4JsonCohortBudget, limits: &[Int4JsonSparseLimits], control: &mut C, finalize: F) -> Result<T, E>
where C: DecodeStepControl, E: From<Int4JsonError>, F: FnOnce(Int4JsonCohortRun) -> Result<T, E> {
    execute_scheduled(engine, requests, vocabulary, budget, Projection::Selected(limits), None, control, finalize)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_with_selected_packed<C, T, E, F>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4JsonCohortRequest<'_>], vocabulary: &VocabMaskOracle,
    budget: Int4JsonCohortBudget, limits: &[Int4JsonSparseLimits], prefill: Int4PrefillLimits,
    control: &mut C, finalize: F) -> Result<T, E>
where C: DecodeStepControl, E: From<Int4JsonError>, F: FnOnce(Int4JsonCohortRun) -> Result<T, E> {
    execute_scheduled(engine, requests, vocabulary, budget, Projection::Selected(limits), Some(prefill), control, finalize)
}
