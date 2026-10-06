//! Identity-sealed extraction cohorts over real shared-layer native execution.
//! Per-row semantic identities do not depend on physical slots. The outer
//! version records grouping; it does not claim batch/scalar model parity.
use super::*;
use crate::native_engine::{portable_int8::batch::MAX_BATCH_ROWS,
    strict_int8::cohort::Int8CohortEngine,
    constrained_int8::sparse::cohort::{self as native, Int8JsonCohortBudget,
        Int8JsonCohortRequest, Int8JsonCohortRun, INT8_JSON_COHORT_EXECUTION}};

pub const INT8_EXTRACT_COHORT_EXECUTION: &str = "strict-int8-selected-extraction-cohort-v1";

pub struct Int8ExtractCohortRequest<'a> {
    pub prepared: &'a Int8ExtractPlan,
    pub admitted_identity: &'a ExecutionIdentity,
    pub budget: Int8JsonBudget,
}
#[derive(Clone, Copy, Debug)]
pub struct Int8ExtractCohortBudget {
    pub decode: Int8JsonCohortBudget,
    /// Whole typed task envelope, including every source occurrence and work.
    pub max_result_bytes: u64,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8ExtractCohortRun {
    pub schema_version: u32,
    pub execution: String,
    pub sequences: Vec<Int8ExtractRun>,
    pub group_steps: u64,
    pub planned_work: Int8Work,
    pub model_work: Int8Work,
}
impl Int8ExtractPlan {
    /// Exact task authority, not inferred from this invocation's live prefix.
    pub fn max_kv_bytes(&self) -> u64 { self.extraction.max_kv_bytes }
}
fn check_request(request: &Int8ExtractCohortRequest<'_>, vocabulary: &ExtractionVocabulary)
    -> Result<(), Int8ExtractError> {
    request.prepared.verify_identity(request.admitted_identity)?;
    if request.prepared.selected_rows().is_none() { return Err(Int8ExtractError::Identity); }
    if vocabulary.controls != request.prepared.extraction.controls {
        return Err(ExtractError::Contract("cohort vocabulary control registry differs from sealed plan").into());
    }
    Ok(())
}
pub fn execute<C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>,
    requests: &[Int8ExtractCohortRequest<'_>], vocabulary: &ExtractionVocabulary,
    budget: Int8ExtractCohortBudget, control: &mut C) -> Result<Int8ExtractCohortRun, Int8ExtractError> {
    execute_with(engine, requests, vocabulary, budget, control, Ok)
}
/// Built-in source-task finalization composes here without exposing a public
/// callback or an unchecked wire-result constructor. All finalizers run before
/// the real native session closes; any failure poisons/drains the entire group.
pub(crate) fn execute_with<C, T, E, F>(engine: &mut Int8CohortEngine<'_>,
    requests: &[Int8ExtractCohortRequest<'_>], vocabulary: &ExtractionVocabulary,
    budget: Int8ExtractCohortBudget, control: &mut C, finalize: F) -> Result<T, E>
where C: DecodeStepControl, E: From<Int8ExtractError> + From<Int8JsonError>,
    F: FnOnce(Int8ExtractCohortRun) -> Result<T, E> {
    if requests.is_empty() || requests.len() > MAX_BATCH_ROWS || budget.max_result_bytes == 0 {
        return Err(E::from(Int8ExtractError::from(ExtractError::Contract("invalid extraction cohort bounds"))));
    }
    for request in requests { check_request(request, vocabulary).map_err(E::from)?; }
    let mut rows = Vec::new(); rows.try_reserve_exact(requests.len())
        .map_err(|_| E::from(Int8ExtractError::from(ExtractError::AllocationRefused)))?;
    for request in requests {
        let plan = request.prepared;
        let mut work = request.budget;
        work.json.max_kv_bytes = work.json.max_kv_bytes.min(plan.max_kv_bytes());
        rows.push(Int8JsonCohortRequest { identity: request.admitted_identity,
            prompt: &plan.extraction.prompt, program: &plan.extraction.program, options: plan.options(),
            budget: work, limits: plan.selected_rows().ok_or_else(|| E::from(Int8ExtractError::Identity))? });
    }
    native::execute_with(engine, &rows, &vocabulary.oracle, budget.decode, control,
        |run| finalize(finish(requests, run, budget.max_result_bytes).map_err(E::from)?))
}
/// Closed composition check, not authentication of arbitrary deserialized work.
pub(super) fn finish(requests: &[Int8ExtractCohortRequest<'_>], raw: Int8JsonCohortRun, cap: u64)
    -> Result<Int8ExtractCohortRun, Int8ExtractError> {
    if raw.schema_version != 1 || raw.execution != INT8_JSON_COHORT_EXECUTION
        || raw.sequences.len() != requests.len() || requests.is_empty() {
        return Err(ExtractError::InvalidResult.into());
    }
    let mut expected = Int8Work::default();
    for request in requests {
        if request.prepared.selected_rows().is_none() { return Err(Int8ExtractError::Identity); }
        expected = expected.checked_add(request.prepared.planned_work()).map_err(Int8JsonError::from)?;
    }
    if raw.planned_work != expected { return Err(Int8JsonError::WorkMismatch.into()); }
    let mut sequences = Vec::new(); sequences.try_reserve_exact(requests.len()).map_err(|_| ExtractError::AllocationRefused)?;
    let mut model_work = Int8Work::default(); let mut longest = 0;
    for (request, sequence) in requests.iter().zip(raw.sequences) {
        let result = request.prepared.finalize(sequence)?;
        model_work = model_work.checked_add(result.model_work).map_err(Int8JsonError::from)?;
        longest = longest.max(result.model_work.forward_positions);
        sequences.push(result);
    }
    if raw.model_work != model_work || raw.group_steps != longest { return Err(Int8JsonError::WorkMismatch.into()); }
    let output = Int8ExtractCohortRun { schema_version: 1, execution: INT8_EXTRACT_COHORT_EXECUTION.to_owned(),
        sequences, group_steps: raw.group_steps, planned_work: expected, model_work };
    check_size(&output, cap)?;
    Ok(output)
}
#[cfg(test)] mod tests;
