//! Independent schema/source plans sharing one actual INT4 decoder cohort.
//! Original full-vocabulary task identities and results are unchanged; only
//! this outer envelope describes physical grouping. No parity claim is made.
use super::*;
use crate::native_engine::{strict_int4::cohort::{Int4CohortEngine, MAX_INT4_SEQUENCES},
    constrained_int4::cohort::{self as native, Int4JsonCohortRequest, Int4JsonCohortBudget,
        Int4JsonCohortRun, INT4_JSON_COHORT_EXECUTION}};

pub const INT4_EXTRACT_COHORT_EXECUTION: &str = "grouped-int4-schema-source-extraction-cohort-v1";

pub struct Int4ExtractCohortRequest<'a> {
    pub prepared: &'a Int4ExtractPlan,
    pub admitted_identity: &'a ExecutionIdentity,
    pub budget: Int4JsonBudget,
}
#[derive(Clone, Copy, Debug)]
pub struct Int4ExtractCohortBudget {
    pub decode: Int4JsonCohortBudget,
    /// Whole typed envelope, including every independent source occurrence.
    pub max_result_bytes: u64,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4ExtractCohortRun {
    pub schema_version: u32,
    pub execution: String,
    pub sequences: Vec<Int4ExtractRun>,
    pub group_steps: u64,
    pub projection_groups: u64,
    pub planned_work: Int4Work,
    pub model_work: Int4Work,
}

fn check_request(request: &Int4ExtractCohortRequest<'_>, vocabulary: &ExtractionVocabulary)
    -> Result<(), Int4ExtractError> {
    request.prepared.verify_identity(request.admitted_identity)?;
    if vocabulary.controls != request.prepared.extraction.controls {
        return Err(ExtractError::Contract("cohort vocabulary control registry differs from sealed INT4 plan").into());
    }
    Ok(())
}
/// The batch boundary can reject an impossible row before enqueueing it or
/// allocating native contexts. This checks exact token/control and work facts;
/// the actual resident model and physical KV capacities remain native checks.
pub(crate) fn preflight(request: &Int4ExtractCohortRequest<'_>, vocabulary: &ExtractionVocabulary)
    -> Result<(), Int4ExtractError> {
    check_request(request, vocabulary)?;
    let work = native::preflight_request(request.prepared.group_width(),
        &native_request(request), &vocabulary.oracle)?;
    if work != request.prepared.planned_work() { return Err(Int4JsonError::WorkMismatch.into()); }
    Ok(())
}
pub fn execute<C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4ExtractCohortRequest<'_>], vocabulary: &ExtractionVocabulary,
    budget: Int4ExtractCohortBudget, control: &mut C) -> Result<Int4ExtractCohortRun, Int4ExtractError> {
    execute_with(engine, requests, vocabulary, budget, control, Ok)
}
/// Closed composition for the shipped source-task finalizers. Native ownership
/// remains live through extraction, source semantics, output accounting and
/// the final checkpoint; a later-row failure publishes no partial collection.
pub(crate) fn execute_with<C, T, E, F>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4ExtractCohortRequest<'_>], vocabulary: &ExtractionVocabulary,
    budget: Int4ExtractCohortBudget, control: &mut C, finalize: F) -> Result<T, E>
where C: DecodeStepControl, E: From<Int4ExtractError> + From<Int4JsonError>,
    F: FnOnce(Int4ExtractCohortRun) -> Result<T, E> {
    if requests.is_empty() || requests.len() > MAX_INT4_SEQUENCES || budget.max_result_bytes == 0 {
        return Err(E::from(Int4ExtractError::from(ExtractError::Contract("invalid INT4 extraction cohort bounds"))));
    }
    for request in requests { preflight(request, vocabulary).map_err(E::from)?; }
    let mut rows = Vec::new(); rows.try_reserve_exact(requests.len())
        .map_err(|_| E::from(Int4ExtractError::from(ExtractError::AllocationRefused)))?;
    for request in requests { rows.push(native_request(request)); }
    native::execute_with(engine, &rows, &vocabulary.oracle, budget.decode, control,
        |run| finalize(finish(requests, run, budget.max_result_bytes).map_err(E::from)?))
}
fn native_request<'a>(request: &Int4ExtractCohortRequest<'a>) -> Int4JsonCohortRequest<'a> {
    let plan = request.prepared;
    let mut budget = request.budget;
    budget.json.max_kv_bytes = budget.json.max_kv_bytes.min(plan.max_kv_bytes());
    Int4JsonCohortRequest { identity: request.admitted_identity,
        prompt: &plan.extraction.prompt, program: &plan.extraction.program,
        options: plan.options(), budget }
}

/// Internal reconciliation of an actual native result; never an authentication
/// API for deserialized scores or a substitute for native execution.
pub(super) fn finish(requests: &[Int4ExtractCohortRequest<'_>], raw: Int4JsonCohortRun, cap: u64)
    -> Result<Int4ExtractCohortRun, Int4ExtractError> {
    if raw.schema_version != 1 || raw.execution != INT4_JSON_COHORT_EXECUTION
        || raw.sequences.len() != requests.len() || requests.is_empty() || requests.len() > MAX_INT4_SEQUENCES {
        return Err(ExtractError::InvalidResult.into());
    }
    let mut expected = Int4Work::default();
    for request in requests {
        request.prepared.verify_identity(request.admitted_identity)?;
        if request.prepared.group_width() != requests[0].prepared.group_width() { return Err(Int4ExtractError::Identity); }
        expected = expected.checked_add(request.prepared.planned_work()).map_err(Int4JsonError::from)?;
    }
    if raw.planned_work != expected { return Err(Int4JsonError::WorkMismatch.into()); }
    let mut sequences = Vec::new(); sequences.try_reserve_exact(requests.len()).map_err(|_| ExtractError::AllocationRefused)?;
    let mut model_work = Int4Work::default(); let mut longest = 0;
    let mut all_projections = 0_u64; let mut longest_projection_sequence = 0;
    for (request, sequence) in requests.iter().zip(raw.sequences) {
        let result = request.prepared.finalize(sequence)?;
        model_work = model_work.checked_add(result.model_work).map_err(Int4JsonError::from)?;
        longest = longest.max(result.model_work.forward_positions);
        let projections = u64::try_from(result.result.output.token_ids.len()).map_err(|_| Int4JsonError::WorkMismatch)?;
        all_projections = all_projections.checked_add(projections).ok_or(Int4JsonError::WorkMismatch)?;
        longest_projection_sequence = longest_projection_sequence.max(projections);
        sequences.push(result);
    }
    if raw.model_work != model_work || raw.group_steps != longest
        || raw.projection_groups < longest_projection_sequence || raw.projection_groups > all_projections {
        return Err(Int4JsonError::WorkMismatch.into());
    }
    let output = Int4ExtractCohortRun { schema_version: 1, execution: INT4_EXTRACT_COHORT_EXECUTION.to_owned(),
        sequences, group_steps: raw.group_steps, projection_groups: raw.projection_groups,
        planned_work: expected, model_work };
    super::super::quantized::check_size(&output, cap)?;
    Ok(output)
}

#[cfg(test)] mod tests;
