//! One native cohort retains all four independent source-task finalizers.
use super::*;
use crate::{native_engine::strict_int4::cohort::{Int4CohortEngine, MAX_INT4_SEQUENCES},
    tasks::extract::grouped_int4::cohort::{self as extraction_group,
        Int4ExtractCohortRequest, Int4ExtractCohortBudget, Int4ExtractCohortRun,
        INT4_EXTRACT_COHORT_EXECUTION, INT4_EXTRACT_PACKED_COHORT_EXECUTION}};

pub const INT4_SOURCE_COHORT_EXECUTION: &str = "grouped-int4-source-portfolio-cohort-v1";
pub const INT4_SOURCE_PACKED_COHORT_EXECUTION: &str = "grouped-int4-source-portfolio-packed-cohort-v1";

pub struct Int4SourceCohortRequest<'a> {
    pub prepared: &'a PreparedInt4SourceTask,
    pub admitted_identity: &'a ExecutionIdentity,
    pub budget: Int4JsonBudget,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4SourceCohortRun {
    pub schema_version: u32,
    pub execution: String,
    pub sequences: Vec<Int4SourceTaskRun>,
    pub group_steps: u64,
    pub projection_groups: u64,
    pub planned_work: Int4Work,
    pub model_work: Int4Work,
}

/// Model-free row admission for hosted callers before native KV allocation.
/// The extraction layer retains the exact task/control and work contract.
pub(crate) fn preflight(request: &Int4SourceCohortRequest<'_>, vocabulary: &ExtractionVocabulary)
    -> Result<(), Int4SourceError> {
    request.prepared.require_full_projection()?;
    request.prepared.verify_identity(request.admitted_identity)?;
    extraction_group::preflight(&Int4ExtractCohortRequest { prepared: &request.prepared.extraction,
        admitted_identity: request.admitted_identity, budget: request.budget }, vocabulary)?;
    Ok(())
}

/// Sources, task kinds, schemas and passage partitions may differ. Each row
/// retains its own grammar and attention state, full identity, and finalizer.
/// Any failed semantic finalizer discards the entire still-owned native group.
pub fn execute<C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4SourceCohortRequest<'_>], vocabulary: &ExtractionVocabulary,
    budget: Int4ExtractCohortBudget, control: &mut C) -> Result<Int4SourceCohortRun, Int4SourceError> {
    execute_scheduled(engine, requests, vocabulary, budget, None, control)
}
/// Explicit packed prompt execution with the original four source finalizers.
/// The host separately admits the scratch returned by `prefill.validate()`;
/// grouping changes only the outer marker and physical invocation counters.
pub fn execute_packed<C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4SourceCohortRequest<'_>], vocabulary: &ExtractionVocabulary,
    budget: Int4ExtractCohortBudget, prefill: Int4PrefillLimits, control: &mut C)
    -> Result<Int4SourceCohortRun, Int4SourceError> {
    execute_scheduled(engine, requests, vocabulary, budget, Some(prefill), control)
}
fn execute_scheduled<C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4SourceCohortRequest<'_>], vocabulary: &ExtractionVocabulary,
    budget: Int4ExtractCohortBudget, prefill: Option<Int4PrefillLimits>, control: &mut C)
    -> Result<Int4SourceCohortRun, Int4SourceError> {
    if let Some(limits) = prefill { limits.validate().map_err(Int4JsonError::from)?; }
    if requests.is_empty() || requests.len() > MAX_INT4_SEQUENCES { return Err(Int4SourceError::InvalidResult); }
    for request in requests { preflight(request, vocabulary)?; }
    let mut extraction = Vec::new(); extraction.try_reserve_exact(requests.len())
        .map_err(|_| Int4SourceError::from(ExtractError::AllocationRefused))?;
    for request in requests {
        extraction.push(Int4ExtractCohortRequest { prepared: &request.prepared.extraction,
            admitted_identity: request.admitted_identity, budget: request.budget });
    }
    let finalize = |run| match prefill {
        Some(_) => finish_scheduled(requests, run, budget.max_result_bytes, prefill),
        None => finish(requests, run, budget.max_result_bytes),
    };
    match prefill {
        Some(limits) => extraction_group::execute_with_packed(engine, &extraction, vocabulary,
            budget, limits, control, finalize),
        None => extraction_group::execute_with(engine, &extraction, vocabulary, budget, control, finalize),
    }
}

pub(super) fn finish(requests: &[Int4SourceCohortRequest<'_>], raw: Int4ExtractCohortRun, cap: u64)
    -> Result<Int4SourceCohortRun, Int4SourceError> {
    finish_scheduled(requests, raw, cap, None)
}
fn finish_scheduled(requests: &[Int4SourceCohortRequest<'_>], raw: Int4ExtractCohortRun, cap: u64,
    prefill: Option<Int4PrefillLimits>) -> Result<Int4SourceCohortRun, Int4SourceError> {
    if let Some(limits) = prefill { limits.validate().map_err(Int4JsonError::from)?; }
    let extraction_execution = if prefill.is_some() { INT4_EXTRACT_PACKED_COHORT_EXECUTION } else { INT4_EXTRACT_COHORT_EXECUTION };
    if raw.schema_version != 1 || raw.execution != extraction_execution
        || raw.sequences.len() != requests.len() || requests.is_empty() || requests.len() > MAX_INT4_SEQUENCES {
        return Err(Int4SourceError::InvalidResult);
    }
    for request in requests { request.prepared.require_full_projection()?; }
    let mut sequences = Vec::new(); sequences.try_reserve_exact(requests.len())
        .map_err(|_| Int4SourceError::from(ExtractError::AllocationRefused))?;
    let mut planned_work = Int4Work::default(); let mut model_work = Int4Work::default(); let mut longest = 0;
    let mut all_projections = 0_u64; let mut longest_projection_sequence = 0;
    for (request, row) in requests.iter().zip(raw.sequences) {
        request.prepared.verify_identity(request.admitted_identity)?;
        if request.prepared.group_width() != requests[0].prepared.group_width() { return Err(Int4SourceError::InvalidResult); }
        planned_work = planned_work.checked_add(request.prepared.planned_work()).map_err(Int4JsonError::from)?;
        let projections = u64::try_from(row.result.output.token_ids.len()).map_err(|_| Int4SourceError::InvalidResult)?;
        all_projections = all_projections.checked_add(projections).ok_or(Int4SourceError::InvalidResult)?;
        longest_projection_sequence = longest_projection_sequence.max(projections);
        let output = request.prepared.finish(row)?;
        model_work = model_work.checked_add(output.model_work).map_err(Int4JsonError::from)?;
        longest = longest.max(output.model_work.forward_positions);
        sequences.push(output);
    }
    if planned_work != raw.planned_work || model_work != raw.model_work {
        return Err(Int4SourceError::InvalidResult);
    }
    constrained_int4::cohort::check_group_counts(prefill, model_work.forward_positions, longest,
        all_projections, longest_projection_sequence, raw.group_steps, raw.projection_groups)
        .map_err(|_| Int4SourceError::InvalidResult)?;
    let execution = if prefill.is_some() { INT4_SOURCE_PACKED_COHORT_EXECUTION } else { INT4_SOURCE_COHORT_EXECUTION };
    let output = Int4SourceCohortRun { schema_version: 1, execution: execution.to_owned(),
        sequences, group_steps: raw.group_steps, projection_groups: raw.projection_groups, planned_work, model_work };
    check_size(&output, cap)?;
    Ok(output)
}

#[cfg(test)] mod tests;
