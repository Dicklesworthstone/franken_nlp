//! Shared-layer cohorts retaining the four existing source-task finalizers.
use super::*;
use crate::{native_engine::{portable_int8::batch::MAX_BATCH_ROWS, strict_int8::cohort::Int8CohortEngine},
    tasks::extract::quantized::cohort::{self as extraction_group,
        Int8ExtractCohortRequest, Int8ExtractCohortBudget, Int8ExtractCohortRun,
        INT8_EXTRACT_COHORT_EXECUTION}};

pub const INT8_SOURCE_COHORT_EXECUTION: &str = "portable-int8-selected-source-cohort-v1";

pub struct Int8SourceCohortRequest<'a> {
    pub prepared: &'a PreparedInt8SourceTask,
    pub admitted_identity: &'a ExecutionIdentity,
    pub budget: Int8JsonBudget,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8SourceCohortRun {
    pub schema_version: u32,
    pub execution: String,
    pub sequences: Vec<Int8SourceTaskRun>,
    pub group_steps: u64,
    pub planned_work: Int8Work,
    pub model_work: Int8Work,
}
/// All rows must be explicitly selected-row plans. Their task kinds, schemas,
/// source texts and passage partitions may differ; they never share a grammar
/// state or attention context. Grouping does not rewrite their semantic keys.
/// A failure of ANY source semantic finalizer discards the whole cohort.
pub fn execute<C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>,
    requests: &[Int8SourceCohortRequest<'_>], vocabulary: &ExtractionVocabulary,
    budget: Int8ExtractCohortBudget, control: &mut C) -> Result<Int8SourceCohortRun, Int8SourceError> {
    if requests.is_empty() || requests.len() > MAX_BATCH_ROWS { return Err(Int8SourceError::InvalidResult); }
    for request in requests {
        request.prepared.verify_identity(request.admitted_identity)?;
        if request.prepared.selected_rows().is_none() { return Err(Int8ExtractError::Identity.into()); }
    }
    let mut extraction = Vec::new(); extraction.try_reserve_exact(requests.len())
        .map_err(|_| Int8SourceError::from(ExtractError::AllocationRefused))?;
    for request in requests {
        extraction.push(Int8ExtractCohortRequest { prepared: &request.prepared.extraction,
            admitted_identity: request.admitted_identity, budget: request.budget });
    }
    // The extraction adapter itself composes with the still-open native session.
    // No native ownership boundary is crossed before these semantic checks.
    extraction_group::execute_with(engine, &extraction, vocabulary, budget, control,
        |run| finish(requests, run, budget.max_result_bytes))
}
pub(super) fn finish(requests: &[Int8SourceCohortRequest<'_>], raw: Int8ExtractCohortRun, cap: u64)
    -> Result<Int8SourceCohortRun, Int8SourceError> {
    if raw.schema_version != 1 || raw.execution != INT8_EXTRACT_COHORT_EXECUTION
        || raw.sequences.len() != requests.len() || requests.is_empty() {
        return Err(Int8SourceError::InvalidResult);
    }
    let mut sequences = Vec::new(); sequences.try_reserve_exact(requests.len())
        .map_err(|_| Int8SourceError::from(ExtractError::AllocationRefused))?;
    let mut planned_work = Int8Work::default(); let mut model_work = Int8Work::default(); let mut longest = 0;
    for (request, row) in requests.iter().zip(raw.sequences) {
        if request.prepared.selected_rows().is_none() { return Err(Int8ExtractError::Identity.into()); }
        planned_work = planned_work.checked_add(request.prepared.planned_work()).map_err(Int8JsonError::from)?;
        let output = request.prepared.finish(row)?;
        model_work = model_work.checked_add(output.model_work).map_err(Int8JsonError::from)?;
        longest = longest.max(output.model_work.forward_positions);
        sequences.push(output);
    }
    if planned_work != raw.planned_work || model_work != raw.model_work || longest != raw.group_steps {
        return Err(Int8SourceError::InvalidResult);
    }
    let output = Int8SourceCohortRun { schema_version: 1, execution: INT8_SOURCE_COHORT_EXECUTION.to_owned(),
        sequences, group_steps: raw.group_steps, planned_work, model_work };
    extract_int8::check_size(&output, cap)?;
    Ok(output)
}
#[cfg(test)] mod tests;
