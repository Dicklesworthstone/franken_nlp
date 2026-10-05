//! Pinned task finalization for the mixed prompt/decode token-pack strategy.
//! Ordinary cohort finalization is unchanged and cannot accept this schedule
//! merely because someone relabeled its longest-sequence tick count.
use super::*;
use crate::native_engine::strict_int8::{prefill::Int8PrefillLimits,
    cohort::packed::INT8_PACKED_COHORT_EXECUTION};

pub fn execute<C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>, requests: &[Int8ChatCohortRequest<'_>],
    budget: Int8ChatCohortBudget, limits: Int8PrefillLimits, control: &mut C)
    -> Result<Int8ChatCohortResult, Int8ChatError> {
    execute_with_sink(engine, requests, budget, limits, &mut Discard, control)
}
/// Every provisional event remains subject to the same independent pinned
/// text finalizer and complete result-byte ceiling. No sibling result escapes
/// a native or task failure; this is not an alternate decoding authority.
pub fn execute_with_sink<S: DecodeEventSink, C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>,
    requests: &[Int8ChatCohortRequest<'_>], budget: Int8ChatCohortBudget, limits: Int8PrefillLimits,
    sink: &mut S, control: &mut C) -> Result<Int8ChatCohortResult, Int8ChatError> {
    limits.validate().map_err(Int8GenerationError::from)?;
    let required = super::preflight(engine, requests, budget)?;
    let rows = native_requests(requests)?;
    let raw = native::packed::execute_with_sink(engine, &rows, budget.generation, limits, sink, control)?;
    finalize(requests, raw, required, budget.max_result_bytes, limits)
}
fn finalize(requests: &[Int8ChatCohortRequest<'_>], raw: Int8GenerationCohortRun,
    required: Int8CohortRequirements, max_result_bytes: u64, limits: Int8PrefillLimits)
    -> Result<Int8ChatCohortResult, Int8ChatError> {
    limits.validate().map_err(Int8GenerationError::from)?;
    if raw.schema_version != 1 || raw.execution != INT8_PACKED_COHORT_EXECUTION
        || raw.sequences.len() != requests.len() || raw.planned_work != required.planned_work {
        return Err(Int8ChatError::WorkMismatch);
    }
    let mut results = reserve(requests.len())?; let mut actual = Int8Work::default();
    let mut prompts = reserve(requests.len())?; let mut positions = reserve(requests.len())?;
    for (request, row) in requests.iter().zip(raw.sequences) {
        if row.sequence.request_seq != request.request_seq || row.sequence.sample_index != request.prepared.sample_index
            || row.sequence.execution != INT8_GENERATION_VERSION { return Err(Int8ChatError::WorkMismatch); }
        actual = actual.checked_add(row.model_work).map_err(|_| Int8ChatError::WorkMismatch)?;
        prompts.push(request.prepared.native.prompt_tokens()); positions.push(row.model_work.forward_positions);
        results.push(request.prepared.finish(row)?);
    }
    if actual != raw.model_work
        || native::packed::expected_group_steps(&prompts, &positions, limits.max_batch_rows)? != raw.group_steps {
        return Err(Int8ChatError::WorkMismatch);
    }
    let result = Int8ChatCohortResult { schema_version: 1, execution: INT8_PACKED_COHORT_EXECUTION.to_owned(), results,
        group_steps: raw.group_steps, planned_work: raw.planned_work, model_work: actual };
    bounds::result(&result, max_result_bytes)?; Ok(result)
}
#[cfg(test)] mod tests;
