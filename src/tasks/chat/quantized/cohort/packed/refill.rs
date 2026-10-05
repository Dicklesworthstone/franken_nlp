//! Pinned chat/generate finalization for a bounded slot-refilling INT8 epoch.
//! Every raw request still passes the ordinary text/UTF-8/control finalizer.
use super::*;
use crate::native_engine::strict_int8::cohort::packed::refill::INT8_REFILL_EXECUTION;

pub fn preflight(engine: &Int8CohortEngine<'_>, requests: &[Int8ChatCohortRequest<'_>],
    budget: Int8ChatCohortBudget, limits: Int8PrefillLimits) -> Result<Int8CohortRequirements, Int8ChatError> {
    if budget.max_result_bytes == 0 { return Err(ChatError::Limit("refilling epoch result bytes").into()); }
    let rows = native_requests(requests)?;
    native::packed::refill::preflight(engine, &rows, budget.generation, limits).map_err(Into::into)
}
pub fn execute<C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>, requests: &[Int8ChatCohortRequest<'_>],
    budget: Int8ChatCohortBudget, limits: Int8PrefillLimits, control: &mut C)
    -> Result<Int8ChatCohortResult, Int8ChatError> {
    execute_with_sink(engine, requests, budget, limits, &mut Discard, control)
}
pub fn execute_with_sink<S: DecodeEventSink, C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>,
    requests: &[Int8ChatCohortRequest<'_>], budget: Int8ChatCohortBudget, limits: Int8PrefillLimits,
    sink: &mut S, control: &mut C) -> Result<Int8ChatCohortResult, Int8ChatError> {
    let required = preflight(engine, requests, budget, limits)?;
    let slots = engine.sequence_count(); let rows = native_requests(requests)?;
    let raw = native::packed::refill::execute_with_sink(engine, &rows, budget.generation, limits, sink, control)?;
    finalize(requests, raw, required, budget.max_result_bytes, slots, limits)
}
fn finalize(requests: &[Int8ChatCohortRequest<'_>], raw: Int8GenerationCohortRun,
    required: Int8CohortRequirements, max_result_bytes: u64, slots: usize, limits: Int8PrefillLimits)
    -> Result<Int8ChatCohortResult, Int8ChatError> {
    limits.validate().map_err(Int8GenerationError::from)?;
    if raw.schema_version != 1 || raw.execution != INT8_REFILL_EXECUTION
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
        || native::packed::refill::expected_steps(&prompts, &positions, slots, limits.max_batch_rows)? != raw.group_steps {
        return Err(Int8ChatError::WorkMismatch);
    }
    let result = Int8ChatCohortResult { schema_version: 1, execution: INT8_REFILL_EXECUTION.to_owned(), results,
        group_steps: raw.group_steps, planned_work: raw.planned_work, model_work: actual };
    bounds::result(&result, max_result_bytes)?; Ok(result)
}
#[cfg(test)] mod tests;
