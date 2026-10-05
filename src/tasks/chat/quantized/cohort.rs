//! Pinned chat/generate finalization for real cross-document INT8 cohorts.
//! No alternate tokenizer, prompt compiler, sampler or raw-text finalizer.
use super::*;
use crate::{tokenizer::bpe::SpBpeTokenizer, native_engine::{
    generation::quantized::{INT8_GENERATION_VERSION, cohort::{self as native,
        Int8CohortBudget, Int8CohortRequest, Int8CohortRequirements, Int8GenerationCohortRun}},
    strict_int8::cohort::{Int8CohortEngine, INT8_COHORT_EXECUTION}, portable_int8::batch::MAX_BATCH_ROWS,
}};

pub struct Int8ChatCohortRequest<'a> {
    pub prepared: &'a PreparedInt8Chat,
    pub admitted_identity: &'a ExecutionIdentity,
    pub request_seq: u64,
    pub budget: Int8GenerationBudget,
}
#[derive(Clone, Copy, Debug)]
pub struct Int8ChatCohortBudget {
    pub generation: Int8CohortBudget,
    /// The COMPLETE canonical envelope, not just assistant content.
    pub max_result_bytes: u64,
}
#[derive(Clone, PartialEq, Serialize)]
pub struct Int8ChatCohortResult {
    pub schema_version: u32,
    pub execution: String,
    pub results: Vec<Int8ChatResult>,
    pub group_steps: u64,
    pub planned_work: Int8Work,
    pub model_work: Int8Work,
}

pub fn preflight(engine: &Int8CohortEngine<'_>, requests: &[Int8ChatCohortRequest<'_>], budget: Int8ChatCohortBudget)
    -> Result<Int8CohortRequirements, Int8ChatError> {
    if budget.max_result_bytes == 0 { return Err(ChatError::Limit("cohort result bytes").into()); }
    let rows = native_requests(requests)?;
    native::preflight(engine, &rows, budget.generation).map_err(Into::into)
}
pub fn execute<C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>, requests: &[Int8ChatCohortRequest<'_>],
    budget: Int8ChatCohortBudget, control: &mut C) -> Result<Int8ChatCohortResult, Int8ChatError> {
    execute_with_sink(engine, requests, budget, &mut Discard, control)
}
/// This bounded API is all-or-error, including task-level no-result. No sibling
/// result or success frame escapes when finalization fails. Provisional token
/// frames may already have been delivered; they are never retracted or retried.
/// Native scope/resource ownership remains the caller's responsibility.
pub fn execute_with_sink<S: DecodeEventSink, C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>,
    requests: &[Int8ChatCohortRequest<'_>], budget: Int8ChatCohortBudget, sink: &mut S, control: &mut C)
    -> Result<Int8ChatCohortResult, Int8ChatError> {
    let required = preflight(engine, requests, budget)?;
    let rows = native_requests(requests)?;
    let raw = native::execute_with_sink(engine, &rows, budget.generation, sink, control)?;
    finalize(requests, raw, required, budget.max_result_bytes)
}
fn native_requests<'a>(requests: &[Int8ChatCohortRequest<'a>])
    -> Result<Vec<Int8CohortRequest<'a, SpBpeTokenizer>>, Int8ChatError> {
    if requests.is_empty() || requests.len() > MAX_BATCH_ROWS { return Err(ChatError::Contract("INT8 cohort width").into()); }
    let mut rows = reserve(requests.len())?;
    for request in requests {
        rows.push(Int8CohortRequest { plan: &request.prepared.native, admitted_identity: request.admitted_identity,
            decoder: request.prepared.tokenizer.tokenizer(), request_seq: request.request_seq,
            budget: request.prepared.task_budget(request.budget) });
    }
    Ok(rows)
}
fn finalize(requests: &[Int8ChatCohortRequest<'_>], raw: Int8GenerationCohortRun,
    required: Int8CohortRequirements, max_result_bytes: u64) -> Result<Int8ChatCohortResult, Int8ChatError> {
    if raw.schema_version != 1 || raw.execution != INT8_COHORT_EXECUTION
        || raw.sequences.len() != requests.len() || raw.planned_work != required.planned_work {
        return Err(Int8ChatError::WorkMismatch);
    }
    let mut results = reserve(requests.len())?; let mut actual = Int8Work::default(); let mut longest = 0;
    for (request, row) in requests.iter().zip(raw.sequences) {
        if row.sequence.request_seq != request.request_seq || row.sequence.sample_index != request.prepared.sample_index
            || row.sequence.execution != INT8_GENERATION_VERSION { return Err(Int8ChatError::WorkMismatch); }
        actual = actual.checked_add(row.model_work).map_err(|_| Int8ChatError::WorkMismatch)?;
        longest = longest.max(row.model_work.forward_positions);
        results.push(request.prepared.finish(row)?);
    }
    if actual != raw.model_work || longest != raw.group_steps { return Err(Int8ChatError::WorkMismatch); }
    let result = Int8ChatCohortResult { schema_version: 1, execution: INT8_COHORT_EXECUTION.to_owned(), results,
        group_steps: raw.group_steps, planned_work: raw.planned_work, model_work: actual };
    bounds::result(&result, max_result_bytes)?; Ok(result)
}
fn reserve<T>(count: usize) -> Result<Vec<T>, Int8ChatError> {
    let mut rows = Vec::new(); rows.try_reserve_exact(count).map_err(|_| ChatError::Allocation)?; Ok(rows)
}
struct Discard;
impl DecodeEventSink for Discard {
    type Permit = (); type Error = std::convert::Infallible;
    fn reserve(&mut self, _: &crate::native_engine::decode::DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
    fn permit(&mut self, _: (), _: crate::native_engine::decode::DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
}
#[cfg(test)] mod tests;
pub mod packed;
