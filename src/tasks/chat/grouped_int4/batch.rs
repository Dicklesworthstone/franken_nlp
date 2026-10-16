//! Whole-cohort text/chat finalization over the real INT4 generation driver.
//! Compiled prompts and the pinned tokenizer are reused, not reinterpreted.
use super::*;
use crate::native_engine::{
    generation::grouped_int4::batch::{self as native, Int4BatchGenerationRequest,
        Int4BatchGenerationBudget, Int4BatchGenerationRequirements, Int4BatchGenerationOutput,
        INT4_BATCH_GENERATION_VERSION},
    strict_int4::{prefill::Int4PrefillLimits,
        cohort::{Int4CohortEngine, MAX_INT4_SEQUENCES, packed::INT4_PACKED_COHORT_EXECUTION}},
};

pub struct Int4ChatBatchRequest<'a> {
    pub prepared: &'a PreparedInt4Chat,
    pub admitted_identity: &'a ExecutionIdentity,
    pub request_seq: u64,
    pub budget: Int4GenerationBudget,
}
#[derive(Clone, Copy, Debug)]
pub struct Int4ChatBatchLimits {
    pub generation: Int4BatchGenerationBudget,
    /// Complete aggregate wrapper, not a per-member byte allowance.
    pub max_result_bytes: u64,
}
#[derive(Serialize)]
pub struct Int4ChatBatchResult {
    pub schema_version: u32,
    pub execution: &'static str,
    pub results: Vec<Int4ChatResult>,
    pub group_steps: u64,
    pub planned_work: Int4Work,
    pub model_work: Int4Work,
}

pub fn requirements(requests: &[Int4ChatBatchRequest<'_>], contexts: &[usize], limits: Int4ChatBatchLimits)
    -> Result<Int4BatchGenerationRequirements, Int4ChatError> {
    check_result_budget(requests, limits.max_result_bytes)?;
    let native_requests = bindings(requests)?;
    native::requirements(&native_requests, contexts, limits.generation).map_err(Into::into)
}
/// Includes the explicit packed prompt rails in the aggregate native envelope.
pub fn requirements_packed(requests: &[Int4ChatBatchRequest<'_>], contexts: &[usize],
    limits: Int4ChatBatchLimits, prefill: Int4PrefillLimits)
    -> Result<Int4BatchGenerationRequirements, Int4ChatError> {
    check_result_budget(requests, limits.max_result_bytes)?;
    let native_requests = bindings(requests)?;
    native::packed::requirements(&native_requests, contexts, limits.generation, prefill).map_err(Into::into)
}

pub fn execute<C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4ChatBatchRequest<'_>], limits: Int4ChatBatchLimits, control: &mut C)
    -> Result<Int4ChatBatchResult, Int4ChatError> {
    check_result_budget(requests, limits.max_result_bytes)?;
    let native_requests = bindings(requests)?;
    // Compatibility validation binds the same real tokenizer assets in every
    // row. There is no caller-provided replacement decoder on this task API.
    let raw = native::execute(engine, &native_requests, limits.generation,
        requests[0].prepared.tokenizer.tokenizer(), control)?;
    finish(requests, raw, limits.max_result_bytes)
}

/// Events retain row-local request_seq and token_index. They are provisional
/// until every row AND the aggregate wrapper pass independent task validation.
pub fn execute_with_sink<S: DecodeEventSink, C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4ChatBatchRequest<'_>], limits: Int4ChatBatchLimits, sink: &mut S, control: &mut C)
    -> Result<Int4ChatBatchResult, Int4ChatError> {
    check_result_budget(requests, limits.max_result_bytes)?;
    let native_requests = bindings(requests)?;
    let raw = native::execute_with_sink(engine, &native_requests, limits.generation,
        requests[0].prepared.tokenizer.tokenizer(), sink, control)?;
    finish(requests, raw, limits.max_result_bytes)
}

/// Explicit token packing changes only physical scheduling. Per-task identity,
/// output limits, the pinned decoder and the independent finalizers are shared.
pub fn execute_packed<C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4ChatBatchRequest<'_>], limits: Int4ChatBatchLimits,
    prefill: Int4PrefillLimits, control: &mut C) -> Result<Int4ChatBatchResult, Int4ChatError> {
    check_result_budget(requests, limits.max_result_bytes)?;
    let native_requests = bindings(requests)?;
    let raw = native::packed::execute(engine, &native_requests, limits.generation, prefill,
        requests[0].prepared.tokenizer.tokenizer(), control)?;
    finish_with_schedule(requests, raw, limits.max_result_bytes, Some(prefill))
}
pub fn execute_packed_with_sink<S: DecodeEventSink, C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4ChatBatchRequest<'_>], limits: Int4ChatBatchLimits,
    prefill: Int4PrefillLimits, sink: &mut S, control: &mut C) -> Result<Int4ChatBatchResult, Int4ChatError> {
    check_result_budget(requests, limits.max_result_bytes)?;
    let native_requests = bindings(requests)?;
    let raw = native::packed::execute_with_sink(engine, &native_requests, limits.generation, prefill,
        requests[0].prepared.tokenizer.tokenizer(), sink, control)?;
    finish_with_schedule(requests, raw, limits.max_result_bytes, Some(prefill))
}

fn bindings<'a>(requests: &'a [Int4ChatBatchRequest<'_>]) -> Result<Vec<Int4BatchGenerationRequest<'a>>, Int4ChatError> {
    if requests.is_empty() || requests.len() > MAX_INT4_SEQUENCES { return Err(ChatError::Limit("INT4 batch width").into()); }
    let mut native_requests = Vec::new();
    native_requests.try_reserve_exact(requests.len()).map_err(|_| ChatError::Limit("INT4 batch allocation"))?;
    for request in requests {
        request.prepared.verify_identity(request.admitted_identity)?;
        native_requests.push(Int4BatchGenerationRequest { plan: &request.prepared.native,
            admitted_identity: request.admitted_identity, request_seq: request.request_seq,
            budget: request.prepared.task_budget(request.budget) });
    }
    Ok(native_requests)
}

fn check_result_budget(requests: &[Int4ChatBatchRequest<'_>], maximum: u64) -> Result<(), Int4ChatError> {
    if requests.is_empty() || requests.len() > MAX_INT4_SEQUENCES { return Err(ChatError::Limit("INT4 batch width").into()); }
    // Reserve every member's full typed/serialized ceiling and a fixed wrapper
    // allowance before inference. A tiny aggregate limit cannot underprice all
    // the retained member results while waiting for final serialization.
    let required = wrapper_bound(requests.iter().map(|r| r.prepared.task.ir().budget().max_output_bytes))?;
    if required > maximum { return Err(ChatError::Limit("INT4 batch result bytes").into()); }
    Ok(())
}
fn wrapper_bound(mut bytes: impl Iterator<Item = u64>) -> Result<u64, Int4ChatError> {
    bytes.try_fold(16_384_u64, |sum, bytes| sum.checked_add(bytes)
        .ok_or(ChatError::Limit("INT4 batch result arithmetic").into()))
}

fn finish(requests: &[Int4ChatBatchRequest<'_>], raw: Int4BatchGenerationOutput, max_bytes: u64)
    -> Result<Int4ChatBatchResult, Int4ChatError> {
    finish_with_schedule(requests, raw, max_bytes, None)
}
fn finish_with_schedule(requests: &[Int4ChatBatchRequest<'_>], raw: Int4BatchGenerationOutput, max_bytes: u64,
    prefill: Option<Int4PrefillLimits>) -> Result<Int4ChatBatchResult, Int4ChatError> {
    let execution = if prefill.is_some() { INT4_PACKED_COHORT_EXECUTION } else { INT4_BATCH_GENERATION_VERSION };
    if requests.is_empty() || requests.len() > MAX_INT4_SEQUENCES || raw.schema_version != 1
        || raw.execution != execution || raw.runs.len() != requests.len() {
        return Err(Int4ChatError::WorkMismatch);
    }
    let mut planned = Int4Work::default(); let mut actual = Int4Work::default(); let mut steps = 0;
    let mut prompts = [0_usize; MAX_INT4_SEQUENCES]; let mut forwards = [0_u64; MAX_INT4_SEQUENCES];
    for (index, (request, run)) in requests.iter().zip(&raw.runs).enumerate() {
        request.prepared.verify_identity(request.admitted_identity)?;
        if run.sequence.request_seq != request.request_seq { return Err(Int4ChatError::WorkMismatch); }
        planned = planned.checked_add(request.prepared.planned_work()).map_err(|_| Int4ChatError::WorkMismatch)?;
        actual = actual.checked_add(run.model_work).map_err(|_| Int4ChatError::WorkMismatch)?;
        steps = steps.max(run.model_work.forward_positions);
        prompts[index] = request.prepared.native.prompt_tokens(); forwards[index] = run.model_work.forward_positions;
    }
    if let Some(prefill) = prefill {
        prefill.validate().map_err(|_| Int4ChatError::WorkMismatch)?;
        steps = native::packed::expected_group_steps(&prompts[..requests.len()], &forwards[..requests.len()], prefill.max_batch_rows)?;
    }
    if raw.planned_work != planned || raw.actual_work != actual || raw.group_steps != steps {
        return Err(Int4ChatError::WorkMismatch);
    }
    let mut results = Vec::new();
    results.try_reserve_exact(requests.len()).map_err(|_| ChatError::Limit("INT4 batch allocation"))?;
    for (request, run) in requests.iter().zip(raw.runs) {
        // Existing finalizer re-decodes tokens and checks UTF-8, stops, controls,
        // sample identity, complete work, task validity and each wrapper's size.
        results.push(request.prepared.finish(run)?);
    }
    let output = Int4ChatBatchResult { schema_version: 1, execution,
        results, group_steps: raw.group_steps, planned_work: planned, model_work: actual };
    bounds::result(&output, max_bytes)?;
    Ok(output)
}

pub mod refill;
#[cfg(test)] mod tests;

