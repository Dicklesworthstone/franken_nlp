//! Pinned text/chat finalization for a finite slot-refilling INT4 epoch.
//! Every completed raw row still passes the ordinary independent finalizer.
use super::*;
use crate::native_engine::strict_int4::cohort::packed::refill::INT4_REFILL_EXECUTION;

pub fn requirements(requests: &[Int4ChatBatchRequest<'_>], contexts: &[usize],
    limits: Int4ChatBatchLimits, prefill: Int4PrefillLimits)
    -> Result<Int4BatchGenerationRequirements, Int4ChatError> {
    check_result_budget(requests, limits.max_result_bytes)?;
    let native_requests = bindings(requests)?;
    native::packed::refill::requirements(&native_requests, contexts, limits.generation, prefill)
        .map_err(Into::into)
}

pub fn preflight(engine: &Int4CohortEngine<'_>, requests: &[Int4ChatBatchRequest<'_>],
    limits: Int4ChatBatchLimits, prefill: Int4PrefillLimits)
    -> Result<Int4BatchGenerationRequirements, Int4ChatError> {
    check_result_budget(requests, limits.max_result_bytes)?;
    let native_requests = bindings(requests)?;
    native::packed::refill::preflight(engine, &native_requests, limits.generation, prefill)
        .map_err(Into::into)
}

pub fn execute<C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4ChatBatchRequest<'_>], limits: Int4ChatBatchLimits,
    prefill: Int4PrefillLimits, control: &mut C) -> Result<Int4ChatBatchResult, Int4ChatError> {
    execute_with_sink(engine, requests, limits, prefill, &mut Discard, control)
}

pub fn execute_with_sink<S: DecodeEventSink, C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4ChatBatchRequest<'_>], limits: Int4ChatBatchLimits, prefill: Int4PrefillLimits,
    sink: &mut S, control: &mut C) -> Result<Int4ChatBatchResult, Int4ChatError> {
    let required = preflight(engine, requests, limits, prefill)?;
    let slots = engine.sequence_count();
    let native_requests = bindings(requests)?;
    let raw = native::packed::refill::execute_with_sink(engine, &native_requests, limits.generation,
        prefill, requests[0].prepared.tokenizer.tokenizer(), sink, control)?;
    finalize(requests, raw, required, limits.max_result_bytes, slots, prefill)
}

fn finalize(requests: &[Int4ChatBatchRequest<'_>], raw: Int4BatchGenerationOutput,
    required: Int4BatchGenerationRequirements, max_result_bytes: u64, slots: usize,
    prefill: Int4PrefillLimits) -> Result<Int4ChatBatchResult, Int4ChatError> {
    prefill.validate().map_err(|_| Int4ChatError::WorkMismatch)?;
    if raw.schema_version != 1 || raw.execution != INT4_REFILL_EXECUTION
        || raw.runs.len() != requests.len() || raw.planned_work != required.planned_work {
        return Err(Int4ChatError::WorkMismatch);
    }
    let mut results = Vec::new();
    results.try_reserve_exact(requests.len()).map_err(|_| ChatError::Limit("INT4 refill result allocation"))?;
    let mut actual = Int4Work::default();
    let mut prompts = Vec::new(); let mut positions = Vec::new();
    prompts.try_reserve_exact(requests.len()).map_err(|_| ChatError::Limit("INT4 refill prompt receipt"))?;
    positions.try_reserve_exact(requests.len()).map_err(|_| ChatError::Limit("INT4 refill position receipt"))?;
    for (request, run) in requests.iter().zip(raw.runs) {
        request.prepared.verify_identity(request.admitted_identity)?;
        if run.sequence.request_seq != request.request_seq { return Err(Int4ChatError::WorkMismatch); }
        actual = actual.checked_add(run.model_work).map_err(|_| Int4ChatError::WorkMismatch)?;
        prompts.push(request.prepared.native.prompt_tokens());
        positions.push(run.model_work.forward_positions);
        results.push(request.prepared.finish(run)?);
    }
    if raw.actual_work != actual
        || native::packed::refill::expected_steps(&prompts, &positions, slots, prefill.max_batch_rows)?
            != raw.group_steps {
        return Err(Int4ChatError::WorkMismatch);
    }
    let result = Int4ChatBatchResult {
        schema_version: 1, execution: INT4_REFILL_EXECUTION, results,
        group_steps: raw.group_steps, planned_work: raw.planned_work, model_work: actual,
    };
    bounds::result(&result, max_result_bytes)?;
    Ok(result)
}

#[cfg(test)] mod tests;
