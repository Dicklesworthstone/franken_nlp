//! Pinned PreparedChat/Generate task cohorts on shared-weight native batches.
//! This library API is distinct from the frozen robot/ordered-NDJSON protocol.
//! It reuses each task's pinned decoder and independent finalizer, not a second
//! prompt compiler, raw-text finalizer, or unqualified model activation path.
use super::*;
use crate::native_engine::{
    batchsched::{EagerBatchEngine, MAX_BATCH_ROWS},
    generation::batched::{self as native, BatchGenerationBudget, BatchGenerationOutput,
        BatchGenerationRequest, BatchGenerationRequirements},
    kv::KV_BYTES_PER_TOKEN,
};

pub const CHAT_BATCH_VERSION: &str = "pinned-chat-batch-v1";
pub struct BatchChatRequest<'a> {
    pub prepared: &'a PreparedChat,
    pub admitted_identity: &'a ExecutionIdentity,
    pub slot: usize,
    pub request_seq: u64,
}
#[derive(Clone, Copy, Debug)]
pub struct BatchChatBudget {
    pub generation: BatchGenerationBudget,
    /// Complete canonical cohort envelope, including per-row errors/metadata.
    pub max_result_bytes: u64,
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchChatNoResult { IncompleteUtf8, ResultByteLimit }
#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum BatchChatItem {
    Completed { result: ChatResult },
    /// Expected task-level refusal, not a forged successful replacement string.
    /// Fixed categories only; private decoder/source details are never echoed.
    NoResult { request_seq: u64, sample_index: u64, reason: BatchChatNoResult },
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchChatResult {
    pub schema_version: u32,
    pub execution: String,
    pub results: Vec<BatchChatItem>,
    pub group_steps: u64,
    pub planned_work: GenerationWork,
    pub actual_work: GenerationWork,
}

pub fn preflight(engine: &EagerBatchEngine<'_>, loaded_model: &ExecutionIdentity,
    requests: &[BatchChatRequest<'_>], budget: BatchChatBudget) -> Result<BatchGenerationRequirements, ChatError> {
    if budget.max_result_bytes == 0 { return Err(ChatError::Limit("cohort result bytes")); }
    check_task_kv_limits(engine.envelope().capacities(), requests)?;
    let native_requests = native_requests(requests)?;
    native::preflight(engine, loaded_model, &native_requests, budget.generation).map_err(ChatError::from)
}

pub fn execute<C: DecodeStepControl>(engine: &mut EagerBatchEngine<'_>, loaded_model: &ExecutionIdentity,
    requests: &[BatchChatRequest<'_>], budget: BatchChatBudget, control: &mut C) -> Result<BatchChatResult, ChatError> {
    execute_with_sink(engine, loaded_model, requests, budget, &mut Discard, control)
}

/// Native/cancellation/integrity failures abort the cohort. Ordinary finalizer
/// no-result states remain per-row, so incomplete UTF-8 in one output does not
/// erase independently valid siblings. Token events are untrusted incremental
/// data until finalization; already delivered bytes cannot be retracted.
/// The host retains real admission and output guards through caller-side flush.
pub fn execute_with_sink<S: DecodeEventSink, C: DecodeStepControl>(engine: &mut EagerBatchEngine<'_>,
    loaded_model: &ExecutionIdentity, requests: &[BatchChatRequest<'_>], budget: BatchChatBudget,
    sink: &mut S, control: &mut C) -> Result<BatchChatResult, ChatError> {
    let required = preflight(engine, loaded_model, requests, budget)?;
    let first = requests.first().ok_or(ChatError::Contract("empty chat cohort"))?;
    let native_requests = native_requests(requests)?;
    // Native preflight compares every model/tokenizer/template binding. All
    // PreparedChat values own the checked pinned tokenizer; no caller-supplied
    // byte decoder can be swapped in at this task execution boundary.
    let raw = native::execute_with_sink(engine, loaded_model, &native_requests, budget.generation,
        first.prepared.tokenizer.tokenizer(), sink, control)?;
    finalize(requests, raw, required, budget.max_result_bytes)
}
fn native_requests<'a>(requests: &[BatchChatRequest<'a>]) -> Result<Vec<BatchGenerationRequest<'a>>, ChatError> {
    if requests.is_empty() || requests.len() > MAX_BATCH_ROWS { return Err(ChatError::Contract("chat cohort width")); }
    let mut rows = reserve(requests.len())?;
    for request in requests {
        rows.push(BatchGenerationRequest { plan: &request.prepared.native, admitted_identity: request.admitted_identity,
            slot: request.slot, request_seq: request.request_seq });
    }
    Ok(rows)
}
fn check_task_kv_limits(capacities: &[usize], requests: &[BatchChatRequest<'_>]) -> Result<(), ChatError> {
    if requests.is_empty() || requests.len() > MAX_BATCH_ROWS { return Err(ChatError::Contract("chat cohort width")); }
    for request in requests {
        let cap = capacities.get(request.slot).ok_or(ChatError::Limit("chat cohort slot"))?;
        let bytes = (*cap as u64).checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(ChatError::Limit("task KV arithmetic"))?;
        if bytes > request.prepared.task.ir().budget().max_kv_bytes { return Err(ChatError::Limit("task complete KV reservation")); }
    }
    Ok(())
}
fn finalize(requests: &[BatchChatRequest<'_>], raw: BatchGenerationOutput,
    required: BatchGenerationRequirements, max_result_bytes: u64) -> Result<BatchChatResult, ChatError> {
    if raw.sequences.len() != requests.len() || raw.planned_work != required.planned_work {
        return Err(ChatError::NoResult("cohort envelope or planned work"));
    }
    let mut actual = GenerationWork::default(); let mut max_positions = 0_u64;
    for row in &raw.sequences { actual = sum(actual, row.native_work)?; max_positions = max_positions.max(row.native_work.forward_positions); }
    if actual != raw.actual_work || raw.group_steps != max_positions
        || actual.forward_positions > required.planned_work.forward_positions
        || actual.projected_logits > required.planned_work.projected_logits || actual.sampled_steps > required.planned_work.sampled_steps {
        return Err(ChatError::NoResult("cohort actual work"));
    }
    let mut results = reserve(requests.len())?;
    for (request, row) in requests.iter().zip(raw.sequences) {
        if row.request_seq != request.request_seq || row.execution != BATCH_GENERATION_VERSION {
            return Err(ChatError::NoResult("cohort row routing"));
        }
        let item = match request.prepared.finish(row) {
            Ok(result) => BatchChatItem::Completed { result },
            Err(ChatError::NoResult("incomplete UTF-8")) => BatchChatItem::NoResult {
                request_seq: request.request_seq, sample_index: request.prepared.sample_index, reason: BatchChatNoResult::IncompleteUtf8,
            },
            Err(ChatError::Limit("complete result bytes")) => BatchChatItem::NoResult {
                request_seq: request.request_seq, sample_index: request.prepared.sample_index, reason: BatchChatNoResult::ResultByteLimit,
            },
            Err(error) => return Err(error),
        };
        results.push(item);
    }
    let result = BatchChatResult { schema_version: 1, execution: CHAT_BATCH_VERSION.to_owned(), results,
        group_steps: raw.group_steps, planned_work: raw.planned_work, actual_work: actual };
    bounds::result(&result, max_result_bytes)?;
    Ok(result)
}
fn sum(a: GenerationWork, b: GenerationWork) -> Result<GenerationWork, ChatError> {
    Ok(GenerationWork {
        forward_positions: a.forward_positions.checked_add(b.forward_positions).ok_or(ChatError::NoResult("cohort work arithmetic"))?,
        projected_logits: a.projected_logits.checked_add(b.projected_logits).ok_or(ChatError::NoResult("cohort work arithmetic"))?,
        sampled_steps: a.sampled_steps.checked_add(b.sampled_steps).ok_or(ChatError::NoResult("cohort work arithmetic"))?,
    })
}
fn reserve<T>(count: usize) -> Result<Vec<T>, ChatError> {
    let mut result = Vec::new(); result.try_reserve_exact(count).map_err(|_| ChatError::Allocation)?; Ok(result)
}
struct Discard;
impl DecodeEventSink for Discard {
    type Permit = (); type Error = std::convert::Infallible;
    fn reserve(&mut self, _: &crate::native_engine::decode::DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
    fn permit(&mut self, _: (), _: crate::native_engine::decode::DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
}
#[cfg(test)] mod tests;
