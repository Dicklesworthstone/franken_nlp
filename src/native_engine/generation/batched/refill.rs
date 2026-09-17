//! FIFO slot refill inside one finite, fully admitted generation window.
//!
//! A finished sequence retires its generation-scoped native handle immediately.
//! At the next token boundary its slot starts the next assigned request, while
//! other slots continue prefill/decode. There is no whole-wave barrier, new input
//! admission, runtime, worker, sampler, or change to per-row reduction order.
//! All window identities and resource ceilings are checked before any forward.
//! The host retains the window's reservation through final result delivery.

use super::*;

/// A request window may exceed physical row capacity; simultaneous native rows
/// remain bounded by the existing engine envelope and MAX_BATCH_ROWS.
pub const MAX_REFILL_REQUESTS: usize = 128;

/// Unlike the fixed-cohort preflight, repeated slots are legal: each slot owns
/// an input-order FIFO. Every request must fit its assigned slot independently.
/// Duplicate delivery sequences are still forbidden, even in different FIFOs.
pub fn preflight(engine: &EagerBatchEngine<'_>, loaded_model: &ExecutionIdentity,
    requests: &[BatchGenerationRequest<'_>], budget: BatchGenerationBudget)
    -> Result<BatchGenerationRequirements, GenerationError> {
    validate_requests(loaded_model, requests)?;
    let required = requirements(requests, engine.envelope().payload().total_bytes)?;
    check_budget(required, budget)?;
    for request in requests {
        let positions = usize::try_from(request.plan.bound.forward_positions)
            .map_err(|_| GenerationError::Limit("refill positions"))?;
        engine.preflight_sequence_slot(request.slot, positions).map_err(native_error)?;
    }
    Ok(required)
}
fn validate_requests(loaded_model: &ExecutionIdentity, requests: &[BatchGenerationRequest<'_>]) -> Result<(), GenerationError> {
    if requests.is_empty() || requests.len() > MAX_REFILL_REQUESTS { return Err(GenerationError::Contract("refill window width")); }
    loaded_model.validate().map_err(|_| GenerationError::Identity)?;
    let expected = model_binding(loaded_model)?;
    for (index, request) in requests.iter().enumerate() {
        request.plan.verify_identity(request.admitted_identity)?;
        if model_binding(request.admitted_identity)? != expected { return Err(GenerationError::Identity); }
        if request.slot >= MAX_BATCH_ROWS || request.request_seq == 0
            || requests[..index].iter().any(|prior| prior.request_seq == request.request_seq) {
            return Err(GenerationError::Contract("refill slot or delivery sequence"));
        }
    }
    Ok(())
}

pub fn execute<D: DecodeByteDecoder, C: DecodeStepControl>(engine: &mut EagerBatchEngine<'_>,
    loaded_model: &ExecutionIdentity, requests: &[BatchGenerationRequest<'_>], budget: BatchGenerationBudget,
    decoder: &D, control: &mut C) -> Result<BatchGenerationOutput, GenerationError> {
    execute_with_sink(engine, loaded_model, requests, budget, decoder, &mut Discard, control)
}

/// Streaming uses the same two-phase token delivery as fixed-cohort generation.
/// A later failure never returns partial success; delivered events are neither
/// retracted nor retried. All live handles close on error/unwind. An old handle
/// cannot close a replacement sequence, and unrelated slots are not touched.
///
/// `group_steps` counts real native ticks. With continuously occupied per-slot
/// FIFOs it equals max_slot(sum of actual forward positions assigned to slot),
/// NOT the longest individual request. Row execution labels stay unchanged:
/// refill changes placement, not the addressed generation/sampling semantics.
pub fn execute_with_sink<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl>(
    engine: &mut EagerBatchEngine<'_>, loaded_model: &ExecutionIdentity,
    requests: &[BatchGenerationRequest<'_>], budget: BatchGenerationBudget, decoder: &D,
    sink: &mut S, control: &mut C) -> Result<BatchGenerationOutput, GenerationError> {
    let required = preflight(engine, loaded_model, requests, budget)?;
    checkpoint(control, 0)?;
    let mut slots = reserved(requests.len())?;
    let mut cursors = reserved(requests.len())?;
    for request in requests {
        slots.push(request.slot);
        cursors.push(Cursor::new(request.plan, request.request_seq, BATCH_GENERATION_VERSION)?);
    }
    let mut queue = SlotQueue::new(&slots)?;
    let mut handles = reserved(requests.len())?; handles.resize(requests.len(), None);
    let mut native = NativeRefill { engine, slots: &slots, handles };
    let group_steps = drive(&mut cursors, &mut queue, &mut native, decoder, sink, control)?;
    let mut sequences = reserved(cursors.len())?;
    let mut actual_work = GenerationWork::default();
    let mut slot_work = [0_u64; MAX_BATCH_ROWS];
    for (index, cursor) in cursors.into_iter().enumerate() {
        let result = cursor.finish()?;
        actual_work = sum_work(actual_work, result.native_work)?;
        slot_work[slots[index]] = plus(slot_work[slots[index]], result.native_work.forward_positions)?;
        sequences.push(result);
    }
    if group_steps != slot_work.into_iter().max().unwrap_or(0)
        || actual_work.forward_positions > required.planned_work.forward_positions
        || actual_work.projected_logits > required.planned_work.projected_logits
        || actual_work.sampled_steps > required.planned_work.sampled_steps {
        return Err(GenerationError::Contract("refill work accounting"));
    }
    Ok(BatchGenerationOutput { sequences, group_steps, planned_work: required.planned_work, actual_work })
}

/// Linked input-order FIFOs: construction is O(window), each transition O(1).
/// No waiting request scan, cloned plan, or per-token queue allocation is needed.
struct SlotQueue {
    slots: Vec<usize>, next: Vec<Option<usize>>,
    heads: [Option<usize>; MAX_BATCH_ROWS], active: [Option<usize>; MAX_BATCH_ROWS],
}
impl SlotQueue {
    fn new(slots: &[usize]) -> Result<Self, GenerationError> {
        if slots.is_empty() || slots.len() > MAX_REFILL_REQUESTS || slots.iter().any(|&s| s >= MAX_BATCH_ROWS) {
            return Err(GenerationError::Contract("refill queue shape"));
        }
        let mut owned = reserved(slots.len())?; owned.extend_from_slice(slots);
        let mut next = reserved(slots.len())?; next.resize(slots.len(), None);
        let mut heads = [None; MAX_BATCH_ROWS];
        for row in (0..slots.len()).rev() { next[row] = heads[slots[row]]; heads[slots[row]] = Some(row); }
        Ok(Self { slots: owned, next, heads, active: [None; MAX_BATCH_ROWS] })
    }
    fn fill<F: RefillForward, C: DecodeStepControl>(&mut self, forward: &mut F, control: &mut C) -> Result<(), GenerationError> {
        for slot in 0..MAX_BATCH_ROWS {
            if self.active[slot].is_some() { continue; }
            if let Some(row) = self.heads[slot] {
                checkpoint(control, 0)?;
                forward.open(row)?;
                // No fallible operation between opening and recording ownership.
                self.active[slot] = Some(row); self.heads[slot] = self.next[row];
            }
        }
        Ok(())
    }
    fn retire(&mut self, row: usize) -> Result<(), GenerationError> {
        let slot = *self.slots.get(row).ok_or(GenerationError::Contract("refill retirement row"))?;
        if self.active[slot] != Some(row) { return Err(GenerationError::Contract("refill retirement owner")); }
        self.active[slot] = None; Ok(())
    }
}
trait RefillForward: ForwardGroup {
    fn open(&mut self, row: usize) -> Result<(), GenerationError>;
}
fn drive<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl, F: RefillForward>(
    cursors: &mut [Cursor<'_>], queue: &mut SlotQueue, forward: &mut F,
    decoder: &D, sink: &mut S, control: &mut C) -> Result<u64, GenerationError> {
    if cursors.len() != queue.slots.len() { return Err(GenerationError::Contract("refill cursor count")); }
    let mut steps = 0;
    loop {
        queue.fill(forward, control)?;
        let mut tokens = reserved(MAX_BATCH_ROWS)?;
        for row in queue.active.iter().flatten().copied() {
            let cursor = &cursors[row];
            cursor.before_forward(control)?;
            let (token, project) = cursor.next_token()?;
            tokens.push(ScheduledToken { row, token, project });
        }
        if tokens.is_empty() {
            if cursors.iter().any(|c| !c.done) { return Err(GenerationError::Contract("refill stranded request")); }
            return Ok(steps);
        }
        // Semantic request order, not physical slot order, governs event order
        // within one tick. Per-request counters do not use this ordering.
        tokens.sort_unstable_by_key(|token| token.row);
        let results = forward.step(&tokens, control)?; steps = plus(steps, 1)?;
        if results.len() != tokens.len() { return Err(GenerationError::Contract("refill forward row count")); }
        // Validate the entire native reply before committing any token event.
        for (scheduled, result) in tokens.iter().zip(&results) {
            if result.row != scheduled.row || result.position as u64 != cursors[scheduled.row].output.native_work.forward_positions
                || result.logits.is_some() != scheduled.project {
                return Err(GenerationError::Contract("refill forward routing or projection"));
            }
            if let Some(logits) = &result.logits { check_logits(logits)?; }
        }
        for (scheduled, result) in tokens.into_iter().zip(results) {
            let cursor = &mut cursors[scheduled.row];
            cursor.record_forward(scheduled.project)?;
            if let Some(logits) = result.logits { cursor.emit_next(&logits, decoder, sink, control)?; }
            if cursor.done {
                forward.close(scheduled.row)?;
                queue.retire(scheduled.row)?;
            }
        }
    }
}

struct NativeRefill<'engine, 'weights, 'slots> {
    engine: &'engine mut EagerBatchEngine<'weights>, slots: &'slots [usize],
    handles: Vec<Option<BatchSequence>>,
}
impl RefillForward for NativeRefill<'_, '_, '_> {
    fn open(&mut self, row: usize) -> Result<(), GenerationError> {
        let target = self.handles.get_mut(row).ok_or(GenerationError::Contract("refill open row"))?;
        if target.is_some() { return Err(GenerationError::Contract("refill double open")); }
        let slot = *self.slots.get(row).ok_or(GenerationError::Contract("refill open slot"))?;
        *target = Some(self.engine.open_sequence(slot).map_err(native_error)?); Ok(())
    }
}
impl ForwardGroup for NativeRefill<'_, '_, '_> {
    fn step<C: DecodeStepControl>(&mut self, tokens: &[ScheduledToken], control: &mut C) -> Result<Vec<ForwardedRow>, GenerationError> {
        let mut input = reserved(tokens.len())?;
        for token in tokens {
            let sequence = self.handles.get(token.row).copied().flatten().ok_or(GenerationError::Contract("refill handle routing"))?;
            input.push(BatchToken { sequence, token_id: token.token,
                projection: if token.project { BatchProjection::FullVocabulary } else { BatchProjection::None } });
        }
        let result = self.engine.step(&input, &mut NativeControl(control)).map_err(native_error)?;
        if result.rows.len() != input.len() { return Err(GenerationError::Contract("refill native row count")); }
        let mut output = reserved(tokens.len())?;
        for ((row, input), scheduled) in result.rows.into_iter().zip(&input).zip(tokens) {
            if row.sequence != input.sequence { return Err(GenerationError::Contract("refill native handle mismatch")); }
            output.push(ForwardedRow { row: scheduled.row, position: row.position, logits: row.logits });
        }
        Ok(output)
    }
    fn close(&mut self, row: usize) -> Result<(), GenerationError> {
        let target = self.handles.get_mut(row).ok_or(GenerationError::Contract("refill close row"))?;
        let sequence = target.ok_or(GenerationError::Contract("refill double close"))?;
        self.engine.close_sequence(sequence).map_err(native_error)?;
        *target = None; Ok(())
    }
}
impl Drop for NativeRefill<'_, '_, '_> {
    fn drop(&mut self) {
        for sequence in self.handles.iter().flatten().copied() {
            // Failed steps may already have invalidated a handle. Its generation
            // prevents cleanup from targeting a newer occupant of the same slot.
            let _ = self.engine.close_sequence(sequence);
        }
    }
}

#[cfg(test)] mod tests;
