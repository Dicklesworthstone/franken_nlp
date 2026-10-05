//! Exact existing generation cursors with bounded FIFO slot refill.
//! Request indices own identities/results; physical slots own reusable KV only.
//! A completed row is retired and replaced without waiting for other rows.
use super::*;
use crate::native_engine::strict_int8::cohort::packed::refill::{
    Int8RefillSession, RefillAdmission, RefillRetirement, INT8_REFILL_EXECUTION,
};
mod replay;
pub(crate) use replay::expected_steps;

pub fn preflight<D>(engine: &Int8CohortEngine<'_>, requests: &[Int8CohortRequest<'_, D>],
    budget: Int8CohortBudget, limits: Int8PrefillLimits) -> Result<Int8CohortRequirements, Int8GenerationError> {
    limits.validate()?; validate_requests(requests)?; engine.check_idle()?;
    let slots = engine.sequence_count();
    if slots == 0 || slots > requests.len() { return Err(StrictInt8Error::Input.into()); }
    let mut required = Int8CohortRequirements { planned_work: Int8Work::default(), kv_bytes: 0, sampler_bytes: 0 };
    let mut largest_sampler = 0;
    for slot in 0..slots {
        required.kv_bytes = required.kv_bytes.checked_add((engine.capacity(slot)? as u64)
            .checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(StrictInt8Error::Memory)?).ok_or(StrictInt8Error::Memory)?;
    }
    for request in requests {
        check_model(request.admitted_identity, engine.artifact_identity())?;
        // Completion order is unknown: every request must fit EVERY recyclable
        // slot, including its complete physical capacity and task KV ceiling.
        for slot in 0..slots {
            check_bounds(request.plan.work, request.plan.sampler_bytes(), engine.capacity(slot)?, request.budget)?;
        }
        required.planned_work = required.planned_work.checked_add(request.plan.work)?;
        largest_sampler = largest_sampler.max(request.plan.sampler_bytes());
    }
    required.sampler_bytes = sampler_bound(largest_sampler, slots)?;
    check_aggregate(required, budget)?; Ok(required)
}
fn sampler_bound(largest: u64, slots: usize) -> Result<u64, Int8GenerationError> {
    if slots == 0 || slots > MAX_BATCH_ROWS { return Err(StrictInt8Error::Input.into()); }
    largest.checked_mul(slots as u64).ok_or_else(|| StrictInt8Error::Memory.into())
}
pub fn execute<D: DecodeByteDecoder, C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>,
    requests: &[Int8CohortRequest<'_, D>], budget: Int8CohortBudget, limits: Int8PrefillLimits, control: &mut C)
    -> Result<Int8GenerationCohortRun, Int8GenerationError> {
    execute_with_sink(engine, requests, budget, limits, &mut Discard, control)
}
/// The entire finite epoch is preflighted before a sampler or native session.
/// Completed results remain private until ALL rows finish. Provisional events
/// on any error must not be interpreted as terminal success or retried.
pub fn execute_with_sink<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl>(
    engine: &mut Int8CohortEngine<'_>, requests: &[Int8CohortRequest<'_, D>], budget: Int8CohortBudget,
    limits: Int8PrefillLimits, sink: &mut S, control: &mut C)
    -> Result<Int8GenerationCohortRun, Int8GenerationError> {
    preflight(engine, requests, budget, limits)?;
    let slots = engine.sequence_count();
    let mut budgets = reserved(requests.len())?;
    for request in requests { budgets.push(Int8RunBudget::exact(request.plan.work)); }
    let mut session = engine.refill_session(&budgets, control)?;
    drive(requests, slots, limits, sink, &mut session)
}
struct Active<'a> { request: usize, cursor: cursor::Cursor<'a> }
fn fill<'a, D, B: RefillDriver>(requests: &[Int8CohortRequest<'a, D>], slots: &mut [Option<Active<'a>>],
    queued: &mut usize, driver: &mut B) -> Result<(), Int8GenerationError> {
    for (slot, active) in slots.iter_mut().enumerate() {
        if active.is_some() || *queued == requests.len() { continue; }
        let request = &requests[*queued];
        let cursor = cursor::Cursor::new(&request.plan.plan, request.request_seq, INT8_GENERATION_VERSION)?;
        let expected = RefillAdmission { sequence: slot, request: *queued };
        if driver.admit_next()? != Some(expected) { return Err(Int8GenerationError::WorkMismatch); }
        *active = Some(Active { request: *queued, cursor }); *queued += 1;
    }
    Ok(())
}
fn drive<D: DecodeByteDecoder, S: DecodeEventSink, B: RefillDriver>(requests: &[Int8CohortRequest<'_, D>],
    slot_count: usize, limits: Int8PrefillLimits, sink: &mut S, driver: &mut B)
    -> Result<Int8GenerationCohortRun, Int8GenerationError> {
    let result = (|| {
        limits.validate()?; validate_requests(requests)?;
        if slot_count == 0 || slot_count > requests.len() { return Err(StrictInt8Error::Input.into()); }
        let mut slots = reserved(slot_count)?; slots.resize_with(slot_count, || None);
        let mut completed = reserved(requests.len())?; completed.resize_with(requests.len(), || None);
        let mut planned_work = Int8Work::default();
        for request in requests { planned_work = planned_work.checked_add(request.plan.work)?; }
        let mut queued = 0; let mut next_slot = 0; let mut group_steps = 0_u64;
        fill(requests, &mut slots, &mut queued, driver)?;
        while slots.iter().any(Option::is_some) {
            let mut available = [0_usize; MAX_BATCH_ROWS]; let mut feedback = [0_u32; MAX_BATCH_ROWS];
            for (slot, active) in slots.iter().enumerate() {
                if let Some(active) = active {
                    let row = &active.cursor; feedback[slot] = row.next_token()?.0;
                    available[slot] = if row.prompt_cursor < row.plan.prompt.len() { row.plan.prompt.len() - row.prompt_cursor } else { 1 };
                }
            }
            let (counts, following) = schedule::allocate(&available[..slot_count], next_slot, limits.max_batch_rows)?;
            let mut runs = reserved(slot_count)?; let mut selections = reserved(slot_count)?;
            for (slot, active) in slots.iter().enumerate() {
                let Some(active) = active else { continue }; let row = &active.cursor;
                let count = counts[slot]; if count == 0 { continue; }
                row.before_forward(driver.control())?;
                let prompt = row.prompt_cursor < row.plan.prompt.len();
                let tokens = if prompt { &row.plan.prompt[row.prompt_cursor..row.prompt_cursor + count] }
                    else { &feedback[slot..slot + 1] };
                runs.push(CohortTokenRun { sequence: slot, tokens });
                if !prompt || row.prompt_cursor + count == row.plan.prompt.len() { selections.push(slot); }
            }
            driver.append_packed(&runs, limits)?; drop(runs);
            group_steps = group_steps.checked_add(1).ok_or(Int8GenerationError::WorkMismatch)?;
            let logits = if selections.is_empty() { Vec::new() } else { driver.logits_group(&selections)? };
            if logits.len() != selections.len() * NANBEIGE_VOCAB_SIZE { return Err(Int8GenerationError::WorkMismatch); }
            for row in logits.chunks_exact(NANBEIGE_VOCAB_SIZE) { check_logits(row)?; }
            for (slot, active) in slots.iter_mut().enumerate() {
                if let Some(active) = active {
                    for offset in 0..counts[slot] {
                        active.cursor.record_forward(offset + 1 == counts[slot] && selections.binary_search(&slot).is_ok())?;
                    }
                }
            }
            for (&slot, logits) in selections.iter().zip(logits.chunks_exact(NANBEIGE_VOCAB_SIZE)) {
                let active = slots[slot].as_mut().ok_or(Int8GenerationError::WorkMismatch)?;
                active.cursor.emit_next(logits, requests[active.request].decoder, sink, driver.control())?;
            }
            // Finish/drop the old cursor workspace BEFORE admitting a new one.
            // Store by immutable input index, never request_seq sort or slot.
            for (slot, active) in slots.iter_mut().enumerate() {
                if active.as_ref().is_some_and(|active| active.cursor.done) {
                    let active = active.take().ok_or(Int8GenerationError::WorkMismatch)?;
                    let sequence = active.cursor.finish()?; let retired = driver.retire(slot)?;
                    if retired.request != active.request || completed[active.request].is_some() {
                        return Err(Int8GenerationError::WorkMismatch);
                    }
                    check_completed(sequence.native_work, retired.work)?;
                    completed[active.request] = Some(Int8GenerationRun { schema_version: 1, sequence, model_work: retired.work });
                }
            }
            next_slot = following;
            fill(requests, &mut slots, &mut queued, driver)?;
        }
        if queued != requests.len() { return Err(Int8GenerationError::WorkMismatch); }
        let mut sequences = reserved(requests.len())?; let mut actual = Int8Work::default();
        let mut prompts = reserved(requests.len())?; let mut positions = reserved(requests.len())?;
        for (request, row) in requests.iter().zip(completed) {
            let row = row.ok_or(Int8GenerationError::WorkMismatch)?;
            actual = actual.checked_add(row.model_work)?;
            prompts.push(request.plan.prompt_tokens()); positions.push(row.model_work.forward_positions); sequences.push(row);
        }
        if actual != driver.completed_work()? || expected_steps(&prompts, &positions, slot_count, limits.max_batch_rows)? != group_steps {
            return Err(Int8GenerationError::WorkMismatch);
        }
        Ok(Int8GenerationCohortRun { schema_version: 1, execution: INT8_REFILL_EXECUTION.into(), sequences,
            group_steps, planned_work, model_work: actual })
    })();
    if result.is_err() { driver.abort(); } result
}
trait RefillDriver {
    type Control: DecodeStepControl;
    fn control(&mut self) -> &mut Self::Control;
    fn admit_next(&mut self) -> Result<Option<RefillAdmission>, Int8GenerationError>;
    fn append_packed(&mut self, runs: &[CohortTokenRun<'_>], limits: Int8PrefillLimits) -> Result<(), Int8GenerationError>;
    fn logits_group(&mut self, slots: &[usize]) -> Result<Vec<f32>, Int8GenerationError>;
    fn retire(&mut self, slot: usize) -> Result<RefillRetirement, Int8GenerationError>;
    fn completed_work(&self) -> Result<Int8Work, Int8GenerationError>;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> RefillDriver for Int8RefillSession<'_, '_, C> {
    type Control = C;
    fn control(&mut self) -> &mut C { Int8RefillSession::control(self) }
    fn admit_next(&mut self) -> Result<Option<RefillAdmission>, Int8GenerationError> { Int8RefillSession::admit_next(self).map_err(Into::into) }
    fn append_packed(&mut self, runs: &[CohortTokenRun<'_>], limits: Int8PrefillLimits) -> Result<(), Int8GenerationError> {
        Int8RefillSession::append_packed(self, runs, limits).map_err(Into::into)
    }
    fn logits_group(&mut self, slots: &[usize]) -> Result<Vec<f32>, Int8GenerationError> {
        Int8RefillSession::logits_group(self, slots, LinearRows::All).map_err(Into::into)
    }
    fn retire(&mut self, slot: usize) -> Result<RefillRetirement, Int8GenerationError> { Int8RefillSession::retire(self, slot).map_err(Into::into) }
    fn completed_work(&self) -> Result<Int8Work, Int8GenerationError> { Int8RefillSession::completed_work(self).map_err(Into::into) }
    fn abort(&mut self) { Int8RefillSession::abort(self); }
}
#[cfg(test)] mod tests;
