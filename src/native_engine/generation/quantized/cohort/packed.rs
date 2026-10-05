//! Fair bounded token packs mixing prompt morsels and ordinary decode rows.
//! Uses the existing cursor, addressed sampler, decoder, events and row budgets.
//! A document cannot consume a feedback token until its previous token has been
//! selected. No whole-prompt barrier, new admission, worker team or retry exists.
use super::*;
use crate::native_engine::strict_int8::{prefill::Int8PrefillLimits,
    cohort::packed::{CohortTokenRun, INT8_PACKED_COHORT_EXECUTION}};
mod schedule;
pub(crate) use schedule::expected_group_steps;

pub fn execute<D: DecodeByteDecoder, C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>,
    requests: &[Int8CohortRequest<'_, D>], budget: Int8CohortBudget, limits: Int8PrefillLimits,
    control: &mut C) -> Result<Int8GenerationCohortRun, Int8GenerationError> {
    execute_with_sink(engine, requests, budget, limits, &mut Discard, control)
}
/// In addition to the resident cohort, the caller admits the complete extra
/// packed workspace. Every request and this declaration are checked BEFORE
/// opening the exclusive session. Each row retains its exact original quota.
pub fn execute_with_sink<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl>(
    engine: &mut Int8CohortEngine<'_>, requests: &[Int8CohortRequest<'_, D>], budget: Int8CohortBudget,
    limits: Int8PrefillLimits, sink: &mut S, control: &mut C)
    -> Result<Int8GenerationCohortRun, Int8GenerationError> {
    limits.validate()?;
    super::preflight(engine, requests, budget)?;
    let mut budgets = reserved(requests.len())?;
    for request in requests { budgets.push(Int8RunBudget::exact(request.plan.work)); }
    let mut session = engine.session(&budgets, control)?;
    drive(requests, limits, sink, &mut session)
}

fn drive<D: DecodeByteDecoder, S: DecodeEventSink, B: PackedDriver>(requests: &[Int8CohortRequest<'_, D>],
    limits: Int8PrefillLimits, sink: &mut S, driver: &mut B)
    -> Result<Int8GenerationCohortRun, Int8GenerationError> {
    let result = (|| {
        limits.validate()?; validate_requests(requests)?;
        let mut cursors = reserved(requests.len())?;
        let mut planned_work = Int8Work::default();
        for request in requests {
            cursors.push(cursor::Cursor::new(&request.plan.plan, request.request_seq, INT8_GENERATION_VERSION)?);
            planned_work = planned_work.checked_add(request.plan.work)?;
        }
        let mut next = 0;
        let mut group_steps = 0_u64;
        while cursors.iter().any(|row| !row.done) {
            let mut available = [0_usize; MAX_BATCH_ROWS];
            let mut feedback = [0_u32; MAX_BATCH_ROWS];
            for (slot, row) in cursors.iter().enumerate() {
                if row.done { continue; }
                feedback[slot] = row.next_token()?.0;
                available[slot] = if row.prompt_cursor < row.plan.prompt.len() {
                    row.plan.prompt.len() - row.prompt_cursor
                } else { 1 };
            }
            let (counts, following) = schedule::allocate(&available[..cursors.len()], next, limits.max_batch_rows)?;
            let mut runs = reserved(cursors.len())?;
            let mut selections = reserved(cursors.len())?;
            // Physical runs are sorted for the native address checker, but the
            // fair cursor rotates across packs, including widths < document count.
            for (slot, row) in cursors.iter().enumerate() {
                let count = counts[slot];
                if count == 0 { continue; }
                row.before_forward(driver.control())?;
                let prompt = row.prompt_cursor < row.plan.prompt.len();
                let tokens = if prompt { &row.plan.prompt[row.prompt_cursor..row.prompt_cursor + count] }
                    else { &feedback[slot..slot + 1] };
                runs.push(CohortTokenRun { sequence: slot, tokens });
                if !prompt || row.prompt_cursor + count == row.plan.prompt.len() { selections.push(slot); }
            }
            driver.append_packed(&runs, limits)?;
            drop(runs);
            group_steps = group_steps.checked_add(1).ok_or(Int8GenerationError::WorkMismatch)?;
            let logits = if selections.is_empty() { Vec::new() } else { driver.logits_group(&selections)? };
            if logits.len() != selections.len() * NANBEIGE_VOCAB_SIZE { return Err(Int8GenerationError::WorkMismatch); }
            for row in logits.chunks_exact(NANBEIGE_VOCAB_SIZE) { check_logits(row)?; }
            for (slot, row) in cursors.iter_mut().enumerate() {
                for offset in 0..counts[slot] {
                    let projected = offset + 1 == counts[slot] && selections.binary_search(&slot).is_ok();
                    row.record_forward(projected)?;
                }
            }
            for (&slot, logits) in selections.iter().zip(logits.chunks_exact(NANBEIGE_VOCAB_SIZE)) {
                cursors[slot].emit_next(logits, requests[slot].decoder, sink, driver.control())?;
            }
            next = following;
        }
        let mut sequences = reserved(requests.len())?;
        let mut model_work = Int8Work::default();
        let mut prompts = reserved(requests.len())?; let mut positions = reserved(requests.len())?;
        for (slot, row) in cursors.into_iter().enumerate() {
            let sequence = row.finish()?; let work = driver.work(slot)?;
            check_completed(sequence.native_work, work)?;
            prompts.push(requests[slot].plan.prompt_tokens()); positions.push(work.forward_positions);
            model_work = model_work.checked_add(work)?;
            sequences.push(Int8GenerationRun { schema_version: 1, sequence, model_work: work });
        }
        if expected_group_steps(&prompts, &positions, limits.max_batch_rows)? != group_steps {
            return Err(Int8GenerationError::WorkMismatch);
        }
        Ok(Int8GenerationCohortRun { schema_version: 1, execution: INT8_PACKED_COHORT_EXECUTION.to_owned(),
            sequences, group_steps, planned_work, model_work })
    })();
    if result.is_err() { driver.abort(); }
    result
}

/// Private synthetic seam. Public execution accepts only a real cohort engine.
trait PackedDriver: GroupDriver {
    fn append_packed(&mut self, runs: &[CohortTokenRun<'_>], limits: Int8PrefillLimits)
        -> Result<(), Int8GenerationError>;
}
impl<C: DecodeStepControl> PackedDriver for Int8CohortSession<'_, '_, C> {
    fn append_packed(&mut self, runs: &[CohortTokenRun<'_>], limits: Int8PrefillLimits)
        -> Result<(), Int8GenerationError> {
        Int8CohortSession::append_packed(self, runs, limits).map_err(Into::into)
    }
}
#[cfg(test)] mod tests;
