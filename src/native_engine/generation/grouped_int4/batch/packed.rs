//! Fair prompt/decode token packs for the existing INT4 batch driver.
//! Reuses the INT8 path's data-only round-robin allocation and receipt replay;
//! INT4 retains its own model/work identity, real native session and group scales.
use super::*;
use crate::native_engine::{
    strict_int4::{prefill::Int4PrefillLimits, cohort::packed::{Int4TokenRun, INT4_PACKED_COHORT_EXECUTION}},
    generation::quantized::cohort::packed::schedule,
};

pub fn requirements(requests: &[Int4BatchGenerationRequest<'_>], contexts: &[usize],
    budget: Int4BatchGenerationBudget, limits: Int4PrefillLimits)
    -> Result<Int4BatchGenerationRequirements, Int4GenerationError> {
    limits.validate()?;
    add_workspace(super::requirements(requests, contexts, budget)?, budget, limits)
}
pub fn preflight(engine: &Int4CohortEngine<'_>, requests: &[Int4BatchGenerationRequest<'_>],
    budget: Int4BatchGenerationBudget, limits: Int4PrefillLimits)
    -> Result<Int4BatchGenerationRequirements, Int4GenerationError> {
    limits.validate()?;
    add_workspace(super::preflight(engine, requests, budget)?, budget, limits)
}
fn add_workspace(mut need: Int4BatchGenerationRequirements, budget: Int4BatchGenerationBudget,
    limits: Int4PrefillLimits) -> Result<Int4BatchGenerationRequirements, Int4GenerationError> {
    let extra = limits.validate()?;
    need.native.scratch_payload_bound = plus(need.native.scratch_payload_bound, extra)?;
    need.native_payload_bytes = plus(need.native_payload_bytes, extra)?;
    if need.native_payload_bytes > budget.max_native_payload_bytes { return Err(StrictInt4Error::Memory.into()); }
    Ok(need)
}

pub fn execute<D: DecodeByteDecoder, C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4BatchGenerationRequest<'_>], budget: Int4BatchGenerationBudget,
    limits: Int4PrefillLimits, decoder: &D, control: &mut C)
    -> Result<Int4BatchGenerationOutput, Int4GenerationError> {
    execute_with_sink(engine, requests, budget, limits, decoder, &mut Discard, control)
}
#[allow(clippy::too_many_arguments)]
pub fn execute_with_sink<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl>(
    engine: &mut Int4CohortEngine<'_>, requests: &[Int4BatchGenerationRequest<'_>],
    budget: Int4BatchGenerationBudget, limits: Int4PrefillLimits,
    decoder: &D, sink: &mut S, control: &mut C)
    -> Result<Int4BatchGenerationOutput, Int4GenerationError> {
    let required = preflight(engine, requests, budget, limits)?;
    checkpoint(control, 0)?;
    let mut cursors = reserved(requests.len())?; let mut budgets = reserved(requests.len())?;
    for request in requests {
        cursors.push(cursor::Cursor::new(&request.plan.plan, request.request_seq, INT4_GENERATION_VERSION)?);
        budgets.push(Int4RunBudget::exact(request.plan.work));
    }
    let mut session = engine.session(&budgets, control)?;
    drive(cursors, required, limits, &mut session, decoder, sink)
}

struct ScheduledRun<'a> { row: usize, tokens: &'a [u32], project: bool }
struct ForwardedRun { row: usize, position: usize, count: usize, project: bool }
struct PackedOutput { rows: Vec<ForwardedRun>, logits: Vec<f32> }
trait PackedDriver: BatchDriver {
    fn packed_step(&mut self, runs: &[ScheduledRun<'_>], limits: Int4PrefillLimits)
        -> Result<PackedOutput, Int4GenerationError>;
}
impl<C: DecodeStepControl> PackedDriver for Int4CohortSession<'_, '_, C> {
    fn packed_step(&mut self, runs: &[ScheduledRun<'_>], limits: Int4PrefillLimits)
        -> Result<PackedOutput, Int4GenerationError> {
        let mut input = reserved(runs.len())?; let mut rows = reserved(runs.len())?;
        let mut selected = reserved(runs.len())?;
        for run in runs {
            // Include the final head in each row's preflight before any append.
            self.preflight(run.row, run.tokens.len(), if run.project { NANBEIGE_VOCAB_SIZE } else { 0 })?;
            rows.push(ForwardedRun { row: run.row, position: self.position(run.row)?, count: run.tokens.len(), project: run.project });
            input.push(Int4TokenRun { sequence: run.row, tokens: run.tokens });
            if run.project { selected.push(run.row); }
        }
        self.append_packed(&input, limits)?;
        let logits = if selected.is_empty() { Vec::new() } else { self.logits_group(&selected, LinearRows::All)? };
        Ok(PackedOutput { rows, logits })
    }
}

fn drive<D: DecodeByteDecoder, S: DecodeEventSink, B: PackedDriver>(mut cursors: Vec<cursor::Cursor<'_>>,
    required: Int4BatchGenerationRequirements, limits: Int4PrefillLimits,
    driver: &mut B, decoder: &D, sink: &mut S) -> Result<Int4BatchGenerationOutput, Int4GenerationError> {
    let result = (|| {
        limits.validate()?;
        let mut group_steps = 0_u64; let mut next = 0;
        while cursors.iter().any(|row| !row.done) {
            let mut available = [0_usize; MAX_INT4_SEQUENCES];
            let mut feedback = [0_u32; MAX_INT4_SEQUENCES];
            for (slot, row) in cursors.iter().enumerate() {
                if row.done { continue; }
                feedback[slot] = row.next_token()?.0;
                available[slot] = if row.prompt_cursor < row.plan.prompt.len() {
                    row.plan.prompt.len() - row.prompt_cursor
                } else { 1 };
            }
            // Shared deterministic policy: even a width smaller than the number
            // of sequences rotates fairly. Feedback can never receive >1 grant.
            let (counts, following) = schedule::allocate(&available[..cursors.len()], next, limits.max_batch_rows)
                .map_err(|_| Int4GenerationError::WorkMismatch)?;
            let mut runs = reserved(cursors.len())?;
            for (slot, row) in cursors.iter().enumerate() {
                let count = counts[slot]; if count == 0 { continue; }
                row.before_forward(driver.control())?;
                let prompt = row.prompt_cursor < row.plan.prompt.len();
                let tokens = if prompt { &row.plan.prompt[row.prompt_cursor..row.prompt_cursor + count] }
                    else { &feedback[slot..slot + 1] };
                runs.push(ScheduledRun { row: slot, tokens,
                    project: !prompt || row.prompt_cursor + count == row.plan.prompt.len() });
            }
            let output = driver.packed_step(&runs, limits)?;
            group_steps = plus(group_steps, 1)?;
            let projected = runs.iter().filter(|run| run.project).count();
            if output.rows.len() != runs.len() || projected.checked_mul(NANBEIGE_VOCAB_SIZE) != Some(output.logits.len()) {
                return Err(GenerationError::Contract("INT4 packed output shape").into());
            }
            // Validate every native row and every logit before cursor mutation
            // or token delivery. A malformed late row cannot emit an early one.
            for (run, row) in runs.iter().zip(&output.rows) {
                if row.row != run.row || row.count != run.tokens.len() || row.project != run.project
                    || row.position as u64 != cursors[run.row].output.native_work.forward_positions {
                    return Err(GenerationError::Contract("INT4 packed output routing").into());
                }
            }
            for logits in output.logits.chunks_exact(NANBEIGE_VOCAB_SIZE) { check_logits(logits)?; }
            drop(runs);
            let mut head = 0;
            for forwarded in output.rows {
                let row = &mut cursors[forwarded.row];
                for offset in 0..forwarded.count {
                    row.record_forward(forwarded.project && offset + 1 == forwarded.count)?;
                }
                if forwarded.project {
                    row.emit_next(&output.logits[head..head + NANBEIGE_VOCAB_SIZE], decoder, sink, driver.control())?;
                    head += NANBEIGE_VOCAB_SIZE;
                }
            }
            next = following;
        }
        let mut runs = reserved(cursors.len())?; let mut prompts = reserved(cursors.len())?;
        let mut positions = reserved(cursors.len())?; let mut actual = Int4Work::default();
        for (index, row) in cursors.into_iter().enumerate() {
            prompts.push(row.plan.prompt.len());
            let sequence = row.finish()?; let work = driver.work(index)?;
            check_completed(required.group_width, sequence.native_work, work)?;
            positions.push(work.forward_positions); actual = actual.checked_add(work)?;
            runs.push(Int4GenerationRun { schema_version: 1, sequence, model_work: work });
        }
        check_work(actual, Int4RunBudget::exact(required.planned_work))?;
        if actual.projected_logits > required.planned_work.projected_logits
            || expected_group_steps(&prompts, &positions, limits.max_batch_rows)? != group_steps {
            return Err(Int4GenerationError::WorkMismatch);
        }
        Ok(Int4BatchGenerationOutput { schema_version: 1, execution: INT4_PACKED_COHORT_EXECUTION,
            runs, group_steps, planned_work: required.planned_work, actual_work: actual })
    })();
    if result.is_err() { driver.abort(); }
    result
}

pub(crate) fn expected_group_steps(prompts: &[usize], forwards: &[u64], width: usize)
    -> Result<u64, Int4GenerationError> {
    schedule::expected_group_steps(prompts, forwards, width).map_err(|_| Int4GenerationError::WorkMismatch)
}
#[cfg(test)] mod tests;
pub mod refill;

