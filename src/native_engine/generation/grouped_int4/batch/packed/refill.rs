//! Exact existing INT4 generation cursors with bounded FIFO slot refill.
//! Request indices own semantic identity/results; physical slots own reusable KV.
//! A completed row retires immediately and the next immutable ticket may enter.
use super::*;
use crate::native_engine::strict_int4::cohort::packed::refill::{
    Int4RefillSession, RefillAdmission, RefillRetirement, INT4_REFILL_EXECUTION,
};

pub fn preflight(engine: &Int4CohortEngine<'_>, requests: &[Int4BatchGenerationRequest<'_>],
    budget: Int4BatchGenerationBudget, limits: Int4PrefillLimits)
    -> Result<Int4BatchGenerationRequirements, Int4GenerationError> {
    limits.validate()?;
    engine.check_idle()?;
    let slots = engine.sequence_count();
    if requests.is_empty() || requests.len() > 1_000_000 || slots == 0
        || slots > requests.len() || slots > MAX_INT4_SEQUENCES {
        return Err(GenerationError::Contract("INT4 refill geometry").into());
    }
    let first = requests[0].plan;
    let mut planned = Int4Work::default();
    let mut output = 0_u64;
    let mut largest_sampler = 0_u64;
    let mut contexts = reserved(slots)?;
    for slot in 0..slots { contexts.push(engine.capacity(slot)?); }
    for (index, request) in requests.iter().enumerate() {
        request.plan.verify_identity(request.admitted_identity)?;
        if request.plan.group_width != first.group_width
            || !super::super::same_model(first.execution_identity(), request.admitted_identity) {
            return Err(Int4GenerationError::ModelIdentity);
        }
        if request.request_seq == 0
            || requests[..index].iter().any(|prior| prior.request_seq == request.request_seq) {
            return Err(GenerationError::Contract("INT4 refill delivery sequence").into());
        }
        check_model(request.admitted_identity, request.plan.group_width, engine.model_identity())?;
        // Completion order is data-dependent. Every request must fit every
        // recyclable physical slot, not merely the slot initially assigned.
        for &context in &contexts {
            check_bounds(request.plan.work, request.plan.sampler_bytes(), context, request.budget)?;
        }
        planned = planned.checked_add(request.plan.work)?;
        largest_sampler = largest_sampler.max(request.plan.sampler_bytes());
        let options = request.plan.options();
        let ids_scores = times(options.max_new_tokens as u64,
            8 + if options.capture_logprobs { 4 } else { 0 })?;
        output = plus(output, plus(times(options.max_output_bytes as u64, 3)?, ids_scores)?)?;
        output = plus(output, 4096)?;
    }
    check_work(planned, budget.native)?;
    let mut native = Int4MemoryRequirement::for_cohort(&contexts, first.group_width)?;
    let extra = limits.validate()?;
    native.scratch_payload_bound = plus(native.scratch_payload_bound, extra)?;
    let native_payload_bytes = plus(plus(native.kv_bytes, native.rope_bytes)?, native.scratch_payload_bound)?;
    let sampler_payload_bytes = times(largest_sampler, slots as u64)?;
    if native_payload_bytes > budget.max_native_payload_bytes
        || sampler_payload_bytes > budget.max_sampler_payload_bytes
        || output > budget.max_output_payload_bytes {
        return Err(StrictInt4Error::Memory.into());
    }
    Ok(Int4BatchGenerationRequirements {
        planned_work: planned, group_width: first.group_width, native,
        native_payload_bytes, sampler_payload_bytes, output_payload_upper_bytes: output,
    })
}

pub fn execute<D: DecodeByteDecoder, C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4BatchGenerationRequest<'_>], budget: Int4BatchGenerationBudget,
    limits: Int4PrefillLimits, decoder: &D, control: &mut C)
    -> Result<Int4BatchGenerationOutput, Int4GenerationError> {
    execute_with_sink(engine, requests, budget, limits, decoder, &mut Discard, control)
}

pub fn execute_with_sink<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl>(
    engine: &mut Int4CohortEngine<'_>, requests: &[Int4BatchGenerationRequest<'_>],
    budget: Int4BatchGenerationBudget, limits: Int4PrefillLimits,
    decoder: &D, sink: &mut S, control: &mut C)
    -> Result<Int4BatchGenerationOutput, Int4GenerationError> {
    let required = preflight(engine, requests, budget, limits)?;
    checkpoint(control, 0)?;
    let slots = engine.sequence_count();
    let mut budgets = reserved(requests.len())?;
    for request in requests { budgets.push(Int4RunBudget::exact(request.plan.work)); }
    let mut session = engine.refill_session(&budgets, control)?;
    drive(requests, required, slots, limits, decoder, sink, &mut session)
}

struct Active<'a> { request: usize, cursor: cursor::Cursor<'a> }

fn fill<'a, B: RefillDriver>(requests: &'a [Int4BatchGenerationRequest<'a>],
    slots: &mut [Option<Active<'a>>], queued: &mut usize, driver: &mut B)
    -> Result<(), Int4GenerationError> {
    for (slot, active) in slots.iter_mut().enumerate() {
        if active.is_some() || *queued == requests.len() { continue; }
        let request = &requests[*queued];
        let cursor = cursor::Cursor::new(&request.plan.plan, request.request_seq, INT4_GENERATION_VERSION)?;
        let expected = RefillAdmission { sequence: slot, request: *queued };
        if driver.admit_next()? != Some(expected) { return Err(Int4GenerationError::WorkMismatch); }
        *active = Some(Active { request: *queued, cursor });
        *queued += 1;
    }
    Ok(())
}

fn drive<'a, D: DecodeByteDecoder, S: DecodeEventSink, B: RefillDriver>(
    requests: &'a [Int4BatchGenerationRequest<'a>], required: Int4BatchGenerationRequirements,
    slot_count: usize, limits: Int4PrefillLimits, decoder: &D, sink: &mut S, driver: &mut B)
    -> Result<Int4BatchGenerationOutput, Int4GenerationError> {
    let result = (|| {
        limits.validate()?;
        if slot_count == 0 || slot_count > requests.len() { return Err(StrictInt4Error::Input.into()); }
        let mut slots = reserved(slot_count)?; slots.resize_with(slot_count, || None);
        let mut completed = reserved(requests.len())?; completed.resize_with(requests.len(), || None);
        let mut queued = 0_usize; let mut next_slot = 0_usize; let mut group_steps = 0_u64;
        fill(requests, &mut slots, &mut queued, driver)?;
        while slots.iter().any(Option::is_some) {
            let mut available = [0_usize; MAX_INT4_SEQUENCES];
            let mut feedback = [0_u32; MAX_INT4_SEQUENCES];
            for (slot, active) in slots.iter().enumerate() {
                let Some(active) = active else { continue; };
                feedback[slot] = active.cursor.next_token()?.0;
                available[slot] = if active.cursor.prompt_cursor < active.cursor.plan.prompt.len() {
                    active.cursor.plan.prompt.len() - active.cursor.prompt_cursor
                } else { 1 };
            }
            let (counts, following) = schedule::allocate(&available[..slot_count], next_slot, limits.max_batch_rows)
                .map_err(|_| Int4GenerationError::WorkMismatch)?;
            let mut native = reserved(slot_count)?; let mut selections = reserved(slot_count)?;
            for (slot, active) in slots.iter().enumerate() {
                let Some(active) = active else { continue; };
                let count = counts[slot]; if count == 0 { continue; }
                active.cursor.before_forward(driver.control())?;
                let prompt = active.cursor.prompt_cursor < active.cursor.plan.prompt.len();
                let tokens = if prompt {
                    &active.cursor.plan.prompt[active.cursor.prompt_cursor..active.cursor.prompt_cursor + count]
                } else { &feedback[slot..slot + 1] };
                let project = !prompt || active.cursor.prompt_cursor + count == active.cursor.plan.prompt.len();
                native.push(Int4TokenRun { sequence: slot, tokens });
                if project { selections.push(slot); }
            }
            driver.append_packed(&native, limits)?;
            group_steps = plus(group_steps, 1)?;
            let logits = if selections.is_empty() { Vec::new() } else { driver.logits_group(&selections)? };
            if logits.len() != selections.len() * NANBEIGE_VOCAB_SIZE {
                return Err(Int4GenerationError::WorkMismatch);
            }
            for row in logits.chunks_exact(NANBEIGE_VOCAB_SIZE) { check_logits(row)?; }
            for (slot, active) in slots.iter_mut().enumerate() {
                let Some(active) = active else { continue; };
                for offset in 0..counts[slot] {
                    let projected = offset + 1 == counts[slot] && selections.binary_search(&slot).is_ok();
                    active.cursor.record_forward(projected)?;
                }
            }
            for (&slot, logits) in selections.iter().zip(logits.chunks_exact(NANBEIGE_VOCAB_SIZE)) {
                let active = slots[slot].as_mut().ok_or(Int4GenerationError::WorkMismatch)?;
                active.cursor.emit_next(logits, decoder, sink, driver.control())?;
            }
            // Finish and drop cursor-owned bytes before exposing the physical
            // slot to the next immutable ticket.
            for (slot, active) in slots.iter_mut().enumerate() {
                if active.as_ref().is_some_and(|active| active.cursor.done) {
                    let active = active.take().ok_or(Int4GenerationError::WorkMismatch)?;
                    let sequence = active.cursor.finish()?;
                    let retired = driver.retire(slot)?;
                    if retired.request != active.request || completed[active.request].is_some() {
                        return Err(Int4GenerationError::WorkMismatch);
                    }
                    check_completed(required.group_width, sequence.native_work, retired.work)?;
                    completed[active.request] = Some(Int4GenerationRun {
                        schema_version: 1, sequence, model_work: retired.work,
                    });
                }
            }
            next_slot = following;
            fill(requests, &mut slots, &mut queued, driver)?;
        }
        if queued != requests.len() { return Err(Int4GenerationError::WorkMismatch); }
        let mut runs = reserved(requests.len())?; let mut actual = Int4Work::default();
        let mut prompts = reserved(requests.len())?; let mut positions = reserved(requests.len())?;
        for (request, row) in requests.iter().zip(completed) {
            let row = row.ok_or(Int4GenerationError::WorkMismatch)?;
            actual = actual.checked_add(row.model_work)?;
            prompts.push(request.plan.prompt_tokens());
            positions.push(row.model_work.forward_positions);
            runs.push(row);
        }
        if actual != driver.completed_work()? || actual.projected_logits > required.planned_work.projected_logits
            || expected_steps(&prompts, &positions, slot_count, limits.max_batch_rows)? != group_steps {
            return Err(Int4GenerationError::WorkMismatch);
        }
        check_work(actual, Int4RunBudget::exact(required.planned_work))?;
        Ok(Int4BatchGenerationOutput {
            schema_version: 1, execution: INT4_REFILL_EXECUTION, runs, group_steps,
            planned_work: required.planned_work, actual_work: actual,
        })
    })();
    if result.is_err() { driver.abort(); }
    result
}

fn expected_steps(prompts: &[usize], forwards: &[u64], slots: usize, width: usize)
    -> Result<u64, Int4GenerationError> {
    if prompts.is_empty() || prompts.len() != forwards.len() || slots == 0 || slots > prompts.len()
        || slots > MAX_INT4_SEQUENCES || width == 0 || width > MAX_INT4_SEQUENCES {
        return Err(Int4GenerationError::WorkMismatch);
    }
    // Replay the same immutable FIFO ticket order and round-robin physical-slot
    // scheduler from lengths only. Completion immediately admits the next ticket.
    let mut active = [None; MAX_INT4_SEQUENCES];
    let mut positions = [0_usize; MAX_INT4_SEQUENCES];
    let mut queued = 0_usize; let mut next = 0_usize; let mut steps = 0_u64;
    for slot in 0..slots {
        active[slot] = Some(queued); queued += 1;
    }
    while active[..slots].iter().any(Option::is_some) {
        let mut available = [0_usize; MAX_INT4_SEQUENCES];
        for slot in 0..slots {
            let Some(request) = active[slot] else { continue; };
            let limit = usize::try_from(forwards[request]).map_err(|_| Int4GenerationError::WorkMismatch)?;
            if prompts[request] == 0 || prompts[request] > limit || positions[slot] >= limit {
                return Err(Int4GenerationError::WorkMismatch);
            }
            available[slot] = if positions[slot] < prompts[request] {
                prompts[request] - positions[slot]
            } else { 1 };
        }
        let (counts, following) = schedule::allocate(&available[..slots], next, width)
            .map_err(|_| Int4GenerationError::WorkMismatch)?;
        for slot in 0..slots {
            let Some(request) = active[slot] else { continue; };
            positions[slot] += counts[slot];
            let limit = usize::try_from(forwards[request]).map_err(|_| Int4GenerationError::WorkMismatch)?;
            if positions[slot] == limit {
                if queued < prompts.len() {
                    active[slot] = Some(queued); positions[slot] = 0; queued += 1;
                } else { active[slot] = None; positions[slot] = 0; }
            }
        }
        steps = steps.checked_add(1).ok_or(Int4GenerationError::WorkMismatch)?;
        next = following;
    }
    if queued != prompts.len() { return Err(Int4GenerationError::WorkMismatch); }
    Ok(steps)
}

trait RefillDriver {
    type Control: DecodeStepControl;
    fn control(&mut self) -> &mut Self::Control;
    fn admit_next(&mut self) -> Result<Option<RefillAdmission>, Int4GenerationError>;
    fn append_packed(&mut self, runs: &[Int4TokenRun<'_>], limits: Int4PrefillLimits)
        -> Result<(), Int4GenerationError>;
    fn logits_group(&mut self, slots: &[usize]) -> Result<Vec<f32>, Int4GenerationError>;
    fn retire(&mut self, slot: usize) -> Result<RefillRetirement, Int4GenerationError>;
    fn completed_work(&self) -> Result<Int4Work, Int4GenerationError>;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> RefillDriver for Int4RefillSession<'_, '_, C> {
    type Control = C;
    fn control(&mut self) -> &mut C { Int4RefillSession::control(self) }
    fn admit_next(&mut self) -> Result<Option<RefillAdmission>, Int4GenerationError> {
        Int4RefillSession::admit_next(self).map_err(Into::into)
    }
    fn append_packed(&mut self, runs: &[Int4TokenRun<'_>], limits: Int4PrefillLimits)
        -> Result<(), Int4GenerationError> {
        Int4RefillSession::append_packed(self, runs, limits).map_err(Into::into)
    }
    fn logits_group(&mut self, slots: &[usize]) -> Result<Vec<f32>, Int4GenerationError> {
        Int4RefillSession::logits_group(self, slots, LinearRows::All).map_err(Into::into)
    }
    fn retire(&mut self, slot: usize) -> Result<RefillRetirement, Int4GenerationError> {
        Int4RefillSession::retire(self, slot).map_err(Into::into)
    }
    fn completed_work(&self) -> Result<Int4Work, Int4GenerationError> {
        Int4RefillSession::completed_work(self).map_err(Into::into)
    }
    fn abort(&mut self) { Int4RefillSession::abort(self); }
}

#[cfg(test)] mod tests;
