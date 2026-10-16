//! Fixed-cohort INT4 generation over one real ragged layer-major engine.
//! Sampling and token delivery remain in the shared Cursor. Physical slot,
//! batch membership and tick never enter its semantic random address.
use super::*;
use crate::native_engine::strict_int4::{
    Int4MemoryRequirement,
    cohort::{Int4CohortEngine, Int4CohortSession, Int4CohortToken, MAX_INT4_SEQUENCES},
};
pub mod packed;

pub const INT4_BATCH_GENERATION_VERSION: &str = "grouped-int4-ragged-generation-v1";

/// Request order maps to fixed physical slots 0..N. All slots are owned until
/// this finite run ends. Different prompts, seeds, stops and lengths are legal.
pub struct Int4BatchGenerationRequest<'a> {
    pub plan: &'a Int4GenerationPlan,
    pub admitted_identity: &'a ExecutionIdentity,
    pub request_seq: u64,
    pub budget: Int4GenerationBudget,
}
#[derive(Clone, Copy, Debug)]
pub struct Int4BatchGenerationBudget {
    pub native: Int4RunBudget,
    /// All allocated cohort KV, shared RoPE and native scratch, not live tokens.
    /// Weights, allocator reserves and hosted serialization are separate.
    pub max_native_payload_bytes: u64,
    pub max_sampler_payload_bytes: u64,
    pub max_output_payload_bytes: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4BatchGenerationRequirements {
    pub planned_work: Int4Work,
    pub group_width: usize,
    pub native: Int4MemoryRequirement,
    pub native_payload_bytes: u64,
    pub sampler_payload_bytes: u64,
    pub output_payload_upper_bytes: u64,
}
#[derive(Serialize)]
pub struct Int4BatchGenerationOutput {
    pub schema_version: u32,
    pub execution: &'static str,
    /// Input order, regardless of completion order. Each row keeps the existing
    /// INT4 semantic execution label; this wrapper identifies the scheduling path.
    pub runs: Vec<Int4GenerationRun>,
    pub group_steps: u64,
    pub planned_work: Int4Work,
    pub actual_work: Int4Work,
}

/// Pure complete-cohort admission, suitable for use BEFORE engine allocation.
/// Actual model ownership/identity is additionally checked by preflight.
/// No work allowance can be borrowed from another member or renewed per tick.
pub fn requirements(requests: &[Int4BatchGenerationRequest<'_>], contexts: &[usize],
    budget: Int4BatchGenerationBudget) -> Result<Int4BatchGenerationRequirements, Int4GenerationError> {
    if requests.is_empty() || requests.len() > MAX_INT4_SEQUENCES || contexts.len() != requests.len() {
        return Err(GenerationError::Contract("INT4 generation cohort width").into());
    }
    let first = requests[0].plan;
    let mut work = Int4Work::default();
    let mut sampler = 0_u64;
    let mut output = 0_u64;
    for (index, (request, &context)) in requests.iter().zip(contexts).enumerate() {
        request.plan.verify_identity(request.admitted_identity)?;
        if request.plan.group_width != first.group_width
            || !same_model(first.execution_identity(), request.admitted_identity) {
            return Err(Int4GenerationError::ModelIdentity);
        }
        if request.request_seq == 0 || requests[..index].iter().any(|prior| prior.request_seq == request.request_seq) {
            return Err(GenerationError::Contract("INT4 generation delivery sequence").into());
        }
        check_bounds(request.plan.work, request.plan.sampler_bytes(), context, request.budget)?;
        // Sum independent attention triangles; never make one cross-document context.
        work = work.checked_add(request.plan.work)?;
        sampler = plus(sampler, request.plan.sampler_bytes())?;
        let options = request.plan.options();
        let ids_and_scores = times(options.max_new_tokens as u64,
            8 + if options.capture_logprobs { 4 } else { 0 })?;
        // Current/replacement bytes and a pending event; token ids may coexist
        // while committing a proposal. Fixed per-row allowance covers scheduler
        // descriptors, cursor/output headers and the bounded native-budget vector.
        output = plus(output, plus(times(options.max_output_bytes as u64, 3)?, ids_and_scores)?)?;
        output = plus(output, 4096)?;
    }
    let native = Int4MemoryRequirement::for_cohort(contexts, first.group_width)?;
    let native_payload_bytes = plus(plus(native.kv_bytes, native.rope_bytes)?, native.scratch_payload_bound)?;
    check_work(work, budget.native)?;
    if native_payload_bytes > budget.max_native_payload_bytes || sampler > budget.max_sampler_payload_bytes
        || output > budget.max_output_payload_bytes {
        return Err(StrictInt4Error::Memory.into());
    }
    Ok(Int4BatchGenerationRequirements { planned_work: work, group_width: first.group_width, native, native_payload_bytes,
        sampler_payload_bytes: sampler, output_payload_upper_bytes: output })
}

pub fn preflight(engine: &Int4CohortEngine<'_>, requests: &[Int4BatchGenerationRequest<'_>],
    budget: Int4BatchGenerationBudget) -> Result<Int4BatchGenerationRequirements, Int4GenerationError> {
    engine.check_idle()?;
    if requests.len() != engine.sequence_count() || requests.len() > MAX_INT4_SEQUENCES {
        return Err(GenerationError::Contract("INT4 generation slot count").into());
    }
    let mut contexts = [0_usize; MAX_INT4_SEQUENCES];
    for (index, request) in requests.iter().enumerate() {
        contexts[index] = engine.capacity(index)?;
        check_model(request.admitted_identity, request.plan.group_width, engine.model_identity())?;
    }
    requirements(requests, &contexts[..requests.len()], budget)
}

pub fn execute<D: DecodeByteDecoder, C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4BatchGenerationRequest<'_>], budget: Int4BatchGenerationBudget,
    decoder: &D, control: &mut C) -> Result<Int4BatchGenerationOutput, Int4GenerationError> {
    execute_with_sink(engine, requests, budget, decoder, &mut Discard, control)
}

/// One native session owns all rows. Later failure is not a partial success;
/// already committed streaming events are provisional and cannot be retracted.
/// No failed row, delivery or native operation is retried. Session Drop clears
/// every cache on success/error/unwind; execution failures poison the engine.
pub fn execute_with_sink<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl>(
    engine: &mut Int4CohortEngine<'_>, requests: &[Int4BatchGenerationRequest<'_>],
    budget: Int4BatchGenerationBudget, decoder: &D, sink: &mut S, control: &mut C)
    -> Result<Int4BatchGenerationOutput, Int4GenerationError> {
    let required = preflight(engine, requests, budget)?;
    checkpoint(control, 0)?;
    let mut cursors = reserved(requests.len())?;
    let mut budgets = reserved(requests.len())?;
    for request in requests {
        cursors.push(cursor::Cursor::new(&request.plan.plan, request.request_seq, INT4_GENERATION_VERSION)?);
        // Tighten the physical account to the compiled bound, even when the
        // caller supplied excess allowance. Admission never creates refill credit.
        budgets.push(Int4RunBudget::exact(request.plan.work));
    }
    let mut session = engine.session(&budgets, control)?;
    drive(cursors, required, &mut session, decoder, sink)
}

#[derive(Clone, Copy)]
struct Step { row: usize, token: u32, project: bool }
struct Forwarded { row: usize, position: usize, project: bool }
struct StepOutput { rows: Vec<Forwarded>, logits: Vec<f32> }
// Static internal seam only: public APIs cannot substitute an invented model.
trait BatchDriver {
    type Control: DecodeStepControl;
    fn control(&mut self) -> &mut Self::Control;
    fn step(&mut self, tokens: &[Step]) -> Result<StepOutput, Int4GenerationError>;
    fn work(&self, row: usize) -> Result<Int4Work, Int4GenerationError>;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> BatchDriver for Int4CohortSession<'_, '_, C> {
    type Control = C;
    fn control(&mut self) -> &mut C { Int4CohortSession::control(self) }
    fn abort(&mut self) { Int4CohortSession::abort(self); }
    fn work(&self, row: usize) -> Result<Int4Work, Int4GenerationError> {
        Int4CohortSession::work(self, row).map_err(Into::into)
    }
    fn step(&mut self, tokens: &[Step]) -> Result<StepOutput, Int4GenerationError> {
        let mut input = reserved(tokens.len())?;
        let mut selected = reserved(tokens.len())?;
        let mut rows = reserved(tokens.len())?;
        // Every combined forward+head quote is checked before the first append.
        // Independent accounts plus the native aggregate ledger own the totals.
        for token in tokens {
            self.preflight(token.row, 1, if token.project { NANBEIGE_VOCAB_SIZE } else { 0 })?;
            rows.push(Forwarded { row: token.row, position: self.position(token.row)?, project: token.project });
            input.push(Int4CohortToken { sequence: token.row, token: token.token });
            if token.project { selected.push(token.row); }
        }
        self.append_group(&input)?;
        let logits = if selected.is_empty() { Vec::new() }
            else { self.logits_group(&selected, LinearRows::All)? };
        Ok(StepOutput { rows, logits })
    }
}

fn drive<D: DecodeByteDecoder, S: DecodeEventSink, B: BatchDriver>(mut cursors: Vec<cursor::Cursor<'_>>,
    required: Int4BatchGenerationRequirements, driver: &mut B, decoder: &D, sink: &mut S)
    -> Result<Int4BatchGenerationOutput, Int4GenerationError> {
    let result = (|| {
        let mut group_steps = 0_u64;
        loop {
            let mut tokens = reserved(cursors.len())?;
            for (row, cursor) in cursors.iter().enumerate() {
                if cursor.done { continue; }
                cursor.before_forward(driver.control())?;
                let (token, project) = cursor.next_token()?;
                tokens.push(Step { row, token, project });
            }
            if tokens.is_empty() { break; }
            let output = driver.step(&tokens)?;
            group_steps = plus(group_steps, 1)?;
            let projected = tokens.iter().filter(|token| token.project).count();
            if output.rows.len() != tokens.len()
                || projected.checked_mul(NANBEIGE_VOCAB_SIZE) != Some(output.logits.len()) {
                return Err(GenerationError::Contract("INT4 batch output shape").into());
            }
            // Check ALL native routing and logits before emitting any row of
            // this tick. A late malformed row cannot commit an earlier token.
            for (scheduled, forwarded) in tokens.iter().zip(&output.rows) {
                if forwarded.row != scheduled.row || forwarded.project != scheduled.project
                    || forwarded.position as u64 != cursors[scheduled.row].output.native_work.forward_positions {
                    return Err(GenerationError::Contract("INT4 batch output routing").into());
                }
            }
            for logits in output.logits.chunks_exact(NANBEIGE_VOCAB_SIZE) { check_logits(logits)?; }
            let mut head = 0;
            for token in tokens {
                let cursor = &mut cursors[token.row];
                cursor.record_forward(token.project)?;
                if token.project {
                    let logits = &output.logits[head..head + NANBEIGE_VOCAB_SIZE];
                    cursor.emit_next(logits, decoder, sink, driver.control())?;
                    head += NANBEIGE_VOCAB_SIZE;
                }
            }
        }
        let mut runs = reserved(cursors.len())?;
        let mut actual = Int4Work::default();
        for (index, cursor) in cursors.into_iter().enumerate() {
            let sequence = cursor.finish()?;
            let work = driver.work(index)?;
            check_completed(required.group_width, sequence.native_work, work)?;
            actual = actual.checked_add(work)?;
            runs.push(Int4GenerationRun { schema_version: 1, sequence, model_work: work });
        }
        check_work(actual, Int4RunBudget::exact(required.planned_work))?;
        if actual.projected_logits > required.planned_work.projected_logits { return Err(Int4GenerationError::WorkMismatch); }
        Ok(Int4BatchGenerationOutput { schema_version: 1, execution: INT4_BATCH_GENERATION_VERSION,
            runs, group_steps, planned_work: required.planned_work, actual_work: actual })
    })();
    if result.is_err() { driver.abort(); }
    result
}

fn same_model(a: &ExecutionIdentity, b: &ExecutionIdentity) -> bool {
    a.source_revision == b.source_revision && a.logical_model_digest == b.logical_model_digest
        && a.artifact_format == b.artifact_format && a.quant_recipe == b.quant_recipe
        && a.packing_set_digest == b.packing_set_digest && a.tokenizer_digest == b.tokenizer_digest
        && a.template_digest == b.template_digest && a.numerics_profile == b.numerics_profile
        && a.kv_dtype == b.kv_dtype && a.backend_semantic_version == b.backend_semantic_version
        && a.host_class == b.host_class && a.compiler_identity == b.compiler_identity
}
fn check_work(work: Int4Work, budget: Int4RunBudget) -> Result<(), Int4GenerationError> {
    if work.forward_positions > budget.max_forward_positions || work.attention_pairs > budget.max_attention_pairs
        || !work.projections.fits(budget.max_projection_work) { return Err(StrictInt4Error::Work.into()); }
    Ok(())
}
fn plus(a: u64, b: u64) -> Result<u64, Int4GenerationError> { a.checked_add(b).ok_or(StrictInt4Error::Memory.into()) }
fn times(a: u64, b: u64) -> Result<u64, Int4GenerationError> { a.checked_mul(b).ok_or(StrictInt4Error::Memory.into()) }

#[cfg(test)] mod tests;