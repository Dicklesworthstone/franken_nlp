//! Exact existing generation cursors on shared-weight, ragged INT8 cohorts.
//! Physical slots/ticks never enter the request key, sampler or prompt policy.
//! Events are provisional and interleaved by request_seq. Any native, decoder,
//! delivery or cancellation failure aborts the entire cohort without retry.
use super::*;
use crate::native_engine::{portable_int8::batch::MAX_BATCH_ROWS,
    strict_int8::cohort::{CohortToken, Int8CohortEngine, Int8CohortSession, INT8_COHORT_EXECUTION}};

pub struct Int8CohortRequest<'a, D> {
    pub plan: &'a Int8GenerationPlan,
    pub admitted_identity: &'a ExecutionIdentity,
    pub decoder: &'a D,
    pub request_seq: u64,
    pub budget: Int8GenerationBudget,
}
/// Aggregate ceilings, in ADDITION to each row's independent admission. These
/// are not permits. The caller retains preparation and all completed outputs;
/// full grouped native scratch is priced by Int8MemoryRequirement::for_cohort.
#[derive(Clone, Copy, Debug)]
pub struct Int8CohortBudget {
    pub native: Int8RunBudget,
    pub max_kv_bytes: u64,
    pub max_sampler_bytes: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8CohortRequirements {
    pub planned_work: Int8Work,
    pub kv_bytes: u64,
    pub sampler_bytes: u64,
}
#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8GenerationCohortRun {
    pub schema_version: u32,
    pub execution: String,
    /// Input order, not completion order. Per-row semantic execution is the
    /// ordinary INT8 generation version; this wrapper records physical grouping.
    pub sequences: Vec<Int8GenerationRun>,
    pub group_steps: u64,
    pub planned_work: Int8Work,
    pub model_work: Int8Work,
}

pub fn preflight<D>(engine: &Int8CohortEngine<'_>, requests: &[Int8CohortRequest<'_, D>], budget: Int8CohortBudget)
    -> Result<Int8CohortRequirements, Int8GenerationError> {
    validate_requests(requests)?;
    engine.check_idle()?;
    if requests.len() != engine.sequence_count() { return Err(StrictInt8Error::Input.into()); }
    let mut required = Int8CohortRequirements { planned_work: Int8Work::default(), kv_bytes: 0, sampler_bytes: 0 };
    for (slot, request) in requests.iter().enumerate() {
        check_model(request.admitted_identity, engine.artifact_identity())?;
        let capacity = engine.capacity(slot)?;
        check_bounds(request.plan.work, request.plan.sampler_bytes(), capacity, request.budget)?;
        required.planned_work = required.planned_work.checked_add(request.plan.work)?;
        required.kv_bytes = required.kv_bytes.checked_add((capacity as u64)
            .checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(StrictInt8Error::Memory)?).ok_or(StrictInt8Error::Memory)?;
        required.sampler_bytes = required.sampler_bytes.checked_add(request.plan.sampler_bytes()).ok_or(StrictInt8Error::Memory)?;
    }
    check_aggregate(required, budget)?; Ok(required)
}
fn check_aggregate(required: Int8CohortRequirements, budget: Int8CohortBudget) -> Result<(), Int8GenerationError> {
    if required.kv_bytes > budget.max_kv_bytes || required.sampler_bytes > budget.max_sampler_bytes {
        return Err(StrictInt8Error::Memory.into());
    }
    let work = required.planned_work;
    if work.forward_positions > budget.native.max_forward_positions || work.attention_pairs > budget.native.max_attention_pairs
        || !work.projections.fits(budget.native.max_projection_work) { return Err(StrictInt8Error::Work.into()); }
    Ok(())
}
fn validate_requests<D>(requests: &[Int8CohortRequest<'_, D>]) -> Result<(), Int8GenerationError> {
    if requests.is_empty() || requests.len() > MAX_BATCH_ROWS { return Err(StrictInt8Error::Input.into()); }
    for (index, request) in requests.iter().enumerate() {
        request.plan.verify_identity(request.admitted_identity)?;
        if request.request_seq == 0 || requests[..index].iter().any(|other| other.request_seq == request.request_seq) {
            return Err(GenerationError::Contract("cohort delivery sequences must be nonzero and unique").into());
        }
    }
    Ok(())
}
pub fn execute<D: DecodeByteDecoder, C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>,
    requests: &[Int8CohortRequest<'_, D>], budget: Int8CohortBudget, control: &mut C)
    -> Result<Int8GenerationCohortRun, Int8GenerationError> {
    execute_with_sink(engine, requests, budget, &mut Discard, control)
}
/// Admission checks every row before opening a session or creating a sampler.
/// No token/terminal frame is delivered for a failed preflight. After execution
/// begins, already delivered token frames remain provisional on ANY error.
pub fn execute_with_sink<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>,
    requests: &[Int8CohortRequest<'_, D>], budget: Int8CohortBudget, sink: &mut S, control: &mut C)
    -> Result<Int8GenerationCohortRun, Int8GenerationError> {
    preflight(engine, requests, budget)?;
    let mut budgets = reserved(requests.len())?;
    // Exact row work, not M copies of the enclosing aggregate ceiling.
    for request in requests { budgets.push(Int8RunBudget::exact(request.plan.work)); }
    let mut session = engine.session(&budgets, control)?;
    drive(requests, sink, &mut session)
}
fn drive<D: DecodeByteDecoder, S: DecodeEventSink, B: GroupDriver>(requests: &[Int8CohortRequest<'_, D>],
    sink: &mut S, driver: &mut B) -> Result<Int8GenerationCohortRun, Int8GenerationError> {
    let result = (|| {
        validate_requests(requests)?;
        let mut cursors = reserved(requests.len())?;
        let mut planned_work = Int8Work::default();
        for request in requests {
            cursors.push(cursor::Cursor::new(&request.plan.plan, request.request_seq, INT8_GENERATION_VERSION)?);
            planned_work = planned_work.checked_add(request.plan.work)?;
        }
        let mut steps = reserved(requests.len())?; let mut selections = reserved(requests.len())?;
        let mut group_steps = 0_u64;
        while cursors.iter().any(|row| !row.done) {
            steps.clear(); selections.clear();
            for (slot, row) in cursors.iter().enumerate() {
                if row.done { continue; }
                row.before_forward(driver.control())?;
                let (token, selection) = row.next_token()?;
                steps.push(CohortToken { sequence: slot, token });
                if selection { selections.push(slot); }
            }
            driver.append_group(&steps)?;
            group_steps = group_steps.checked_add(1).ok_or(Int8GenerationError::WorkMismatch)?;
            let logits = if selections.is_empty() { Vec::new() } else { driver.logits_group(&selections)? };
            if logits.len() != selections.len() * NANBEIGE_VOCAB_SIZE { return Err(Int8GenerationError::WorkMismatch); }
            // Validate every projected row before emitting any event from this
            // step. Ordinary prompt positions never project the vocabulary.
            for row in logits.chunks_exact(NANBEIGE_VOCAB_SIZE) { check_logits(row)?; }
            for step in &steps { cursors[step.sequence].record_forward(selections.binary_search(&step.sequence).is_ok())?; }
            for (&slot, logits) in selections.iter().zip(logits.chunks_exact(NANBEIGE_VOCAB_SIZE)) {
                cursors[slot].emit_next(logits, requests[slot].decoder, sink, driver.control())?;
            }
        }
        let mut sequences = reserved(requests.len())?; let mut model_work = Int8Work::default(); let mut longest = 0;
        for (slot, row) in cursors.into_iter().enumerate() {
            let sequence = row.finish()?; let work = driver.work(slot)?;
            check_completed(sequence.native_work, work)?;
            longest = longest.max(work.forward_positions); model_work = model_work.checked_add(work)?;
            sequences.push(Int8GenerationRun { schema_version: 1, sequence, model_work: work });
        }
        if group_steps != longest { return Err(Int8GenerationError::WorkMismatch); }
        Ok(Int8GenerationCohortRun { schema_version: 1, execution: INT8_COHORT_EXECUTION.to_owned(),
            sequences, group_steps, planned_work, model_work })
    })();
    if result.is_err() { driver.abort(); } result
}

/// Private synthetic seam exercises the REAL cursor and policy, never public
/// fake-native receipts. Production accepts only Int8CohortEngine above.
trait GroupDriver {
    type Control: DecodeStepControl;
    fn control(&mut self) -> &mut Self::Control;
    fn append_group(&mut self, steps: &[CohortToken]) -> Result<(), Int8GenerationError>;
    fn logits_group(&mut self, sequences: &[usize]) -> Result<Vec<f32>, Int8GenerationError>;
    fn work(&self, sequence: usize) -> Result<Int8Work, Int8GenerationError>;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> GroupDriver for Int8CohortSession<'_, '_, C> {
    type Control = C;
    fn control(&mut self) -> &mut C { Int8CohortSession::control(self) }
    fn append_group(&mut self, steps: &[CohortToken]) -> Result<(), Int8GenerationError> {
        Int8CohortSession::append_group(self, steps).map_err(Into::into)
    }
    fn logits_group(&mut self, sequences: &[usize]) -> Result<Vec<f32>, Int8GenerationError> {
        Int8CohortSession::logits_group(self, sequences, LinearRows::All).map_err(Into::into)
    }
    fn work(&self, sequence: usize) -> Result<Int8Work, Int8GenerationError> {
        Int8CohortSession::work(self, sequence).map_err(Into::into)
    }
    fn abort(&mut self) { Int8CohortSession::abort(self); }
}
#[cfg(test)] mod tests;
