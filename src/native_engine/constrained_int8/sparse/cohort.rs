//! Shared-layer INT8 decoding of independent source/schema languages.
//!
//! Fixed membership, one token per unfinished row per tick, independent causal
//! KV and grammars. Heads reuse complete-legal-set bounded chunk selection.
//! Any failure aborts the cohort; no output precedes all task finalizers.
use super::*;
use crate::native_engine::{portable_int8::batch::MAX_BATCH_ROWS,
    strict_int8::cohort::{CohortToken, Int8CohortEngine, Int8CohortSession}};
mod execution;

pub const INT8_JSON_COHORT_EXECUTION: &str = "portable-int8-selected-json-cohort-v1";

/// The task binds exact prompt/schema/options to each admitted identity. This
/// low-level boundary additionally checks every row's actual model binding.
pub struct Int8JsonCohortRequest<'a> {
    pub identity: &'a ExecutionIdentity,
    pub prompt: &'a [u32],
    pub program: &'a JsonProgram,
    pub options: &'a JsonDecodeOptions,
    pub budget: Int8JsonBudget,
    pub limits: Int8JsonSparseLimits,
}
/// Aggregate ceilings supplement, never replace, the row ceilings. All input,
/// grammar, token and retained-result allocations require host admission.
/// max_result_bytes bounds the serialized native envelope, not process RSS.
#[derive(Clone, Copy, Debug)]
pub struct Int8JsonCohortBudget {
    pub native: Int8RunBudget,
    pub max_kv_bytes: u64,
    pub max_mask_node_visits: u64,
    pub max_result_bytes: u64,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8JsonCohortRun {
    pub schema_version: u32,
    pub execution: String,
    /// Input order; rows retain their selected-row semantic execution label.
    pub sequences: Vec<Int8JsonRun>,
    pub group_steps: u64,
    pub planned_work: Int8Work,
    pub model_work: Int8Work,
}
fn requests_preflight<V: Vocabulary>(requests: &[Int8JsonCohortRequest<'_>], vocabulary: &V,
    budget: Int8JsonCohortBudget) -> Result<Int8Work, Int8JsonError> {
    if requests.is_empty() || requests.len() > MAX_BATCH_ROWS || budget.max_result_bytes == 0 {
        return Err(JsonDecodeError::InvalidRequest("invalid structured cohort bounds").into());
    }
    let mut work = Int8Work::default(); let mut masks = 0_u64;
    for request in requests {
        work = work.checked_add(super::preflight(request.prompt, vocabulary, request.options,
            request.budget, request.limits)?)?;
        masks = masks.checked_add((vocabulary.mask_charge(request.budget.json.mask_limits) as u64)
            .checked_mul(request.options.max_new_tokens as u64).ok_or(StrictInt8Error::Work)?)
            .ok_or(StrictInt8Error::Work)?;
    }
    if work.forward_positions > budget.native.max_forward_positions
        || work.attention_pairs > budget.native.max_attention_pairs
        || !work.projections.fits(budget.native.max_projection_work)
        || masks > budget.max_mask_node_visits {
        return Err(JsonDecodeError::BudgetExceeded("aggregate structured cohort work").into());
    }
    Ok(work)
}

pub fn decode_json_int8_sparse_cohort<C: DecodeStepControl>(engine: &mut Int8CohortEngine<'_>,
    requests: &[Int8JsonCohortRequest<'_>], vocabulary: &VocabMaskOracle,
    budget: Int8JsonCohortBudget, control: &mut C) -> Result<Int8JsonCohortRun, Int8JsonError> {
    execute_with(engine, requests, vocabulary, budget, control, Ok)
}
/// Only closed task modules finalize inside the exclusive native session.
/// Errors and late cancellation poison it before RAII clears every KV slot.
pub(crate) fn execute_with<C, T, E, F>(engine: &mut Int8CohortEngine<'_>,
    requests: &[Int8JsonCohortRequest<'_>], vocabulary: &VocabMaskOracle,
    budget: Int8JsonCohortBudget, control: &mut C, finalize: F) -> Result<T, E>
where C: DecodeStepControl, E: From<Int8JsonError>, F: FnOnce(Int8JsonCohortRun) -> Result<T, E> {
    let preflight = (|| -> Result<(), Int8JsonError> {
        engine.check_idle()?;
        if requests.len() != engine.sequence_count() || vocabulary.width() != NANBEIGE_VOCAB_SIZE {
            return Err(JsonDecodeError::InvalidRequest("cohort model/tokenizer/row mismatch").into());
        }
        requests_preflight(requests, vocabulary, budget)?;
        let mut kv = 0_u64;
        for (slot, request) in requests.iter().enumerate() {
            check_model(request.identity, engine.artifact_identity())?;
            let work = planned_work(request.prompt.len(), request.options.max_new_tokens, request.limits)?;
            let capacity = engine.capacity(slot)?;
            check_capacity(work, capacity, request.budget.json.max_kv_bytes)?;
            kv = kv.checked_add((capacity as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
                .ok_or(StrictInt8Error::Memory)?).ok_or(StrictInt8Error::Memory)?;
        }
        if kv > budget.max_kv_bytes { return Err(StrictInt8Error::Memory.into()); }
        Ok(())
    })();
    preflight.map_err(E::from)?;
    let mut budgets = reserved(requests.len()).map_err(E::from)?;
    for request in requests {
        budgets.push(Int8RunBudget::exact(planned_work(request.prompt.len(),
            request.options.max_new_tokens, request.limits).map_err(E::from)?));
    }
    let mut session = engine.session(&budgets, control).map_err(Int8JsonError::from).map_err(E::from)?;
    execution::drive(requests, vocabulary, budget, &mut session, finalize)
}
fn reserved<T>(count: usize) -> Result<Vec<T>, Int8JsonError> {
    let mut values = Vec::new(); values.try_reserve_exact(count).map_err(|_| JsonDecodeError::AllocationRefused)?;
    Ok(values)
}
// Private synthetic seam; public callers must supply the real native engine.
trait GroupDriver {
    type Control: DecodeStepControl;
    fn control(&mut self) -> &mut Self::Control;
    fn append_group(&mut self, steps: &[CohortToken]) -> Result<(), Int8JsonError>;
    fn logits(&mut self, slot: usize, rows: &[u32]) -> Result<Vec<f32>, Int8JsonError>;
    fn work(&self, slot: usize) -> Result<Int8Work, Int8JsonError>;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> GroupDriver for Int8CohortSession<'_, '_, C> {
    type Control = C;
    fn control(&mut self) -> &mut C { Int8CohortSession::control(self) }
    fn append_group(&mut self, steps: &[CohortToken]) -> Result<(), Int8JsonError> { Ok(Int8CohortSession::append_group(self, steps)?) }
    fn logits(&mut self, slot: usize, rows: &[u32]) -> Result<Vec<f32>, Int8JsonError> {
        Ok(Int8CohortSession::logits_group(self, &[slot], LinearRows::Selected(rows))?)
    }
    fn work(&self, slot: usize) -> Result<Int8Work, Int8JsonError> { Ok(Int8CohortSession::work(self, slot)?) }
    fn abort(&mut self) { Int8CohortSession::abort(self); }
}
struct RowHead<'a, D> { driver: &'a mut D, slot: usize }
impl<D: GroupDriver> selection::Head for RowHead<'_, D> {
    type Control = D::Control;
    fn control(&mut self) -> &mut Self::Control { self.driver.control() }
    fn logits(&mut self, rows: &[u32]) -> Result<Vec<f32>, Int8JsonError> { self.driver.logits(self.slot, rows) }
}
#[cfg(test)] mod tests;
