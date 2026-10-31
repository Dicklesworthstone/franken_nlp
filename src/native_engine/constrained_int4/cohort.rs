//! Full-vocabulary structured decoding over independent native INT4 sequences.
//!
//! Each row keeps its own prompt, grammar/source language, EOS and complete
//! work allowance. Ready rows share an actual full-head projection; unfinished
//! rows share one decoder step without padding their causal positions.
use super::*;
use crate::native_engine::strict_int4::cohort::{Int4CohortEngine, Int4CohortSession,
    Int4CohortToken, MAX_INT4_SEQUENCES};
mod execution;

pub const INT4_JSON_COHORT_EXECUTION: &str = "grouped-int4-constrained-json-cohort-v1";

/// Additional live cohort buffers beyond Int4MemoryRequirement::for_cohort.
/// The token allowance covers token vectors plus cursor/row bookkeeping; the
/// byte allowance covers retained JSON/results and independent validation.
/// Fixed row storage includes parser-stack/control overhead, and one dense
/// vocabulary mask is live at a time. Full-head rails belong to native memory.
///
/// This modeled payload quote does NOT replace per-document preparation:
/// schema/source state, variable parser choices and mask-traversal clones must
/// remain covered by that existing reserve throughout execution, as in serial
/// decoding. Allocator slack also remains the embedding host's responsibility.
pub fn required_workspace_bytes(sequence_count: usize, max_generated_tokens: u64, max_result_bytes: u64)
    -> Result<u64, Int4JsonError> {
    if sequence_count == 0 || sequence_count > MAX_INT4_SEQUENCES || max_generated_tokens == 0
        || max_generated_tokens > (sequence_count as u64) * DEFAULT_ADMITTED_CONTEXT_CAP as u64
        || max_result_bytes == 0 {
        return Err(JsonDecodeError::InvalidRequest("invalid structured cohort workspace bounds").into());
    }
    max_generated_tokens.checked_mul(64)
        .and_then(|bytes| max_result_bytes.checked_mul(8).and_then(|output| bytes.checked_add(output)))
        .and_then(|bytes| bytes.checked_add((sequence_count as u64) * 65536))
        .and_then(|bytes| bytes.checked_add(NANBEIGE_VOCAB_SIZE.div_ceil(8) as u64))
        .ok_or_else(|| StrictInt4Error::Memory.into())
}

/// Exact prompt/schema/options binding is supplied by the sealed task plan.
/// This native boundary also checks every row against the actual loaded model.
pub struct Int4JsonCohortRequest<'a> {
    pub identity: &'a ExecutionIdentity,
    pub prompt: &'a [u32],
    pub program: &'a JsonProgram,
    pub options: &'a JsonDecodeOptions,
    pub budget: Int4JsonBudget,
}
/// Aggregate ceilings supplement every independent row's budget. The embedding
/// host owns all input/grammar/result admission, including the complete native
/// cohort payload. KV covers allocated capacity, not only consumed positions.
#[derive(Clone, Copy, Debug)]
pub struct Int4JsonCohortBudget {
    pub native: Int4RunBudget,
    pub max_kv_bytes: u64,
    pub max_mask_node_visits: u64,
    pub max_result_bytes: u64,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4JsonCohortRun {
    pub schema_version: u32,
    pub execution: String,
    /// Stable input order; inner results retain the existing dense semantics.
    pub sequences: Vec<Int4JsonRun>,
    pub group_steps: u64,
    /// Actual shared full-vocabulary head invocations, not projected row count.
    pub projection_groups: u64,
    pub planned_work: Int4Work,
    pub model_work: Int4Work,
}

/// Model-free row checks let a bounded batch reject one inadmissible document
/// before enqueueing it. This does not authorize resident model/KV resources;
/// execution independently checks the actual model and allocated capacities.
pub(crate) fn preflight_request(group_width: usize, request: &Int4JsonCohortRequest<'_>,
    vocabulary: &VocabMaskOracle) -> Result<Int4Work, Int4JsonError> {
    check_profile(request.identity, group_width)?;
    if vocabulary.width() != NANBEIGE_VOCAB_SIZE {
        return Err(JsonDecodeError::InvalidRequest("model/tokenizer vocabulary mismatch").into());
    }
    super::preflight(group_width, request.prompt, vocabulary, request.options, request.budget)
}

fn requests_preflight<V: Vocabulary>(group_width: usize, requests: &[Int4JsonCohortRequest<'_>],
    vocabulary: &V, budget: Int4JsonCohortBudget) -> Result<Int4Work, Int4JsonError> {
    if requests.is_empty() || requests.len() > MAX_INT4_SEQUENCES || budget.max_result_bytes == 0 {
        return Err(JsonDecodeError::InvalidRequest("invalid structured cohort bounds").into());
    }
    let mut work = Int4Work::default(); let mut masks = 0_u64;
    let mut tokens = 0_u64; let mut bytes = 0_u64;
    for request in requests {
        same_model(requests[0].identity, request.identity)?;
        work = work.checked_add(super::preflight(group_width, request.prompt, vocabulary,
            request.options, request.budget)?)?;
        masks = masks.checked_add((vocabulary.mask_charge(request.budget.json.mask_limits) as u64)
            .checked_mul(request.options.max_new_tokens as u64).ok_or(StrictInt4Error::Work)?)
            .ok_or(StrictInt4Error::Work)?;
        tokens = tokens.checked_add(request.options.max_new_tokens as u64).ok_or(StrictInt4Error::Memory)?;
        bytes = bytes.checked_add(request.program.max_output_bytes() as u64).ok_or(StrictInt4Error::Memory)?;
    }
    if work.forward_positions > budget.native.max_forward_positions
        || work.attention_pairs > budget.native.max_attention_pairs
        || !work.projections.fits(budget.native.max_projection_work)
        || masks > budget.max_mask_node_visits {
        return Err(JsonDecodeError::BudgetExceeded("aggregate structured cohort work").into());
    }
    if bytes > budget.max_result_bytes {
        return Err(JsonDecodeError::BudgetExceeded("complete cohort JSON capacity").into());
    }
    required_workspace_bytes(requests.len(), tokens, budget.max_result_bytes)?;
    Ok(work)
}

pub fn decode_json_int4_cohort<C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4JsonCohortRequest<'_>], vocabulary: &VocabMaskOracle,
    budget: Int4JsonCohortBudget, control: &mut C) -> Result<Int4JsonCohortRun, Int4JsonError> {
    execute_with(engine, requests, vocabulary, budget, control, Ok)
}
/// Closed task finalizers run before the exclusive session can release any KV.
/// Error, cancellation and unwind invalidate the complete cohort; session Drop
/// clears every row before its embedding host can release resource ownership.
pub(crate) fn execute_with<C, T, E, F>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4JsonCohortRequest<'_>], vocabulary: &VocabMaskOracle,
    budget: Int4JsonCohortBudget, control: &mut C, finalize: F) -> Result<T, E>
where C: DecodeStepControl, E: From<Int4JsonError>, F: FnOnce(Int4JsonCohortRun) -> Result<T, E> {
    let group_width = engine.model_identity().group_width();
    let preflight = (|| -> Result<Vec<Int4RunBudget>, Int4JsonError> {
        poll(control, 0)?; engine.check_idle()?;
        if requests.len() != engine.sequence_count() || vocabulary.width() != NANBEIGE_VOCAB_SIZE {
            return Err(JsonDecodeError::InvalidRequest("cohort model/tokenizer/row mismatch").into());
        }
        requests_preflight(group_width, requests, vocabulary, budget)?;
        let mut capacities = reserved(requests.len())?;
        let mut budgets = reserved(requests.len())?;
        for (slot, request) in requests.iter().enumerate() {
            poll(control, 0)?;
            check_model(request.identity, engine.model_identity())?;
            capacities.push(engine.capacity(slot)?);
            budgets.push(Int4RunBudget::exact(planned_work(group_width, request.prompt.len(),
                request.options.max_new_tokens)?));
        }
        check_capacities(group_width, requests, &capacities, budget.max_kv_bytes)?;
        Ok(budgets)
    })();
    let budgets = preflight.map_err(E::from)?;
    let mut session = engine.session(&budgets, control).map_err(Int4JsonError::from).map_err(E::from)?;
    execution::drive(group_width, requests, vocabulary, budget, &mut session, finalize)
}
fn check_capacities(group_width: usize, requests: &[Int4JsonCohortRequest<'_>],
    capacities: &[usize], max_kv_bytes: u64) -> Result<(), Int4JsonError> {
    if requests.len() != capacities.len() { return Err(Int4JsonError::WorkMismatch); }
    let mut kv = 0_u64;
    for (request, &capacity) in requests.iter().zip(capacities) {
        let work = planned_work(group_width, request.prompt.len(), request.options.max_new_tokens)?;
        if capacity == 0 || capacity > DEFAULT_ADMITTED_CONTEXT_CAP { return Err(StrictInt4Error::Context.into()); }
        check_capacity(work, capacity, request.budget.json.max_kv_bytes)?;
        kv = kv.checked_add((capacity as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
            .ok_or(StrictInt4Error::Memory)?).ok_or(StrictInt4Error::Memory)?;
    }
    if kv > max_kv_bytes { return Err(StrictInt4Error::Memory.into()); }
    Ok(())
}
fn same_model(a: &ExecutionIdentity, b: &ExecutionIdentity) -> Result<(), Int4JsonError> {
    // Normalize only fields owned by an independent task. Every model,
    // tokenizer, numerics and host field (including future fields) stays equal.
    let mut common = a.clone();
    common.template_digest = b.template_digest; common.task_spec = b.task_spec.clone();
    common.taskir_digest = b.taskir_digest; common.prompt_digest = b.prompt_digest;
    common.grammar_compiler_version = b.grammar_compiler_version.clone(); common.schema_digest = b.schema_digest;
    common.sampler_version = b.sampler_version.clone(); common.calibration_digest = b.calibration_digest;
    common.decision_policy_digest = b.decision_policy_digest;
    if &common != b { return Err(Int4JsonError::ModelIdentity); }
    Ok(())
}
fn poll(control: &mut impl DecodeStepControl, step: usize) -> Result<(), Int4JsonError> {
    match control.checkpoint(step) { Some(cause) => Err(JsonDecodeError::Cancelled(cause).into()), None => Ok(()) }
}
fn reserved<T>(count: usize) -> Result<Vec<T>, Int4JsonError> {
    let mut values = Vec::new(); values.try_reserve_exact(count).map_err(|_| JsonDecodeError::AllocationRefused)?;
    Ok(values)
}

// Private deterministic fixture seam; public entrypoints require real weights
// and VocabMaskOracle rather than accepting injected model or vocabulary work.
trait GroupDriver {
    type Control: DecodeStepControl;
    fn control(&mut self) -> &mut Self::Control;
    fn append_group(&mut self, steps: &[Int4CohortToken]) -> Result<(), Int4JsonError>;
    fn logits_group(&mut self, slots: &[usize]) -> Result<Vec<f32>, Int4JsonError>;
    fn work(&self, slot: usize) -> Result<Int4Work, Int4JsonError>;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> GroupDriver for Int4CohortSession<'_, '_, C> {
    type Control = C;
    fn control(&mut self) -> &mut C { Int4CohortSession::control(self) }
    fn append_group(&mut self, steps: &[Int4CohortToken]) -> Result<(), Int4JsonError> {
        Ok(Int4CohortSession::append_group(self, steps)?)
    }
    fn logits_group(&mut self, slots: &[usize]) -> Result<Vec<f32>, Int4JsonError> {
        Ok(Int4CohortSession::logits_group(self, slots, LinearRows::All)?)
    }
    fn work(&self, slot: usize) -> Result<Int4Work, Int4JsonError> { Ok(Int4CohortSession::work(self, slot)?) }
    fn abort(&mut self) { Int4CohortSession::abort(self); }
}

#[cfg(test)] mod tests;
