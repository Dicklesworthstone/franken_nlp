//! Structured decoding over independent native INT4 sequences.
//!
//! Each row keeps its own prompt, grammar/source language, EOS and complete
//! work allowance. Full heads share a projection; explicit selected heads
//! project every legal row in bounded chunks. Both use the same causal schedule.
use super::*;
use crate::native_engine::strict_int4::cohort::{Int4CohortEngine, Int4CohortSession,
    Int4CohortToken, packed::Int4TokenRun, MAX_INT4_SEQUENCES};
mod execution;
mod projection;
pub mod selected;
use projection::Projection;
pub(crate) use projection::check_selected_group_counts;

pub const INT4_JSON_COHORT_EXECUTION: &str = "grouped-int4-constrained-json-cohort-v1";
pub const INT4_JSON_PACKED_COHORT_EXECUTION: &str = "grouped-int4-constrained-json-packed-cohort-v1";
pub const INT4_JSON_SELECTED_COHORT_EXECUTION: &str = "grouped-int4-grammar-first-selected-rows-cohort-v1";
pub const INT4_JSON_SELECTED_PACKED_COHORT_EXECUTION: &str = "grouped-int4-grammar-first-selected-rows-packed-cohort-v1";

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
    /// Stable input order; inner results retain their original head semantics.
    pub sequences: Vec<Int4JsonRun>,
    /// Actual native forward invocations. Packed execution can append several
    /// known prompt positions per invocation; this is not a position count.
    pub group_steps: u64,
    /// Actual head invocations. Selected heads count each bounded row chunk;
    /// they can exceed decoder groups. This is never projected-logit count.
    pub projection_groups: u64,
    pub planned_work: Int4Work,
    pub model_work: Int4Work,
}

/// Model-free row checks let a bounded batch reject one inadmissible document
/// before enqueueing it. This does not authorize resident model/KV resources;
/// execution independently checks the actual model and allocated capacities.
pub(crate) fn preflight_request(group_width: usize, request: &Int4JsonCohortRequest<'_>,
    vocabulary: &VocabMaskOracle) -> Result<Int4Work, Int4JsonError> {
    preflight_request_for(group_width, request, vocabulary, Projection::Full)
}
pub(crate) fn preflight_request_selected(group_width: usize, request: &Int4JsonCohortRequest<'_>,
    vocabulary: &VocabMaskOracle, limits: sparse::Int4JsonSparseLimits) -> Result<Int4Work, Int4JsonError> {
    preflight_request_for(group_width, request, vocabulary, Projection::Selected(&[limits]))
}
fn preflight_request_for(group_width: usize, request: &Int4JsonCohortRequest<'_>,
    vocabulary: &VocabMaskOracle, projection: Projection<'_>) -> Result<Int4Work, Int4JsonError> {
    check_profile(request.identity, group_width)?;
    if vocabulary.width() != NANBEIGE_VOCAB_SIZE {
        return Err(JsonDecodeError::InvalidRequest("model/tokenizer vocabulary mismatch").into());
    }
    projection.validate(1)?;
    projection.preflight(group_width, 0, request, vocabulary)
}

fn requests_preflight<V: Vocabulary>(group_width: usize, requests: &[Int4JsonCohortRequest<'_>],
    vocabulary: &V, budget: Int4JsonCohortBudget, projection: Projection<'_>) -> Result<Int4Work, Int4JsonError> {
    if requests.is_empty() || requests.len() > MAX_INT4_SEQUENCES || budget.max_result_bytes == 0 {
        return Err(JsonDecodeError::InvalidRequest("invalid structured cohort bounds").into());
    }
    projection.validate(requests.len())?;
    let mut work = Int4Work::default(); let mut masks = 0_u64;
    let mut tokens = 0_u64; let mut bytes = 0_u64;
    for (slot, request) in requests.iter().enumerate() {
        same_model(requests[0].identity, request.identity)?;
        work = work.checked_add(projection.preflight(group_width, slot, request, vocabulary)?)?;
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
/// Explicit packed prompt scheduling over the same native math and independent
/// grammars. The caller must admit `prefill.validate()` extra scratch bytes in
/// addition to the ordinary native cohort and constrained-workspace quotes.
/// Ready decode rows consume only one sampled feedback token per invocation.
pub fn decode_json_int4_cohort_packed<C: DecodeStepControl>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4JsonCohortRequest<'_>], vocabulary: &VocabMaskOracle,
    budget: Int4JsonCohortBudget, prefill: Int4PrefillLimits, control: &mut C)
    -> Result<Int4JsonCohortRun, Int4JsonError> {
    execute_with_packed(engine, requests, vocabulary, budget, prefill, control, Ok)
}
/// Closed task finalizers run before the exclusive session can release any KV.
/// Error, cancellation and unwind invalidate the complete cohort; session Drop
/// clears every row before its embedding host can release resource ownership.
pub(crate) fn execute_with<C, T, E, F>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4JsonCohortRequest<'_>], vocabulary: &VocabMaskOracle,
    budget: Int4JsonCohortBudget, control: &mut C, finalize: F) -> Result<T, E>
where C: DecodeStepControl, E: From<Int4JsonError>, F: FnOnce(Int4JsonCohortRun) -> Result<T, E> {
    execute_scheduled(engine, requests, vocabulary, budget, Projection::Full, None, control, finalize)
}
pub(crate) fn execute_with_packed<C, T, E, F>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4JsonCohortRequest<'_>], vocabulary: &VocabMaskOracle,
    budget: Int4JsonCohortBudget, prefill: Int4PrefillLimits, control: &mut C, finalize: F) -> Result<T, E>
where C: DecodeStepControl, E: From<Int4JsonError>, F: FnOnce(Int4JsonCohortRun) -> Result<T, E> {
    execute_scheduled(engine, requests, vocabulary, budget, Projection::Full, Some(prefill), control, finalize)
}
#[allow(clippy::too_many_arguments)]
fn execute_scheduled<C, T, E, F>(engine: &mut Int4CohortEngine<'_>,
    requests: &[Int4JsonCohortRequest<'_>], vocabulary: &VocabMaskOracle,
    budget: Int4JsonCohortBudget, projection: Projection<'_>, prefill: Option<Int4PrefillLimits>,
    control: &mut C, finalize: F) -> Result<T, E>
where C: DecodeStepControl, E: From<Int4JsonError>, F: FnOnce(Int4JsonCohortRun) -> Result<T, E> {
    let group_width = engine.model_identity().group_width();
    let preflight = (|| -> Result<Vec<Int4RunBudget>, Int4JsonError> {
        poll(control, 0)?; engine.check_idle()?;
        if let Some(limits) = prefill { limits.validate()?; }
        if requests.len() != engine.sequence_count() || vocabulary.width() != NANBEIGE_VOCAB_SIZE {
            return Err(JsonDecodeError::InvalidRequest("cohort model/tokenizer/row mismatch").into());
        }
        requests_preflight(group_width, requests, vocabulary, budget, projection)?;
        let mut capacities = reserved(requests.len())?;
        let mut budgets = reserved(requests.len())?;
        for (slot, request) in requests.iter().enumerate() {
            poll(control, 0)?;
            check_model(request.identity, engine.model_identity())?;
            capacities.push(engine.capacity(slot)?);
            budgets.push(Int4RunBudget::exact(projection.preflight(group_width, slot, request, vocabulary)?));
        }
        check_capacities(group_width, requests, &capacities, budget.max_kv_bytes)?;
        Ok(budgets)
    })();
    let budgets = preflight.map_err(E::from)?;
    let mut session = engine.session(&budgets, control).map_err(Int4JsonError::from).map_err(E::from)?;
    match projection {
        Projection::Full => match prefill {
            Some(limits) => execution::drive_scheduled(group_width, requests, vocabulary, budget,
                Some(limits), &mut session, finalize),
            None => execution::drive(group_width, requests, vocabulary, budget, &mut session, finalize),
        },
        Projection::Selected(_) => execution::drive_with_projection(group_width, requests, vocabulary,
            budget, projection, prefill, &mut session, finalize),
    }
}

/// Reconcile physical counters without confusing a packed invocation with one
/// causal position. Per-row native work is checked independently and exactly;
/// these bounds supplement that check for the closed scheduling strategy.
pub(crate) fn check_group_counts(prefill: Option<Int4PrefillLimits>, total_positions: u64,
    longest_positions: u64, total_projections: u64, longest_projection_sequence: u64,
    group_steps: u64, projection_groups: u64) -> Result<(), Int4JsonError> {
    check_forward_groups(prefill, total_positions, longest_positions, group_steps)?;
    if projection_groups == 0 || projection_groups > group_steps || projection_groups < longest_projection_sequence
        || projection_groups > total_projections {
        return Err(Int4JsonError::WorkMismatch);
    }
    Ok(())
}
fn check_forward_groups(prefill: Option<Int4PrefillLimits>, total_positions: u64,
    longest_positions: u64, group_steps: u64) -> Result<(), Int4JsonError> {
    let valid_steps = match prefill {
        None => group_steps == longest_positions,
        Some(limits) => {
            limits.validate()?;
            group_steps >= total_positions.div_ceil(limits.max_batch_rows as u64)
                && group_steps <= total_positions
        }
    };
    if !valid_steps || longest_positions == 0 || longest_positions > total_positions {
        return Err(Int4JsonError::WorkMismatch);
    }
    Ok(())
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
    fn append_packed(&mut self, runs: &[Int4TokenRun<'_>], limits: Int4PrefillLimits) -> Result<(), Int4JsonError>;
    fn logits_group(&mut self, slots: &[usize]) -> Result<Vec<f32>, Int4JsonError>;
    fn logits_selected(&mut self, slot: usize, rows: &[u32]) -> Result<Vec<f32>, Int4JsonError>;
    fn work(&self, slot: usize) -> Result<Int4Work, Int4JsonError>;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> GroupDriver for Int4CohortSession<'_, '_, C> {
    type Control = C;
    fn control(&mut self) -> &mut C { Int4CohortSession::control(self) }
    fn append_group(&mut self, steps: &[Int4CohortToken]) -> Result<(), Int4JsonError> {
        Ok(Int4CohortSession::append_group(self, steps)?)
    }
    fn append_packed(&mut self, runs: &[Int4TokenRun<'_>], limits: Int4PrefillLimits) -> Result<(), Int4JsonError> {
        Ok(Int4CohortSession::append_packed(self, runs, limits)?)
    }
    fn logits_group(&mut self, slots: &[usize]) -> Result<Vec<f32>, Int4JsonError> {
        Ok(Int4CohortSession::logits_group(self, slots, LinearRows::All)?)
    }
    fn logits_selected(&mut self, slot: usize, rows: &[u32]) -> Result<Vec<f32>, Int4JsonError> {
        Ok(Int4CohortSession::logits_group(self, &[slot], LinearRows::Selected(rows))?)
    }
    fn work(&self, slot: usize) -> Result<Int4Work, Int4JsonError> { Ok(Int4CohortSession::work(self, slot)?) }
    fn abort(&mut self) { Int4CohortSession::abort(self); }
}

#[cfg(test)] mod tests;
