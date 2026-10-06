//! Explicit grammar-first selected-row execution on the real strict INT8 engine.
//!
//! Every legal token is projected, including EOS exactly when the grammar is
//! accepting. A row ceiling REFUSES, never truncates the language. This path
//! checks only requested logits; it makes no claim about unrequested rows and
//! does not replace the full-vocabulary reference or certify model parity.
use super::*;

pub const INT8_SPARSE_JSON_EXECUTION: &str = "portable-int8-grammar-first-selected-rows-v1";

/// An explicit per-selection ceiling, not top-k pruning or a dispatch award.
/// Complete worst-case native and mask work is admitted before prefill.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8JsonSparseLimits { pub max_rows_per_step: usize }
impl Int8JsonSparseLimits {
    pub fn validate(self) -> Result<usize, Int8JsonError> {
        if self.max_rows_per_step == 0 || self.max_rows_per_step > NANBEIGE_VOCAB_SIZE {
            return Err(JsonDecodeError::InvalidRequest("invalid selected-row ceiling").into());
        }
        Ok(self.max_rows_per_step)
    }
}

pub fn planned_work(prompt_tokens: usize, max_new_tokens: usize, limits: Int8JsonSparseLimits)
    -> Result<Int8Work, Int8JsonError> {
    work_for_width(prompt_tokens, max_new_tokens, limits.validate()?)
}

#[allow(clippy::too_many_arguments)]
pub fn decode_json_int8_sparse<C: DecodeStepControl>(engine: &mut StrictInt8Engine<'_>,
    identity: &ExecutionIdentity, prompt: &[u32], program: &JsonProgram,
    vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions, budget: Int8JsonBudget,
    limits: Int8JsonSparseLimits, control: &mut C) -> Result<Int8JsonRun, Int8JsonError> {
    decode_json_int8_sparse_with(engine, identity, prompt, program, vocabulary, options,
        budget, limits, control, Ok)
}

/// Task/source/output finalization stays inside the exclusively owned native
/// session. Errors and late cancellation poison it; RAII drains all 44 KV slots.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_json_int8_sparse_with<C, T, E, F>(engine: &mut StrictInt8Engine<'_>,
    identity: &ExecutionIdentity, prompt: &[u32], program: &JsonProgram,
    vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions, budget: Int8JsonBudget,
    limits: Int8JsonSparseLimits, control: &mut C, finalize: F) -> Result<T, E>
where C: DecodeStepControl, E: From<Int8JsonError>, F: FnOnce(Int8JsonRun) -> Result<T, E> {
    check_model(identity, engine.artifact_identity()).map_err(E::from)?;
    if engine.profile() != STRICT_INT8_PROFILE || engine.is_poisoned() {
        return Err(E::from(StrictInt8Error::EngineUnavailable.into()));
    }
    if !engine.kv_cache().all_slots_have_len(0) {
        return Err(E::from(JsonDecodeError::EngineAlreadyPrimed.into()));
    }
    if vocabulary.width() != NANBEIGE_VOCAB_SIZE {
        return Err(E::from(JsonDecodeError::InvalidRequest("model/tokenizer vocabulary mismatch").into()));
    }
    let work = preflight(prompt, vocabulary, options, budget, limits).map_err(E::from)?;
    check_capacity(work, engine.kv_cache().capacity_positions(), budget.json.max_kv_bytes).map_err(E::from)?;
    let mut session = engine.session(budget.native, control).map_err(Int8JsonError::from).map_err(E::from)?;
    drive(prompt, program, vocabulary, options, budget, limits, &mut session, finalize)
}

fn preflight<V: Vocabulary>(prompt: &[u32], vocabulary: &V, options: &JsonDecodeOptions,
    budget: Int8JsonBudget, limits: Int8JsonSparseLimits) -> Result<Int8Work, Int8JsonError> {
    let width = vocabulary.width();
    let cap = limits.validate()?;
    if width == 0 || width > NANBEIGE_VOCAB_SIZE || cap > width {
        return Err(JsonDecodeError::InvalidRequest("selected rows exceed vocabulary").into());
    }
    let work = planned_work(prompt.len(), options.max_new_tokens, limits)?;
    if prompt.iter().chain(options.excluded_token_ids.iter()).chain(std::iter::once(&options.eos_token_id))
        .any(|&id| id as usize >= width) {
        return Err(JsonDecodeError::InvalidRequest("out-of-vocabulary token").into());
    }
    if budget.json.mask_limits.max_trie_node_visits == 0 || budget.json.mask_limits.checkpoint_interval_nodes == 0 {
        return Err(JsonDecodeError::InvalidRequest("zero mask work bound").into());
    }
    // Sparse projection does NOT discount full grammar-mask traversal work.
    let masks = (vocabulary.mask_charge(budget.json.mask_limits) as u64)
        .checked_mul(options.max_new_tokens as u64).ok_or(StrictInt8Error::Work)?;
    if masks > budget.json.max_total_mask_node_visits
        || work.forward_positions > budget.json.max_forward_positions
        || work.projected_logits > budget.json.max_projected_logits
        || work.forward_positions > budget.native.max_forward_positions
        || work.attention_pairs > budget.native.max_attention_pairs
        || !work.projections.fits(budget.native.max_projection_work) {
        return Err(JsonDecodeError::BudgetExceeded("complete selected-row work").into());
    }
    Ok(work)
}

// A private static seam, never an external backend or native-receipt factory.
trait Driver {
    type Control: DecodeStepControl;
    fn control(&mut self) -> &mut Self::Control;
    fn append(&mut self, token: u32) -> Result<(), Int8JsonError>;
    fn logits(&mut self, rows: &[u32]) -> Result<Vec<f32>, Int8JsonError>;
    fn work(&self) -> Int8Work;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> Driver for Int8Session<'_, '_, C> {
    type Control = C;
    fn control(&mut self) -> &mut C { Int8Session::control(self) }
    fn append(&mut self, token: u32) -> Result<(), Int8JsonError> { Ok(Int8Session::append(self, token)?) }
    fn logits(&mut self, rows: &[u32]) -> Result<Vec<f32>, Int8JsonError> {
        Ok(Int8Session::logits(self, LinearRows::Selected(rows))?)
    }
    fn work(&self) -> Int8Work { Int8Session::work(self) }
    fn abort(&mut self) { Int8Session::abort(self); }
}
fn poll<C: DecodeStepControl>(control: &mut C, step: usize) -> Result<(), Int8JsonError> {
    match control.checkpoint(step) { Some(cause) => Err(JsonDecodeError::Cancelled(cause).into()), None => Ok(()) }
}

/// Enumerate the ENTIRE legal set in canonical token-ID order. The first pass
/// rejects oversized sets before allocating/projecting; the second fills the
/// exactly sized array. No sort, sampling policy, or physical-row address exists.
fn legal_rows<C: DecodeStepControl>(mask: &DenseTokenMask, accepting: bool,
    options: &JsonDecodeOptions, cap: usize, control: &mut C, step: usize) -> Result<Vec<u32>, Int8JsonError> {
    let legal = |id: u32| if id == options.eos_token_id { accepting }
        else { mask.contains(id) && !options.excluded_token_ids.contains(&id) };
    let mut count = 0;
    for index in 0..mask.vocab_size() {
        if index % 1024 == 0 { poll(control, step)?; }
        if legal(index as u32) {
            count += 1;
            if count > cap { return Err(JsonDecodeError::BudgetExceeded("complete legal row set").into()); }
        }
    }
    if count == 0 { return Err(JsonDecodeError::NoLegalToken.into()); }
    let mut rows = Vec::new();
    rows.try_reserve_exact(count).map_err(|_| JsonDecodeError::AllocationRefused)?;
    for index in 0..mask.vocab_size() {
        if index % 1024 == 0 { poll(control, step)?; }
        if legal(index as u32) { rows.push(index as u32); }
    }
    Ok(rows)
}
fn select(rows: &[u32], logits: &[f32]) -> Result<u32, Int8JsonError> {
    if rows.is_empty() || rows.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(JsonDecodeError::InvalidLogits.into());
    }
    checked_logits(logits, rows.len())?;
    let mut best = 0;
    for index in 1..rows.len() {
        // Exactly the full reference's first-index comparison, even for +/-0.
        if logits[index] > logits[best] { best = index; }
    }
    Ok(rows[best])
}

#[allow(clippy::too_many_arguments)]
fn drive<V, D, T, E, F>(prompt: &[u32], program: &JsonProgram, vocabulary: &V,
    options: &JsonDecodeOptions, budget: Int8JsonBudget, limits: Int8JsonSparseLimits,
    driver: &mut D, finalize: F) -> Result<T, E>
where V: Vocabulary, D: Driver, E: From<Int8JsonError>, F: FnOnce(Int8JsonRun) -> Result<T, E> {
    let result = run(prompt, program, vocabulary, options, budget, limits, driver)
        .map_err(E::from).and_then(|run| {
            let next_token = run.output.token_ids.len();
            let output = finalize(run)?;
            poll(driver.control(), next_token).map_err(E::from)?;
            Ok(output)
        });
    if result.is_err() { driver.abort(); }
    result
}
fn run<V: Vocabulary, D: Driver>(prompt: &[u32], program: &JsonProgram, vocabulary: &V,
    options: &JsonDecodeOptions, budget: Int8JsonBudget, limits: Int8JsonSparseLimits,
    driver: &mut D) -> Result<Int8JsonRun, Int8JsonError> {
    preflight(prompt, vocabulary, options, budget, limits)?;
    let mut tokens = Vec::new();
    tokens.try_reserve_exact(options.max_new_tokens).map_err(|_| JsonDecodeError::AllocationRefused)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(program.max_output_bytes()).map_err(|_| JsonDecodeError::AllocationRefused)?;
    for (index, &token) in prompt.iter().enumerate() {
        if let Some(cause) = driver.control().prefill_checkpoint(index) { return Err(JsonDecodeError::Cancelled(cause).into()); }
        driver.append(token)?;
    }
    let mut positions = prompt.len();
    let mut projected = 0_usize;
    let mut charged = 0_u64;
    let mut state = program.initial_state();
    for step in 0..options.max_new_tokens {
        poll(driver.control(), step)?;
        charged = charged.checked_add(vocabulary.mask_charge(budget.json.mask_limits) as u64)
            .filter(|&n| n <= budget.json.max_total_mask_node_visits)
            .ok_or(JsonDecodeError::BudgetExceeded("mask work"))?;
        let mask = vocabulary.mask(&state, budget.json.mask_limits, driver.control(), step)?;
        if mask.vocab_size() != vocabulary.width() { return Err(JsonDecodeError::IllegalTransition.into()); }
        let rows = legal_rows(&mask, state.is_accepting(), options, limits.max_rows_per_step, driver.control(), step)?;
        projected = projected.checked_add(rows.len()).filter(|&n| n as u64 <= budget.json.max_projected_logits)
            .ok_or(JsonDecodeError::BudgetExceeded("selected projection work"))?;
        poll(driver.control(), step)?;
        let logits = driver.logits(&rows)?;
        let selected = select(&rows, &logits)?;
        drop(logits); drop(rows); drop(mask);
        poll(driver.control(), step)?;
        tokens.push(selected);
        if selected == options.eos_token_id {
            let json = String::from_utf8(bytes).map_err(|_| JsonDecodeError::IndependentValidation)?;
            program.validate_json(&json).map_err(|_| JsonDecodeError::IndependentValidation)?;
            let model_work = driver.work();
            if model_work != Int8Work::for_sequence(0, positions, projected)? { return Err(Int8JsonError::WorkMismatch); }
            poll(driver.control(), step)?;
            return Ok(Int8JsonRun { schema_version: 1, execution: INT8_SPARSE_JSON_EXECUTION.to_owned(), model_work,
                output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.to_owned(),
                    token_ids: tokens, json, forward_positions: positions as u64,
                    projected_logits: projected as u64, mask_node_visit_charge: charged } });
        }
        let emitted = vocabulary.bytes(selected).filter(|b| !b.is_empty()).ok_or(JsonDecodeError::IllegalTransition)?;
        if bytes.len().checked_add(emitted.len()).is_none_or(|n| n > program.max_output_bytes()) {
            return Err(JsonDecodeError::BudgetExceeded("JSON bytes").into());
        }
        if !state.consume_bytes(emitted) { return Err(JsonDecodeError::IllegalTransition.into()); }
        bytes.extend_from_slice(emitted);
        if step + 1 == options.max_new_tokens {
            return Err(JsonDecodeError::BudgetExceeded("output tokens before EOS").into());
        }
        // Singleton choices and accepting prefixes STILL evolve all 44 KV slots.
        // EOS is always explicitly projected, never fabricated or auto-selected.
        driver.append(selected)?;
        positions += 1;
    }
    Err(JsonDecodeError::BudgetExceeded("output tokens").into())
}

#[cfg(test)] mod tests;
