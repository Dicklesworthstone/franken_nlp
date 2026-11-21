//! Explicit grammar-first selected-row execution on the existing grouped INT4 engine.
//!
//! Every legal token is projected, including EOS exactly when the grammar is
//! accepting. A row ceiling REFUSES, never truncates the language. This path
//! checks only requested logits; it makes no claim about unrequested rows and
//! does not replace the full-vocabulary reference or certify model parity.
use super::*;
pub(super) mod selection;
pub mod prefill;

pub const INT4_SPARSE_JSON_EXECUTION: &str = "grouped-int4-grammar-first-selected-rows-v1";

/// An explicit per-selection ceiling, not top-k pruning or a dispatch award.
/// Complete worst-case native and mask work is admitted before prefill.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4JsonSparseLimits { pub max_rows_per_step: usize }
impl Int4JsonSparseLimits {
    pub fn validate(self) -> Result<usize, Int4JsonError> {
        if self.max_rows_per_step == 0 || self.max_rows_per_step > NANBEIGE_VOCAB_SIZE {
            return Err(JsonDecodeError::InvalidRequest("invalid selected-row ceiling").into());
        }
        Ok(self.max_rows_per_step)
    }
}

pub fn planned_work(group_width: usize, prompt_tokens: usize, max_new_tokens: usize, limits: Int4JsonSparseLimits)
    -> Result<Int4Work, Int4JsonError> {
    work_for_width(group_width, prompt_tokens, max_new_tokens, limits.validate()?)
}

#[allow(clippy::too_many_arguments)]
pub fn decode_json_int4_sparse<C: DecodeStepControl>(engine: &mut StrictInt4Engine<'_>,
    identity: &ExecutionIdentity, prompt: &[u32], program: &JsonProgram,
    vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions, budget: Int4JsonBudget,
    limits: Int4JsonSparseLimits, control: &mut C) -> Result<Int4JsonRun, Int4JsonError> {
    decode_json_int4_sparse_with(engine, identity, prompt, program, vocabulary, options,
        budget, limits, control, Ok)
}

/// Task/source/output finalization stays inside the exclusively owned native
/// session. Errors and late cancellation poison it; RAII drains all 44 KV slots.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_json_int4_sparse_with<C, T, E, F>(engine: &mut StrictInt4Engine<'_>,
    identity: &ExecutionIdentity, prompt: &[u32], program: &JsonProgram,
    vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions, budget: Int4JsonBudget,
    limits: Int4JsonSparseLimits, control: &mut C, finalize: F) -> Result<T, E>
where C: DecodeStepControl, E: From<Int4JsonError>, F: FnOnce(Int4JsonRun) -> Result<T, E> {
    check_request(engine, identity, prompt, vocabulary, options, budget, limits).map_err(E::from)?;
    let group_width = engine.model_identity().group_width();
    let mut session = engine.session(budget.native, control).map_err(Int4JsonError::from).map_err(E::from)?;
    drive(group_width, prompt, program, vocabulary, options, budget, limits, &mut session, finalize)
}

pub(super) fn check_request(engine: &StrictInt4Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions, budget: Int4JsonBudget,
    limits: Int4JsonSparseLimits) -> Result<(), Int4JsonError> {
    check_model(identity, engine.model_identity())?;
    if engine.profile() != STRICT_INT4_PROFILE || engine.is_poisoned() {
        return Err(StrictInt4Error::EngineUnavailable.into());
    }
    if !engine.kv_cache().all_slots_have_len(0) {
        return Err(JsonDecodeError::EngineAlreadyPrimed.into());
    }
    if vocabulary.width() != NANBEIGE_VOCAB_SIZE {
        return Err(JsonDecodeError::InvalidRequest("model/tokenizer vocabulary mismatch").into());
    }
    let work = preflight(engine.model_identity().group_width(), prompt, vocabulary, options, budget, limits)?;
    check_capacity(work, engine.kv_cache().capacity_positions(), budget.json.max_kv_bytes)
}

pub(super) fn preflight<V: Vocabulary>(group_width: usize, prompt: &[u32], vocabulary: &V, options: &JsonDecodeOptions,
    budget: Int4JsonBudget, limits: Int4JsonSparseLimits) -> Result<Int4Work, Int4JsonError> {
    let width = vocabulary.width();
    let cap = limits.validate()?;
    if width == 0 || width > NANBEIGE_VOCAB_SIZE || cap > width {
        return Err(JsonDecodeError::InvalidRequest("selected rows exceed vocabulary").into());
    }
    let work = planned_work(group_width, prompt.len(), options.max_new_tokens, limits)?;
    if prompt.iter().chain(options.excluded_token_ids.iter()).chain(std::iter::once(&options.eos_token_id))
        .any(|&id| id as usize >= width) {
        return Err(JsonDecodeError::InvalidRequest("out-of-vocabulary token").into());
    }
    if budget.json.mask_limits.max_trie_node_visits == 0 || budget.json.mask_limits.checkpoint_interval_nodes == 0 {
        return Err(JsonDecodeError::InvalidRequest("zero mask work bound").into());
    }
    // Sparse projection does NOT discount full grammar-mask traversal work.
    let masks = (vocabulary.mask_charge(budget.json.mask_limits) as u64)
        .checked_mul(options.max_new_tokens as u64).ok_or(StrictInt4Error::Work)?;
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
pub(super) trait Driver {
    type Control: DecodeStepControl;
    fn control(&mut self) -> &mut Self::Control;
    fn append(&mut self, token: u32) -> Result<(), Int4JsonError>;
    fn append_prompt(&mut self, prompt: &[u32]) -> Result<(), Int4JsonError> {
        for (index, &token) in prompt.iter().enumerate() {
            if let Some(cause) = self.control().prefill_checkpoint(index) {
                return Err(JsonDecodeError::Cancelled(cause).into());
            }
            self.append(token)?;
        }
        Ok(())
    }
    fn logits(&mut self, rows: &[u32]) -> Result<Vec<f32>, Int4JsonError>;
    fn work(&self) -> Result<Int4Work, Int4JsonError>;
    /// Same-session causal prompt positions retained for independent replay.
    /// Work returned by this driver covers only this sample's new operations.
    fn retained_prompt_positions(&self) -> usize { 0 }
    fn abort(&mut self);
}
impl<C: DecodeStepControl> Driver for Int4Session<'_, '_, C> {
    type Control = C;
    fn control(&mut self) -> &mut C { Int4Session::control(self) }
    fn append(&mut self, token: u32) -> Result<(), Int4JsonError> { Ok(Int4Session::append(self, token)?) }
    fn logits(&mut self, rows: &[u32]) -> Result<Vec<f32>, Int4JsonError> {
        Ok(Int4Session::logits(self, LinearRows::Selected(rows))?)
    }
    fn work(&self) -> Result<Int4Work, Int4JsonError> { Ok(Int4Session::work(self)) }
    fn abort(&mut self) { Int4Session::abort(self); }
}
fn poll<C: DecodeStepControl>(control: &mut C, step: usize) -> Result<(), Int4JsonError> {
    match control.checkpoint(step) { Some(cause) => Err(JsonDecodeError::Cancelled(cause).into()), None => Ok(()) }
}

fn select(rows: &[u32], logits: &[f32]) -> Result<u32, Int4JsonError> {
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
pub(super) fn drive<V, D, T, E, F>(group_width: usize, prompt: &[u32], program: &JsonProgram, vocabulary: &V,
    options: &JsonDecodeOptions, budget: Int4JsonBudget, limits: Int4JsonSparseLimits,
    driver: &mut D, finalize: F) -> Result<T, E>
where V: Vocabulary, D: Driver, E: From<Int4JsonError>, F: FnOnce(Int4JsonRun) -> Result<T, E> {
    drive_selected(group_width, prompt, program, vocabulary, options, budget, limits, driver, &mut Greedy, finalize)
}

/// Private selection seam shared by exact greedy and addressed sampling. The
/// grammar, complete legal-set admission, causal work and finalizer stay in
/// one decoder loop; callers cannot provide an alternate backend or sampler.
pub(super) trait Selection {
    #[allow(clippy::too_many_arguments)]
    fn project<D: selection::Head>(&mut self, mask: &DenseTokenMask, accepting: bool,
        options: &JsonDecodeOptions, cap: usize, remaining: u64, driver: &mut D, step: usize)
        -> Result<(u32, usize), Int4JsonError>;
    fn execution(&self) -> &'static str;
}
struct Greedy;
impl Selection for Greedy {
    #[allow(clippy::too_many_arguments)]
    fn project<D: selection::Head>(&mut self, mask: &DenseTokenMask, accepting: bool,
        options: &JsonDecodeOptions, cap: usize, remaining: u64, driver: &mut D, step: usize)
        -> Result<(u32, usize), Int4JsonError> {
        selection::project(mask, accepting, options, cap, remaining, driver, step)
    }
    fn execution(&self) -> &'static str { INT4_SPARSE_JSON_EXECUTION }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn drive_selected<V, D, S, T, E, F>(group_width: usize, prompt: &[u32], program: &JsonProgram,
    vocabulary: &V, options: &JsonDecodeOptions, budget: Int4JsonBudget, limits: Int4JsonSparseLimits,
    driver: &mut D, selection: &mut S, finalize: F) -> Result<T, E>
where V: Vocabulary, D: Driver, S: Selection, E: From<Int4JsonError>, F: FnOnce(Int4JsonRun) -> Result<T, E> {
    let result = run(group_width, prompt, program, vocabulary, options, budget, limits, driver, selection)
        .map_err(E::from).and_then(|run| {
            let next_token = run.output.token_ids.len();
            let output = finalize(run)?;
            poll(driver.control(), next_token).map_err(E::from)?;
            Ok(output)
        });
    if result.is_err() { driver.abort(); }
    result
}
#[allow(clippy::too_many_arguments)]
fn run<V: Vocabulary, D: Driver, S: Selection>(group_width: usize, prompt: &[u32], program: &JsonProgram, vocabulary: &V,
    options: &JsonDecodeOptions, budget: Int4JsonBudget, limits: Int4JsonSparseLimits,
    driver: &mut D, selection: &mut S) -> Result<Int4JsonRun, Int4JsonError> {
    preflight(group_width, prompt, vocabulary, options, budget, limits)?;
    let retained = driver.retained_prompt_positions();
    if retained >= prompt.len() {
        return Err(JsonDecodeError::InvalidRequest("replayed prompt must forward its final token").into());
    }
    let mut tokens = Vec::new();
    tokens.try_reserve_exact(options.max_new_tokens).map_err(|_| JsonDecodeError::AllocationRefused)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(program.max_output_bytes()).map_err(|_| JsonDecodeError::AllocationRefused)?;
    driver.append_prompt(prompt)?;
    let mut positions = prompt.len() - retained;
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
        let remaining = budget.json.max_projected_logits.checked_sub(projected as u64).ok_or(Int4JsonError::WorkMismatch)?;
        let (selected, count) = selection.project(&mask, state.is_accepting(), options,
            limits.max_rows_per_step, remaining, driver, step)?;
        projected = projected.checked_add(count).ok_or(StrictInt4Error::Work)?;
        drop(mask);
        poll(driver.control(), step)?;
        tokens.push(selected);
        if selected == options.eos_token_id {
            let json = String::from_utf8(bytes).map_err(|_| JsonDecodeError::IndependentValidation)?;
            program.validate_json(&json).map_err(|_| JsonDecodeError::IndependentValidation)?;
            let model_work = driver.work()?;
            if model_work != Int4Work::for_sequence(group_width, retained, positions, projected)? { return Err(Int4JsonError::WorkMismatch); }
            poll(driver.control(), step)?;
            return Ok(Int4JsonRun { schema_version: 1, execution: selection.execution().to_owned(), group_width, model_work,
                output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT4_PROFILE.to_owned(),
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
