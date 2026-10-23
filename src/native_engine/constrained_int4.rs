//! Full-vocabulary constrained JSON on the existing grouped-INT4 engine.
//!
//! The grammar, source language and independent validator are shared with the
//! other native profiles. Only the last prompt position projects the head;
//! every generated non-EOS token still executes both loops before selection
//! resumes. The caller retains process, model and output admission authority.
//! This explicit candidate does not qualify INT4 fidelity or task quality.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{
    execution_identity::{ExecutionIdentity, ThinkingMode, ToolMode},
    grammar::{mask::{DenseTokenMask, MaskWorkLimits, VocabMaskOracle}, runtime::{JsonProgram, JsonState}},
};
use super::{
    constrained::{JsonDecodeError, JsonDecodeOptions, JsonDecodeOutput, JsonWorkBudget},
    decode::{DecodeCancellationKind, DecodeStepControl},
    generation::grouped_int4,
    kv::KV_BYTES_PER_TOKEN,
    lmhead::NANBEIGE_VOCAB_SIZE,
    portable_int8::LinearRows,
    rope::DEFAULT_ADMITTED_CONTEXT_CAP,
    strict_int4::{Int4ModelIdentity, Int4RunBudget, Int4Session, Int4Work,
        StrictInt4Engine, StrictInt4Error, STRICT_INT4_PROFILE},
};

pub const INT4_JSON_EXECUTION: &str = "grouped-int4-constrained-json-final-prefill-head-v1";

/// Group dots, MACs and attention pairs are additional to JSON/mask work.
/// The KV ceiling prices the engine's complete resident capacity.
#[derive(Clone, Copy, Debug)]
pub struct Int4JsonBudget { pub native: Int4RunBudget, pub json: JsonWorkBudget }

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int4JsonRun {
    pub schema_version: u32,
    pub execution: String,
    pub group_width: usize,
    pub output: JsonDecodeOutput,
    pub model_work: Int4Work,
}

#[derive(Debug)]
pub enum Int4JsonError {
    Decode(JsonDecodeError), Native(StrictInt4Error), ModelIdentity, WorkMismatch,
}
impl fmt::Display for Int4JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(error) => write!(f, "INT4 structured decoding refused: {error}"),
            Self::Native(error) => write!(f, "INT4 structured native execution refused: {error}"),
            Self::ModelIdentity => f.write_str("INT4 structured model, partition or candidate identity mismatch"),
            Self::WorkMismatch => f.write_str("INT4 structured work differs from its native session"),
        }
    }
}
impl Error for Int4JsonError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Decode(error) => Some(error), Self::Native(error) => Some(error), _ => None }
    }
}
impl From<JsonDecodeError> for Int4JsonError { fn from(error: JsonDecodeError) -> Self { Self::Decode(error) } }
impl From<StrictInt4Error> for Int4JsonError { fn from(error: StrictInt4Error) -> Self { Self::Native(error) } }
impl Int4JsonError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Decode(JsonDecodeError::Cancelled(cause))
                | Self::Native(StrictInt4Error::Cancelled(cause)) => Some(*cause),
            _ => None,
        }
    }
}

/// Complete final-prefill-head ceiling, including a scored terminal EOS.
/// Exact group partition affects dot counts even at the same model shape.
pub fn planned_work(group_width: usize, prompt_tokens: usize, max_new_tokens: usize)
    -> Result<Int4Work, Int4JsonError> {
    work_for_width(group_width, prompt_tokens, max_new_tokens, NANBEIGE_VOCAB_SIZE)
}
fn work_for_width(group_width: usize, prompt_tokens: usize, max_new_tokens: usize, width: usize)
    -> Result<Int4Work, Int4JsonError> {
    if prompt_tokens == 0 || max_new_tokens == 0 || width == 0 || width > u32::MAX as usize {
        return Err(JsonDecodeError::InvalidRequest("empty prompt or output, or invalid vocabulary").into());
    }
    let positions = prompt_tokens.checked_add(max_new_tokens - 1).ok_or(StrictInt4Error::Work)?;
    if positions > DEFAULT_ADMITTED_CONTEXT_CAP { return Err(StrictInt4Error::Context.into()); }
    let rows = max_new_tokens.checked_mul(width).ok_or(StrictInt4Error::Work)?;
    Ok(Int4Work::for_sequence(group_width, 0, positions, rows)?)
}

/// The task compiler binds prompt, schema, vocabulary and control policy to
/// the identity. This boundary independently checks its native model facts.
#[allow(clippy::too_many_arguments)]
pub fn decode_json_int4<C: DecodeStepControl>(
    engine: &mut StrictInt4Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    budget: Int4JsonBudget, control: &mut C,
) -> Result<Int4JsonRun, Int4JsonError> {
    decode_json_int4_with(engine, identity, prompt, program, vocabulary, options, budget, control, Ok)
}

/// Task-specific evidence and envelope validation execute while the same
/// exclusive session owns KV. Any failure aborts it; Drop clears every slot
/// on success, error and unwind before the host can release its reservations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_json_int4_with<C, T, E, F>(
    engine: &mut StrictInt4Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    budget: Int4JsonBudget, control: &mut C, finalize: F,
) -> Result<T, E>
where C: DecodeStepControl, E: From<Int4JsonError>, F: FnOnce(Int4JsonRun) -> Result<T, E> {
    check_request(engine, identity, prompt, vocabulary, options, budget).map_err(E::from)?;
    let group_width = engine.model_identity().group_width();
    let mut session = engine.session(budget.native, control).map_err(Int4JsonError::from).map_err(E::from)?;
    drive(group_width, prompt, program, vocabulary, options, budget, &mut session, finalize)
}

fn check_request(engine: &StrictInt4Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions, budget: Int4JsonBudget)
    -> Result<(), Int4JsonError> {
    check_model(identity, engine.model_identity())?;
    if engine.profile() != STRICT_INT4_PROFILE || engine.is_poisoned() {
        return Err(StrictInt4Error::EngineUnavailable.into());
    }
    if !engine.kv_cache().all_slots_have_len(0) { return Err(JsonDecodeError::EngineAlreadyPrimed.into()); }
    if vocabulary.width() != NANBEIGE_VOCAB_SIZE {
        return Err(JsonDecodeError::InvalidRequest("model/tokenizer vocabulary mismatch").into());
    }
    let work = preflight(engine.model_identity().group_width(), prompt, vocabulary, options, budget)?;
    check_capacity(work, engine.kv_cache().capacity_positions(), budget.json.max_kv_bytes)
}

pub(crate) fn check_profile(identity: &ExecutionIdentity, group_width: usize) -> Result<(), Int4JsonError> {
    identity.validate().map_err(|_| Int4JsonError::ModelIdentity)?;
    grouped_int4::check_contract(identity, group_width).map_err(|_| Int4JsonError::ModelIdentity)?;
    if identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None {
        return Err(Int4JsonError::ModelIdentity);
    }
    Ok(())
}
pub(crate) fn check_model(identity: &ExecutionIdentity, source: &Int4ModelIdentity) -> Result<(), Int4JsonError> {
    check_profile(identity, source.group_width())?;
    grouped_int4::check_model(identity, source.group_width(), source).map_err(|_| Int4JsonError::ModelIdentity)
}
fn check_capacity(work: Int4Work, capacity: usize, max_kv_bytes: u64) -> Result<(), Int4JsonError> {
    let capacity = u64::try_from(capacity).map_err(|_| StrictInt4Error::Memory)?;
    if work.forward_positions > capacity { return Err(StrictInt4Error::Context.into()); }
    let bytes = capacity.checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(StrictInt4Error::Memory)?;
    if bytes > max_kv_bytes { return Err(StrictInt4Error::Memory.into()); }
    Ok(())
}
fn preflight<V: Vocabulary>(group_width: usize, prompt: &[u32], vocabulary: &V,
    options: &JsonDecodeOptions, budget: Int4JsonBudget) -> Result<Int4Work, Int4JsonError> {
    let width = vocabulary.width();
    let work = work_for_width(group_width, prompt.len(), options.max_new_tokens, width)?;
    if prompt.iter().chain(options.excluded_token_ids.iter()).chain(std::iter::once(&options.eos_token_id))
        .any(|&id| id as usize >= width) {
        return Err(JsonDecodeError::InvalidRequest("out-of-vocabulary token").into());
    }
    if budget.json.mask_limits.max_trie_node_visits == 0 || budget.json.mask_limits.checkpoint_interval_nodes == 0 {
        return Err(JsonDecodeError::InvalidRequest("zero mask work bound").into());
    }
    let masks = (vocabulary.mask_charge(budget.json.mask_limits) as u64)
        .checked_mul(options.max_new_tokens as u64).ok_or(StrictInt4Error::Work)?;
    if masks > budget.json.max_total_mask_node_visits
        || work.forward_positions > budget.json.max_forward_positions
        || work.projected_logits > budget.json.max_projected_logits
        || work.forward_positions > budget.native.max_forward_positions
        || work.attention_pairs > budget.native.max_attention_pairs
        || !work.projections.fits(budget.native.max_projection_work) {
        return Err(JsonDecodeError::BudgetExceeded("complete structured work").into());
    }
    Ok(work)
}

// These static seams are private. Public callers must supply the real INT4
// engine and the token-byte table owned by VocabMaskOracle.
trait Driver {
    type Control: DecodeStepControl;
    fn control(&mut self) -> &mut Self::Control;
    fn append(&mut self, token: u32) -> Result<(), Int4JsonError>;
    fn logits(&mut self) -> Result<Vec<f32>, Int4JsonError>;
    fn work(&self) -> Int4Work;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> Driver for Int4Session<'_, '_, C> {
    type Control = C;
    fn control(&mut self) -> &mut C { Int4Session::control(self) }
    fn append(&mut self, token: u32) -> Result<(), Int4JsonError> { Ok(Int4Session::append(self, token)?) }
    fn logits(&mut self) -> Result<Vec<f32>, Int4JsonError> { Ok(Int4Session::logits(self, LinearRows::All)?) }
    fn work(&self) -> Int4Work { Int4Session::work(self) }
    fn abort(&mut self) { Int4Session::abort(self); }
}
trait Vocabulary {
    fn width(&self) -> usize;
    fn bytes(&self, id: u32) -> Option<&[u8]>;
    fn mask_charge(&self, limits: MaskWorkLimits) -> usize;
    fn mask<C: DecodeStepControl>(&self, state: &JsonState<'_>, limits: MaskWorkLimits, control: &mut C, step: usize)
        -> Result<DenseTokenMask, Int4JsonError>;
}
impl Vocabulary for VocabMaskOracle {
    fn width(&self) -> usize { self.trie().vocab_size() }
    fn bytes(&self, id: u32) -> Option<&[u8]> { self.trie().token_bytes(id) }
    fn mask_charge(&self, limits: MaskWorkLimits) -> usize { self.trie().node_count().min(limits.max_trie_node_visits) }
    fn mask<C: DecodeStepControl>(&self, state: &JsonState<'_>, limits: MaskWorkLimits, control: &mut C, step: usize)
        -> Result<DenseTokenMask, Int4JsonError> {
        let mut cancellation = None;
        let mask = self.materialize(state, limits, |_| {
            cancellation = control.checkpoint(step); cancellation.is_none()
        });
        if let Some(cause) = cancellation { return Err(JsonDecodeError::Cancelled(cause).into()); }
        mask.map_err(JsonDecodeError::Mask).map_err(Into::into)
    }
}
fn checked_logits(logits: &[f32], width: usize) -> Result<(), Int4JsonError> {
    if logits.len() != width || logits.iter().any(|value| !value.is_finite()) {
        return Err(JsonDecodeError::InvalidLogits.into());
    }
    Ok(())
}
fn select(logits: &[f32], mask: &DenseTokenMask, accepting: bool, options: &JsonDecodeOptions)
    -> Result<u32, Int4JsonError> {
    checked_logits(logits, mask.vocab_size())?;
    let mut best = None;
    for (id, &value) in logits.iter().enumerate() {
        let id = id as u32;
        let legal = if id == options.eos_token_id { accepting }
            else { mask.contains(id) && !options.excluded_token_ids.contains(&id) };
        // Keep native first-id tie ordering, including signed zero.
        if legal && best.is_none_or(|(_, score)| value > score) { best = Some((id, value)); }
    }
    best.map(|(id, _)| id).ok_or_else(|| JsonDecodeError::NoLegalToken.into())
}

#[allow(clippy::too_many_arguments)]
fn drive<V, D, T, E, F>(group_width: usize, prompt: &[u32], program: &JsonProgram, vocabulary: &V,
    options: &JsonDecodeOptions, budget: Int4JsonBudget, driver: &mut D, finalize: F) -> Result<T, E>
where V: Vocabulary, D: Driver, E: From<Int4JsonError>, F: FnOnce(Int4JsonRun) -> Result<T, E> {
    let result = run(group_width, prompt, program, vocabulary, options, budget, driver).map_err(E::from).and_then(|run| {
        let next_token = run.output.token_ids.len();
        let output = finalize(run)?;
        if let Some(cause) = driver.control().checkpoint(next_token) {
            return Err(E::from(Int4JsonError::Decode(JsonDecodeError::Cancelled(cause))));
        }
        Ok(output)
    });
    if result.is_err() { driver.abort(); }
    result
}
fn run<V: Vocabulary, D: Driver>(group_width: usize, prompt: &[u32], program: &JsonProgram, vocabulary: &V,
    options: &JsonDecodeOptions, budget: Int4JsonBudget, driver: &mut D) -> Result<Int4JsonRun, Int4JsonError> {
    preflight(group_width, prompt, vocabulary, options, budget)?;
    let mut tokens = Vec::new();
    tokens.try_reserve_exact(options.max_new_tokens).map_err(|_| JsonDecodeError::AllocationRefused)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(program.max_output_bytes()).map_err(|_| JsonDecodeError::AllocationRefused)?;
    for (index, &token) in prompt.iter().enumerate() {
        if let Some(cause) = driver.control().prefill_checkpoint(index) {
            return Err(JsonDecodeError::Cancelled(cause).into());
        }
        driver.append(token)?;
    }
    let mut logits = driver.logits()?;
    checked_logits(&logits, vocabulary.width())?;
    let mut positions = prompt.len() as u64;
    let mut projected = vocabulary.width() as u64;
    let mut charged = 0_u64;
    let mut state = program.initial_state();
    for step in 0..options.max_new_tokens {
        if let Some(cause) = driver.control().checkpoint(step) { return Err(JsonDecodeError::Cancelled(cause).into()); }
        charged = charged.checked_add(vocabulary.mask_charge(budget.json.mask_limits) as u64)
            .filter(|&count| count <= budget.json.max_total_mask_node_visits)
            .ok_or(JsonDecodeError::BudgetExceeded("mask work"))?;
        let mask = vocabulary.mask(&state, budget.json.mask_limits, driver.control(), step)?;
        let selected = select(&logits, &mask, state.is_accepting(), options)?;
        if let Some(cause) = driver.control().checkpoint(step) { return Err(JsonDecodeError::Cancelled(cause).into()); }
        tokens.push(selected);
        if selected == options.eos_token_id {
            let json = String::from_utf8(bytes).map_err(|_| JsonDecodeError::IndependentValidation)?;
            program.validate_json(&json).map_err(|_| JsonDecodeError::IndependentValidation)?;
            let model_work = driver.work();
            if model_work != Int4Work::for_sequence(group_width, 0, positions as usize, projected as usize)? {
                return Err(Int4JsonError::WorkMismatch);
            }
            if let Some(cause) = driver.control().checkpoint(step) { return Err(JsonDecodeError::Cancelled(cause).into()); }
            return Ok(Int4JsonRun { schema_version: 1, execution: INT4_JSON_EXECUTION.to_owned(), group_width, model_work,
                output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT4_PROFILE.to_owned(),
                    token_ids: tokens, json, forward_positions: positions, projected_logits: projected,
                    mask_node_visit_charge: charged } });
        }
        let emitted = vocabulary.bytes(selected).filter(|bytes| !bytes.is_empty()).ok_or(JsonDecodeError::IllegalTransition)?;
        if bytes.len().checked_add(emitted.len()).is_none_or(|count| count > program.max_output_bytes()) {
            return Err(JsonDecodeError::BudgetExceeded("JSON bytes").into());
        }
        if !state.consume_bytes(emitted) { return Err(JsonDecodeError::IllegalTransition.into()); }
        bytes.extend_from_slice(emitted);
        if step + 1 == options.max_new_tokens { return Err(JsonDecodeError::BudgetExceeded("output tokens before EOS").into()); }
        // An accepting value still needs a newly scored EOS. Never fabricate
        // completion or skip the causal forward for its final content token.
        drop(logits);
        driver.append(selected)?;
        logits = driver.logits()?;
        checked_logits(&logits, vocabulary.width())?;
        positions += 1; projected += vocabulary.width() as u64;
    }
    Err(JsonDecodeError::BudgetExceeded("output tokens").into())
}

#[cfg(test)] mod tests;
