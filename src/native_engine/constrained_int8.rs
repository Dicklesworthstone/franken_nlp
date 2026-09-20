//! Full-vocabulary, schema-constrained generation on the real strict-int8 engine.
//!
//! Reuses the executable JSON/source grammar and pinned vocabulary mask oracle.
//! Only the last prompt position projects the head; every selected non-EOS
//! token is fed back before another full projection. No sparse shortcut, forced
//! byte shortcut, parse retry, model loader or runtime is introduced here.
//! The embedding host owns artifact authenticity and process admission. This
//! candidate does not assert BF16 parity or ratify quantized task quality.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::{mask::{DenseTokenMask, MaskWorkLimits, VocabMaskOracle}, runtime::{JsonProgram, JsonState}},
};
use super::{
    artifact_bridge::ArtifactIdentity,
    constrained::{JsonDecodeError, JsonDecodeOptions, JsonDecodeOutput, JsonWorkBudget},
    decode::{DecodeCancellationKind, DecodeStepControl},
    kv::KV_BYTES_PER_TOKEN,
    lmhead::NANBEIGE_VOCAB_SIZE,
    portable_int8::LinearRows,
    rope::DEFAULT_ADMITTED_CONTEXT_CAP,
    strict_int8::{Int8RunBudget, Int8Session, Int8Work, StrictInt8Engine, StrictInt8Error,
        STRICT_INT8_EXECUTION, STRICT_INT8_PROFILE},
};

pub const INT8_JSON_EXECUTION: &str = "portable-int8-constrained-json-final-prefill-head-v1";

/// Both budgets are enforced: JSON work is not a substitute for integer
/// decoder/head MACs and the 44-slot attention-pair ceiling. KV covers the
/// engine's COMPLETE resident capacity, not only this request's live prefix.
#[derive(Clone, Copy, Debug)]
pub struct Int8JsonBudget { pub native: Int8RunBudget, pub json: JsonWorkBudget }

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8JsonRun {
    pub schema_version: u32,
    pub execution: String,
    pub output: JsonDecodeOutput,
    pub model_work: Int8Work,
}

#[derive(Debug)]
pub enum Int8JsonError {
    Decode(JsonDecodeError), Native(StrictInt8Error), ModelIdentity, WorkMismatch,
}
impl fmt::Display for Int8JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(error) => write!(f, "int8 structured decoding refused: {error}"),
            Self::Native(error) => write!(f, "int8 structured native execution refused: {error}"),
            Self::ModelIdentity => f.write_str("int8 structured model/profile identity mismatch"),
            Self::WorkMismatch => f.write_str("int8 structured work differs from its native session"),
        }
    }
}
impl Error for Int8JsonError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Decode(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<JsonDecodeError> for Int8JsonError { fn from(e: JsonDecodeError) -> Self { Self::Decode(e) } }
impl From<StrictInt8Error> for Int8JsonError { fn from(e: StrictInt8Error) -> Self { Self::Native(e) } }
impl Int8JsonError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self {
            Self::Decode(JsonDecodeError::Cancelled(cause)) | Self::Native(StrictInt8Error::Cancelled(cause)) => Some(*cause),
            _ => None,
        }
    }
}

/// Exact ceiling for final-prefill-head execution. Head rows are charged at
/// every potential selection, including explicit terminal EOS. Integer decoder
/// projections and triangular causal attention are included by Int8Work.
/// This arithmetic does not allocate, load a model or confer admission.
pub fn planned_work(prompt_tokens: usize, max_new_tokens: usize) -> Result<Int8Work, Int8JsonError> {
    work_for_width(prompt_tokens, max_new_tokens, NANBEIGE_VOCAB_SIZE)
}
fn work_for_width(prompt_tokens: usize, max_new_tokens: usize, width: usize) -> Result<Int8Work, Int8JsonError> {
    if prompt_tokens == 0 || max_new_tokens == 0 || width == 0 || width > u32::MAX as usize {
        return Err(JsonDecodeError::InvalidRequest("empty prompt or output, or invalid vocabulary").into());
    }
    let positions = prompt_tokens.checked_add(max_new_tokens - 1).ok_or(StrictInt8Error::Work)?;
    if positions > DEFAULT_ADMITTED_CONTEXT_CAP { return Err(StrictInt8Error::Context.into()); }
    let rows = max_new_tokens.checked_mul(width).ok_or(StrictInt8Error::Work)?;
    Ok(Int8Work::for_sequence(0, positions, rows)?)
}

/// Low-level native boundary. The task layer must independently bind exact
/// prompt/schema/options/control IDs to the admitted identity before calling.
/// Here model/profile fields are checked against actual materialized weights;
/// no caller-supplied identity can relabel BF16 as this quantized execution.
#[allow(clippy::too_many_arguments)]
pub fn decode_json_int8<C: DecodeStepControl>(
    engine: &mut StrictInt8Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    budget: Int8JsonBudget, control: &mut C,
) -> Result<Int8JsonRun, Int8JsonError> {
    decode_json_int8_with(engine, identity, prompt, program, vocabulary, options, budget, control, Ok)
}

/// Finalization occurs INSIDE the session lifetime. A source-evidence or whole
/// envelope failure poisons the session just like a native/mask failure. RAII
/// clears all logical KV on success, error and unwind before returning to the
/// host, whose process/output reservations remain separately owned.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_json_int8_with<C, T, E, F>(
    engine: &mut StrictInt8Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    budget: Int8JsonBudget, control: &mut C, finalize: F,
) -> Result<T, E>
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
    let work = preflight(prompt, vocabulary, options, budget).map_err(E::from)?;
    check_capacity(work, engine.kv_cache().capacity_positions(), budget.json.max_kv_bytes).map_err(E::from)?;
    let mut session = engine.session(budget.native, control).map_err(Int8JsonError::from).map_err(E::from)?;
    drive(prompt, program, vocabulary, options, budget, &mut session, finalize)
}

pub(crate) fn check_profile(identity: &ExecutionIdentity) -> Result<(), Int8JsonError> {
    identity.validate().map_err(|_| Int8JsonError::ModelIdentity)?;
    if identity.numerics_profile != (NumericsProfile::StrictQuantized { version: 1 })
        || identity.backend_semantic_version != STRICT_INT8_EXECUTION || identity.kv_dtype != "bf16"
        || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None {
        return Err(Int8JsonError::ModelIdentity);
    }
    Ok(())
}
fn check_model(identity: &ExecutionIdentity, source: &ArtifactIdentity) -> Result<(), Int8JsonError> {
    check_profile(identity)?;
    if source.model_id != "Nanbeige4.2-3B" || source.revision != identity.source_revision
        || source.recipe_id != identity.quant_recipe
        || Sha256Digest::from_hex(&source.logical_model_sha256).ok() != Some(identity.logical_model_digest) {
        return Err(Int8JsonError::ModelIdentity);
    }
    Ok(())
}
fn check_capacity(work: Int8Work, capacity: usize, max_kv_bytes: u64) -> Result<(), Int8JsonError> {
    let capacity = u64::try_from(capacity).map_err(|_| StrictInt8Error::Memory)?;
    if work.forward_positions > capacity { return Err(StrictInt8Error::Context.into()); }
    let bytes = capacity.checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(StrictInt8Error::Memory)?;
    if bytes > max_kv_bytes { return Err(StrictInt8Error::Memory.into()); }
    Ok(())
}
fn preflight<V: Vocabulary>(prompt: &[u32], vocabulary: &V, options: &JsonDecodeOptions, budget: Int8JsonBudget)
    -> Result<Int8Work, Int8JsonError> {
    let width = vocabulary.width();
    let work = work_for_width(prompt.len(), options.max_new_tokens, width)?;
    if prompt.iter().chain(options.excluded_token_ids.iter()).chain(std::iter::once(&options.eos_token_id))
        .any(|&id| id as usize >= width) {
        return Err(JsonDecodeError::InvalidRequest("out-of-vocabulary token").into());
    }
    if budget.json.mask_limits.max_trie_node_visits == 0 || budget.json.mask_limits.checkpoint_interval_nodes == 0 {
        return Err(JsonDecodeError::InvalidRequest("zero mask work bound").into());
    }
    let masks = (vocabulary.mask_charge(budget.json.mask_limits) as u64)
        .checked_mul(options.max_new_tokens as u64).ok_or(StrictInt8Error::Work)?;
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

// Private static test seams. Public callers cannot mint a native result using
// a fake backend or unchecked byte table; they must supply the real engine and
// existing VocabMaskOracle, which owns token bytes and their trie together.
trait Driver {
    type Control: DecodeStepControl;
    fn control(&mut self) -> &mut Self::Control;
    fn append(&mut self, token: u32) -> Result<(), Int8JsonError>;
    fn logits(&mut self) -> Result<Vec<f32>, Int8JsonError>;
    fn work(&self) -> Int8Work;
    fn abort(&mut self);
}
impl<C: DecodeStepControl> Driver for Int8Session<'_, '_, C> {
    type Control = C;
    fn control(&mut self) -> &mut C { Int8Session::control(self) }
    fn append(&mut self, token: u32) -> Result<(), Int8JsonError> { Ok(Int8Session::append(self, token)?) }
    fn logits(&mut self) -> Result<Vec<f32>, Int8JsonError> { Ok(Int8Session::logits(self, LinearRows::All)?) }
    fn work(&self) -> Int8Work { Int8Session::work(self) }
    fn abort(&mut self) { Int8Session::abort(self); }
}
trait Vocabulary {
    fn width(&self) -> usize;
    fn bytes(&self, id: u32) -> Option<&[u8]>;
    fn mask_charge(&self, limits: MaskWorkLimits) -> usize;
    fn mask<C: DecodeStepControl>(&self, state: &JsonState<'_>, limits: MaskWorkLimits, control: &mut C, step: usize)
        -> Result<DenseTokenMask, Int8JsonError>;
}
impl Vocabulary for VocabMaskOracle {
    fn width(&self) -> usize { self.trie().vocab_size() }
    fn bytes(&self, id: u32) -> Option<&[u8]> { self.trie().token_bytes(id) }
    fn mask_charge(&self, limits: MaskWorkLimits) -> usize { self.trie().node_count().min(limits.max_trie_node_visits) }
    fn mask<C: DecodeStepControl>(&self, state: &JsonState<'_>, limits: MaskWorkLimits, control: &mut C, step: usize)
        -> Result<DenseTokenMask, Int8JsonError> {
        let mut cancellation = None;
        let mask = self.materialize(state, limits, |_| {
            cancellation = control.checkpoint(step); cancellation.is_none()
        });
        if let Some(cause) = cancellation { return Err(JsonDecodeError::Cancelled(cause).into()); }
        mask.map_err(JsonDecodeError::Mask).map_err(Into::into)
    }
}

fn checked_logits(logits: &[f32], width: usize) -> Result<(), Int8JsonError> {
    if logits.len() != width || logits.iter().any(|v| !v.is_finite()) {
        return Err(JsonDecodeError::InvalidLogits.into());
    }
    Ok(())
}
fn select(logits: &[f32], mask: &DenseTokenMask, accepting: bool, options: &JsonDecodeOptions) -> Result<u32, Int8JsonError> {
    checked_logits(logits, mask.vocab_size())?;
    let mut best = None;
    for (id, &value) in logits.iter().enumerate() {
        let id = id as u32;
        let legal = if id == options.eos_token_id { accepting }
            else { mask.contains(id) && !options.excluded_token_ids.contains(&id) };
        // Same first-index strict comparison as eager, including signed zero.
        if legal && best.is_none_or(|(_, score)| value > score) { best = Some((id, value)); }
    }
    best.map(|(id, _)| id).ok_or_else(|| JsonDecodeError::NoLegalToken.into())
}

#[allow(clippy::too_many_arguments)]
fn drive<V, D, T, E, F>(prompt: &[u32], program: &JsonProgram, vocabulary: &V,
    options: &JsonDecodeOptions, budget: Int8JsonBudget, driver: &mut D, finalize: F) -> Result<T, E>
where V: Vocabulary, D: Driver, E: From<Int8JsonError>, F: FnOnce(Int8JsonRun) -> Result<T, E> {
    let result = run(prompt, program, vocabulary, options, budget, driver).map_err(E::from).and_then(finalize);
    if result.is_err() { driver.abort(); }
    result
}
fn run<V: Vocabulary, D: Driver>(prompt: &[u32], program: &JsonProgram, vocabulary: &V,
    options: &JsonDecodeOptions, budget: Int8JsonBudget, driver: &mut D) -> Result<Int8JsonRun, Int8JsonError> {
    preflight(prompt, vocabulary, options, budget)?;
    let mut tokens = Vec::new();
    tokens.try_reserve_exact(options.max_new_tokens).map_err(|_| JsonDecodeError::AllocationRefused)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(program.max_output_bytes()).map_err(|_| JsonDecodeError::AllocationRefused)?;
    for (index, &token) in prompt.iter().enumerate() {
        if let Some(cause) = driver.control().prefill_checkpoint(index) { return Err(JsonDecodeError::Cancelled(cause).into()); }
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
            .filter(|&n| n <= budget.json.max_total_mask_node_visits).ok_or(JsonDecodeError::BudgetExceeded("mask work"))?;
        let mask = vocabulary.mask(&state, budget.json.mask_limits, driver.control(), step)?;
        let selected = select(&logits, &mask, state.is_accepting(), options)?;
        if let Some(cause) = driver.control().checkpoint(step) { return Err(JsonDecodeError::Cancelled(cause).into()); }
        tokens.push(selected);
        if selected == options.eos_token_id {
            let json = String::from_utf8(bytes).map_err(|_| JsonDecodeError::IndependentValidation)?;
            program.validate_json(&json).map_err(|_| JsonDecodeError::IndependentValidation)?;
            let model_work = driver.work();
            if model_work != Int8Work::for_sequence(0, positions as usize, projected as usize)? {
                return Err(Int8JsonError::WorkMismatch);
            }
            if let Some(cause) = driver.control().checkpoint(step) { return Err(JsonDecodeError::Cancelled(cause).into()); }
            return Ok(Int8JsonRun { schema_version: 1, execution: INT8_JSON_EXECUTION.to_owned(), model_work,
                output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.to_owned(),
                    token_ids: tokens, json, forward_positions: positions, projected_logits: projected,
                    mask_node_visit_charge: charged } });
        }
        let emitted = vocabulary.bytes(selected).filter(|b| !b.is_empty()).ok_or(JsonDecodeError::IllegalTransition)?;
        if bytes.len().checked_add(emitted.len()).is_none_or(|n| n > program.max_output_bytes()) {
            return Err(JsonDecodeError::BudgetExceeded("JSON bytes").into());
        }
        if !state.consume_bytes(emitted) { return Err(JsonDecodeError::IllegalTransition.into()); }
        bytes.extend_from_slice(emitted);
        if step + 1 == options.max_new_tokens { return Err(JsonDecodeError::BudgetExceeded("output tokens before EOS").into()); }
        // Even accepting JSON requires a newly scored EOS; never fabricate it.
        drop(logits);
        driver.append(selected)?;
        logits = driver.logits()?;
        checked_logits(&logits, vocabulary.width())?;
        positions += 1; projected += vocabulary.width() as u64;
    }
    Err(JsonDecodeError::BudgetExceeded("output tokens").into())
}

#[cfg(test)] mod tests;
