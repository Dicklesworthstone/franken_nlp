//! Explicit addressed sampling over the complete INT8 JSON/source grammar.
//!
//! The shared decoder still owns exact masking, scored terminal EOS, native
//! work reconciliation and independent validation. Sampling reuses the existing
//! generation processors; it never retries, truncates the legal set, or adds
//! implicit entropy. Prompt scheduling is absent from the private draw key.

use super::*;
use sha2::{Digest, Sha256};
use crate::native_engine::{
    generation::{GenerationError, GenerationOptions, GenerationSampling, policy},
    sampler::{MAX_TOP_K, SAMPLER_VERSION, StableRequestKey},
    strict_int8::prefill::Int8PrefillLimits,
};

mod reuse;
pub use reuse::{INT8_JSON_SAMPLED_REUSED_PREFILL_EXECUTION, json_sampling_reuse_workspace_bytes};
pub(crate) use reuse::decode_json_int8_sampled_reusing_prefill_with;
mod selected;
pub use selected::{INT8_JSON_SAMPLED_SPARSE_EXECUTION, INT8_JSON_SAMPLED_SPARSE_REUSED_PREFILL_EXECUTION,
    decode_json_int8_sampled_sparse, decode_json_int8_sampled_sparse_layer_major};
pub(crate) use selected::{decode_json_int8_sampled_sparse_with, decode_json_int8_sampled_sparse_layer_major_with,
    decode_json_int8_sampled_sparse_reusing_prefill_with};

pub const INT8_JSON_SAMPLED_EXECUTION: &str =
    "portable-int8-sampled-constrained-json-final-prefill-head-v1";
/// Grammar legality, then temperature, optional top-k and exact nucleus,
/// followed by the pinned addressed sampler. This is not a confidence score.
pub const JSON_SAMPLER_VERSION: &str = "fnlp-json-mask-temperature-k-p-addressed-v1";

const MAX_IDENTITY_TEXT_BYTES: usize = 4096;
const IDENTITY_WORKSPACE_BYTES: u64 = 128 * 1024;

/// An effective seed and semantic sample address supplied by the task owner.
/// No global RNG, implicit entropy, physical row, or scheduling order is used.
#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8JsonSamplingOptions {
    pub effective_seed: [u8; 32],
    pub temperature_milli: u32,
    /// None computes nucleus over every legal token without a hidden top-k.
    pub top_k: Option<usize>,
    pub top_p_ppm: u32,
    pub sample_index: u64,
}
impl fmt::Debug for Int8JsonSamplingOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Int8JsonSamplingOptions").field("effective_seed", &"[redacted]")
            .field("temperature_milli", &self.temperature_milli).field("top_k", &self.top_k)
            .field("top_p_ppm", &self.top_p_ppm).field("sample_index", &self.sample_index).finish()
    }
}
impl Int8JsonSamplingOptions {
    /// Validate before prompt construction, model loading or sampler allocation.
    pub fn validate(&self) -> Result<(), Int8JsonError> {
        if !(1..=100_000).contains(&self.temperature_milli)
            || !(1..=1_000_000).contains(&self.top_p_ppm)
            || self.top_k.is_some_and(|k| !(1..=MAX_TOP_K).contains(&k)) {
            return Err(JsonDecodeError::InvalidRequest("JSON sampling temperature, top-k or top-p").into());
        }
        Ok(())
    }
}

/// Additional payload admission for one sampled native sequence, including
/// the existing full-vocabulary sampler and bounded identity construction.
/// Sequential samples reuse this amount; concurrent samples need independent
/// reservations. Host overhead, output and KV are accounted for separately.
pub fn json_sampling_workspace_bytes() -> Result<u64, Int8JsonError> {
    policy::workspace_bytes().map_err(sampling_error)?.checked_add(IDENTITY_WORKSPACE_BYTES)
        .ok_or_else(|| JsonDecodeError::BudgetExceeded("JSON sampler accounting").into())
}

/// Produce one independently validated JSON result on the actual strict INT8
/// engine. EOS is projected and sampled only in an accepting grammar state.
/// Token/byte/work exhaustion and cancellation return no completed result.
#[allow(clippy::too_many_arguments)]
pub fn decode_json_int8_sampled<C: DecodeStepControl>(
    engine: &mut StrictInt8Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    sampling: &Int8JsonSamplingOptions, budget: Int8JsonBudget, max_sampler_bytes: u64,
    control: &mut C,
) -> Result<Int8JsonRun, Int8JsonError> {
    decode_json_int8_sampled_with(engine, identity, prompt, program, vocabulary,
        options, sampling, budget, max_sampler_bytes, control, Ok)
}

/// Use bounded layer-major prompt morsels before the same addressed decoder.
/// The extra prompt workspace is admitted by the caller. The exact prompt,
/// sample address, grammar, EOS policy and per-row arithmetic are unchanged.
#[allow(clippy::too_many_arguments)]
pub fn decode_json_int8_sampled_layer_major<C: DecodeStepControl>(
    engine: &mut StrictInt8Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    sampling: &Int8JsonSamplingOptions, budget: Int8JsonBudget, max_sampler_bytes: u64,
    prefill: Int8PrefillLimits, control: &mut C,
) -> Result<Int8JsonRun, Int8JsonError> {
    decode_json_int8_sampled_layer_major_with(engine, identity, prompt, program, vocabulary,
        options, sampling, budget, max_sampler_bytes, prefill, control, Ok)
}

/// Task-specific validation remains inside the same exclusive native session.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_json_int8_sampled_with<C, T, E, F>(
    engine: &mut StrictInt8Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    sampling: &Int8JsonSamplingOptions, budget: Int8JsonBudget, max_sampler_bytes: u64,
    control: &mut C, finalize: F,
) -> Result<T, E>
where C: DecodeStepControl, E: From<Int8JsonError>, F: FnOnce(Int8JsonRun) -> Result<T, E> {
    decode_sampled_scheduled(engine, identity, prompt, program, vocabulary, options,
        sampling, budget, max_sampler_bytes, None, control, finalize)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_json_int8_sampled_layer_major_with<C, T, E, F>(
    engine: &mut StrictInt8Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    sampling: &Int8JsonSamplingOptions, budget: Int8JsonBudget, max_sampler_bytes: u64,
    prefill: Int8PrefillLimits, control: &mut C, finalize: F,
) -> Result<T, E>
where C: DecodeStepControl, E: From<Int8JsonError>, F: FnOnce(Int8JsonRun) -> Result<T, E> {
    decode_sampled_scheduled(engine, identity, prompt, program, vocabulary, options,
        sampling, budget, max_sampler_bytes, Some(prefill), control, finalize)
}

#[allow(clippy::too_many_arguments)]
fn decode_sampled_scheduled<C, T, E, F>(
    engine: &mut StrictInt8Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    sampling: &Int8JsonSamplingOptions, budget: Int8JsonBudget, max_sampler_bytes: u64,
    prefill: Option<Int8PrefillLimits>, control: &mut C, finalize: F,
) -> Result<T, E>
where C: DecodeStepControl, E: From<Int8JsonError>, F: FnOnce(Int8JsonRun) -> Result<T, E> {
    if let Some(limits) = prefill { limits.validate().map_err(Int8JsonError::from).map_err(E::from)?; }
    sampling.validate().map_err(E::from)?;
    check_sampler_budget(max_sampler_bytes).map_err(E::from)?;
    check_request(engine, identity, prompt, vocabulary, options, budget).map_err(E::from)?;
    if let Some(cause) = control.checkpoint(0) {
        return Err(E::from(Int8JsonError::Decode(JsonDecodeError::Cancelled(cause))));
    }
    let mut selection = Sampled::new(identity, prompt, program, options, sampling, max_sampler_bytes)
        .map_err(E::from)?;
    let mut session = engine.session(budget.native, control).map_err(Int8JsonError::from).map_err(E::from)?;
    match prefill {
        Some(limits) => prefill::drive_layer_major_selected(prompt, program, vocabulary, options,
            budget, limits, &mut session, &mut selection, finalize),
        None => drive_selected(prompt, program, vocabulary, options, budget,
            &mut session, &mut selection, finalize),
    }
}

fn check_sampler_budget(max_sampler_bytes: u64) -> Result<(), Int8JsonError> {
    if json_sampling_workspace_bytes()? > max_sampler_bytes {
        return Err(JsonDecodeError::BudgetExceeded("JSON sampler storage").into());
    }
    Ok(())
}

struct Sampled {
    workspace: policy::Workspace,
    options: GenerationOptions,
    key: StableRequestKey,
    sample_index: u64,
}
impl Sampled {
    fn new(identity: &ExecutionIdentity, prompt: &[u32], program: &JsonProgram,
        options: &JsonDecodeOptions, sampling: &Int8JsonSamplingOptions, max_sampler_bytes: u64)
        -> Result<Self, Int8JsonError> {
        sampling.validate()?;
        check_sampler_budget(max_sampler_bytes)?;
        let key = request_key(identity, prompt, program, options, sampling)?;
        // Other processors stay neutral. Legality is the exact grammar plus
        // excluded template controls and the explicit EOS acceptance rule.
        let mut processors = GenerationOptions::greedy(options.max_new_tokens,
            program.max_output_bytes(), options.eos_token_id);
        processors.sampling = GenerationSampling::Seeded {
            effective_seed: sampling.effective_seed, temperature_milli: sampling.temperature_milli,
            top_k: sampling.top_k, top_p_ppm: sampling.top_p_ppm,
        };
        Ok(Self { workspace: policy::Workspace::for_generation(&[], &processors).map_err(sampling_error)?,
            options: processors, key, sample_index: sampling.sample_index })
    }
}
impl Selection for Sampled {
    fn select<C: DecodeStepControl>(&mut self, logits: &[f32], mask: &DenseTokenMask, accepting: bool,
        options: &JsonDecodeOptions, step: usize, control: &mut C) -> Result<u32, Int8JsonError> {
        if logits.len() != mask.vocab_size() { return Err(JsonDecodeError::InvalidLogits.into()); }
        self.workspace.select_legal_with_control(logits, &self.options, self.key, self.sample_index, step, |id| {
            if id == options.eos_token_id { accepting }
            else { mask.contains(id) && !options.excluded_token_ids.contains(&id) }
        }, control).map_err(sampling_error)
    }
    fn execution(&self) -> &'static str { INT8_JSON_SAMPLED_EXECUTION }
}

/// Bind actual grammar/source bytes as well as the declared execution
/// identity, prompt, exclusions and every sampling setting. The digest is
/// private; it never becomes a public document fingerprint or cache key.
fn request_key(identity: &ExecutionIdentity, prompt: &[u32], program: &JsonProgram,
    options: &JsonDecodeOptions, sampling: &Int8JsonSamplingOptions)
    -> Result<StableRequestKey, Int8JsonError> {
    check_profile(identity)?;
    if identity.sampler_version != JSON_SAMPLER_VERSION { return Err(Int8JsonError::ModelIdentity); }
    [&identity.source_revision, &identity.artifact_format, &identity.quant_recipe,
        &identity.task_spec, &identity.grammar_compiler_version, &identity.kv_dtype,
        &identity.sampler_version, &identity.backend_semantic_version]
        .into_iter().map(String::len)
        .chain(identity.host_class.as_ref().map(String::len))
        .chain(identity.compiler_identity.as_ref().map(String::len))
        .try_fold(0_usize, usize::checked_add).filter(|&bytes| bytes <= MAX_IDENTITY_TEXT_BYTES)
        .ok_or(JsonDecodeError::InvalidRequest("JSON sampling identity text bound"))?;
    let identity_key = identity.receipt_identity().map_err(|_| Int8JsonError::ModelIdentity)?;
    let mut hasher = Sha256::new();
    frame(&mut hasher, INT8_JSON_SAMPLED_EXECUTION.as_bytes());
    frame(&mut hasher, JSON_SAMPLER_VERSION.as_bytes());
    frame(&mut hasher, SAMPLER_VERSION.as_bytes());
    frame(&mut hasher, identity_key.to_hex().as_bytes());
    hasher.update(program.sampling_digest());
    hasher.update((prompt.len() as u64).to_be_bytes());
    for &token in prompt { hasher.update(token.to_be_bytes()); }
    hasher.update((options.max_new_tokens as u64).to_be_bytes());
    hasher.update(options.eos_token_id.to_be_bytes());
    hasher.update((options.excluded_token_ids.len() as u64).to_be_bytes());
    for &token in &options.excluded_token_ids { hasher.update(token.to_be_bytes()); }
    hasher.update(sampling.effective_seed);
    hasher.update(sampling.temperature_milli.to_be_bytes());
    hasher.update([u8::from(sampling.top_k.is_some())]);
    hasher.update((sampling.top_k.unwrap_or(0) as u64).to_be_bytes());
    hasher.update(sampling.top_p_ppm.to_be_bytes());
    hasher.update(sampling.sample_index.to_be_bytes());
    Ok(StableRequestKey::from_canonical_digest(hasher.finalize().into()))
}
fn frame(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}
fn sampling_error(error: GenerationError) -> Int8JsonError {
    match error {
        GenerationError::Allocation => JsonDecodeError::AllocationRefused.into(),
        GenerationError::InvalidLogits => JsonDecodeError::InvalidLogits.into(),
        GenerationError::NoLegalToken => JsonDecodeError::NoLegalToken.into(),
        GenerationError::Limit(axis) => JsonDecodeError::BudgetExceeded(axis).into(),
        GenerationError::Contract(axis) => JsonDecodeError::InvalidRequest(axis).into(),
        GenerationError::Cancelled(cause) => JsonDecodeError::Cancelled(cause).into(),
        GenerationError::Identity => Int8JsonError::ModelIdentity,
        _ => JsonDecodeError::InvalidRequest("JSON sampling processor contract").into(),
    }
}

#[cfg(test)]
mod tests;
