//! Addressed sampling under the existing exact JSON/source-language mask.
//!
//! This is a full-vocabulary candidate path. It does not claim that sampled
//! scores or justifications are calibrated, and it never samples from sparse
//! rows, a guessed mask, or a partially completed JSON value.

use super::*;
use sha2::{Digest, Sha256};
use crate::native_engine::{
    generation::{GenerationError, GenerationOptions, GenerationSampling, policy},
    sampler::{MAX_TOP_K, SAMPLER_VERSION, StableRequestKey},
};

pub const INT4_JSON_SAMPLED_EXECUTION: &str = "grouped-int4-sampled-constrained-json-final-prefill-head-v1";
/// Grammar legality precedes temperature, bounded top-k and exact nucleus.
/// The addressed draw itself uses the pinned `fnlp-sampler-v1` primitive.
pub const JSON_SAMPLER_VERSION: &str = "fnlp-json-mask-temperature-k-p-addressed-v1";

const MAX_IDENTITY_TEXT_BYTES: usize = 4096;
const IDENTITY_WORKSPACE_BYTES: u64 = 128 * 1024;

/// An explicit effective seed and semantic sample address. There is no global
/// RNG, implicit entropy, retry, repetition penalty or physical-row address.
/// The task compiler must bind these settings to its admitted identity.
#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JsonSamplingOptions {
    pub effective_seed: [u8; 32],
    pub temperature_milli: u32,
    /// None applies nucleus to every legal token, without a hidden top-k cap.
    pub top_k: Option<usize>,
    pub top_p_ppm: u32,
    pub sample_index: u64,
}

impl fmt::Debug for JsonSamplingOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JsonSamplingOptions").field("effective_seed", &"[redacted]")
            .field("temperature_milli", &self.temperature_milli).field("top_k", &self.top_k)
            .field("top_p_ppm", &self.top_p_ppm).field("sample_index", &self.sample_index).finish()
    }
}

impl JsonSamplingOptions {
    /// Validate before model loading, prompt construction or sampler allocation.
    pub fn validate(&self) -> Result<(), Int4JsonError> {
        if !(1..=100_000).contains(&self.temperature_milli)
            || !(1..=1_000_000).contains(&self.top_p_ppm)
            || self.top_k.is_some_and(|k| !(1..=MAX_TOP_K).contains(&k)) {
            return Err(JsonDecodeError::InvalidRequest("JSON sampling temperature, top-k or top-p").into());
        }
        Ok(())
    }
}

/// Conservative additional payload admission for one sampled sequence. It
/// includes the existing full-vocabulary sampler and bounded identity-key
/// construction. Sequential samples reuse this allowance; concurrent samples
/// would require independent reservations. This is not an RSS measurement.
pub fn json_sampling_workspace_bytes() -> Result<u64, Int4JsonError> {
    policy::workspace_bytes().map_err(sampling_error)?.checked_add(IDENTITY_WORKSPACE_BYTES)
        .ok_or_else(|| JsonDecodeError::BudgetExceeded("JSON sampler accounting").into())
}

/// Sample one independently validated complete JSON result on the admitted
/// grouped-INT4 engine. A terminal EOS must be scored in an accepting state.
/// Cancellation, token/byte exhaustion and invalid logits return no result.
#[allow(clippy::too_many_arguments)]
pub fn decode_json_int4_sampled<C: DecodeStepControl>(
    engine: &mut StrictInt4Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    sampling: &JsonSamplingOptions, budget: Int4JsonBudget, max_sampler_bytes: u64, control: &mut C,
) -> Result<Int4JsonRun, Int4JsonError> {
    decode_json_int4_sampled_with(engine, identity, prompt, program, vocabulary, options,
        sampling, budget, max_sampler_bytes, control, Ok)
}

/// Finalize while the exclusive native session still owns its complete KV.
/// The same session-drop cleanup applies on success, typed failure and unwind.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_json_int4_sampled_with<C, T, E, F>(
    engine: &mut StrictInt4Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    sampling: &JsonSamplingOptions, budget: Int4JsonBudget, max_sampler_bytes: u64,
    control: &mut C, finalize: F,
) -> Result<T, E>
where C: DecodeStepControl, E: From<Int4JsonError>, F: FnOnce(Int4JsonRun) -> Result<T, E> {
    sampling.validate().map_err(E::from)?;
    check_sampler_budget(max_sampler_bytes).map_err(E::from)?;
    check_request(engine, identity, prompt, vocabulary, options, budget).map_err(E::from)?;
    if let Some(cause) = control.checkpoint(0) {
        return Err(E::from(Int4JsonError::Decode(JsonDecodeError::Cancelled(cause))));
    }
    let mut selection = Sampled::new(identity, prompt, program, options, sampling, max_sampler_bytes)
        .map_err(E::from)?;
    let group_width = engine.model_identity().group_width();
    let mut session = engine.session(budget.native, control).map_err(Int4JsonError::from).map_err(E::from)?;
    drive_selected(group_width, prompt, program, vocabulary, options, budget,
        &mut session, &mut selection, finalize)
}

fn check_sampler_budget(max_sampler_bytes: u64) -> Result<(), Int4JsonError> {
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
        options: &JsonDecodeOptions, sampling: &JsonSamplingOptions, max_sampler_bytes: u64)
        -> Result<Self, Int4JsonError> {
        sampling.validate()?;
        check_sampler_budget(max_sampler_bytes)?;
        let key = request_key(identity, prompt, program, options, sampling)?;
        // Every processor except the explicitly named sampling policy is
        // neutral. The exact grammar/excluded/EOS mask is supplied per step.
        let mut processors = GenerationOptions::greedy(options.max_new_tokens,
            program.max_output_bytes(), options.eos_token_id);
        processors.sampling = GenerationSampling::Seeded {
            effective_seed: sampling.effective_seed, temperature_milli: sampling.temperature_milli,
            top_k: sampling.top_k, top_p_ppm: sampling.top_p_ppm,
        };
        Ok(Self { workspace: policy::Workspace::new(&[]).map_err(sampling_error)?,
            options: processors, key, sample_index: sampling.sample_index })
    }
}

impl Selection for Sampled {
    fn select(&mut self, logits: &[f32], mask: &DenseTokenMask, accepting: bool,
        options: &JsonDecodeOptions, step: usize) -> Result<u32, Int4JsonError> {
        checked_logits(logits, mask.vocab_size())?;
        self.workspace.select_legal(logits, &self.options, self.key, self.sample_index, step as u64, |id| {
            if id == options.eos_token_id { accepting }
            else { mask.contains(id) && !options.excluded_token_ids.contains(&id) }
        }).map_err(sampling_error)
    }
    fn execution(&self) -> &'static str { INT4_JSON_SAMPLED_EXECUTION }
}

/// Complete identity authority comes from its existing canonical projection.
/// Exact prompt, mask options and sampling settings are additionally framed
/// without allocating a second prompt or a vocabulary-sized banned-ID vector.
/// This key is private and never returned as a public content fingerprint.
fn request_key(identity: &ExecutionIdentity, prompt: &[u32], program: &JsonProgram, options: &JsonDecodeOptions,
    sampling: &JsonSamplingOptions) -> Result<StableRequestKey, Int4JsonError> {
    identity.validate().map_err(|_| Int4JsonError::ModelIdentity)?;
    if identity.sampler_version != JSON_SAMPLER_VERSION { return Err(Int4JsonError::ModelIdentity); }
    [&identity.source_revision, &identity.artifact_format, &identity.quant_recipe,
        &identity.task_spec, &identity.grammar_compiler_version, &identity.kv_dtype,
        &identity.sampler_version, &identity.backend_semantic_version]
        .into_iter().map(String::len)
        .chain(identity.host_class.as_ref().map(String::len))
        .chain(identity.compiler_identity.as_ref().map(String::len))
        .try_fold(0_usize, usize::checked_add).filter(|&bytes| bytes <= MAX_IDENTITY_TEXT_BYTES)
        .ok_or(JsonDecodeError::InvalidRequest("JSON sampling identity text bound"))?;
    let identity_key = identity.receipt_identity().map_err(|_| Int4JsonError::ModelIdentity)?;
    let mut hasher = Sha256::new();
    frame(&mut hasher, INT4_JSON_SAMPLED_EXECUTION.as_bytes());
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

fn sampling_error(error: GenerationError) -> Int4JsonError {
    match error {
        GenerationError::Allocation => JsonDecodeError::AllocationRefused.into(),
        GenerationError::InvalidLogits => JsonDecodeError::InvalidLogits.into(),
        GenerationError::NoLegalToken => JsonDecodeError::NoLegalToken.into(),
        GenerationError::Limit(axis) => JsonDecodeError::BudgetExceeded(axis).into(),
        GenerationError::Contract(axis) => JsonDecodeError::InvalidRequest(axis).into(),
        GenerationError::Cancelled(cause) => JsonDecodeError::Cancelled(cause).into(),
        GenerationError::Identity => Int4JsonError::ModelIdentity,
        _ => JsonDecodeError::InvalidRequest("JSON sampling processor contract").into(),
    }
}

#[cfg(test)]
mod tests;
