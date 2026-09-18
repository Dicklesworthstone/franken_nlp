//! Bounded, identity-bound greedy and addressably sampled generation.
//!
//! This executes the existing eager model and pinned sampler, not a second
//! runtime or model loader. Compilation binds all effective options and the
//! private exact prompt. Execution requires the exact admitted identity. The
//! caller owns process/model admission, output delivery and panic supervision.
//! Unlike the legacy raw greedy seam, EOS has no content bytes; byte-stop
//! suffixes retain their bytes. No tools, thinking mode or hidden retries exist.
//! The quantized module reuses this compiler/cursor through a distinct plan
//! type, so a quantized plan cannot enter an eager or grouped-eager driver.

use std::{collections::BTreeMap, error::Error, fmt};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use crate::{canonjson, execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode}};
use super::{
    decode::{DecodeByteDecoder, DecodeCancellationKind, DecodeEventSink, DecodeScoreSpace,
        DecodeStepControl, DecodeTokenEvent, DECODE_TOKEN_EVENT_SCHEMA_VERSION},
    hf_bf16_eager::{HfBf16EagerEngine, HfBf16EagerError, HF_BF16_EAGER_PROFILE, candidate_scoring::ClearCache},
    kv::KV_BYTES_PER_TOKEN, lmhead::NANBEIGE_VOCAB_SIZE,
    sampler::{Seed256, StableRequestKey, SAMPLER_VERSION},
};
pub mod batched;
pub mod quantized;
mod config;
mod cursor;
mod policy;
#[cfg(test)] mod tests;

pub const GENERATION_VERSION: &str = "eager-addressed-generation-v1";
pub const PROCESSOR_VERSION: &str = "ban-minimum-repeat-presence-frequency-bias-temperature-k-p-v1";

/// An explicit seed is mandatory here. Only the admission owner may obtain a
/// missing seed from Cx::random_bytes; a hot-loop/global RNG is never consulted.
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GenerationSampling {
    Greedy,
    Seeded {
        effective_seed: [u8; 32],
        temperature_milli: u32,
        /// None means no top-k restriction: nucleus uses the full legal vocab.
        top_k: Option<usize>,
        top_p_ppm: u32,
    },
}

/// Fixed-point processor settings avoid nonfinite/ambiguous wire options.
/// Repetition counts include prompt and previously committed generated tokens.
/// Bias is applied after penalties; temperature, top-k, then exact top-p follow.
/// Logprobs, when requested, are RAW full-vocabulary scores, not the processed
/// sampling probability and not confidence. All unsupported options are refused.
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    /// Minimum NON-EOS generated tokens before EOS may be selected.
    pub min_new_tokens: usize,
    pub max_output_bytes: usize,
    pub eos_token_ids: Vec<u32>,
    pub banned_token_ids: Vec<u32>,
    /// Match exact suffixes after a commit, including across token boundaries.
    /// Matching bytes are retained, so streaming never retracts published bytes.
    pub stop_suffixes: Vec<Vec<u8>>,
    pub capture_logprobs: bool,
    pub repetition_penalty_milli: u32,
    pub presence_penalty_milli: i32,
    pub frequency_penalty_milli: i32,
    /// Wire keys must be canonical unsigned decimal token IDs. Alternate
    /// spellings cannot collapse two distinct JSON keys into one token policy.
    #[serde(deserialize_with = "config::deserialize_bias")]
    pub logit_bias_milli: BTreeMap<u32, i32>,
    pub sampling: GenerationSampling,
}
impl GenerationOptions {
    pub fn greedy(max_new_tokens: usize, max_output_bytes: usize, eos: u32) -> Self {
        Self { max_new_tokens, min_new_tokens: 0, max_output_bytes, eos_token_ids: vec![eos],
            banned_token_ids: Vec::new(), stop_suffixes: Vec::new(), capture_logprobs: false,
            repetition_penalty_milli: 1000, presence_penalty_milli: 0, frequency_penalty_milli: 0,
            logit_bias_milli: BTreeMap::new(), sampling: GenerationSampling::Greedy }
    }
    /// Check caller-owned option sizes and domains without cloning options,
    /// rendering/tokenizing text, allocating a sampler or touching the engine.
    pub fn validate(&self, limits: GenerationLimits) -> Result<(), GenerationError> {
        policy::validate(self, limits)?;
        if policy::workspace_bytes()? > limits.max_sampler_bytes { return Err(GenerationError::Limit("sampler storage")); }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GenerationLimits {
    pub max_prompt_tokens: usize,
    pub max_new_tokens: usize,
    pub max_output_bytes: usize,
    pub max_sampler_bytes: u64,
}
impl Default for GenerationLimits {
    fn default() -> Self {
        Self { max_prompt_tokens: 8192, max_new_tokens: 1024, max_output_bytes: 4 * 1024 * 1024,
            max_sampler_bytes: 32 * 1024 * 1024 }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct GenerationBudget {
    pub max_forward_positions: u64,
    pub max_projected_logits: u64,
    pub max_kv_bytes: u64,
    pub max_sampler_bytes: u64,
}
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationWork {
    pub forward_positions: u64,
    pub projected_logits: u64,
    /// Actual addressed draws, including a final byte-budget-refused proposal.
    pub sampled_steps: u64,
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationFinish { Eos, StopSuffix, TokenLimit, ByteLimit }

/// A completed sequence, never a cancellation disguised as successful output.
/// token_ids includes terminal EOS; content_bytes excludes EOS. Stop suffixes
/// remain. Token events concatenate to exactly content_bytes. Raw scores include
/// a scored EOS, with its actual full-vocabulary denominator. `execution` names
/// whether intermediate prompt positions also computed a full vocabulary row.
#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedSequence {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub request_seq: u64,
    pub sample_index: u64,
    pub token_ids: Vec<u32>,
    pub content_bytes: Vec<u8>,
    pub finish_reason: GenerationFinish,
    pub effective_seed: Option<String>,
    pub token_logprobs: Option<Vec<f32>>,
    pub logprob_score_space: Option<DecodeScoreSpace>,
    pub native_work: GenerationWork,
}

#[derive(Debug)]
pub enum GenerationError {
    Contract(&'static str), Limit(&'static str), Identity, Allocation, InvalidLogits,
    NoLegalToken, Decoder, DecoderNotPrefixStable, Stream,
    EngineAlreadyPrimed, Engine(HfBf16EagerError), Cancelled(DecodeCancellationKind),
}
impl fmt::Display for GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(axis) => write!(f, "generation contract refused: {axis}"),
            Self::Limit(axis) => write!(f, "generation budget exceeded: {axis}"),
            Self::Cancelled(kind) => write!(f, "generation cancelled: {kind:?}"),
            Self::Identity => f.write_str("generation admitted identity mismatch"),
            Self::Allocation => f.write_str("generation allocation refused"),
            Self::InvalidLogits => f.write_str("generation requires a finite complete projection"),
            Self::NoLegalToken => f.write_str("generation has no legal next token"),
            Self::Decoder => f.write_str("generation byte decoding failed"),
            Self::DecoderNotPrefixStable => f.write_str("generation decoder rewrote committed bytes"),
            Self::Stream => f.write_str("generation stream delivery failed"),
            Self::EngineAlreadyPrimed => f.write_str("generation requires an empty admitted engine"),
            Self::Engine(_) => f.write_str("generation native forward failed"),
        }
    }
}
impl Error for GenerationError {}

#[derive(Clone, Copy, Eq, PartialEq)]
enum GenerationBackend { Eager, Int8 }
impl GenerationBackend {
    fn profile(self) -> NumericsProfile {
        match self { Self::Eager => NumericsProfile::HfBf16Eager, Self::Int8 => NumericsProfile::StrictQuantized { version: 1 } }
    }
    fn version(self) -> &'static str {
        match self { Self::Eager => GENERATION_VERSION, Self::Int8 => quantized::INT8_GENERATION_VERSION }
    }
}

/// Immutable private compiled input. No Debug/Deserialize/Serialize: neither
/// source tokens nor the raw content-derived sampling key enter telemetry.
pub struct GenerationPlan {
    prompt: Vec<u32>, options: GenerationOptions, identity: ExecutionIdentity,
    key: StableRequestKey, sample_index: u64, bound: GenerationWork, sampler_bytes: u64,
}
impl GenerationPlan {
    /// Bind task-owned prompt/processor/sampler fields at COMPILE time while
    /// preserving artifact/backend facts. This does not activate a model or
    /// certify their truth. The host must admit the resulting exact identity.
    /// item_id is a stable caller/job id, never a physical row/request sequence.
    pub fn compile(prompt: Vec<u32>, options: GenerationOptions, identity: ExecutionIdentity,
        item_id: &str, sample_index: u64, limits: GenerationLimits) -> Result<Self, GenerationError> {
        Self::compile_for_backend(prompt, options, identity, item_id, sample_index, limits, GenerationBackend::Eager)
    }
    /// Only closed, statically typed native drivers may select this backend.
    /// Eager identity/key bytes retain their previous version and framing.
    fn compile_for_backend(prompt: Vec<u32>, mut options: GenerationOptions, mut identity: ExecutionIdentity,
        item_id: &str, sample_index: u64, limits: GenerationLimits, backend: GenerationBackend) -> Result<Self, GenerationError> {
        identity.validate().map_err(|_| GenerationError::Identity)?;
        if !matches!(identity.task_spec.as_str(), "generate-v1" | "chat-v1")
            || identity.numerics_profile != backend.profile() || identity.kv_dtype != "bf16"
            || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None {
            return Err(GenerationError::Contract("task/profile/thinking/tools"));
        }
        if backend == GenerationBackend::Int8
            && identity.backend_semantic_version != super::strict_int8::STRICT_INT8_EXECUTION {
            return Err(GenerationError::Identity);
        }
        if item_id.is_empty() || item_id.len() > 256 || item_id.chars().any(char::is_control) {
            return Err(GenerationError::Contract("stable item id"));
        }
        if prompt.is_empty() || prompt.len() > limits.max_prompt_tokens
            || prompt.iter().any(|&id| id as usize >= NANBEIGE_VOCAB_SIZE) {
            return Err(GenerationError::Contract("prompt"));
        }
        options.validate(limits)?;
        options.eos_token_ids.sort_unstable(); options.eos_token_ids.dedup();
        options.banned_token_ids.sort_unstable(); options.banned_token_ids.dedup();
        let positions = prompt.len().checked_add(options.max_new_tokens - 1)
            .and_then(|n| u64::try_from(n).ok()).ok_or(GenerationError::Limit("positions"))?;
        let head_positions = if backend == GenerationBackend::Eager { positions } else { options.max_new_tokens as u64 };
        let bound = GenerationWork { forward_positions: positions,
            projected_logits: head_positions.checked_mul(NANBEIGE_VOCAB_SIZE as u64).ok_or(GenerationError::Limit("logits"))?,
            sampled_steps: if matches!(&options.sampling, GenerationSampling::Seeded { .. }) { options.max_new_tokens as u64 } else { 0 } };
        let sampler_bytes = policy::workspace_bytes()?;
        identity.prompt_digest = digest(&prompt)?;
        // Item/sample selection affects addressed draws, so admission/resume
        // identity must bind it too, not just the decoder's private request key.
        let version = backend.version();
        identity.decision_policy_digest = digest(&(version, PROCESSOR_VERSION, &options, item_id, sample_index))?;
        identity.sampler_version = match &options.sampling {
            GenerationSampling::Greedy => "fnlp-greedy-v1", GenerationSampling::Seeded { .. } => SAMPLER_VERSION,
        }.to_owned();
        let bytes = canonjson::canonical_bytes(&(version, PROCESSOR_VERSION, &identity, &prompt, &options, item_id))
            .map_err(|_| GenerationError::Identity)?;
        let key = StableRequestKey::from_canonical_digest(Sha256::digest(&bytes).into());
        Ok(Self { prompt, options, identity, key, sample_index, bound, sampler_bytes })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn options(&self) -> &GenerationOptions { &self.options }
    pub fn planned_work(&self) -> GenerationWork { self.bound }
    pub fn sampler_bytes(&self) -> u64 { self.sampler_bytes }
    pub fn prompt_tokens(&self) -> usize { self.prompt.len() }
    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), GenerationError> {
        admitted.validate().map_err(|_| GenerationError::Identity)?;
        if canonjson::canonical_bytes(admitted).map_err(|_| GenerationError::Identity)?
            != canonjson::canonical_bytes(&self.identity).map_err(|_| GenerationError::Identity)? {
            return Err(GenerationError::Identity);
        }
        Ok(())
    }
    pub fn preflight_eager(&self, admitted: &ExecutionIdentity, engine: &HfBf16EagerEngine,
        budget: GenerationBudget) -> Result<(), GenerationError> {
        self.verify_identity(admitted)?;
        if self.identity.numerics_profile != NumericsProfile::HfBf16Eager { return Err(GenerationError::Identity); }
        if !engine.kv_cache().all_slots_have_len(0) { return Err(GenerationError::EngineAlreadyPrimed); }
        if engine.profile() != HF_BF16_EAGER_PROFILE { return Err(GenerationError::Identity); }
        let capacity = engine.kv_cache().capacity_positions() as u64;
        if self.bound.forward_positions > capacity { return Err(GenerationError::Limit("context")); }
        let kv = capacity.checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(GenerationError::Limit("KV arithmetic"))?;
        if kv > budget.max_kv_bytes { return Err(GenerationError::Limit("complete KV reservation")); }
        self.check_budget(budget)
    }
    fn check_budget(&self, budget: GenerationBudget) -> Result<(), GenerationError> {
        if self.bound.forward_positions > budget.max_forward_positions || self.bound.projected_logits > budget.max_projected_logits
            || self.sampler_bytes > budget.max_sampler_bytes { return Err(GenerationError::Limit("work or sampler storage")); }
        Ok(())
    }
    pub fn execute_eager<D: DecodeByteDecoder, C: DecodeStepControl>(&self,
        admitted: &ExecutionIdentity, engine: &mut HfBf16EagerEngine, decoder: &D,
        request_seq: u64, budget: GenerationBudget, control: &mut C) -> Result<GeneratedSequence, GenerationError> {
        self.execute_eager_with_sink(admitted, engine, decoder, request_seq, budget, &mut Discard, control)
    }
    /// One physical forward at a time. ClearCache owns logical KV cleanup on
    /// success, error, cancellation and unwind. No pre-existing KV is discarded.
    /// Stream sinks have the existing reserve/permit contract; a failed permit
    /// is fatal and never retried. The embedding host retains admission guards.
    pub fn execute_eager_with_sink<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl>(&self,
        admitted: &ExecutionIdentity, engine: &mut HfBf16EagerEngine, decoder: &D,
        request_seq: u64, budget: GenerationBudget, sink: &mut S, control: &mut C) -> Result<GeneratedSequence, GenerationError> {
        self.preflight_eager(admitted, engine, budget)?;
        let guard = ClearCache(engine);
        self.run(decoder, request_seq, sink, control, |token| {
            guard.0.decode(token).map(|row| row.logits).map_err(GenerationError::Engine)
        })
    }
    /// The scalar and grouped drivers share this exact cursor/sampler/stop
    /// state machine. Scalar eager still projects every prompt position; the
    /// grouped driver skips only intermediate prompt lm-heads and records that.
    fn run<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl,
        F: FnMut(u32) -> Result<Vec<f32>, GenerationError>>(&self, decoder: &D, request_seq: u64,
        sink: &mut S, control: &mut C, mut forward: F) -> Result<GeneratedSequence, GenerationError> {
        let mut row = cursor::Cursor::new(self, request_seq, GENERATION_VERSION)?;
        while !row.done {
            row.before_forward(control)?;
            let (token, needs_selection) = row.next_token()?;
            let logits = forward(token)?;
            check_logits(&logits)?;
            row.record_forward(true)?;
            if needs_selection { row.emit_next(&logits, decoder, sink, control)?; }
        }
        row.finish()
    }
}
fn checkpoint<C: DecodeStepControl>(control: &mut C, index: usize) -> Result<(), GenerationError> {
    match control.checkpoint(index) { Some(cause) => Err(GenerationError::Cancelled(cause)), None => Ok(()) }
}
fn check_logits(logits: &[f32]) -> Result<(), GenerationError> {
    if logits.len() != NANBEIGE_VOCAB_SIZE || logits.iter().any(|v| !v.is_finite()) { return Err(GenerationError::InvalidLogits); }
    Ok(())
}
fn raw_logprob(logits: &[f32], selected: u32) -> Result<f32, GenerationError> {
    check_logits(logits)?;
    let maximum = logits.iter().copied().map(f64::from).fold(f64::NEG_INFINITY, f64::max);
    let sum = logits.iter().fold(0.0_f64, |s, &x| s + (f64::from(x) - maximum).exp());
    let score = ((f64::from(logits[selected as usize]) - maximum) - sum.ln()) as f32;
    if !score.is_finite() { return Err(GenerationError::InvalidLogits); }
    Ok(score)
}
fn digest<T: Serialize>(value: &T) -> Result<Sha256Digest, GenerationError> {
    canonjson::canonical_bytes(value).map(|bytes| Sha256Digest::of_bytes(&bytes)).map_err(|_| GenerationError::Identity)
}
fn reserved<T>(length: usize) -> Result<Vec<T>, GenerationError> {
    let mut result = Vec::new(); result.try_reserve_exact(length).map_err(|_| GenerationError::Allocation)?; Ok(result)
}
struct Discard;
impl DecodeEventSink for Discard {
    type Permit = (); type Error = std::convert::Infallible;
    fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
    fn permit(&mut self, _: (), _: DecodeTokenEvent) -> Result<(), Self::Error> { Ok(()) }
}
