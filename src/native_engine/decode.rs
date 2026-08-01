//! Frozen greedy decode-loop types and the eager autoregressive core.
//!
//! This module owns the request/output vocabulary shared by the eventual CLI,
//! library, and robot stream.  It intentionally implements only the
//! `hf-bf16-eager` greedy path available today.  Grammar masking, seeded
//! sampling, typed asupersync budgets, and the product bounded-channel adapter
//! remain separate dependencies; callers receive a typed refusal rather than a
//! silently weaker decode mode.

use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};

use crate::tokenizer::bpe::{DecodeBytesError, SpBpeTokenizer};

use super::{
    hf_bf16_eager::{HF_BF16_EAGER_PROFILE, HfBf16EagerEngine, HfBf16EagerError},
    kv::KV_SLOT_COUNT,
    lmhead::NANBEIGE_VOCAB_SIZE,
};

/// Frozen schema version of [`DecodeParams`] and [`DecodeOutput`].
pub const DECODE_SCHEMA_VERSION: u32 = 1;

/// Schema version carried by every per-token stream event.
///
/// This matches the current robot event envelope.  The CLI owns conversion of
/// this typed event into its NDJSON line.
pub const DECODE_TOKEN_EVENT_SCHEMA_VERSION: u32 = 2;

const MAX_STOP_SEQUENCES: usize = 64;
const MAX_STOP_SEQUENCE_BYTES: usize = 4 * 1024;

/// A frozen, serializable numerics-profile selector at the decode boundary.
///
/// The eager core below accepts only [`Self::HfBf16Eager`].  Keeping the full
/// request vocabulary here prevents a later CLI/library split while preserving
/// a typed refusal until the other profile engines own their decode routes.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeNumericsProfile {
    /// The pinned HF-compatible bf16 cast schedule.
    HfBf16Eager,
    /// The structural f32 bisect profile.
    DiagnosticF32,
    /// A preregistered strict quantized recipe.
    StrictQuantized { version: u32 },
    /// A host-scoped performance profile.
    Fast { version: u32 },
}

impl DecodeNumericsProfile {
    /// Stable machine label for receipts and robot output.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::HfBf16Eager => HF_BF16_EAGER_PROFILE.to_owned(),
            Self::DiagnosticF32 => "diagnostic-f32".to_owned(),
            Self::StrictQuantized { version } => format!("strict-quantized-v{version}"),
            Self::Fast { version } => format!("fast-v{version}"),
        }
    }
}

/// Selection policy frozen into the decode request envelope.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum DecodeSamplingMode {
    /// First-index-wins greedy selection.  It requires no RANDOM authority.
    Greedy,
    /// Addressable sampling reserved for the sampler-owner implementation.
    Seeded { seed: [u8; 32] },
}

/// The request's stopping policy, evaluated after each committed token.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeStopPolicy {
    /// Tokens that end a response once `min_new_tokens` has been reached.
    pub eos_token_ids: Vec<u32>,
    /// Exact byte suffixes that end a response once the minimum is reached.
    ///
    /// Byte sequences are used rather than strings so invalid UTF-8 can never
    /// be normalized or silently replaced at this boundary.
    pub stop_sequences: Vec<Vec<u8>>,
    /// Suppress EOS and byte-stop completion until this many tokens commit.
    pub min_new_tokens: usize,
}

impl Default for DecodeStopPolicy {
    fn default() -> Self {
        Self {
            eos_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
            min_new_tokens: 0,
        }
    }
}

/// A data-only hook for the template-owned thinking contract.
///
/// This eager greedy core cannot decide template delimiter semantics by itself.
/// A supplied hook therefore refuses until the template/task owner binds it to
/// concrete control-token and no-result policy.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ThinkingBudgetHook {
    /// Maximum token budget assigned to the thinking region.
    pub max_tokens: usize,
    /// Template-owned byte delimiter that must close the thinking region.
    pub required_close_delimiter: Vec<u8>,
}

/// Frozen request shared by the library, CLI, and eventual robot adapter.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeParams {
    /// Must equal [`DECODE_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Caller-assigned robot request sequence echoed by stream events.
    pub request_seq: u64,
    /// Strict upper bound on generated tokens committed to the output.
    pub max_new_tokens: usize,
    /// Strict upper bound on decoded bytes committed to the output.
    pub max_output_bytes: usize,
    /// EOS, byte-stop, and minimum-length policy.
    pub stop: DecodeStopPolicy,
    /// Greedy by default; seeded sampling remains a typed deferred route.
    pub sampling: DecodeSamplingMode,
    /// Template-owned thinking contract, if one was requested.
    pub thinking: Option<ThinkingBudgetHook>,
    /// Capture the chosen token's full-vocabulary log-softmax score.
    pub capture_logprobs: bool,
    /// The per-request numeric behavior contract.
    pub numerics_profile: DecodeNumericsProfile,
}

impl DecodeParams {
    /// Construct the smallest valid request for this eager greedy core.
    #[must_use]
    pub fn hf_bf16_greedy(
        request_seq: u64,
        max_new_tokens: usize,
        max_output_bytes: usize,
    ) -> Self {
        Self {
            schema_version: DECODE_SCHEMA_VERSION,
            request_seq,
            max_new_tokens,
            max_output_bytes,
            stop: DecodeStopPolicy::default(),
            sampling: DecodeSamplingMode::Greedy,
            thinking: None,
            capture_logprobs: false,
            numerics_profile: DecodeNumericsProfile::HfBf16Eager,
        }
    }

    fn validate_for_eager_greedy(&self) -> Result<(), DecodeError> {
        if self.schema_version != DECODE_SCHEMA_VERSION {
            return Err(DecodeError::UnsupportedSchemaVersion {
                expected: DECODE_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        if self.stop.stop_sequences.len() > MAX_STOP_SEQUENCES {
            return Err(DecodeError::TooManyStopSequences {
                maximum: MAX_STOP_SEQUENCES,
                actual: self.stop.stop_sequences.len(),
            });
        }
        for sequence in &self.stop.stop_sequences {
            if sequence.is_empty() {
                return Err(DecodeError::EmptyStopSequence);
            }
            if sequence.len() > MAX_STOP_SEQUENCE_BYTES {
                return Err(DecodeError::StopSequenceTooLong {
                    maximum: MAX_STOP_SEQUENCE_BYTES,
                    actual: sequence.len(),
                });
            }
        }
        if self.thinking.is_some() {
            return Err(DecodeError::ThinkingBudgetUnavailable);
        }
        if self.numerics_profile != DecodeNumericsProfile::HfBf16Eager {
            return Err(DecodeError::NumericsProfileUnavailable {
                requested: self.numerics_profile.label(),
            });
        }
        if self.sampling != DecodeSamplingMode::Greedy {
            return Err(DecodeError::SamplingModeUnavailable);
        }
        Ok(())
    }
}

/// Why a completed decode stopped emitting tokens.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeFinishReason {
    /// An EOS token matched after the configured minimum length.
    Eos,
    /// A byte stop sequence matched after the configured minimum length.
    StopSequence,
    /// The declared output-token budget was exhausted.
    Budget,
    /// A per-step cancellation checkpoint fired before a new token committed.
    Cancelled,
    /// A future task/template finalizer could not form a valid result.
    NoResult,
}

/// The complete cancellation-kind domain retained by decode output.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeCancellationKind {
    /// The caller explicitly cancelled.
    User,
    /// A timeout fired.
    Timeout,
    /// A deadline expired.
    Deadline,
    /// A poll quota was exhausted.
    PollQuota,
    /// A cost budget was exhausted.
    CostBudget,
    /// A supervised sibling requested fail-fast cancellation.
    FailFast,
    /// A race loser was cancelled.
    RaceLost,
    /// A parent region cancelled.
    ParentCancelled,
    /// A checked resource was unavailable.
    ResourceUnavailable,
    /// The runtime is shutting down.
    Shutdown,
    /// A linked task exited abnormally.
    LinkedExit,
}

/// The probability-space definition carried with optional token logprobs.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeScoreSpace {
    /// Log-softmax over the full 166,144-row f32 lm-head projection.
    FullVocabularyLogSoftmax,
}

/// Frozen successful/raw decode result consumed by task finalizers.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeOutput {
    /// [`DECODE_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Echo of [`DecodeParams::request_seq`].
    pub request_seq: u64,
    /// Actual profile selected by the engine.
    pub numerics_profile: String,
    /// Generated token ids in committed order.
    pub emitted_token_ids: Vec<u32>,
    /// Exact bytes decoded from `emitted_token_ids` by the supplied tokenizer.
    pub decoded_bytes: Vec<u8>,
    /// Terminal reason.  A task finalizer owns whether this is a success.
    pub finish_reason: DecodeFinishReason,
    /// Present only when `capture_logprobs` requested a full projection score.
    pub token_logprobs: Option<Vec<f32>>,
    /// Score-space of `token_logprobs`; absent exactly when they are absent.
    pub logprob_score_space: Option<DecodeScoreSpace>,
    /// Cancellation cause when `finish_reason` is [`DecodeFinishReason::Cancelled`].
    pub cancellation: Option<DecodeCancellationKind>,
}

/// One typed, per-token stream event.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeTokenEvent {
    /// [`DECODE_TOKEN_EVENT_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Echo of [`DecodeParams::request_seq`].
    pub request_seq: u64,
    /// Zero-based index in [`DecodeOutput::emitted_token_ids`].
    pub token_index: usize,
    /// The committed generated token id.
    pub token_id: u32,
    /// Exact newly decoded bytes attributable to this token.
    pub decoded_bytes: Vec<u8>,
    /// Full-vocabulary logprob when capture was requested.
    pub logprob: Option<f32>,
}

/// Two-phase stream adapter owned by the CLI/runtime integration.
///
/// `reserve` obtains bounded output capacity before the event is visible;
/// `permit` makes exactly that event visible.  The eager core calls no sink
/// while it holds an internal engine/admission guard (it owns neither), and it
/// does not add the token to its returned output until `permit` succeeds.
/// Implementations must treat a failed `permit` as uncommitted delivery.
pub trait DecodeEventSink {
    /// The bounded-capacity permit returned by [`Self::reserve`].
    type Permit;
    /// Typed adapter refusal, for example a cancelled sender or full channel.
    type Error: fmt::Display;

    /// Reserve capacity for this exact event without making it visible.
    fn reserve(&mut self, event: &DecodeTokenEvent) -> Result<Self::Permit, Self::Error>;

    /// Commit the reserved event exactly once.
    fn permit(&mut self, permit: Self::Permit, event: DecodeTokenEvent) -> Result<(), Self::Error>;
}

/// Token-byte decoder used for exact stop matching and final byte output.
pub trait DecodeByteDecoder {
    /// Refusal from the tokenizer/decoder.
    type Error: fmt::Display;

    /// Decode the complete generated token prefix without lossy replacement.
    fn decode_token_ids(&self, token_ids: &[u32]) -> Result<Vec<u8>, Self::Error>;
}

impl DecodeByteDecoder for SpBpeTokenizer {
    type Error = DecodeBytesError;

    fn decode_token_ids(&self, token_ids: &[u32]) -> Result<Vec<u8>, Self::Error> {
        self.decode_bytes(token_ids)
    }
}

/// The cancellation checkpoint invoked before every generated-token commit.
pub trait DecodeStepControl {
    /// Return a cancellation chain to stop before the pending token commits.
    fn checkpoint(&mut self, next_token_index: usize) -> Option<DecodeCancellationKind>;
}

/// Typed refusals from the eager greedy loop.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecodeError {
    /// A blank prompt has no final prefill row from which to select a token.
    EmptyPrompt,
    /// A request schema revision is not understood by this frozen loop.
    UnsupportedSchemaVersion { expected: u32, actual: u32 },
    /// Stop matching refuses ambiguous immediate completion.
    EmptyStopSequence,
    /// Stop matching has a fixed per-request sequence count cap.
    TooManyStopSequences { maximum: usize, actual: usize },
    /// Stop matching has a fixed per-sequence byte cap.
    StopSequenceTooLong { maximum: usize, actual: usize },
    /// The template-owned thinking close contract has no eager-loop binding.
    ThinkingBudgetUnavailable,
    /// This engine does not implement the requested profile.
    NumericsProfileUnavailable { requested: String },
    /// Addressable seeded sampling has no implementation at this seam yet.
    SamplingModeUnavailable,
    /// A caller attempted to reuse an engine with retained prefix/KV state.
    EngineAlreadyPrimed {
        slot: usize,
        cached_positions: usize,
    },
    /// Prompt plus required feedback positions exceed the admitted K/V cache.
    ContextBudgetExceeded {
        capacity_positions: usize,
        prompt_tokens: usize,
        requested_new_tokens: usize,
        required_positions: usize,
    },
    /// Arithmetic could not express a requested position count.
    ContextBudgetOverflow,
    /// The eager engine rejected a model, K/V, or token invariant.
    Engine(HfBf16EagerError),
    /// The eager lm-head returned an impossible token id.
    GreedyTokenOutOfRange { token_id: usize, vocab_size: usize },
    /// The supplied decoder rejected a token prefix.
    Decoder { detail: String },
    /// Decoding a longer token prefix rewrote previously emitted bytes.
    DecoderNotPrefixStable {
        previous_len: usize,
        decoded_len: usize,
    },
    /// A requested full-vocabulary score could not be formed from logits.
    LogprobUnavailable { detail: String },
    /// A two-phase output adapter refused reservation or delivery.
    Stream { detail: String },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPrompt => formatter.write_str("decode prompt must contain at least one token"),
            Self::UnsupportedSchemaVersion { expected, actual } => write!(
                formatter,
                "decode schema version {actual} is unsupported; expected {expected}"
            ),
            Self::EmptyStopSequence => formatter.write_str("decode stop sequences must not be empty"),
            Self::TooManyStopSequences { maximum, actual } => write!(
                formatter,
                "decode has {actual} stop sequences; maximum is {maximum}"
            ),
            Self::StopSequenceTooLong { maximum, actual } => write!(
                formatter,
                "decode stop sequence has {actual} bytes; maximum is {maximum}"
            ),
            Self::ThinkingBudgetUnavailable => formatter.write_str(
                "thinking budget requires the template-owned delimiter binding, unavailable in eager decode",
            ),
            Self::NumericsProfileUnavailable { requested } => {
                write!(formatter, "decode numerics profile {requested} is unavailable in eager decode")
            }
            Self::SamplingModeUnavailable => formatter.write_str(
                "seeded sampling requires the addressable sampler, unavailable in eager greedy decode",
            ),
            Self::EngineAlreadyPrimed {
                slot,
                cached_positions,
            } => write!(
                formatter,
                "decode engine already contains {cached_positions} positions in K/V slot {slot}"
            ),
            Self::ContextBudgetExceeded {
                capacity_positions,
                prompt_tokens,
                requested_new_tokens,
                required_positions,
            } => write!(
                formatter,
                "decode needs {required_positions} K/V positions for prompt={prompt_tokens} new={requested_new_tokens}; admitted capacity is {capacity_positions}"
            ),
            Self::ContextBudgetOverflow => formatter.write_str("decode context-budget arithmetic overflowed"),
            Self::Engine(error) => write!(formatter, "eager decode rejected: {error:?}"),
            Self::GreedyTokenOutOfRange {
                token_id,
                vocab_size,
            } => write!(
                formatter,
                "eager greedy token {token_id} is outside vocabulary {vocab_size}"
            ),
            Self::Decoder { detail } => write!(formatter, "decode byte decoder rejected token prefix: {detail}"),
            Self::DecoderNotPrefixStable {
                previous_len,
                decoded_len,
            } => write!(
                formatter,
                "decode byte decoder rewrote prior output: previous={previous_len} new={decoded_len}"
            ),
            Self::LogprobUnavailable { detail } => {
                write!(formatter, "decode full-vocabulary logprob unavailable: {detail}")
            }
            Self::Stream { detail } => write!(formatter, "decode stream delivery refused: {detail}"),
        }
    }
}

impl Error for DecodeError {}

impl From<HfBf16EagerError> for DecodeError {
    fn from(value: HfBf16EagerError) -> Self {
        Self::Engine(value)
    }
}

/// Run a non-streaming, greedy eager decode with no cancellation source.
pub fn greedy_decode<D: DecodeByteDecoder>(
    engine: &mut HfBf16EagerEngine,
    prompt_token_ids: &[u32],
    params: &DecodeParams,
    decoder: &D,
) -> Result<DecodeOutput, DecodeError> {
    let mut sink = DiscardDecodeEventSink;
    let mut control = NeverCancelled;
    greedy_decode_with_hooks(
        engine,
        prompt_token_ids,
        params,
        decoder,
        &mut sink,
        &mut control,
    )
}

/// Run a greedy eager decode through a caller-owned two-phase stream adapter.
pub fn greedy_decode_with_sink<D: DecodeByteDecoder, S: DecodeEventSink>(
    engine: &mut HfBf16EagerEngine,
    prompt_token_ids: &[u32],
    params: &DecodeParams,
    decoder: &D,
    sink: &mut S,
) -> Result<DecodeOutput, DecodeError> {
    let mut control = NeverCancelled;
    greedy_decode_with_hooks(
        engine,
        prompt_token_ids,
        params,
        decoder,
        sink,
        &mut control,
    )
}

/// Run a greedy eager decode with explicit streaming and cancellation hooks.
///
/// The Hf eager engine appends all 44 K/V slots for every prefilling and
/// feedback position.  The last prefill forward chooses the first generated
/// token; each subsequent generated token is fed back through `decode` before
/// selecting its successor.  No extra final feedback forward is performed,
/// because it cannot affect the already-complete response.
pub fn greedy_decode_with_hooks<D: DecodeByteDecoder, S: DecodeEventSink, C: DecodeStepControl>(
    engine: &mut HfBf16EagerEngine,
    prompt_token_ids: &[u32],
    params: &DecodeParams,
    decoder: &D,
    sink: &mut S,
    control: &mut C,
) -> Result<DecodeOutput, DecodeError> {
    params.validate_for_eager_greedy()?;
    if prompt_token_ids.is_empty() {
        return Err(DecodeError::EmptyPrompt);
    }
    refuse_primed_engine(engine)?;
    validate_context_budget(
        engine.kv_cache().capacity_positions(),
        prompt_token_ids.len(),
        params.max_new_tokens,
    )?;

    let mut output = empty_output(params, engine.profile());
    if params.max_new_tokens == 0 {
        return Ok(output);
    }

    let prefill = engine.prefill(prompt_token_ids)?;
    let mut selected = checked_greedy_token(
        prefill
            .last()
            .expect("nonempty prompt produces one eager forward per token")
            .greedy_token,
    )?;
    let mut selected_logprob = if params.capture_logprobs {
        Some(full_vocabulary_logprob(
            &prefill
                .last()
                .expect("nonempty prompt produces one eager forward per token")
                .logits,
            selected,
        )?)
    } else {
        None
    };

    loop {
        let token_index = output.emitted_token_ids.len();
        if let Some(cancellation) = control.checkpoint(token_index) {
            output.finish_reason = DecodeFinishReason::Cancelled;
            output.cancellation = Some(cancellation);
            return Ok(output);
        }

        let mut candidate_token_ids = output.emitted_token_ids.clone();
        candidate_token_ids.push(selected);
        let decoded = decoder
            .decode_token_ids(&candidate_token_ids)
            .map_err(|error| DecodeError::Decoder {
                detail: error.to_string(),
            })?;
        if !decoded.starts_with(&output.decoded_bytes) {
            return Err(DecodeError::DecoderNotPrefixStable {
                previous_len: output.decoded_bytes.len(),
                decoded_len: decoded.len(),
            });
        }
        if decoded.len() > params.max_output_bytes {
            output.finish_reason = DecodeFinishReason::Budget;
            return Ok(output);
        }
        let token_bytes = decoded[output.decoded_bytes.len()..].to_vec();
        let event = DecodeTokenEvent {
            schema_version: DECODE_TOKEN_EVENT_SCHEMA_VERSION,
            request_seq: params.request_seq,
            token_index,
            token_id: selected,
            decoded_bytes: token_bytes,
            logprob: selected_logprob,
        };
        deliver_stream_event(sink, event)?;

        output.emitted_token_ids = candidate_token_ids;
        output.decoded_bytes = decoded;
        if let Some(logprobs) = &mut output.token_logprobs {
            logprobs.push(selected_logprob.expect("capture_logprobs selected this logprob"));
        }

        if let Some(finish_reason) = finish_reason_after_commit(&output, params, selected) {
            output.finish_reason = finish_reason;
            return Ok(output);
        }

        let forward = engine.decode(selected)?;
        selected = checked_greedy_token(forward.greedy_token)?;
        selected_logprob = if params.capture_logprobs {
            Some(full_vocabulary_logprob(&forward.logits, selected)?)
        } else {
            None
        };
    }
}

fn empty_output(params: &DecodeParams, profile: &str) -> DecodeOutput {
    DecodeOutput {
        schema_version: DECODE_SCHEMA_VERSION,
        request_seq: params.request_seq,
        numerics_profile: profile.to_owned(),
        emitted_token_ids: Vec::with_capacity(params.max_new_tokens),
        decoded_bytes: Vec::new(),
        finish_reason: DecodeFinishReason::Budget,
        token_logprobs: params.capture_logprobs.then(Vec::new),
        logprob_score_space: params
            .capture_logprobs
            .then_some(DecodeScoreSpace::FullVocabularyLogSoftmax),
        cancellation: None,
    }
}

fn refuse_primed_engine(engine: &HfBf16EagerEngine) -> Result<(), DecodeError> {
    for slot in 0..KV_SLOT_COUNT {
        let cached_positions = engine
            .kv_cache()
            .len_for_slot(slot)
            .map_err(HfBf16EagerError::from)?;
        if cached_positions != 0 {
            return Err(DecodeError::EngineAlreadyPrimed {
                slot,
                cached_positions,
            });
        }
    }
    Ok(())
}

fn validate_context_budget(
    capacity_positions: usize,
    prompt_tokens: usize,
    requested_new_tokens: usize,
) -> Result<(), DecodeError> {
    // The first emitted token comes from the final prompt logits and needs no
    // K/V append.  Only later generated tokens are fed back for a successor.
    let feedback_positions = requested_new_tokens.saturating_sub(1);
    let required_positions = prompt_tokens
        .checked_add(feedback_positions)
        .ok_or(DecodeError::ContextBudgetOverflow)?;
    if required_positions > capacity_positions {
        return Err(DecodeError::ContextBudgetExceeded {
            capacity_positions,
            prompt_tokens,
            requested_new_tokens,
            required_positions,
        });
    }
    Ok(())
}

fn checked_greedy_token(token_id: usize) -> Result<u32, DecodeError> {
    if token_id >= NANBEIGE_VOCAB_SIZE {
        return Err(DecodeError::GreedyTokenOutOfRange {
            token_id,
            vocab_size: NANBEIGE_VOCAB_SIZE,
        });
    }
    u32::try_from(token_id).map_err(|_| DecodeError::GreedyTokenOutOfRange {
        token_id,
        vocab_size: NANBEIGE_VOCAB_SIZE,
    })
}

fn finish_reason_after_commit(
    output: &DecodeOutput,
    params: &DecodeParams,
    selected: u32,
) -> Option<DecodeFinishReason> {
    if output.emitted_token_ids.len() >= params.stop.min_new_tokens {
        if params.stop.eos_token_ids.contains(&selected) {
            return Some(DecodeFinishReason::Eos);
        }
        if params
            .stop
            .stop_sequences
            .iter()
            .any(|sequence| output.decoded_bytes.ends_with(sequence))
        {
            return Some(DecodeFinishReason::StopSequence);
        }
    }
    if output.emitted_token_ids.len() == params.max_new_tokens {
        return Some(DecodeFinishReason::Budget);
    }
    None
}

fn full_vocabulary_logprob(logits: &[f32], selected: u32) -> Result<f32, DecodeError> {
    let selected_index =
        usize::try_from(selected).map_err(|_| DecodeError::LogprobUnavailable {
            detail: "selected token does not fit usize".to_owned(),
        })?;
    let selected_logit =
        *logits
            .get(selected_index)
            .ok_or_else(|| DecodeError::LogprobUnavailable {
                detail: format!("selected token {selected} has no logit"),
            })?;
    if !selected_logit.is_finite() {
        return Err(DecodeError::LogprobUnavailable {
            detail: format!("selected token {selected} has non-finite logit"),
        });
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !maximum.is_finite() {
        return Err(DecodeError::LogprobUnavailable {
            detail: "full vocabulary has no finite maximum logit".to_owned(),
        });
    }
    let exp_sum = logits
        .iter()
        .copied()
        .map(|logit| f64::from(logit - maximum).exp())
        .sum::<f64>();
    if !exp_sum.is_finite() || exp_sum <= 0.0 {
        return Err(DecodeError::LogprobUnavailable {
            detail: "full vocabulary log-sum-exp is non-finite".to_owned(),
        });
    }
    Ok((f64::from(selected_logit - maximum) - exp_sum.ln()) as f32)
}

fn deliver_stream_event<S: DecodeEventSink>(
    sink: &mut S,
    event: DecodeTokenEvent,
) -> Result<(), DecodeError> {
    let permit = sink.reserve(&event).map_err(|error| DecodeError::Stream {
        detail: error.to_string(),
    })?;
    sink.permit(permit, event)
        .map_err(|error| DecodeError::Stream {
            detail: error.to_string(),
        })
}

struct DiscardDecodeEventSink;

impl DecodeEventSink for DiscardDecodeEventSink {
    type Permit = ();
    type Error = std::convert::Infallible;

    fn reserve(&mut self, _event: &DecodeTokenEvent) -> Result<Self::Permit, Self::Error> {
        Ok(())
    }

    fn permit(
        &mut self,
        _permit: Self::Permit,
        _event: DecodeTokenEvent,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

struct NeverCancelled;

impl DecodeStepControl for NeverCancelled {
    fn checkpoint(&mut self, _next_token_index: usize) -> Option<DecodeCancellationKind> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::fmt;

    use crate::native_engine::{hf_bf16_eager::HfBf16EagerError, kv::KvCacheError};

    use super::{
        DECODE_TOKEN_EVENT_SCHEMA_VERSION, DecodeError, DecodeEventSink, DecodeFinishReason,
        DecodeParams, DecodeStopPolicy, DecodeTokenEvent, deliver_stream_event, empty_output,
        finish_reason_after_commit, full_vocabulary_logprob, validate_context_budget,
    };
    use serde_json::json;

    #[test]
    fn budget_allows_the_first_token_without_an_extra_kv_append() {
        assert_eq!(validate_context_budget(3, 3, 1), Ok(()));
        assert_eq!(
            validate_context_budget(3, 3, 2),
            Err(DecodeError::ContextBudgetExceeded {
                capacity_positions: 3,
                prompt_tokens: 3,
                requested_new_tokens: 2,
                required_positions: 4,
            })
        );
    }

    #[test]
    fn context_budget_refuses_position_arithmetic_overflow() {
        assert_eq!(
            validate_context_budget(usize::MAX, usize::MAX, 2),
            Err(DecodeError::ContextBudgetOverflow)
        );
    }

    #[test]
    fn cache_introspection_refusal_stays_a_typed_engine_error() {
        let error: DecodeError =
            HfBf16EagerError::from(KvCacheError::InvalidSlot { slot: 44 }).into();
        assert_eq!(
            error,
            DecodeError::Engine(HfBf16EagerError::Kv(KvCacheError::InvalidSlot { slot: 44 }))
        );
    }

    #[test]
    fn byte_stop_sequences_match_only_after_minimum_length() {
        let mut params = DecodeParams::hf_bf16_greedy(7, 2, 32);
        params.stop = DecodeStopPolicy {
            eos_token_ids: vec![9],
            stop_sequences: vec![vec![0xe2, 0x80, 0xa6]],
            min_new_tokens: 2,
        };
        let mut output = empty_output(&params, "hf-bf16-eager");
        output.emitted_token_ids.push(9);
        output.decoded_bytes = vec![0xe2, 0x80];
        assert_eq!(finish_reason_after_commit(&output, &params, 9), None);

        output.emitted_token_ids.push(10);
        output.decoded_bytes.push(0xa6);
        assert_eq!(
            finish_reason_after_commit(&output, &params, 10),
            Some(DecodeFinishReason::StopSequence)
        );
    }

    #[test]
    fn eos_has_priority_over_the_same_step_output_budget() {
        let mut params = DecodeParams::hf_bf16_greedy(7, 1, 32);
        params.stop.eos_token_ids.push(9);
        let mut output = empty_output(&params, "hf-bf16-eager");
        output.emitted_token_ids.push(9);
        assert_eq!(
            finish_reason_after_commit(&output, &params, 9),
            Some(DecodeFinishReason::Eos)
        );
    }

    #[test]
    fn output_budget_still_terminates_when_the_minimum_is_larger() {
        let mut params = DecodeParams::hf_bf16_greedy(7, 1, 32);
        params.stop.min_new_tokens = 2;
        let mut output = empty_output(&params, "hf-bf16-eager");
        output.emitted_token_ids.push(41);
        assert_eq!(
            finish_reason_after_commit(&output, &params, 41),
            Some(DecodeFinishReason::Budget)
        );
    }

    #[test]
    fn empty_stop_sequence_is_refused_before_engine_use() {
        let mut params = DecodeParams::hf_bf16_greedy(0, 1, 32);
        params.stop.stop_sequences.push(Vec::new());
        assert_eq!(
            params.validate_for_eager_greedy(),
            Err(DecodeError::EmptyStopSequence)
        );
    }

    #[test]
    fn frozen_request_and_output_wire_shapes_remain_reviewable() {
        let params = DecodeParams::hf_bf16_greedy(41, 2, 64);
        assert_eq!(
            serde_json::to_value(&params).expect("serialize frozen decode params"),
            json!({
                "schema_version": 1,
                "request_seq": 41,
                "max_new_tokens": 2,
                "max_output_bytes": 64,
                "stop": {
                    "eos_token_ids": [],
                    "stop_sequences": [],
                    "min_new_tokens": 0,
                },
                "sampling": {"kind": "greedy"},
                "thinking": null,
                "capture_logprobs": false,
                "numerics_profile": "hf_bf16_eager",
            })
        );
        assert_eq!(
            serde_json::to_value(empty_output(&params, "hf-bf16-eager"))
                .expect("serialize frozen decode output"),
            json!({
                "schema_version": 1,
                "request_seq": 41,
                "numerics_profile": "hf-bf16-eager",
                "emitted_token_ids": [],
                "decoded_bytes": [],
                "finish_reason": "budget",
                "token_logprobs": null,
                "logprob_score_space": null,
                "cancellation": null,
            })
        );
    }

    #[test]
    fn full_vocabulary_logprob_is_named_and_finite() {
        let observed = full_vocabulary_logprob(&[0.0, 0.0], 0)
            .expect("finite logits must have a full-vocabulary logprob");
        assert!((observed + std::f32::consts::LN_2).abs() < 1.0e-6);
    }

    #[derive(Debug)]
    struct TestStreamError;

    impl fmt::Display for TestStreamError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("test stream unavailable")
        }
    }

    struct RefusingReservationSink {
        permit_calls: usize,
    }

    impl DecodeEventSink for RefusingReservationSink {
        type Permit = ();
        type Error = TestStreamError;

        fn reserve(&mut self, _event: &DecodeTokenEvent) -> Result<Self::Permit, Self::Error> {
            Err(TestStreamError)
        }

        fn permit(
            &mut self,
            _permit: Self::Permit,
            _event: DecodeTokenEvent,
        ) -> Result<(), Self::Error> {
            self.permit_calls += 1;
            Ok(())
        }
    }

    #[test]
    fn reservation_refusal_never_attempts_stream_delivery() {
        let mut sink = RefusingReservationSink { permit_calls: 0 };
        let event = DecodeTokenEvent {
            schema_version: DECODE_TOKEN_EVENT_SCHEMA_VERSION,
            request_seq: 7,
            token_index: 0,
            token_id: 1,
            decoded_bytes: vec![b'x'],
            logprob: None,
        };
        assert_eq!(
            deliver_stream_event(&mut sink, event),
            Err(DecodeError::Stream {
                detail: "test stream unavailable".to_owned(),
            })
        );
        assert_eq!(sink.permit_calls, 0);
    }
}
