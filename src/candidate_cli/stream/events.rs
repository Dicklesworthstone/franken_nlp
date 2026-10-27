//! Bounded canonical token framing. No stdout write occurs in reserve.
//! A complete native task result must reconcile every delivered byte/token/score
//! before a terminal success frame is even staged. This sink owns no inference.
use super::*;
use std::sync::Arc;
use crate::{execution_identity::Sha256Digest,
    native_engine::{artifact_bridge::ArtifactIdentity,
        decode::{DecodeEventSink, DecodeTokenEvent, DecodeScoreSpace, DECODE_TOKEN_EVENT_SCHEMA_VERSION},
        generation::{GenerationFinish, quantized::INT8_GENERATION_VERSION},
        lmhead::NANBEIGE_VOCAB_SIZE, sampler::Seed256, strict_int8::STRICT_INT8_PROFILE},
    tasks::chat::quantized::Int8ChatResult};

#[derive(Serialize)]
struct Provenance {
    model_id: String, source_revision: String, source_root_sha256: String,
    logical_model_sha256: String, quant_recipe: String,
}
#[derive(Serialize)]
struct Frame<'a, T: Serialize> {
    protocol: &'static str, schema_version: u32, event: &'static str,
    scope: &'static str, evidence: &'static str, request_seq: u64,
    task: &'static str, provisional: bool,
    #[serde(flatten)] provenance: &'a Provenance,
    data: &'a T,
}
#[derive(Serialize)]
struct Start {
    token_event_schema_version: u32, byte_encoding: &'static str,
    completion_required: bool,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum State { Fresh, Open, Reserved, Failed, Finished }

/// Unforgeable outside this module and non-Clone. A permit contains the exact
/// reserved event and staged wire bytes, not an unchecked capacity count.
/// Its owner marker prevents interchange between two otherwise identical sinks.
pub(in crate::candidate_cli) struct TokenPermit {
    owner: Arc<()>, expected: DecodeTokenEvent, start: Vec<u8>, token: Vec<u8>,
    next_total: u64,
}
pub(in crate::candidate_cli) struct TokenSink<W> {
    writer: W, provenance: Provenance, task: &'static str, bounds: StreamBounds,
    owner: Arc<()>, state: State, emitted: u64,
    token_ids: Vec<u32>, bytes: Vec<u8>, scores: Option<Vec<f32>>,
    eos: u32, saw_eos: bool, saw_stop: bool, expected_seed: Option<String>,
    min_new_tokens: usize, stop_suffixes: Vec<Vec<u8>>,
}
fn reserved<T>(count: usize) -> Result<Vec<T>, Failure> {
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| Failure::output("stream_allocation"))?;
    Ok(values)
}
impl<W: Write> TokenSink<W> {
    pub(in crate::candidate_cli) fn new(writer: W, task: Task, bounds: StreamBounds,
        facts: &ArtifactIdentity, options: &GenerationOptions) -> Result<Self, Failure> {
        options.validate(GenerationLimits { max_prompt_tokens: 8192, max_new_tokens: bounds.tokens,
            max_output_bytes: bounds.content_bytes, max_sampler_bytes: SAMPLER_BYTES })
            .map_err(|_| Failure::usage("stream_generation_policy"))?;
        let [eos] = options.eos_token_ids.as_slice() else {
            return Err(Failure::usage("stream_eos_contract"));
        };
        let seed = match &options.sampling {
            GenerationSampling::Greedy => None,
            GenerationSampling::Seeded { effective_seed, .. } => Some(Seed256::from(*effective_seed).to_lower_hex()),
        };
        // Runtime also checks these actual artifact facts before loading. The
        // sink independently bounds its repeated public provenance fields.
        if facts.model_id != "Nanbeige4.2-3B"
            || facts.revision != "f56ec5a9650268aa098496734743c25ea778bd2d"
            || facts.recipe_id.is_empty() || facts.recipe_id.len() > 256
            || Sha256Digest::from_hex(&facts.source_root_sha256).is_err()
            || Sha256Digest::from_hex(&facts.logical_model_sha256).is_err()
            || *eos as usize >= NANBEIGE_VOCAB_SIZE
            || options.max_new_tokens != bounds.tokens || options.max_output_bytes != bounds.content_bytes
            || options.capture_logprobs != bounds.capture_logprobs
            || bounds.tokens == 0 || bounds.tokens > 1024
            || bounds.content_bytes == 0 || bounds.content_bytes > MAX_CONTENT_BYTES
            || bounds.event_bytes < FRAME_BYTES || bounds.terminal_bytes < FRAME_BYTES
            || bounds.stream_bytes < bounds.terminal_bytes as u64 + FRAME_BYTES as u64 {
            return Err(Failure::usage("stream_contract"));
        }
        Ok(Self { writer, provenance: Provenance { model_id: facts.model_id.clone(),
            source_revision: facts.revision.clone(), source_root_sha256: facts.source_root_sha256.clone(),
            logical_model_sha256: facts.logical_model_sha256.clone(), quant_recipe: facts.recipe_id.clone() },
            task: match task { Task::Generate => "generate-v1", Task::Chat => "chat-v1" }, bounds,
            owner: Arc::new(()), state: State::Fresh, emitted: 0,
            token_ids: reserved(bounds.tokens)?, bytes: reserved(bounds.content_bytes)?,
            scores: if bounds.capture_logprobs { Some(reserved(bounds.tokens)?) } else { None },
            eos: *eos, saw_eos: false, saw_stop: false, expected_seed: seed,
            min_new_tokens: options.min_new_tokens, stop_suffixes: options.stop_suffixes.clone() })
    }
    fn frame<T: Serialize>(&self, event: &'static str, provisional: bool, data: &T,
        cap: usize) -> Result<Vec<u8>, Failure> {
        encode(&Frame { protocol: PROTOCOL, schema_version: 1, event,
            scope: "real-artifact-current-candidate", evidence: "non_authoritative", request_seq: 1,
            task: self.task, provisional, provenance: &self.provenance, data }, cap)
    }
    fn start(&self) -> Result<Vec<u8>, Failure> {
        self.frame("run_start", true, &Start { token_event_schema_version: DECODE_TOKEN_EVENT_SCHEMA_VERSION,
            byte_encoding: "u8-array", completion_required: true }, FRAME_BYTES)
    }
    fn validate_event(&self, event: &DecodeTokenEvent) -> Result<(), Failure> {
        let score_ok = match (self.bounds.capture_logprobs, event.logprob) {
            (false, None) => true, (true, Some(s)) => s.is_finite() && s <= 0.0, _ => false,
        };
        let bytes = self.bytes.len().checked_add(event.decoded_bytes.len())
            .ok_or_else(|| Failure::output("stream_content_limit"))?;
        if event.schema_version != DECODE_TOKEN_EVENT_SCHEMA_VERSION || event.request_seq != 1
            || event.token_index != self.token_ids.len() || event.token_id as usize >= NANBEIGE_VOCAB_SIZE
            || self.token_ids.len() >= self.bounds.tokens || bytes > self.bounds.content_bytes
            || !score_ok || self.saw_eos || self.saw_stop
            || (event.token_id == self.eos && (!event.decoded_bytes.is_empty()
                || self.token_ids.len() < self.min_new_tokens)) {
            return Err(Failure::output("stream_token_contract"));
        }
        Ok(())
    }
    fn total(&self, first: usize, second: usize, terminal: bool) -> Result<u64, Failure> {
        let next = self.emitted.checked_add(first as u64).and_then(|n| n.checked_add(second as u64))
            .ok_or_else(|| Failure::output("stream_transport_limit"))?;
        let cap = if terminal { self.bounds.stream_bytes }
            else { self.bounds.stream_bytes.checked_sub(self.bounds.terminal_bytes as u64)
                .ok_or_else(|| Failure::output("stream_transport_limit"))? };
        if next > cap { return Err(Failure::output("stream_transport_limit")); }
        Ok(next)
    }
    fn write_frames(&mut self, first: &[u8], second: &[u8]) -> Result<(), Failure> {
        // Caller sets Failed BEFORE entering any writer callback. Never retry
        // even a flush-only error: the peer may have received an unknown prefix.
        if !first.is_empty() { self.writer.write_all(first).map_err(|_| Failure::output("stream_output_io"))?; }
        self.writer.write_all(second).and_then(|()| self.writer.flush())
            .map_err(|_| Failure::output("stream_output_io"))
    }
    /// Only the runtime's guarded, post-drain completion calls this in product
    /// code. Unit fixtures exercise framing without manufacturing model proof.
    pub(in crate::candidate_cli) fn finish(&mut self, result: &Int8ChatResult) -> Result<(), Failure> {
        let old = self.state; self.state = State::Failed;
        if !matches!(old, State::Fresh | State::Open) { return Err(Failure::output("stream_state")); }
        self.reconcile(result)?;
        let start = if old == State::Fresh { self.start()? } else { Vec::new() };
        let terminal = self.frame("run_complete", false, result, self.bounds.terminal_bytes)?;
        let total = self.total(start.len(), terminal.len(), true)?;
        self.write_frames(&start, &terminal)?;
        self.emitted = total; self.state = State::Finished; Ok(())
    }
    fn reconcile(&self, result: &Int8ChatResult) -> Result<(), Failure> {
        let r = &result.result;
        let scores_match = match (&self.scores, &r.token_logprobs, r.logprob_score_space) {
            (None, None, None) => true,
            (Some(a), Some(b), Some(DecodeScoreSpace::FullVocabularyLogSoftmax)) =>
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
            _ => false,
        };
        let termination = match r.finish_reason {
            GenerationFinish::Eos => self.saw_eos,
            GenerationFinish::StopSuffix => !self.saw_eos && self.saw_stop,
            GenerationFinish::TokenLimit => !self.saw_eos && !self.saw_stop && self.token_ids.len() == self.bounds.tokens,
            GenerationFinish::ByteLimit => !self.saw_eos && !self.saw_stop && self.token_ids.len() < self.bounds.tokens,
        };
        if result.schema_version != 1 || r.schema_version != 1 || r.task != self.task
            || r.execution != INT8_GENERATION_VERSION || r.numerics_profile != STRICT_INT8_PROFILE
            || r.request_seq != 1 || r.sample_index != 0 || r.effective_seed != self.expected_seed
            || r.token_ids != self.token_ids || r.content.as_bytes() != self.bytes
            || !scores_match || !termination {
            return Err(Failure::output("stream_completion_mismatch"));
        }
        Ok(())
    }
}
impl<W: Write> DecodeEventSink for TokenSink<W> {
    type Permit = TokenPermit;
    type Error = Failure;
    fn reserve(&mut self, event: &DecodeTokenEvent) -> Result<TokenPermit, Failure> {
        let old = self.state; self.state = State::Failed;
        if !matches!(old, State::Fresh | State::Open) { return Err(Failure::output("stream_state")); }
        self.validate_event(event)?;
        let start = if old == State::Fresh { self.start()? } else { Vec::new() };
        let token = self.frame("token", true, event, self.bounds.event_bytes)?;
        let next_total = self.total(start.len(), token.len(), false)?;
        // No writes: cancellation between reserve and permit drops these bytes.
        // The sink stays Reserved and cannot later fabricate a completion.
        let mut bytes = reserved(event.decoded_bytes.len())?; bytes.extend_from_slice(&event.decoded_bytes);
        let expected = DecodeTokenEvent { decoded_bytes: bytes, ..event.clone() };
        let permit = TokenPermit { owner: Arc::clone(&self.owner), expected, start, token, next_total };
        self.state = State::Reserved; Ok(permit)
    }
    fn permit(&mut self, permit: TokenPermit, event: DecodeTokenEvent) -> Result<(), Failure> {
        let old = self.state; self.state = State::Failed;
        if old != State::Reserved || !Arc::ptr_eq(&permit.owner, &self.owner)
            || event != permit.expected || event.logprob.map(f32::to_bits) != permit.expected.logprob.map(f32::to_bits) {
            return Err(Failure::output("stream_permit_mismatch"));
        }
        self.validate_event(&event)?;
        self.write_frames(&permit.start, &permit.token)?;
        // All these buffers were reserved to their finite full-run capacities
        // in new(), and validate_event checked every append BEFORE delivery.
        self.bytes.extend_from_slice(&event.decoded_bytes); self.token_ids.push(event.token_id);
        if let (Some(scores), Some(score)) = (&mut self.scores, event.logprob) { scores.push(score); }
        self.saw_eos = event.token_id == self.eos;
        // Match the native cursor's priority: EOS, then a qualifying suffix,
        // then token capacity. Stops retain all delivered bytes, even when a
        // UTF-8 suffix spans several token events. Before the minimum they do
        // not terminate, and a hard byte/token budget may still end the run.
        self.saw_stop = !self.saw_eos && self.token_ids.len() >= self.min_new_tokens
            && self.stop_suffixes.iter().any(|suffix| self.bytes.ends_with(suffix));
        self.emitted = permit.next_total; self.state = State::Open; Ok(())
    }
}

fn encode<T: Serialize>(value: &T, cap: usize) -> Result<Vec<u8>, Failure> {
    let mut count = Counter { remaining: cap.checked_sub(1).ok_or_else(|| Failure::output("stream_frame_limit"))? };
    serde_json::to_writer(&mut count, value).map_err(|_| Failure::output("stream_frame_limit"))?;
    // Canonicalize the original typed value, not the sizing pass's JSON: this
    // preserves strict nonfinite-number rejection and deterministic key order.
    let mut bytes = canonjson::canonical_bytes(value).map_err(|_| Failure::output("stream_serialization"))?;
    if bytes.len() >= cap { return Err(Failure::output("stream_frame_limit")); }
    bytes.try_reserve_exact(1).map_err(|_| Failure::output("stream_allocation"))?;
    bytes.push(b'\n'); Ok(bytes)
}
struct Counter { remaining: usize }
impl Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.remaining = self.remaining.checked_sub(bytes.len()).ok_or_else(|| io::Error::other("stream bound"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

#[cfg(test)] mod tests;
