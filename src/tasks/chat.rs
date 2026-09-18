//! Pinned, bounded generation and multi-turn chat planning.
//!
//! The template renderer sees only authored placeholders and typed roles.
//! Caller message bytes are separately encoded without privileged controls.
//! History is cold-prefilled in full; no silent truncation, cross-request KV
//! reuse, tool execution or thinking-mode downgrade is performed.

use std::{error::Error, fmt, sync::Arc};
use serde::{Deserialize, Serialize};
use crate::{
    canonjson, execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{decode::{DecodeEventSink, DecodeScoreSpace, DecodeStepControl},
        generation::{GeneratedSequence, GenerationBudget, GenerationError, GenerationFinish, GenerationLimits,
            GenerationOptions, GenerationPlan, GenerationSampling, GenerationWork, GENERATION_VERSION,
            batched::BATCH_GENERATION_VERSION},
        hf_bf16_eager::{HfBf16EagerEngine, HF_BF16_EAGER_PROFILE}, lmhead::NANBEIGE_VOCAB_SIZE,
        sampler::Seed256},
    template::{Conversation, Message, MessageRole, RenderOptions, TemplateBuilder, ToolFormat, IM_START, IM_END, THINK_START, THINK_END},
    tokenizer::{bpe::EncodeOptions, embedded::{EmbeddedTokenizer, PINNED_TOKENIZER_MODEL_BYTES,
        PINNED_ADDED_TOKENS_BYTES, PINNED_TOKENIZER_CONFIG_BYTES, PINNED_SPECIAL_TOKENS_MAP_BYTES}, specials::TemplateControlIds},
};
use super::{BuiltInTask, extract::SourceDocumentEncoder,
    ir::{DecodeBudget, DecodeStrategy, DependencyScope, FinitePostcondition, GrammarReference, PlanContext,
        PromptSegment, PromptSegmentKind, TaskBudget, TaskIR, TaskPlan, TokenSequence}};
pub mod batched;
mod bounds;

pub const CHAT_PROMPT_VERSION: &str = "pinned-segmented-chat-no-thinking-no-tools-v1";
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole { System, User, Assistant }
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage { pub role: ChatRole, pub content: String }
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    pub item_id: String,
    pub sample_index: u64,
    pub messages: Vec<ChatMessage>,
    pub generation: GenerationOptions,
    pub budget: TaskBudget,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GenerateRequest {
    pub item_id: String,
    pub sample_index: u64,
    pub prompt: String,
    pub generation: GenerationOptions,
    pub budget: TaskBudget,
}
#[derive(Clone, Copy, Debug)]
pub struct ChatLimits {
    pub max_messages: usize,
    pub max_message_bytes: usize,
    pub max_total_message_bytes: usize,
    pub generation: GenerationLimits,
}
impl Default for ChatLimits {
    fn default() -> Self { Self { max_messages: 128, max_message_bytes: 64 * 1024,
        max_total_message_bytes: 512 * 1024, generation: GenerationLimits::default() } }
}
#[derive(Debug)]
pub enum ChatError { Contract(&'static str), Limit(&'static str), Identity, Encoding, Allocation,
    Native(GenerationError), NoResult(&'static str), Serialization }
impl fmt::Display for ChatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(axis) => write!(f, "chat contract refused: {axis}"),
            Self::Limit(axis) => write!(f, "chat budget exceeded: {axis}"),
            Self::Identity => f.write_str("chat execution identity refused"),
            Self::Encoding => f.write_str("chat exact token encoding failed"),
            Self::Allocation => f.write_str("chat allocation refused"),
            Self::Native(error) => write!(f, "chat native execution failed: {error}"),
            Self::NoResult(axis) => write!(f, "chat has no valid result: {axis}"),
            Self::Serialization => f.write_str("chat result serialization refused"),
        }
    }
}
impl Error for ChatError {}
impl From<GenerationError> for ChatError { fn from(error: GenerationError) -> Self { Self::Native(error) } }

/// Reused immutable assets and frozen host ceilings. This planner does not
/// load/activate model weights or invent model/backend identity facts.
pub struct ChatPlanner {
    tokenizer: Arc<EmbeddedTokenizer>, encoder: SourceDocumentEncoder, controls: TemplateControlIds,
    eos: u32, identity: ExecutionIdentity, ceiling: TaskBudget, limits: ChatLimits,
}
pub struct PreparedChat {
    task: TaskPlan, native: GenerationPlan, tokenizer: Arc<EmbeddedTokenizer>, sample_index: u64,
}
#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatResult {
    pub schema_version: u32,
    pub task: String,
    pub execution: String,
    pub numerics_profile: String,
    pub request_seq: u64,
    pub sample_index: u64,
    /// Untrusted assistant data. No tool is parsed or executed by this surface.
    pub content: String,
    pub token_ids: Vec<u32>,
    pub finish_reason: GenerationFinish,
    pub effective_seed: Option<String>,
    pub token_logprobs: Option<Vec<f32>>,
    pub logprob_score_space: Option<DecodeScoreSpace>,
    pub native_work: GenerationWork,
}

impl ChatPlanner {
    pub fn pinned(controls: &TemplateControlIds, eos: u32, mut identity: ExecutionIdentity,
        ceiling: TaskBudget, limits: ChatLimits) -> Result<Self, ChatError> {
        identity.validate().map_err(|_| ChatError::Identity)?;
        ceiling.validate().map_err(|_| ChatError::Limit("host task ceiling"))?;
        if identity.numerics_profile != NumericsProfile::HfBf16Eager || identity.kv_dtype != "bf16"
            || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None {
            return Err(ChatError::Contract("profile/thinking/tools"));
        }
        if limits.max_messages == 0 || limits.max_messages > 128 || limits.max_message_bytes == 0
            || limits.max_total_message_bytes == 0 || limits.generation.max_prompt_tokens == 0 {
            return Err(ChatError::Limit("planner limits"));
        }
        let tokenizer = Arc::new(EmbeddedTokenizer::pinned().map_err(|_| ChatError::Encoding)?);
        if tokenizer.eos_token_id() != Some(eos) {
            return Err(ChatError::Contract("chat EOS differs from tokenizer configuration"));
        }
        for marker in [IM_START, IM_END, THINK_START, THINK_END] {
            let ids = tokenizer.tokenizer().encode_ids_with_options(marker, EncodeOptions { add_bos: false, add_eos: false })
                .map_err(|_| ChatError::Encoding)?;
            if ids.len() != 1 || !controls.entry(ids[0]).is_some_and(|e| e.surface == marker) { return Err(ChatError::Identity); }
        }
        if controls.ids().iter().any(|&id| id as usize >= NANBEIGE_VOCAB_SIZE)
            || !controls.entry(eos).is_some_and(|entry| entry.special && entry.surface == IM_END) {
            return Err(ChatError::Contract("terminal EOS/control registry"));
        }
        let exemplar = render(&[Message::text(MessageRole::System, "FNLP_SYSTEM"), Message::text(MessageRole::User, "FNLP_USER"),
            Message::text(MessageRole::Assistant, "FNLP_ASSISTANT"), Message::text(MessageRole::User, "FNLP_LAST")])?;
        let census: Vec<_> = controls.entries().iter().map(|e| (e.id, e.special, e.surface.as_str())).collect();
        let assets = [PINNED_TOKENIZER_MODEL_BYTES, PINNED_ADDED_TOKENS_BYTES,
            PINNED_TOKENIZER_CONFIG_BYTES, PINNED_SPECIAL_TOKENS_MAP_BYTES].map(Sha256Digest::of_bytes);
        identity.template_digest = hash(&(CHAT_PROMPT_VERSION, exemplar, census, assets, eos))?;
        identity.tokenizer_digest = Sha256Digest::of_bytes(PINNED_TOKENIZER_MODEL_BYTES);
        let encoder = SourceDocumentEncoder::pinned(controls).map_err(|_| ChatError::Encoding)?;
        Ok(Self { tokenizer, encoder, controls: controls.clone(), eos, identity, ceiling, limits })
    }
    pub fn plan_chat(&self, request: &ChatRequest) -> Result<PreparedChat, ChatError> {
        self.compile(BuiltInTask::Chat, &request.item_id, request.sample_index, &request.messages, &request.generation, request.budget)
    }
    pub fn plan_generate(&self, request: &GenerateRequest) -> Result<PreparedChat, ChatError> {
        // Check before the owned single-message conversion, not after cloning
        // an unbounded caller string into a second request representation.
        if request.prompt.len() > self.limits.max_message_bytes || request.prompt.len() > self.limits.max_total_message_bytes {
            return Err(ChatError::Limit("prompt bytes"));
        }
        self.compile(BuiltInTask::Generate, &request.item_id, request.sample_index,
            &[ChatMessage { role: ChatRole::User, content: request.prompt.clone() }], &request.generation, request.budget)
    }
    fn compile(&self, kind: BuiltInTask, item: &str, sample: u64, messages: &[ChatMessage],
        options: &GenerationOptions, budget: TaskBudget) -> Result<PreparedChat, ChatError> {
        // Reject invalid/oversized caller-owned option collections before
        // rendering, tokenizing messages or cloning them into a sealed plan.
        options.validate(self.limits.generation)?;
        validate_messages(messages, self.limits)?;
        budget.validate().map_err(|_| ChatError::Limit("task budget"))?;
        let ceiling = self.ceiling;
        if budget.max_input_tokens > ceiling.max_input_tokens || budget.max_output_tokens > ceiling.max_output_tokens
            || budget.max_output_bytes > ceiling.max_output_bytes || budget.max_grammar_states > ceiling.max_grammar_states
            || budget.max_kv_bytes > ceiling.max_kv_bytes { return Err(ChatError::Limit("task exceeds host ceiling")); }
        if options.eos_token_ids != [self.eos] || options.banned_token_ids.contains(&self.eos) {
            return Err(ChatError::Contract("chat EOS cannot be changed or banned"));
        }
        if options.max_new_tokens > budget.max_output_tokens as usize || options.max_output_bytes as u64 > budget.max_output_bytes {
            return Err(ChatError::Limit("generation exceeds task budget"));
        }
        let placeholders: Vec<_> = (0..messages.len()).map(|index| format!("FNLP_CHAT_SLOT_{index:04}_778cf")).collect();
        let authored: Vec<_> = messages.iter().zip(&placeholders).map(|(message, slot)| Message::text(match message.role {
            ChatRole::System => MessageRole::System, ChatRole::User => MessageRole::User, ChatRole::Assistant => MessageRole::Assistant,
        }, slot.clone())).collect();
        let rendered = render(&authored)?;
        // Reference flow is apply_chat_template(tokenize=false), followed by
        // tokenization with add_special_tokens=false. The template itself owns
        // the leading <|im_start|>; never prepend configured BOS here.
        let mut tail = rendered.as_str(); let mut segments = Vec::new();
        let mut token_total = 0_usize;
        let cap = (budget.max_input_tokens as usize).min(self.limits.generation.max_prompt_tokens);
        for (message, slot) in messages.iter().zip(&placeholders) {
            let (before, rest) = tail.split_once(slot).ok_or(ChatError::Contract("template placeholder"))?;
            let scaffold = self.tokenizer.tokenizer().encode_ids_with_options(before, EncodeOptions { add_bos: false, add_eos: false })
                .map_err(|_| ChatError::Encoding)?;
            token_total = add_tokens(token_total, scaffold.len(), cap)?;
            let source = self.encoder.encode(&message.content, self.limits.max_message_bytes, cap.saturating_sub(token_total))
                .map_err(|_| ChatError::Encoding)?;
            token_total = add_tokens(token_total, source.token_ids().len(), cap)?;
            segments.push(PromptSegment::new(PromptSegmentKind::TaskInstruction, scaffold));
            segments.push(PromptSegment::new(PromptSegmentKind::Document, source.token_ids().to_vec()));
            tail = rest;
        }
        let suffix = self.tokenizer.tokenizer().encode_ids_with_options(tail, EncodeOptions { add_bos: false, add_eos: false })
            .map_err(|_| ChatError::Encoding)?;
        token_total = add_tokens(token_total, suffix.len(), cap)?;
        segments.push(PromptSegment::new(PromptSegmentKind::AnswerScaffold, suffix));
        let ir = TaskIR::new(segments, DecodeStrategy::FreeText { stops: vec![TokenSequence::new(vec![self.eos])],
            budget: DecodeBudget { max_tokens: u32::try_from(options.max_new_tokens).map_err(|_| ChatError::Limit("decode tokens"))?,
                max_bytes: options.max_output_bytes as u64 } }, GrammarReference::none(), None,
            vec![FinitePostcondition::OutputWithinBudget], budget, DependencyScope::ItemLocal).map_err(|_| ChatError::Contract("TaskIR"))?;
        let mut identity = self.identity.clone(); identity.task_spec = kind.spec().identity();
        identity.taskir_digest = ir.digest().map_err(|_| ChatError::Identity)?;
        identity.grammar_compiler_version = "none".to_owned();
        identity.schema_digest = hash(&(kind.spec().request_schema(), kind.spec().response_schema(), CHAT_PROMPT_VERSION))?;
        let context = PlanContext::new(&identity, self.ceiling).map_err(|_| ChatError::Identity)?;
        let task = TaskPlan::new(kind.spec(), &context, ir).map_err(|_| ChatError::Contract("task plan"))?;
        let mut prompt = Vec::new(); prompt.try_reserve_exact(token_total).map_err(|_| ChatError::Allocation)?;
        prompt.extend(task.ir().prompt_segments().iter().flat_map(|s| s.token_ids().iter().copied()));
        let mut effective = options.clone();
        // Request options cannot turn role/thinking controls back on. Exact
        // rendered scaffold controls remain allowed only in the input prompt.
        effective.banned_token_ids.extend(self.controls.ids().iter().copied().filter(|&id| id != self.eos));
        let native = GenerationPlan::compile(prompt, effective, identity, item, sample, self.limits.generation)?;
        Ok(PreparedChat { task, native, tokenizer: Arc::clone(&self.tokenizer), sample_index: sample })
    }
}
impl PreparedChat {
    pub fn task_plan(&self) -> &TaskPlan { &self.task }
    pub fn native_plan(&self) -> &GenerationPlan { &self.native }
    pub fn execution_identity(&self) -> &ExecutionIdentity { self.native.execution_identity() }
    pub fn planned_work(&self) -> GenerationWork { self.native.planned_work() }
    pub fn preflight_eager(&self, admitted: &ExecutionIdentity, engine: &HfBf16EagerEngine,
        mut budget: GenerationBudget) -> Result<(), ChatError> {
        budget.max_kv_bytes = budget.max_kv_bytes.min(self.task.ir().budget().max_kv_bytes);
        self.native.preflight_eager(admitted, engine, budget).map_err(ChatError::from)
    }
    pub fn execute_eager<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut HfBf16EagerEngine, request_seq: u64, mut budget: GenerationBudget, control: &mut C) -> Result<ChatResult, ChatError> {
        budget.max_kv_bytes = budget.max_kv_bytes.min(self.task.ir().budget().max_kv_bytes);
        let raw = self.native.execute_eager(admitted, engine, self.tokenizer.tokenizer(), request_seq, budget, control)?;
        self.finish(raw)
    }
    pub fn execute_eager_with_sink<S: DecodeEventSink, C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut HfBf16EagerEngine, request_seq: u64, mut budget: GenerationBudget, sink: &mut S, control: &mut C) -> Result<ChatResult, ChatError> {
        budget.max_kv_bytes = budget.max_kv_bytes.min(self.task.ir().budget().max_kv_bytes);
        let raw = self.native.execute_eager_with_sink(admitted, engine, self.tokenizer.tokenizer(), request_seq, budget, sink, control)?;
        self.finish(raw)
    }
    fn finish(&self, raw: GeneratedSequence) -> Result<ChatResult, ChatError> {
        let p = self.native.options();
        let expected_seed = match &p.sampling { GenerationSampling::Greedy => None,
            GenerationSampling::Seeded { effective_seed, .. } => Some(Seed256::from(*effective_seed).to_lower_hex()) };
        let selective_projection = raw.execution == BATCH_GENERATION_VERSION;
        if raw.schema_version != 1 || (raw.execution != GENERATION_VERSION && !selective_projection)
            || raw.numerics_profile != HF_BF16_EAGER_PROFILE
            || raw.sample_index != self.sample_index || raw.effective_seed != expected_seed
            || raw.token_ids.len() > p.max_new_tokens || raw.content_bytes.len() > p.max_output_bytes
            || raw.token_ids.iter().any(|&id| id as usize >= NANBEIGE_VOCAB_SIZE || p.banned_token_ids.binary_search(&id).is_ok()) {
            return Err(ChatError::NoResult("generation envelope"));
        }
        let eos = raw.finish_reason == GenerationFinish::Eos;
        let body_len = raw.token_ids.len().checked_sub(usize::from(eos)).ok_or(ChatError::NoResult("missing EOS"))?;
        if raw.token_ids[..body_len].iter().any(|id| p.eos_token_ids.contains(id))
            || (eos && (!raw.token_ids.last().is_some_and(|id| p.eos_token_ids.contains(id)) || body_len < p.min_new_tokens))
            || (raw.finish_reason == GenerationFinish::TokenLimit && raw.token_ids.len() != p.max_new_tokens)
            || (raw.finish_reason == GenerationFinish::StopSuffix && (body_len < p.min_new_tokens
                || !p.stop_suffixes.iter().any(|s| raw.content_bytes.ends_with(s)))) {
            return Err(ChatError::NoResult("termination"));
        }
        let decoded = self.tokenizer.tokenizer().decode_bytes(&raw.token_ids[..body_len]).map_err(|_| ChatError::NoResult("decoding"))?;
        if decoded != raw.content_bytes { return Err(ChatError::NoResult("exact content bytes")); }
        let proposals = raw.token_ids.len().checked_add(usize::from(raw.finish_reason == GenerationFinish::ByteLimit))
            .ok_or(ChatError::NoResult("work arithmetic"))?;
        let positions = self.native.prompt_tokens().checked_add(proposals).and_then(|n| n.checked_sub(1)).ok_or(ChatError::NoResult("work arithmetic"))? as u64;
        let sampled = if expected_seed.is_some() { proposals as u64 } else { 0 };
        // Batch execution intentionally skips intermediate prompt lm heads.
        // Do not weaken the denominator check or rewrite work to look scalar.
        let projection_rows = if selective_projection { proposals as u64 } else { positions };
        if raw.native_work.forward_positions != positions || positions > self.native.planned_work().forward_positions
            || projection_rows.checked_mul(NANBEIGE_VOCAB_SIZE as u64) != Some(raw.native_work.projected_logits)
            || raw.native_work.sampled_steps != sampled { return Err(ChatError::NoResult("native work")); }
        match (&raw.token_logprobs, raw.logprob_score_space, p.capture_logprobs) {
            (Some(scores), Some(DecodeScoreSpace::FullVocabularyLogSoftmax), true)
                if scores.len() == raw.token_ids.len() && scores.iter().all(|s| s.is_finite() && *s <= 0.0) => {},
            (None, None, false) => {}, _ => return Err(ChatError::NoResult("raw score space")),
        }
        let content = String::from_utf8(raw.content_bytes).map_err(|_| ChatError::NoResult("incomplete UTF-8"))?;
        let result = ChatResult { schema_version: 1, task: self.task.task_spec_identity().to_owned(), execution: raw.execution,
            numerics_profile: raw.numerics_profile, request_seq: raw.request_seq, sample_index: raw.sample_index, content,
            token_ids: raw.token_ids, finish_reason: raw.finish_reason, effective_seed: raw.effective_seed,
            token_logprobs: raw.token_logprobs, logprob_score_space: raw.logprob_score_space, native_work: raw.native_work };
        bounds::result(&result, self.task.ir().budget().max_output_bytes)?;
        Ok(result)
    }
}
fn render(messages: &[Message]) -> Result<String, ChatError> {
    TemplateBuilder::with_options(RenderOptions { add_generation_prompt: true, enable_thinking: false,
        preserve_thinking: false, tool_format: ToolFormat::Xml }).render(&Conversation::new(messages.to_vec())).map_err(|_| ChatError::Encoding)
}
fn validate_messages(messages: &[ChatMessage], limits: ChatLimits) -> Result<(), ChatError> {
    if messages.is_empty() || messages.len() > limits.max_messages { return Err(ChatError::Limit("message count")); }
    let mut expected = ChatRole::User; let mut bytes = 0_usize;
    for (index, message) in messages.iter().enumerate() {
        bytes = bytes.checked_add(message.content.len()).ok_or(ChatError::Limit("message bytes"))?;
        if message.content.len() > limits.max_message_bytes || bytes > limits.max_total_message_bytes { return Err(ChatError::Limit("message bytes")); }
        if index == 0 && message.role == ChatRole::System { continue; }
        if message.role != expected { return Err(ChatError::Contract("roles must alternate user/assistant after optional first system")); }
        expected = if expected == ChatRole::User { ChatRole::Assistant } else { ChatRole::User };
    }
    if messages.last().is_none_or(|m| m.role != ChatRole::User) { return Err(ChatError::Contract("last message must be user")); }
    Ok(())
}
fn add_tokens(total: usize, count: usize, cap: usize) -> Result<usize, ChatError> {
    total.checked_add(count).filter(|&n| n <= cap).ok_or(ChatError::Limit("complete transcript tokens"))
}
fn hash<T: Serialize>(value: &T) -> Result<Sha256Digest, ChatError> {
    canonjson::canonical_bytes(value).map(|bytes| Sha256Digest::of_bytes(&bytes)).map_err(|_| ChatError::Serialization)
}

#[cfg(test)] mod tests;
