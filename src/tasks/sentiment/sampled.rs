//! Independent seeded sentiment generations, each constrained to a bounded
//! score/justification object. No distribution-mode scores are sampled here.
//!
//! The complete axis × sample schedule is admitted before the first forward.
//! One schema and one vocabulary serve every fresh, causal native session.
//! Descriptive spread measures variation between generated scores, never
//! confidence, calibrated uncertainty, human agreement, or factual support.

use std::{collections::BTreeSet, error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{
    canonjson,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::{CompileLimits, mask::{VocabMaskOracle, VocabTrie}, runtime::{JsonProgram, JSON_RUNTIME_VERSION}},
    native_engine::{
        constrained::JsonDecodeOptions,
        constrained_int4::{self, Int4JsonBudget, Int4JsonError, Int4JsonRun, JsonSamplingOptions,
            INT4_JSON_SAMPLED_EXECUTION, JSON_SAMPLER_VERSION, json_sampling_workspace_bytes},
        decode::{DecodeCancellationKind, DecodeStepControl},
        generation::grouped_int4::bind_candidate_model,
        lmhead::NANBEIGE_VOCAB_SIZE,
        strict_int4::{Int4ModelIdentity, Int4RunBudget, Int4Work, StrictInt4Engine, StrictInt4Error,
            STRICT_INT4_EXECUTION, STRICT_INT4_PROFILE},
    },
    tasks::{BuiltInTask, ir::{DecodeStrategy, DependencyScope, FinitePostcondition, GrammarReference,
        PlanContext, PromptSegment, PromptSegmentKind, TaskBudget, TaskIR, TaskPlan}},
    template::{Conversation, Message, MessageRole, RenderOptions, TemplateBuilder, ToolFormat},
    tokenizer::{bpe::EncodeOptions, embedded::{EmbeddedTokenizer, PINNED_TOKENIZER_MODEL_BYTES,
        PINNED_ADDED_TOKENS_BYTES, PINNED_TOKENIZER_CONFIG_BYTES, PINNED_SPECIAL_TOKENS_MAP_BYTES},
        pinned_controls, specials::TemplateControlIds, untrusted::UntrustedDocumentEncoder},
};
use super::{SentimentAxis, SentimentRequest};

pub const SAMPLED_SENTIMENT_EXECUTION: &str = "grouped-int4-independent-sampled-sentiment-v1";
pub const SAMPLED_SENTIMENT_PROMPT_VERSION: &str = "sentiment-sampled-score-justification-v1";
pub const SAMPLED_SENTIMENT_AGGREGATION_VERSION: &str = "canonical-axis-sample-population-moments-v1";
pub const MAX_SENTIMENT_SAMPLES: usize = 32;
const PROMPT_DATA: &str = include_str!("../presets/sentiment_sampled_v1.json");
const DOCUMENT_SLOT: &str = "FNLP_INTERNAL_SAMPLED_SENTIMENT_DOCUMENT_61ed7a";
const MIB: u64 = 1024 * 1024;

/// An explicit seed family with addressed, independent sample ordinals. The
/// seed is private admission data: Debug redacts it and results omit it.
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SampledSentimentOptions {
    pub effective_seed: [u8; 32],
    pub samples: usize,
    pub temperature_milli: u32,
    pub top_k: Option<usize>,
    pub top_p_ppm: u32,
    /// Per-generation token limit, including EOS. The request TaskBudget has
    /// the separate, aggregate axis × sample output-token ceiling.
    pub max_sample_tokens: usize,
    pub max_justification_chars: usize,
}
impl fmt::Debug for SampledSentimentOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SampledSentimentOptions").field("effective_seed", &"REDACTED")
            .field("samples", &self.samples).field("temperature_milli", &self.temperature_milli)
            .field("top_k", &self.top_k).field("top_p_ppm", &self.top_p_ppm)
            .field("max_sample_tokens", &self.max_sample_tokens)
            .field("max_justification_chars", &self.max_justification_chars).finish()
    }
}
impl SampledSentimentOptions {
    pub fn validate(&self) -> Result<(), SampledSentimentError> {
        if !(1..=MAX_SENTIMENT_SAMPLES).contains(&self.samples)
            || !(1..=1024).contains(&self.max_sample_tokens)
            || !(1..=4096).contains(&self.max_justification_chars) {
            return Err(SampledSentimentError::Limits);
        }
        self.sampling(SentimentAxis::Valence, 0)?.validate()?;
        Ok(())
    }
    fn sampling(&self, axis: SentimentAxis, sample: usize) -> Result<JsonSamplingOptions, SampledSentimentError> {
        if sample >= self.samples || sample >= MAX_SENTIMENT_SAMPLES { return Err(SampledSentimentError::Accounting); }
        Ok(JsonSamplingOptions { effective_seed: self.effective_seed,
            temperature_milli: self.temperature_milli, top_k: self.top_k, top_p_ppm: self.top_p_ppm,
            sample_index: (axis as u64) * MAX_SENTIMENT_SAMPLES as u64 + sample as u64 })
    }
}

/// Grammar/storage limits for the entire operation. Prompt-token work counts
/// repeated sample execution, although a prompt is stored only once per axis.
#[derive(Clone, Copy, Debug)]
pub struct SampledSentimentLimits {
    pub max_input_bytes: usize,
    pub max_total_prompt_tokens: usize,
    pub max_sample_json_bytes: usize,
    pub max_result_bytes: u64,
    pub compile: CompileLimits,
}
impl Default for SampledSentimentLimits {
    fn default() -> Self {
        Self { max_input_bytes: 65_536, max_total_prompt_tokens: 131_072,
            max_sample_json_bytes: 8192, max_result_bytes: MIB, compile: CompileLimits::default() }
    }
}
impl SampledSentimentLimits {
    pub fn validate(&self) -> Result<(), SampledSentimentError> {
        if !(1..=16 * 1024 * 1024).contains(&self.max_input_bytes)
            || !(1..=16 * 1024 * 1024).contains(&self.max_total_prompt_tokens)
            || !(32..=65_536).contains(&self.max_sample_json_bytes)
            || !(1..=64 * MIB).contains(&self.max_result_bytes)
            || self.compile.max_schema_bytes == 0 || self.compile.max_string_bytes == 0
            || self.compile.max_output_bytes == 0 || self.compile.max_states == 0
            || self.compile.max_transitions == 0 || self.compile.max_mask_bytes == 0 {
            return Err(SampledSentimentError::Limits);
        }
        Ok(())
    }
}

/// Privacy-safe typed failure; native cancellation causes are never replaced.
pub enum SampledSentimentError {
    Limits, Identity, Tokenizer, Template, Schema, Allocation, Serialization,
    WorkBudget, OutputBudget, NoResult, Accounting, Native(Int4JsonError), Cancelled(DecodeCancellationKind),
}
impl fmt::Display for SampledSentimentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Limits => "sampled sentiment request or grammar limits refused",
            Self::Identity => "sampled sentiment model or sealed execution identity differs",
            Self::Tokenizer => "sampled sentiment lossless control-free encoding refused",
            Self::Template => "sampled sentiment pinned prompt data refused",
            Self::Schema => "sampled sentiment bounded score/justification schema refused",
            Self::Allocation => "sampled sentiment bounded allocation refused",
            Self::Serialization => "sampled sentiment canonical serialization refused",
            Self::WorkBudget => "sampled sentiment complete generation work exceeds admission",
            Self::OutputBudget => "sampled sentiment complete result exceeds its byte budget",
            Self::NoResult => "sampled sentiment sample failed independent content validation",
            Self::Accounting => "sampled sentiment native output or aggregate accounting differs",
            Self::Native(_) => "sampled sentiment native constrained generation failed",
            Self::Cancelled(_) => "sampled sentiment cancelled",
        })
    }
}
impl fmt::Debug for SampledSentimentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for SampledSentimentError {}
impl From<Int4JsonError> for SampledSentimentError { fn from(error: Int4JsonError) -> Self { Self::Native(error) } }
impl From<StrictInt4Error> for SampledSentimentError {
    fn from(error: StrictInt4Error) -> Self { Self::Native(Int4JsonError::Native(error)) }
}
impl SampledSentimentError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Cancelled(cause) => Some(*cause), Self::Native(error) => error.cancellation(), _ => None }
    }
}

/// Pinned decoder vocabulary, reusable across independently admitted requests.
pub struct SampledSentimentVocabulary { oracle: VocabMaskOracle, controls: BTreeSet<u32> }
impl SampledSentimentVocabulary {
    pub fn pinned() -> Result<Self, SampledSentimentError> {
        let registry = pinned_controls::pinned().map_err(|_| SampledSentimentError::Tokenizer)?;
        let tokenizer = EmbeddedTokenizer::pinned().map_err(|_| SampledSentimentError::Tokenizer)?;
        let trie = VocabTrie::from_tokenizer(tokenizer.tokenizer(), NANBEIGE_VOCAB_SIZE)
            .map_err(|_| SampledSentimentError::Tokenizer)?;
        Ok(Self { oracle: VocabMaskOracle::new(trie), controls: registry.template_controls().ids().clone() })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptData { schema_version: u32, version: String, origin: String, license: String,
    system: String, response: String, axes: Vec<AxisPrompt> }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AxisPrompt { axis: SentimentAxis, instruction: String }
struct AxisTemplate { axis: SentimentAxis, instruction: Vec<u32>, scaffold: Vec<u32> }
struct Compiler { tokenizer: EmbeddedTokenizer, controls: TemplateControlIds,
    global: Vec<u32>, axes: Vec<AxisTemplate>, template_digest: Sha256Digest,
    schema: String, schema_digest: Sha256Digest, decode: JsonDecodeOptions }

/// Pinned task compiler derived from an actual authenticated native model.
/// It retains identity and prompts, never a second copy of model weights.
pub struct SampledSentimentPlanner {
    compiler: Compiler, identity: ExecutionIdentity, group_width: usize,
    options: SampledSentimentOptions, ceiling: TaskBudget, limits: SampledSentimentLimits,
}
impl SampledSentimentPlanner {
    pub fn for_model(model: &Int4ModelIdentity, options: SampledSentimentOptions,
        ceiling: TaskBudget, limits: SampledSentimentLimits) -> Result<Self, SampledSentimentError> {
        options.validate()?; limits.validate()?;
        ceiling.validate().map_err(|_| SampledSentimentError::Limits)?;
        let compiler = Compiler::pinned(&options, ceiling, limits)?;
        let identity = bind_candidate_model(unbound_identity(&compiler), model)
            .map_err(|_| SampledSentimentError::Identity)?;
        constrained_int4::check_model(&identity, model)?;
        Ok(Self { compiler, identity, group_width: model.group_width(), options, ceiling, limits })
    }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn plan(&self, request: &SentimentRequest) -> Result<PreparedSampledSentiment, SampledSentimentError> {
        self.plan_with_control(request, &mut Continue)
    }
    pub fn plan_with_control<C: DecodeStepControl>(&self, request: &SentimentRequest, control: &mut C)
        -> Result<PreparedSampledSentiment, SampledSentimentError> {
        compile_request(&self.compiler, &self.identity, self.group_width, &self.options,
            self.ceiling, self.limits, request, control)
    }
}

struct PreparedAxis { axis: SentimentAxis, prompt: Vec<u32>, identity: ExecutionIdentity, work: Int4Work }

/// Sealed prompts, sampled policy and complete native schedule. There is no
/// serde/Debug constructor and no public path accepting invented native runs.
pub struct PreparedSampledSentiment {
    axes: Vec<PreparedAxis>, program: JsonProgram, options: SampledSentimentOptions,
    decode: JsonDecodeOptions, controls: BTreeSet<u32>, identity: ExecutionIdentity,
    group_width: usize, work: Int4Work, budget: TaskBudget, context: usize,
    max_result_bytes: u64, preparation_bytes: u64,
}
impl PreparedSampledSentiment {
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn planned_work(&self) -> Int4Work { self.work }
    pub fn required_context(&self) -> usize { self.context }
    pub fn task_budget(&self) -> TaskBudget { self.budget }
    pub fn group_width(&self) -> usize { self.group_width }
    pub fn max_result_bytes(&self) -> u64 { self.max_result_bytes }
    pub fn sample_count(&self) -> usize { self.axes.len() * self.options.samples }
    /// Declared payload reserve, not measured RSS. Includes pinned compiler/
    /// vocabulary construction allowance and simultaneous result serialization.
    pub fn preparation_requirement_bytes(&self) -> u64 { self.preparation_bytes }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_control<C: DecodeStepControl>(&self, engine: &mut StrictInt4Engine<'_>,
        admitted: &ExecutionIdentity, vocabulary: &SampledSentimentVocabulary,
        budget: Int4JsonBudget, max_sampler_bytes: u64, control: &mut C)
        -> Result<SampledSentimentRun, SampledSentimentError> {
        checkpoint(control)?;
        self.verify_identity(admitted)?;
        constrained_int4::check_model(&self.identity, engine.model_identity())?;
        if engine.model_identity().group_width() != self.group_width || engine.is_poisoned()
            || !engine.kv_cache().all_slots_have_len(0) || vocabulary.controls != self.controls {
            return Err(SampledSentimentError::Identity);
        }
        let capacity = engine.kv_cache().capacity_positions();
        let resident_kv = (capacity as u64).checked_mul(crate::native_engine::kv::KV_BYTES_PER_TOKEN as u64)
            .ok_or(SampledSentimentError::WorkBudget)?;
        if capacity < self.context || resident_kv > budget.json.max_kv_bytes {
            return Err(SampledSentimentError::WorkBudget);
        }
        let mask_charge = self.preflight(vocabulary, budget, max_sampler_bytes)?;
        let mut dimensions = reserved(self.axes.len())?;
        let mut actual_work = Int4Work::default();
        let mut actual_mask = 0_u64;
        let mut actual_tokens = 0_u64;
        let mut retained_sample_bytes = 0_u64;
        for axis in &self.axes {
            let mut samples = reserved(self.options.samples)?;
            for sample_index in 0..self.options.samples {
                checkpoint(control)?;
                let sampling = self.options.sampling(axis.axis, sample_index)?;
                let sample_budget = Int4JsonBudget { native: Int4RunBudget::exact(axis.work),
                    json: crate::native_engine::constrained::JsonWorkBudget {
                        max_forward_positions: axis.work.forward_positions,
                        max_projected_logits: axis.work.projected_logits,
                        max_kv_bytes: budget.json.max_kv_bytes,
                        max_total_mask_node_visits: mask_charge,
                        mask_limits: budget.json.mask_limits } };
                let checked = constrained_int4::decode_json_int4_sampled_with(engine, &axis.identity,
                    &axis.prompt, &self.program, &vocabulary.oracle, &self.decode, &sampling,
                    sample_budget, max_sampler_bytes, control,
                    |run| self.finalize_sample(axis, sample_index, run))?;
                // Check cumulative retained sample storage before committing
                // this sample to the bundle; the final envelope is checked too.
                let serialized = canonjson::canonical_bytes(&checked).map_err(|_| SampledSentimentError::Serialization)?;
                retained_sample_bytes = retained_sample_bytes.checked_add(serialized.len() as u64)
                    .filter(|&bytes| bytes <= self.max_result_bytes).ok_or(SampledSentimentError::OutputBudget)?;
                drop(serialized);
                actual_work = actual_work.checked_add(checked.model_work)?;
                actual_mask = actual_mask.checked_add(checked.mask_node_visit_charge).ok_or(SampledSentimentError::Accounting)?;
                actual_tokens = actual_tokens.checked_add(checked.generated_tokens as u64).ok_or(SampledSentimentError::Accounting)?;
                samples.push(checked);
                checkpoint(control)?;
            }
            dimensions.push(aggregate(axis.axis, samples)?);
        }
        let result = SampledSentimentRun { schema_version: 1,
            execution: SAMPLED_SENTIMENT_EXECUTION.to_owned(), task_spec: "sentiment-v1".to_owned(),
            prompt_template_sha256: self.identity.template_digest,
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: self.group_width,
            aggregation: SAMPLED_SENTIMENT_AGGREGATION_VERSION.to_owned(),
            interpretation: "generated_score_variation_not_calibrated_confidence".to_owned(),
            justification_status: "unverified_model_explanations_from_untrusted_document".to_owned(),
            sampler_version: JSON_SAMPLER_VERSION.to_owned(),
            samples_per_axis: self.options.samples, dimensions, model_work: actual_work,
            generated_tokens: actual_tokens, mask_node_visit_charge: actual_mask };
        checkpoint(control)?;
        if !fits_work(actual_work, self.work) || actual_tokens > u64::from(self.budget.max_output_tokens)
            || actual_mask > budget.json.max_total_mask_node_visits {
            return Err(SampledSentimentError::Accounting);
        }
        check_output_size(&result, self.max_result_bytes)?;
        checkpoint(control)?;
        Ok(result)
    }

    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), SampledSentimentError> {
        admitted.validate().map_err(|_| SampledSentimentError::Identity)?;
        if canonjson::canonical_bytes(admitted).map_err(|_| SampledSentimentError::Identity)?
            != canonjson::canonical_bytes(&self.identity).map_err(|_| SampledSentimentError::Identity)? {
            return Err(SampledSentimentError::Identity);
        }
        Ok(())
    }
    fn preflight(&self, vocabulary: &SampledSentimentVocabulary, budget: Int4JsonBudget,
        max_sampler_bytes: u64) -> Result<u64, SampledSentimentError> {
        if self.work.forward_positions > budget.native.max_forward_positions
            || self.work.attention_pairs > budget.native.max_attention_pairs
            || !self.work.projections.fits(budget.native.max_projection_work)
            || self.work.forward_positions > budget.json.max_forward_positions
            || self.work.projected_logits > budget.json.max_projected_logits
            || self.budget.max_kv_bytes < budget.json.max_kv_bytes
            || max_sampler_bytes < json_sampling_workspace_bytes()?
            || budget.json.mask_limits.max_trie_node_visits == 0
            || budget.json.mask_limits.checkpoint_interval_nodes == 0 {
            return Err(SampledSentimentError::WorkBudget);
        }
        let per_step = vocabulary.oracle.trie().node_count().min(budget.json.mask_limits.max_trie_node_visits) as u64;
        let one = per_step.checked_mul(self.options.max_sample_tokens as u64).ok_or(SampledSentimentError::WorkBudget)?;
        let all = one.checked_mul(self.sample_count() as u64).ok_or(SampledSentimentError::WorkBudget)?;
        if all > budget.json.max_total_mask_node_visits { return Err(SampledSentimentError::WorkBudget); }
        Ok(one)
    }
    fn finalize_sample(&self, axis: &PreparedAxis, sample_index: usize, run: Int4JsonRun)
        -> Result<SampledSentimentSample, SampledSentimentError> {
        let output = &run.output;
        if sample_index >= self.options.samples || run.schema_version != 1
            || run.execution != INT4_JSON_SAMPLED_EXECUTION || run.group_width != self.group_width
            || output.schema_version != 1 || output.numerics_profile != STRICT_INT4_PROFILE
            || output.token_ids.is_empty() || output.token_ids.len() > self.options.max_sample_tokens
            || output.token_ids.last() != Some(&self.decode.eos_token_id)
            || output.token_ids[..output.token_ids.len() - 1].iter().any(|id|
                *id as usize >= NANBEIGE_VOCAB_SIZE || self.decode.excluded_token_ids.contains(id)) {
            return Err(SampledSentimentError::Accounting);
        }
        let positions = axis.prompt.len().checked_add(output.token_ids.len() - 1).ok_or(SampledSentimentError::Accounting)?;
        let logits = output.token_ids.len().checked_mul(NANBEIGE_VOCAB_SIZE).ok_or(SampledSentimentError::Accounting)?;
        let expected = Int4Work::for_sequence(self.group_width, 0, positions, logits)?;
        if expected != run.model_work || !fits_work(expected, axis.work)
            || output.forward_positions != expected.forward_positions || output.projected_logits != expected.projected_logits {
            return Err(SampledSentimentError::Accounting);
        }
        self.program.validate_json(&output.json).map_err(|_| SampledSentimentError::NoResult)?;
        let value = canonjson::parse_str_with_limits(&output.json, canonjson::ParseLimits {
            max_depth: 4, max_string_bytes: self.program.max_output_bytes() })
            .map_err(|_| SampledSentimentError::NoResult)?;
        let decoded: SampleValue = serde_json::from_value(value).map_err(|_| SampledSentimentError::NoResult)?;
        if !(-100..=100).contains(&decoded.score) || decoded.justification.trim().is_empty()
            || decoded.justification.chars().count() > self.options.max_justification_chars {
            return Err(SampledSentimentError::NoResult);
        }
        Ok(SampledSentimentSample { sample_index, score: decoded.score, justification: decoded.justification,
            generated_tokens: output.token_ids.len(), model_work: run.model_work,
            mask_node_visit_charge: output.mask_node_visit_charge })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SampleValue { score: i16, justification: String }

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SampledSentimentSample {
    pub sample_index: usize, pub score: i16, pub justification: String,
    pub generated_tokens: usize, pub model_work: Int4Work, pub mask_node_visit_charge: u64,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SampledSentimentDimension {
    pub axis: SentimentAxis, pub low_anchor: String, pub high_anchor: String,
    pub mean: f64, pub median: f64, pub minimum: i16, pub maximum: i16,
    pub population_stddev: f64, pub samples: Vec<SampledSentimentSample>,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SampledSentimentRun {
    pub schema_version: u32, pub execution: String, pub task_spec: String,
    pub prompt_template_sha256: Sha256Digest, pub numerics_profile: String,
    pub group_width: usize, pub aggregation: String, pub interpretation: String,
    pub justification_status: String, pub sampler_version: String,
    pub samples_per_axis: usize, pub dimensions: Vec<SampledSentimentDimension>,
    pub model_work: Int4Work, pub generated_tokens: u64, pub mask_node_visit_charge: u64,
}

fn aggregate(axis: SentimentAxis, mut samples: Vec<SampledSentimentSample>)
    -> Result<SampledSentimentDimension, SampledSentimentError> {
    if samples.is_empty() || samples.len() > MAX_SENTIMENT_SAMPLES { return Err(SampledSentimentError::Accounting); }
    samples.sort_unstable_by_key(|sample| sample.sample_index);
    if samples.iter().enumerate().any(|(index, sample)| sample.sample_index != index || !(-100..=100).contains(&sample.score)) {
        return Err(SampledSentimentError::Accounting);
    }
    let mut scores = reserved(samples.len())?;
    scores.extend(samples.iter().map(|sample| sample.score)); scores.sort_unstable();
    let sum: i64 = scores.iter().map(|&score| i64::from(score)).sum();
    let mean = sum as f64 / scores.len() as f64;
    let middle = scores.len() / 2;
    let median = if scores.len() % 2 == 0 { (f64::from(scores[middle - 1]) + f64::from(scores[middle])) / 2.0 }
        else { f64::from(scores[middle]) };
    let variance = scores.iter().map(|&score| (f64::from(score) - mean).powi(2)).sum::<f64>() / scores.len() as f64;
    let (low, high) = axis.anchors();
    Ok(SampledSentimentDimension { axis, low_anchor: low.to_owned(), high_anchor: high.to_owned(),
        mean, median, minimum: scores[0], maximum: scores[scores.len() - 1],
        population_stddev: variance.sqrt(), samples })
}

impl Compiler {
    fn pinned(options: &SampledSentimentOptions, ceiling: TaskBudget, limits: SampledSentimentLimits)
        -> Result<Self, SampledSentimentError> {
        let value = canonjson::parse_str(PROMPT_DATA).map_err(|_| SampledSentimentError::Template)?;
        let data: PromptData = serde_json::from_value(value).map_err(|_| SampledSentimentError::Template)?;
        if data.schema_version != 1 || data.version != SAMPLED_SENTIMENT_PROMPT_VERSION
            || data.origin.is_empty() || data.license.is_empty() || data.axes.len() != SentimentAxis::ALL.len()
            || data.axes.iter().zip(SentimentAxis::ALL).any(|(entry, axis)| entry.axis != axis) {
            return Err(SampledSentimentError::Template);
        }
        let registry = pinned_controls::pinned().map_err(|_| SampledSentimentError::Tokenizer)?;
        let controls = registry.template_controls().clone();
        let tokenizer = EmbeddedTokenizer::pinned().map_err(|_| SampledSentimentError::Tokenizer)?;
        let eos = tokenizer.eos_token_id().ok_or(SampledSentimentError::Tokenizer)?;
        if !controls.entry(eos).is_some_and(|entry| entry.special) { return Err(SampledSentimentError::Tokenizer); }
        let render = |generation| RenderOptions { add_generation_prompt: generation, enable_thinking: false,
            preserve_thinking: false, tool_format: ToolFormat::Xml };
        let global_text = TemplateBuilder::with_options(render(false)).render(&Conversation::new(vec![
            Message::text(MessageRole::System, &data.system)])).map_err(|_| SampledSentimentError::Template)?;
        let encode = |text: &str| tokenizer.tokenizer().encode_ids_with_options(text,
            EncodeOptions { add_bos: false, add_eos: false }).map_err(|_| SampledSentimentError::Tokenizer);
        let global = encode(&global_text)?;
        let mut axes = reserved(data.axes.len())?;
        for entry in data.axes {
            let user = format!("{}\n{}\nThe justification may contain at most {} Unicode scalar values.\n\nDocument:\n{}",
                entry.instruction, data.response, options.max_justification_chars, DOCUMENT_SLOT);
            let text = TemplateBuilder::with_options(render(true)).render(&Conversation::new(vec![
                Message::text(MessageRole::System, &data.system), Message::text(MessageRole::User, user)]))
                .map_err(|_| SampledSentimentError::Template)?;
            let tail = text.strip_prefix(&global_text).ok_or(SampledSentimentError::Template)?;
            let (instruction, scaffold) = tail.split_once(DOCUMENT_SLOT).ok_or(SampledSentimentError::Template)?;
            if scaffold.contains(DOCUMENT_SLOT) || instruction.is_empty() || scaffold.is_empty() {
                return Err(SampledSentimentError::Template);
            }
            axes.push(AxisTemplate { axis: entry.axis, instruction: encode(instruction)?, scaffold: encode(scaffold)? });
        }
        let schema = canonjson::canonical_bytes(&serde_json::json!({"type":"object",
            "additionalProperties":false,"required":["score","justification"],
            "properties":{"score":{"type":"integer","enum":(-100..=100).collect::<Vec<i32>>()},
                // JsonProgram normalizes the legacy compiler's byte dialect:
                // maxLength bounds Unicode scalars while max_string_bytes
                // separately reserves up to four UTF-8 bytes per scalar.
                "justification":{"type":"string","maxLength":options.max_justification_chars}}}))
            .map_err(|_| SampledSentimentError::Serialization)?;
        let schema_digest = Sha256Digest::of_bytes(&schema);
        let mut compile = limits.compile;
        compile.max_output_bytes = compile.max_output_bytes.min(limits.max_sample_json_bytes);
        compile.max_states = compile.max_states.min(ceiling.max_grammar_states as usize);
        compile.max_string_bytes = compile.max_string_bytes.min(options.max_justification_chars.saturating_mul(4));
        let schema = String::from_utf8(schema).map_err(|_| SampledSentimentError::Serialization)?;
        JsonProgram::compile(&schema, compile).map_err(|_| SampledSentimentError::Schema)?;
        let template_digest = digest(&(SAMPLED_SENTIMENT_PROMPT_VERSION, PROMPT_DATA,
            options.max_justification_chars,
            [PINNED_TOKENIZER_MODEL_BYTES, PINNED_ADDED_TOKENS_BYTES, PINNED_TOKENIZER_CONFIG_BYTES,
                PINNED_SPECIAL_TOKENS_MAP_BYTES].map(Sha256Digest::of_bytes), controls.ids()))?;
        Ok(Self { tokenizer, controls: controls.clone(), global, axes, template_digest, schema, schema_digest,
            decode: JsonDecodeOptions { max_new_tokens: options.max_sample_tokens, eos_token_id: eos,
                excluded_token_ids: controls.ids().clone() } })
    }
}

#[allow(clippy::too_many_arguments)]
fn compile_request<C: DecodeStepControl>(compiler: &Compiler, identity: &ExecutionIdentity, group_width: usize,
    options: &SampledSentimentOptions, ceiling: TaskBudget, limits: SampledSentimentLimits,
    request: &SentimentRequest, control: &mut C) -> Result<PreparedSampledSentiment, SampledSentimentError> {
    checkpoint(control)?; options.validate()?; limits.validate()?;
    request.budget.validate().map_err(|_| SampledSentimentError::Limits)?;
    if request.document.is_empty() || request.document.len() > limits.max_input_bytes
        || request.axes.is_empty() || request.axes.len() > SentimentAxis::ALL.len()
        || !budget_fits(request.budget, ceiling) || identity.template_digest != compiler.template_digest
        || identity.task_spec != "sentiment-v1" || identity.sampler_version != JSON_SAMPLER_VERSION
        || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None {
        return Err(SampledSentimentError::Limits);
    }
    let mut ordered = request.axes.clone(); ordered.sort_unstable();
    if ordered.windows(2).any(|pair| pair[0] == pair[1]) { return Err(SampledSentimentError::Limits); }
    let generations = ordered.len().checked_mul(options.samples).ok_or(SampledSentimentError::WorkBudget)?;
    let output_tokens = generations.checked_mul(options.max_sample_tokens).ok_or(SampledSentimentError::WorkBudget)?;
    if output_tokens as u64 > u64::from(request.budget.max_output_tokens) { return Err(SampledSentimentError::WorkBudget); }
    let mut executed_prompt_tokens = 0_usize;
    let mut retained_prompt_tokens = 0_usize;
    let mut context = 0_usize;
    let mut work = Int4Work::default();
    let mut lengths = reserved(ordered.len())?;
    for axis in &ordered {
        let template = compiler.axes.iter().find(|entry| entry.axis == *axis).ok_or(SampledSentimentError::Template)?;
        let length = compiler.global.len().checked_add(template.instruction.len())
            .and_then(|n| n.checked_add(request.document.len())).and_then(|n| n.checked_add(template.scaffold.len()))
            .ok_or(SampledSentimentError::WorkBudget)?;
        executed_prompt_tokens = executed_prompt_tokens.checked_add(length.checked_mul(options.samples)
            .ok_or(SampledSentimentError::WorkBudget)?).ok_or(SampledSentimentError::WorkBudget)?;
        retained_prompt_tokens = retained_prompt_tokens.checked_add(length).ok_or(SampledSentimentError::WorkBudget)?;
        if length > request.budget.max_input_tokens as usize || executed_prompt_tokens > limits.max_total_prompt_tokens {
            return Err(SampledSentimentError::WorkBudget);
        }
        context = context.max(length.checked_add(options.max_sample_tokens - 1).ok_or(SampledSentimentError::WorkBudget)?);
        let one = constrained_int4::planned_work(group_width, length, options.max_sample_tokens)?;
        for _ in 0..options.samples { work = work.checked_add(one)?; }
        lengths.push((length, one));
    }
    let bytes = (context as u64).checked_mul(crate::native_engine::kv::KV_BYTES_PER_TOKEN as u64)
        .ok_or(SampledSentimentError::WorkBudget)?;
    if bytes > request.budget.max_kv_bytes { return Err(SampledSentimentError::WorkBudget); }
    let max_result_bytes = request.budget.max_output_bytes.min(limits.max_result_bytes);
    // A tighter child request recompiles the single shared grammar under its
    // actual caps; a planner's larger grammar allowance cannot leak through.
    let mut compile = limits.compile;
    compile.max_output_bytes = compile.max_output_bytes.min(limits.max_sample_json_bytes)
        .min(usize::try_from(max_result_bytes).map_err(|_| SampledSentimentError::Limits)?);
    compile.max_states = compile.max_states.min(request.budget.max_grammar_states as usize);
    compile.max_string_bytes = compile.max_string_bytes.min(options.max_justification_chars.saturating_mul(4));
    let program = JsonProgram::compile(&compiler.schema, compile).map_err(|_| SampledSentimentError::Schema)?;
    let preparation_bytes = preparation_quote(retained_prompt_tokens, max_result_bytes, program.max_output_bytes())?;
    checkpoint(control)?;
    let document = UntrustedDocumentEncoder::new(compiler.tokenizer.tokenizer(), &compiler.controls)
        .encode(request.document.as_bytes()).map_err(|_| SampledSentimentError::Tokenizer)?;
    if document.ids().len() != request.document.len() { return Err(SampledSentimentError::Tokenizer); }
    checkpoint(control)?;
    let context_identity = PlanContext::new(identity, ceiling).map_err(|_| SampledSentimentError::Identity)?;
    let mut axes = reserved(ordered.len())?;
    for (axis, (length, one)) in ordered.into_iter().zip(lengths) {
        checkpoint(control)?;
        let template = compiler.axes.iter().find(|entry| entry.axis == axis).ok_or(SampledSentimentError::Template)?;
        let sample_budget = TaskBudget { max_output_tokens: options.max_sample_tokens as u32,
            max_output_bytes: program.max_output_bytes() as u64, ..request.budget };
        let ir = TaskIR::new(vec![
            PromptSegment::new(PromptSegmentKind::GlobalPolicy, compiler.global.clone()),
            PromptSegment::new(PromptSegmentKind::TaskInstruction, template.instruction.clone()),
            PromptSegment::new(PromptSegmentKind::Document, document.ids().to_vec()),
            PromptSegment::new(PromptSegmentKind::AnswerScaffold, template.scaffold.clone()),
        ], DecodeStrategy::ConstrainedJson, GrammarReference::json_schema(compiler.schema_digest, JSON_RUNTIME_VERSION),
            None, vec![FinitePostcondition::JsonValid, FinitePostcondition::MatchesGrammar, FinitePostcondition::OutputWithinBudget],
            sample_budget, DependencyScope::ItemLocal).map_err(|_| SampledSentimentError::Identity)?;
        let task = TaskPlan::new(BuiltInTask::Sentiment.spec(), &context_identity, ir).map_err(|_| SampledSentimentError::Identity)?;
        let mut prompt = reserved(length)?;
        for segment in task.ir().prompt_segments() { prompt.extend_from_slice(segment.token_ids()); }
        if prompt.len() != length { return Err(SampledSentimentError::Accounting); }
        let mut axis_identity = identity.clone();
        axis_identity.taskir_digest = task.ir().digest().map_err(|_| SampledSentimentError::Serialization)?;
        axis_identity.prompt_digest = digest(task.ir().prompt_segments())?;
        axis_identity.schema_digest = compiler.schema_digest;
        axis_identity.grammar_compiler_version = JSON_RUNTIME_VERSION.to_owned();
        axis_identity.decision_policy_digest = digest(&(SAMPLED_SENTIMENT_EXECUTION, SAMPLED_SENTIMENT_AGGREGATION_VERSION,
            INT4_JSON_SAMPLED_EXECUTION, axis, options, group_width, request.budget, program.max_output_bytes()))?;
        axis_identity.validate().map_err(|_| SampledSentimentError::Identity)?;
        axes.push(PreparedAxis { axis, prompt, identity: axis_identity, work: one });
    }
    let mut sealed = identity.clone();
    sealed.taskir_digest = digest(&axes.iter().map(|axis| (axis.axis, axis.identity.taskir_digest)).collect::<Vec<_>>())?;
    sealed.prompt_digest = digest(&axes.iter().map(|axis| (axis.axis, axis.identity.prompt_digest)).collect::<Vec<_>>())?;
    sealed.schema_digest = compiler.schema_digest; sealed.grammar_compiler_version = JSON_RUNTIME_VERSION.to_owned();
    sealed.decision_policy_digest = digest(&(SAMPLED_SENTIMENT_EXECUTION, SAMPLED_SENTIMENT_AGGREGATION_VERSION,
        options, request.budget, max_result_bytes, group_width, &sealed.taskir_digest))?;
    sealed.validate().map_err(|_| SampledSentimentError::Identity)?;
    checkpoint(control)?;
    Ok(PreparedSampledSentiment { axes, program, options: options.clone(),
        decode: compiler.decode.clone(), controls: compiler.controls.ids().clone(), identity: sealed,
        group_width, work, budget: request.budget, context, max_result_bytes, preparation_bytes })
}

fn unbound_identity(compiler: &Compiler) -> ExecutionIdentity {
    let absent = Sha256Digest::of_bytes(b"null");
    ExecutionIdentity { schema_version: 1, source_revision: "unbound".to_owned(), logical_model_digest: absent,
        artifact_format: "unbound".to_owned(), quant_recipe: "unbound".to_owned(), packing_set_digest: absent,
        tokenizer_digest: Sha256Digest::of_bytes(PINNED_TOKENIZER_MODEL_BYTES), template_digest: compiler.template_digest,
        task_spec: "sentiment-v1".to_owned(), taskir_digest: absent, prompt_digest: absent,
        grammar_compiler_version: JSON_RUNTIME_VERSION.to_owned(), schema_digest: compiler.schema_digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: JSON_SAMPLER_VERSION.to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent, backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn preparation_quote(prompt_tokens: usize, result_bytes: u64, sample_bytes: usize) -> Result<u64, SampledSentimentError> {
    // The pinned tokenizer/trie and finite grammar share the same conservative
    // 512-MiB construction envelope used by structured candidate operations.
    // Exact prompts, identity serialization and completed-result copies are
    // additional; sampler and native buffers are separate host charges.
    (prompt_tokens as u64).checked_mul(64).and_then(|n| n.checked_add(512 * MIB))
        .and_then(|n| n.checked_add(result_bytes.checked_mul(4)?))
        .and_then(|n| n.checked_add((sample_bytes as u64).checked_mul(2)?))
        .ok_or(SampledSentimentError::Limits)
}
fn fits_work(work: Int4Work, cap: Int4Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}
fn budget_fits(value: TaskBudget, ceiling: TaskBudget) -> bool {
    value.max_input_tokens <= ceiling.max_input_tokens && value.max_output_tokens <= ceiling.max_output_tokens
        && value.max_output_bytes <= ceiling.max_output_bytes && value.max_grammar_states <= ceiling.max_grammar_states
        && value.max_kv_bytes <= ceiling.max_kv_bytes
}
fn digest<T: Serialize + ?Sized>(value: &T) -> Result<Sha256Digest, SampledSentimentError> {
    canonjson::canonical_bytes(value).map(|bytes| Sha256Digest::of_bytes(&bytes)).map_err(|_| SampledSentimentError::Serialization)
}
fn reserved<T>(count: usize) -> Result<Vec<T>, SampledSentimentError> {
    let mut values = Vec::new(); values.try_reserve_exact(count).map_err(|_| SampledSentimentError::Allocation)?; Ok(values)
}
fn check_output_size<T: Serialize>(value: &T, cap: u64) -> Result<(), SampledSentimentError> {
    let bytes = canonjson::canonical_bytes(value).map_err(|_| SampledSentimentError::Serialization)?;
    if bytes.len() as u64 > cap { return Err(SampledSentimentError::OutputBudget); }
    Ok(())
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), SampledSentimentError> {
    match control.prefill_checkpoint(0) { Some(cause) => Err(SampledSentimentError::Cancelled(cause)), None => Ok(()) }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }

#[cfg(test)] mod tests;
