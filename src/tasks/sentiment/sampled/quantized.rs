//! Independent seeded score/justification generations on portable INT8 weights.
//!
//! Prompt bytes, control-free source encoding, bounded grammar, sample addresses,
//! validation and aggregation are shared with the grouped-INT4 task. Every
//! sample owns a fresh causal session and projects the complete vocabulary.
//! This is a distinct sampled estimator, never sampling distribution-mode bins.
//! No sampled quality, calibration, public-format or release claim is made here.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{
    canonjson,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest},
    native_engine::{
        artifact_bridge::ArtifactIdentity,
        constrained::JsonWorkBudget,
        constrained_int8::{self, Int8JsonBudget, Int8JsonError, Int8JsonRun,
            Int8JsonSamplingOptions, INT8_JSON_SAMPLED_EXECUTION, JSON_SAMPLER_VERSION,
            json_sampling_workspace_bytes},
        decode::{DecodeCancellationKind, DecodeStepControl},
        kv::KV_BYTES_PER_TOKEN,
        lmhead::NANBEIGE_VOCAB_SIZE,
        strict_int8::{Int8RunBudget, Int8Work, StrictInt8Engine, StrictInt8Error,
            STRICT_INT8_EXECUTION, STRICT_INT8_PROFILE, prefill::Int8PrefillLimits},
    },
    tasks::ir::TaskBudget,
    tokenizer::untrusted::{SafeBpeWork, UntrustedEncoding},
};
use super::{
    Compiler, CompiledSampledSentiment, Continue, PreparedAxis, SampledSchedule,
    SampledSentimentDimension, SampledSentimentError, SampledSentimentLimits,
    SampledSentimentOptions, SampledSentimentSample, SampledSentimentVocabulary,
    SAMPLED_SENTIMENT_AGGREGATION_VERSION, MAX_SENTIMENT_SAMPLES, SentimentAxis, SentimentRequest,
    aggregate, check_output_size, checkpoint, compile_sampled_request, digest,
    item_address_digest, reserved, unbound_identity, validate_sample_json,
};

pub mod long;

pub const INT8_SAMPLED_SENTIMENT_EXECUTION: &str = "portable-int8-independent-sampled-sentiment-v1";

/// Existing shared task failures retain their precise category. Native INT8
/// failures stay separate from the grouped-INT4 error variant and preserve
/// cancellation all the way to the existing library/CLI policy boundary.
pub enum Int8SampledSentimentError {
    Shared(SampledSentimentError),
    Native(Int8JsonError),
}
impl fmt::Display for Int8SampledSentimentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shared(error) => fmt::Display::fmt(error, f),
            Self::Native(_) => f.write_str("int8 sampled sentiment native constrained generation failed"),
        }
    }
}
impl fmt::Debug for Int8SampledSentimentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, f) }
}
impl Error for Int8SampledSentimentError {}
impl From<SampledSentimentError> for Int8SampledSentimentError {
    fn from(error: SampledSentimentError) -> Self { Self::Shared(error) }
}
impl From<Int8JsonError> for Int8SampledSentimentError {
    fn from(error: Int8JsonError) -> Self { Self::Native(error) }
}
impl From<StrictInt8Error> for Int8SampledSentimentError {
    fn from(error: StrictInt8Error) -> Self { Self::Native(Int8JsonError::Native(error)) }
}
impl Int8SampledSentimentError {
    pub fn cancellation(&self) -> Option<DecodeCancellationKind> {
        match self { Self::Shared(error) => error.cancellation(), Self::Native(error) => error.cancellation() }
    }
}

struct Int8SampledSchedule;
impl SampledSchedule for Int8SampledSchedule {
    type Work = Int8Work;
    // The numerical profile has no INT4 group-width parameter.
    type Profile = ();
    const EXECUTION: &'static str = INT8_SAMPLED_SENTIMENT_EXECUTION;
    const NATIVE_EXECUTION: &'static str = INT8_JSON_SAMPLED_EXECUTION;
    fn profile(&self) {}
    fn planned_work(&self, prompt: usize, tokens: usize) -> Result<Int8Work, SampledSentimentError> {
        constrained_int8::planned_work(prompt, tokens).map_err(|_| SampledSentimentError::WorkBudget)
    }
    fn add_work(&self, left: Int8Work, right: Int8Work) -> Result<Int8Work, SampledSentimentError> {
        left.checked_add(right).map_err(|_| SampledSentimentError::WorkBudget)
    }
}

/// Compiler bound to the facts of the held, authenticated candidate artifact.
/// It owns prompts and identity, never model weights or a second tokenizer.
pub struct Int8SampledSentimentPlanner {
    compiler: Compiler,
    identity: ExecutionIdentity,
    options: SampledSentimentOptions,
    ceiling: TaskBudget,
    limits: SampledSentimentLimits,
}
impl Int8SampledSentimentPlanner {
    pub fn for_model(model: &ArtifactIdentity, options: SampledSentimentOptions,
        ceiling: TaskBudget, limits: SampledSentimentLimits) -> Result<Self, Int8SampledSentimentError> {
        options.validate()?;
        limits.validate()?;
        ceiling.validate().map_err(|_| SampledSentimentError::Limits)?;
        let compiler = Compiler::pinned(&options, ceiling, limits)?;
        let identity = bind_model(unbound_identity(&compiler), model)?;
        Ok(Self { compiler, identity, options, ceiling, limits })
    }

    pub fn for_model_with_encoding(model: &ArtifactIdentity, options: SampledSentimentOptions,
        ceiling: TaskBudget, limits: SampledSentimentLimits, encoding: UntrustedEncoding)
        -> Result<Self, Int8SampledSentimentError> {
        encoding.validate().map_err(|_| SampledSentimentError::WorkBudget)?;
        Self::for_model(model, options, ceiling, limits)?.with_input_encoding(encoding)
    }

    /// One compact encoding serves the entire axis/sample bundle. Its declared
    /// workspace and spent work remain distinct from native projection work.
    pub fn with_input_encoding(mut self, encoding: UntrustedEncoding)
        -> Result<Self, Int8SampledSentimentError> {
        self.compiler.set_input_encoding(encoding)?;
        self.identity.template_digest = self.compiler.template_digest;
        self.identity.validate().map_err(|_| SampledSentimentError::Identity)?;
        Ok(self)
    }

    pub fn input_encoding(&self) -> &UntrustedEncoding { &self.compiler.input_encoding }
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    pub fn task_budget(&self) -> TaskBudget { self.ceiling }
    pub fn limits(&self) -> SampledSentimentLimits { self.limits }
    pub fn encoding_workspace_bytes(&self) -> Result<u64, Int8SampledSentimentError> {
        Ok(self.compiler.encoding_workspace_bytes()?)
    }
    pub fn plan(&self, request: &SentimentRequest)
        -> Result<PreparedInt8SampledSentiment, Int8SampledSentimentError> {
        self.plan_with_control(request, &mut Continue)
    }
    pub fn plan_with_control<C: DecodeStepControl>(&self, request: &SentimentRequest, control: &mut C)
        -> Result<PreparedInt8SampledSentiment, Int8SampledSentimentError> {
        self.plan_with_encoding_work(request, control, &mut SafeBpeWork::default())
    }

    /// Failed preparation retains the caller's tokenizer work. Neither axes,
    /// samples nor subsequent records silently receive a fresh allowance.
    pub fn plan_with_encoding_work<C: DecodeStepControl>(&self, request: &SentimentRequest,
        control: &mut C, work: &mut SafeBpeWork)
        -> Result<PreparedInt8SampledSentiment, Int8SampledSentimentError> {
        constrained_int8::check_profile(&self.identity)?;
        let inner = compile_sampled_request(&self.compiler, &self.identity, &Int8SampledSchedule,
            &self.options, self.ceiling, self.limits, request, control, work)?;
        Ok(PreparedInt8SampledSentiment { inner })
    }

    /// Optional stable population addressing excludes ordering/flush epochs.
    /// It is private semantic identity, never an output field or model prompt.
    pub fn plan_for_item_with_control<C: DecodeStepControl>(&self, request: &SentimentRequest,
        item_id: &str, control: &mut C) -> Result<PreparedInt8SampledSentiment, Int8SampledSentimentError> {
        checkpoint(control)?;
        if item_id.is_empty() || item_id.len() > 256 || item_id.chars().any(char::is_control) {
            return Err(SampledSentimentError::Limits.into());
        }
        let mut prepared = self.plan_with_control(request, control)?;
        for axis in &mut prepared.inner.axes {
            axis.identity.decision_policy_digest = item_address_digest(&axis.identity, item_id)?;
            axis.identity.validate().map_err(|_| SampledSentimentError::Identity)?;
        }
        prepared.inner.identity.decision_policy_digest = item_address_digest(&prepared.inner.identity, item_id)?;
        prepared.inner.identity.validate().map_err(|_| SampledSentimentError::Identity)?;
        checkpoint(control)?;
        Ok(prepared)
    }
}

pub type Int8SampledSentimentSample = SampledSentimentSample<Int8Work>;
pub type Int8SampledSentimentDimension = SampledSentimentDimension<Int8Work>;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Int8SampledSentimentRun {
    pub schema_version: u32,
    pub execution: String,
    pub task_spec: String,
    pub prompt_template_sha256: Sha256Digest,
    pub numerics_profile: String,
    pub aggregation: String,
    pub interpretation: String,
    pub justification_status: String,
    pub sampler_version: String,
    pub samples_per_axis: usize,
    pub dimensions: Vec<Int8SampledSentimentDimension>,
    pub model_work: Int8Work,
    pub generated_tokens: u64,
    pub mask_node_visit_charge: u64,
}

/// Sealed exact prompts, native schedules and sample policy. Public callers
/// cannot construct or deserialize this capability from a result envelope.
pub struct PreparedInt8SampledSentiment { inner: CompiledSampledSentiment<Int8Work> }
impl PreparedInt8SampledSentiment {
    pub fn execution_identity(&self) -> &ExecutionIdentity { &self.inner.identity }
    pub fn planned_work(&self) -> Int8Work { self.inner.work }
    pub fn required_context(&self) -> usize { self.inner.context }
    pub fn task_budget(&self) -> TaskBudget { self.inner.budget }
    pub fn max_result_bytes(&self) -> u64 { self.inner.max_result_bytes }
    pub fn sample_count(&self) -> usize { self.inner.axes.len() * self.inner.options.samples }
    pub fn max_output_tokens(&self) -> u64 {
        self.sample_count() as u64 * self.inner.options.max_sample_tokens as u64
    }
    pub fn input_encoding(&self) -> &UntrustedEncoding { &self.inner.input_encoding }
    pub fn encoding_work(&self) -> SafeBpeWork { self.inner.encoding_work }
    pub fn preparation_requirement_bytes(&self) -> u64 { self.inner.preparation_bytes }

    pub fn verify_identity(&self, admitted: &ExecutionIdentity) -> Result<(), Int8SampledSentimentError> {
        admitted.validate().map_err(|_| SampledSentimentError::Identity)?;
        if admitted != self.execution_identity() { return Err(SampledSentimentError::Identity.into()); }
        Ok(())
    }

    /// Admit all independent samples before opening the first native session.
    /// The returned value is a bound, not measured trie traversal.
    pub fn complete_mask_charge(&self, vocabulary: &SampledSentimentVocabulary,
        budget: Int8JsonBudget, max_sampler_bytes: u64) -> Result<u64, Int8SampledSentimentError> {
        self.preflight(vocabulary, budget, max_sampler_bytes)?.checked_mul(self.sample_count() as u64)
            .ok_or_else(|| SampledSentimentError::WorkBudget.into())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_control<C: DecodeStepControl>(&self, engine: &mut StrictInt8Engine<'_>,
        admitted: &ExecutionIdentity, vocabulary: &SampledSentimentVocabulary,
        budget: Int8JsonBudget, max_sampler_bytes: u64, control: &mut C)
        -> Result<Int8SampledSentimentRun, Int8SampledSentimentError> {
        self.execute_scheduled(engine, admitted, vocabulary, budget, max_sampler_bytes, None, control)
    }

    /// The extra causal prompt buffers are separately admitted by the host.
    /// Morsel width does not enter identities, sample addresses or work limits.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_layer_major_with_control<C: DecodeStepControl>(&self, engine: &mut StrictInt8Engine<'_>,
        admitted: &ExecutionIdentity, vocabulary: &SampledSentimentVocabulary,
        budget: Int8JsonBudget, max_sampler_bytes: u64, prefill: Int8PrefillLimits, control: &mut C)
        -> Result<Int8SampledSentimentRun, Int8SampledSentimentError> {
        self.execute_scheduled(engine, admitted, vocabulary, budget, max_sampler_bytes, Some(prefill), control)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_scheduled<C: DecodeStepControl>(&self, engine: &mut StrictInt8Engine<'_>,
        admitted: &ExecutionIdentity, vocabulary: &SampledSentimentVocabulary,
        budget: Int8JsonBudget, max_sampler_bytes: u64, prefill: Option<Int8PrefillLimits>, control: &mut C)
        -> Result<Int8SampledSentimentRun, Int8SampledSentimentError> {
        checkpoint(control)?;
        if let Some(limits) = prefill { limits.validate()?; }
        self.verify_identity(admitted)?;
        check_model(self.execution_identity(), engine.artifact_identity())?;
        if engine.profile() != STRICT_INT8_PROFILE || engine.is_poisoned()
            || !engine.kv_cache().all_slots_have_len(0) || vocabulary.controls != self.inner.controls {
            return Err(SampledSentimentError::Identity.into());
        }
        let capacity = engine.kv_cache().capacity_positions();
        let resident_kv = (capacity as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
            .ok_or(SampledSentimentError::WorkBudget)?;
        if capacity < self.required_context() || resident_kv > budget.json.max_kv_bytes {
            return Err(SampledSentimentError::WorkBudget.into());
        }
        let mask_charge = self.preflight(vocabulary, budget, max_sampler_bytes)?;
        let mut dimensions = reserved(self.inner.axes.len())?;
        let mut actual_work = Int8Work::default();
        let mut actual_mask = 0_u64;
        let mut actual_tokens = 0_u64;
        let mut retained_sample_bytes = 0_u64;
        for axis in &self.inner.axes {
            let mut samples = reserved(self.inner.options.samples)?;
            for sample_index in 0..self.inner.options.samples {
                checkpoint(control)?;
                let sampling = sampling(&self.inner.options, axis.axis, sample_index)?;
                let sample_budget = Int8JsonBudget { native: Int8RunBudget::exact(axis.work),
                    json: JsonWorkBudget { max_forward_positions: axis.work.forward_positions,
                        max_projected_logits: axis.work.projected_logits, max_kv_bytes: budget.json.max_kv_bytes,
                        max_total_mask_node_visits: mask_charge, mask_limits: budget.json.mask_limits } };
                // Content validation and retained-output charging occur inside
                // the session lifetime, so refusal poisons/cleans that session.
                let finalize = |run| -> Result<Int8SampledSentimentSample, Int8SampledSentimentError> {
                    let sample = self.finalize_sample(axis, sample_index, run)?;
                    let bytes = canonjson::canonical_bytes(&sample).map_err(|_| SampledSentimentError::Serialization)?;
                    retained_sample_bytes = retained_sample_bytes.checked_add(bytes.len() as u64)
                        .filter(|&n| n <= self.max_result_bytes()).ok_or(SampledSentimentError::OutputBudget)?;
                    Ok(sample)
                };
                let sample = match prefill {
                    None => constrained_int8::decode_json_int8_sampled_with(engine, &axis.identity,
                        &axis.prompt, &self.inner.program, &vocabulary.oracle, &self.inner.decode,
                        &sampling, sample_budget, max_sampler_bytes, control, finalize),
                    Some(limits) => constrained_int8::decode_json_int8_sampled_layer_major_with(engine,
                        &axis.identity, &axis.prompt, &self.inner.program, &vocabulary.oracle,
                        &self.inner.decode, &sampling, sample_budget, max_sampler_bytes, limits, control, finalize),
                }?;
                if engine.is_poisoned() || !engine.kv_cache().all_slots_have_len(0) {
                    return Err(SampledSentimentError::Accounting.into());
                }
                actual_work = actual_work.checked_add(sample.model_work)?;
                actual_mask = actual_mask.checked_add(sample.mask_node_visit_charge)
                    .ok_or(SampledSentimentError::Accounting)?;
                actual_tokens = actual_tokens.checked_add(sample.generated_tokens as u64)
                    .ok_or(SampledSentimentError::Accounting)?;
                samples.push(sample);
                checkpoint(control)?;
            }
            dimensions.push(aggregate(axis.axis, samples)?);
        }
        let run = Int8SampledSentimentRun { schema_version: 1,
            execution: INT8_SAMPLED_SENTIMENT_EXECUTION.to_owned(), task_spec: "sentiment-v1".to_owned(),
            prompt_template_sha256: self.inner.identity.template_digest,
            numerics_profile: STRICT_INT8_PROFILE.to_owned(),
            aggregation: SAMPLED_SENTIMENT_AGGREGATION_VERSION.to_owned(),
            interpretation: "generated_score_variation_not_calibrated_confidence".to_owned(),
            justification_status: "unverified_model_explanations_from_untrusted_document".to_owned(),
            sampler_version: JSON_SAMPLER_VERSION.to_owned(), samples_per_axis: self.inner.options.samples,
            dimensions, model_work: actual_work, generated_tokens: actual_tokens, mask_node_visit_charge: actual_mask };
        let per_token = vocabulary.oracle.trie().node_count().min(budget.json.mask_limits.max_trie_node_visits) as u64;
        self.verify_completed_with_control(&run, per_token, budget.json.max_total_mask_node_visits, control)?;
        Ok(run)
    }

    fn preflight(&self, vocabulary: &SampledSentimentVocabulary, budget: Int8JsonBudget,
        max_sampler_bytes: u64) -> Result<u64, Int8SampledSentimentError> {
        if vocabulary.controls != self.inner.controls || vocabulary.oracle.trie().vocab_size() != NANBEIGE_VOCAB_SIZE {
            return Err(SampledSentimentError::Identity.into());
        }
        let per_token = vocabulary.oracle.trie().node_count().min(budget.json.mask_limits.max_trie_node_visits) as u64;
        self.preflight_bounds(per_token, budget, max_sampler_bytes)
    }

    fn preflight_bounds(&self, per_token: u64, budget: Int8JsonBudget, max_sampler_bytes: u64)
        -> Result<u64, Int8SampledSentimentError> {
        let work = self.planned_work();
        let required_kv = (self.required_context() as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
            .ok_or(SampledSentimentError::WorkBudget)?;
        if per_token == 0 || work.forward_positions > budget.native.max_forward_positions
            || work.attention_pairs > budget.native.max_attention_pairs
            || !work.projections.fits(budget.native.max_projection_work)
            || work.forward_positions > budget.json.max_forward_positions
            || work.projected_logits > budget.json.max_projected_logits
            || required_kv > budget.json.max_kv_bytes || budget.json.max_kv_bytes > self.task_budget().max_kv_bytes
            || max_sampler_bytes < json_sampling_workspace_bytes()?
            || budget.json.mask_limits.max_trie_node_visits == 0
            || budget.json.mask_limits.checkpoint_interval_nodes == 0 {
            return Err(SampledSentimentError::WorkBudget.into());
        }
        let one = per_token.checked_mul(self.inner.options.max_sample_tokens as u64)
            .ok_or(SampledSentimentError::WorkBudget)?;
        let all = one.checked_mul(self.sample_count() as u64).ok_or(SampledSentimentError::WorkBudget)?;
        if all > budget.json.max_total_mask_node_visits { return Err(SampledSentimentError::WorkBudget.into()); }
        Ok(one)
    }

    fn finalize_sample(&self, axis: &PreparedAxis<Int8Work>, sample_index: usize, run: Int8JsonRun)
        -> Result<Int8SampledSentimentSample, Int8SampledSentimentError> {
        let output = &run.output;
        if sample_index >= self.inner.options.samples || run.schema_version != 1
            || run.execution != INT8_JSON_SAMPLED_EXECUTION || output.schema_version != 1
            || output.numerics_profile != STRICT_INT8_PROFILE || output.token_ids.is_empty()
            || output.token_ids.len() > self.inner.options.max_sample_tokens
            || output.token_ids.last() != Some(&self.inner.decode.eos_token_id)
            || output.token_ids[..output.token_ids.len() - 1].iter().any(|id|
                *id as usize >= NANBEIGE_VOCAB_SIZE || self.inner.decode.excluded_token_ids.contains(id)) {
            return Err(SampledSentimentError::Accounting.into());
        }
        let expected = constrained_int8::planned_work(axis.prompt.len(), output.token_ids.len())?;
        if run.model_work != expected || !fits_work(expected, axis.work)
            || output.forward_positions != expected.forward_positions || output.projected_logits != expected.projected_logits {
            return Err(SampledSentimentError::Accounting.into());
        }
        let value = validate_sample_json(&self.inner.program, &self.inner.options, &output.json)?;
        Ok(SampledSentimentSample { sample_index, score: value.score, justification: value.justification,
            generated_tokens: output.token_ids.len(), model_work: run.model_work,
            mask_node_visit_charge: output.mask_node_visit_charge })
    }

    fn verify_completed_with_control<C: DecodeStepControl>(&self, run: &Int8SampledSentimentRun,
        mask_charge_per_token: u64, mask_ceiling: u64, control: &mut C)
        -> Result<(), Int8SampledSentimentError> {
        checkpoint(control)?;
        if mask_charge_per_token == 0 || run.schema_version != 1 || run.execution != INT8_SAMPLED_SENTIMENT_EXECUTION
            || run.task_spec != "sentiment-v1" || run.prompt_template_sha256 != self.inner.identity.template_digest
            || run.numerics_profile != STRICT_INT8_PROFILE || run.aggregation != SAMPLED_SENTIMENT_AGGREGATION_VERSION
            || run.interpretation != "generated_score_variation_not_calibrated_confidence"
            || run.justification_status != "unverified_model_explanations_from_untrusted_document"
            || run.sampler_version != JSON_SAMPLER_VERSION || run.samples_per_axis != self.inner.options.samples
            || run.dimensions.len() != self.inner.axes.len() {
            return Err(SampledSentimentError::Accounting.into());
        }
        let mut work = Int8Work::default();
        let mut tokens = 0_u64;
        let mut masks = 0_u64;
        for (axis, dimension) in self.inner.axes.iter().zip(&run.dimensions) {
            checkpoint(control)?;
            if dimension.axis != axis.axis || dimension.samples.len() != self.inner.options.samples {
                return Err(SampledSentimentError::Accounting.into());
            }
            for (index, sample) in dimension.samples.iter().enumerate() {
                checkpoint(control)?;
                if sample.sample_index != index || !(-100..=100).contains(&sample.score)
                    || sample.justification.trim().is_empty()
                    || sample.justification.chars().count() > self.inner.options.max_justification_chars
                    || sample.generated_tokens == 0 || sample.generated_tokens > self.inner.options.max_sample_tokens {
                    return Err(SampledSentimentError::Accounting.into());
                }
                let expected = constrained_int8::planned_work(axis.prompt.len(), sample.generated_tokens)?;
                let expected_masks = (sample.generated_tokens as u64).checked_mul(mask_charge_per_token)
                    .ok_or(SampledSentimentError::Accounting)?;
                if sample.model_work != expected || !fits_work(expected, axis.work)
                    || sample.mask_node_visit_charge != expected_masks {
                    return Err(SampledSentimentError::Accounting.into());
                }
                work = work.checked_add(expected)?;
                tokens = tokens.checked_add(sample.generated_tokens as u64).ok_or(SampledSentimentError::Accounting)?;
                masks = masks.checked_add(expected_masks).ok_or(SampledSentimentError::Accounting)?;
            }
            if aggregate(axis.axis, dimension.samples.clone())? != *dimension {
                return Err(SampledSentimentError::Accounting.into());
            }
        }
        if work != run.model_work || tokens != run.generated_tokens || masks != run.mask_node_visit_charge
            || !fits_work(work, self.planned_work()) || tokens > u64::from(self.task_budget().max_output_tokens)
            || masks > mask_ceiling {
            return Err(SampledSentimentError::Accounting.into());
        }
        check_output_size(run, self.max_result_bytes())?;
        checkpoint(control)?;
        Ok(())
    }
}

fn sampling(options: &SampledSentimentOptions, axis: SentimentAxis, sample: usize)
    -> Result<Int8JsonSamplingOptions, Int8SampledSentimentError> {
    if sample >= options.samples || sample >= MAX_SENTIMENT_SAMPLES {
        return Err(SampledSentimentError::Accounting.into());
    }
    let sampling = Int8JsonSamplingOptions { effective_seed: options.effective_seed,
        temperature_milli: options.temperature_milli, top_k: options.top_k, top_p_ppm: options.top_p_ppm,
        sample_index: axis as u64 * MAX_SENTIMENT_SAMPLES as u64 + sample as u64 };
    sampling.validate()?;
    Ok(sampling)
}

fn bind_model(mut identity: ExecutionIdentity, model: &ArtifactIdentity)
    -> Result<ExecutionIdentity, Int8SampledSentimentError> {
    if model.model_id != "Nanbeige4.2-3B" || model.revision != "f56ec5a9650268aa098496734743c25ea778bd2d"
        || model.recipe_id.is_empty() || model.recipe_id.len() > 256 {
        return Err(SampledSentimentError::Identity.into());
    }
    Sha256Digest::from_hex(&model.source_root_sha256).map_err(|_| SampledSentimentError::Identity)?;
    identity.source_revision = model.revision.clone();
    identity.logical_model_digest = Sha256Digest::from_hex(&model.logical_model_sha256)
        .map_err(|_| SampledSentimentError::Identity)?;
    identity.quant_recipe = model.recipe_id.clone();
    identity.artifact_format = "fnlpq-current-candidate-v1".to_owned();
    // Identical portable layout/empty sidecar vocabulary to the checked
    // candidate CLI. This does not invent a file digest or active catalog.
    identity.packing_set_digest = digest(&("current-candidate-portable-int8-v1", Vec::<String>::new()))?;
    identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 };
    identity.backend_semantic_version = STRICT_INT8_EXECUTION.to_owned();
    constrained_int8::check_profile(&identity)?;
    check_model(&identity, model)?;
    Ok(identity)
}

fn check_model(identity: &ExecutionIdentity, model: &ArtifactIdentity) -> Result<(), Int8SampledSentimentError> {
    constrained_int8::check_profile(identity)?;
    if model.model_id != "Nanbeige4.2-3B" || model.revision != identity.source_revision
        || model.recipe_id != identity.quant_recipe
        || Sha256Digest::from_hex(&model.logical_model_sha256).ok() != Some(identity.logical_model_digest) {
        return Err(SampledSentimentError::Identity.into());
    }
    Ok(())
}

fn fits_work(work: Int8Work, cap: Int8Work) -> bool {
    work.forward_positions <= cap.forward_positions && work.projected_logits <= cap.projected_logits
        && work.attention_pairs <= cap.attention_pairs && work.projections.fits(cap.projections)
}

#[cfg(test)] mod tests;
