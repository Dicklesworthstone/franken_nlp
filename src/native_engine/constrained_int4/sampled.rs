//! Addressed sampling under the existing exact JSON/source-language mask.
//!
//! Full-vocabulary defaults and an explicit complete-legal-row candidate share
//! one addressed sampler. Neither claims calibrated scores or explanations;
//! no path samples from guessed/pruned masks or partially completed JSON.

use super::*;
use sha2::{Digest, Sha256};
use crate::native_engine::{
    generation::{GenerationError, GenerationOptions, GenerationSampling, policy},
    sampler::{MAX_TOP_K, SAMPLER_VERSION, StableRequestKey},
};
mod selected;
pub use selected::{INT4_JSON_SAMPLED_SPARSE_EXECUTION, INT4_JSON_SAMPLED_SPARSE_REUSED_PREFILL_EXECUTION,
    decode_json_int4_sampled_sparse, decode_json_int4_sampled_sparse_layer_major};
pub(crate) use selected::{decode_json_int4_sampled_sparse_with,
    decode_json_int4_sampled_sparse_layer_major_with, decode_json_int4_sampled_sparse_reusing_prefill_with};

pub const INT4_JSON_SAMPLED_EXECUTION: &str = "grouped-int4-sampled-constrained-json-final-prefill-head-v1";
/// Explicit physical schedule: only the first sample prefills the complete
/// prompt. Later independent samples replay its final token from the retained
/// same-session prefix. Draw addresses still use INT4_JSON_SAMPLED_EXECUTION.
pub const INT4_JSON_SAMPLED_REUSED_PREFILL_EXECUTION: &str =
    "grouped-int4-sampled-constrained-json-reused-prefill-v1";
/// Grammar legality precedes temperature, bounded top-k and exact nucleus.
/// The addressed draw itself uses the pinned `fnlp-sampler-v1` primitive.
pub const JSON_SAMPLER_VERSION: &str = "fnlp-json-mask-temperature-k-p-addressed-v1";

const MAX_IDENTITY_TEXT_BYTES: usize = 4096;
const IDENTITY_WORKSPACE_BYTES: u64 = 128 * 1024;
const MAX_REUSED_PREFILL_SAMPLES: usize = 32;

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

/// Run the exact sampled decoder after bounded causal layer-major prompt
/// morsels. The caller admits the additional workspace before execution.
/// Prompt scheduling never enters the private draw key or changes the sample
/// address, grammar mask, final-prefill head or scored terminal EOS.
#[allow(clippy::too_many_arguments)]
pub fn decode_json_int4_sampled_layer_major<C: DecodeStepControl>(
    engine: &mut StrictInt4Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    sampling: &JsonSamplingOptions, budget: Int4JsonBudget, max_sampler_bytes: u64,
    prefill: Int4PrefillLimits, control: &mut C,
) -> Result<Int4JsonRun, Int4JsonError> {
    decode_json_int4_sampled_layer_major_with(engine, identity, prompt, program, vocabulary,
        options, sampling, budget, max_sampler_bytes, prefill, control, Ok)
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
    decode_json_int4_sampled_scheduled(engine, identity, prompt, program, vocabulary,
        options, sampling, budget, max_sampler_bytes, None, control, finalize)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_json_int4_sampled_layer_major_with<C, T, E, F>(
    engine: &mut StrictInt4Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    sampling: &JsonSamplingOptions, budget: Int4JsonBudget, max_sampler_bytes: u64,
    prefill: Int4PrefillLimits, control: &mut C, finalize: F,
) -> Result<T, E>
where C: DecodeStepControl, E: From<Int4JsonError>, F: FnOnce(Int4JsonRun) -> Result<T, E> {
    decode_json_int4_sampled_scheduled(engine, identity, prompt, program, vocabulary,
        options, sampling, budget, max_sampler_bytes, Some(prefill), control, finalize)
}

#[allow(clippy::too_many_arguments)]
fn decode_json_int4_sampled_scheduled<C, T, E, F>(
    engine: &mut StrictInt4Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    sampling: &JsonSamplingOptions, budget: Int4JsonBudget, max_sampler_bytes: u64,
    prefill: Option<Int4PrefillLimits>, control: &mut C, finalize: F,
) -> Result<T, E>
where C: DecodeStepControl, E: From<Int4JsonError>, F: FnOnce(Int4JsonRun) -> Result<T, E> {
    if let Some(limits) = prefill { limits.validate().map_err(Int4JsonError::from).map_err(E::from)?; }
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
    match prefill {
        Some(limits) => prefill::drive_layer_major_selected(group_width, prompt, program, vocabulary,
            options, budget, &mut session, &mut selection, limits, finalize),
        None => drive_selected(group_width, prompt, program, vocabulary, options, budget,
            &mut session, &mut selection, finalize),
    }
}

/// Execute independently addressed samples with one exclusive causal cache.
/// Only this crate's sealed task composition may use the replay seam. Its
/// complete cold-work admission remains a valid conservative ceiling; this
/// function also checks the exact reused maximum before opening the session.
///
/// Every returned run reports ACTUAL work. The first sample forwards the full
/// prompt, while later samples retain prompt_len-1 completed positions and
/// reforward the final token before selecting anything. No hidden-state/KV
/// copy, fork, new native budget, cross-axis cache or cross-request cache exists.
/// All samples keep the exact original prompt/identity/address for sampling.
/// The finalizer runs inside the live session, and any error poisons it before
/// Drop clears all 44 slots. Already executed work is never refunded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_json_int4_sampled_reusing_prefill_with<C, T, E, F>(
    engine: &mut StrictInt4Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    sampling: &[JsonSamplingOptions], budget: Int4JsonBudget, max_sampler_bytes: u64,
    prefill: Option<Int4PrefillLimits>, control: &mut C, finalize: F,
) -> Result<Vec<T>, E>
where C: DecodeStepControl, E: From<Int4JsonError>,
    F: FnMut(usize, Int4JsonRun) -> Result<T, E> {
    if let Some(limits) = prefill { limits.validate().map_err(Int4JsonError::from).map_err(E::from)?; }
    check_reused_sampling(sampling, max_sampler_bytes).map_err(E::from)?;
    check_request(engine, identity, prompt, vocabulary, options, budget).map_err(E::from)?;
    let group_width = engine.model_identity().group_width();
    repeated_preflight(group_width, prompt, vocabulary, options, sampling.len(), budget)
        .map_err(E::from)?;
    if let Some(cause) = control.checkpoint(0) {
        return Err(E::from(Int4JsonError::Decode(JsonDecodeError::Cancelled(cause))));
    }
    let mut session = engine.session(budget.native, control).map_err(Int4JsonError::from).map_err(E::from)?;
    drive_reused_prefill(group_width, identity, prompt, program, vocabulary, options,
        sampling, budget, max_sampler_bytes, prefill, &mut session, finalize)
}

fn check_reused_sampling(sampling: &[JsonSamplingOptions], max_sampler_bytes: u64)
    -> Result<(), Int4JsonError> {
    if sampling.is_empty() || sampling.len() > MAX_REUSED_PREFILL_SAMPLES {
        return Err(JsonDecodeError::InvalidRequest("reused-prefill sample count").into());
    }
    for sample in sampling { sample.validate()?; }
    check_sampler_budget(max_sampler_bytes)
}

fn repeated_preflight<V: Vocabulary>(group_width: usize, prompt: &[u32], vocabulary: &V,
    options: &JsonDecodeOptions, samples: usize, budget: Int4JsonBudget)
    -> Result<Int4Work, Int4JsonError> {
    if samples == 0 || samples > MAX_REUSED_PREFILL_SAMPLES {
        return Err(JsonDecodeError::InvalidRequest("reused-prefill sample count").into());
    }
    let cold = preflight(group_width, prompt, vocabulary, options, budget)?;
    let projected = options.max_new_tokens.checked_mul(vocabulary.width()).ok_or(StrictInt4Error::Work)?;
    // A later sample performs the final prompt position and each generated
    // non-EOS token. Attention still sees every retained earlier position.
    let warm = Int4Work::for_sequence(group_width, prompt.len() - 1, options.max_new_tokens, projected)?;
    let mut complete = cold;
    for _ in 1..samples { complete = complete.checked_add(warm)?; }
    let mask_charge = (vocabulary.mask_charge(budget.json.mask_limits) as u64)
        .checked_mul(options.max_new_tokens as u64)
        .and_then(|one| one.checked_mul(samples as u64)).ok_or(StrictInt4Error::Work)?;
    if complete.forward_positions > budget.native.max_forward_positions
        || complete.attention_pairs > budget.native.max_attention_pairs
        || !complete.projections.fits(budget.native.max_projection_work)
        || complete.forward_positions > budget.json.max_forward_positions
        || complete.projected_logits > budget.json.max_projected_logits
        || mask_charge > budget.json.max_total_mask_node_visits {
        return Err(JsonDecodeError::BudgetExceeded("complete reused-prefill work").into());
    }
    Ok(complete)
}

// This seam is private and cannot be implemented by a product caller. A real
// session keeps its native ledger, model, controller and cache owner together.
trait ReplayDriver: prefill::PrefillDriver {
    fn rewind_for_replay(&mut self, retain: usize) -> Result<(), Int4JsonError>;
}
impl<C: DecodeStepControl> ReplayDriver for Int4Session<'_, '_, C> {
    fn rewind_for_replay(&mut self, retain: usize) -> Result<(), Int4JsonError> {
        Ok(Int4Session::rewind_for_replay(self, retain)?)
    }
}

struct PromptReplay<'a, D> {
    inner: &'a mut D, baseline: Int4Work, retained: usize, replay: bool,
    prefill: Option<Int4PrefillLimits>,
}
impl<D: ReplayDriver> Driver for PromptReplay<'_, D> {
    type Control = D::Control;
    fn control(&mut self) -> &mut Self::Control { self.inner.control() }
    fn append(&mut self, token: u32) -> Result<(), Int4JsonError> { self.inner.append(token) }
    fn append_prompt(&mut self, prompt: &[u32]) -> Result<(), Int4JsonError> {
        if self.retained >= prompt.len() || (!self.replay && self.retained != 0)
            || (self.replay && self.retained + 1 != prompt.len()) {
            return Err(Int4JsonError::WorkMismatch);
        }
        if self.replay {
            self.inner.rewind_for_replay(self.retained)?;
            if let Some(cause) = self.inner.control().prefill_checkpoint(self.retained) {
                return Err(JsonDecodeError::Cancelled(cause).into());
            }
            self.inner.append(prompt[self.retained])
        } else {
            match self.prefill {
                Some(limits) => self.inner.append_layer_major(prompt, limits),
                None => self.inner.append_prompt(prompt),
            }
        }
    }
    fn logits(&mut self) -> Result<Vec<f32>, Int4JsonError> { self.inner.logits() }
    fn work(&self) -> Result<Int4Work, Int4JsonError> { work_delta(self.inner.work()?, self.baseline) }
    fn retained_prompt_positions(&self) -> usize { self.retained }
    fn abort(&mut self) { self.inner.abort(); }
}

fn work_delta(total: Int4Work, before: Int4Work) -> Result<Int4Work, Int4JsonError> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or(Int4JsonError::WorkMismatch);
    Ok(Int4Work { forward_positions: sub(total.forward_positions, before.forward_positions)?,
        projected_logits: sub(total.projected_logits, before.projected_logits)?,
        attention_pairs: sub(total.attention_pairs, before.attention_pairs)?,
        projections: super::super::portable_int8::ProjectionWork {
            dot_products: sub(total.projections.dot_products, before.projections.dot_products)?,
            multiply_accumulates: sub(total.projections.multiply_accumulates, before.projections.multiply_accumulates)?,
        } })
}

struct ReusedSelection(Sampled);
impl Selection for ReusedSelection {
    fn select<C: DecodeStepControl>(&mut self, logits: &[f32], mask: &DenseTokenMask, accepting: bool,
        options: &JsonDecodeOptions, step: usize, control: &mut C) -> Result<u32, Int4JsonError> {
        self.0.select(logits, mask, accepting, options, step, control)
    }
    fn execution(&self) -> &'static str { INT4_JSON_SAMPLED_REUSED_PREFILL_EXECUTION }
}

#[allow(clippy::too_many_arguments)]
fn drive_reused_prefill<V, D, T, E, F>(group_width: usize, identity: &ExecutionIdentity,
    prompt: &[u32], program: &JsonProgram, vocabulary: &V, options: &JsonDecodeOptions,
    sampling: &[JsonSamplingOptions], budget: Int4JsonBudget, max_sampler_bytes: u64,
    prefill: Option<Int4PrefillLimits>, driver: &mut D, mut finalize: F) -> Result<Vec<T>, E>
where V: Vocabulary, D: ReplayDriver, E: From<Int4JsonError>,
    F: FnMut(usize, Int4JsonRun) -> Result<T, E> {
    // Validate the complete operation before the first forward, not one fresh
    // independent allowance for every sample. The native driver never resets.
    check_reused_sampling(sampling, max_sampler_bytes).map_err(E::from)?;
    if let Some(limits) = prefill { limits.validate().map_err(Int4JsonError::from).map_err(E::from)?; }
    let complete = repeated_preflight(group_width, prompt, vocabulary, options, sampling.len(), budget)
        .map_err(E::from)?;
    let result = (|| {
        if driver.work().map_err(E::from)? != Int4Work::default() {
            return Err(E::from(Int4JsonError::WorkMismatch));
        }
        let mut output = Vec::new();
        output.try_reserve_exact(sampling.len())
            .map_err(|_| E::from(Int4JsonError::Decode(JsonDecodeError::AllocationRefused)))?;
        let mut work = Int4Work::default();
        let mut masks = 0_u64;
        for (index, sample) in sampling.iter().enumerate() {
            if let Some(cause) = driver.control().checkpoint(0) {
                return Err(E::from(Int4JsonError::Decode(JsonDecodeError::Cancelled(cause))));
            }
            let mut selection = ReusedSelection(Sampled::new(identity, prompt, program, options,
                sample, max_sampler_bytes).map_err(E::from)?);
            let baseline = driver.work().map_err(E::from)?;
            let mut replay = PromptReplay { inner: &mut *driver, baseline,
                retained: if index == 0 { 0 } else { prompt.len() - 1 }, replay: index != 0, prefill };
            let checked = drive_selected(group_width, prompt, program, vocabulary, options,
                budget, &mut replay, &mut selection, |run| {
                    work = work.checked_add(run.model_work).map_err(Int4JsonError::from).map_err(E::from)?;
                    masks = masks.checked_add(run.output.mask_node_visit_charge)
                        .filter(|&charge| charge <= budget.json.max_total_mask_node_visits)
                        .ok_or_else(|| E::from(Int4JsonError::WorkMismatch))?;
                    finalize(index, run)
                })?;
            output.push(checked);
        }
        if driver.work().map_err(E::from)? != work || work.forward_positions > complete.forward_positions
            || work.projected_logits > complete.projected_logits || work.attention_pairs > complete.attention_pairs
            || !work.projections.fits(complete.projections) {
            return Err(E::from(Int4JsonError::WorkMismatch));
        }
        if let Some(cause) = driver.control().checkpoint(0) {
            return Err(E::from(Int4JsonError::Decode(JsonDecodeError::Cancelled(cause))));
        }
        Ok(output)
    })();
    if result.is_err() { driver.abort(); }
    result
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
    fn select<C: DecodeStepControl>(&mut self, logits: &[f32], mask: &DenseTokenMask, accepting: bool,
        options: &JsonDecodeOptions, step: usize, control: &mut C) -> Result<u32, Int4JsonError> {
        if logits.len() != mask.vocab_size() { return Err(JsonDecodeError::InvalidLogits.into()); }
        self.workspace.select_legal_with_control(logits, &self.options, self.key, self.sample_index, step, |id| {
            if id == options.eos_token_id { accepting }
            else { mask.contains(id) && !options.excluded_token_ids.contains(&id) }
        }, control).map_err(sampling_error)
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
