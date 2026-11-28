//! Exact grammar-conditioned sampling from every legal output row.
//!
//! Projection policy never changes the semantic request key or draw address.
//! The explicit row ceiling refuses a dense state; it is not top-k pruning,
//! an automatic routing threshold or a measured performance claim. Illegal
//! logits are not computed, so no full-head diagnostics are implied.
use super::*;
use super::reuse::{check_sample_count, check_sampling, work_delta};
use crate::native_engine::constrained_int8::sparse::{self as selected_head, Int8JsonSparseLimits};

pub const INT8_JSON_SAMPLED_SPARSE_EXECUTION: &str = "portable-int8-sampled-grammar-first-selected-rows-v1";
pub const INT8_JSON_SAMPLED_SPARSE_REUSED_PREFILL_EXECUTION: &str =
    "portable-int8-sampled-grammar-first-selected-rows-reused-prefill-v1";

/// Project the complete legal set, including accepting EOS, then use the same
/// temperature/top-k/exact-nucleus sampler as the full-head reference. The
/// original semantic identity is used for draws; task owners separately bind
/// this execution profile and its cap into their admission and durable recipe.
/// Full native and sampler storage remains required.
#[allow(clippy::too_many_arguments)]
pub fn decode_json_int8_sampled_sparse<C: DecodeStepControl>(engine: &mut StrictInt8Engine<'_>,
    identity: &ExecutionIdentity, prompt: &[u32], program: &JsonProgram, vocabulary: &VocabMaskOracle,
    options: &JsonDecodeOptions, sampling: &Int8JsonSamplingOptions, budget: Int8JsonBudget,
    max_sampler_bytes: u64, limits: Int8JsonSparseLimits, control: &mut C)
    -> Result<Int8JsonRun, Int8JsonError> {
    decode_json_int8_sampled_sparse_with(engine, identity, prompt, program, vocabulary,
        options, sampling, budget, max_sampler_bytes, limits, control, Ok)
}

/// Bounded causal prompt morsels with the identical selected-row decoder.
/// The host must independently retain the additional prefill workspace.
#[allow(clippy::too_many_arguments)]
pub fn decode_json_int8_sampled_sparse_layer_major<C: DecodeStepControl>(engine: &mut StrictInt8Engine<'_>,
    identity: &ExecutionIdentity, prompt: &[u32], program: &JsonProgram, vocabulary: &VocabMaskOracle,
    options: &JsonDecodeOptions, sampling: &Int8JsonSamplingOptions, budget: Int8JsonBudget,
    max_sampler_bytes: u64, limits: Int8JsonSparseLimits, prefill: Int8PrefillLimits, control: &mut C)
    -> Result<Int8JsonRun, Int8JsonError> {
    decode_json_int8_sampled_sparse_layer_major_with(engine, identity, prompt, program, vocabulary,
        options, sampling, budget, max_sampler_bytes, limits, prefill, control, Ok)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_json_int8_sampled_sparse_with<C, T, E, F>(engine: &mut StrictInt8Engine<'_>,
    identity: &ExecutionIdentity, prompt: &[u32], program: &JsonProgram, vocabulary: &VocabMaskOracle,
    options: &JsonDecodeOptions, sampling: &Int8JsonSamplingOptions, budget: Int8JsonBudget,
    max_sampler_bytes: u64, limits: Int8JsonSparseLimits, control: &mut C, finalize: F) -> Result<T, E>
where C: DecodeStepControl, E: From<Int8JsonError>, F: FnOnce(Int8JsonRun) -> Result<T, E> {
    scheduled(engine, identity, prompt, program, vocabulary, options, sampling, budget,
        max_sampler_bytes, limits, None, control, finalize)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_json_int8_sampled_sparse_layer_major_with<C, T, E, F>(engine: &mut StrictInt8Engine<'_>,
    identity: &ExecutionIdentity, prompt: &[u32], program: &JsonProgram, vocabulary: &VocabMaskOracle,
    options: &JsonDecodeOptions, sampling: &Int8JsonSamplingOptions, budget: Int8JsonBudget,
    max_sampler_bytes: u64, limits: Int8JsonSparseLimits, prefill: Int8PrefillLimits,
    control: &mut C, finalize: F) -> Result<T, E>
where C: DecodeStepControl, E: From<Int8JsonError>, F: FnOnce(Int8JsonRun) -> Result<T, E> {
    scheduled(engine, identity, prompt, program, vocabulary, options, sampling, budget,
        max_sampler_bytes, limits, Some(prefill), control, finalize)
}

#[allow(clippy::too_many_arguments)]
fn scheduled<C, T, E, F>(engine: &mut StrictInt8Engine<'_>, identity: &ExecutionIdentity,
    prompt: &[u32], program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    sampling: &Int8JsonSamplingOptions, budget: Int8JsonBudget, max_sampler_bytes: u64,
    limits: Int8JsonSparseLimits, prefill: Option<Int8PrefillLimits>, control: &mut C, finalize: F)
    -> Result<T, E>
where C: DecodeStepControl, E: From<Int8JsonError>, F: FnOnce(Int8JsonRun) -> Result<T, E> {
    if let Some(prefill) = prefill { prefill.validate().map_err(Int8JsonError::from).map_err(E::from)?; }
    sampling.validate().map_err(E::from)?;
    check_sampler_budget(max_sampler_bytes).map_err(E::from)?;
    selected_head::check_request(engine, identity, prompt, vocabulary, options, budget, limits).map_err(E::from)?;
    if let Some(cause) = control.checkpoint(0) {
        return Err(E::from(Int8JsonError::Decode(JsonDecodeError::Cancelled(cause))));
    }
    let mut selection = Selected::new(identity, prompt, program, options, sampling, max_sampler_bytes, false)
        .map_err(E::from)?;
    let mut session = engine.session(budget.native, control).map_err(Int8JsonError::from).map_err(E::from)?;
    let mut driver = Prompt { inner: &mut session, baseline: Int8Work::default(), retained: 0,
        replay: false, prefill };
    selected_head::drive_selected(prompt, program, vocabulary, options, budget,
        limits, &mut driver, &mut selection, finalize)
}

struct Selected { sampled: Sampled, reused: bool }
impl Selected {
    fn new(identity: &ExecutionIdentity, prompt: &[u32], program: &JsonProgram, options: &JsonDecodeOptions,
        sampling: &Int8JsonSamplingOptions, bytes: u64, reused: bool) -> Result<Self, Int8JsonError> {
        // This is the unchanged full-head key and processor construction.
        Ok(Self { sampled: Sampled::new(identity, prompt, program, options, sampling, bytes)?, reused })
    }
}
impl selected_head::Selection for Selected {
    #[allow(clippy::too_many_arguments)]
    fn project<D: selected_head::selection::Head>(&mut self, mask: &DenseTokenMask, accepting: bool,
        options: &JsonDecodeOptions, cap: usize, remaining: u64, driver: &mut D, step: usize)
        -> Result<(u32, usize), Int8JsonError> {
        let sampled = &mut self.sampled;
        sampled.workspace.begin_legal_rows();
        let count = selected_head::selection::visit(mask, accepting, options, cap, remaining, driver, step,
            |rows, logits| sampled.workspace.extend_legal_rows(rows, logits, &sampled.options, step as u64)
                .map_err(sampling_error))?;
        let token = sampled.workspace.finish_legal_rows_with_control(&sampled.options, sampled.key,
            sampled.sample_index, step, driver.control()).map_err(sampling_error)?;
        Ok((token, count))
    }
    fn execution(&self) -> &'static str {
        if self.reused { INT8_JSON_SAMPLED_SPARSE_REUSED_PREFILL_EXECUTION }
        else { INT8_JSON_SAMPLED_SPARSE_EXECUTION }
    }
}

trait Replay: selected_head::Driver {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int8PrefillLimits) -> Result<(), Int8JsonError>;
    fn rewind_for_replay(&mut self, retain: usize) -> Result<(), Int8JsonError>;
}
impl<C: DecodeStepControl> Replay for Int8Session<'_, '_, C> {
    fn append_layer_major(&mut self, prompt: &[u32], limits: Int8PrefillLimits) -> Result<(), Int8JsonError> {
        Ok(Int8Session::append_layer_major(self, prompt, limits)?)
    }
    fn rewind_for_replay(&mut self, retain: usize) -> Result<(), Int8JsonError> {
        Ok(Int8Session::rewind_for_replay(self, retain)?)
    }
}
struct Prompt<'a, D> { inner: &'a mut D, baseline: Int8Work, retained: usize,
    replay: bool, prefill: Option<Int8PrefillLimits> }
impl<D: Replay> selected_head::Driver for Prompt<'_, D> {
    type Control = D::Control;
    fn control(&mut self) -> &mut Self::Control { self.inner.control() }
    fn append(&mut self, token: u32) -> Result<(), Int8JsonError> { self.inner.append(token) }
    fn append_prompt(&mut self, prompt: &[u32]) -> Result<(), Int8JsonError> {
        if self.retained >= prompt.len() || (!self.replay && self.retained != 0)
            || (self.replay && self.retained + 1 != prompt.len()) {
            return Err(Int8JsonError::WorkMismatch);
        }
        if self.replay {
            self.inner.rewind_for_replay(self.retained)?;
            if let Some(cause) = self.inner.control().prefill_checkpoint(self.retained) {
                return Err(JsonDecodeError::Cancelled(cause).into());
            }
            self.inner.append(prompt[self.retained])
        } else { match self.prefill {
            Some(limits) => {
                for index in 0..prompt.len() {
                    if let Some(cause) = self.inner.control().prefill_checkpoint(index) {
                        return Err(JsonDecodeError::Cancelled(cause).into());
                    }
                }
                self.inner.append_layer_major(prompt, limits)
            }
            None => self.inner.append_prompt(prompt),
        } }
    }
    fn logits(&mut self, rows: &[u32]) -> Result<Vec<f32>, Int8JsonError> { self.inner.logits(rows) }
    fn work(&self) -> Result<Int8Work, Int8JsonError> { work_delta(self.inner.work()?, self.baseline) }
    fn retained_prompt_positions(&self) -> usize { self.retained }
    fn abort(&mut self) { self.inner.abort(); }
}

/// Reuse one admitted prompt prefix inside one native session. The address
/// array, sampler, completed outputs and native/prefill storage retain their
/// existing simultaneous-lifetime quotes. Later samples replay the final
/// prompt token and report only newly executed causal work; no work is refunded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_json_int8_sampled_sparse_reusing_prefill_with<C, T, E, F>(
    engine: &mut StrictInt8Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32], program: &JsonProgram,
    vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions, sampling: &[Int8JsonSamplingOptions],
    budget: Int8JsonBudget, max_sampler_bytes: u64, limits: Int8JsonSparseLimits,
    prefill: Option<Int8PrefillLimits>, control: &mut C, finalize: F) -> Result<Vec<T>, E>
where C: DecodeStepControl, E: From<Int8JsonError>, F: FnMut(usize, Int8JsonRun) -> Result<T, E> {
    if let Some(prefill) = prefill { prefill.validate().map_err(Int8JsonError::from).map_err(E::from)?; }
    check_sampling(sampling, max_sampler_bytes).map_err(E::from)?;
    selected_head::check_request(engine, identity, prompt, vocabulary, options, budget, limits).map_err(E::from)?;
    repeated_preflight(prompt, vocabulary, options, sampling.len(), budget, limits).map_err(E::from)?;
    if let Some(cause) = control.checkpoint(0) {
        return Err(E::from(Int8JsonError::Decode(JsonDecodeError::Cancelled(cause))));
    }
    let mut session = engine.session(budget.native, control).map_err(Int8JsonError::from).map_err(E::from)?;
    drive_reused(identity, prompt, program, vocabulary, options, sampling, budget,
        max_sampler_bytes, limits, prefill, &mut session, finalize)
}

fn repeated_preflight<V: Vocabulary>(prompt: &[u32], vocabulary: &V,
    options: &JsonDecodeOptions, samples: usize, budget: Int8JsonBudget, limits: Int8JsonSparseLimits)
    -> Result<Int8Work, Int8JsonError> {
    check_sample_count(samples)?;
    let cold = selected_head::preflight(prompt, vocabulary, options, budget, limits)?;
    let projected = options.max_new_tokens.checked_mul(limits.validate()?).ok_or(StrictInt8Error::Work)?;
    let warm = Int8Work::for_sequence(prompt.len() - 1, options.max_new_tokens, projected)?;
    let mut complete = cold;
    for _ in 1..samples { complete = complete.checked_add(warm)?; }
    let masks = (vocabulary.mask_charge(budget.json.mask_limits) as u64)
        .checked_mul(options.max_new_tokens as u64).and_then(|one| one.checked_mul(samples as u64))
        .ok_or(StrictInt8Error::Work)?;
    if complete.forward_positions > budget.native.max_forward_positions
        || complete.attention_pairs > budget.native.max_attention_pairs
        || !complete.projections.fits(budget.native.max_projection_work)
        || complete.forward_positions > budget.json.max_forward_positions
        || complete.projected_logits > budget.json.max_projected_logits
        || masks > budget.json.max_total_mask_node_visits {
        return Err(JsonDecodeError::BudgetExceeded("complete selected reused-prefill work").into());
    }
    Ok(complete)
}

#[allow(clippy::too_many_arguments)]
fn drive_reused<V, D, T, E, F>(identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &V, options: &JsonDecodeOptions, sampling: &[Int8JsonSamplingOptions],
    budget: Int8JsonBudget, max_sampler_bytes: u64, limits: Int8JsonSparseLimits,
    prefill: Option<Int8PrefillLimits>, driver: &mut D, mut finalize: F) -> Result<Vec<T>, E>
where V: Vocabulary, D: Replay, E: From<Int8JsonError>, F: FnMut(usize, Int8JsonRun) -> Result<T, E> {
    check_sampling(sampling, max_sampler_bytes).map_err(E::from)?;
    if let Some(prefill) = prefill { prefill.validate().map_err(Int8JsonError::from).map_err(E::from)?; }
    let complete = repeated_preflight(prompt, vocabulary, options, sampling.len(), budget, limits).map_err(E::from)?;
    let result = (|| {
        if driver.work().map_err(E::from)? != Int8Work::default() { return Err(E::from(Int8JsonError::WorkMismatch)); }
        let mut output = Vec::new();
        output.try_reserve_exact(sampling.len())
            .map_err(|_| E::from(Int8JsonError::Decode(JsonDecodeError::AllocationRefused)))?;
        let mut work = Int8Work::default(); let mut masks = 0_u64;
        for (index, sample) in sampling.iter().enumerate() {
            if let Some(cause) = driver.control().checkpoint(0) {
                return Err(E::from(Int8JsonError::Decode(JsonDecodeError::Cancelled(cause))));
            }
            let mut selection = Selected::new(identity, prompt, program, options, sample, max_sampler_bytes, true)
                .map_err(E::from)?;
            let baseline = driver.work().map_err(E::from)?;
            let mut replay = Prompt { inner: &mut *driver, baseline,
                retained: if index == 0 { 0 } else { prompt.len() - 1 }, replay: index != 0, prefill };
            output.push(selected_head::drive_selected(prompt, program, vocabulary, options, budget,
                limits, &mut replay, &mut selection, |run| {
                    work = work.checked_add(run.model_work).map_err(Int8JsonError::from).map_err(E::from)?;
                    masks = masks.checked_add(run.output.mask_node_visit_charge)
                        .filter(|&charge| charge <= budget.json.max_total_mask_node_visits)
                        .ok_or_else(|| E::from(Int8JsonError::WorkMismatch))?;
                    finalize(index, run)
                })?);
        }
        if driver.work().map_err(E::from)? != work || work.forward_positions > complete.forward_positions
            || work.projected_logits > complete.projected_logits || work.attention_pairs > complete.attention_pairs
            || !work.projections.fits(complete.projections) { return Err(E::from(Int8JsonError::WorkMismatch)); }
        if let Some(cause) = driver.control().checkpoint(0) {
            return Err(E::from(Int8JsonError::Decode(JsonDecodeError::Cancelled(cause))));
        }
        Ok(output)
    })();
    if result.is_err() { driver.abort(); }
    result
}

#[cfg(test)] mod tests;
