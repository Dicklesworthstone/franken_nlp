//! Independent addressed samples sharing one exclusive prompt prefix.
//!
//! The final prompt token is replayed for every later sample. No KV snapshot,
//! hidden-state history, saved logits or cross-request cache is created.

use super::*;

pub const INT8_JSON_SAMPLED_REUSED_PREFILL_EXECUTION: &str =
    "portable-int8-sampled-constrained-json-reused-prefill-v1";
const MAX_REUSED_PREFILL_SAMPLES: usize = 32;

/// Simultaneous payload for one sampler plus the caller-owned array of sample
/// addresses retained throughout replay. The array is charged in addition to
/// the existing bounded identity-construction allowance, using its actual Rust
/// element size. Completed outputs, native/KV storage and optional prompt
/// batching retain their existing separate quotes; allocator overhead is not
/// represented by this payload bound.
pub fn json_sampling_reuse_workspace_bytes(samples: usize) -> Result<u64, Int8JsonError> {
    check_sample_count(samples)?;
    let addresses = (samples as u64).checked_mul(std::mem::size_of::<Int8JsonSamplingOptions>() as u64)
        .ok_or(JsonDecodeError::BudgetExceeded("JSON sample address storage"))?;
    json_sampling_workspace_bytes()?.checked_add(addresses)
        .ok_or_else(|| JsonDecodeError::BudgetExceeded("JSON sampler accounting").into())
}

fn check_sample_count(samples: usize) -> Result<(), Int8JsonError> {
    if samples == 0 || samples > MAX_REUSED_PREFILL_SAMPLES {
        return Err(JsonDecodeError::InvalidRequest("reused-prefill sample count").into());
    }
    Ok(())
}

fn check_sampling(sampling: &[Int8JsonSamplingOptions], max_sampler_bytes: u64)
    -> Result<(), Int8JsonError> {
    let needed = json_sampling_reuse_workspace_bytes(sampling.len())?;
    for sample in sampling { sample.validate()?; }
    if needed > max_sampler_bytes {
        return Err(JsonDecodeError::BudgetExceeded("JSON reused sampler storage").into());
    }
    Ok(())
}

/// The sealed task owner admits the complete cold sample ceiling. This seam
/// also checks the exact replay maximum before opening the single session.
/// Every completed sample reports only its newly executed physical work;
/// neither native reservations nor remaining ceilings are reset or refunded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_json_int8_sampled_reusing_prefill_with<C, T, E, F>(
    engine: &mut StrictInt8Engine<'_>, identity: &ExecutionIdentity, prompt: &[u32],
    program: &JsonProgram, vocabulary: &VocabMaskOracle, options: &JsonDecodeOptions,
    sampling: &[Int8JsonSamplingOptions], budget: Int8JsonBudget, max_sampler_bytes: u64,
    prefill: Option<Int8PrefillLimits>, control: &mut C, finalize: F,
) -> Result<Vec<T>, E>
where C: DecodeStepControl, E: From<Int8JsonError>,
    F: FnMut(usize, Int8JsonRun) -> Result<T, E> {
    if let Some(limits) = prefill { limits.validate().map_err(Int8JsonError::from).map_err(E::from)?; }
    check_sampling(sampling, max_sampler_bytes).map_err(E::from)?;
    check_request(engine, identity, prompt, vocabulary, options, budget).map_err(E::from)?;
    repeated_preflight(prompt, vocabulary, options, sampling.len(), budget).map_err(E::from)?;
    if let Some(cause) = control.checkpoint(0) {
        return Err(E::from(Int8JsonError::Decode(JsonDecodeError::Cancelled(cause))));
    }
    let mut session = engine.session(budget.native, control).map_err(Int8JsonError::from).map_err(E::from)?;
    drive_reused_prefill(identity, prompt, program, vocabulary, options, sampling,
        budget, max_sampler_bytes, prefill, &mut session, finalize)
}

fn repeated_preflight<V: Vocabulary>(prompt: &[u32], vocabulary: &V, options: &JsonDecodeOptions,
    samples: usize, budget: Int8JsonBudget) -> Result<Int8Work, Int8JsonError> {
    check_sample_count(samples)?;
    let cold = preflight(prompt, vocabulary, options, budget)?;
    let projected = options.max_new_tokens.checked_mul(vocabulary.width()).ok_or(StrictInt8Error::Work)?;
    // Replay forwards one final prompt token plus generated non-EOS tokens.
    // Every new attention row still sees the retained earlier positions.
    let warm = Int8Work::for_sequence(prompt.len() - 1, options.max_new_tokens, projected)?;
    let mut complete = cold;
    for _ in 1..samples { complete = complete.checked_add(warm)?; }
    let masks = (vocabulary.mask_charge(budget.json.mask_limits) as u64)
        .checked_mul(options.max_new_tokens as u64)
        .and_then(|one| one.checked_mul(samples as u64)).ok_or(StrictInt8Error::Work)?;
    if complete.forward_positions > budget.native.max_forward_positions
        || complete.attention_pairs > budget.native.max_attention_pairs
        || !complete.projections.fits(budget.native.max_projection_work)
        || complete.forward_positions > budget.json.max_forward_positions
        || complete.projected_logits > budget.json.max_projected_logits
        || masks > budget.json.max_total_mask_node_visits {
        return Err(JsonDecodeError::BudgetExceeded("complete reused-prefill work").into());
    }
    Ok(complete)
}

// Product callers cannot implement this driver or mint a native result. The
// actual session retains its model, native ledger, controller and cache owner.
trait ReplayDriver: prefill::PromptDriver {
    fn rewind_for_replay(&mut self, retain: usize) -> Result<(), Int8JsonError>;
}
impl<C: DecodeStepControl> ReplayDriver for Int8Session<'_, '_, C> {
    fn rewind_for_replay(&mut self, retain: usize) -> Result<(), Int8JsonError> {
        Ok(Int8Session::rewind_for_replay(self, retain)?)
    }
}

struct PromptReplay<'a, D> {
    inner: &'a mut D, baseline: Int8Work, retained: usize, replay: bool,
    prefill: Option<Int8PrefillLimits>,
}
impl<D: ReplayDriver> Driver for PromptReplay<'_, D> {
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
        } else {
            match self.prefill {
                Some(limits) => {
                    // Match the existing INT8 LayerMajor adapter's prompt
                    // checkpoints before entering its native morsels.
                    for index in 0..prompt.len() {
                        if let Some(cause) = self.inner.control().prefill_checkpoint(index) {
                            return Err(JsonDecodeError::Cancelled(cause).into());
                        }
                    }
                    self.inner.append_layer_major(prompt, limits)
                }
                None => self.inner.append_prompt(prompt),
            }
        }
    }
    fn logits(&mut self) -> Result<Vec<f32>, Int8JsonError> { self.inner.logits() }
    fn work(&self) -> Result<Int8Work, Int8JsonError> { work_delta(self.inner.work()?, self.baseline) }
    fn retained_prompt_positions(&self) -> usize { self.retained }
    fn abort(&mut self) { self.inner.abort(); }
}

fn work_delta(total: Int8Work, before: Int8Work) -> Result<Int8Work, Int8JsonError> {
    let sub = |a: u64, b: u64| a.checked_sub(b).ok_or(Int8JsonError::WorkMismatch);
    Ok(Int8Work { forward_positions: sub(total.forward_positions, before.forward_positions)?,
        projected_logits: sub(total.projected_logits, before.projected_logits)?,
        attention_pairs: sub(total.attention_pairs, before.attention_pairs)?,
        projections: crate::native_engine::portable_int8::ProjectionWork {
            dot_products: sub(total.projections.dot_products, before.projections.dot_products)?,
            multiply_accumulates: sub(total.projections.multiply_accumulates, before.projections.multiply_accumulates)?,
        } })
}

// Physical execution provenance differs; the actual addressed draw keeps the
// original Sampled key, prompt, grammar and semantic sample_index unchanged.
struct ReusedSelection(Sampled);
impl Selection for ReusedSelection {
    fn select<C: DecodeStepControl>(&mut self, logits: &[f32], mask: &DenseTokenMask, accepting: bool,
        options: &JsonDecodeOptions, step: usize, control: &mut C) -> Result<u32, Int8JsonError> {
        self.0.select(logits, mask, accepting, options, step, control)
    }
    fn execution(&self) -> &'static str { INT8_JSON_SAMPLED_REUSED_PREFILL_EXECUTION }
}

#[allow(clippy::too_many_arguments)]
fn drive_reused_prefill<V, D, T, E, F>(identity: &ExecutionIdentity,
    prompt: &[u32], program: &JsonProgram, vocabulary: &V, options: &JsonDecodeOptions,
    sampling: &[Int8JsonSamplingOptions], budget: Int8JsonBudget, max_sampler_bytes: u64,
    prefill: Option<Int8PrefillLimits>, driver: &mut D, mut finalize: F) -> Result<Vec<T>, E>
where V: Vocabulary, D: ReplayDriver, E: From<Int8JsonError>,
    F: FnMut(usize, Int8JsonRun) -> Result<T, E> {
    check_sampling(sampling, max_sampler_bytes).map_err(E::from)?;
    if let Some(limits) = prefill { limits.validate().map_err(Int8JsonError::from).map_err(E::from)?; }
    let complete = repeated_preflight(prompt, vocabulary, options, sampling.len(), budget).map_err(E::from)?;
    let result = (|| {
        if driver.work().map_err(E::from)? != Int8Work::default() {
            return Err(E::from(Int8JsonError::WorkMismatch));
        }
        let mut output = Vec::new();
        output.try_reserve_exact(sampling.len())
            .map_err(|_| E::from(Int8JsonError::Decode(JsonDecodeError::AllocationRefused)))?;
        let mut work = Int8Work::default();
        let mut masks = 0_u64;
        for (index, sample) in sampling.iter().enumerate() {
            if let Some(cause) = driver.control().checkpoint(0) {
                return Err(E::from(Int8JsonError::Decode(JsonDecodeError::Cancelled(cause))));
            }
            let mut selection = ReusedSelection(Sampled::new(identity, prompt, program, options,
                sample, max_sampler_bytes).map_err(E::from)?);
            let baseline = driver.work().map_err(E::from)?;
            let mut replay = PromptReplay { inner: &mut *driver, baseline,
                retained: if index == 0 { 0 } else { prompt.len() - 1 }, replay: index != 0, prefill };
            let checked = drive_selected(prompt, program, vocabulary, options,
                budget, &mut replay, &mut selection, |run| {
                    work = work.checked_add(run.model_work).map_err(Int8JsonError::from).map_err(E::from)?;
                    masks = masks.checked_add(run.output.mask_node_visit_charge)
                        .filter(|&charge| charge <= budget.json.max_total_mask_node_visits)
                        .ok_or_else(|| E::from(Int8JsonError::WorkMismatch))?;
                    finalize(index, run)
                })?;
            output.push(checked);
        }
        if driver.work().map_err(E::from)? != work || work.forward_positions > complete.forward_positions
            || work.projected_logits > complete.projected_logits || work.attention_pairs > complete.attention_pairs
            || !work.projections.fits(complete.projections) {
            return Err(E::from(Int8JsonError::WorkMismatch));
        }
        if let Some(cause) = driver.control().checkpoint(0) {
            return Err(E::from(Int8JsonError::Decode(JsonDecodeError::Cancelled(cause))));
        }
        Ok(output)
    })();
    if result.is_err() { driver.abort(); }
    result
}

#[cfg(test)] mod tests;
