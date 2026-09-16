//! Native finite-candidate execution with one live KV branch. Prefill is
//! computed once, continuation prefixes reuse their longest common prefix,
//! and the head projects only the rows required by the declared score space.
//! This is destructive backtracking, not a retained KV fork. No model loader,
//! weight clone, second KV reservation, worker, or independent runtime exists.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use super::super::{
    HfBf16EagerEngine, HfBf16EagerError, HF_BF16_EAGER_PROFILE, run_hf_bf16_layer,
};
use crate::{
    canonjson,
    native_engine::{
        decode::{DecodeCancellationKind, DecodeStepControl},
        kv::{KvCache, KvRewindError, KV_BYTES_PER_TOKEN, KV_SLOT_COUNT, PHYSICAL_LAYER_COUNT},
        layer::{HfBf16EagerLayerWeights, HfBf16LayerError},
        lmhead::{NANBEIGE_VOCAB_SIZE,
            scoring::{CandidateLogits, ProjectionRows, ScoringLimits, ScoringMode},
            selected::{HeadProjectionError, checked_row_count, project_rows_with_control}},
        looprun::{LayerBinding, LayerExecutor, LoopRunner, PositionContext},
        nn::{RMS_NORM_EPSILON, embedding_row_stays_bf16, rms_norm_f32_reduce_cast_back},
        rope::RopeTablesF32,
        tensor::Bf16,
    },
    tasks::{
        classify::{ClassificationError, ClassificationOptions, ClassificationPlan, ClassificationResult},
        ir::{Candidate, DecodeStrategy, TaskPlan},
    },
};

pub const PREFIX_EXECUTION_VERSION: &str = "eager-single-kv-prefix-rewind-v1";

/// Aggregate REAL forwards and head rows, independent of logical trie edges.
/// The caller still owns process/resource admission of the existing engine.
#[derive(Clone, Copy, Debug)]
pub struct PrefixBudget {
    pub max_forward_positions: u64,
    pub max_projected_logits: u64,
}
impl Default for PrefixBudget {
    fn default() -> Self {
        Self { max_forward_positions: 4096, max_projected_logits: 100_000_000 }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixWork {
    pub prefix_evaluations: u64,
    pub forward_positions: u64,
    pub prompt_positions: u64,
    pub continuation_positions: u64,
    pub projected_logits: u64,
    pub rewound_positions: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CachedClassificationRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub result: ClassificationResult,
    pub native_work: PrefixWork,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrefixScoringError {
    Classification(ClassificationError),
    InvalidPrompt,
    InvalidPrefix,
    EngineAlreadyPrimed,
    ContextBudget,
    KvBudget,
    WorkBudget,
    ArithmeticOverflow,
    AllocationRefused,
    InvalidExecution,
    Poisoned,
    Rewind(KvRewindError),
    Engine(HfBf16EagerError),
    Head(HeadProjectionError),
    Cancelled(DecodeCancellationKind),
    OutputBudget,
    Serialization,
}
impl fmt::Display for PrefixScoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never render private prompt/candidate tokens, hidden states or logits.
        f.write_str(match self {
            Self::Classification(_) => "cached classification plan or result refused",
            Self::InvalidPrompt => "cached scoring requires a valid nonempty exact prompt",
            Self::InvalidPrefix => "cached scoring continuation is invalid",
            Self::EngineAlreadyPrimed => "cached scoring requires an empty admitted engine",
            Self::ContextBudget => "cached scoring exceeds admitted context",
            Self::KvBudget => "cached scoring exceeds the task KV budget",
            Self::WorkBudget => "cached scoring exceeds aggregate native work budget",
            Self::ArithmeticOverflow => "cached scoring accounting overflow",
            Self::AllocationRefused => "cached scoring allocation refused",
            Self::InvalidExecution => "cached scoring execution contract diverged",
            Self::Poisoned => "cached scoring session failed and cannot be retried",
            Self::Rewind(_) => "cached scoring cannot rewind an incomplete prefix",
            Self::Engine(_) => "cached scoring native forward failed",
            Self::Head(_) => "cached scoring native projection failed",
            Self::Cancelled(_) => "cached scoring cancelled",
            Self::OutputBudget => "cached classification complete output budget exceeded",
            Self::Serialization => "cached classification serialization failed",
        })
    }
}
impl Error for PrefixScoringError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Classification(e) => Some(e), Self::Rewind(e) => Some(e),
            Self::Head(e) => Some(e), _ => None,
        }
    }
}
impl From<ClassificationError> for PrefixScoringError {
    fn from(e: ClassificationError) -> Self { Self::Classification(e) }
}
impl From<HfBf16EagerError> for PrefixScoringError {
    fn from(e: HfBf16EagerError) -> Self { Self::Engine(e) }
}
impl From<KvRewindError> for PrefixScoringError {
    fn from(e: KvRewindError) -> Self { Self::Rewind(e) }
}
impl From<HeadProjectionError> for PrefixScoringError {
    fn from(e: HeadProjectionError) -> Self {
        match e { HeadProjectionError::Cancelled(cause) => Self::Cancelled(cause), other => Self::Head(other) }
    }
}

/// No Debug/Clone/Serialize: this value owns a live private branch and borrows
/// its immutable exact prompt and the caller's exclusive admitted engine.
/// Dropping it clears logical KV positions, retaining the engine's buffers.
pub struct EagerPrefixSession<'a, C: DecodeStepControl> {
    engine: &'a mut HfBf16EagerEngine,
    prompt: &'a [u32],
    control: &'a mut C,
    prefix: Vec<u32>,
    max_prefix: usize,
    hidden: Option<Vec<Bf16>>,
    primed: bool,
    poisoned: bool,
    remaining_positions: u64,
    remaining_logits: u64,
    work: PrefixWork,
    last_error: Option<PrefixScoringError>,
}
impl<'a, C: DecodeStepControl> EagerPrefixSession<'a, C> {
    /// Borrow an ALREADY admitted, empty engine. No extra KV buffer is created.
    /// The task's max_kv_bytes prices the entire engine reservation, not just
    /// the currently occupied prefix. Failure here leaves pre-existing state
    /// untouched. The caller must bind this prompt to its validated TaskPlan.
    pub fn new(
        engine: &'a mut HfBf16EagerEngine,
        prompt: &'a [u32],
        max_prefix_tokens: usize,
        max_kv_bytes: u64,
        budget: PrefixBudget,
        control: &'a mut C,
    ) -> Result<Self, PrefixScoringError> {
        if !engine.kv_cache.all_slots_have_len(0) { return Err(PrefixScoringError::EngineAlreadyPrimed); }
        if prompt.is_empty() || prompt.iter().any(|&id| id as usize >= NANBEIGE_VOCAB_SIZE) {
            return Err(PrefixScoringError::InvalidPrompt);
        }
        let required = prompt.len().checked_add(max_prefix_tokens).ok_or(PrefixScoringError::ArithmeticOverflow)?;
        if required > engine.kv_cache.capacity_positions() { return Err(PrefixScoringError::ContextBudget); }
        let kv_bytes = (engine.kv_cache.capacity_positions() as u64)
            .checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(PrefixScoringError::ArithmeticOverflow)?;
        if kv_bytes > max_kv_bytes { return Err(PrefixScoringError::KvBudget); }
        if prompt.len() as u64 > budget.max_forward_positions || budget.max_projected_logits == 0 {
            return Err(PrefixScoringError::WorkBudget);
        }
        let mut prefix = Vec::new();
        prefix.try_reserve_exact(max_prefix_tokens).map_err(|_| PrefixScoringError::AllocationRefused)?;
        Ok(Self { engine, prompt, control, prefix, max_prefix: max_prefix_tokens,
            hidden: None, primed: false, poisoned: false,
            remaining_positions: budget.max_forward_positions,
            remaining_logits: budget.max_projected_logits,
            work: PrefixWork::default(), last_error: None })
    }
    pub fn work(&self) -> PrefixWork { self.work }
    pub fn is_poisoned(&self) -> bool { self.poisoned }
    pub fn last_error(&self) -> Option<&PrefixScoringError> { self.last_error.as_ref() }

    fn project_inner(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, PrefixScoringError> {
        if self.poisoned { return Err(PrefixScoringError::Poisoned); }
        let row_count = checked_row_count(rows, NANBEIGE_VOCAB_SIZE, NANBEIGE_VOCAB_SIZE)?;
        if prefix.len() > self.max_prefix || prefix.iter().any(|&id| id as usize >= NANBEIGE_VOCAB_SIZE) {
            return Err(PrefixScoringError::InvalidPrefix);
        }
        let previous_positions = if self.primed { self.prompt.len() + self.prefix.len() } else { 0 };
        if !self.engine.kv_cache.all_slots_have_len(previous_positions) {
            return Err(PrefixScoringError::InvalidExecution);
        }
        let transition = transition(self.prompt.len(), &self.prefix, prefix, self.primed)?;
        let prompt_steps = self.prompt.len() - transition.prompt_from;
        let prefix_steps = prefix.len() - transition.prefix_from;
        let steps = prompt_steps.checked_add(prefix_steps).ok_or(PrefixScoringError::ArithmeticOverflow)? as u64;
        let remaining_positions = self.remaining_positions.checked_sub(steps).ok_or(PrefixScoringError::WorkBudget)?;
        let remaining_logits = self.remaining_logits.checked_sub(row_count as u64).ok_or(PrefixScoringError::WorkBudget)?;
        let rewound = previous_positions.checked_sub(transition.retain).ok_or(PrefixScoringError::InvalidExecution)? as u64;
        let total_rewound = self.work.rewound_positions.checked_add(rewound).ok_or(PrefixScoringError::ArithmeticOverflow)?;
        // Precharge the whole operation and poison BEFORE callbacks, native
        // work, or mutation. Failed/cancelled/unwound work cannot be retried for
        // free, even if an outer caller catches an unwind and keeps the session.
        self.remaining_positions = remaining_positions;
        self.remaining_logits = remaining_logits;
        self.poisoned = true;
        self.engine.kv_cache.rewind_completed(transition.retain)?;
        self.work.rewound_positions = total_rewound;
        for index in transition.prompt_from..self.prompt.len() {
            self.hidden = Some(forward_hidden(self.engine, self.prompt[index], self.control)?);
            self.work.forward_positions += 1;
            self.work.prompt_positions += 1;
        }
        for &token in &prefix[transition.prefix_from..] {
            self.hidden = Some(forward_hidden(self.engine, token, self.control)?);
            self.work.forward_positions += 1;
            self.work.continuation_positions += 1;
        }
        let expected_positions = self.prompt.len() + prefix.len();
        if !self.engine.kv_cache.all_slots_have_len(expected_positions) {
            return Err(PrefixScoringError::InvalidExecution);
        }
        let hidden = self.hidden.as_deref().ok_or(PrefixScoringError::InvalidExecution)?;
        // Classification does not commit generated tokens. Keep deadline/poll
        // cancellation on the prefill channel, not the generation token limit.
        let mut head_control = PrefillControl { inner: &mut *self.control, position: expected_positions - 1 };
        let logits = project_rows_with_control(hidden, &self.engine.weights.lm_head, rows,
            row_count, &mut head_control, 0)?;
        self.work.projected_logits += row_count as u64;
        self.work.prefix_evaluations += 1;
        self.prefix.clear();
        self.prefix.extend_from_slice(prefix); // max_prefix was reserved once.
        self.primed = true;
        self.poisoned = false;
        Ok(logits)
    }
}
impl<C: DecodeStepControl> CandidateLogits for EagerPrefixSession<'_, C> {
    type Error = PrefixScoringError;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let result = self.project_inner(prefix, rows);
        if let Err(error) = &result {
            self.poisoned = true;
            if self.last_error.is_none() { self.last_error = Some(error.clone()); }
            self.engine.kv_cache.clear();
        }
        result
    }
}
impl<C: DecodeStepControl> Drop for EagerPrefixSession<'_, C> {
    fn drop(&mut self) { self.engine.kv_cache.clear(); }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Transition { retain: usize, prompt_from: usize, prefix_from: usize }
/// The one shared scheduling authority for arbitrary prefix query order.
/// Repeating a prefix needs no forward. Revisiting an ancestor recomputes just
/// its last token because this constant-memory session retains only one hidden
/// state, not an unpriced hidden-state/logit table for every trie node.
fn transition(prompt_len: usize, previous: &[u32], next: &[u32], primed: bool) -> Result<Transition, PrefixScoringError> {
    if prompt_len == 0 { return Err(PrefixScoringError::InvalidPrompt); }
    if !primed { return Ok(Transition { retain: 0, prompt_from: 0, prefix_from: 0 }); }
    let common = previous.iter().zip(next).take_while(|(a, b)| a == b).count();
    let end = prompt_len.checked_add(common).ok_or(PrefixScoringError::ArithmeticOverflow)?;
    if previous == next || common < next.len() {
        return Ok(Transition { retain: end, prompt_from: prompt_len, prefix_from: common });
    }
    // The requested ancestor's old hidden state was intentionally not retained.
    Ok(if next.is_empty() {
        Transition { retain: prompt_len - 1, prompt_from: prompt_len - 1, prefix_from: 0 }
    } else {
        Transition { retain: end - 1, prompt_from: prompt_len, prefix_from: next.len() - 1 }
    })
}

struct PrefillControl<'a, C> { inner: &'a mut C, position: usize }
impl<C: DecodeStepControl> DecodeStepControl for PrefillControl<'_, C> {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.inner.prefill_checkpoint(self.position)
    }
}
fn checkpoint<C: DecodeStepControl>(control: &mut C, position: usize) -> Result<(), PrefixScoringError> {
    match control.prefill_checkpoint(position) {
        Some(cause) => Err(PrefixScoringError::Cancelled(cause)), None => Ok(()),
    }
}

/// Same validated weights, layer primitive, cast schedule and 44-binding runner
/// as the reference decode path, but no L2 tap copies or unrequested lm_head.
/// Only the final hidden state is retained. A failed call may have partial KV;
/// the owning session always clears it and is permanently poisoned on failure.
fn forward_hidden<C: DecodeStepControl>(engine: &mut HfBf16EagerEngine, token: u32, control: &mut C) -> Result<Vec<Bf16>, PrefixScoringError> {
    let position = engine.sequence_len()?;
    if token as usize >= NANBEIGE_VOCAB_SIZE { return Err(PrefixScoringError::InvalidPrefix); }
    if position >= engine.kv_cache.capacity_positions() { return Err(PrefixScoringError::ContextBudget); }
    checkpoint(control, position)?;
    let mut hidden = embedding_row_stays_bf16(engine.weights.embeddings.row(token as usize)
        .map_err(HfBf16EagerError::from)?);
    let runner = LoopRunner::from_layer_weights(&engine.weights.layers);
    let mut executor = HiddenExecutor { final_norm: &engine.weights.final_norm,
        rope: &engine.rope, completed_layers: 0, control };
    runner.run_token(&mut executor, &mut hidden, PositionContext::at(position), &mut engine.kv_cache)?;
    if executor.completed_layers != KV_SLOT_COUNT { return Err(PrefixScoringError::InvalidExecution); }
    checkpoint(executor.control, position)?;
    Ok(hidden)
}
struct HiddenExecutor<'a, C> {
    final_norm: &'a [Bf16], rope: &'a RopeTablesF32, completed_layers: usize, control: &'a mut C,
}
impl<C: DecodeStepControl> LayerExecutor<HfBf16EagerLayerWeights> for HiddenExecutor<'_, C> {
    type Hidden = Vec<Bf16>;
    type Error = PrefixScoringError;
    fn layer_forward(
        &mut self, binding: &LayerBinding<'_, HfBf16EagerLayerWeights>, hidden: &mut Self::Hidden,
        positions: PositionContext, kv_cache: &mut KvCache,
    ) -> Result<(), Self::Error> {
        checkpoint(self.control, positions.cache_position)?;
        *hidden = run_hf_bf16_layer(binding.weights(), binding.loop_index(), binding.layer_index(),
            binding.kv_slot(), hidden, positions, self.rope, kv_cache)?;
        self.completed_layers += 1;
        Ok(())
    }
    fn final_rms_norm(&mut self, hidden: &mut Self::Hidden, positions: PositionContext) -> Result<(), Self::Error> {
        if self.completed_layers != PHYSICAL_LAYER_COUNT && self.completed_layers != KV_SLOT_COUNT {
            return Err(PrefixScoringError::InvalidExecution);
        }
        checkpoint(self.control, positions.cache_position)?;
        *hidden = rms_norm_f32_reduce_cast_back(hidden, self.final_norm, RMS_NORM_EPSILON)
            .map_err(HfBf16LayerError::from).map_err(HfBf16EagerError::from)?;
        Ok(())
    }
}

/// Real work for the scorer's deterministic prefix-first traversal. A distinct
/// continuation prefix needs one forward; each complete candidate adds one EOS
/// projection edge, but EOS is never fed into KV. Call after plan validation.
fn work_bound(candidates: &[Candidate], prompt_len: usize, mode: ScoringMode) -> Result<PrefixWork, PrefixScoringError> {
    if prompt_len == 0 || candidates.is_empty() { return Err(PrefixScoringError::InvalidPrompt); }
    let mut ordered = Vec::new();
    ordered.try_reserve_exact(candidates.len()).map_err(|_| PrefixScoringError::AllocationRefused)?;
    ordered.extend(candidates.iter().map(|c| c.continuation().token_ids()));
    ordered.sort_unstable();
    let mut previous: &[u32] = &[];
    let mut edges = 0_u64;
    for tokens in ordered {
        if tokens.is_empty() || tokens == previous { return Err(PrefixScoringError::InvalidPrefix); }
        let common = previous.iter().zip(tokens).take_while(|(a, b)| a == b).count();
        edges = edges.checked_add((tokens.len() - common) as u64).ok_or(PrefixScoringError::ArithmeticOverflow)?;
        previous = tokens;
    }
    let prefixes = edges.checked_add(1).ok_or(PrefixScoringError::ArithmeticOverflow)?;
    let projected = match mode {
        ScoringMode::FullVocabulary => prefixes.checked_mul(NANBEIGE_VOCAB_SIZE as u64),
        _ => edges.checked_add(candidates.len() as u64),
    }.ok_or(PrefixScoringError::ArithmeticOverflow)?;
    Ok(PrefixWork { prefix_evaluations: prefixes,
        forward_positions: (prompt_len as u64).checked_add(edges).ok_or(PrefixScoringError::ArithmeticOverflow)?,
        prompt_positions: prompt_len as u64, continuation_positions: edges,
        projected_logits: projected, rewound_positions: 0 })
}

/// Execute the same validated classification plan with native prefix reuse.
/// Score space, EOS treatment, candidate completeness and policy are unchanged;
/// the versioned run records the execution strategy and ACTUAL native work.
pub fn classify_eager_cached(
    engine: &mut HfBf16EagerEngine, task: &TaskPlan, options: ClassificationOptions,
    scoring_limits: ScoringLimits, budget: PrefixBudget,
) -> Result<CachedClassificationRun, PrefixScoringError> {
    classify_eager_cached_with_control(engine, task, options, scoring_limits, budget, &mut super::Continue)
}
pub fn classify_eager_cached_with_control<C: DecodeStepControl>(
    engine: &mut HfBf16EagerEngine, task: &TaskPlan, options: ClassificationOptions,
    scoring_limits: ScoringLimits, budget: PrefixBudget, control: &mut C,
) -> Result<CachedClassificationRun, PrefixScoringError> {
    let classifier = ClassificationPlan::from_task_plan(task, options, scoring_limits)?;
    let ir = task.ir();
    let DecodeStrategy::PrefillOnly { candidates } = ir.decode_strategy() else {
        return Err(ClassificationError::WrongDecodeStrategy.into());
    };
    let prompt_len = ir.prompt_segments().iter().try_fold(0_usize, |total, segment|
        total.checked_add(segment.token_ids().len()).ok_or(PrefixScoringError::ArithmeticOverflow))?;
    let expected = work_bound(candidates, prompt_len, options.mode)?;
    if expected.forward_positions > budget.max_forward_positions || expected.projected_logits > budget.max_projected_logits {
        return Err(PrefixScoringError::WorkBudget);
    }
    let max_prefix = candidates.iter().map(|c| c.continuation().token_ids().len()).max().unwrap_or(0);
    let mut prompt = Vec::new();
    prompt.try_reserve_exact(prompt_len).map_err(|_| PrefixScoringError::AllocationRefused)?;
    prompt.extend(ir.prompt_segments().iter().flat_map(|s| s.token_ids().iter().copied()));
    let mut session = EagerPrefixSession::new(engine, &prompt, max_prefix, ir.budget().max_kv_bytes, budget, control)?;
    let result = match classifier.execute(&mut session) {
        Ok(result) => result,
        Err(error) => return Err(session.last_error.clone().unwrap_or_else(|| error.into())),
    };
    let actual = session.work();
    // Backtracking is traversal-dependent, but all other real work counts are
    // exact. A future scorer reordering must update this contract explicitly.
    if actual.prefix_evaluations != expected.prefix_evaluations
        || actual.forward_positions != expected.forward_positions
        || actual.prompt_positions != expected.prompt_positions
        || actual.continuation_positions != expected.continuation_positions
        || actual.projected_logits != expected.projected_logits {
        return Err(PrefixScoringError::InvalidExecution);
    }
    drop(session); // Empty logical cache also on successful completion.
    let run = CachedClassificationRun { schema_version: 1, execution: PREFIX_EXECUTION_VERSION.to_owned(),
        numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(), result, native_work: actual };
    if canonjson::canonical_bytes(&run).map_err(|_| PrefixScoringError::Serialization)?.len() as u64 > ir.budget().max_output_bytes {
        return Err(PrefixScoringError::OutputBudget);
    }
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::ir::TokenSequence;
    fn candidates() -> Vec<Candidate> {
        vec![Candidate::new("a", TokenSequence::new(vec![1])),
            Candidate::new("b", TokenSequence::new(vec![1, 2])),
            Candidate::new("c", TokenSequence::new(vec![1, 3]))]
    }
    #[test]
    fn real_work_is_one_prefill_plus_unique_prefix_edges() {
        let work = work_bound(&candidates(), 100, ScoringMode::TrieConditional).unwrap();
        assert_eq!(work.forward_positions, 103);
        assert_eq!(work.prompt_positions, 100);
        assert_eq!(work.prefix_evaluations, 4);
        assert_eq!(work.projected_logits, 6); // Three continuation and three EOS edges.
        let full = work_bound(&candidates(), 100, ScoringMode::FullVocabulary).unwrap();
        assert_eq!(full.projected_logits, 4 * NANBEIGE_VOCAB_SIZE as u64);
        let mut reversed = candidates(); reversed.reverse();
        assert_eq!(work_bound(&reversed, 100, ScoringMode::TrieConditional).unwrap(), work);
    }
    #[test]
    fn exact_transition_covers_siblings_repeats_ancestors_and_empty_root() {
        assert_eq!(transition(5, &[], &[], false).unwrap(), Transition { retain: 0, prompt_from: 0, prefix_from: 0 });
        assert_eq!(transition(5, &[1, 2], &[1, 3], true).unwrap(), Transition { retain: 6, prompt_from: 5, prefix_from: 1 });
        assert_eq!(transition(5, &[1, 2], &[1, 2], true).unwrap(), Transition { retain: 7, prompt_from: 5, prefix_from: 2 });
        assert_eq!(transition(5, &[1, 2], &[1], true).unwrap(), Transition { retain: 5, prompt_from: 5, prefix_from: 0 });
        assert_eq!(transition(5, &[1, 2], &[], true).unwrap(), Transition { retain: 4, prompt_from: 4, prefix_from: 0 });
    }
    #[test]
    fn planned_rewinds_and_suffix_appends_reconstruct_every_requested_prefix() {
        let prompt = [7, 8, 9]; let mut current = Vec::new(); let mut previous = Vec::new(); let mut primed = false;
        for next in [&[][..], &[1][..], &[1, 2][..], &[1, 3][..], &[4][..], &[4][..], &[][..], &[1, 2][..], &[1][..]] {
            let t = transition(prompt.len(), &previous, next, primed).unwrap();
            current.truncate(t.retain);
            current.extend_from_slice(&prompt[t.prompt_from..]);
            current.extend_from_slice(&next[t.prefix_from..]);
            assert_eq!(current, prompt.iter().chain(next).copied().collect::<Vec<_>>());
            previous = next.to_vec(); primed = true;
        }
    }
    #[test]
    fn projection_cancellation_uses_prefill_not_generated_token_accounting() {
        struct Control;
        impl DecodeStepControl for Control {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { panic!("scoring generates no tokens"); }
            fn prefill_checkpoint(&mut self, position: usize) -> Option<DecodeCancellationKind> {
                assert_eq!(position, 12); Some(DecodeCancellationKind::CostBudget)
            }
        }
        let mut control = Control;
        assert_eq!(PrefillControl { inner: &mut control, position: 12 }.checkpoint(0), Some(DecodeCancellationKind::CostBudget));
    }
}
