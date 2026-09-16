//! Concrete eager-engine classification with an explicitly bounded replay
//! fallback. This is NOT a KV-fork or logit-slicing performance qualification.
//!
//! The caller supplies an already-admitted engine. No file loader, raw source
//! activation route, runtime spawn, weight clone, or retained prefix snapshot
//! is introduced. Each logical scoring prefix replays the exact TaskIR prompt
//! from position zero; all real eager full-vocabulary projections are charged.

use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};

use super::{HfBf16EagerEngine, HfBf16EagerError};
use crate::{
    canonjson,
    native_engine::{
        decode::{DecodeCancellationKind, DecodeStepControl},
        kv::{KV_ELEMENTS_PER_POSITION, KV_SLOT_COUNT},
        lmhead::{NANBEIGE_VOCAB_SIZE, scoring::{CandidateLogits, ProjectionRows, ScoringLimits}},
    },
    tasks::{
        classify::{ClassificationError, ClassificationOptions, ClassificationPlan, ClassificationResult},
        ir::{Candidate, DecodeStrategy, TaskPlan},
    },
};

/// Real replay work caps, separate from the scorer's logical edge-work caps.
#[derive(Clone, Copy, Debug)]
pub struct ReplayBudget {
    pub max_forward_positions: u64,
    pub max_projected_logits: u64,
}

impl Default for ReplayBudget {
    fn default() -> Self {
        Self { max_forward_positions: 4096, max_projected_logits: 4096 * NANBEIGE_VOCAB_SIZE as u64 }
    }
}

/// Actual completed eager work; selected-row requests still pay full rows.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayWork {
    pub prefix_evaluations: u64,
    pub forward_positions: u64,
    pub projected_logits: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EagerClassificationRun {
    pub result: ClassificationResult,
    pub replay_work: ReplayWork,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EagerClassificationError {
    Classification(ClassificationError),
    InvalidPrompt,
    EngineAlreadyPrimed,
    ContextBudgetExceeded,
    KvBudgetExceeded,
    ReplayBudgetExceeded,
    ArithmeticOverflow,
    AllocationRefused,
    InvalidProjection,
    Engine(HfBf16EagerError),
    Cancelled(DecodeCancellationKind),
    OutputBudgetExceeded,
    Serialization,
}

impl fmt::Display for EagerClassificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Classification(error) => write!(f, "eager classification refused: {error}"),
            Self::InvalidPrompt => f.write_str("eager classification requires valid exact prompt tokens"),
            Self::EngineAlreadyPrimed => f.write_str("eager classification requires an empty engine cache"),
            Self::ContextBudgetExceeded => f.write_str("classification continuation exceeds admitted context"),
            Self::KvBudgetExceeded => f.write_str("admitted eager KV capacity exceeds the task budget"),
            Self::ReplayBudgetExceeded => f.write_str("classification replay exceeds its real work budget"),
            Self::ArithmeticOverflow => f.write_str("classification replay arithmetic overflowed"),
            Self::AllocationRefused => f.write_str("classification replay allocation refused"),
            Self::InvalidProjection => f.write_str("classification projection rows or logits are invalid"),
            Self::Engine(_) => f.write_str("classification eager forward failed"),
            Self::Cancelled(kind) => write!(f, "classification replay cancelled: {kind:?}"),
            Self::OutputBudgetExceeded => f.write_str("eager classification output exceeds its byte budget"),
            Self::Serialization => f.write_str("eager classification serialization failed"),
        }
    }
}
impl Error for EagerClassificationError {}
impl From<ClassificationError> for EagerClassificationError {
    fn from(error: ClassificationError) -> Self { Self::Classification(error) }
}

/// Classify a validated exact-token task using the existing native engine.
/// The engine is returned with an empty logical KV cache, not a retained
/// candidate prefix. Artifact and execution-identity admission remain owned
/// by the caller; this function never constructs or loads a model.
pub fn classify_eager(
    engine: &mut HfBf16EagerEngine,
    task: &TaskPlan,
    options: ClassificationOptions,
    scoring_limits: ScoringLimits,
    replay_budget: ReplayBudget,
) -> Result<EagerClassificationRun, EagerClassificationError> {
    classify_eager_with_control(engine, task, options, scoring_limits, replay_budget, &mut Continue)
}

/// As above, preserving a typed cancellation cause at every replay position.
pub fn classify_eager_with_control<C: DecodeStepControl>(
    engine: &mut HfBf16EagerEngine,
    task: &TaskPlan,
    options: ClassificationOptions,
    scoring_limits: ScoringLimits,
    replay_budget: ReplayBudget,
    control: &mut C,
) -> Result<EagerClassificationRun, EagerClassificationError> {
    let classifier = ClassificationPlan::from_task_plan(task, options, scoring_limits)?;
    let ir = task.ir();
    let DecodeStrategy::PrefillOnly { candidates } = ir.decode_strategy() else {
        return Err(ClassificationError::WrongDecodeStrategy.into());
    };
    if !engine.kv_cache.all_slots_have_len(0) {
        return Err(EagerClassificationError::EngineAlreadyPrimed);
    }
    let prompt_len = ir.prompt_segments().iter().try_fold(0_usize, |total, segment| {
        total.checked_add(segment.token_ids().len()).ok_or(EagerClassificationError::ArithmeticOverflow)
    })?;
    if prompt_len == 0 || ir.prompt_segments().iter().flat_map(|s| s.token_ids())
        .any(|&token| token as usize >= NANBEIGE_VOCAB_SIZE)
    {
        return Err(EagerClassificationError::InvalidPrompt);
    }
    let max_prefix = candidates.iter().map(|c| c.continuation().token_ids().len()).max().unwrap_or(0);
    let required_context = prompt_len.checked_add(max_prefix)
        .ok_or(EagerClassificationError::ArithmeticOverflow)?;
    if required_context > engine.kv_cache.capacity_positions() {
        return Err(EagerClassificationError::ContextBudgetExceeded);
    }
    let kv_bytes_per_position = (KV_SLOT_COUNT * KV_ELEMENTS_PER_POSITION * 2 * size_of::<u16>()) as u64;
    let admitted_kv = (engine.kv_cache.capacity_positions() as u64).checked_mul(kv_bytes_per_position)
        .ok_or(EagerClassificationError::ArithmeticOverflow)?;
    if admitted_kv > ir.budget().max_kv_bytes {
        return Err(EagerClassificationError::KvBudgetExceeded);
    }
    // Include every replayed prompt position and every full lm_head call,
    // even when the logical scorer requests only selected candidate rows.
    let expected_work = replay_work_bound(candidates, prompt_len)?;
    check_replay_budget(expected_work, replay_budget)?;
    let mut prompt = Vec::new();
    prompt.try_reserve_exact(prompt_len).map_err(|_| EagerClassificationError::AllocationRefused)?;
    prompt.extend(ir.prompt_segments().iter().flat_map(|segment| segment.token_ids().iter().copied()));
    let mut backend = EagerReplay {
        engine, prompt: &prompt, max_prefix, control,
        remaining_positions: replay_budget.max_forward_positions,
        remaining_logits: replay_budget.max_projected_logits,
        work: ReplayWork::default(), last_error: None,
    };
    let result = match classifier.execute(&mut backend) {
        Ok(result) => result,
        Err(error) => return Err(backend.last_error.take().unwrap_or_else(|| error.into())),
    };
    let run = EagerClassificationRun { result, replay_work: backend.work };
    if run.replay_work != expected_work {
        return Err(EagerClassificationError::InvalidProjection);
    }
    let bytes = canonjson::canonical_bytes(&run).map_err(|_| EagerClassificationError::Serialization)?;
    if bytes.len() as u64 > ir.budget().max_output_bytes {
        return Err(EagerClassificationError::OutputBudgetExceeded);
    }
    Ok(run)
}

/// Calculate the union of label prefixes without allocating one Vec per
/// prefix. EOS is scored from the full label prefix, never fed back afterward.
fn replay_work_bound(candidates: &[Candidate], prompt_len: usize) -> Result<ReplayWork, EagerClassificationError> {
    let mut ordered = Vec::new();
    ordered.try_reserve_exact(candidates.len()).map_err(|_| EagerClassificationError::AllocationRefused)?;
    ordered.extend(candidates.iter().map(|c| c.continuation().token_ids()));
    ordered.sort_unstable();
    let mut previous: &[u32] = &[];
    let mut prefixes = 1_u64; // prompt-only root
    let mut depth_sum = 0_u64;
    for tokens in ordered {
        let common = previous.iter().zip(tokens).take_while(|(a, b)| a == b).count() as u64;
        let depth = tokens.len() as u64;
        prefixes = prefixes.checked_add(depth - common).ok_or(EagerClassificationError::ArithmeticOverflow)?;
        depth_sum = depth_sum.checked_add(triangular(depth)? - triangular(common)?)
            .ok_or(EagerClassificationError::ArithmeticOverflow)?;
        previous = tokens;
    }
    let forward_positions = prefixes.checked_mul(prompt_len as u64)
        .and_then(|positions| positions.checked_add(depth_sum))
        .ok_or(EagerClassificationError::ArithmeticOverflow)?;
    let projected_logits = forward_positions.checked_mul(NANBEIGE_VOCAB_SIZE as u64)
        .ok_or(EagerClassificationError::ArithmeticOverflow)?;
    Ok(ReplayWork { prefix_evaluations: prefixes, forward_positions, projected_logits })
}

fn triangular(n: u64) -> Result<u64, EagerClassificationError> {
    n.checked_add(1).and_then(|next| n.checked_mul(next)).map(|value| value / 2)
        .ok_or(EagerClassificationError::ArithmeticOverflow)
}

fn check_replay_budget(work: ReplayWork, budget: ReplayBudget) -> Result<(), EagerClassificationError> {
    if work.forward_positions > budget.max_forward_positions || work.projected_logits > budget.max_projected_logits {
        return Err(EagerClassificationError::ReplayBudgetExceeded);
    }
    Ok(())
}

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

/// Only a freshly admitted empty engine enters this guard. Clear on ordinary
/// errors, cancellation, and unwinding, while retaining preallocated buffers.
struct ClearCache<'a>(&'a mut HfBf16EagerEngine);
impl Drop for ClearCache<'_> {
    fn drop(&mut self) { self.0.kv_cache.clear(); }
}

struct EagerReplay<'a, C> {
    engine: &'a mut HfBf16EagerEngine,
    prompt: &'a [u32],
    max_prefix: usize,
    control: &'a mut C,
    remaining_positions: u64,
    remaining_logits: u64,
    work: ReplayWork,
    last_error: Option<EagerClassificationError>,
}

impl<C: DecodeStepControl> EagerReplay<'_, C> {
    fn project_inner(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, EagerClassificationError> {
        validate_rows(rows)?;
        if prefix.len() > self.max_prefix || prefix.iter().any(|&id| id as usize >= NANBEIGE_VOCAB_SIZE) {
            return Err(EagerClassificationError::ContextBudgetExceeded);
        }
        let positions = self.prompt.len().checked_add(prefix.len())
            .ok_or(EagerClassificationError::ArithmeticOverflow)? as u64;
        let projected = positions.checked_mul(NANBEIGE_VOCAB_SIZE as u64)
            .ok_or(EagerClassificationError::ArithmeticOverflow)?;
        let remaining_positions = self.remaining_positions.checked_sub(positions)
            .ok_or(EagerClassificationError::ReplayBudgetExceeded)?;
        let remaining_logits = self.remaining_logits.checked_sub(projected)
            .ok_or(EagerClassificationError::ReplayBudgetExceeded)?;
        self.remaining_positions = remaining_positions;
        self.remaining_logits = remaining_logits;
        let mut guard = ClearCache(&mut *self.engine);
        let work = &mut self.work;
        let logits = replay_logits(self.prompt, prefix, &mut *self.control, |token| {
            let forward = guard.0.decode(token).map_err(EagerClassificationError::Engine)?;
            work.forward_positions += 1;
            work.projected_logits += NANBEIGE_VOCAB_SIZE as u64;
            Ok(forward.logits)
        });
        drop(guard);
        let logits = logits?;
        if logits.len() != NANBEIGE_VOCAB_SIZE || logits.iter().any(|value| !value.is_finite()) {
            return Err(EagerClassificationError::InvalidProjection);
        }
        self.work.prefix_evaluations += 1;
        match rows {
            ProjectionRows::FullVocabulary { .. } => Ok(logits),
            ProjectionRows::Selected(ids) => {
                let mut selected = Vec::new();
                selected.try_reserve_exact(ids.len()).map_err(|_| EagerClassificationError::AllocationRefused)?;
                selected.extend(ids.iter().map(|&id| logits[id as usize]));
                Ok(selected)
            }
        }
    }
}
impl<C: DecodeStepControl> CandidateLogits for EagerReplay<'_, C> {
    type Error = EagerClassificationError;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let result = self.project_inner(prefix, rows);
        if let Err(error) = &result { self.last_error = Some(error.clone()); }
        result
    }
}

fn validate_rows(rows: ProjectionRows<'_>) -> Result<(), EagerClassificationError> {
    let valid = match rows {
        ProjectionRows::FullVocabulary { vocabulary_size } => vocabulary_size == NANBEIGE_VOCAB_SIZE,
        ProjectionRows::Selected(ids) => !ids.is_empty()
            && ids.iter().all(|&id| (id as usize) < NANBEIGE_VOCAB_SIZE)
            && ids.windows(2).all(|pair| pair[0] < pair[1]),
    };
    if valid { Ok(()) } else { Err(EagerClassificationError::InvalidProjection) }
}

fn replay_logits<C, F>(prompt: &[u32], prefix: &[u32], control: &mut C, mut forward: F)
    -> Result<Vec<f32>, EagerClassificationError>
where C: DecodeStepControl, F: FnMut(u32) -> Result<Vec<f32>, EagerClassificationError> {
    if prompt.is_empty() { return Err(EagerClassificationError::InvalidPrompt); }
    let total = prompt.len().checked_add(prefix.len()).ok_or(EagerClassificationError::ArithmeticOverflow)?;
    for (index, &token) in prompt.iter().chain(prefix).enumerate() {
        if let Some(kind) = control.prefill_checkpoint(index) {
            return Err(EagerClassificationError::Cancelled(kind));
        }
        let logits = forward(token)?;
        if index + 1 == total { return Ok(logits); }
        drop(logits);
    }
    Err(EagerClassificationError::InvalidPrompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::ir::TokenSequence;

    fn candidates() -> Vec<Candidate> {
        vec![Candidate::new("short", TokenSequence::new(vec![1])),
            Candidate::new("long", TokenSequence::new(vec![1, 2]))]
    }

    #[test]
    fn shared_prefix_work_includes_all_replayed_prompt_positions() {
        let work = replay_work_bound(&candidates(), 5).unwrap();
        // Prefixes [], [1], [1, 2] replay 5 + 6 + 7 positions.
        assert_eq!(work.prefix_evaluations, 3);
        assert_eq!(work.forward_positions, 18);
        assert_eq!(work.projected_logits, 18 * NANBEIGE_VOCAB_SIZE as u64);
    }

    #[test]
    fn work_bound_is_independent_of_candidate_order() {
        let mut inputs = candidates();
        let expected = replay_work_bound(&inputs, 5).unwrap();
        inputs.reverse();
        assert_eq!(replay_work_bound(&inputs, 5).unwrap(), expected);
    }

    #[test]
    fn replay_caps_charge_full_projection_rows_not_selected_edges() {
        let work = replay_work_bound(&candidates(), 5).unwrap();
        assert_eq!(check_replay_budget(work, ReplayBudget { max_forward_positions: 18,
            max_projected_logits: work.projected_logits - 1 }), Err(EagerClassificationError::ReplayBudgetExceeded));
        assert!(check_replay_budget(work, ReplayBudget { max_forward_positions: 18,
            max_projected_logits: work.projected_logits }).is_ok());
    }

    #[test]
    fn work_bound_rejects_overflow_before_engine_use() {
        assert!(replay_work_bound(&candidates(), usize::MAX).is_err());
        assert!(triangular(u64::MAX).is_err());
    }

    #[test]
    fn replay_uses_exact_prompt_then_prefix_and_only_returns_last_logits() {
        let mut observed = Vec::new();
        let output = replay_logits(&[7, 8], &[9, 10], &mut Continue, |token| {
            observed.push(token); Ok(vec![token as f32])
        }).unwrap();
        assert_eq!(observed, vec![7, 8, 9, 10]);
        assert_eq!(output, vec![10.0]);
    }

    #[test]
    fn cancellation_preserves_cause_and_stops_replay() {
        struct Cancel;
        impl DecodeStepControl for Cancel {
            fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
            fn prefill_checkpoint(&mut self, index: usize) -> Option<DecodeCancellationKind> {
                (index == 1).then_some(DecodeCancellationKind::CostBudget)
            }
        }
        let mut observed = Vec::new();
        let result = replay_logits(&[7, 8], &[9], &mut Cancel, |token| {
            observed.push(token); Ok(vec![0.0])
        });
        assert_eq!(result, Err(EagerClassificationError::Cancelled(DecodeCancellationKind::CostBudget)));
        assert_eq!(observed, vec![7]);
    }

    #[test]
    fn malformed_projection_requests_refuse() {
        for ids in [&[][..], &[2, 1][..], &[1, 1][..], &[NANBEIGE_VOCAB_SIZE as u32][..]] {
            assert_eq!(validate_rows(ProjectionRows::Selected(ids)), Err(EagerClassificationError::InvalidProjection));
        }
        assert!(validate_rows(ProjectionRows::Selected(&[0, 1, 2])).is_ok());
        assert!(validate_rows(ProjectionRows::FullVocabulary { vocabulary_size: 2 }).is_err());
    }

    #[test]
    fn forward_errors_stop_before_later_tokens() {
        let mut observed = Vec::new();
        let result = replay_logits(&[7, 8], &[9], &mut Continue, |token| {
            observed.push(token);
            if token == 8 { Err(EagerClassificationError::InvalidProjection) } else { Ok(vec![0.0]) }
        });
        assert_eq!(result, Err(EagerClassificationError::InvalidProjection));
        assert_eq!(observed, vec![7, 8]);
    }
}
