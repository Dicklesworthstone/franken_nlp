//! One admitted native engine, one live prefix branch, complete multi-head work.
//! No artifact loader, second KV reservation, worker pool or speedup claim.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::native_engine::{
    decode::{DecodeCancellationKind, DecodeStepControl},
    hf_bf16_eager::{HfBf16EagerEngine, HF_BF16_EAGER_PROFILE,
        candidate_scoring::{EagerPrefixSession, PrefixBudget, PrefixScoringError, PrefixWork}},
    kv::KV_BYTES_PER_TOKEN,
    lmhead::scoring::{CandidateScores, ScoringMode},
};
use super::{common::{Bundle, Head}, JudgeError, PairwisePlan, PairwiseResult};

pub const JUDGE_NATIVE_EXECUTION: &str = "eager-complete-judge-prefix-heads-v1";
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EagerJudgeRun<T> {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub result: T,
    pub native_work: PrefixWork,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JudgeNativeError {
    Task(JudgeError),
    /// Includes the original typed cancellation, allocation or engine cause.
    Native(PrefixScoringError),
}
impl fmt::Display for JudgeNativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Task(error) => write!(f, "native judge refused: {error}"),
            Self::Native(error) => write!(f, "native judge failed: {error}"),
        }
    }
}
impl Error for JudgeNativeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Task(error) => Some(error), Self::Native(error) => Some(error) }
    }
}
impl From<JudgeError> for JudgeNativeError { fn from(e: JudgeError) -> Self { Self::Task(e) } }
impl From<PrefixScoringError> for JudgeNativeError { fn from(e: PrefixScoringError) -> Self { Self::Native(e) } }

impl PairwisePlan {
    /// The caller owns artifact/model/profile/process admission. A TaskPlan
    /// does not grant model activation. Both orders execute, or neither wins.
    pub fn execute_eager(&self, engine: &mut HfBf16EagerEngine, budget: PrefixBudget)
        -> Result<EagerJudgeRun<PairwiseResult>, JudgeNativeError> {
        self.execute_eager_with_control(engine, budget, &mut Continue)
    }
    pub fn execute_eager_with_control<C: DecodeStepControl>(&self,
        engine: &mut HfBf16EagerEngine, budget: PrefixBudget, control: &mut C)
        -> Result<EagerJudgeRun<PairwiseResult>, JudgeNativeError> {
        let (scores, work) = score_bundle(&self.bundle, engine, budget, control)?;
        wrap(&self.bundle, self.finish(scores)?, work)
    }
}
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

pub(super) fn score_bundle<C: DecodeStepControl>(bundle: &Bundle,
    engine: &mut HfBf16EagerEngine, budget: PrefixBudget, control: &mut C)
    -> Result<(Vec<CandidateScores>, PrefixWork), JudgeNativeError> {
    if !engine.kv_cache().all_slots_have_len(0) { return Err(PrefixScoringError::EngineAlreadyPrimed.into()); }
    let capacity = engine.kv_cache().capacity_positions();
    let reservation = (capacity as u64).checked_mul(KV_BYTES_PER_TOKEN as u64)
        .ok_or(PrefixScoringError::ArithmeticOverflow)?;
    let mut expected = PrefixWork::default();
    // Preflight ALL heads before touching the engine, including later prompts.
    for head in &bundle.heads {
        let positions = head.prompt_len.checked_add(head.max_prefix).ok_or(PrefixScoringError::ArithmeticOverflow)?;
        if positions > capacity { return Err(PrefixScoringError::ContextBudget.into()); }
        if reservation > head.ir.budget().max_kv_bytes { return Err(PrefixScoringError::KvBudget.into()); }
        expected = sum_work(expected, head_work(head)?)?;
    }
    check_budget(expected, budget)?;
    let mut completed = PrefixWork::default();
    let scores = bundle.score_with(|_, head| -> Result<CandidateScores, JudgeNativeError> {
        if let Some(cause) = control.prefill_checkpoint(0) { return Err(PrefixScoringError::Cancelled(cause).into()); }
        let bound = head_work(head)?;
        // Exact per-head slices, not renewable copies of the aggregate budget.
        let allowance = PrefixBudget { max_forward_positions: bound.forward_positions,
            max_projected_logits: bound.projected_logits };
        let mut prompt = Vec::new();
        prompt.try_reserve_exact(head.prompt_len).map_err(|_| PrefixScoringError::AllocationRefused)?;
        prompt.extend(head.ir.prompt_segments().iter().flat_map(|s| s.token_ids().iter().copied()));
        let mut session = EagerPrefixSession::new(engine, &prompt, head.max_prefix,
            head.ir.budget().max_kv_bytes, allowance, control)?;
        let scores = match head.scorer.score(&mut session, ScoringMode::FullVocabulary) {
            Ok(scores) => scores,
            Err(error) => return Err(match session.last_error() {
                Some(native) => native.clone().into(), None => JudgeError::Scoring(error).into(),
            }),
        };
        let actual = session.work();
        check_work(actual, bound)?;
        drop(session); // Logical KV clears between prompts; buffers survive.
        completed = sum_work(completed, actual)?;
        Ok(scores)
    })?;
    check_work(completed, expected)?;
    Ok((scores, completed))
}

pub(super) fn wrap<T: Serialize>(bundle: &Bundle, result: T, work: PrefixWork)
    -> Result<EagerJudgeRun<T>, JudgeNativeError> {
    let run = EagerJudgeRun { schema_version: 1, execution: JUDGE_NATIVE_EXECUTION.to_owned(),
        numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(), result, native_work: work };
    bundle.check_output(&run)?;
    Ok(run)
}
fn head_work(head: &Head) -> Result<PrefixWork, PrefixScoringError> {
    work_bound(head.prompt_len, head.work.prefix_evaluations, head.work.projected_logits)
}
fn work_bound(prompt: usize, prefixes: usize, rows: u64) -> Result<PrefixWork, PrefixScoringError> {
    let continuation = prefixes.checked_sub(1).ok_or(PrefixScoringError::InvalidExecution)?;
    let forward = prompt.checked_add(continuation).ok_or(PrefixScoringError::ArithmeticOverflow)?;
    Ok(PrefixWork { prefix_evaluations: prefixes as u64, forward_positions: forward as u64,
        prompt_positions: prompt as u64, continuation_positions: continuation as u64,
        projected_logits: rows, rewound_positions: 0 })
}
fn check_budget(work: PrefixWork, budget: PrefixBudget) -> Result<(), PrefixScoringError> {
    if work.forward_positions > budget.max_forward_positions || work.projected_logits > budget.max_projected_logits {
        return Err(PrefixScoringError::WorkBudget);
    }
    Ok(())
}
fn check_work(actual: PrefixWork, expected: PrefixWork) -> Result<(), PrefixScoringError> {
    // Rewinds are observed. Forwards/prefixes/projections are predicted exactly;
    // the scored EOS edge must not become an extra KV position.
    if actual.prefix_evaluations != expected.prefix_evaluations || actual.forward_positions != expected.forward_positions
        || actual.prompt_positions != expected.prompt_positions || actual.continuation_positions != expected.continuation_positions
        || actual.projected_logits != expected.projected_logits { return Err(PrefixScoringError::InvalidExecution); }
    Ok(())
}
fn sum_work(a: PrefixWork, b: PrefixWork) -> Result<PrefixWork, PrefixScoringError> {
    let add = |a: u64, b: u64| a.checked_add(b).ok_or(PrefixScoringError::ArithmeticOverflow);
    Ok(PrefixWork { prefix_evaluations: add(a.prefix_evaluations, b.prefix_evaluations)?,
        forward_positions: add(a.forward_positions, b.forward_positions)?, prompt_positions: add(a.prompt_positions, b.prompt_positions)?,
        continuation_positions: add(a.continuation_positions, b.continuation_positions)?,
        projected_logits: add(a.projected_logits, b.projected_logits)?, rewound_positions: add(a.rewound_positions, b.rewound_positions)? })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_orders_are_charged_and_eos_is_not_a_kv_forward() {
        let one = work_bound(100, 3, 498432).unwrap(); let both = sum_work(one, one).unwrap();
        assert_eq!(both.prompt_positions, 200); assert_eq!(both.continuation_positions, 4); assert_eq!(both.forward_positions, 204);
        assert!(check_budget(both, PrefixBudget { max_forward_positions: 203, max_projected_logits: 996864 }).is_err());
        assert!(check_budget(both, PrefixBudget { max_forward_positions: 204, max_projected_logits: 996863 }).is_err());
    }
    #[test]
    fn actual_work_cannot_hide_replay_or_missing_denominators() {
        let expected = work_bound(100, 3, 498432).unwrap(); let mut actual = expected; actual.forward_positions += 100;
        assert!(check_work(actual, expected).is_err()); actual = expected; actual.projected_logits = 4;
        assert!(check_work(actual, expected).is_err()); actual = expected; actual.rewound_positions = 1;
        assert!(check_work(actual, expected).is_ok());
    }
    #[test]
    fn cancellation_and_overflow_remain_typed() {
        assert!(work_bound(usize::MAX, 2, 1).is_err());
        let error: JudgeNativeError = PrefixScoringError::Cancelled(DecodeCancellationKind::Deadline).into();
        assert!(matches!(error, JudgeNativeError::Native(PrefixScoringError::Cancelled(DecodeCancellationKind::Deadline))));
    }
}
