//! Execute all classification heads on ONE caller-admitted eager engine.
//! Each head uses the existing exact-prefix session, including scored EOS,
//! full-vocabulary denominators, cancellation and automatic KV cleanup.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{
    canonjson,
    execution_identity::ExecutionIdentity,
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        hf_bf16_eager::{HfBf16EagerEngine, HF_BF16_EAGER_PROFILE,
            candidate_scoring::{EagerPrefixSession, PrefixBudget, PrefixScoringError, PrefixWork}},
        kv::KV_BYTES_PER_TOKEN, lmhead::scoring::ScoringMode},
};
use super::{ClassificationError, ClassificationTaskResult, PreparedClassification,
    planning::{self, ClassificationPlanningError}};

pub const CLASSIFICATION_NATIVE_EXECUTION: &str = "eager-full-vocabulary-classification-prefix-heads-v1";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EagerClassificationRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub result: ClassificationTaskResult,
    pub native_work: PrefixWork,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClassificationNativeError {
    Planning(ClassificationPlanningError), Native(PrefixScoringError),
    EngineAlreadyPrimed, ContextBudget, KvBudget, WorkBudget, ExecutionDiverged,
    Cancelled(DecodeCancellationKind),
}
impl fmt::Display for ClassificationNativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Planning(_) => "native classification planning or finalization failed",
            Self::Native(_) => "native classification prefix execution failed",
            Self::EngineAlreadyPrimed => "native classification requires an empty admitted engine",
            Self::ContextBudget => "a complete classification head exceeds admitted context",
            Self::KvBudget => "engine KV reservation exceeds classification budget",
            Self::WorkBudget => "classification aggregate native work budget exceeded",
            Self::ExecutionDiverged => "classification native execution disagrees with its complete plan",
            Self::Cancelled(_) => "native classification cancelled",
        })
    }
}
impl Error for ClassificationNativeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Planning(e) => Some(e), Self::Native(e) => Some(e), _ => None }
    }
}
impl From<ClassificationPlanningError> for ClassificationNativeError {
    fn from(e: ClassificationPlanningError) -> Self {
        match e { ClassificationPlanningError::Cancelled(reason) => Self::Cancelled(reason), other => Self::Planning(other) }
    }
}
impl From<PrefixScoringError> for ClassificationNativeError {
    fn from(e: PrefixScoringError) -> Self {
        match e { PrefixScoringError::Cancelled(reason) => Self::Cancelled(reason), other => Self::Native(other) }
    }
}

impl PreparedClassification {
    /// Model identity is checked as a whole, not repaired at execution time.
    /// All heads and the FULL resident KV allocation are preflighted before a
    /// first forward. This method neither admits resources nor mutates the KV.
    pub fn preflight_eager(&self, admitted: &ExecutionIdentity, engine: &HfBf16EagerEngine,
        budget: PrefixBudget) -> Result<(), ClassificationNativeError> {
        self.verify_identity(admitted)?;
        check_budget(self.work, budget)?;
        let capacity = engine.kv_cache().capacity_positions();
        let mut requirements = planning::reserved(self.heads.len())?;
        for head in &self.heads {
            requirements.push(head.prompt_len.checked_add(head.max_prefix).ok_or(ClassificationNativeError::ContextBudget)?);
        }
        check_capacity(engine.kv_cache().all_slots_have_len(0), capacity, &requirements, self.budget.max_kv_bytes)
    }

    /// No loader, runtime, model clone, alternate numerical implementation or
    /// per-label renewable work allowance. An error never becomes an abstention
    /// or a partial multi-label result. The host owns admission/output guards.
    pub fn execute_eager_with_control<C: DecodeStepControl>(&self, admitted: &ExecutionIdentity,
        engine: &mut HfBf16EagerEngine, budget: PrefixBudget, control: &mut C)
        -> Result<EagerClassificationRun, ClassificationNativeError> {
        planning::checkpoint(control)?;
        self.preflight_eager(admitted, engine, budget)?;
        let mut completed = PrefixWork::default();
        let result = self.execute_heads::<ClassificationNativeError, _>(|head| {
            planning::checkpoint(control)?;
            let mut prompt = planning::reserved(head.prompt_len)?;
            prompt.extend(head.task.ir().prompt_segments().iter().flat_map(|s| s.token_ids().iter().copied()));
            if prompt.len() != head.prompt_len { return Err(ClassificationNativeError::ExecutionDiverged); }
            let head_budget = PrefixBudget { max_forward_positions: head.work.forward_positions,
                max_projected_logits: head.work.projected_logits };
            let mut session = EagerPrefixSession::new(engine, &prompt, head.max_prefix,
                self.budget.max_kv_bytes, head_budget, control)?;
            let scores = match head.classifier.scorer.score(&mut session, ScoringMode::FullVocabulary) {
                Ok(scores) => scores,
                Err(error) => return Err(match session.last_error() {
                    Some(native) => native.clone().into(),
                    None => ClassificationPlanningError::Task(ClassificationError::Scoring(error)).into(),
                }),
            };
            let actual = session.work();
            check_work(actual, head.work)?;
            drop(session);
            if !engine.kv_cache().all_slots_have_len(0) { return Err(ClassificationNativeError::ExecutionDiverged); }
            completed = planning::add_work(completed, actual)?;
            Ok(scores)
        })?;
        check_work(completed, self.work)?;
        let run = EagerClassificationRun { schema_version: 1, execution: CLASSIFICATION_NATIVE_EXECUTION.to_owned(),
            numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(), result, native_work: completed };
        planning::check_output(&run, self.budget.max_output_bytes)?;
        // Serialize the ORIGINAL value: a sizing pass's JSON would turn NaN
        // into null and lose canonjson's independent nonfinite-number check.
        let bytes = canonjson::canonical_bytes(&run).map_err(|_| ClassificationPlanningError::Serialization)?;
        if bytes.len() as u64 > self.budget.max_output_bytes { return Err(ClassificationPlanningError::OutputBudget.into()); }
        planning::checkpoint(control)?;
        Ok(run)
    }
}
fn check_budget(work: PrefixWork, budget: PrefixBudget) -> Result<(), ClassificationNativeError> {
    if work.forward_positions > budget.max_forward_positions || work.projected_logits > budget.max_projected_logits {
        return Err(ClassificationNativeError::WorkBudget);
    }
    Ok(())
}
fn check_capacity(empty: bool, capacity: usize, requirements: &[usize], kv_budget: u64) -> Result<(), ClassificationNativeError> {
    if !empty { return Err(ClassificationNativeError::EngineAlreadyPrimed); }
    if requirements.is_empty() || requirements.iter().any(|&n| n == 0 || n > capacity) {
        return Err(ClassificationNativeError::ContextBudget);
    }
    let bytes = (capacity as u64).checked_mul(KV_BYTES_PER_TOKEN as u64).ok_or(ClassificationNativeError::KvBudget)?;
    if bytes > kv_budget { return Err(ClassificationNativeError::KvBudget); }
    Ok(())
}
fn check_work(actual: PrefixWork, expected: PrefixWork) -> Result<(), ClassificationNativeError> {
    if actual.prefix_evaluations != expected.prefix_evaluations || actual.forward_positions != expected.forward_positions
        || actual.prompt_positions != expected.prompt_positions || actual.continuation_positions != expected.continuation_positions
        || actual.projected_logits != expected.projected_logits {
        return Err(ClassificationNativeError::ExecutionDiverged);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn work() -> PrefixWork {
        PrefixWork { prefix_evaluations: 6, forward_positions: 24, prompt_positions: 20,
            continuation_positions: 4, projected_logits: 6 * 166_144, rewound_positions: 2 }
    }
    #[test]
    fn all_heads_preflight_against_one_full_kv_reservation() {
        let bytes = 12 * KV_BYTES_PER_TOKEN as u64;
        assert!(check_capacity(true, 12, &[10, 12, 11], bytes).is_ok());
        assert_eq!(check_capacity(true, 12, &[10, 13], bytes), Err(ClassificationNativeError::ContextBudget));
        assert_eq!(check_capacity(false, 12, &[10], bytes), Err(ClassificationNativeError::EngineAlreadyPrimed));
        assert_eq!(check_capacity(true, 12, &[10], bytes - 1), Err(ClassificationNativeError::KvBudget));
        assert!(check_capacity(true, usize::MAX, &[1], u64::MAX).is_err());
    }
    #[test]
    fn whole_run_budget_does_not_renew_between_labels() {
        let w = work();
        assert!(check_budget(w, PrefixBudget { max_forward_positions: 24, max_projected_logits: 6 * 166_144 }).is_ok());
        assert_eq!(check_budget(w, PrefixBudget { max_forward_positions: 23, max_projected_logits: u64::MAX }), Err(ClassificationNativeError::WorkBudget));
        assert_eq!(check_budget(w, PrefixBudget { max_forward_positions: u64::MAX, max_projected_logits: w.projected_logits - 1 }), Err(ClassificationNativeError::WorkBudget));
    }
    #[test]
    fn native_receipt_checks_every_exact_work_axis() {
        let expected = work();
        for axis in 0..5 {
            let mut actual = expected;
            match axis { 0 => actual.prefix_evaluations += 1, 1 => actual.forward_positions += 1,
                2 => actual.prompt_positions += 1, 3 => actual.continuation_positions += 1, _ => actual.projected_logits += 1 }
            assert_eq!(check_work(actual, expected), Err(ClassificationNativeError::ExecutionDiverged));
        }
    }
    #[test]
    fn native_and_planning_cancellation_preserve_deadline() {
        let expected = ClassificationNativeError::Cancelled(DecodeCancellationKind::Deadline);
        assert_eq!(ClassificationNativeError::from(PrefixScoringError::Cancelled(DecodeCancellationKind::Deadline)), expected);
        assert_eq!(ClassificationNativeError::from(ClassificationPlanningError::Cancelled(DecodeCancellationKind::Deadline)), expected);
    }
}
