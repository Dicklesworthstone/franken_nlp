//! One already-admitted engine for every semantic field. No model activation,
//! additional KV reservation, thread pool, retry, or acceptance policy.
use serde::{Deserialize, Serialize};
use crate::{native_engine::{
    decode::{DecodeCancellationKind, DecodeStepControl},
    hf_bf16_eager::{HfBf16EagerEngine, HF_BF16_EAGER_PROFILE,
        candidate_scoring::{PrefixBudget, PrefixScoringError, PrefixWork}},
}, tasks::judge::JudgeNativeError};
use super::*;

pub const SEMANTIC_NATIVE_EXECUTION: &str = "eager-complete-correlated-extraction-fields-v1";
#[derive(Debug)]
pub enum SemanticNativeError { Semantic(SemanticError), Native(JudgeNativeError) }
impl fmt::Display for SemanticNativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Semantic(e) => write!(f, "native semantic verification refused: {e}"),
            Self::Native(e) => write!(f, "native semantic verification failed: {e}") }
    }
}
impl Error for SemanticNativeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Semantic(e) => Some(e), Self::Native(e) => Some(e) }
    }
}
impl From<SemanticError> for SemanticNativeError { fn from(e: SemanticError) -> Self { Self::Semantic(e) } }
impl From<JudgeNativeError> for SemanticNativeError { fn from(e: JudgeNativeError) -> Self { Self::Native(e) } }
impl From<PrefixScoringError> for SemanticNativeError { fn from(e: PrefixScoringError) -> Self { Self::Native(e.into()) } }

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticNativeRun {
    pub schema_version: u32,
    pub execution: String,
    pub numerics_profile: String,
    pub result: SemanticExtractionResult,
    /// ONLY the second-reader work. The original extraction's own work is
    /// retained separately inside result.extraction.output, not counted twice.
    pub semantic_native_work: PrefixWork,
}
impl PreparedSemanticVerification {
    pub fn execute_eager(&self, admitted: &[ExecutionIdentity], engine: &mut HfBf16EagerEngine,
        budget: PrefixBudget) -> Result<SemanticNativeRun, SemanticNativeError> {
        self.execute_eager_with_control(admitted, engine, budget, &mut Continue)
    }
    /// `budget` is the aggregate allowance for ALL second-reader fields,
    /// including each field's full source and every evidence window. The
    /// caller retains model/artifact/process admission and supplies each exact
    /// prepared judge identity; nothing is repaired or authorized by this API.
    pub fn execute_eager_with_control<C: DecodeStepControl>(&self, admitted: &[ExecutionIdentity],
        engine: &mut HfBf16EagerEngine, budget: PrefixBudget, control: &mut C)
        -> Result<SemanticNativeRun, SemanticNativeError> {
        self.verify_identities(admitted)?;
        if !engine.kv_cache().all_slots_have_len(0) { return Err(PrefixScoringError::EngineAlreadyPrimed.into()); }
        let mut expected = PrefixBudget { max_forward_positions: 0, max_projected_logits: 0 };
        // Validate every later field's identity, full engine reservation and
        // context before callbacks or the first forward, not halfway through.
        for (judge, identity) in self.judges.iter().zip(admitted) {
            let bound = judge.planned_native_budget()?;
            judge.preflight_eager(identity, engine, bound)?;
            expected = add_budget(expected, bound)?;
        }
        check_budget(expected, budget)?;
        if let Some(cause) = control.prefill_checkpoint(0) { return Err(PrefixScoringError::Cancelled(cause).into()); }
        let mut results = Vec::new(); results.try_reserve_exact(self.judges.len()).map_err(|_| SemanticError::AllocationRefused)?;
        let mut work = PrefixWork::default();
        for (judge, identity) in self.judges.iter().zip(admitted) {
            // The field receives its exact precharged slice, not a fresh copy
            // of the aggregate budget. Shared prefix-engine cleanup empties
            // logical KV between fields while retaining weights and buffers.
            let run = judge.execute_eager_with_control(identity, engine, judge.planned_native_budget()?, control)?;
            let JudgeResult::Faithfulness(result) = run.result else {
                return Err(SemanticError::Contract("semantic native execution returned another judge mode").into());
            };
            work = sum_native_work(work, run.native_work)?;
            results.push(result);
        }
        if work.forward_positions != expected.max_forward_positions || work.projected_logits != expected.max_projected_logits
            || work.prefix_evaluations != self.work.prefix_evaluations as u64 {
            return Err(PrefixScoringError::InvalidExecution.into());
        }
        let run = SemanticNativeRun { schema_version: 1, execution: SEMANTIC_NATIVE_EXECUTION.to_owned(),
            numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(), result: self.finish(results)?, semantic_native_work: work };
        self.check_output(&run)?;
        Ok(run)
    }
}
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn add_budget(a: PrefixBudget, b: PrefixBudget) -> Result<PrefixBudget, PrefixScoringError> {
    Ok(PrefixBudget { max_forward_positions: a.max_forward_positions.checked_add(b.max_forward_positions).ok_or(PrefixScoringError::ArithmeticOverflow)?,
        max_projected_logits: a.max_projected_logits.checked_add(b.max_projected_logits).ok_or(PrefixScoringError::ArithmeticOverflow)? })
}
fn check_budget(required: PrefixBudget, admitted: PrefixBudget) -> Result<(), PrefixScoringError> {
    if required.max_forward_positions > admitted.max_forward_positions || required.max_projected_logits > admitted.max_projected_logits {
        return Err(PrefixScoringError::WorkBudget);
    }
    Ok(())
}
fn sum_native_work(a: PrefixWork, b: PrefixWork) -> Result<PrefixWork, PrefixScoringError> {
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
    fn aggregate_budget_is_not_renewed_for_each_field() {
        let one = PrefixBudget { max_forward_positions: 100, max_projected_logits: 664576 };
        let both = add_budget(one, one).unwrap();
        assert!(check_budget(both, one).is_err()); assert!(check_budget(both, both).is_ok());
        let overflow = PrefixBudget { max_forward_positions: u64::MAX, ..one };
        assert!(add_budget(overflow, one).is_err());
    }
    #[test]
    fn cancellation_retains_its_typed_native_cause() {
        let error: SemanticNativeError = PrefixScoringError::Cancelled(DecodeCancellationKind::Deadline).into();
        assert!(matches!(error, SemanticNativeError::Native(JudgeNativeError::Native(
            PrefixScoringError::Cancelled(DecodeCancellationKind::Deadline)))));
    }
}
