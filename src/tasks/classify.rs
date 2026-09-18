//! Exclusive finite-label classification over the shared continuation scorer.
//!
//! This executes an already-tokenized TaskIR, not an alternate text/template
//! path. The caller binds the model to the plan's trusted prompt segments and
//! owns runtime, context/KV, and model admission. No artifact gate is bypassed.
//! Multi-label classification needs independently qualified binary decisions;
//! a distribution over exclusive candidates must not masquerade as that API.

use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};

use crate::{
    canonjson,
    execution_identity::Sha256Digest,
    native_engine::lmhead::{
        NANBEIGE_VOCAB_SIZE,
        scoring::{CandidateLogits, CandidateScorer, CandidateScores, ScoringError, ScoringLimits, ScoringMode},
    },
};

use super::ir::{DecodeStrategy, ScoreSpace, TaskIR, TaskPlan};

const POLICY_SCALE: u32 = 1_000_000;

/// An explicit, uncalibrated decision rule, in millionths of candidate weight.
/// These thresholds are not accuracy or correctness-confidence guarantees.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationPolicy {
    pub minimum_candidate_weight_ppm: u32,
    pub minimum_margin_ppm: u32,
}

impl ClassificationPolicy {
    fn validate(self) -> Result<(), ClassificationError> {
        if self.minimum_candidate_weight_ppm > POLICY_SCALE
            || self.minimum_margin_ppm > POLICY_SCALE
        {
            return Err(ClassificationError::InvalidPolicy);
        }
        Ok(())
    }
}

/// Scoring semantics must be bound along with the TaskIR before model execution.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationOptions {
    pub mode: ScoringMode,
    pub eos_token_id: u32,
    pub policy: ClassificationPolicy,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassificationDecision {
    Classified,
    Abstained,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassificationCalibration {
    Uncalibrated,
}

/// Complete deterministic result. Abstention is a successful typed decision.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationResult {
    pub schema_version: u32,
    pub decision: ClassificationDecision,
    pub selected_id: Option<String>,
    /// Complete ranking by raw sequence score, then lexical id. Sorting only
    /// weights would incorrectly tie distinct scores after exp underflow.
    pub ranking: Vec<String>,
    pub best_candidate_weight: f64,
    pub candidate_weight_margin: f64,
    pub calibration: ClassificationCalibration,
    pub policy: ClassificationPolicy,
    /// Retains every candidate and the full score-space/EOS disclosure.
    pub scores: CandidateScores,
}

impl ClassificationResult {
    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, ClassificationError> {
        canonjson::canonical_bytes(self).map_err(|_| ClassificationError::Serialization)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClassificationError {
    InvalidPolicy,
    InvalidTaskPlan,
    WrongTask,
    WrongDecodeStrategy,
    Scoring(ScoringError),
    IncompleteScores,
    InvalidScores,
    OutputBudgetExceeded,
    AllocationRefused,
    Serialization,
}

impl fmt::Display for ClassificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicy => f.write_str("classification thresholds must be in 0..=1000000"),
            Self::InvalidTaskPlan => f.write_str("classification TaskIR validation failed"),
            Self::WrongTask => f.write_str("classification requires a classify-v1 task plan"),
            Self::WrongDecodeStrategy => f.write_str("classification requires finite prefill-only candidates"),
            Self::Scoring(error) => write!(f, "classification scoring failed: {error}"),
            Self::IncompleteScores => f.write_str("classification requires every planned candidate exactly once"),
            Self::InvalidScores => f.write_str("classification score metadata or values are invalid"),
            Self::OutputBudgetExceeded => f.write_str("classification output exceeds its byte budget"),
            Self::AllocationRefused => f.write_str("classification allocation refused"),
            Self::Serialization => f.write_str("classification canonical serialization failed"),
        }
    }
}

impl Error for ClassificationError {}

impl From<ScoringError> for ClassificationError {
    fn from(error: ScoringError) -> Self {
        Self::Scoring(error)
    }
}

#[derive(Debug)]
pub struct ClassificationPlan {
    scorer: CandidateScorer,
    options: ClassificationOptions,
    /// Id and scored continuation length, in canonical id order.
    expected: Vec<(String, usize)>,
    max_output_bytes: u64,
    binding_digest: Sha256Digest,
}

impl ClassificationPlan {
    /// Execute only the matching closed-registry task, preserving TaskPlan's
    /// existing identity and resource checks.
    pub fn from_task_plan(
        plan: &TaskPlan,
        options: ClassificationOptions,
        limits: ScoringLimits,
    ) -> Result<Self, ClassificationError> {
        if plan.task_spec_identity() != "classify-v1" {
            return Err(ClassificationError::WrongTask);
        }
        Self::compile_ir(plan.ir(), options, limits)
    }

    /// Compile the internal exact-token execution contract before model work.
    /// This does not grant runtime or artifact authority. `binding_digest`
    /// binds the TaskIR AND score/decision options for the caller's identity.
    pub fn compile_ir(
        ir: &TaskIR,
        options: ClassificationOptions,
        mut limits: ScoringLimits,
    ) -> Result<Self, ClassificationError> {
        ir.validate().map_err(|_| ClassificationError::InvalidTaskPlan)?;
        options.policy.validate()?;
        let DecodeStrategy::PrefillOnly { candidates } = ir.decode_strategy() else {
            return Err(ClassificationError::WrongDecodeStrategy);
        };
        // Scored EOS consumes a continuation position; never widen IR bounds.
        limits.max_depth = limits.max_depth.min(ir.budget().max_output_tokens as usize);
        limits.max_nodes = limits.max_nodes.min(ir.budget().max_grammar_states as usize);
        let scorer = CandidateScorer::compile(
            candidates, NANBEIGE_VOCAB_SIZE, options.eos_token_id, limits,
        )?;
        let mut expected = Vec::new();
        expected.try_reserve_exact(candidates.len())
            .map_err(|_| ClassificationError::AllocationRefused)?;
        expected.extend(candidates.iter().map(|candidate| {
            (candidate.id().to_owned(), candidate.continuation().token_ids().len() + 1)
        }));
        expected.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        #[derive(Serialize)]
        struct Binding<'a> {
            version: &'static str,
            task_ir: &'a TaskIR,
            options: ClassificationOptions,
        }
        let bytes = canonjson::canonical_bytes(&Binding {
            version: "classification-execution-v1", task_ir: ir, options,
        }).map_err(|_| ClassificationError::Serialization)?;
        Ok(Self {
            scorer,
            options,
            expected,
            max_output_bytes: ir.budget().max_output_bytes,
            binding_digest: Sha256Digest::of_bytes(&bytes),
        })
    }

    /// Internal semantic identity input. Contains a digest of private prompt
    /// tokens: never log/export it as a public receipt or telemetry field.
    #[must_use]
    pub const fn binding_digest(&self) -> &Sha256Digest {
        &self.binding_digest
    }

    /// All candidates must score and independently pass finalization before
    /// any result is returned. The backend must be bound to this plan's prompt.
    pub fn execute<M: CandidateLogits>(
        &self,
        model: &mut M,
    ) -> Result<ClassificationResult, ClassificationError> {
        let scores = self.scorer.score(model, self.options.mode)?;
        self.finalize(scores)
    }

    fn finalize(&self, scores: CandidateScores) -> Result<ClassificationResult, ClassificationError> {
        if scores.candidates.len() != self.expected.len() {
            return Err(ClassificationError::IncompleteScores);
        }
        let expected_space = match self.options.mode {
            ScoringMode::FullVocabulary => ScoreSpace::FullVocabSequenceLogprob,
            ScoringMode::TrieConditional => ScoreSpace::TrieLocalConditionalProbability,
            ScoringMode::SequenceSoftmax { .. } => ScoreSpace::SequenceScoreSoftmax,
        };
        if scores.score_space != expected_space
            || scores.eos_token_id != self.options.eos_token_id
            || scores.full_vocab_denominators_computed != (self.options.mode == ScoringMode::FullVocabulary)
        {
            return Err(ClassificationError::InvalidScores);
        }
        let mut total = 0.0;
        for ((id, length), score) in self.expected.iter().zip(&scores.candidates) {
            if id != &score.id || *length != score.scored_tokens {
                return Err(ClassificationError::IncompleteScores);
            }
            if !score.sequence_score.is_finite() || !score.candidate_weight.is_finite()
                || !(0.0..=1.0).contains(&score.candidate_weight)
            {
                return Err(ClassificationError::InvalidScores);
            }
            total += score.candidate_weight;
        }
        // Summation roundoff scales with candidate count; it is not permission
        // to accept a truncated distribution or silently renormalize one.
        let tolerance = 32.0 * f64::EPSILON * self.expected.len() as f64;
        if (total - 1.0).abs() > tolerance {
            return Err(ClassificationError::InvalidScores);
        }
        let mut order = Vec::new();
        order.try_reserve_exact(scores.candidates.len())
            .map_err(|_| ClassificationError::AllocationRefused)?;
        order.extend(0..scores.candidates.len());
        order.sort_unstable_by(|&a, &b| {
            let left = &scores.candidates[a];
            let right = &scores.candidates[b];
            if left.sequence_score == right.sequence_score {
                left.id.cmp(&right.id)
            } else {
                right.sequence_score.total_cmp(&left.sequence_score)
            }
        });
        let best = &scores.candidates[order[0]];
        let runner_up = order.get(1).map_or(0.0, |&index| scores.candidates[index].candidate_weight);
        let margin = (best.candidate_weight - runner_up).max(0.0);
        let accepted = best.candidate_weight * f64::from(POLICY_SCALE)
            >= f64::from(self.options.policy.minimum_candidate_weight_ppm)
            && margin * f64::from(POLICY_SCALE) >= f64::from(self.options.policy.minimum_margin_ppm);
        let mut ranking = Vec::new();
        ranking.try_reserve_exact(order.len()).map_err(|_| ClassificationError::AllocationRefused)?;
        ranking.extend(order.into_iter().map(|index| scores.candidates[index].id.clone()));
        let result = ClassificationResult {
            schema_version: 1,
            decision: if accepted { ClassificationDecision::Classified } else { ClassificationDecision::Abstained },
            selected_id: accepted.then(|| best.id.clone()),
            ranking,
            best_candidate_weight: best.candidate_weight,
            candidate_weight_margin: margin,
            calibration: ClassificationCalibration::Uncalibrated,
            policy: self.options.policy,
            scores,
        };
        let bytes = result.canonical_json_bytes()?;
        if bytes.len() as u64 > self.max_output_bytes {
            return Err(ClassificationError::OutputBudgetExceeded);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ir::{Candidate, DependencyScope, FinitePostcondition, GrammarReference,
        PromptSegment, PromptSegmentKind, TaskBudget, TokenSequence};
    use crate::native_engine::lmhead::scoring::ProjectionRows;

    fn ir() -> TaskIR {
        TaskIR::new(
            vec![PromptSegment::new(PromptSegmentKind::Document, vec![3]),
                PromptSegment::new(PromptSegmentKind::AnswerScaffold, vec![4])],
            DecodeStrategy::PrefillOnly { candidates: vec![
                Candidate::new("b", TokenSequence::new(vec![2])),
                Candidate::new("a", TokenSequence::new(vec![1])),
            ] }, GrammarReference::none(), None,
            vec![FinitePostcondition::CandidateSetComplete, FinitePostcondition::OutputWithinBudget],
            TaskBudget { max_input_tokens: 16, max_output_tokens: 8, max_output_bytes: 65_536,
                max_grammar_states: 64, max_kv_bytes: 1_000_000 }, DependencyScope::ItemLocal,
        ).unwrap()
    }
    fn options() -> ClassificationOptions {
        ClassificationOptions { mode: ScoringMode::TrieConditional, eos_token_id: 0,
            policy: ClassificationPolicy::default() }
    }
    struct Flat;
    impl CandidateLogits for Flat {
        type Error = &'static str;
        fn project(&mut self, _: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
            Ok(vec![0.0; match rows { ProjectionRows::Selected(ids) => ids.len(),
                ProjectionRows::FullVocabulary { vocabulary_size } => vocabulary_size }])
        }
    }
    fn plan(options: ClassificationOptions) -> ClassificationPlan {
        ClassificationPlan::compile_ir(&ir(), options, ScoringLimits::default()).unwrap()
    }

    #[test]
    fn equal_scores_choose_lexical_id_and_preserve_all_candidates() {
        let result = plan(options()).execute(&mut Flat).unwrap();
        assert_eq!(result.selected_id.as_deref(), Some("a"));
        assert_eq!(result.ranking, ["a", "b"]);
        assert_eq!(result.scores.candidates.len(), 2);
        assert_eq!(result.calibration, ClassificationCalibration::Uncalibrated);
        assert_eq!(result.candidate_weight_margin, 0.0);
    }

    #[test]
    fn abstention_is_a_complete_successful_result() {
        let mut options = options();
        options.policy.minimum_margin_ppm = 1;
        let result = plan(options).execute(&mut Flat).unwrap();
        assert_eq!(result.decision, ClassificationDecision::Abstained);
        assert!(result.selected_id.is_none());
        assert_eq!(result.scores.candidates.len(), 2);
    }

    #[test]
    fn thresholds_cannot_claim_more_than_unit_weight() {
        let mut options = options();
        options.policy.minimum_candidate_weight_ppm = 1_000_001;
        assert!(matches!(ClassificationPlan::compile_ir(&ir(), options, ScoringLimits::default()),
            Err(ClassificationError::InvalidPolicy)));
    }

    #[test]
    fn scoring_semantics_and_policy_change_the_internal_binding() {
        let original = plan(options());
        let mut different = options();
        different.policy.minimum_margin_ppm = 1;
        assert_ne!(original.binding_digest(), plan(different).binding_digest());
        different = options(); different.mode = ScoringMode::FullVocabulary;
        assert_ne!(original.binding_digest(), plan(different).binding_digest());
    }

    #[test]
    fn finalization_refuses_truncated_or_forged_distributions() {
        let plan = plan(options());
        let scores = plan.scorer.score(&mut Flat, options().mode).unwrap();
        let mut incomplete = scores.clone(); incomplete.candidates.pop();
        assert_eq!(plan.finalize(incomplete).unwrap_err(), ClassificationError::IncompleteScores);
        let mut corrupt = scores; corrupt.candidates[0].candidate_weight = f64::NAN;
        assert_eq!(plan.finalize(corrupt).unwrap_err(), ClassificationError::InvalidScores);
    }

    #[test]
    fn result_budget_failure_never_returns_partial_success() {
        let mut plan = plan(options()); plan.max_output_bytes = 1;
        assert_eq!(plan.execute(&mut Flat).unwrap_err(), ClassificationError::OutputBudgetExceeded);
    }

    #[test]
    fn eos_position_cannot_exceed_task_output_bound() {
        let limits = ScoringLimits { max_depth: 1, ..ScoringLimits::default() };
        assert!(matches!(ClassificationPlan::compile_ir(&ir(), options(), limits),
            Err(ClassificationError::Scoring(ScoringError::LimitExceeded("depth")))));
    }

    #[test]
    fn canonical_result_round_trips_without_confidence_claims() {
        let result = plan(options()).execute(&mut Flat).unwrap();
        let bytes = result.canonical_json_bytes().unwrap();
        let round_trip: ClassificationResult = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(result, round_trip);
        assert!(!String::from_utf8(bytes).unwrap().contains("confidence"));
    }
}

// Raw-text requests share the exact continuation scorer above. Multi-label
// decisions are independent binary heads, never an exclusive-score relabeling.
mod planning;
mod native;
pub use planning::{
    CLASSIFICATION_PROMPT_VERSION, ClassificationLabel, ClassificationLimits,
    ClassificationMode, ClassificationPlanner, ClassificationPlanningError,
    ClassificationRequest, ClassificationTaskResult, MultiLabelDecision,
    MultiLabelLabelResult, MultiLabelResult, PreparedClassification,
};
pub use native::{CLASSIFICATION_NATIVE_EXECUTION, ClassificationNativeError, EagerClassificationRun};
mod model;
pub use model::ClassificationLogits;
pub mod quantized;
