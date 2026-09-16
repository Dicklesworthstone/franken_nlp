//! Independent ordinal criteria and host-computed, all-criteria aggregation.

use serde::{Deserialize, Serialize};
use crate::{
    execution_identity::Sha256Digest,
    native_engine::{decode::DecodeStepControl, hf_bf16_eager::{HfBf16EagerEngine,
        candidate_scoring::PrefixBudget}, lmhead::scoring::{CandidateScores, ScoringWork}},
    tasks::ir::{DecodeStrategy, PromptSegmentKind, TaskPlan},
    tokenizer::specials::TemplateControlIds,
};
use super::{common::{Bundle, reserved}, native, EagerJudgeRun, JudgeError, JudgeLimits, JudgeLogits, JudgeNativeError};

pub const RUBRIC_VERSION: &str = "judge-independent-ordinal-weighted-mean-v1";
pub const MAX_RUBRIC_CRITERIA: usize = 16;
pub const MAX_RUBRIC_SCORE: u8 = 10;
pub const MAX_CRITERION_WEIGHT: u32 = 1_000_000;

/// Integer weights avoid nonfinite/negative numeric inputs. All weights must
/// be positive and explicitly bounded; no criterion disappears at aggregation.
pub struct RubricHeadInput {
    pub criterion_id: String,
    pub weight: u32,
    pub task: TaskPlan,
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RubricPolicy {
    pub minimum_peak_weight_ppm: u32,
    pub maximum_normalized_entropy_ppm: u32,
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RubricDecision { Scored, Abstained }

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RubricCriterionResult {
    pub criterion_id: String,
    pub weight: u32,
    pub decision: RubricDecision,
    pub estimate: Option<f64>,
    pub distribution_mean: f64,
    pub distribution_mode: u8,
    pub normalized_entropy: f64,
    pub scores: CandidateScores,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RubricResult {
    pub schema_version: u32,
    pub task_spec: String,
    pub algorithm: String,
    pub calibration: String,
    pub scale_maximum: u8,
    pub decision: RubricDecision,
    /// Present only when EVERY criterion passes its explicit policy.
    pub weighted_estimate: Option<f64>,
    pub weighted_modal_score: Option<f64>,
    /// Diagnostic projection, retained even when the task abstains.
    pub weighted_distribution_mean: f64,
    pub total_weight: u64,
    pub criteria: Vec<RubricCriterionResult>,
    pub policy: RubricPolicy,
    pub work: ScoringWork,
}

pub struct RubricPlan {
    pub(super) bundle: Bundle,
    criteria: Vec<(String, u32)>,
    scale: u8,
    policy: RubricPolicy,
}
impl RubricPlan {
    /// Each head uses global, instruction, criterion-data, instruction,
    /// document-data, answer-scaffold. Document, scaffold, instructions and
    /// numeric verbalizers must match across criteria. Only criterion-data
    /// differs. The shared integer scale is 0..=scale, with 1 <= scale <= 10.
    pub fn from_task_plans(inputs: &[RubricHeadInput], scale: u8, eos: u32,
        controls: &TemplateControlIds, policy: RubricPolicy, limits: JudgeLimits) -> Result<Self, JudgeError> {
        if inputs.is_empty() || inputs.len() > MAX_RUBRIC_CRITERIA || !(1..=MAX_RUBRIC_SCORE).contains(&scale) {
            return Err(JudgeError::Contract("rubric criterion count or scale"));
        }
        if policy.minimum_peak_weight_ppm > 1_000_000 || policy.maximum_normalized_entropy_ppm > 1_000_000 {
            return Err(JudgeError::Contract("rubric probability thresholds"));
        }
        let mut ordered = reserved(inputs.len())?;
        ordered.extend(inputs.iter());
        ordered.sort_unstable_by(|a, b| a.criterion_id.cmp(&b.criterion_id));
        if ordered.windows(2).any(|w| w[0].criterion_id == w[1].criterion_id) {
            return Err(JudgeError::Contract("duplicate rubric criterion"));
        }
        let mut criteria = reserved(inputs.len())?;
        for input in &ordered {
            check_criterion(&input.criterion_id, input.weight)?;
            validate_head(&input.task, &ordered[0].task, scale)?;
            criteria.push((input.criterion_id.clone(), input.weight));
        }
        #[derive(Serialize)]
        struct Policy<'a> { version: &'static str, scale: u8, criteria: &'a [(String, u32)], thresholds: RubricPolicy }
        let tasks: Vec<_> = ordered.iter().map(|input| &input.task).collect();
        let bundle = Bundle::compile(&tasks, eos, controls,
            &Policy { version: RUBRIC_VERSION, scale, criteria: &criteria, thresholds: policy }, limits)?;
        Ok(Self { bundle, criteria, scale, policy })
    }
    /// Prompt-derived private identity, never public telemetry.
    pub fn binding_digest(&self) -> &Sha256Digest { &self.bundle.binding }
    pub fn planned_work(&self) -> ScoringWork { self.bundle.work }
    pub fn execute<M: JudgeLogits>(&self, model: &mut M) -> Result<RubricResult, JudgeError> {
        self.finish(self.bundle.score(model)?)
    }
    pub fn execute_eager_with_control<C: DecodeStepControl>(&self, engine: &mut HfBf16EagerEngine,
        budget: PrefixBudget, control: &mut C) -> Result<EagerJudgeRun<RubricResult>, JudgeNativeError> {
        let (scores, work) = native::score_bundle(&self.bundle, engine, budget, control)?;
        native::wrap(&self.bundle, self.finish(scores)?, work)
    }
    pub(super) fn finish(&self, scores: Vec<CandidateScores>) -> Result<RubricResult, JudgeError> {
        if scores.len() != self.criteria.len() { return Err(JudgeError::InvalidScores); }
        let mut criteria = reserved(scores.len())?;
        for ((id, weight), scores) in self.criteria.iter().zip(scores) {
            criteria.push(summarize(id, *weight, self.scale, self.policy, scores)?);
        }
        let (total_weight, mean, modal, accepted) = aggregate(&criteria)?;
        let result = RubricResult {
            schema_version: 1, task_spec: "judge-v1".to_owned(), algorithm: RUBRIC_VERSION.to_owned(),
            calibration: "uncalibrated_caller_rubric_not_a_qualified_preset".to_owned(),
            scale_maximum: self.scale, decision: if accepted { RubricDecision::Scored } else { RubricDecision::Abstained },
            weighted_estimate: accepted.then_some(mean), weighted_modal_score: accepted.then_some(modal),
            weighted_distribution_mean: mean, total_weight, criteria, policy: self.policy, work: self.bundle.work,
        };
        self.bundle.check_output(&result)?;
        Ok(result)
    }
}

pub(super) fn check_criterion(id: &str, weight: u32) -> Result<(), JudgeError> {
    if id.is_empty() || id.len() > 64 || !id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
        || weight == 0 || weight > MAX_CRITERION_WEIGHT {
        return Err(JudgeError::Contract("rubric criterion identifier or weight"));
    }
    Ok(())
}
fn validate_head(task: &TaskPlan, first: &TaskPlan, scale: u8) -> Result<(), JudgeError> {
    use PromptSegmentKind::{GlobalPolicy, TaskInstruction, Document, AnswerScaffold};
    let layout = [GlobalPolicy, TaskInstruction, Document, TaskInstruction, Document, AnswerScaffold];
    let segments = task.ir().prompt_segments(); let baseline = first.ir().prompt_segments();
    if segments.len() != layout.len() || baseline.len() != layout.len()
        || segments.iter().zip(layout).any(|(s, k)| s.kind() != k || s.token_ids().is_empty()) {
        return Err(JudgeError::Contract("rubric segmented prompt ABI"));
    }
    if (0..layout.len()).any(|i| i != 2 && segments[i] != baseline[i]) {
        return Err(JudgeError::Contract("rubric document or trusted prompt differs"));
    }
    let DecodeStrategy::PrefillOnly { candidates } = task.ir().decode_strategy() else {
        return Err(JudgeError::Contract("rubric requires finite scores"));
    };
    let DecodeStrategy::PrefillOnly { candidates: expected } = first.ir().decode_strategy() else {
        return Err(JudgeError::Contract("rubric requires finite scores"));
    };
    if candidates.len() != usize::from(scale) + 1 || candidates.len() != expected.len()
        || (0..=scale).any(|point| {
            let id = format!("score-{point}");
            let a = candidates.iter().find(|c| c.id() == id);
            let b = expected.iter().find(|c| c.id() == id);
            a.is_none() || b.is_none() || a != b
        }) { return Err(JudgeError::Contract("rubric requires the complete shared ordinal scale")); }
    Ok(())
}
fn summarize(id: &str, weight: u32, scale: u8, policy: RubricPolicy, scores: CandidateScores)
    -> Result<RubricCriterionResult, JudgeError> {
    let mut mean = 0.0; let mut entropy = 0.0; let mut peak = 0.0_f64;
    let mut mode = 0; let mut best = f64::NEG_INFINITY;
    for point in 0..=scale {
        let id = format!("score-{point}");
        let score = scores.candidates.iter().find(|c| c.id == id).ok_or(JudgeError::InvalidScores)?;
        mean += f64::from(point) * score.candidate_weight;
        if score.candidate_weight > 0.0 { entropy -= score.candidate_weight * score.candidate_weight.ln(); }
        peak = peak.max(score.candidate_weight);
        // Raw log probabilities, not underflowed exp weights. Exact ties use
        // the lower numeric score, independent of lexical candidate-id order.
        if score.sequence_score > best { best = score.sequence_score; mode = point; }
    }
    let entropy = (entropy / (f64::from(scale) + 1.0).ln()).clamp(0.0, 1.0);
    let accepted = peak >= f64::from(policy.minimum_peak_weight_ppm) / 1_000_000.0
        && entropy <= f64::from(policy.maximum_normalized_entropy_ppm) / 1_000_000.0;
    Ok(RubricCriterionResult { criterion_id: id.to_owned(), weight,
        decision: if accepted { RubricDecision::Scored } else { RubricDecision::Abstained },
        estimate: accepted.then_some(mean), distribution_mean: mean, distribution_mode: mode,
        normalized_entropy: entropy, scores })
}
fn aggregate(criteria: &[RubricCriterionResult]) -> Result<(u64, f64, f64, bool), JudgeError> {
    let mut total = 0_u64; let mut sum = 0.0; let mut modal = 0.0; let mut accepted = true;
    for criterion in criteria {
        total = total.checked_add(u64::from(criterion.weight)).ok_or(JudgeError::Limit("rubric_weight"))?;
        sum += f64::from(criterion.weight) * criterion.distribution_mean;
        modal += f64::from(criterion.weight) * f64::from(criterion.distribution_mode);
        accepted &= criterion.decision == RubricDecision::Scored;
    }
    if total == 0 || !sum.is_finite() || !modal.is_finite() { return Err(JudgeError::InvalidScores); }
    Ok((total, sum / total as f64, modal / total as f64, accepted))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{native_engine::lmhead::scoring::CandidateScore, tasks::ir::ScoreSpace};
    fn scores(weights: &[f64]) -> CandidateScores {
        CandidateScores { score_space: ScoreSpace::FullVocabSequenceLogprob, normalization_scope: "fixture".to_owned(),
            length_rule: "fixture".to_owned(), eos_rule: "fixture".to_owned(), eos_token_id: 0, full_vocab_denominators_computed: true,
            candidates: weights.iter().enumerate().map(|(i, &w)| CandidateScore { id: format!("score-{i}"), scored_tokens: 2,
                sequence_score: w.max(1e-300).ln(), candidate_weight: w }).collect(),
            work: ScoringWork { prefix_evaluations: 0, scored_edges: 0, projected_logits: 0 } }
    }
    fn policy() -> RubricPolicy { RubricPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 900000 } }
    #[test]
    fn criteria_have_independent_distributions_and_host_weighted_totals() {
        let a = summarize("accuracy", 3, 2, policy(), scores(&[0.0, 0.0, 1.0])).unwrap();
        let b = summarize("style", 1, 2, policy(), scores(&[1.0, 0.0, 0.0])).unwrap();
        let (weight, mean, modal, accepted) = aggregate(&[a, b]).unwrap();
        assert_eq!((weight, mean, modal, accepted), (4, 1.5, 1.5, true));
    }
    #[test]
    fn diffuse_criterion_abstains_without_being_dropped_from_the_total() {
        let a = summarize("accuracy", 3, 2, policy(), scores(&[0.0, 0.0, 1.0])).unwrap();
        let b = summarize("style", 1, 2, policy(), scores(&[1.0/3.0; 3])).unwrap();
        assert_eq!(b.estimate, None);
        let (weight, mean, _, accepted) = aggregate(&[a, b]).unwrap();
        assert_eq!(weight, 4); assert!((mean - 1.75).abs() < 1e-12); assert!(!accepted);
    }
    #[test]
    fn numeric_ties_are_conservative_and_missing_bins_refuse() {
        let result = summarize("criterion", 1, 2, policy(), scores(&[0.5, 0.0, 0.5])).unwrap();
        assert_eq!(result.distribution_mode, 0);
        assert!(summarize("criterion", 1, 2, policy(), scores(&[0.5, 0.5])).is_err());
    }
    #[test]
    fn malformed_weights_and_identifiers_fail_before_inference() {
        assert!(check_criterion("quality", 0).is_err());
        assert!(check_criterion("quality", MAX_CRITERION_WEIGHT + 1).is_err());
        assert!(check_criterion("private text\n", 1).is_err());
        assert!(check_criterion("quality-v1", 1).is_ok());
        assert!(aggregate(&[]).is_err());
    }
}
