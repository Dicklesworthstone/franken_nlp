//! Two-order preference with exact original-identity remapping.

use serde::{Deserialize, Serialize};
use crate::{
    execution_identity::Sha256Digest,
    native_engine::lmhead::scoring::{CandidateScores, ScoringWork},
    tasks::ir::{DecodeStrategy, PromptSegmentKind, TaskPlan},
    tokenizer::specials::TemplateControlIds,
};
use super::common::{Bundle, JudgeError, JudgeLimits, JudgeLogits};

pub const PAIRWISE_VERSION: &str = "judge-pairwise-mean-oriented-log-odds-v1";

/// Explicit uncalibrated policy. Units are thousandths of a natural-log score
/// ratio. Small combined margins and inconsistent orders may abstain. These
/// thresholds have no calibrated-confidence or population-risk interpretation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairwisePolicy {
    pub minimum_margin_milli: u32,
    pub maximum_order_disagreement_milli: u32,
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PairwiseDecision { PreferA, PreferB, Tie, Abstained }

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairwiseResult {
    pub schema_version: u32,
    pub task_spec: String,
    pub algorithm: String,
    pub calibration: String,
    pub decision: PairwiseDecision,
    /// Positive favors original A, not the first displayed answer.
    pub mean_log_odds_a: f64,
    /// Each order is remapped to original A minus original B BEFORE averaging.
    pub order_log_odds_a: [f64; 2],
    pub order_disagreement: f64,
    /// Logistic of mean_log_odds_a: a candidate-conditional projection only.
    pub candidate_conditional_weight_a: f64,
    /// Complete scores for display A/B, then display B/A.
    pub orders: [CandidateScores; 2],
    pub policy: PairwisePolicy,
    pub work: ScoringWork,
}

pub struct PairwisePlan {
    pub(super) bundle: Bundle,
    policy: PairwisePolicy,
}
impl PairwisePlan {
    /// Both TaskPlans must use the eight-segment pairwise ABI: global,
    /// instruction, criterion-data, instruction, first-data, instruction,
    /// second-data, answer-scaffold. The reverse plan swaps ONLY the answers.
    /// Criteria, instructions, scaffold and first/second continuations match.
    pub fn from_task_plans(ab: &TaskPlan, ba: &TaskPlan, eos: u32, controls: &TemplateControlIds,
        policy: PairwisePolicy, limits: JudgeLimits) -> Result<Self, JudgeError> {
        validate_orders(ab, ba)?;
        #[derive(Serialize)]
        struct Policy { algorithm: &'static str, thresholds: PairwisePolicy }
        let bundle = Bundle::compile(&[ab, ba], eos, controls,
            &Policy { algorithm: PAIRWISE_VERSION, thresholds: policy }, limits)?;
        Ok(Self { bundle, policy })
    }

    /// Prompt-derived private identity input; do not place in public telemetry.
    pub fn binding_digest(&self) -> &Sha256Digest { &self.bundle.binding }
    pub fn planned_work(&self) -> ScoringWork { self.bundle.work }

    pub fn execute<M: JudgeLogits>(&self, model: &mut M) -> Result<PairwiseResult, JudgeError> {
        self.finish(self.bundle.score(model)?)
    }

    pub(super) fn finish(&self, scores: Vec<CandidateScores>) -> Result<PairwiseResult, JudgeError> {
        let orders: [CandidateScores; 2] = scores.try_into().map_err(|_| JudgeError::InvalidScores)?;
        let margin = |scores: &CandidateScores| -> Result<f64, JudgeError> {
            let first = scores.candidates.iter().find(|c| c.id == "first").ok_or(JudgeError::InvalidScores)?;
            let second = scores.candidates.iter().find(|c| c.id == "second").ok_or(JudgeError::InvalidScores)?;
            Ok(first.sequence_score - second.sequence_score)
        };
        let oriented = [margin(&orders[0])?, -margin(&orders[1])?];
        let (mean, disagreement, weight, decision) = combine(oriented[0], oriented[1], self.policy)?;
        let result = PairwiseResult {
            schema_version: 1, task_spec: "judge-v1".to_owned(), algorithm: PAIRWISE_VERSION.to_owned(),
            calibration: "uncalibrated_not_a_correctness_probability".to_owned(),
            decision, mean_log_odds_a: mean, order_log_odds_a: oriented,
            order_disagreement: disagreement, candidate_conditional_weight_a: weight,
            orders, policy: self.policy, work: self.bundle.work,
        };
        self.bundle.check_output(&result)?;
        Ok(result)
    }
}

fn validate_orders(ab: &TaskPlan, ba: &TaskPlan) -> Result<(), JudgeError> {
    use PromptSegmentKind::{GlobalPolicy, TaskInstruction, Document, AnswerScaffold};
    let layout = [GlobalPolicy, TaskInstruction, Document, TaskInstruction, Document,
        TaskInstruction, Document, AnswerScaffold];
    let a = ab.ir().prompt_segments();
    let b = ba.ir().prompt_segments();
    for segments in [a, b] {
        if segments.len() != layout.len() || segments.iter().zip(layout).any(|(s, kind)|
            s.kind() != kind || s.token_ids().is_empty()) {
            return Err(JudgeError::Contract("pairwise segmented prompt ABI"));
        }
    }
    for index in 0..layout.len() {
        let opposite = match index { 4 => 6, 6 => 4, other => other };
        if a[index].token_ids() != b[opposite].token_ids() {
            return Err(JudgeError::Contract("reverse order must swap exactly the answer data"));
        }
    }
    let DecodeStrategy::PrefillOnly { candidates: first } = ab.ir().decode_strategy() else {
        return Err(JudgeError::Contract("pairwise candidates"));
    };
    let DecodeStrategy::PrefillOnly { candidates: second } = ba.ir().decode_strategy() else {
        return Err(JudgeError::Contract("pairwise candidates"));
    };
    if first.len() != 2 || second.len() != 2 || ["first", "second"].iter().any(|id| {
        let a = first.iter().find(|c| c.id() == *id);
        let b = second.iter().find(|c| c.id() == *id);
        a.is_none() || b.is_none() || a != b
    }) { return Err(JudgeError::Contract("pairwise first/second continuations differ")); }
    Ok(())
}

fn combine(ab: f64, ba: f64, policy: PairwisePolicy)
    -> Result<(f64, f64, f64, PairwiseDecision), JudgeError> {
    let mean = ab * 0.5 + ba * 0.5;
    let disagreement = (ab - ba).abs();
    if !ab.is_finite() || !ba.is_finite() || !mean.is_finite() || !disagreement.is_finite() {
        return Err(JudgeError::InvalidScores);
    }
    let weight = if mean >= 0.0 { 1.0 / (1.0 + (-mean).exp()) }
        else { let exp = mean.exp(); exp / (1.0 + exp) };
    let decision = if disagreement > f64::from(policy.maximum_order_disagreement_milli) / 1000.0 {
        PairwiseDecision::Abstained
    } else if mean == 0.0 { PairwiseDecision::Tie }
    else if mean.abs() < f64::from(policy.minimum_margin_milli) / 1000.0 { PairwiseDecision::Abstained }
    else if mean > 0.0 { PairwiseDecision::PreferA } else { PairwiseDecision::PreferB };
    Ok((mean, disagreement, weight, decision))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        execution_identity::{ExecutionIdentity, NumericsProfile, ThinkingMode, ToolMode},
        native_engine::lmhead::{NANBEIGE_VOCAB_SIZE, scoring::{ProjectionRows, ScoringError}},
        tasks::{BuiltInTask, ir::{Candidate, DependencyScope, FinitePostcondition,
            GrammarReference, PlanContext, PromptSegment, TaskBudget, TaskIR, TokenSequence}},
        tokenizer::specials::ArchivedControlRegistries,
    };
    fn identity() -> ExecutionIdentity {
        let d = Sha256Digest::of_bytes(b"judge fixture");
        ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
            artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
            tokenizer_digest: d, template_digest: d, task_spec: "judge-v1".to_owned(), taskir_digest: d,
            prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
            numerics_profile: NumericsProfile::HfBf16Eager, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
            thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d,
            decision_policy_digest: d, backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None }
    }
    fn task(reverse: bool, criterion: u32) -> TaskPlan {
        use PromptSegmentKind::*;
        let budget = TaskBudget { max_input_tokens: 64, max_output_tokens: 8, max_output_bytes: 100000,
            max_grammar_states: 64, max_kv_bytes: 1 << 30 };
        let ids = if reverse { [7, 8, criterion, 9, 21, 10, 20, 11] } else { [7, 8, criterion, 9, 20, 10, 21, 11] };
        let kinds = [GlobalPolicy, TaskInstruction, Document, TaskInstruction, Document, TaskInstruction, Document, AnswerScaffold];
        let ir = TaskIR::new(kinds.into_iter().zip(ids).map(|(kind, id)| PromptSegment::new(kind, vec![id])).collect(),
            DecodeStrategy::PrefillOnly { candidates: vec![Candidate::new("first", TokenSequence::new(vec![1])),
                Candidate::new("second", TokenSequence::new(vec![2]))] }, GrammarReference::none(), None,
            vec![FinitePostcondition::CandidateSetComplete, FinitePostcondition::OutputWithinBudget], budget, DependencyScope::ItemLocal).unwrap();
        TaskPlan::new(BuiltInTask::Judge.spec(), &PlanContext::new(&identity(), budget).unwrap(), ir).unwrap()
    }
    fn registry() -> ArchivedControlRegistries {
        ArchivedControlRegistries::from_archived_json(
            r#"{"schema_version":1,"registry":"TokenizerSpecialIds","entries":[{"id":0,"special":true,"surface":"eos"}]}"#,
            r#"{"schema_version":1,"registry":"TemplateControlIds","entries":[{"id":0,"special":true,"surface":"eos"}]}"#).unwrap()
    }
    fn policy() -> PairwisePolicy { PairwisePolicy { minimum_margin_milli: 100, maximum_order_disagreement_milli: 100000 } }
    fn plan(limits: JudgeLimits) -> PairwisePlan {
        PairwisePlan::from_task_plans(&task(false, 12), &task(true, 12), 0, registry().template_controls(), policy(), limits).unwrap()
    }
    struct Model { position_bias: bool, calls: usize, fail_after: usize }
    impl JudgeLogits for Model {
        type Error = &'static str;
        fn project(&mut self, head: usize, _: &[PromptSegment], prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
            self.calls += 1;
            if self.calls > self.fail_after { return Err("private fixture diagnostic"); }
            let ProjectionRows::FullVocabulary { vocabulary_size } = rows else { panic!("requires full denominator") };
            let mut logits = vec![0.0; vocabulary_size];
            if prefix.is_empty() { logits[if self.position_bias || head == 0 { 1 } else { 2 }] = 8.0; }
            Ok(logits)
        }
    }
    fn model(position_bias: bool) -> Model { Model { position_bias, calls: 0, fail_after: usize::MAX } }
    #[test]
    fn answer_identity_is_remapped_before_order_averaging() {
        let mut model = model(false);
        let result = plan(JudgeLimits::default()).execute(&mut model).unwrap();
        assert_eq!(result.decision, PairwiseDecision::PreferA);
        assert!(result.order_log_odds_a.iter().all(|&x| (x - 8.0).abs() < 1e-10));
        assert!(result.order_disagreement < 1e-10);
        assert_eq!(model.calls, 6);
        assert_eq!(result.work.projected_logits, 6 * NANBEIGE_VOCAB_SIZE as u64);
    }
    #[test]
    fn pure_first_position_bias_cancels_instead_of_becoming_a_preference() {
        let result = plan(JudgeLimits::default()).execute(&mut model(true)).unwrap();
        assert_eq!(result.decision, PairwiseDecision::Tie);
        assert_eq!(result.mean_log_odds_a, 0.0);
        assert_eq!(result.candidate_conditional_weight_a, 0.5);
        assert!(result.order_disagreement > 15.0);
    }
    #[test]
    fn changed_criterion_or_unswapped_answers_are_not_a_second_order() {
        assert!(validate_orders(&task(false, 12), &task(false, 12)).is_err());
        assert!(validate_orders(&task(false, 12), &task(true, 13)).is_err());
    }
    #[test]
    fn disagreement_and_weak_margins_can_abstain() {
        let strict = PairwisePolicy { minimum_margin_milli: 500, maximum_order_disagreement_milli: 1000 };
        assert_eq!(combine(8.0, -8.0, strict).unwrap().3, PairwiseDecision::Abstained);
        assert_eq!(combine(0.1, 0.2, strict).unwrap().3, PairwiseDecision::Abstained);
        assert_eq!(combine(-2.0, -2.0, strict).unwrap().3, PairwiseDecision::PreferB);
        assert_eq!(combine(0.0, 0.0, strict).unwrap().3, PairwiseDecision::Tie);
    }
    #[test]
    fn averaging_log_ratios_is_not_averaging_probabilities() {
        let (mean, _, weight, _) = combine(4.0, 0.0, policy()).unwrap();
        assert_eq!(mean, 2.0);
        assert!((weight - 0.8807970779778823).abs() < 1e-12);
        assert!(combine(f64::NAN, 0.0, policy()).is_err());
        assert_eq!(combine(-10000.0, -10000.0, policy()).unwrap().2, 0.0);
    }
    #[test]
    fn reverse_failure_returns_no_partial_preference() {
        let mut model = model(false); model.fail_after = 3;
        assert!(matches!(plan(JudgeLimits::default()).execute(&mut model), Err(JudgeError::Scoring(ScoringError::ProjectionFailed))));
    }
    #[test]
    fn aggregate_projection_and_complete_output_budgets_are_enforced() {
        let limits = JudgeLimits { max_total_projected_logits: 6 * NANBEIGE_VOCAB_SIZE as u64 - 1, ..JudgeLimits::default() };
        assert!(PairwisePlan::from_task_plans(&task(false, 12), &task(true, 12), 0, registry().template_controls(), policy(), limits).is_err());
        let limited = plan(JudgeLimits { max_output_bytes: 50, ..JudgeLimits::default() });
        assert!(matches!(limited.execute(&mut model(false)), Err(JudgeError::Limit("complete_output_bytes"))));
    }
    #[test]
    fn policy_and_criterion_change_private_execution_binding() {
        let first = plan(JudgeLimits::default());
        let changed = PairwisePlan::from_task_plans(&task(false, 13), &task(true, 13), 0, registry().template_controls(), policy(), JudgeLimits::default()).unwrap();
        assert_ne!(first.binding_digest(), changed.binding_digest());
        let changed = PairwisePlan::from_task_plans(&task(false, 12), &task(true, 12), 0, registry().template_controls(),
            PairwisePolicy { minimum_margin_milli: 101, ..policy() }, JudgeLimits::default()).unwrap();
        assert_ne!(first.binding_digest(), changed.binding_digest());
    }
}
