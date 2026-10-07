//! Pinned pairwise/rubric/evidence finalizers; synthetic logits are not truth evidence.
use super::*;
use crate::{native_engine::strict_int8::scoring::cohort::task::fixtures::{self, Constant},
    tasks::judge::{PairwisePolicy, RubricPolicy, RubricDefinition, RubricCriterion, FaithfulnessPolicy},
    tokenizer::pinned_controls};
fn requests() -> Vec<JudgeRequest> {
    vec![JudgeRequest::Pairwise { criterion: "Accuracy".into(), a: "café <think>".into(), b: "other".into(),
        policy: PairwisePolicy { minimum_margin_milli: 100, maximum_order_disagreement_milli: 20000 }, budget: fixtures::budget() },
        JudgeRequest::Rubric { document: "Example text".into(), rubric: RubricDefinition { schema_version: 1,
            revision: "local-v1".into(), declared_origin_digest: Sha256Digest::of_bytes(b"declared"), scale_maximum: 5,
            criteria: vec![RubricCriterion { id: "style".into(), description: "Clear prose".into(), weight: 1 },
                RubricCriterion { id: "accuracy".into(), description: "Accurate claims".into(), weight: 3 }] },
            policy: RubricPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 900000 }, budget: fixtures::budget() },
        JudgeRequest::Faithfulness { source: "Alice left. Bob stayed. café closed.".into(), claim: "Alice left.".into(),
            policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 600000, minimum_margin_milli: 100,
                evidence_window_bytes: 16, max_evidence_windows: 8, max_evidence_spans: 8 }, budget: fixtures::budget() }]
}
fn prepare(request: &JudgeRequest) -> PreparedInt8Judge {
    let registry = pinned_controls::pinned().unwrap();
    let planner = JudgePlanner::pinned(registry.template_controls(), 166_101).unwrap();
    let identity = fixtures::identity("judge-v1", planner.tokenizer_digest(), *planner.template_digest());
    planner.plan_int8_with_control(request, &PlanContext::new(&identity, request.budget()).unwrap(),
        JudgeLimits::default(), &mut fixtures::Continue).unwrap()
}
fn fixture(plan: &PreparedInt8Judge) -> Int8CandidateCohortRun {
    fixtures::run(plan.inner.executable.bundle().heads.iter().zip(&plan.schedules).map(|(head, &schedule)|
        (head.scorer.score(&mut Constant, ScoringMode::FullVocabulary).unwrap(), schedule)))
}
#[test]
fn both_orders_all_rubric_criteria_and_all_evidence_heads_use_the_original_finalizer() {
    for request in requests() {
        let plan = prepare(&request); let run = fixture(&plan);
        let expected = plan.evaluate_heads(&mut fixtures::Continue, |index, _, _, _| Ok(run.heads[index].clone())).unwrap();
        let result = plan.finish_cohort(run, plan.max_result_bytes(), &mut fixtures::Continue).unwrap();
        assert_eq!(result.output, expected); assert!(result.output.head_count >= 2);
        if let JudgeResult::Faithfulness(result) = result.output.result {
            assert!(result.windows.len() > 1); assert_eq!(result.heads.len(), result.windows.len() + 1);
        }
    }
}
#[test]
fn raw_evidence_and_canonical_head_order_are_not_changed_by_cohort_geometry() {
    for request in requests() {
        let plan = prepare(&request); let identity = plan.execution_identity().clone();
        let prompts = plan.cohort_prompts(&mut fixtures::Continue).unwrap(); let contexts = plan.cohort_contexts().unwrap();
        assert_eq!(contexts.len(), plan.head_count());
        for (prompt, head) in prompts.iter().zip(&plan.inner.executable.bundle().heads) {
            assert_eq!(prompt.len(), head.prompt_len);
            assert_eq!(*prompt, head.ir.prompt_segments().iter().flat_map(|s| s.token_ids().iter().copied()).collect::<Vec<_>>());
        }
        assert_eq!(plan.execution_identity(), &identity);
        assert!(contexts.iter().sum::<usize>() > plan.required_context());
    }
}
#[test]
fn missing_or_corrupt_later_judgment_head_never_becomes_partial_success() {
    for request in requests() {
        let plan = prepare(&request);
        for fault in 0..5 {
            let mut run = fixture(&plan);
            match fault { 0 => { run.heads.pop(); }, 1 => run.heads.last_mut().unwrap().rewound_positions += 1,
                2 => run.heads.last_mut().unwrap().scores.candidates[0].id = "foreign".into(),
                3 => run.heads.last_mut().unwrap().scores.candidates[0].candidate_weight = f64::NAN,
                _ => run.model_work.projections.multiply_accumulates += 1 }
            assert!(plan.finish_cohort(run, plan.max_result_bytes(), &mut fixtures::Continue).is_err());
        }
    }
}
#[test]
fn complete_judgment_outer_bytes_and_finalizer_cancellation_remain_enforced() {
    let request = requests().remove(0); let plan = prepare(&request); let run = fixture(&plan);
    let result = plan.finish_cohort(run.clone(), plan.max_result_bytes(), &mut fixtures::Continue).unwrap();
    let bytes = crate::canonjson::canonical_bytes(&result).unwrap().len() as u64;
    assert!(plan.finish_cohort(run.clone(), bytes, &mut fixtures::Continue).is_ok());
    assert!(matches!(plan.finish_cohort(run.clone(), bytes - 1, &mut fixtures::Continue),
        Err(Int8JudgeError::Scoring(Int8ScoringError::OutputBudget))));
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    assert_eq!(plan.finish_cohort(run, bytes, &mut Stop).unwrap_err().cancellation(), Some(DecodeCancellationKind::Deadline));
}
