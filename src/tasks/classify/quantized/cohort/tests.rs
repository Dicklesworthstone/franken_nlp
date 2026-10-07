//! Pinned task planning and private finalizer fixtures, not model inference.
use super::*;
use crate::{native_engine::strict_int8::scoring::cohort::task::fixtures::{self, Continue, Constant},
    tasks::classify::{ClassificationLabel, ClassificationMode, ClassificationPolicy}, tokenizer::pinned_controls};
fn prepare(mode: ClassificationMode) -> PreparedInt8Classification {
    let registry = pinned_controls::pinned().unwrap();
    let planner = ClassificationPlanner::pinned(registry.template_controls(), 166_101).unwrap();
    let identity = fixtures::identity("classify-v1", planner.tokenizer_digest(), *planner.template_digest());
    let request = ClassificationRequest { document: "é Refund <think>literal data</think>".into(),
        labels: ["Billing", "Refund", "étiquette"].into_iter().map(|id| ClassificationLabel {
            id: id.into(), description: format!("Definition of {id}") }).collect(), mode,
        policy: ClassificationPolicy { minimum_candidate_weight_ppm: 600_000, minimum_margin_ppm: 200_000 },
        budget: fixtures::budget() };
    planner.plan_int8_with_control(&request, &PlanContext::new(&identity, request.budget).unwrap(),
        ClassificationLimits::default(), &mut Continue).unwrap()
}
fn fixture(plan: &PreparedInt8Classification) -> Int8CandidateCohortRun {
    fixtures::run(plan.inner.heads.iter().zip(&plan.schedules).map(|(head, &schedule)|
        (head.classifier.scorer.score(&mut Constant, ScoringMode::FullVocabulary).unwrap(), schedule)))
}
#[test]
fn exclusive_and_independent_multilabel_cohorts_use_the_existing_complete_finalizer() {
    for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
        let plan = prepare(mode); let run = fixture(&plan);
        let expected = plan.execute_heads(&mut Continue, |index, _, _, _| Ok(run.heads[index].clone())).unwrap();
        let actual = plan.finish_cohort(run, plan.task_budget().max_output_bytes, &mut Continue).unwrap();
        assert_eq!(actual.output, expected);
        assert_eq!(actual.output.model_work, plan.planned_work());
        if let ClassificationTaskResult::MultiLabel(result) = actual.output.result {
            assert_eq!(result.labels.len(), 3); assert_eq!(result.abstained_ids.len(), 3);
        }
    }
}
#[test]
fn complete_prompt_and_context_order_preserve_the_sealed_identity() {
    let plan = prepare(ClassificationMode::MultiLabel); let before = plan.execution_identity().clone();
    let prompts = plan.cohort_prompts(&mut Continue).unwrap(); let contexts = plan.cohort_contexts().unwrap();
    assert_eq!(contexts.len(), 3);
    for ((prompt, head), schedule) in prompts.iter().zip(&plan.inner.heads).zip(&plan.schedules) {
        assert_eq!(prompt.len(), head.prompt_len);
        assert_eq!(*prompt, head.task.ir().prompt_segments().iter().flat_map(|s| s.token_ids().iter().copied()).collect::<Vec<_>>());
        assert!(schedule.context >= prompt.len());
    }
    assert_eq!(contexts, plan.schedules.iter().map(|s| s.context).collect::<Vec<_>>());
    assert!(contexts.iter().sum::<usize>() > plan.required_context());
    assert_eq!(plan.execution_identity(), &before);
    let mut changed = before; changed.prompt_digest = Sha256Digest::of_bytes(b"wrong task");
    assert!(plan.verify_identity(&changed).is_err());
}
#[test]
fn missing_foreign_or_corrupt_last_head_cannot_become_partial_label_success() {
    let plan = prepare(ClassificationMode::MultiLabel);
    for fault in 0..8 {
        let mut run = fixture(&plan);
        match fault { 0 => { run.heads.pop(); }, 1 => run.execution.push('x'), 2 => run.model_work.attention_pairs += 1,
            3 => run.heads[2].rewound_positions += 1, 4 => run.heads[2].scores.candidates[0].id = "foreign".into(),
            5 => run.heads[2].scores.candidates[0].candidate_weight = f64::NAN,
            6 => run.group_steps = 0, _ => run.projection_groups = u64::MAX }
        assert!(plan.finish_cohort(run, plan.task_budget().max_output_bytes, &mut Continue).is_err());
    }
}
#[test]
fn outer_result_cap_and_finalizer_cancellation_are_not_bypassed_by_completed_native_heads() {
    let plan = prepare(ClassificationMode::MultiLabel); let run = fixture(&plan);
    let actual = plan.finish_cohort(run.clone(), plan.task_budget().max_output_bytes, &mut Continue).unwrap();
    let cap = canonjson::canonical_bytes(&actual).unwrap().len() as u64;
    assert!(plan.finish_cohort(run.clone(), cap, &mut Continue).is_ok());
    assert!(matches!(plan.finish_cohort(run.clone(), cap - 1, &mut Continue),
        Err(Int8ClassificationError::Scoring(Int8ScoringError::OutputBudget))));
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    assert_eq!(plan.finish_cohort(run, cap, &mut Stop).unwrap_err().cancellation(), Some(DecodeCancellationKind::Deadline));
}
#[test]
fn cohort_width_and_complete_consumption_cannot_silently_drop_heads() {
    let plan = prepare(ClassificationMode::MultiLabel); let schedule = plan.schedules[0];
    assert!(native::task::contexts(&[]).is_err());
    assert_eq!(native::task::contexts(&[schedule; 64]).unwrap().len(), 64);
    assert!(native::task::contexts(&[schedule; 65]).is_err());
    let error = native::task::finish(fixture(&plan), &plan.schedules, plan.work, 1_000_000,
        |heads| { heads.next(); Ok::<_, Int8ScoringError>(()) }).unwrap_err();
    assert_eq!(error, Int8ScoringError::Accounting);
}
