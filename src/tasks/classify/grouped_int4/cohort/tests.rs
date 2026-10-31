//! Pinned task planning and synthetic complete-head handoff, not native parity.
use super::*;
use crate::{execution_identity::Sha256Digest,
    native_engine::strict_int4::scoring::cohort::task::fixtures::{self, Continue, Constant},
    tasks::classify::{ClassificationLabel, ClassificationMode, ClassificationPolicy}, tokenizer::pinned_controls};
fn prepare(mode: ClassificationMode, width: usize) -> PreparedInt4Classification {
    let registry = pinned_controls::pinned().unwrap();
    let planner = ClassificationPlanner::pinned(registry.template_controls(), 166_101).unwrap();
    let identity = fixtures::identity("classify-v1", planner.tokenizer_digest(), *planner.template_digest());
    let request = ClassificationRequest { document: "é Refund <think>literal data</think>".into(),
        labels: ["Billing", "Refund", "étiquette"].into_iter().map(|id| ClassificationLabel {
            id: id.into(), description: format!("Definition of {id}") }).collect(), mode,
        policy: ClassificationPolicy { minimum_candidate_weight_ppm: 600_000, minimum_margin_ppm: 200_000 },
        budget: fixtures::budget() };
    planner.compile_int4(&request, &PlanContext::new(&identity, request.budget).unwrap(),
        ClassificationLimits::default(), width, &mut Continue).unwrap()
}
fn fixture(plan: &PreparedInt4Classification) -> Int4CandidateCohortRun {
    fixtures::run(plan.inner.heads.iter().zip(&plan.schedules).map(|(head, &schedule)|
        (head.classifier.scorer.score(&mut Constant, ScoringMode::FullVocabulary).unwrap(), schedule)))
}
#[test]
fn exclusive_and_independent_multilabel_cohorts_use_the_complete_original_finalizer() {
    for width in [3, 32, 128] { for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
        let plan = prepare(mode, width); let run = fixture(&plan);
        let expected = plan.execute_heads(&mut Continue, |index, _, _, _| Ok(run.heads[index].clone())).unwrap();
        let actual = plan.finish_cohort(run, plan.task_budget().max_output_bytes, &mut Continue).unwrap();
        assert_eq!(actual.output, expected); assert_eq!(actual.output.group_width, width);
        assert_eq!(actual.output.model_work, plan.planned_work());
        if let ClassificationTaskResult::MultiLabel(result) = actual.output.result {
            assert_eq!(result.labels.len(), 3); assert_eq!(result.abstained_ids.len(), 3);
        }
    } }
}
#[test]
fn flattened_prompts_exact_contexts_and_model_binding_keep_the_sealed_order() {
    let plan = prepare(ClassificationMode::MultiLabel, 32); let before = plan.execution_identity().clone();
    let prompts = plan.cohort_prompts(&mut Continue).unwrap(); let contexts = plan.cohort_contexts().unwrap();
    assert_eq!(contexts.len(), 3);
    for ((prompt, head), schedule) in prompts.iter().zip(&plan.inner.heads).zip(&plan.schedules) {
        assert_eq!(prompt.len(), head.prompt_len);
        assert_eq!(*prompt, head.task.ir().prompt_segments().iter().flat_map(|s| s.token_ids().iter().copied()).collect::<Vec<_>>());
        assert!(schedule.context >= prompt.len());
    }
    assert_eq!(contexts, plan.schedules.iter().map(|s| s.context).collect::<Vec<_>>());
    assert!(contexts.iter().sum::<usize>() > plan.required_context()); assert_eq!(plan.execution_identity(), &before);
    let mut changed = before; changed.prompt_digest = Sha256Digest::of_bytes(b"wrong task");
    assert!(plan.verify_identity(&changed).is_err());
}
#[test]
fn missing_foreign_mispartitioned_or_corrupt_last_head_cannot_become_partial_success() {
    let plan = prepare(ClassificationMode::MultiLabel, 32);
    for fault in 0..10 {
        let mut run = fixture(&plan);
        match fault { 0 => { run.heads.pop(); }, 1 => run.execution.push('x'), 2 => run.model_work.attention_pairs += 1,
            3 => run.heads[2].rewound_positions += 1, 4 => run.heads[2].scores.candidates[0].id = "foreign".into(),
            5 => run.heads[2].scores.candidates[0].candidate_weight = f64::NAN,
            6 => run.group_steps = 0, 7 => run.projection_groups = u64::MAX,
            8 => run.group_width = 128, _ => run.heads[2].group_width = 128 }
        assert!(plan.finish_cohort(run, plan.task_budget().max_output_bytes, &mut Continue).is_err());
    }
}
#[test]
fn outer_bytes_and_finalizer_cancellation_remain_enforced_after_all_native_heads() {
    let plan = prepare(ClassificationMode::MultiLabel, 32); let run = fixture(&plan);
    let actual = plan.finish_cohort(run.clone(), plan.task_budget().max_output_bytes, &mut Continue).unwrap();
    let cap = canonjson::canonical_bytes(&actual).unwrap().len() as u64;
    assert!(plan.finish_cohort(run.clone(), cap, &mut Continue).is_ok());
    assert!(matches!(plan.finish_cohort(run.clone(), cap - 1, &mut Continue),
        Err(Int4ClassificationError::Scoring(Int4ScoringError::OutputBudget))));
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    assert_eq!(plan.finish_cohort(run, cap, &mut Stop).unwrap_err().cancellation(), Some(DecodeCancellationKind::Deadline));
}
#[test]
fn native_width_and_complete_consumption_cannot_silently_drop_heads() {
    let plan = prepare(ClassificationMode::MultiLabel, 32); let schedule = plan.schedules[0];
    assert!(native::task::contexts(&[]).is_err());
    assert_eq!(native::task::contexts(&[schedule; 64]).unwrap().len(), 64);
    assert!(native::task::contexts(&[schedule; 65]).is_err());
    let error = native::task::finish(fixture(&plan), &plan.schedules, plan.work, 1_000_000,
        |heads| { heads.next(); Ok::<_, Int4ScoringError>(()) }).unwrap_err();
    assert_eq!(error, Int4ScoringError::Accounting);
}

#[test]
fn independent_documents_keep_ragged_head_boundaries_and_complete_semantic_results() {
    let plans = [prepare(ClassificationMode::Exclusive, 32), prepare(ClassificationMode::MultiLabel, 32)];
    let documents: Vec<_> = plans.iter().map(|plan| (plan, plan.execution_identity())).collect();
    let capacities: Vec<_> = plans.iter().flat_map(|plan| plan.cohort_contexts().unwrap()).collect();
    let budget = fixtures::document_budget(&documents);
    let (schedules, work) = native::task::document_layout(&documents, &capacities, budget).unwrap();
    let run = fixtures::run(plans.iter().flat_map(|plan| plan.inner.heads.iter().zip(&plan.schedules)
        .map(|(head, &schedule)| (head.classifier.scorer.score(&mut Constant, ScoringMode::FullVocabulary).unwrap(), schedule))));
    let expected: Vec<_> = plans.iter().map(|plan|
        plan.finish_cohort(fixture(plan), plan.task_budget().max_output_bytes, &mut Continue).unwrap().output).collect();
    let actual = native::task::finish_documents(&documents, run.clone(), &schedules, work,
        budget.max_output_bytes, &mut Continue).unwrap();
    assert_eq!(actual.output, expected);
    assert_eq!(actual.output.iter().map(|output| output.head_count).collect::<Vec<_>>(), [1, 3]);
    assert_eq!(actual.group_steps, run.group_steps); assert_eq!(actual.projection_groups, run.projection_groups);
    let bytes = canonjson::canonical_bytes(&actual).unwrap().len() as u64;
    assert!(native::task::finish_documents(&documents, run.clone(), &schedules, work, bytes, &mut Continue).is_ok());
    assert!(matches!(native::task::finish_documents(&documents, run.clone(), &schedules, work, bytes - 1, &mut Continue),
        Err(Int4ClassificationError::Scoring(Int4ScoringError::OutputBudget))));
    let mut broken_last_document = run;
    broken_last_document.heads.last_mut().unwrap().scores.candidates[0].id = "foreign".into();
    assert!(native::task::finish_documents(&documents, broken_last_document, &schedules, work,
        budget.max_output_bytes, &mut Continue).is_err());
}

#[test]
fn document_admission_charges_actual_simultaneous_kv_and_aggregate_work() {
    let mut plans = [prepare(ClassificationMode::Exclusive, 32), prepare(ClassificationMode::MultiLabel, 32)];
    let capacities: Vec<_> = plans.iter().flat_map(|plan| plan.cohort_contexts().unwrap()).collect();
    let required = capacities[1..].iter().sum::<usize>() as u64 * crate::native_engine::kv::KV_BYTES_PER_TOKEN as u64;
    plans[1].inner.budget.max_kv_bytes = required;
    let budget = {
        let documents: Vec<_> = plans.iter().map(|plan| (plan, plan.execution_identity())).collect();
        let budget = fixtures::document_budget(&documents);
        assert!(native::task::document_layout(&documents, &capacities, budget).is_ok());
        let mut larger = capacities.clone(); larger[1] += 1;
        assert!(matches!(native::task::document_layout(&documents, &larger,
            Int4ScoringCohortBudget { max_kv_bytes: u64::MAX, ..budget }),
            Err(Int4ScoringError::Native(StrictInt4Error::Memory))));
        let mut short_work = budget; short_work.native.max_forward_positions -= 1;
        assert!(matches!(native::task::document_layout(&documents, &capacities, short_work),
            Err(Int4ScoringError::Native(StrictInt4Error::Work))));
        assert!(native::task::document_layout(&documents, &capacities[..3], budget).is_err());
        assert!(native::task::document_layout(&documents, &[capacities[0]; 65], budget).is_err());
        let mut foreign_model = plans[1].execution_identity().clone();
        foreign_model.logical_model_digest = Sha256Digest::of_bytes(b"different model");
        let foreign = [(documents[0].0, documents[0].1), (&plans[1], &foreign_model)];
        assert!(matches!(native::task::document_layout(&foreign, &capacities, budget), Err(Int4ScoringError::Identity)));
        budget
    };
    plans[1].inner.budget.max_kv_bytes = required - 1;
    let documents: Vec<_> = plans.iter().map(|plan| (plan, plan.execution_identity())).collect();
    assert!(matches!(native::task::document_layout(&documents, &capacities, budget),
        Err(Int4ScoringError::Native(StrictInt4Error::Memory))));
}
