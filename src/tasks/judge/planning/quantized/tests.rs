//! Synthetic logits exercise real judge semantics, not native fidelity claims.
use super::*;
use super::super::tests as fixtures;
use crate::native_engine::lmhead::scoring::{CandidateLogits, ProjectionRows};

fn identity(planner: &JudgePlanner) -> ExecutionIdentity {
    let mut id = fixtures::identity(planner);
    id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 };
    id.backend_semantic_version = STRICT_INT8_EXECUTION.to_owned();
    id.quant_recipe = "nanbeige42-int8-v1".to_owned();
    id
}
fn pairwise() -> JudgeRequest {
    JudgeRequest::Pairwise { criterion: "Accuracy".to_owned(), a: "café <think>".to_owned(),
        b: "other".to_owned(), policy: PairwisePolicy { minimum_margin_milli: 100,
            maximum_order_disagreement_milli: 20000 }, budget: fixtures::budget() }
}
fn plan(request: &JudgeRequest) -> PreparedInt8Judge {
    let planner = fixtures::planner(); let id = identity(&planner);
    planner.plan_int8_with_control(request, &PlanContext::new(&id, fixtures::budget()).unwrap(),
        JudgeLimits::default(), &mut Continue).unwrap()
}
struct Flat;
impl CandidateLogits for Flat {
    type Error = &'static str;
    fn project(&mut self, _: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        match rows {
            ProjectionRows::FullVocabulary { vocabulary_size } => Ok(vec![0.0; vocabulary_size]),
            ProjectionRows::Selected(_) => Err("missing full denominator"),
        }
    }
}
fn synthetic(head: &Head, schedule: CandidateSchedule) -> Int8CandidateRun {
    Int8CandidateRun { schema_version: 1, execution: INT8_SCORING_EXECUTION.to_owned(),
        numerics_profile: STRICT_INT8_PROFILE.to_owned(), model_work: schedule.model,
        rewound_positions: 0, scores: head.scorer.score(&mut Flat, ScoringMode::FullVocabulary).unwrap() }
}
fn evaluate(plan: &PreparedInt8Judge) -> Int8JudgeRun {
    plan.evaluate_heads(&mut Continue, |_, h, s, _| Ok(synthetic(h, s))).unwrap()
}
#[test]
fn pairwise_plans_both_orders_and_charges_complete_native_geometry() {
    let p = plan(&pairwise());
    assert_eq!(p.head_count(), 2);
    let a = p.schedules[0]; let b = p.schedules[1];
    assert_eq!(p.planned_work(), a.model.checked_add(b.model).unwrap());
    assert_eq!(p.required_context(), a.context.max(b.context));
    assert!(p.planned_work().forward_positions > p.required_context() as u64);
    let run = evaluate(&p);
    assert_eq!(run.head_count, 2);
    assert_eq!(run.model_work, p.planned_work());
    assert!(matches!(run.result, JudgeResult::Pairwise(_)));
}
#[test]
fn raw_text_and_pinned_template_do_not_change_with_numeric_profile() {
    let planner = fixtures::planner(); let strict = identity(&planner); let eager = fixtures::identity(&planner);
    let request = pairwise();
    let q = planner.plan_int8_with_control(&request, &PlanContext::new(&strict, fixtures::budget()).unwrap(),
        JudgeLimits::default(), &mut Continue).unwrap();
    let e = planner.plan(&request, &PlanContext::new(&eager, fixtures::budget()).unwrap(), JudgeLimits::default()).unwrap();
    assert_eq!(q.execution_identity().prompt_digest, e.execution_identity().prompt_digest);
    assert_eq!(q.execution_identity().template_digest, e.execution_identity().template_digest);
    assert_eq!(q.execution_identity().decision_policy_digest, e.execution_identity().decision_policy_digest);
    assert_ne!(q.execution_identity().numerics_profile, e.execution_identity().numerics_profile);
}
#[test]
fn eager_context_cannot_be_rewritten_into_an_int8_plan() {
    let planner = fixtures::planner(); let id = fixtures::identity(&planner);
    assert!(matches!(planner.plan_int8_with_control(&pairwise(), &PlanContext::new(&id, fixtures::budget()).unwrap(),
        JudgeLimits::default(), &mut Continue), Err(Int8JudgeError::Identity)));
    let strict = identity(&planner);
    assert!(planner.plan(&pairwise(), &PlanContext::new(&strict, fixtures::budget()).unwrap(), JudgeLimits::default()).is_err());
}
#[test]
fn full_identity_is_bound_not_only_model_or_task_name() {
    let p = plan(&pairwise());
    p.verify_identity(p.execution_identity()).unwrap();
    for axis in 0..4 {
        let mut id = p.execution_identity().clone();
        match axis { 0 => id.prompt_digest = Sha256Digest::of_bytes(b"other"),
            1 => id.decision_policy_digest = Sha256Digest::of_bytes(b"other"),
            2 => id.backend_semantic_version.push('x'), _ => id.kv_dtype = "int8".to_owned() }
        assert!(p.verify_identity(&id).is_err());
    }
}
#[test]
fn wrong_actual_model_facts_or_malformed_digest_refuse() {
    let p = plan(&pairwise()); let id = p.execution_identity();
    let digest = id.logical_model_digest.to_hex();
    let source = ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: id.source_revision.clone(),
        recipe_id: id.quant_recipe.clone(), source_root_sha256: "0".repeat(64), logical_model_sha256: digest };
    check_model(id, &source).unwrap();
    for axis in 0..4 {
        let mut source = source.clone();
        match axis { 0 => source.model_id.push('x'), 1 => source.revision.push('x'),
            2 => source.recipe_id.push('x'), _ => source.logical_model_sha256 = "not-a-digest".to_owned() }
        assert!(matches!(check_model(id, &source), Err(Int8JudgeError::Identity)));
    }
}
#[test]
fn every_native_allowance_axis_is_checked_for_the_complete_bundle() {
    let p = plan(&pairwise()); let work = p.planned_work();
    check_work(work, Int8RunBudget::exact(work)).unwrap();
    for axis in 0..4 {
        let mut budget = Int8RunBudget::exact(work);
        match axis { 0 => budget.max_forward_positions -= 1, 1 => budget.max_attention_pairs -= 1,
            2 => budget.max_projection_work.dot_products -= 1, _ => budget.max_projection_work.multiply_accumulates -= 1 }
        assert!(matches!(check_work(work, budget), Err(Int8JudgeError::WorkBudget)));
    }
}
#[test]
fn one_failed_order_cannot_become_a_partial_pairwise_success() {
    let p = plan(&pairwise()); let mut count = 0;
    let result = p.evaluate_heads(&mut Continue, |i, h, s, _| {
        count += 1;
        if i == 1 { Err(Int8JudgeError::Native(StrictInt8Error::Work)) } else { Ok(synthetic(h, s)) }
    });
    assert_eq!(count, 2);
    assert!(matches!(result, Err(Int8JudgeError::Native(StrictInt8Error::Work))));
}
#[test]
fn corrupted_native_envelopes_or_any_work_counter_refuse() {
    let p = plan(&pairwise());
    for axis in 0..8 {
        let result = p.evaluate_heads(&mut Continue, |_, h, s, _| {
            let mut run = synthetic(h, s);
            match axis { 0 => run.schema_version += 1, 1 => run.execution.push('x'), 2 => run.numerics_profile.push('x'),
                3 => run.model_work.forward_positions += 1, 4 => run.model_work.projected_logits += 1,
                5 => run.model_work.attention_pairs += 1, 6 => run.model_work.projections.dot_products += 1,
                _ => run.model_work.projections.multiply_accumulates += 1 }
            Ok(run)
        });
        assert!(matches!(result, Err(Int8JudgeError::Accounting)));
    }
}
#[test]
fn missing_denominators_eos_or_candidates_are_not_accepted() {
    let p = plan(&pairwise());
    for axis in 0..3 {
        let result = p.evaluate_heads(&mut Continue, |_, h, s, _| {
            let mut run = synthetic(h, s);
            match axis { 0 => run.scores.full_vocab_denominators_computed = false,
                1 => run.scores.eos_token_id = 0, _ => { run.scores.candidates.pop(); } }
            Ok(run)
        });
        assert!(result.is_err());
    }
}
struct Stop { remaining: usize }
impl DecodeStepControl for Stop {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if self.remaining == 0 { Some(DecodeCancellationKind::Deadline) }
        else { self.remaining -= 1; None }
    }
}
#[test]
fn cancellation_before_planning_or_between_heads_returns_no_result() {
    let planner = fixtures::planner(); let id = identity(&planner);
    let result = planner.plan_int8_with_control(&pairwise(), &PlanContext::new(&id, fixtures::budget()).unwrap(),
        JudgeLimits::default(), &mut Stop { remaining: 0 });
    assert!(matches!(result, Err(Int8JudgeError::Cancelled(DecodeCancellationKind::Deadline))));
    let p = plan(&pairwise()); let mut count = 0;
    let result = p.evaluate_heads(&mut Stop { remaining: 1 }, |_, h, s, _| { count += 1; Ok(synthetic(h, s)) });
    assert_eq!(count, 1);
    assert_eq!(result.unwrap_err().cancellation(), Some(DecodeCancellationKind::Deadline));
}
#[test]
fn late_cancellation_cannot_publish_a_finalized_judgment() {
    let p = plan(&pairwise());
    // Two head checkpoints and the pre-finalizer checkpoint pass, final one stops.
    let result = p.evaluate_heads(&mut Stop { remaining: 3 }, |_, h, s, _| Ok(synthetic(h, s)));
    assert_eq!(result.unwrap_err().cancellation(), Some(DecodeCancellationKind::Deadline));
}
#[test]
fn complete_output_envelope_obeys_its_byte_limit() {
    let mut p = plan(&pairwise());
    let length = canonjson::canonical_bytes(&evaluate(&p)).unwrap().len() as u64;
    match &mut p.inner.executable { Executable::Pairwise(v) => v.bundle.max_output_bytes = length - 1, _ => unreachable!() }
    assert!(matches!(p.evaluate_heads(&mut Continue, |_, h, s, _| Ok(synthetic(h, s))),
        Err(Int8JudgeError::Task(JudgeError::Limit("complete_output_bytes")))));
}
#[test]
fn rubric_executes_every_criterion_and_uses_existing_semantic_finalizer() {
    let request = JudgeRequest::Rubric { document: "Example text".to_owned(),
        rubric: RubricDefinition { schema_version: 1, revision: "local-v1".to_owned(),
            declared_origin_digest: Sha256Digest::of_bytes(b"caller"), scale_maximum: 3,
            criteria: vec![RubricCriterion { id: "style".to_owned(), description: "Clear prose".to_owned(), weight: 1 },
                RubricCriterion { id: "accuracy".to_owned(), description: "Correct statements".to_owned(), weight: 3 }] },
        policy: RubricPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1000000 }, budget: fixtures::budget() };
    let p = plan(&request); let run = evaluate(&p);
    assert_eq!(run.head_count, 2); assert!(matches!(run.result, JudgeResult::Rubric(_)));
    let scores = p.inner.executable.bundle().score_with::<JudgeError, _>(|_, h|
        h.scorer.score(&mut Flat, ScoringMode::FullVocabulary).map_err(Into::into)).unwrap();
    assert_eq!(run.result, p.inner.executable.finish(scores).unwrap());
}
#[test]
fn faithfulness_retains_full_source_and_every_evidence_window() {
    let request = JudgeRequest::Faithfulness { source: "Alpha. Beta. Gamma.".to_owned(), claim: "Alpha".to_owned(),
        policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 900000, minimum_margin_milli: 100,
            evidence_window_bytes: 8, max_evidence_windows: 8, max_evidence_spans: 8 }, budget: fixtures::budget() };
    let p = plan(&request); let run = evaluate(&p);
    assert!(p.head_count() > 1);
    let JudgeResult::Faithfulness(result) = run.result else { panic!("faithfulness mode") };
    assert_eq!(result.heads.len(), p.head_count());
    assert_eq!(result.windows.len() + 1, p.head_count());
    assert!(result.relation.is_none());
    assert!(result.evidence.is_empty());
}
#[test]
fn native_cancellation_cause_survives_nested_scoring_error() {
    let error = Int8JudgeError::Scoring(Int8ScoringError::Native(StrictInt8Error::Cancelled(DecodeCancellationKind::CostBudget)));
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::CostBudget));
    assert!(error.source().is_some());
}
