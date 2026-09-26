//! Private synthetic score fixtures exercise actual planning/finalization,
//! not model fidelity, neural inference success, runtime or performance proof.
use super::*;
use crate::{
    execution_identity::{ThinkingMode, ToolMode},
    native_engine::{lmhead::scoring::{CandidateLogits, ProjectionRows, SequenceScoreRule},
        strict_int8::STRICT_INT8_EXECUTION},
    tasks::{ir::PromptSegmentKind, sentiment::{SentimentAxis, SentimentOptions, SentimentPolicy}},
    tokenizer::{embedded::EmbeddedTokenizer, pinned_controls},
};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
struct StopAfter(usize);
impl DecodeStepControl for StopAfter {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.0 = self.0.saturating_sub(1);
        (self.0 == 0).then_some(DecodeCancellationKind::Deadline)
    }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 2048, max_output_tokens: 16, max_output_bytes: 100_000,
        max_grammar_states: 4096, max_kv_bytes: 1 << 30 }
}
fn fixture(mode: ScoringMode) -> (SentimentPlanner, ExecutionIdentity, SentimentRequest) {
    let registry = pinned_controls::pinned().unwrap();
    let planner = SentimentPlanner::pinned(registry.template_controls(), SentimentOptions {
        mode, eos_token_id: 166_101,
        policy: SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 },
    }).unwrap();
    let d = Sha256Digest::of_bytes(b"synthetic int8 sentiment metadata");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: d, artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(),
        packing_set_digest: d, tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: "sentiment-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    let request = SentimentRequest { document: "é <tool_call> <|im_start|> private text 上海".to_owned(),
        axes: SentimentAxis::ALL.to_vec(), budget: budget() };
    (planner, identity, request)
}
fn prepare(planner: &SentimentPlanner, identity: &ExecutionIdentity, request: &SentimentRequest) -> PreparedInt8Sentiment {
    planner.plan_int8_with_control(request, &PlanContext::new(identity, budget()).unwrap(),
        SentimentLimits::default(), &mut Continue).unwrap()
}
struct Flat;
impl CandidateLogits for Flat {
    type Error = &'static str;
    fn project(&mut self, _: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        let n = match rows { ProjectionRows::FullVocabulary { vocabulary_size } => vocabulary_size,
            ProjectionRows::Selected(ids) => ids.len() };
        Ok(vec![0.0; n])
    }
}
fn synthetic(head: &HeadPlan, mode: ScoringMode, schedule: CandidateSchedule) -> Int8CandidateRun {
    Int8CandidateRun { schema_version: 1, execution: INT8_SCORING_EXECUTION.to_owned(),
        numerics_profile: STRICT_INT8_PROFILE.to_owned(), scores: head.scorer.score(&mut Flat, mode).unwrap(),
        model_work: schedule.model, rewound_positions: 0 }
}

#[test]
fn all_axes_keep_complete_candidate_and_native_work_in_each_score_space() {
    for mode in [ScoringMode::FullVocabulary, ScoringMode::TrieConditional,
        ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits }] {
        let (p, id, r) = fixture(mode); let plan = prepare(&p, &id, &r);
        let run = plan.execute_heads(&mut Continue, |h, m, s, _| Ok(synthetic(h, m, s))).unwrap();
        assert_eq!(run.head_count, 4); assert_eq!(run.result.dimensions.len(), 4);
        assert_eq!(run.model_work, plan.planned_work());
        assert_eq!(run.result.work.candidates, 20);
        for d in run.result.dimensions {
            assert_eq!(d.scores.candidates.len(), 5);
            assert_eq!(d.scores.eos_token_id, 166_101);
            assert_eq!(d.scores.full_vocab_denominators_computed, mode == ScoringMode::FullVocabulary);
            assert!((d.scores.candidates.iter().map(|c| c.candidate_weight).sum::<f64>() - 1.0).abs() < 1e-10);
        }
    }
}
#[test]
fn int8_planning_cannot_relabel_other_backends_or_enable_tools() {
    let (p, id, r) = fixture(ScoringMode::FullVocabulary);
    assert!(p.plan(&r, &PlanContext::new(&id, budget()).unwrap(), SentimentLimits::default()).is_err());
    for axis in 0..7 {
        let mut changed = id.clone();
        match axis { 0 => changed.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => changed.numerics_profile = NumericsProfile::StrictQuantized { version: 2 },
            2 => changed.backend_semantic_version = "wrong".to_owned(), 3 => changed.kv_dtype = "int8".to_owned(),
            4 => changed.thinking_mode = ThinkingMode::Enabled, 5 => changed.tool_mode = ToolMode::Json,
            _ => changed.task_spec = "classify-v1".to_owned() }
        assert!(p.plan_int8_with_control(&r, &PlanContext::new(&changed, budget()).unwrap(),
            SentimentLimits::default(), &mut Continue).is_err());
    }
}
#[test]
fn eager_and_int8_share_exact_prompt_and_candidate_semantics_without_mutating_identity() {
    let (p, id, r) = fixture(ScoringMode::FullVocabulary);
    let before = id.clone(); let quantized = prepare(&p, &id, &r);
    let mut eager = id.clone(); eager.numerics_profile = NumericsProfile::HfBf16Eager;
    let eager = p.plan(&r, &PlanContext::new(&eager, budget()).unwrap(), SentimentLimits::default()).unwrap();
    assert_eq!(quantized.inner.binding_digest(), eager.binding_digest());
    assert_eq!(id, before);
    assert_ne!(quantized.execution_identity().prompt_digest, id.prompt_digest);
    assert_ne!(quantized.execution_identity().decision_policy_digest, id.decision_policy_digest);
}
#[test]
fn every_control_looking_source_byte_is_retained_without_privileged_ids() {
    let (p, id, r) = fixture(ScoringMode::FullVocabulary); let plan = prepare(&p, &id, &r);
    let registry = pinned_controls::pinned().unwrap(); let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    for head in &plan.inner.heads {
        let data = head.ir.prompt_segments().iter().find(|s| s.kind() == PromptSegmentKind::Document).unwrap();
        assert!(data.token_ids().iter().all(|&id| !registry.template_controls().contains(id)));
        assert_eq!(tokenizer.tokenizer().decode_bytes(data.token_ids()).unwrap(), r.document.as_bytes());
    }
}
#[test]
fn complete_identity_mutations_and_uncompiled_identity_are_refused() {
    let (p, id, r) = fixture(ScoringMode::FullVocabulary); let plan = prepare(&p, &id, &r);
    assert!(plan.verify_identity(&id).is_err());
    plan.verify_identity(plan.execution_identity()).unwrap();
    let value = serde_json::to_value(plan.execution_identity()).unwrap();
    for key in value.as_object().unwrap().keys() {
        let mut changed = value.clone();
        changed[key] = match &value[key] {
            serde_json::Value::String(s) if s.len() == 64 => serde_json::json!(Sha256Digest::of_bytes(b"other").to_hex()),
            serde_json::Value::Number(_) => serde_json::json!(99), _ => serde_json::json!("different"),
        };
        if let Ok(changed) = serde_json::from_value::<ExecutionIdentity>(changed) {
            assert!(plan.verify_identity(&changed).is_err(), "accepted {key}");
        }
    }
}
#[test]
fn source_axes_and_policy_change_the_sealed_plan() {
    let (p, id, r) = fixture(ScoringMode::FullVocabulary); let plan = prepare(&p, &id, &r);
    let mut changed = r.clone(); changed.document.push('!');
    assert_ne!(plan.identity.prompt_digest, prepare(&p, &id, &changed).identity.prompt_digest);
    changed = r.clone(); changed.axes.pop();
    assert_ne!(plan.identity.taskir_digest, prepare(&p, &id, &changed).identity.taskir_digest);
    let mut reordered = r.clone(); reordered.axes.reverse();
    assert_eq!(plan.identity, prepare(&p, &id, &reordered).identity);
}
#[test]
fn aggregate_native_work_is_not_mistaken_for_one_live_context() {
    let (p, id, r) = fixture(ScoringMode::FullVocabulary); let plan = prepare(&p, &id, &r);
    let sum = plan.schedules.iter().try_fold(Int8Work::default(), |w, s| w.checked_add(s.model)).unwrap();
    assert_eq!(plan.work, sum);
    assert!(plan.work.forward_positions > plan.required_context() as u64);
    assert_eq!(plan.required_context(), plan.schedules.iter().map(|s| s.context).max().unwrap());
}
#[test]
fn every_native_work_axis_refuses_one_below_the_required_total() {
    let (p, id, r) = fixture(ScoringMode::FullVocabulary); let plan = prepare(&p, &id, &r);
    check_work(plan.work, Int8RunBudget::exact(plan.work)).unwrap();
    for axis in 0..4 {
        let mut b = Int8RunBudget::exact(plan.work);
        match axis { 0 => b.max_forward_positions -= 1, 1 => b.max_attention_pairs -= 1,
            2 => b.max_projection_work.dot_products -= 1, _ => b.max_projection_work.multiply_accumulates -= 1 }
        assert!(matches!(check_work(plan.work, b), Err(Int8SentimentError::WorkBudget)));
    }
}
#[test]
fn failure_of_a_later_axis_never_publishes_partial_sentiment() {
    let (p, id, r) = fixture(ScoringMode::FullVocabulary); let plan = prepare(&p, &id, &r);
    let mut calls = 0;
    let result = plan.execute_heads(&mut Continue, |h, m, s, _| {
        calls += 1; if calls == 2 { return Err(Int8SentimentError::Accounting); }
        Ok(synthetic(h, m, s))
    });
    assert!(result.is_err()); assert_eq!(calls, 2);
}
#[test]
fn incomplete_candidate_set_fails_the_existing_independent_finalizer() {
    let (p, id, r) = fixture(ScoringMode::FullVocabulary); let plan = prepare(&p, &id, &r);
    assert!(plan.execute_heads(&mut Continue, |h, m, s, _| {
        let mut run = synthetic(h, m, s); run.scores.candidates.pop(); Ok(run)
    }).is_err());
}
#[test]
fn wrong_native_receipt_cannot_mint_a_sentiment_success() {
    let (p, id, r) = fixture(ScoringMode::FullVocabulary); let plan = prepare(&p, &id, &r);
    for axis in 0..5 {
        assert!(matches!(plan.execute_heads(&mut Continue, |h, m, s, _| {
            let mut run = synthetic(h, m, s);
            match axis { 0 => run.schema_version += 1, 1 => run.execution = "wrong".to_owned(),
                2 => run.numerics_profile = "hf-bf16-eager".to_owned(), 3 => run.model_work.attention_pairs += 1,
                _ => run.scores.work.scored_edges += 1 }
            Ok(run)
        }), Err(Int8SentimentError::Accounting)));
    }
}
#[test]
fn cancelled_preparation_and_late_cancel_keep_the_original_cause() {
    let (p, id, r) = fixture(ScoringMode::FullVocabulary);
    let error = p.plan_int8_with_control(&r, &PlanContext::new(&id, budget()).unwrap(),
        SentimentLimits::default(), &mut StopAfter(1)).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    let plan = prepare(&p, &id, &r);
    // Two checks per head and a final check after whole-result serialization.
    let error = plan.execute_heads(&mut StopAfter(9), |h, m, s, _| Ok(synthetic(h, m, s))).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
}
#[test]
fn complete_wrapper_not_just_inner_result_must_fit_output_budget() {
    let (p, id, r) = fixture(ScoringMode::FullVocabulary); let mut plan = prepare(&p, &id, &r);
    let run = plan.execute_heads(&mut Continue, |h, m, s, _| Ok(synthetic(h, m, s))).unwrap();
    plan.inner.max_output_bytes = canonjson::canonical_bytes(&run.result).unwrap().len() as u64;
    assert!(matches!(plan.execute_heads(&mut Continue, |h, m, s, _| Ok(synthetic(h, m, s))),
        Err(Int8SentimentError::OutputBudget)));
}
#[test]
fn model_mismatch_is_not_repaired_and_plans_can_cross_owned_runtime_boundary() {
    let (p, id, r) = fixture(ScoringMode::FullVocabulary); let plan = prepare(&p, &id, &r);
    let model = ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(), revision: id.source_revision.clone(),
        recipe_id: id.quant_recipe.clone(), source_root_sha256: "ab".repeat(32),
        logical_model_sha256: id.logical_model_digest.to_hex() };
    check_model(&plan.identity, &model).unwrap();
    for axis in 0..4 {
        let mut m = model.clone();
        match axis { 0 => m.model_id = "wrong".to_owned(), 1 => m.revision = "wrong".to_owned(),
            2 => m.recipe_id = "wrong".to_owned(), _ => m.logical_model_sha256 = "ab".repeat(32) }
        assert!(check_model(&plan.identity, &m).is_err());
    }
    fn send<T: Send + 'static>() {}
    send::<PreparedInt8Sentiment>(); send::<Int8SentimentRun>();
}
