//! Pinned compilation and fixed admission only, never synthetic native success.
use super::*;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::strict_int8::STRICT_INT8_EXECUTION,
    tasks::judge::{PairwisePolicy, RubricDefinition, RubricCriterion, RubricPolicy, FaithfulnessPolicy},
    tokenizer::pinned_controls,
};

pub(in crate::hosted::corpus::judge) struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
pub(in crate::hosted::corpus::judge) fn fixture() -> (JudgePlanner, JudgeCorpusConfig) {
    let registry = pinned_controls::pinned().unwrap();
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    let planner = JudgePlanner::pinned(controls, eos).unwrap();
    let d = Sha256Digest::of_bytes(b"judge-corpus-model-free-fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".into(), logical_model_digest: d,
        artifact_format: "fixture".into(), quant_recipe: "fixture-int8".into(), packing_set_digest: d,
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(), task_spec: "judge-v1".into(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".into(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".into(),
        sampler_version: "fixture".into(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.into(),
        host_class: None, compiler_identity: None };
    let task_ceiling = TaskBudget { max_input_tokens: 4096, max_output_tokens: 16,
        max_output_bytes: 1 << 20, max_grammar_states: 4096, max_kv_bytes: 1 << 30 };
    let config = JudgeCorpusConfig { identity, task_ceiling, planning: JudgeLimits::default(),
        defaults: Some(pairwise(task_ceiling)), max_model_work: Int8Work::for_sequence(0, 8192, 100_000_000).unwrap() };
    (planner, config)
}
pub(in crate::hosted::corpus::judge) fn pairwise(budget: TaskBudget) -> JudgeBatchArgs {
    JudgeBatchArgs::Pairwise { criterion: "Accuracy".into(), b: "Bob lives in London.".into(),
        policy: PairwisePolicy { minimum_margin_milli: 100, maximum_order_disagreement_milli: 20_000 }, budget }
}
fn rubric(budget: TaskBudget) -> JudgeBatchArgs {
    JudgeBatchArgs::Rubric { rubric: RubricDefinition { schema_version: 1, revision: "local-v1".into(),
        declared_origin_digest: Sha256Digest::of_bytes(b"declared-not-authenticated"), scale_maximum: 1,
        criteria: vec![RubricCriterion { id: "accuracy".into(), description: "Accuracy".into(), weight: 2 },
            RubricCriterion { id: "clarity".into(), description: "Clarity".into(), weight: 1 }] },
        policy: RubricPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 }, budget }
}
fn faithful(budget: TaskBudget) -> JudgeBatchArgs {
    JudgeBatchArgs::Faithfulness { claim: "Alice lives in Paris.".into(), policy: FaithfulnessPolicy {
        minimum_candidate_weight_ppm: 900_000, minimum_margin_milli: 100,
        evidence_window_bytes: 16, max_evidence_windows: 8, max_evidence_spans: 8 }, budget }
}
fn document(args: Option<JudgeBatchArgs>) -> BatchDocument<JudgeBatchArgs> {
    BatchDocument { id: "row".into(), text: "Alice lives in Paris. She writes.".into(), task_args: args }
}
#[test]
fn all_three_modes_use_real_pinned_complete_head_planning() {
    let (planner, config) = fixture(); config.validate(&planner).unwrap();
    let pair = prepare(&planner, &config, document(None), &mut Continue).unwrap();
    assert_eq!(pair.head_count(), 2);
    let rubric = prepare(&planner, &config, document(Some(rubric(config.task_ceiling))), &mut Continue).unwrap();
    assert_eq!(rubric.head_count(), 2);
    let faith = prepare(&planner, &config, document(Some(faithful(config.task_ceiling))), &mut Continue).unwrap();
    assert!(faith.head_count() > 1);
    for plan in [&pair, &rubric, &faith] {
        assert!(plan.planned_work().attention_pairs > 0);
        assert!(plan.planned_work().projections.multiply_accumulates > 0);
        assert_eq!(plan.execution_identity().logical_model_digest, config.identity.logical_model_digest);
        assert!(plan.max_result_bytes() <= config.task_ceiling.max_output_bytes);
    }
    let again = prepare(&planner, &config, document(None), &mut Continue).unwrap();
    assert_eq!(pair.execution_identity(), again.execution_identity());
}
#[test]
fn every_record_budget_axis_is_bounded_by_the_host_not_by_its_json() {
    let (_, config) = fixture();
    for axis in 0..5 {
        let mut budget = config.task_ceiling;
        match axis { 0 => budget.max_input_tokens += 1, 1 => budget.max_output_tokens += 1,
            2 => budget.max_output_bytes += 1, 3 => budget.max_grammar_states += 1, _ => budget.max_kv_bytes += 1 }
        let error = check_args(&pairwise(budget), config.task_ceiling).unwrap_err();
        assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::Planning);
    }
    let mut smaller = config.task_ceiling; smaller.max_output_bytes -= 1;
    check_args(&pairwise(smaller), config.task_ceiling).unwrap();
}
#[test]
fn controlled_preparation_preserves_each_cancellation_cause() {
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    let (planner, config) = fixture();
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::CostBudget] {
        let error = prepare(&planner, &config, document(None), &mut Stop(cause)).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.cancellation, Some(cause));
    }
}
#[test]
fn missing_defaults_do_not_become_a_successful_empty_judgment() {
    let (planner, mut config) = fixture(); config.defaults = None;
    let error = prepare(&planner, &config, document(None), &mut Continue).err().unwrap();
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::Planning);
    assert!(prepare(&planner, &config, document(Some(pairwise(config.task_ceiling))), &mut Continue).is_ok());
}
#[test]
fn fixed_defaults_refuse_invalid_policy_budget_and_empty_private_text() {
    let (planner, mut config) = fixture();
    let mut args = pairwise(config.task_ceiling);
    if let JudgeBatchArgs::Pairwise { b, .. } = &mut args { b.clear(); }
    config.defaults = Some(args); assert!(config.validate(&planner).is_err());
    let mut args = faithful(config.task_ceiling);
    if let JudgeBatchArgs::Faithfulness { policy, .. } = &mut args { policy.max_evidence_windows = 0; }
    config.defaults = Some(args); assert!(config.validate(&planner).is_err());
    let mut args = rubric(config.task_ceiling);
    if let JudgeBatchArgs::Rubric { policy, .. } = &mut args { policy.minimum_peak_weight_ppm = 1_000_001; }
    config.defaults = Some(args); assert!(config.validate(&planner).is_err());
}
#[test]
fn milli_log_odds_are_not_accidentally_limited_as_ppm_weights() {
    let (_, config) = fixture(); let mut args = pairwise(config.task_ceiling);
    if let JudgeBatchArgs::Pairwise { policy, .. } = &mut args {
        policy.minimum_margin_milli = u32::MAX; policy.maximum_order_disagreement_milli = u32::MAX;
    }
    check_args(&args, config.task_ceiling).unwrap();
}
#[test]
fn borrowed_size_counter_includes_escaping_and_has_an_exact_boundary() {
    let value = vec!["é\n\"\\", "private"];
    let bytes = serde_json::to_vec(&value).unwrap().len();
    check_serialized_size(&value, bytes).unwrap();
    assert!(check_serialized_size(&value, bytes - 1).is_err());
    assert!(check_serialized_size(&value, 0).is_err());
}
#[test]
fn malformed_fixed_identity_and_unbounded_defaults_refuse_before_stream_use() {
    let (planner, mut config) = fixture();
    config.identity.task_spec = "sentiment-v1".into(); assert!(config.validate(&planner).is_err());
    config.identity.task_spec = "judge-v1".into();
    if let Some(JudgeBatchArgs::Pairwise { b, .. }) = &mut config.defaults { *b = "\n".repeat(MAX_DEFAULT_BYTES); }
    assert!(config.validate(&planner).is_err());
}
