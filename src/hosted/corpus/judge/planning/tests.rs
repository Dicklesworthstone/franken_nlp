//! Pinned compilation and fixed admission only, never synthetic native success.
use super::*;
use crate::{
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::strict_int8::STRICT_INT8_EXECUTION,
    tasks::judge::{PairwisePolicy, RubricDefinition, RubricCriterion, RubricPolicy, FaithfulnessPolicy},
    tokenizer::pinned_controls,
    tokenizer::untrusted::SafeBpeLimits,
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

pub(in crate::hosted::corpus::judge) fn compact_fixture()
    -> (JudgePlanner, JudgeCorpusConfig, SafeBpeWorkLimits) {
    let (planner, mut config) = fixture();
    let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1_000_000, max_pair_bytes: 16_000_000,
    });
    let planner = planner.with_document_encoding(policy).unwrap();
    config.identity.template_digest = *planner.template_digest();
    let total = SafeBpeWorkLimits { max_input_bytes: 65_536, max_output_tokens: 65_536,
        max_pair_checks: 16_000_000, max_pair_bytes: 256_000_000 };
    (planner, config, total)
}

#[test]
fn compact_host_quote_prices_the_actual_policy_and_one_shared_owner() {
    let (planner, config, total) = compact_fixture();
    let (policy, caps, bytes) = config.encoding_requirements(&planner, Some(total)).unwrap();
    assert_eq!(&policy, planner.document_encoding());
    assert_eq!(caps, Some(total));
    let UntrustedEncoding::SafeBpeV1(call) = policy else { panic!("compact fixture") };
    assert_eq!(bytes, call.requirements().unwrap().workspace_bytes + 1024);
    assert!(config.encoding_requirements(&planner, None).is_err());
    let mut invalid = total; invalid.max_pair_checks = 0;
    assert!(config.encoding_requirements(&planner, Some(invalid)).is_err());
    let (plain, config) = fixture();
    assert_eq!(config.encoding_requirements(&plain, None).unwrap(), (UntrustedEncoding::ByteFallback, None, 0));
    assert!(config.encoding_requirements(&plain, Some(total)).is_err());
}

#[test]
fn compact_pairwise_rubric_and_evidence_share_real_work_without_rebinding_prompts() {
    let (planner, config, total) = compact_fixture();
    let encoding = EncodingWorkScope::new(planner.document_encoding(), Some(total)).unwrap();
    for args in [None, Some(rubric(config.task_ceiling)), Some(faithful(config.task_ceiling))] {
        let before = encoding.work().unwrap();
        let first = prepare_with_encoding_scope(&planner, &config, &encoding,
            document(args.clone()), &mut Continue).unwrap();
        let charged = encoding.work().unwrap().checked_sub(before).unwrap();
        assert_eq!(first.input_encoding_work(), Some(charged));
        assert!(charged.input_bytes() > 0 && charged.output_tokens() > 0 && charged.pair_checks() > 0);
        assert_eq!(first.input_encoding(), planner.document_encoding());
        let again = prepare_with_encoding_scope(&planner, &config, &encoding,
            document(args), &mut Continue).unwrap();
        assert_eq!(first.execution_identity(), again.execution_identity());
        assert_eq!(first.input_encoding_work(), again.input_encoding_work());
        assert_eq!(first.planned_work(), again.planned_work());
    }
}

#[test]
fn compact_pairwise_and_rubric_reuse_encoded_segments_across_heads() {
    let (planner, config, total) = compact_fixture();
    let encoding = EncodingWorkScope::new(planner.document_encoding(), Some(total)).unwrap();
    let pair = prepare_with_encoding_scope(&planner, &config, &encoding,
        document(None), &mut Continue).unwrap();
    assert_eq!(pair.head_count(), 2);
    assert_eq!(pair.input_encoding_work().unwrap().input_bytes(),
        "Accuracy".len() + "Bob lives in London.".len() + document(None).text.len());
    let rubric = prepare_with_encoding_scope(&planner, &config, &encoding,
        document(Some(rubric(config.task_ceiling))), &mut Continue).unwrap();
    assert_eq!(rubric.head_count(), 2);
    assert_eq!(rubric.input_encoding_work().unwrap().input_bytes(),
        document(None).text.len() + "Accuracy".len() + "Clarity".len());
}

#[test]
fn rejected_partial_encoding_stays_charged_and_a_bounded_record_can_continue() {
    let (planner, config, total) = compact_fixture();
    let encoding = EncodingWorkScope::new(planner.document_encoding(), Some(total)).unwrap();
    let mut invalid = pairwise(config.task_ceiling);
    if let JudgeBatchArgs::Pairwise { b, .. } = &mut invalid { *b = "x".repeat(4096); }
    let error = prepare_with_encoding_scope(&planner, &config, &encoding,
        document(Some(invalid)), &mut Continue).err().unwrap();
    assert!(!error.stop);
    let failed = encoding.work().unwrap();
    assert!(failed.input_bytes() > 0 && failed.pair_checks() > 0);
    let next = prepare_with_encoding_scope(&planner, &config, &encoding,
        document(None), &mut Continue).unwrap();
    assert_eq!(encoding.work().unwrap(), failed.checked_add(next.input_encoding_work().unwrap()).unwrap());
}

#[test]
fn every_invocation_axis_stays_spent_across_records() {
    let (planner, config, total) = compact_fixture();
    let probe = EncodingWorkScope::new(planner.document_encoding(), Some(total)).unwrap();
    prepare_with_encoding_scope(&planner, &config, &probe, document(None), &mut Continue).unwrap();
    let work = probe.work().unwrap();
    for axis in 0..4 {
        let mut bounded = total;
        match axis {
            0 => bounded.max_input_bytes = work.input_bytes() as u64,
            1 => bounded.max_output_tokens = work.output_tokens() as u64,
            2 => bounded.max_pair_checks = work.pair_checks(),
            _ => bounded.max_pair_bytes = work.pair_bytes(),
        }
        let encoding = EncodingWorkScope::new(planner.document_encoding(), Some(bounded)).unwrap();
        prepare_with_encoding_scope(&planner, &config, &encoding, document(None), &mut Continue).unwrap();
        assert_eq!(encoding.work().unwrap(), work);
        let error = prepare_with_encoding_scope(&planner, &config, &encoding,
            document(None), &mut Continue).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::WorkLimit);
        assert_eq!(encoding.work().unwrap(), work);
    }
}

#[test]
fn compact_preparation_retains_all_cancellation_causes_and_spent_work() {
    use DecodeCancellationKind::*;
    struct Count(usize);
    impl DecodeStepControl for Count {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { self.0 += 1; None }
    }
    struct Stop { seen: usize, at: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.seen += 1; (self.seen >= self.at).then_some(self.cause)
        }
    }
    let (planner, config, total) = compact_fixture();
    let item = || {
        let mut item = document(None); item.text = "Alice writes accurate answers. ".repeat(16); item
    };
    let probe = EncodingWorkScope::new(planner.document_encoding(), Some(total)).unwrap();
    let mut count = Count(0);
    prepare_with_encoding_scope(&planner, &config, &probe, item(), &mut count).unwrap();
    assert!(count.0 > 16);
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast,
        RaceLost, ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let encoding = EncodingWorkScope::new(planner.document_encoding(), Some(total)).unwrap();
        let error = prepare_with_encoding_scope(&planner, &config, &encoding, item(),
            &mut Stop { seen: 0, at: count.0 / 2, cause }).err().unwrap();
        assert!(error.stop); assert_eq!(error.fault.cancellation, Some(cause));
        assert!(encoding.work().unwrap().input_bytes() > 0);
    }
}

#[test]
fn another_policy_cannot_borrow_a_scope_even_when_workspace_geometry_matches() {
    let (planner, config, total) = compact_fixture();
    let UntrustedEncoding::SafeBpeV1(mut call) = planner.document_encoding().clone() else { panic!("compact fixture") };
    call.max_pair_checks -= 1;
    let encoding = EncodingWorkScope::new(&UntrustedEncoding::SafeBpeV1(call), Some(total)).unwrap();
    let error = prepare_with_encoding_scope(&planner, &config, &encoding,
        document(None), &mut Continue).err().unwrap();
    assert!(error.stop); assert_eq!(error.fault.code, BatchCode::Admission);
    assert_eq!(encoding.work().unwrap(), SafeBpeWork::default());
}

#[test]
fn output_count_includes_metadata_and_retains_terminal_cancellation_precedence() {
    let value = serde_json::json!({ "result": "é\n\"", "input_encoding": "safe-bpe-v1" });
    let bytes = serde_json::to_vec(&value).unwrap().len() as u64;
    check_output_bytes(&value, bytes, &mut Continue).unwrap();
    let error = check_output_bytes(&value, bytes - 1, &mut Continue).unwrap_err();
    assert!(!error.stop); assert_eq!(error.fault.code, BatchCode::OutputLineLimit);
    struct StopOnTerminal(usize);
    impl DecodeStepControl for StopOnTerminal {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.0 += 1; (self.0 >= 3).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let error = check_output_bytes(&value, 0, &mut StopOnTerminal(0)).unwrap_err();
    assert!(error.stop); assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline));
}
