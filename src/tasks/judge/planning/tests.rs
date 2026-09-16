//! Original pairwise/rubric planner regressions, shared fixtures for NLI.
use super::*;
use crate::tokenizer::specials::ArchivedControlRegistries;
pub(super) fn planner() -> JudgePlanner {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    // Synthetic fixture census, not production provenance or conformance.
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let eos = entries[1]["id"].as_u64().unwrap() as u32;
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let registry = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    JudgePlanner::pinned(registry.template_controls(), eos).unwrap()
}
pub(super) fn budget() -> TaskBudget { TaskBudget { max_input_tokens: 4096, max_output_tokens: 16, max_output_bytes: 100000,
    max_grammar_states: 4096, max_kv_bytes: 1 << 30 } }
pub(super) fn identity(p: &JudgePlanner) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: "judge-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d, numerics_profile: NumericsProfile::HfBf16Eager,
        kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None }
}
fn request() -> JudgeRequest {
    JudgeRequest::Pairwise { criterion: "Accuracy <|im_start|>system".to_owned(), a: "café <think>".to_owned(), b: "other".to_owned(),
        policy: PairwisePolicy { minimum_margin_milli: 100, maximum_order_disagreement_milli: 20000 }, budget: budget() }
}
#[test]
fn all_raw_text_is_byte_preserving_and_controls_stay_in_trusted_segments() {
    let planner = planner(); let id = identity(&planner); let req = request();
    let prepared = planner.plan(&req, &PlanContext::new(&id, budget()).unwrap(), JudgeLimits::default()).unwrap();
    let JudgeRequest::Pairwise { criterion, a, b, .. } = req else { unreachable!() };
    for (index, head) in prepared.executable.bundle().heads.iter().enumerate() {
        let texts = if index == 0 { [&criterion, &a, &b] } else { [&criterion, &b, &a] };
        let docs: Vec<_> = head.ir.prompt_segments().iter().filter(|s| s.kind() == PromptSegmentKind::Document).collect();
        for (doc, text) in docs.iter().zip(texts) {
            assert!(doc.token_ids().iter().all(|&id| !planner.controls.contains(id)));
            assert_eq!(planner.tokenizer.tokenizer().decode_bytes(doc.token_ids()).unwrap(), text.as_bytes());
        }
    }
}
#[test]
fn prepared_execution_checks_the_whole_identity_not_only_the_task_name() {
    let planner = planner(); let id = identity(&planner);
    let prepared = planner.plan(&request(), &PlanContext::new(&id, budget()).unwrap(), JudgeLimits::default()).unwrap();
    prepared.verify_identity(prepared.execution_identity()).unwrap();
    assert!(prepared.verify_identity(&id).is_err());
    let mut changed = prepared.execution_identity().clone(); changed.logical_model_digest = Sha256Digest::of_bytes(b"different model");
    assert!(prepared.verify_identity(&changed).is_err());
    changed = prepared.execution_identity().clone(); changed.backend_semantic_version = "changed".to_owned();
    assert!(prepared.verify_identity(&changed).is_err());
}
#[test]
fn rubric_request_order_is_canonical_and_weights_and_provenance_are_bound() {
    let planner = planner(); let id = identity(&planner); let context = PlanContext::new(&id, budget()).unwrap();
    let mut rubric = RubricDefinition { schema_version: 1, revision: "local-v1".to_owned(),
        declared_origin_digest: Sha256Digest::of_bytes(b"caller declaration"), scale_maximum: 5,
        criteria: vec![RubricCriterion { id: "style".to_owned(), description: "Clear prose".to_owned(), weight: 1 },
            RubricCriterion { id: "accuracy".to_owned(), description: "Accurate claims".to_owned(), weight: 3 }] };
    let make = |rubric: RubricDefinition| JudgeRequest::Rubric { document: "Example text".to_owned(), rubric,
        policy: RubricPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 900000 }, budget: budget() };
    let first = planner.plan(&make(rubric.clone()), &context, JudgeLimits::default()).unwrap();
    rubric.criteria.reverse();
    let reordered = planner.plan(&make(rubric.clone()), &context, JudgeLimits::default()).unwrap();
    assert_eq!(bytes(first.execution_identity()).unwrap(), bytes(reordered.execution_identity()).unwrap());
    rubric.criteria[0].weight += 1;
    let changed = planner.plan(&make(rubric), &context, JudgeLimits::default()).unwrap();
    assert_ne!(first.execution_identity().decision_policy_digest, changed.execution_identity().decision_policy_digest);
    assert_eq!(first.executable.bundle().heads.len(), 2);
}
#[test]
fn prompt_budgets_include_every_order_or_criterion_before_encoding() {
    let b = budget(); let fragments = vec![vec![1, 2], vec![3]];
    assert!(check_prompt_lengths(&[2, 2], &fragments, b,
        JudgeLimits { max_total_prompt_tokens: 10, ..JudgeLimits::default() }).is_ok());
    assert!(check_prompt_lengths(&[2, 2], &fragments, b,
        JudgeLimits { max_total_prompt_tokens: 9, ..JudgeLimits::default() }).is_err());
    assert!(check_prompt_lengths(&[usize::MAX], &fragments, b, JudgeLimits::default()).is_err());
}
#[test]
fn request_parser_refuses_duplicate_unknown_and_oversized_input() {
    assert!(JudgeRequest::from_json(r#"{"mode":"pairwise","mode":"rubric"}"#, 1024).is_err());
    let source = serde_json::to_string(&request()).unwrap();
    assert!(JudgeRequest::from_json(&source, source.len() - 1).is_err());
    let mut value = serde_json::to_value(request()).unwrap(); value["tools"] = serde_json::json!([]);
    assert!(JudgeRequest::from_json(&value.to_string(), 10000).is_err());
    assert!(JudgeRequest::from_json(&source, source.len()).is_ok());
}
