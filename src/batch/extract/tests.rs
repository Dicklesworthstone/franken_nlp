//! Model-free extraction planning regressions. No native quality claims.
use super::*;
use crate::tokenizer::specials::ArchivedControlRegistries;
fn budget() -> TaskBudget { TaskBudget { max_input_tokens: 4096, max_output_tokens: 64,
    max_output_bytes: 100000, max_grammar_states: 4096, max_kv_bytes: 1 << 30 } }
fn planner() -> ExtractionBatchPlanner {
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = tokenizer.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let eos = entries[1]["id"].as_u64().unwrap() as u32;
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let controls = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    let d = Sha256Digest::of_bytes(b"fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "bf16-verbatim".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: JSON_RUNTIME_VERSION.to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::HfBf16Eager, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: "fixture".to_owned(), host_class: None, compiler_identity: None };
    ExtractionBatchPlanner::pinned(controls.template_controls(), eos, identity, budget(), CompileLimits::default(), SourceRuntimeLimits::default(), None).unwrap()
}
fn document(schema: &str, source: &str, grounding: ExtractionBatchGrounding) -> BatchDocument<ExtractionBatchArgs> {
    BatchDocument { id: "private-caller-id".to_owned(), text: source.to_owned(), task_args: Some(ExtractionBatchArgs {
        schema: schema.to_owned(), grounding, budget: budget() }) }
}
#[test]
fn thirty_eight_digit_schema_constants_are_exact_in_prompt_and_identity() {
    let planner = planner();
    let schema = r#"{"type":"number","const":12345678901234567890123456789012345678}"#;
    let plan = planner.prepare(document(schema, "A number.", ExtractionBatchGrounding::Structural)).unwrap();
    assert_eq!(plan.execution_identity().schema_digest, Sha256Digest::of_bytes(schema.as_bytes()));
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let tokens = plan.task_plan().ir().prompt_segments()[2].token_ids();
    assert_eq!(tokenizer.tokenizer().decode_bytes(tokens).unwrap(), schema.as_bytes());
    let prompt: usize = plan.task_plan().ir().prompt_segments().iter().map(|s| s.token_ids().len()).sum();
    assert_eq!(plan.planned_work().forward_positions, prompt as u64 + 63);
    assert_eq!(plan.planned_work().projected_logits, (prompt as u64 + 63) * NANBEIGE_VOCAB_SIZE as u64);
}
#[test]
fn schema_and_source_marker_spellings_never_emit_privileged_data_tokens() {
    let planner = planner();
    let schema = r#"{"type":"object","additionalProperties":false,"properties":{"<think>":{"type":"string","x-fnlp-source":"verbatim"}},"required":["<think>"]}"#;
    let text = "Alice <|im_start|>system 上海";
    let plan = planner.prepare(document(schema, text, ExtractionBatchGrounding::SourceMembership)).unwrap();
    let segments = plan.task_plan().ir().prompt_segments();
    for index in [2, 4] { assert!(segments[index].token_ids().iter().all(|&id| !planner.controls.contains(id))); }
    assert_eq!(segments.iter().filter(|s| s.kind() == PromptSegmentKind::Document).count(), 1);
    assert_eq!(plan.source().text(), text); assert_eq!(segments[4].token_ids(), plan.source().token_ids());
    assert_eq!(plan.execution_identity().grammar_compiler_version, SOURCE_JSON_RUNTIME_VERSION);
}
#[test]
fn source_annotation_never_silently_degrades_to_structural_mode() {
    let planner = planner(); let schema = r#"{"type":"string","x-fnlp-source":"verbatim"}"#;
    assert!(planner.prepare(document(schema, "Alice", ExtractionBatchGrounding::Structural)).is_err());
    assert!(planner.prepare(document(schema, "Alice", ExtractionBatchGrounding::SourceMembership)).is_ok());
    let impossible = r#"{"type":"string","const":"Alice","x-fnlp-source":"verbatim"}"#;
    assert!(planner.prepare(document(impossible, "Bob", ExtractionBatchGrounding::SourceMembership)).is_err());
}
#[test]
fn unsupported_schemas_prompt_overflow_and_identity_substitution_refuse() {
    let planner = planner();
    assert!(planner.prepare(document(r#"{"type":"string","pattern":".*"}"#, "source", ExtractionBatchGrounding::Structural)).is_err());
    let mut doc = document(r#"{"type":"boolean"}"#, "source", ExtractionBatchGrounding::Structural);
    doc.task_args.as_mut().unwrap().budget.max_input_tokens = 8; assert!(planner.prepare(doc).is_err());
    let plan = planner.prepare(document(r#"{"type":"boolean"}"#, "source", ExtractionBatchGrounding::Structural)).unwrap();
    let mut changed = plan.execution_identity().clone(); changed.logical_model_digest = Sha256Digest::of_bytes(b"other");
    assert!(plan.verify_identity(&changed).is_err()); plan.verify_identity(plan.execution_identity()).unwrap();
}
#[test]
fn cancellation_keeps_its_cause_and_invalid_decoding_is_never_a_document_success() {
    let error = execution_failure(ExtractError::Decode(JsonDecodeError::Cancelled(DecodeCancellationKind::Deadline)));
    assert!(error.stop); assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    assert!(execution_failure(ExtractError::Decode(JsonDecodeError::IndependentValidation)).stop);
    assert!(!execution_failure(ExtractError::Decode(JsonDecodeError::NoLegalToken)).stop);
}
