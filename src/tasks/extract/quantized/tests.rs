//! Synthetic compiler/finalizer fixtures, not quantized-model quality evidence.
use super::*;
use crate::{tasks::{BuiltInTask, ir::{TaskBudget, PlanContext, PromptSegment}},
    native_engine::strict_int8::STRICT_INT8_EXECUTION,
    validation::grounded_fields::SourceOccurrence};
use super::super::tests::{registry, options, identity as eager_identity};

const BOOLEAN: &str = r#"{"type":"boolean"}"#;
const SOURCE: &str = r#"{"type":"string","maxLength":64,"x-fnlp-source":"verbatim"}"#;
fn identity() -> ExecutionIdentity {
    let mut identity = eager_identity();
    identity.numerics_profile = NumericsProfile::StrictQuantized { version: 1 };
    identity.backend_semantic_version = STRICT_INT8_EXECUTION.to_owned();
    identity.quant_recipe = "portable-quant-v1".to_owned(); identity
}
fn task(schema: &str, documents: Vec<Vec<u32>>, source: bool, output_bytes: u64) -> TaskPlan {
    let id = identity();
    let budget = TaskBudget { max_input_tokens: 1024, max_output_tokens: 8,
        max_output_bytes: output_bytes, max_grammar_states: 4096, max_kv_bytes: 1 << 30 };
    let mut segments = vec![PromptSegment::new(PromptSegmentKind::TaskInstruction, vec![1])];
    segments.extend(documents.into_iter().map(|ids| PromptSegment::new(PromptSegmentKind::Document, ids)));
    segments.push(PromptSegment::new(PromptSegmentKind::AnswerScaffold, vec![2]));
    let mut post = vec![FinitePostcondition::JsonValid, FinitePostcondition::OutputWithinBudget];
    if source { post.push(FinitePostcondition::SourceSpansVerified); }
    let ir = TaskIR::new(segments, DecodeStrategy::ConstrainedJson,
        GrammarReference::json_schema(Sha256Digest::of_bytes(schema.as_bytes()),
            if source { SOURCE_JSON_RUNTIME_VERSION } else { JSON_RUNTIME_VERSION }),
        None, post, budget, DependencyScope::ItemLocal).unwrap();
    TaskPlan::new(BuiltInTask::Extract.spec(), &PlanContext::new(&id, budget).unwrap(), ir).unwrap()
}
fn plan(schema: &str) -> Int8ExtractPlan {
    Int8ExtractPlan::from_task_plan(&task(schema, vec![vec![7]], false, 16384), schema, options(),
        CompileLimits::default(), registry().template_controls(), identity()).unwrap()
}
fn raw(p: &Int8ExtractPlan, json: &str) -> Int8JsonRun {
    let work = constrained_int8::planned_work(p.prompt_tokens(), 2).unwrap();
    Int8JsonRun { schema_version: 1, execution: INT8_JSON_EXECUTION.to_owned(), model_work: work,
        output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.to_owned(),
            token_ids: vec![1, 0], json: json.to_owned(), forward_positions: work.forward_positions,
            projected_logits: work.projected_logits, mask_node_visit_charge: 20 } }
}
fn source_plan(text: &str, contexts: &[&str]) -> Int8ExtractPlan {
    let r = registry(); let encoder = SourceDocumentEncoder::pinned(r.template_controls()).unwrap();
    let d = encoder.encode_with_context(text, contexts, 1024, 1024).unwrap();
    let docs = d.context_token_ids().chain(std::iter::once(d.token_ids())).map(|ids| ids.to_vec()).collect();
    Int8ExtractPlan::from_task_plan_with_source(&task(SOURCE, docs, true, 16384), SOURCE, options(),
        CompileLimits::default(), r.template_controls(), &d, SourceRuntimeLimits::default(), identity()).unwrap()
}

#[test]
fn compilation_binds_quantized_work_and_keeps_eager_identity_incompatible() {
    let p = plan(BOOLEAN);
    p.verify_identity(p.execution_identity()).unwrap();
    assert_eq!(p.planned_work().forward_positions, p.prompt_tokens() as u64 + 7);
    assert_eq!(p.planned_work().projected_logits, 8 * NANBEIGE_VOCAB_SIZE as u64);
    assert!(p.extraction.bind_identity(identity()).is_err());
    assert!(p.extraction.verify_identity(p.execution_identity()).is_err());
    let t = task(BOOLEAN, vec![vec![7]], false, 16384);
    assert!(Int8ExtractPlan::from_task_plan(&t, BOOLEAN, options(), CompileLimits::default(),
        registry().template_controls(), eager_identity()).is_err());
}

#[test]
fn every_execution_identity_field_is_checked_not_just_task_digests() {
    let p = plan(BOOLEAN);
    let expected = canonjson::canonical_bytes(p.execution_identity()).unwrap();
    let value = serde_json::to_value(p.execution_identity()).unwrap();
    for key in value.as_object().unwrap().keys() {
        let mut changed = value.clone();
        let old = &changed[key];
        let new = match old {
            serde_json::Value::String(s) if s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()) =>
                serde_json::Value::String(Sha256Digest::of_bytes(b"different").to_hex()),
            serde_json::Value::String(_) => serde_json::Value::String("different".to_owned()),
            serde_json::Value::Number(_) => serde_json::json!(99),
            serde_json::Value::Null => serde_json::Value::String("different".to_owned()),
            _ => serde_json::Value::Null,
        };
        changed[key] = new;
        // A wire/type refusal is also fail-closed; all representable changes
        // must be rejected by the sealed plan itself.
        if let Ok(changed) = serde_json::from_value::<ExecutionIdentity>(changed) {
            assert!(p.verify_identity(&changed).is_err(), "accepted changed field {key}");
        }
    }
    assert_eq!(expected, canonjson::canonical_bytes(p.execution_identity()).unwrap());
}

#[test]
fn profile_backend_kv_thinking_and_tools_are_never_silently_repaired() {
    let t = task(BOOLEAN, vec![vec![7]], false, 16384);
    for axis in 0..6 {
        let mut id = identity();
        match axis {
            0 => id.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => id.numerics_profile = NumericsProfile::StrictQuantized { version: 2 },
            2 => id.backend_semantic_version = "other".to_owned(), 3 => id.kv_dtype = "int8".to_owned(),
            4 => id.thinking_mode = ThinkingMode::Enabled, _ => id.tool_mode = ToolMode::Json,
        }
        assert!(Int8ExtractPlan::from_task_plan(&t, BOOLEAN, options(), CompileLimits::default(),
            registry().template_controls(), id).is_err());
    }
}

#[test]
fn options_caps_and_control_exclusions_are_bound_without_changing_eager_policy() {
    let t = task(BOOLEAN, vec![vec![7]], false, 16384); let r = registry();
    let a = plan(BOOLEAN);
    assert!(a.options().excluded_token_ids.contains(&0)); assert!(a.options().excluded_token_ids.contains(&3));
    for axis in 0..3 {
        let mut o = options(); let mut l = CompileLimits::default();
        match axis { 0 => o.max_new_tokens = 7, 1 => { o.excluded_token_ids.insert(8); }, _ => l.max_array_items = 1 }
        let b = Int8ExtractPlan::from_task_plan(&t, BOOLEAN, o, l, r.template_controls(), identity()).unwrap();
        assert_ne!(a.execution_identity().decision_policy_digest, b.execution_identity().decision_policy_digest);
        assert!(a.verify_identity(b.execution_identity()).is_err());
    }
    let eager = a.extraction.bind_identity(eager_identity()).unwrap();
    assert_ne!(eager.decision_policy_digest, a.execution_identity().decision_policy_digest);
}

#[test]
fn source_swap_extra_context_and_missing_source_postcondition_fail_before_execution() {
    let r = registry(); let e = SourceDocumentEncoder::pinned(r.template_controls()).unwrap();
    let a = e.encode("Alice", 1024, 1024).unwrap(); let b = e.encode("Bob", 1024, 1024).unwrap();
    for mode in 0..3 {
        let mut docs = vec![a.token_ids().to_vec()]; if mode == 2 { docs.push(vec![7]); }
        let t = task(SOURCE, docs, mode != 1, 16384);
        let result = Int8ExtractPlan::from_task_plan_with_source(&t, SOURCE, options(), CompileLimits::default(),
            r.template_controls(), if mode == 0 { &b } else { &a }, SourceRuntimeLimits::default(), identity());
        assert!(result.is_err());
    }
}

#[test]
fn repeated_unicode_source_evidence_keeps_all_original_coordinates() {
    let p = source_plan("é Alice 上海 Alice", &[]);
    let out = p.finalize(raw(&p, r#""Alice""#)).unwrap();
    assert_eq!(out.result.schema_version, 2); assert_eq!(out.result.grounding, ExtractionGrounding::SourceMembership);
    let evidence = &out.result.source_fields[0];
    assert_eq!(evidence.occurrence, SourceOccurrence::Ambiguous); assert_eq!(evidence.spans.len(), 2);
    for span in &evidence.spans {
        let source = "é Alice 上海 Alice";
        assert_eq!(&source[span.byte_start..span.byte_end], "Alice");
        assert_eq!(source[..span.byte_start].chars().count(), span.scalar_start);
    }
    assert_eq!(out.result.output.numerics_profile, STRICT_INT8_PROFILE);
}

#[test]
fn auxiliary_question_or_metadata_cannot_authorize_a_citation() {
    let p = source_plan("Alice", &["Carol?", "Bob metadata"]);
    assert!(p.finalize(raw(&p, r#""Alice""#)).is_ok());
    for text in [r#""Carol""#, r#""Bob""#, r#""Mallory""#] { assert!(p.finalize(raw(&p, text)).is_err()); }
}

#[test]
fn finalizer_refuses_wrong_profile_eos_tokens_schema_and_native_accounting() {
    let p = plan(BOOLEAN);
    for axis in 0..8 {
        let mut run = raw(&p, "true");
        match axis {
            0 => run.output.numerics_profile = HF_BF16_EAGER_PROFILE.to_owned(),
            1 => { run.output.token_ids.pop(); }, 2 => run.output.token_ids[0] = 3,
            3 => run.output.json = "tru".to_owned(), 4 => run.output.json = "null".to_owned(),
            5 => run.model_work.attention_pairs += 1, 6 => run.output.projected_logits += 1,
            _ => run.execution = "eager".to_owned(),
        }
        assert!(p.finalize(run).is_err());
    }
}

#[test]
fn complete_outer_envelope_has_an_exact_byte_limit_including_model_work_and_evidence() {
    let mut p = source_plan("Alice Alice", &[]);
    let run = p.finalize(raw(&p, r#""Alice""#)).unwrap();
    let bytes = canonjson::canonical_bytes(&run).unwrap().len() as u64;
    assert!(bytes > canonjson::canonical_bytes(&run.result).unwrap().len() as u64);
    p.extraction.max_result_bytes = bytes;
    assert!(p.finalize(raw(&p, r#""Alice""#)).is_ok());
    p.extraction.max_result_bytes -= 1;
    assert!(matches!(p.finalize(raw(&p, r#""Alice""#)),
        Err(Int8ExtractError::Extraction(ExtractError::OutputBudgetExceeded))));
}

#[test]
fn exact_decimal_text_is_not_rounded_through_a_float_value() {
    let p = plan(r#"{"type":"integer"}"#);
    let decimal = "12345678901234567890123456789012345678";
    let out = p.finalize(raw(&p, decimal)).unwrap();
    assert_eq!(out.result.output.json, decimal);
}

#[test]
fn compiler_cannot_drop_a_source_annotation_or_accept_untrusted_control_tokens() {
    for (schema, docs) in [(SOURCE, vec![vec![7]]), (BOOLEAN, vec![vec![3]])] {
        assert!(Int8ExtractPlan::from_task_plan(&task(schema, docs, false, 16384), schema, options(),
            CompileLimits::default(), registry().template_controls(), identity()).is_err());
    }
}

#[test]
fn result_replay_is_canonical_without_exporting_private_identity_or_confidence() {
    let p = plan(BOOLEAN);
    let a = canonjson::canonical_string(&p.finalize(raw(&p, "true")).unwrap()).unwrap();
    let b = canonjson::canonical_string(&p.finalize(raw(&p, "true")).unwrap()).unwrap();
    assert_eq!(a, b);
    for forbidden in ["prompt_digest", "decision_policy_digest", "logical_model_digest", "confidence"] {
        assert!(!a.contains(forbidden));
    }
}
