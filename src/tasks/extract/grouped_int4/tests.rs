//! Synthetic compiler/finalizer cases; these are not model-quality evidence.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::STRICT_INT4_EXECUTION},
    tasks::{BuiltInTask, ir::{TaskBudget, PlanContext, PromptSegment}},
    validation::grounded_fields::SourceOccurrence,
};
use super::super::tests::{registry, options, identity as eager_identity};

const BOOLEAN: &str = r#"{"type":"boolean"}"#;
const SOURCE: &str = r#"{"type":"string","maxLength":64,"x-fnlp-source":"verbatim"}"#;

fn identity(width: usize) -> ExecutionIdentity {
    let mut id = eager_identity();
    id.source_revision = PINNED_REVISION.to_owned();
    id.numerics_profile = NumericsProfile::StrictQuantized { version: 2 };
    id.backend_semantic_version = STRICT_INT4_EXECUTION.to_owned();
    id.artifact_format = INT4_SOURCE_FORMAT.to_owned();
    id.quant_recipe = format!("{GROUPED_INT4_RECIPE}:group={width}");
    id.packing_set_digest = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", id.logical_model_digest, width,
    )).unwrap());
    id
}
fn task(schema: &str, documents: Vec<Vec<u32>>, source: bool, output_bytes: u64) -> TaskPlan {
    let id = identity(127);
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
pub(super) fn plan(schema: &str, width: usize) -> Int4ExtractPlan {
    Int4ExtractPlan::from_task_plan(&task(schema, vec![vec![7]], false, 16384), schema, options(),
        CompileLimits::default(), registry().template_controls(), identity(width), width).unwrap()
}
pub(super) fn raw(plan: &Int4ExtractPlan, json: &str) -> Int4JsonRun {
    let work = constrained_int4::planned_work(plan.group_width(), plan.prompt_tokens(), 2).unwrap();
    Int4JsonRun { schema_version: 1, execution: INT4_JSON_EXECUTION.to_owned(), group_width: plan.group_width(),
        model_work: work, output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT4_PROFILE.to_owned(),
            token_ids: vec![1, 0], json: json.to_owned(), forward_positions: work.forward_positions,
            projected_logits: work.projected_logits, mask_node_visit_charge: 20 } }
}
pub(super) fn source_plan(text: &str, contexts: &[&str]) -> Int4ExtractPlan {
    let registry = registry(); let encoder = SourceDocumentEncoder::pinned(registry.template_controls()).unwrap();
    let source = encoder.encode_with_context(text, contexts, 1024, 1024).unwrap();
    let documents = source.context_token_ids().chain(std::iter::once(source.token_ids())).map(|ids| ids.to_vec()).collect();
    Int4ExtractPlan::from_task_plan_with_source(&task(SOURCE, documents, true, 16384), SOURCE, options(),
        CompileLimits::default(), registry.template_controls(), &source, SourceRuntimeLimits::default(), identity(127), 127).unwrap()
}

#[test]
fn group_partition_controls_full_model_work_and_cannot_relabel_eager_or_int8() {
    let a = plan(BOOLEAN, 16); let b = plan(BOOLEAN, 128);
    a.verify_identity(a.execution_identity()).unwrap();
    assert_eq!(a.planned_work().forward_positions, a.prompt_tokens() as u64 + 7);
    assert_eq!(a.planned_work().projected_logits, 8 * NANBEIGE_VOCAB_SIZE as u64);
    assert_eq!(a.planned_work().projections.multiply_accumulates, b.planned_work().projections.multiply_accumulates);
    assert!(a.planned_work().projections.dot_products > b.planned_work().projections.dot_products);
    assert!(a.verify_identity(b.execution_identity()).is_err());
    assert!(a.extraction.bind_identity(identity(16)).is_err());
    let task = task(BOOLEAN, vec![vec![7]], false, 16384);
    for profile in [NumericsProfile::HfBf16Eager, NumericsProfile::DiagnosticF32,
        NumericsProfile::StrictQuantized { version: 1 }, NumericsProfile::StrictQuantized { version: 3 }] {
        let mut id = identity(16); id.numerics_profile = profile;
        assert!(Int4ExtractPlan::from_task_plan(&task, BOOLEAN, options(), CompileLimits::default(),
            registry().template_controls(), id, 16).is_err());
    }
}

#[test]
fn candidate_namespace_packing_group_controls_and_task_are_checked_before_compilation() {
    let task = task(BOOLEAN, vec![vec![7]], false, 16384);
    for axis in 0..9 {
        let mut id = identity(127);
        match axis {
            0 => id.artifact_format = "fnlpq-v1".to_owned(),
            1 => id.quant_recipe = format!("{GROUPED_INT4_RECIPE}:group=128"),
            2 => id.packing_set_digest = Sha256Digest::of_bytes(b"wrong-layout"),
            3 => id.backend_semantic_version.push('x'), 4 => id.source_revision.push('x'),
            5 => id.kv_dtype = "int8".to_owned(), 6 => id.thinking_mode = ThinkingMode::Enabled,
            7 => id.tool_mode = ToolMode::Json, _ => id.task_spec = "classify-v1".to_owned(),
        }
        assert!(Int4ExtractPlan::from_task_plan(&task, BOOLEAN, options(), CompileLimits::default(),
            registry().template_controls(), id, 127).is_err());
    }
    for width in [0, 128, usize::MAX] {
        assert!(Int4ExtractPlan::from_task_plan(&task, BOOLEAN, options(), CompileLimits::default(),
            registry().template_controls(), identity(127), width).is_err());
    }
}

#[test]
fn every_representable_identity_change_is_refused_without_rebinding() {
    let plan = plan(BOOLEAN, 32);
    let original = canonjson::canonical_bytes(plan.execution_identity()).unwrap();
    let value = serde_json::to_value(plan.execution_identity()).unwrap();
    for key in value.as_object().unwrap().keys() {
        let mut changed = value.clone();
        changed[key] = match &value[key] {
            serde_json::Value::String(s) if s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()) =>
                serde_json::Value::String(Sha256Digest::of_bytes(b"different").to_hex()),
            serde_json::Value::String(_) | serde_json::Value::Null => serde_json::Value::String("different".to_owned()),
            serde_json::Value::Number(_) => serde_json::json!(99),
            _ => serde_json::Value::Null,
        };
        if let Ok(changed) = serde_json::from_value::<ExecutionIdentity>(changed) {
            assert!(plan.verify_identity(&changed).is_err(), "accepted identity field {key}");
        }
    }
    assert_eq!(original, canonjson::canonical_bytes(plan.execution_identity()).unwrap());
}

#[test]
fn bounds_and_exclusions_are_bound_into_the_exact_task_policy() {
    let task = task(BOOLEAN, vec![vec![7]], false, 16384); let registry = registry();
    let original = plan(BOOLEAN, 127);
    assert!(original.options().excluded_token_ids.contains(&0));
    assert!(original.options().excluded_token_ids.contains(&3));
    for axis in 0..3 {
        let mut options = options(); let mut limits = CompileLimits::default();
        match axis { 0 => options.max_new_tokens -= 1, 1 => { options.excluded_token_ids.insert(8); }, _ => limits.max_array_items = 1 }
        let changed = Int4ExtractPlan::from_task_plan(&task, BOOLEAN, options, limits,
            registry.template_controls(), identity(127), 127).unwrap();
        assert_ne!(original.execution_identity().decision_policy_digest, changed.execution_identity().decision_policy_digest);
    }
}

#[test]
fn source_swaps_untrusted_controls_and_missing_postconditions_fail_before_execution() {
    let registry = registry(); let encoder = SourceDocumentEncoder::pinned(registry.template_controls()).unwrap();
    let alice = encoder.encode("Alice", 1024, 1024).unwrap(); let bob = encoder.encode("Bob", 1024, 1024).unwrap();
    for axis in 0..3 {
        let mut documents = vec![alice.token_ids().to_vec()]; if axis == 2 { documents.push(vec![7]); }
        assert!(Int4ExtractPlan::from_task_plan_with_source(&task(SOURCE, documents, axis != 1, 16384), SOURCE,
            options(), CompileLimits::default(), registry.template_controls(), if axis == 0 { &bob } else { &alice },
            SourceRuntimeLimits::default(), identity(127), 127).is_err());
    }
    for (schema, document) in [(SOURCE, vec![7]), (BOOLEAN, vec![3])] {
        assert!(Int4ExtractPlan::from_task_plan(&task(schema, vec![document], false, 16384), schema, options(),
            CompileLimits::default(), registry.template_controls(), identity(127), 127).is_err());
    }
}

#[test]
fn unicode_repeated_mentions_keep_every_original_byte_and_scalar_coordinate() {
    let text = "é Alice 上海 Alice"; let plan = source_plan(text, &[]);
    let result = plan.finalize(raw(&plan, r#""Alice""#)).unwrap();
    assert_eq!(result.result.grounding, ExtractionGrounding::SourceMembership);
    assert_eq!(result.result.output.numerics_profile, STRICT_INT4_PROFILE);
    assert_eq!(result.group_width, 127);
    let field = &result.result.source_fields[0];
    assert_eq!(field.occurrence, SourceOccurrence::Ambiguous); assert_eq!(field.spans.len(), 2);
    for span in &field.spans {
        assert_eq!(&text[span.byte_start..span.byte_end], "Alice");
        assert_eq!(text[..span.byte_start].chars().count(), span.scalar_start);
        assert_eq!(text[..span.byte_end].chars().count(), span.scalar_end);
    }
}

#[test]
fn auxiliary_context_cannot_authorize_a_source_citation() {
    let plan = source_plan("Alice", &["Bob metadata", "Is Carol present?"]);
    assert!(plan.finalize(raw(&plan, r#""Alice""#)).is_ok());
    for text in [r#""Bob""#, r#""Carol""#, r#""Mallory""#] { assert!(plan.finalize(raw(&plan, text)).is_err()); }
}

#[test]
fn independent_finalization_rejects_truncation_schema_control_and_work_mismatches() {
    let plan = plan(BOOLEAN, 127);
    for axis in 0..11 {
        let mut run = raw(&plan, "true");
        match axis {
            0 => run.output.numerics_profile = HF_BF16_EAGER_PROFILE.to_owned(),
            1 => { run.output.token_ids.pop(); }, 2 => run.output.token_ids[0] = 3,
            3 => run.output.json = "tru".to_owned(), 4 => run.output.json = "null".to_owned(),
            5 => run.model_work.attention_pairs += 1, 6 => run.model_work.projections.dot_products += 1,
            7 => run.output.projected_logits += 1, 8 => run.group_width = 128,
            9 => run.execution = "int8".to_owned(), _ => run.output.token_ids.clear(),
        }
        assert!(plan.finalize(run).is_err());
    }
}

#[test]
fn whole_result_bound_includes_group_work_and_all_source_evidence() {
    let mut plan = source_plan("Alice Alice", &[]);
    let result = plan.finalize(raw(&plan, r#""Alice""#)).unwrap();
    let bytes = canonjson::canonical_bytes(&result).unwrap().len() as u64;
    assert!(bytes > canonjson::canonical_bytes(&result.result).unwrap().len() as u64);
    plan.extraction.max_result_bytes = bytes;
    assert!(plan.finalize(raw(&plan, r#""Alice""#)).is_ok());
    plan.extraction.max_result_bytes -= 1;
    assert!(matches!(plan.finalize(raw(&plan, r#""Alice""#)),
        Err(Int4ExtractError::Extraction(ExtractError::OutputBudgetExceeded))));
}

#[test]
fn finalization_preserves_exact_decimal_and_integer_boundaries() {
    let number = plan(r#"{"type":"number"}"#, 127);
    let decimal = "1.2345678901234567890123456789012345678e37";
    assert_eq!(number.finalize(raw(&number, decimal)).unwrap().result.output.json, decimal);
    let integer = plan(r#"{"type":"integer"}"#, 32);
    for value in ["-9223372036854775808", "9007199254740993", "18446744073709551615"] {
        assert_eq!(integer.finalize(raw(&integer, value)).unwrap().result.output.json, value);
    }
    for value in ["-9223372036854775809", "18446744073709551616"] {
        assert!(integer.finalize(raw(&integer, value)).is_err());
    }
}

#[test]
fn canonical_output_does_not_export_private_identity_or_confidence() {
    let plan = plan(BOOLEAN, 127);
    let first = canonjson::canonical_string(&plan.finalize(raw(&plan, "true")).unwrap()).unwrap();
    assert_eq!(first, canonjson::canonical_string(&plan.finalize(raw(&plan, "true")).unwrap()).unwrap());
    for private in ["prompt_digest", "decision_policy_digest", "logical_model_digest", "confidence"] {
        assert!(!first.contains(private));
    }
}
