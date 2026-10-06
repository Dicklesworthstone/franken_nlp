//! Synthetic compiler/finalizer fixtures, not native-model receipts.
use super::*;
use crate::{tasks::{BuiltInTask, ir::{TaskBudget, PlanContext, PromptSegment}},
    tokenizer::specials::ArchivedControlRegistries,
    native_engine::{strict_int8::STRICT_INT8_EXECUTION, constrained_int8::sparse::Int8JsonSparseLimits},
    validation::grounded_fields::SourceOccurrence};
const SOURCE: &str = r#"{"type":"string","maxLength":64,"x-fnlp-source":"verbatim"}"#;
fn registry() -> ArchivedControlRegistries {
    ArchivedControlRegistries::from_archived_json(
        r#"{"schema_version":1,"registry":"TokenizerSpecialIds","entries":[{"id":0,"special":true,"surface":"<eos>"}]}"#,
        r#"{"schema_version":1,"registry":"TemplateControlIds","entries":[{"id":0,"special":true,"surface":"<eos>"},{"id":3,"special":false,"surface":"<think>"}]}"#).unwrap()
}
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"extraction cohort fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: d, template_digest: d, task_spec: "extract-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "fixture".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn plan(schema: &str, source: Option<&str>, selected: bool) -> Int8ExtractPlan {
    let registry = registry(); let id = identity();
    let encoder = SourceDocumentEncoder::pinned(registry.template_controls()).unwrap();
    let document = source.map(|text| encoder.encode(text, 1024, 1024).unwrap());
    let tokens = document.as_ref().map_or_else(|| vec![7], |d| d.token_ids().to_vec());
    let budget = TaskBudget { max_input_tokens: 1024, max_output_tokens: 8,
        max_output_bytes: 16384, max_grammar_states: 4096, max_kv_bytes: 1 << 30 };
    let mut post = vec![FinitePostcondition::JsonValid, FinitePostcondition::OutputWithinBudget];
    if document.is_some() { post.push(FinitePostcondition::SourceSpansVerified); }
    let ir = TaskIR::new(vec![PromptSegment::new(PromptSegmentKind::TaskInstruction, vec![1]),
        PromptSegment::new(PromptSegmentKind::Document, tokens),
        PromptSegment::new(PromptSegmentKind::AnswerScaffold, vec![2])], DecodeStrategy::ConstrainedJson,
        GrammarReference::json_schema(Sha256Digest::of_bytes(schema.as_bytes()),
            if document.is_some() { SOURCE_JSON_RUNTIME_VERSION } else { JSON_RUNTIME_VERSION }),
        None, post, budget, DependencyScope::ItemLocal).unwrap();
    let task = TaskPlan::new(BuiltInTask::Extract.spec(), &PlanContext::new(&id, budget).unwrap(), ir).unwrap();
    let options = JsonDecodeOptions { max_new_tokens: 8, eos_token_id: 0, excluded_token_ids: Default::default() };
    let plan = match document {
        Some(document) => Int8ExtractPlan::from_task_plan_with_source(&task, schema, options, CompileLimits::default(),
            registry.template_controls(), &document, SourceRuntimeLimits::default(), id).unwrap(),
        None => Int8ExtractPlan::from_task_plan(&task, schema, options, CompileLimits::default(), registry.template_controls(), id).unwrap(),
    };
    if selected { plan.with_selected_rows(Int8JsonSparseLimits { max_rows_per_step: 7 }).unwrap() } else { plan }
}
fn requests(plans: &[Int8ExtractPlan]) -> Vec<Int8ExtractCohortRequest<'_>> {
    plans.iter().map(|p| {
        let work = p.planned_work();
        Int8ExtractCohortRequest { prepared: p, admitted_identity: p.execution_identity(),
            budget: Int8JsonBudget { native: crate::native_engine::strict_int8::Int8RunBudget::exact(work),
                json: JsonWorkBudget { max_forward_positions: work.forward_positions, max_projected_logits: work.projected_logits,
                    max_kv_bytes: p.max_kv_bytes(), max_total_mask_node_visits: 1_000_000_000,
                    mask_limits: crate::grammar::mask::MaskWorkLimits::default() } } }
    }).collect()
}
fn raw(requests: &[Int8ExtractCohortRequest<'_>], json: &[&str]) -> Int8JsonCohortRun {
    let mut planned_work = Int8Work::default(); let mut model_work = Int8Work::default(); let mut group_steps = 0;
    let sequences = requests.iter().zip(json).map(|(r, &json)| {
        let work = Int8Work::for_sequence(0, r.prepared.prompt_tokens() + 1, 3).unwrap();
        planned_work = planned_work.checked_add(r.prepared.planned_work()).unwrap();
        model_work = model_work.checked_add(work).unwrap(); group_steps = group_steps.max(work.forward_positions);
        Int8JsonRun { schema_version: 1, execution: selected_head::INT8_SPARSE_JSON_EXECUTION.to_owned(), model_work: work,
            output: JsonDecodeOutput { schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.to_owned(),
                token_ids: vec![1,0], json: json.to_owned(), forward_positions: work.forward_positions,
                projected_logits: work.projected_logits, mask_node_visit_charge: 20 } }
    }).collect();
    Int8JsonCohortRun { schema_version: 1, execution: INT8_JSON_COHORT_EXECUTION.to_owned(), sequences,
        group_steps, planned_work, model_work }
}
#[test]
fn row_identity_controls_and_explicit_mode_are_checked_before_native_execution() {
    let plans = [plan(r#"{"type":"boolean"}"#, None, true)]; let requests = requests(&plans);
    let mut vocabulary = ExtractionVocabulary::pinned(registry().template_controls()).unwrap();
    check_request(&requests[0], &vocabulary).unwrap();
    let mut changed = plans[0].execution_identity().clone(); changed.prompt_digest = Sha256Digest::of_bytes(b"changed");
    let wrong = Int8ExtractCohortRequest { prepared: &plans[0], admitted_identity: &changed, budget: requests[0].budget };
    assert!(check_request(&wrong, &vocabulary).is_err());
    vocabulary.controls.insert(55); assert!(check_request(&requests[0], &vocabulary).is_err());
    let dense = [plan(r#"{"type":"boolean"}"#, None, false)];
    assert!(check_request(&self::requests(&dense)[0], &vocabulary).is_err());
}
#[test]
fn independent_source_occurrences_and_input_order_survive_cohort_finalization() {
    let plans = [plan(SOURCE, Some("é Alice Alice"), true), plan(SOURCE, Some("Bob"), true)];
    let requests = requests(&plans);
    let out = finish(&requests, raw(&requests, &["\"Alice\"", "\"Bob\""]), 1 << 20).unwrap();
    assert_eq!(out.sequences[0].result.output.json, "\"Alice\"");
    assert_eq!(out.sequences[1].result.output.json, "\"Bob\"");
    assert_eq!(out.sequences[0].result.source_fields[0].occurrence, SourceOccurrence::Ambiguous);
    assert_eq!(out.sequences[0].result.source_fields[0].spans.len(), 2);
    assert_eq!(out.sequences[0].result.source_fields[0].spans[0].byte_start, 3);
    assert_eq!(out.sequences[0].result.source_fields[0].spans[0].scalar_start, 2);
    let mut swapped = raw(&requests, &["\"Alice\"", "\"Bob\""]); swapped.sequences.swap(0,1);
    assert!(finish(&requests, swapped, 1 << 20).is_err());
}
#[test]
fn exact_decimal_strings_do_not_pass_through_a_float_value() {
    let plans = [plan(r#"{"type":"integer"}"#, None, true)]; let requests = requests(&plans);
    let number = "12345678901234567890123456789012345678";
    let out = finish(&requests, raw(&requests, &[number]), 1 << 20).unwrap();
    assert_eq!(out.sequences[0].result.output.json, number);
}
#[test]
fn all_envelope_versions_row_counts_and_work_axes_are_reconciled() {
    let plans = [plan(r#"{"type":"boolean"}"#, None, true)]; let requests = requests(&plans);
    for axis in 0..8 {
        let mut run = raw(&requests, &["true"]);
        match axis { 0 => run.schema_version = 2, 1 => run.execution = "other".to_owned(),
            2 => { run.sequences.clear(); }, 3 => run.planned_work.forward_positions += 1,
            4 => run.model_work.attention_pairs += 1, 5 => run.group_steps += 1,
            6 => run.sequences[0].output.token_ids.pop().map(|_| ()).unwrap(),
            _ => run.sequences[0].model_work.projected_logits += 1 }
        assert!(finish(&requests, run, 1 << 20).is_err());
    }
}
#[test]
fn a_late_invalid_source_row_refuses_the_whole_typed_cohort() {
    let plans = [plan(SOURCE, Some("Alice"), true), plan(SOURCE, Some("Bob"), true)]; let requests = requests(&plans);
    assert!(finish(&requests, raw(&requests, &["\"Alice\"", "\"Mallory\""]), 1 << 20).is_err());
}
#[test]
fn complete_outer_result_has_an_exact_bound_and_keeps_semantic_keys_unchanged() {
    let plans = [plan(r#"{"type":"boolean"}"#, None, true)]; let requests = requests(&plans);
    let before = canonjson::canonical_bytes(plans[0].execution_identity()).unwrap();
    let out = finish(&requests, raw(&requests, &["true"]), 1 << 20).unwrap();
    let cap = canonjson::canonical_bytes(&out).unwrap().len() as u64;
    assert!(finish(&requests, raw(&requests, &["true"]), cap).is_ok());
    assert!(finish(&requests, raw(&requests, &["true"]), cap - 1).is_err());
    assert_eq!(before, canonjson::canonical_bytes(plans[0].execution_identity()).unwrap());
}
