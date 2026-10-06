//! Pinned planning and synthetic semantic fixtures, not neural execution.
use super::*;
use crate::{grammar::runtime::JsonProgram,
    native_engine::{constrained::JsonDecodeOutput, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace},
    tokenizer::specials::ArchivedControlRegistries};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn planner() -> SourceTaskPlanner {
    let t = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = t.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let registry = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    SourceTaskPlanner::pinned(registry.template_controls(), t.eos_token_id().unwrap()).unwrap()
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 512, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn prepare(p: &SourceTaskPlanner, request: &SourceTaskRequest, selected: bool) -> PreparedInt8SourceTask {
    let d = Sha256Digest::of_bytes(b"source cohort fixture");
    let id = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(),
        task_spec: request.task().spec().identity(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    let plan = p.plan_int8_with_control(request, &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), &mut Continue).unwrap();
    if selected { plan.with_selected_rows(crate::native_engine::constrained_int8::sparse::Int8JsonSparseLimits {
        max_rows_per_step: 7 }).unwrap() } else { plan }
}
fn fixtures() -> Vec<(SourceTaskRequest, &'static str, &'static str)> {
    vec![
        (SourceTaskRequest::Ner { document: "é上海 and 上海".to_owned(), options: NerOptions::default(), budget: budget() },
            "é上海 and 上海", r#"[{"text":"上海","type":"location"}]"#),
        (SourceTaskRequest::Keyphrases { document: "Rust compiler Rust".to_owned(), options: KeyphraseOptions::default(), budget: budget() },
            "Rust compiler Rust", r#"["compiler","Rust","Rust"]"#),
        (SourceTaskRequest::Summarize { document: "Alice".to_owned(), options: SummaryOptions::default(), budget: budget() },
            "Alice", r#"[{"citations":["Alice"],"text":"An unverified assertion."}]"#),
        (SourceTaskRequest::Answer { question: "Where?".to_owned(), passages: vec![
            AnswerPassage { id: "one".to_owned(), text: "é上海".to_owned() },
            AnswerPassage { id: "two".to_owned(), text: "上海".to_owned() }], options: AnswerOptions::default(), budget: budget() },
            "é上海\n\n上海", r#"{"answer":"A location.","answerable":true,"citations":["上海"]}"#),
    ]
}
fn schema(r: &SourceTaskRequest) -> String {
    match r { SourceTaskRequest::Ner { options, .. } => options.schema_source().unwrap(),
        SourceTaskRequest::Keyphrases { options, .. } => options.schema_source().unwrap(),
        SourceTaskRequest::Summarize { options, .. } => options.schema_source().unwrap(),
        SourceTaskRequest::Answer { options, .. } => options.schema_source().unwrap() }
}
fn raw_row(p: &PreparedInt8SourceTask, r: &SourceTaskRequest, source: &str, json: &str) -> Int8ExtractRun {
    let grammar = JsonProgram::compile_with_source(&schema(r), source, CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
    let work = Int8Work::for_sequence(0, p.prompt_tokens() + 1, 3).unwrap();
    Int8ExtractRun { schema_version: 1, execution: p.extraction.execution_version().to_owned(), model_work: work,
        result: ExtractResult { schema_version: 2, task_spec_version: r.task().spec().identity(),
            score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
            source_fields: grammar.source_fields(json).unwrap(), output: JsonDecodeOutput {
                schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.to_owned(), token_ids: vec![1,0],
                json: json.to_owned(), forward_positions: work.forward_positions,
                projected_logits: work.projected_logits, mask_node_visit_charge: 20 } } }
}
fn requests(plans: &[PreparedInt8SourceTask]) -> Vec<Int8SourceCohortRequest<'_>> {
    plans.iter().map(|plan| {
        let w = plan.planned_work();
        Int8SourceCohortRequest { prepared: plan, admitted_identity: plan.execution_identity(),
            budget: Int8JsonBudget { native: crate::native_engine::strict_int8::Int8RunBudget::exact(w),
                json: crate::native_engine::constrained::JsonWorkBudget { max_forward_positions: w.forward_positions,
                    max_projected_logits: w.projected_logits, max_kv_bytes: plan.task_budget().max_kv_bytes,
                    max_total_mask_node_visits: 1_000_000_000, mask_limits: crate::grammar::mask::MaskWorkLimits::default() } } }
    }).collect()
}
fn raw(plans: &[PreparedInt8SourceTask], fixtures: &[(SourceTaskRequest, &str, &str)]) -> Int8ExtractCohortRun {
    let mut planned_work = Int8Work::default(); let mut model_work = Int8Work::default(); let mut group_steps = 0;
    let sequences = plans.iter().zip(fixtures).map(|(p, (r, source, json))| {
        let out = raw_row(p, r, source, json);
        planned_work = planned_work.checked_add(p.planned_work()).unwrap();
        model_work = model_work.checked_add(out.model_work).unwrap(); group_steps = group_steps.max(out.model_work.forward_positions);
        out
    }).collect();
    Int8ExtractCohortRun { schema_version: 1, execution: INT8_EXTRACT_COHORT_EXECUTION.to_owned(),
        sequences, group_steps, planned_work, model_work }
}
#[test]
fn all_four_semantic_finalizers_match_their_independent_results_in_input_order() {
    let planner = planner(); let fixtures = fixtures();
    let plans: Vec<_> = fixtures.iter().map(|(r, _, _)| prepare(&planner, r, true)).collect();
    let out = finish(&requests(&plans), raw(&plans, &fixtures), 8 << 20).unwrap();
    assert_eq!(out.sequences.len(), 4);
    for ((p, (r, source, json)), grouped) in plans.iter().zip(&fixtures).zip(&out.sequences) {
        let single = p.finish(raw_row(p, r, source, json)).unwrap();
        assert_eq!(canonjson::canonical_bytes(&single.result).unwrap(), canonjson::canonical_bytes(&grouped.result).unwrap());
        assert_eq!(single.execution, grouped.execution); p.verify_identity(p.execution_identity()).unwrap();
    }
}
#[test]
fn semantic_failure_after_valid_rows_aborts_the_complete_cohort() {
    let planner = planner(); let mut fixtures = fixtures();
    let plans: Vec<_> = fixtures.iter().map(|(r, _, _)| prepare(&planner, r, true)).collect();
    fixtures[2].2 = r#"[{"citations":[],"text":"A claim."}]"#;
    assert!(finish(&requests(&plans), raw(&plans, &fixtures), 8 << 20).is_err());
}
#[test]
fn cross_task_swaps_and_altered_group_work_are_refused() {
    let planner = planner(); let fixtures = fixtures();
    let plans: Vec<_> = fixtures.iter().map(|(r, _, _)| prepare(&planner, r, true)).collect();
    for axis in 0..5 {
        let mut raw = raw(&plans, &fixtures);
        match axis { 0 => raw.sequences.swap(0,1), 1 => raw.planned_work.projected_logits += 1,
            2 => raw.model_work.attention_pairs += 1, 3 => raw.group_steps += 1, _ => raw.execution = "other".to_owned() }
        assert!(finish(&requests(&plans), raw, 8 << 20).is_err());
    }
}
#[test]
fn a_dense_plan_cannot_silently_enter_a_selected_cohort() {
    let planner = planner(); let fixtures = fixtures();
    let plans: Vec<_> = fixtures.iter().map(|(r, _, _)| prepare(&planner, r, false)).collect();
    assert!(finish(&requests(&plans), raw(&plans, &fixtures), 8 << 20).is_err());
}
#[test]
fn all_source_evidence_and_group_metadata_count_toward_the_outer_bound() {
    let planner = planner(); let fixtures = fixtures();
    let plans: Vec<_> = fixtures.iter().map(|(r, _, _)| prepare(&planner, r, true)).collect();
    let out = finish(&requests(&plans), raw(&plans, &fixtures), 8 << 20).unwrap();
    let cap = canonjson::canonical_bytes(&out).unwrap().len() as u64;
    assert!(finish(&requests(&plans), raw(&plans, &fixtures), cap).is_ok());
    assert!(finish(&requests(&plans), raw(&plans, &fixtures), cap - 1).is_err());
}
