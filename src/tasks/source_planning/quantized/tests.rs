//! Synthetic planning/semantic fixtures, not neural inference or quality evidence.
use super::*;
use crate::{
    grammar::runtime::JsonProgram,
    native_engine::{constrained::JsonDecodeOutput, strict_int8::STRICT_INT8_EXECUTION},
    tasks::{answer::{AnswerCalibration, AnswerStatus}, extract::ExtractionGrounding,
        ir::ScoreSpace, summarize::{CitationGuarantee, SummarySemanticSupport}},
    tokenizer::specials::ArchivedControlRegistries,
    validation::{SourceSpan, validate_source_span, grounded_fields::SourceOccurrence},
};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
struct StopAfter { calls: usize, at: usize }
impl DecodeStepControl for StopAfter {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.calls >= self.at).then_some(DecodeCancellationKind::Deadline)
    }
}
fn planner() -> SourceTaskPlanner {
    let t = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = t.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let eos = t.eos_token_id().unwrap();
    let specials: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let registry = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":specials}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    SourceTaskPlanner::pinned(registry.template_controls(), eos).unwrap()
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 512, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn identity(p: &SourceTaskPlanner, request: &SourceTaskRequest) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(),
        task_spec: request.task().spec().identity(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn prepare(p: &SourceTaskPlanner, r: &SourceTaskRequest) -> PreparedInt8SourceTask {
    p.plan_int8_with_control(r, &PlanContext::new(&identity(p, r), budget()).unwrap(),
        SourcePlanningLimits::default(), &mut Continue).unwrap()
}
fn ner(source: &str) -> SourceTaskRequest {
    SourceTaskRequest::Ner { document: source.to_owned(), options: NerOptions::default(), budget: budget() }
}
fn keyphrases(source: &str) -> SourceTaskRequest {
    SourceTaskRequest::Keyphrases { document: source.to_owned(), options: KeyphraseOptions::default(), budget: budget() }
}
fn summary(source: &str) -> SourceTaskRequest {
    SourceTaskRequest::Summarize { document: source.to_owned(), options: SummaryOptions::default(), budget: budget() }
}
fn answer(values: &[(&str, &str)]) -> SourceTaskRequest {
    SourceTaskRequest::Answer { question: "Who is named?".to_owned(),
        passages: values.iter().map(|&(id, text)| AnswerPassage { id: id.to_owned(), text: text.to_owned() }).collect(),
        options: AnswerOptions::default(), budget: budget() }
}
fn schema(r: &SourceTaskRequest) -> String {
    match r { SourceTaskRequest::Ner { options, .. } => options.schema_source().unwrap(),
        SourceTaskRequest::Keyphrases { options, .. } => options.schema_source().unwrap(),
        SourceTaskRequest::Summarize { options, .. } => options.schema_source().unwrap(),
        SourceTaskRequest::Answer { options, .. } => options.schema_source().unwrap() }
}
// Independent source membership is real; numbers and token IDs below are
// deliberately synthetic. Only the private semantic-finalization seam sees them.
fn synthetic(p: &PreparedInt8SourceTask, r: &SourceTaskRequest, source: &str, json: &str) -> Int8ExtractRun {
    let grammar = JsonProgram::compile_with_source(&schema(r), source, CompileLimits::default(),
        SourceRuntimeLimits::default()).unwrap();
    let work = constrained_int8::planned_work(p.prompt_tokens(), 2).unwrap();
    Int8ExtractRun { schema_version: 1, execution: INT8_EXTRACT_VERSION.to_owned(), model_work: work,
        result: ExtractResult { schema_version: 2, task_spec_version: r.task().spec().identity(),
            score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
            source_fields: grammar.source_fields(json).unwrap(), output: JsonDecodeOutput {
                schema_version: 1, numerics_profile: STRICT_INT8_PROFILE.to_owned(), token_ids: vec![1, 0],
                json: json.to_owned(), forward_positions: work.forward_positions,
                projected_logits: work.projected_logits, mask_node_visit_charge: 20,
            } } }
}

#[test]
fn all_four_tasks_compile_exact_int8_work_without_relabeling_eager_plans() {
    let p = planner();
    for r in [ner("Alice"), keyphrases("Alice"), summary("Alice"), answer(&[("p", "Alice")])] {
        let id = identity(&p, &r); let before = canonjson::canonical_bytes(&id).unwrap();
        let context = PlanContext::new(&id, budget()).unwrap();
        let prepared = p.plan_int8_with_control(&r, &context, SourcePlanningLimits::default(), &mut Continue).unwrap();
        assert_eq!(prepared.execution_identity().task_spec, r.task().spec().identity());
        assert_eq!(prepared.planned_work(), constrained_int8::planned_work(prepared.prompt_tokens(), 512).unwrap());
        prepared.verify_identity(prepared.execution_identity()).unwrap();
        assert!(prepared.verify_identity(&id).is_err());
        assert!(p.plan(&r, &context, SourcePlanningLimits::default()).is_err());
        assert_eq!(before, canonjson::canonical_bytes(&id).unwrap());
        let mut eager = id.clone(); eager.numerics_profile = NumericsProfile::HfBf16Eager;
        assert!(p.plan(&r, &PlanContext::new(&eager, budget()).unwrap(), SourcePlanningLimits::default()).is_ok());
    }
}

#[test]
fn profile_backend_kv_thinking_and_tools_are_not_repaired() {
    let p = planner(); let r = ner("Alice");
    for axis in 0..6 {
        let mut id = identity(&p, &r);
        match axis { 0 => id.numerics_profile = NumericsProfile::HfBf16Eager,
            1 => id.numerics_profile = NumericsProfile::StrictQuantized { version: 2 },
            2 => id.backend_semantic_version = "other".to_owned(), 3 => id.kv_dtype = "int8".to_owned(),
            4 => id.thinking_mode = ThinkingMode::Enabled, _ => id.tool_mode = ToolMode::Json }
        let context = PlanContext::new(&id, budget()).unwrap();
        assert!(p.plan_int8_with_control(&r, &context, SourcePlanningLimits::default(), &mut Continue).is_err());
    }
}

#[test]
fn all_representable_identity_mutations_are_refused() {
    let p = planner(); let r = ner("Alice"); let prepared = prepare(&p, &r);
    let value = serde_json::to_value(prepared.execution_identity()).unwrap();
    for key in value.as_object().unwrap().keys() {
        let mut changed = value.clone();
        changed[key] = match &value[key] {
            serde_json::Value::String(s) if s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()) =>
                serde_json::json!(Sha256Digest::of_bytes(b"different").to_hex()),
            serde_json::Value::String(_) | serde_json::Value::Null => serde_json::json!("different"),
            serde_json::Value::Number(_) => serde_json::json!(99), _ => serde_json::Value::Null,
        };
        if let Ok(changed) = serde_json::from_value::<ExecutionIdentity>(changed) {
            assert!(prepared.verify_identity(&changed).is_err(), "accepted {key}");
        }
    }
}

#[test]
fn source_options_and_passage_partition_are_bound() {
    let p = planner();
    let a = prepare(&p, &ner("Alice")); let b = prepare(&p, &ner("Bob"));
    assert_ne!(a.execution_identity().prompt_digest, b.execution_identity().prompt_digest);
    let mut r = ner("Alice");
    if let SourceTaskRequest::Ner { options, .. } = &mut r { options.max_entities = 2; }
    let b = prepare(&p, &r);
    assert_ne!(a.execution_identity().schema_digest, b.execution_identity().schema_digest);
    let a = prepare(&p, &answer(&[("a", "Alice"), ("b", "Bob")]));
    let b = prepare(&p, &answer(&[("a", "Alice\n\nBob")]));
    assert_ne!(a.execution_identity().prompt_digest, b.execution_identity().prompt_digest);
    let c = prepare(&p, &answer(&[("renamed", "Alice"), ("b", "Bob")]));
    assert_ne!(a.execution_identity().prompt_digest, c.execution_identity().prompt_digest);
}

#[test]
fn context_input_and_host_task_ceilings_are_enforced_before_execution() {
    let p = planner(); let r = ner("Alice"); let id = identity(&p, &r);
    let context = PlanContext::new(&id, budget()).unwrap();
    for limits in [SourcePlanningLimits { max_input_bytes: 4, ..SourcePlanningLimits::default() },
        SourcePlanningLimits { max_context_tokens: 1, ..SourcePlanningLimits::default() }] {
        assert!(p.plan_int8_with_control(&r, &context, limits, &mut Continue).is_err());
    }
    let mut ceiling = budget(); ceiling.max_output_tokens -= 1;
    assert!(p.plan_int8_with_control(&r, &PlanContext::new(&id, ceiling).unwrap(),
        SourcePlanningLimits::default(), &mut Continue).is_err());
}

#[test]
fn cancellation_before_and_after_compilation_remains_typed() {
    let p = planner(); let r = ner("Alice"); let id = identity(&p, &r);
    for at in [1, 2, 3, 4] {
        let error = p.plan_int8_with_control(&r, &PlanContext::new(&id, budget()).unwrap(),
            SourcePlanningLimits::default(), &mut StopAfter { calls: 0, at }).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    }
}

#[test]
fn ner_preserves_all_unicode_occurrences_and_the_quantized_profile() {
    let p = planner(); let source = "é上海 and 上海"; let r = ner(source); let prepared = prepare(&p, &r);
    let run = prepared.finish(synthetic(&prepared, &r, source, r#"[{"text":"上海","type":"location"}]"#)).unwrap();
    let SourceTaskResult::Ner(result) = run.result else { panic!("wrong task") };
    assert_eq!(result.entities[0].occurrence, SourceOccurrence::Ambiguous);
    assert_eq!(result.entities[0].spans.len(), 2);
    assert_eq!(result.numerics_profile, STRICT_INT8_PROFILE);
    for s in &result.entities[0].spans {
        validate_source_span(source, "上海", SourceSpan::new(s.byte_start, s.byte_end, s.scalar_start, s.scalar_end)).unwrap();
    }
}

#[test]
fn keyphrase_model_order_dedup_and_ambiguity_survive_native_composition() {
    let p = planner(); let source = "Rust compiler Rust"; let r = keyphrases(source); let prepared = prepare(&p, &r);
    let run = prepared.finish(synthetic(&prepared, &r, source, r#"["compiler","Rust","Rust"]"#)).unwrap();
    let SourceTaskResult::Keyphrases(result) = run.result else { panic!("wrong task") };
    assert_eq!(result.phrases.iter().map(|p| (p.rank, p.text.as_str())).collect::<Vec<_>>(), vec![(1, "compiler"), (2, "Rust")]);
    assert_eq!(result.phrases[1].occurrence, SourceOccurrence::Ambiguous);
    assert_eq!(result.phrases[1].spans.len(), 2);
}

#[test]
fn summaries_require_citations_without_claiming_entailment() {
    let p = planner(); let r = summary("Alice"); let prepared = prepare(&p, &r);
    let json = r#"[{"citations":["Alice"],"text":"An unverified assertion."}]"#;
    let run = prepared.finish(synthetic(&prepared, &r, "Alice", json)).unwrap();
    let SourceTaskResult::Summarize(result) = run.result else { panic!("wrong task") };
    assert_eq!(result.citation_guarantee, CitationGuarantee::StructuralSourceMembership);
    assert_eq!(result.semantic_support, SummarySemanticSupport::NotAssessed);
    assert!(prepared.finish(synthetic(&prepared, &r, "Alice", r#"[{"citations":[],"text":"A claim."}]"#)).is_err());
}

#[test]
fn passage_qa_retains_original_offsets_and_all_ambiguous_passages() {
    let p = planner(); let r = answer(&[("one", "é上海"), ("two", "上海")]); let prepared = prepare(&p, &r);
    let json = r#"{"answer":"A location.","answerable":true,"citations":["上海"]}"#;
    let run = prepared.finish(synthetic(&prepared, &r, "é上海\n\n上海", json)).unwrap();
    let SourceTaskResult::Answer(result) = run.result else { panic!("wrong task") };
    assert_eq!(result.status, AnswerStatus::Answered);
    assert_eq!(result.calibration, AnswerCalibration::Uncalibrated);
    assert_eq!(result.citations[0].occurrence, SourceOccurrence::Ambiguous);
    let occurrences = &result.citations[0].spans;
    assert_eq!(occurrences.len(), 2);
    assert_eq!((occurrences[0].passage_id.as_str(), occurrences[0].span.byte_start, occurrences[0].span.scalar_start), ("one", 2, 1));
    assert_eq!((occurrences[1].passage_id.as_str(), occurrences[1].span.byte_start), ("two", 0));
}

#[test]
fn a_quote_crossing_a_passage_join_cannot_become_answer_evidence() {
    let p = planner(); let r = answer(&[("a", "ab"), ("b", "cd")]); let prepared = prepare(&p, &r);
    let json = r#"{"answer":"A claim.","answerable":true,"citations":["b\n\nc"]}"#;
    assert!(prepared.finish(synthetic(&prepared, &r, "ab\n\ncd", json)).is_err());
    let r = answer(&[("real", "b\n\nc"), ("a", "b"), ("b", "c")]); let prepared = prepare(&p, &r);
    let run = prepared.finish(synthetic(&prepared, &r, "b\n\nc\n\nb\n\nc", json)).unwrap();
    let SourceTaskResult::Answer(result) = run.result else { panic!("wrong task") };
    assert_eq!(result.citations[0].spans.len(), 1);
    assert_eq!(result.citations[0].spans[0].passage_id, "real");
}

#[test]
fn qa_abstention_is_success_but_inconsistent_answerability_is_not() {
    let p = planner(); let r = answer(&[("p", "Alice")]); let prepared = prepare(&p, &r);
    let run = prepared.finish(synthetic(&prepared, &r, "Alice", r#"{"answer":"","answerable":false,"citations":[]}"#)).unwrap();
    let SourceTaskResult::Answer(result) = run.result else { panic!("wrong task") };
    assert_eq!(result.status, AnswerStatus::Abstained); assert!(result.answer.is_none());
    for json in [r#"{"answer":"Alice","answerable":true,"citations":[]}"#,
        r#"{"answer":"Alice","answerable":false,"citations":[]}"#] {
        assert!(prepared.finish(synthetic(&prepared, &r, "Alice", json)).is_err());
    }
}

#[test]
fn every_source_task_refuses_incomplete_or_corrupted_evidence() {
    let p = planner();
    for (r, json) in [(ner("Alice"), r#"[{"text":"Alice","type":"person"}]"#),
        (keyphrases("Alice"), r#"["Alice"]"#),
        (summary("Alice"), r#"[{"citations":["Alice"],"text":"A name."}]"#),
        (answer(&[("p", "Alice")]), r#"{"answer":"Alice","answerable":true,"citations":["Alice"]}"#)] {
        let prepared = prepare(&p, &r);
        for axis in 0..4 {
            let mut raw = synthetic(&prepared, &r, "Alice", json);
            match axis { 0 => raw.result.source_fields.clear(),
                1 => raw.result.source_fields[0].json_pointer = "/other".to_owned(),
                2 => raw.result.source_fields.push(raw.result.source_fields[0].clone()),
                _ => raw.result.source_fields[0].spans[0].scalar_end += 1 }
            assert!(prepared.finish(raw).is_err());
        }
    }
}

#[test]
fn task_profile_and_wrapper_versions_cannot_be_relabeled() {
    let p = planner(); let r = ner("Alice"); let prepared = prepare(&p, &r);
    for axis in 0..5 {
        let mut raw = synthetic(&prepared, &r, "Alice", "[]");
        match axis { 0 => raw.schema_version += 1, 1 => raw.execution = "other".to_owned(),
            2 => raw.result.output.numerics_profile = "hf-bf16-eager".to_owned(),
            3 => raw.result.task_spec_version = "summarize-v1".to_owned(), _ => raw.result.output.schema_version += 1 }
        assert!(prepared.finish(raw).is_err());
    }
}

#[test]
fn outer_result_budget_includes_native_work_and_task_envelope() {
    let p = planner(); let mut r = ner("Alice"); let prepared = prepare(&p, &r);
    let run = prepared.finish(synthetic(&prepared, &r, "Alice", r#"[{"text":"Alice","type":"person"}]"#)).unwrap();
    let exact = canonjson::canonical_bytes(&run).unwrap().len() as u64;
    for (cap, success) in [(exact, true), (exact - 1, false)] {
        if let SourceTaskRequest::Ner { budget, .. } = &mut r { budget.max_output_bytes = cap; }
        let prepared = prepare(&p, &r);
        assert_eq!(prepared.finish(synthetic(&prepared, &r, "Alice", r#"[{"text":"Alice","type":"person"}]"#)).is_ok(), success);
    }
    let text = canonjson::canonical_string(&run).unwrap();
    for absent in ["prompt_digest", "taskir_digest", "confidence", "logical_model_digest"] { assert!(!text.contains(absent)); }
}

#[test]
fn empty_source_lists_are_valid_complete_results() {
    let p = planner();
    for r in [ner("plain"), keyphrases("plain"), summary("plain")] {
        let prepared = prepare(&p, &r);
        let run = prepared.finish(synthetic(&prepared, &r, "plain", "[]")).unwrap();
        assert_eq!(run.execution, INT8_SOURCE_EXECUTION);
    }
}

#[test]
fn native_cancellation_keeps_its_cause_and_default_diagnostics_redact_content() {
    use crate::native_engine::strict_int8::StrictInt8Error;
    let error: Int8SourceError = Int8JsonError::Native(StrictInt8Error::Cancelled(DecodeCancellationKind::Deadline)).into();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    let error = Int8SourceError::Planning(SourcePlanningError::Contract("PRIVATE_SOURCE_SENTINEL"));
    assert!(!format!("{error:?} {error}").contains("PRIVATE_SOURCE_SENTINEL"));
    assert!(error.source().is_some());
}
