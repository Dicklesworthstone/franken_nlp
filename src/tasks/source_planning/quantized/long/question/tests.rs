//! Pinned planning plus PRIVATE scripted corruption fixtures. These are not
//! full-model inference, QA quality, calibrated abstention or throughput evidence.
use super::*;
use std::{cell::Cell, collections::VecDeque, rc::Rc};
use crate::{native_engine::strict_int8::STRICT_INT8_EXECUTION,
    tasks::answer::PassageOccurrence, tokenizer::pinned_controls};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn planner() -> SourceTaskPlanner {
    let controls = pinned_controls::pinned().unwrap();
    let eos = controls.template_controls().entries().iter().find(|e| e.special && e.surface == IM_END).unwrap().id;
    SourceTaskPlanner::pinned(controls.template_controls(), eos).unwrap()
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn question() -> SourceQuestion {
    SourceQuestion { question: "Who is named? private-question-token".to_owned(), options: AnswerOptions::default(),
        verification: GroundingBudget::default() }
}
fn identity(p: &SourceTaskPlanner) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"private question-map test fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(),
        task_spec: ANSWER_TASK_VERSION.to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn limits() -> Int8SourceMapLimits {
    let mut work = Int8Work::default();
    work.forward_positions = u64::MAX; work.projected_logits = u64::MAX; work.attention_pairs = u64::MAX;
    work.projections.dot_products = u64::MAX; work.projections.multiply_accumulates = u64::MAX;
    Int8SourceMapLimits { chunks: ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 6, max_chunk_tokens: 6,
            context_tokens: 8192, reserved_tokens: 64, max_chunks: 256, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { map_batch_chunks: 2, reduce_fan_in: 2,
            max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
        max_model_work: work, mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 256_000 }
}
fn prepare<'s>(p: &SourceTaskPlanner, source: &'s str, q: &SourceQuestion, l: Int8SourceMapLimits) -> PreparedInt8Question<'s> {
    let id = identity(p);
    p.plan_int8_question_with_control(source, q, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), l, &mut Continue).unwrap()
}
fn raw(chunk: &SourceChunk<'_>) -> AnswerResult {
    let quote = chunk.text().chars().find(|c| !c.is_whitespace()).unwrap().to_string();
    let spans = scan_occurrences(chunk.text(), &quote, &mut GroundingBudget::default()).unwrap();
    let occurrence = if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous };
    AnswerResult { schema_version: 1, task_spec_version: ANSWER_TASK_VERSION.to_owned(), numerics_profile: STRICT_INT8_PROFILE.to_owned(),
        status: AnswerStatus::Answered, answerable: true, answer: Some("same answer".to_owned()),
        citations: vec![PassageCitation { quote, occurrence, spans: spans.into_iter()
            .map(|span| PassageOccurrence { passage_id: PASSAGE_ID.to_owned(), span }).collect() }],
        calibration: AnswerCalibration::Uncalibrated, citation_guarantee: CitationGuarantee::StructuralSourceMembership,
        semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
        untrusted_fields: ["answer".to_owned(), "citations".to_owned()], generated_token_ids: vec![1, 0],
        forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 20 }
}
struct Script {
    outputs: VecDeque<AnswerResult>, calls: Rc<Cell<usize>>, checkpoints: Rc<Cell<usize>>,
    bad_axis: Option<usize>, fail_at: Option<usize>, cancel_at: Option<usize>,
}
fn script(p: &PreparedInt8Question<'_>) -> Script {
    Script { outputs: p.chunks.chunks().iter().filter(|c| has_text(c.text())).map(raw).collect(),
        calls: Rc::new(Cell::new(0)), checkpoints: Rc::new(Cell::new(0)), bad_axis: None, fail_at: None, cancel_at: None }
}
impl SourceDriver for Script {
    fn checkpoint(&mut self) -> Result<(), Int8SourceError> {
        let n = self.checkpoints.get() + 1; self.checkpoints.set(n);
        if self.cancel_at == Some(n) { return Err(Int8SourceError::Cancelled(DecodeCancellationKind::Deadline)); }
        Ok(())
    }
    fn run(&mut self, plan: &PreparedInt8SourceTask, _: &ExecutionIdentity) -> Result<Int8SourceTaskRun, Int8SourceError> {
        let n = self.calls.get() + 1; self.calls.set(n);
        if self.fail_at == Some(n) { return Err(Int8SourceError::Cancelled(DecodeCancellationKind::Deadline)); }
        let mut result = self.outputs.pop_front().ok_or(Int8SourceError::InvalidResult)?;
        let work = constrained_int8::planned_work(plan.prompt_tokens(), result.generated_token_ids.len()).unwrap();
        result.forward_positions = work.forward_positions; result.projected_logits = work.projected_logits;
        let mut model_work = work;
        if let Some(axis) = self.bad_axis {
            match axis { 0 => model_work.forward_positions += 1, 1 => model_work.projected_logits += 1,
                2 => model_work.attention_pairs += 1, 3 => model_work.projections.dot_products += 1,
                _ => model_work.projections.multiply_accumulates += 1 }
        }
        Ok(Int8SourceTaskRun { schema_version: 1, execution: INT8_SOURCE_EXECUTION.to_owned(),
            result: SourceTaskResult::Answer(result), model_work })
    }
}
fn run(p: PreparedInt8Question<'_>, driver: Script) -> Result<Int8QuestionRun, Int8SourceMapError> {
    let admitted: Vec<_> = p.execution_identities().cloned().collect();
    p.execute_with_driver(&admitted, driver)
}

#[test]
fn exact_preflight_matches_all_actual_plans_including_question_manifest_and_blank_ranges() {
    let p = planner(); let q = question(); let id = identity(&p);
    for source in ["éAéA      éAéA", "Alice <tool_call> 上海\r\n Bob"] {
        let prepared = prepare(&p, source, &q, limits());
        let expected = p.preflight_int8_question_with_control(source, &q, budget(), &PlanContext::new(&id, budget()).unwrap(),
            SourcePlanningLimits::default(), limits(), &mut Continue).unwrap();
        assert_eq!(expected, prepared.preflight_metadata());
        assert_eq!(prepared.chunks.chunks().iter().map(|c| c.text()).collect::<String>(), source);
        assert_eq!(expected.source_span().byte_end, source.len());
        assert_eq!(expected.source_span().scalar_end, source.chars().count());
        for plan in &prepared.plans {
            assert_eq!(plan.execution_identity().task_spec, ANSWER_TASK_VERSION);
            assert!(plan.prompt_tokens() + budget().max_output_tokens as usize <= limits().chunks.context_tokens);
        }
    }
}
#[test]
fn changing_only_the_question_changes_every_native_prompt_identity() {
    let p = planner(); let a = question(); let mut b = question(); b.question = "What is located here?".to_owned();
    let pa = prepare(&p, "Alice Bob Carol Dave", &a, limits()); let pb = prepare(&p, "Alice Bob Carol Dave", &b, limits());
    assert_eq!(pa.plans.len(), pb.plans.len());
    for (a, b) in pa.execution_identities().zip(pb.execution_identities()) {
        assert_ne!(a.prompt_digest, b.prompt_digest); assert_ne!(a.taskir_digest, b.taskir_digest);
        assert_eq!(a.logical_model_digest, b.logical_model_digest);
    }
}
#[test]
fn unicode_citations_lift_all_occurrences_and_blank_chunks_make_no_model_call() {
    let p = planner(); let source = "éAéA      éAéA"; let prepared = prepare(&p, source, &question(), limits());
    let expected = prepared.preflight_metadata(); let driver = script(&prepared); let calls = Rc::clone(&driver.calls);
    let result = run(prepared, driver).unwrap(); expected.verify_completed(&result).unwrap();
    assert_eq!(calls.get(), 2); assert_eq!(result.native_chunks, 2); assert_eq!(result.whitespace_chunks, 1);
    assert_eq!(result.outcome, QuestionOutcome::OneAnswerText); assert_eq!(result.distinct_answer_texts, 1);
    let chunks: Vec<_> = result.mapped.root().value().chunks().collect();
    assert_eq!(chunks[1].status, QuestionChunkStatus::WhitespaceOnly);
    assert_eq!(chunks[1].model_work, Int8Work::default()); assert_eq!(chunks[1].mask_node_visit_charge, 0);
    assert!(chunks[1].answer.is_none()); assert!(chunks[1].citations.is_empty());
    assert_eq!(chunks[2].citations[0].spans[0].byte_start, 12);
    assert_eq!(chunks[2].citations[0].spans[0].scalar_start, 10);
    for chunk in chunks { for citation in &chunk.citations { for span in &citation.spans {
        assert_eq!(&source[span.byte_start..span.byte_end], citation.quote);
        assert_eq!(source[..span.byte_start].chars().count(), span.scalar_start);
    } } }
}
#[test]
fn distinct_answers_are_retained_without_majority_voting_or_a_fabricated_global_answer() {
    let p = planner(); let prepared = prepare(&p, "éAéAéAéAéAéA", &question(), limits());
    let mut driver = script(&prepared); driver.outputs.back_mut().unwrap().answer = Some("different answer".to_owned());
    let result = run(prepared, driver).unwrap();
    assert_eq!(result.outcome, QuestionOutcome::MultipleAnswerTexts); assert_eq!(result.distinct_answer_texts, 2);
    assert_eq!(result.answered_chunks, 3);
    assert_eq!(result.mapped.root().value().chunks().last().unwrap().answer.as_deref(), Some("different answer"));
}
#[test]
fn all_native_abstentions_are_uncalibrated_and_not_confused_with_skipped_whitespace() {
    let p = planner(); let prepared = prepare(&p, "éAéA      éAéA", &question(), limits()); let mut driver = script(&prepared);
    for raw in &mut driver.outputs { raw.status = AnswerStatus::Abstained; raw.answerable = false; raw.answer = None; raw.citations.clear(); }
    let result = run(prepared, driver).unwrap();
    assert_eq!(result.outcome, QuestionOutcome::NoAnswerProposed);
    assert_eq!(result.abstained_chunks, 2); assert_eq!(result.whitespace_chunks, 1);
    assert_eq!(result.calibration, AnswerCalibration::Uncalibrated);
    assert_eq!(result.semantic_support, SummarySemanticSupport::NotAssessed);
}
#[test]
fn all_last_passage_citations_are_independently_rechecked() {
    let p = planner();
    for axis in 0..5 {
        let prepared = prepare(&p, "éAéAéAéA", &question(), limits()); let mut driver = script(&prepared);
        let citation = &mut driver.outputs.back_mut().unwrap().citations[0];
        match axis { 0 => { citation.spans.pop(); }, 1 => citation.spans[0].span.scalar_end += 1,
            2 => citation.spans[0].passage_id = "question".to_owned(), 3 => citation.occurrence = SourceOccurrence::Anchored,
            _ => citation.quote = "absent".to_owned() }
        let calls = Rc::clone(&driver.calls); assert!(run(prepared, driver).is_err()); assert_eq!(calls.get(), 2);
    }
}
#[test]
fn final_identity_and_each_native_receipt_axis_refuse_before_publication() {
    let p = planner(); let prepared = prepare(&p, "éAéAéAéA", &question(), limits()); let driver = script(&prepared);
    let calls = Rc::clone(&driver.calls); let mut admitted: Vec<_> = prepared.execution_identities().cloned().collect();
    admitted.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"wrong final question");
    assert!(prepared.execute_with_driver(&admitted, driver).is_err()); assert_eq!(calls.get(), 0);
    for axis in 0..5 {
        let prepared = prepare(&p, "éAéA", &question(), limits()); let mut driver = script(&prepared); driver.bad_axis = Some(axis);
        assert!(run(prepared, driver).is_err());
    }
}
#[test]
fn verification_fields_occurrences_and_scan_work_are_whole_document_allowances() {
    let p = planner(); let prepared = prepare(&p, "éAéAéAéA", &question(), limits()); let driver = script(&prepared);
    let scans = run(prepared, driver).unwrap().verification_scan_steps;
    for axis in 0..3 {
        let mut q = question();
        match axis { 0 => q.verification.max_fields = 1, 1 => q.verification.max_matches = 3,
            _ => q.verification.max_scan_steps = scans - 1 }
        let prepared = prepare(&p, "éAéAéAéA", &q, limits()); let driver = script(&prepared);
        assert!(run(prepared, driver).is_err());
    }
}
#[test]
fn empty_documents_invalid_questions_and_excess_whole_work_fail_before_inference() {
    let p = planner(); let id = identity(&p); let ctx = PlanContext::new(&id, budget()).unwrap();
    for source in ["", "                  "] {
        assert!(p.plan_int8_question_with_control(source, &question(), budget(), &ctx,
            SourcePlanningLimits::default(), limits(), &mut Continue).is_err());
    }
    for text in [" ".to_owned(), "q".repeat(8193)] {
        let mut q = question(); q.question = text;
        assert!(p.preflight_int8_question_with_control("Alice", &q, budget(), &ctx,
            SourcePlanningLimits::default(), limits(), &mut Continue).is_err());
    }
    let expected = prepare(&p, "éAéAéAéA", &question(), limits()).preflight_metadata();
    for axis in 0..6 {
        let mut l = limits(); l.max_model_work = expected.planned_work(); l.max_mask_visits = expected.reserved_mask_visits();
        match axis { 0 => l.max_model_work.forward_positions -= 1, 1 => l.max_model_work.projected_logits -= 1,
            2 => l.max_model_work.attention_pairs -= 1, 3 => l.max_model_work.projections.dot_products -= 1,
            4 => l.max_model_work.projections.multiply_accumulates -= 1, _ => l.max_mask_visits -= 1 }
        assert!(p.preflight_int8_question_with_control("éAéAéAéA", &question(), budget(), &ctx,
            SourcePlanningLimits::default(), l, &mut Continue).is_err());
    }
}
#[test]
fn malformed_abstention_profile_and_mask_receipts_never_become_answers() {
    let p = planner();
    for axis in 0..4 {
        let prepared = prepare(&p, "éAéA", &question(), limits()); let mut driver = script(&prepared);
        let raw = driver.outputs.front_mut().unwrap();
        match axis { 0 => raw.answerable = false, 1 => raw.status = AnswerStatus::Abstained,
            2 => raw.numerics_profile = "hf-bf16-eager".to_owned(), _ => raw.mask_node_visit_charge = 1001 }
        assert!(run(prepared, driver).is_err());
    }
}
#[test]
fn all_answer_values_are_identical_across_reduction_trees() {
    let p = planner(); let mut baseline = None;
    for fan_in in [2, 3, 8] {
        let mut l = limits(); l.reduction.reduce_fan_in = fan_in;
        let prepared = prepare(&p, "éAéAéAéAéAéAéAéA", &question(), l); let driver = script(&prepared);
        let result = run(prepared, driver).unwrap();
        let value = canonjson::canonical_string(result.mapped.root().value()).unwrap();
        if let Some(expected) = &baseline { assert_eq!(&value, expected); } else { baseline = Some(value); }
    }
}
#[test]
fn late_native_and_final_coordinator_cancellation_keep_the_exact_reason() {
    let p = planner(); let prepared = prepare(&p, "éAéAéAéA", &question(), limits()); let driver = script(&prepared);
    let checkpoints = Rc::clone(&driver.checkpoints); run(prepared, driver).unwrap();
    for native in [true, false] {
        let prepared = prepare(&p, "éAéAéAéA", &question(), limits()); let mut driver = script(&prepared);
        if native { driver.fail_at = Some(2); } else { driver.cancel_at = Some(checkpoints.get()); }
        let error = match run(prepared, driver) { Err(e) => e, Ok(_) => panic!("cancelled") };
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    }
}
#[test]
fn complete_envelope_is_bounded_without_exporting_question_or_generated_tokens() {
    let p = planner(); let prepared = prepare(&p, "éAéAéAéA", &question(), limits()); let driver = script(&prepared);
    let result = run(prepared, driver).unwrap(); let bytes = canonjson::canonical_bytes(&result).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    for private in ["private-question-token", "generated_token_ids", "prompt_digest"] { assert!(!text.contains(private)); }
    for delta in [0, 1] {
        let mut l = limits(); l.reduction.max_result_bytes = bytes.len() - delta;
        let prepared = prepare(&p, "éAéAéAéA", &question(), l); let driver = script(&prepared);
        assert_eq!(run(prepared, driver).is_ok(), delta == 0);
    }
}
