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
    for private in ["private-question-token", "generated_token_ids", "prompt_digest", "request_binding", "result_binding"] {
        assert!(!text.contains(private));
    }
    for delta in [0, 1] {
        let mut l = limits(); l.reduction.max_result_bytes = bytes.len() - delta;
        let prepared = prepare(&p, "éAéAéAéA", &question(), l); let driver = script(&prepared);
        assert_eq!(run(prepared, driver).is_ok(), delta == 0);
    }
}

#[test]
fn equal_geometry_does_not_authorize_a_different_source_question_or_policy() {
    let p = planner();
    let mut q = question(); q.question = "Who??".to_owned();
    let prepared = prepare(&p, "Alice!", &q, limits());
    let expected = prepared.preflight_metadata();
    let driver = script(&prepared); let result = run(prepared, driver).unwrap();
    expected.verify_completed(&result).unwrap();
    for axis in 0..4 {
        let mut q = question(); q.question = "Who??".to_owned();
        let mut mapping = limits();
        let source = if axis == 0 { "Carol!" } else { "Alice!" };
        match axis {
            1 => q.question = "What?".to_owned(),
            2 => q.verification.max_scan_steps -= 1,
            3 => mapping.max_mask_visits += 1,
            _ => {},
        }
        let foreign = prepare(&p, source, &q, mapping).preflight_metadata();
        assert_eq!(foreign.source_span(), expected.source_span());
        assert_eq!(foreign.planned_work(), expected.planned_work());
        assert_eq!(foreign.reserved_mask_visits(), expected.reserved_mask_visits());
        assert!(foreign.verify_completed(&result).is_err(), "changed request axis {axis}");
    }
    let diagnostic = format!("{expected:?}");
    for private in ["request_binding", "Alice!", "Who??"] { assert!(!diagnostic.contains(private)); }
}

#[test]
fn changing_only_answer_text_or_citations_invalidates_the_complete_payload() {
    let p = planner(); let q = question();
    for axis in 0..2 {
        let prepared = prepare(&p, "Alice!", &q, limits());
        let expected = prepared.preflight_metadata();
        let driver = script(&prepared); let mut original = run(prepared, driver).unwrap();
        let prepared = prepare(&p, "Alice!", &q, limits());
        let mut driver = script(&prepared);
        let raw = driver.outputs.front_mut().unwrap();
        if axis == 0 {
            raw.answer = Some("fake answer".to_owned());
        } else {
            raw.citations[0].quote = "l".to_owned();
            raw.citations[0].spans[0].span = VerifiedSourceSpan {
                byte_start: 1, byte_end: 2, scalar_start: 1, scalar_end: 2 };
        }
        let other = run(prepared, driver).unwrap();
        // Both outputs are independently valid for this exact request. The
        // completion seal also binds the particular output that was produced.
        expected.verify_completed(&other).unwrap();
        assert_eq!(original.model_work, other.model_work);
        assert_eq!(original.distinct_answer_texts, other.distinct_answer_texts);
        original.mapped = other.mapped;
        assert!(expected.verify_completed(&original).is_err());
    }
}

#[test]
fn verification_keeps_the_original_typed_cancellation() {
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    let p = planner(); let prepared = prepare(&p, "Alice!", &question(), limits());
    let expected = prepared.preflight_metadata();
    let driver = script(&prepared); let result = run(prepared, driver).unwrap();
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        assert_eq!(expected.verify_completed_with_control(&result, &mut Stop(cause))
            .unwrap_err().cancellation(), Some(cause));
    }
}

const COMPACT_SOURCE: &str = "The quick brown fox jumps over the lazy dog. 上海 <think> <|im_start|> ▁\r\n";
fn compact_policy() -> crate::tokenizer::untrusted::UntrustedEncoding {
    use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 })
}
fn encoding_allowance() -> crate::tokenizer::untrusted::SafeBpeWorkLimits {
    crate::tokenizer::untrusted::SafeBpeWorkLimits { max_input_bytes: 65_536, max_output_tokens: 65_536,
        max_pair_checks: 1 << 26, max_pair_bytes: 1 << 30 }
}
fn compact(work: crate::tokenizer::untrusted::SafeBpeWorkLimits) -> SourceTaskPlanner {
    planner().with_map_document_encoding(compact_policy(), work).unwrap()
}
fn compact_mapping() -> Int8SourceMapLimits {
    let mut limits = limits();
    limits.chunks.max_chunk_bytes = 4096; limits.chunks.max_chunk_tokens = 4096;
    limits
}
fn encoding_used(planner: &SourceTaskPlanner) -> crate::tokenizer::untrusted::SafeBpeWork {
    planner.map_document_encoding_work().unwrap().unwrap()
}

#[test]
fn compact_question_preflight_and_planning_share_all_question_manifest_and_source_work() {
    let q = question(); let byte = planner(); let p = compact(encoding_allowance());
    let id = identity(&p); let context = PlanContext::new(&id, budget()).unwrap();
    let original = prepare(&byte, COMPACT_SOURCE, &q, compact_mapping());
    let expected = p.preflight_int8_question_with_control(COMPACT_SOURCE, &q, budget(), &context,
        SourcePlanningLimits::default(), compact_mapping(), &mut Continue).unwrap();
    let preflight_work = encoding_used(&p);
    let prepared = prepare(&p, COMPACT_SOURCE, &q, compact_mapping());
    assert_eq!(expected, prepared.preflight_metadata());
    assert_eq!(encoding_used(&p), preflight_work.checked_add(preflight_work).unwrap());
    assert_eq!(expected.source_span(), original.preflight_metadata().source_span());
    assert!(expected.planned_work().forward_positions < original.preflight_metadata().planned_work().forward_positions);
    assert_ne!(prepared.execution_identities().next().unwrap().prompt_digest,
        original.execution_identities().next().unwrap().prompt_digest);
    let passages = [AnswerPassage { id: PASSAGE_ID.to_owned(), text: COMPACT_SOURCE.to_owned() }];
    let complete = AnswerContext::encode(&byte.encoder, &q.question, &passages,
        AnswerInputLimits { max_passages: 1, max_input_bytes: 4096, max_input_tokens: 8192 }).unwrap();
    assert_eq!(preflight_work.input_bytes(), q.question.len() + COMPACT_SOURCE.len()
        + complete.document().total_input_bytes());
    assert!(preflight_work.output_tokens() < preflight_work.input_bytes());
    assert!(preflight_work.pair_checks() > 0 && preflight_work.pair_bytes() > 0);
    let mut driver = script(&prepared);
    let start = COMPACT_SOURCE.find("上海").unwrap(); let end = start + "上海".len();
    driver.outputs[0].citations = vec![PassageCitation { quote: "上海".to_owned(),
        occurrence: SourceOccurrence::Anchored, spans: vec![PassageOccurrence {
            passage_id: PASSAGE_ID.to_owned(), span: VerifiedSourceSpan { byte_start: start, byte_end: end,
                scalar_start: COMPACT_SOURCE[..start].chars().count(), scalar_end: COMPACT_SOURCE[..end].chars().count() } }] }];
    let result = run(prepared, driver).unwrap(); expected.verify_completed(&result).unwrap();
    for chunk in result.mapped.root().value().chunks() {
        for citation in &chunk.citations { for span in &citation.spans {
            assert_eq!(&COMPACT_SOURCE[span.byte_start..span.byte_end], citation.quote);
            assert_eq!(COMPACT_SOURCE[..span.byte_start].chars().count(), span.scalar_start);
        } }
    }
}

#[test]
fn compact_question_total_limits_are_exact_and_never_renew_after_failed_preparation() {
    use crate::tokenizer::untrusted::SafeBpeWorkLimits;
    let q = question(); let baseline = compact(encoding_allowance());
    let _ = prepare(&baseline, COMPACT_SOURCE, &q, compact_mapping());
    let consumed = encoding_used(&baseline);
    let exact = SafeBpeWorkLimits { max_input_bytes: consumed.input_bytes() as u64,
        max_output_tokens: consumed.output_tokens() as u64, max_pair_checks: consumed.pair_checks(),
        max_pair_bytes: consumed.pair_bytes() };
    for axis in 0..5 {
        let mut allowance = exact;
        match axis { 0 => allowance.max_input_bytes -= 1, 1 => allowance.max_output_tokens -= 1,
            2 => allowance.max_pair_checks -= 1, 3 => allowance.max_pair_bytes -= 1, _ => {} }
        let p = compact(allowance); let id = identity(&p); let context = PlanContext::new(&id, budget()).unwrap();
        let first = p.plan_int8_question_with_control(COMPACT_SOURCE, &q, budget(), &context,
            SourcePlanningLimits::default(), compact_mapping(), &mut Continue);
        assert_eq!(first.is_ok(), axis == 4); drop(first);
        let failed = encoding_used(&p);
        assert!(failed.input_bytes() > 0);
        if axis == 4 { assert_eq!(failed, consumed); }
        assert!(p.plan_int8_question_with_control(COMPACT_SOURCE, &q, budget(), &context,
            SourcePlanningLimits::default(), compact_mapping(), &mut Continue).is_err());
        let repeated = encoding_used(&p);
        assert!(repeated.input_bytes() >= failed.input_bytes() && repeated.output_tokens() >= failed.output_tokens());
        assert!(repeated.pair_checks() >= failed.pair_checks() && repeated.pair_bytes() >= failed.pair_bytes());
        assert!(repeated.input_bytes() as u64 <= allowance.max_input_bytes);
        assert!(repeated.output_tokens() as u64 <= allowance.max_output_tokens);
        assert!(repeated.pair_checks() <= allowance.max_pair_checks && repeated.pair_bytes() <= allowance.max_pair_bytes);
    }
}

#[test]
fn compact_question_byte_budget_is_independent_of_actual_prompt_token_capacity() {
    let mut q = question(); q.question = " the".repeat(600);
    let mut b = budget(); b.max_input_tokens = 2048;
    assert!(q.question.len() > b.max_input_tokens as usize);
    let p = compact(encoding_allowance()); let id = identity(&p);
    let plan = p.plan_int8_question_with_control("Alice", &q, b, &PlanContext::new(&id, b).unwrap(),
        SourcePlanningLimits::default(), compact_mapping(), &mut Continue).unwrap();
    assert!(plan.plans.iter().all(|plan| plan.prompt_tokens() <= b.max_input_tokens as usize));
    let p = planner(); let id = identity(&p);
    assert!(p.plan_int8_question_with_control("Alice", &q, b, &PlanContext::new(&id, b).unwrap(),
        SourcePlanningLimits::default(), compact_mapping(), &mut Continue).is_err());
}

#[test]
fn compact_question_discarded_probes_preserve_utf8_crlf_and_spend_the_same_ledger() {
    let p = compact(encoding_allowance()); let mut l = compact_mapping(); l.chunks.max_chunk_tokens = 4;
    let prepared = prepare(&p, COMPACT_SOURCE, &question(), l);
    assert!(prepared.chunks.tokenizer_calls() > prepared.chunks.chunks().len());
    assert_eq!(prepared.chunks.chunks().iter().map(SourceChunk::text).collect::<String>(), COMPACT_SOURCE);
    assert!(encoding_used(&p).input_bytes() > COMPACT_SOURCE.len() * 2 + question().question.len());
    for chunk in prepared.chunks.chunks() {
        assert!(chunk.tokens() <= 4);
        assert!(!chunk.text().ends_with('\r'));
    }
}

#[test]
fn compact_question_identity_binds_declared_encoding_caps_without_hashing_remaining_work() {
    use crate::tokenizer::untrusted::UntrustedEncoding;
    let q = question(); let p = compact(encoding_allowance());
    let first = prepare(&p, COMPACT_SOURCE, &q, compact_mapping());
    let original: Vec<_> = first.execution_identities().cloned().collect();
    let second = prepare(&p, COMPACT_SOURCE, &q, compact_mapping());
    assert_eq!(original, second.execution_identities().cloned().collect::<Vec<_>>());
    for axis in 0..8 {
        let UntrustedEncoding::SafeBpeV1(mut call) = compact_policy() else { panic!("compact policy") };
        let mut work = encoding_allowance();
        match axis { 0 => call.max_input_bytes -= 1, 1 => call.max_output_tokens -= 1,
            2 => call.max_pair_checks -= 1, 3 => call.max_pair_bytes -= 1,
            4 => work.max_input_bytes -= 1, 5 => work.max_output_tokens -= 1,
            6 => work.max_pair_checks -= 1, _ => work.max_pair_bytes -= 1 }
        let changed = planner().with_map_document_encoding(UntrustedEncoding::SafeBpeV1(call), work).unwrap();
        let actual = prepare(&changed, COMPACT_SOURCE, &q, compact_mapping());
        assert_eq!(first.preflight_metadata().planned_work(), actual.preflight_metadata().planned_work());
        for (a, b) in original.iter().zip(actual.execution_identities()) {
            assert_ne!(a.template_digest, b.template_digest);
            assert_eq!(a.taskir_digest, b.taskir_digest);
        }
    }
}

#[test]
fn compact_question_requires_a_whole_operation_budget_and_keeps_synthesis_explicitly_byte_only() {
    let q = question(); let p = planner().with_document_encoding(compact_policy()).unwrap();
    let id = identity(&p); let context = PlanContext::new(&id, budget()).unwrap();
    assert!(p.plan_int8_question_with_control(COMPACT_SOURCE, &q, budget(), &context,
        SourcePlanningLimits::default(), compact_mapping(), &mut Continue).is_err());
    let p = compact(encoding_allowance()); let id = identity(&p); let context = PlanContext::new(&id, budget()).unwrap();
    let request = synthesis::SourceQuestionSynthesis { question: q,
        limits: synthesis::QuestionSynthesisLimits { max_evidence_passages: 8, max_evidence_bytes: 1024 } };
    assert!(p.preflight_int8_question_synthesis_with_control(COMPACT_SOURCE, &request, budget(), &context,
        SourcePlanningLimits::default(), compact_mapping(), &mut Continue).is_err());
    assert!(p.plan_int8_question_synthesis_with_control(COMPACT_SOURCE, &request, budget(), &context,
        SourcePlanningLimits::default(), compact_mapping(), &mut Continue).is_err());
    assert_eq!(encoding_used(&p).input_bytes(), 0);
}

struct StopInsideQuestionEncoding<'a> {
    planner: &'a SourceTaskPlanner, cause: DecodeCancellationKind,
    wait_for_question: bool, armed: bool, busy_polls: usize,
}
impl DecodeStepControl for StopInsideQuestionEncoding<'_> {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        match self.planner.map_document_encoding_work() {
            Ok(Some(work)) if work.input_bytes() > 0 => self.armed = true,
            Err(_) if !self.wait_for_question || self.armed => {
                self.busy_polls += 1;
                if self.busy_polls == 8 { return Some(self.cause); }
            },
            _ => {},
        }
        None
    }
}
#[test]
fn compact_question_and_chunk_encoding_preserve_every_original_cancellation_cause() {
    use DecodeCancellationKind::*;
    for wait_for_question in [false, true] {
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let p = compact(encoding_allowance()); let mut q = question(); q.question = " the".repeat(200);
            let id = identity(&p); let context = PlanContext::new(&id, budget()).unwrap();
            let mut control = StopInsideQuestionEncoding { planner: &p, cause, wait_for_question,
                armed: false, busy_polls: 0 };
            let error = p.plan_int8_question_with_control(COMPACT_SOURCE, &q, budget(), &context,
                SourcePlanningLimits::default(), compact_mapping(), &mut control).err().unwrap();
            assert_eq!(error.cancellation(), Some(cause));
            assert_eq!(control.busy_polls, 8);
            assert!(encoding_used(&p).input_bytes() > 0);
            if wait_for_question { assert!(encoding_used(&p).input_bytes() > q.question.len()); }
        }
    }
}
