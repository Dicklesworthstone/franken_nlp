//! Real pinned plans with a private scripted driver: these establish admission,
//! complete-source accounting and evidence contracts, not model answer quality.
use super::*;
use super::super::{execution::SourceDriver, tests as fixtures};
use crate::tasks::answer::PassageOccurrence;
use std::{cell::Cell, collections::VecDeque, rc::Rc};

pub(super) fn question() -> SourceQuestion {
    SourceQuestion { question: "Who is named? private-question-token".to_owned(),
        options: AnswerOptions::default(), verification: GroundingBudget::default() }
}
fn identity(planner: &SourceTaskPlanner, width: usize) -> ExecutionIdentity {
    let mut identity = fixtures::identity(planner, &SourceMapTask::Ner(NerOptions::default()), width);
    identity.task_spec = ANSWER_TASK_VERSION.to_owned(); identity
}
pub(super) fn limits() -> Int4SourceMapLimits {
    let mut limits = fixtures::limits();
    limits.chunks.max_chunk_bytes = 6; limits.chunks.max_chunk_tokens = 6;
    limits.reduction.map_batch_chunks = 2; limits
}
pub(super) fn prepare<'s>(planner: &SourceTaskPlanner, source: &'s str, question: &SourceQuestion,
    limits: Int4SourceMapLimits, width: usize) -> PreparedInt4Question<'s> {
    let id = identity(planner, width); let budget = fixtures::budget();
    planner.plan_int4_question_with_control(source, question, budget, &PlanContext::new(&id, budget).unwrap(),
        SourcePlanningLimits::default(), limits, width, &mut fixtures::Continue).unwrap()
}
fn raw(plan: &PreparedInt4SourceTask, chunk: &SourceChunk<'_>) -> AnswerResult {
    let options = plan.extraction.options();
    let ordinary = (0..crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE as u32)
        .find(|id| *id != options.eos_token_id && !options.excluded_token_ids.contains(id)).unwrap();
    let quote = chunk.text().chars().find(|c| !c.is_whitespace()).unwrap().to_string();
    let spans = scan_occurrences(chunk.text(), &quote, &mut GroundingBudget::default()).unwrap();
    let occurrence = if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous };
    AnswerResult { schema_version: 1, task_spec_version: ANSWER_TASK_VERSION.to_owned(),
        numerics_profile: STRICT_INT4_PROFILE.to_owned(), status: AnswerStatus::Answered, answerable: true,
        answer: Some("same answer".to_owned()), citations: vec![PassageCitation { quote, occurrence,
            spans: spans.into_iter().map(|span| PassageOccurrence { passage_id: PASSAGE_ID.to_owned(), span }).collect() }],
        calibration: AnswerCalibration::Uncalibrated, citation_guarantee: CitationGuarantee::StructuralSourceMembership,
        semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
        untrusted_fields: ["answer".to_owned(), "citations".to_owned()],
        generated_token_ids: vec![ordinary, options.eos_token_id], forward_positions: 0,
        projected_logits: 0, mask_node_visit_charge: 20 }
}
pub(super) struct Script {
    pub(super) outputs: VecDeque<AnswerResult>,
    pub(super) calls: Rc<Cell<usize>>,
    pub(super) checkpoints: Rc<Cell<usize>>,
    model_checks: Rc<Cell<usize>>,
    pub(super) width: usize,
    pub(super) capacity: usize,
    pub(super) fail_model_at: Option<usize>,
    pub(super) fail_run_at: Option<usize>,
    pub(super) cancel_at: Option<usize>,
    pub(super) dirty_after: Option<usize>,
    pub(super) corrupt_axis: Option<usize>,
    pub(super) receipt_width: Option<usize>,
}
pub(super) fn script(prepared: &PreparedInt4Question<'_>) -> Script {
    let outputs = prepared.chunks.chunks().iter().zip(&prepared.plan_indices)
        .filter_map(|(chunk, index)| index.map(|index| raw(&prepared.plans[index], chunk))).collect();
    Script { outputs, calls: Rc::new(Cell::new(0)), checkpoints: Rc::new(Cell::new(0)),
        model_checks: Rc::new(Cell::new(0)), width: prepared.group_width(), capacity: 8192,
        fail_model_at: None, fail_run_at: None, cancel_at: None, dirty_after: None,
        corrupt_axis: None, receipt_width: None }
}
impl SourceDriver for Script {
    fn capacity(&self) -> usize { self.capacity }
    fn group_width(&self) -> usize { self.width }
    fn clean(&self) -> bool { self.dirty_after.is_none_or(|at| self.calls.get() < at) }
    fn check_model(&self, _: &ExecutionIdentity) -> Result<(), Int4SourceError> {
        let next = self.model_checks.get() + 1; self.model_checks.set(next);
        if self.fail_model_at == Some(next) { Err(Int4SourceError::InvalidResult) } else { Ok(()) }
    }
    fn checkpoint(&mut self) -> Result<(), Int4SourceError> {
        let next = self.checkpoints.get() + 1; self.checkpoints.set(next);
        if self.cancel_at == Some(next) { Err(Int4SourceError::Cancelled(DecodeCancellationKind::Deadline)) } else { Ok(()) }
    }
    fn run(&mut self, plan: &PreparedInt4SourceTask, _: &ExecutionIdentity) -> Result<Int4SourceTaskRun, Int4SourceError> {
        let next = self.calls.get() + 1; self.calls.set(next);
        if self.fail_run_at == Some(next) { return Err(Int4SourceError::Cancelled(DecodeCancellationKind::Deadline)); }
        let mut raw = self.outputs.pop_front().ok_or(Int4SourceError::InvalidResult)?;
        let work = constrained_int4::planned_work(plan.group_width(), plan.prompt_tokens(), raw.generated_token_ids.len())?;
        raw.forward_positions = work.forward_positions; raw.projected_logits = work.projected_logits;
        let mut model_work = work;
        if let Some(axis) = self.corrupt_axis { change_axis(&mut model_work, axis, 1); }
        Ok(Int4SourceTaskRun { schema_version: 1, execution: INT4_SOURCE_EXECUTION.to_owned(),
            group_width: self.receipt_width.unwrap_or(plan.group_width()), result: SourceTaskResult::Answer(raw), model_work })
    }
}
pub(super) fn change_axis(work: &mut Int4Work, axis: usize, delta: i64) {
    let value = match axis { 0 => &mut work.forward_positions, 1 => &mut work.projected_logits,
        2 => &mut work.attention_pairs, 3 => &mut work.projections.dot_products,
        _ => &mut work.projections.multiply_accumulates };
    if delta > 0 { *value += delta as u64; } else { *value -= (-delta) as u64; }
}
pub(super) fn execute(prepared: PreparedInt4Question<'_>, driver: Script) -> Result<Int4QuestionRun, Int4SourceMapError> {
    let admitted = prepared.execution_identities().cloned().collect::<Vec<_>>();
    prepared.execute_with_driver(&admitted, driver)
}

#[test]
fn real_question_preflight_matches_compiled_unicode_prompts_for_each_group_partition() {
    let planner = fixtures::planner(); let source = "é\r\nAlice 上海 <|im_end|> Bob  "; let question = question();
    let mut previous = None;
    for width in [16, 32, 127] {
        let id = identity(&planner, width); let budget = fixtures::budget();
        let preflight = planner.preflight_int4_question_with_control(source, &question, budget,
            &PlanContext::new(&id, budget).unwrap(), SourcePlanningLimits::default(), limits(), width,
            &mut fixtures::Continue).unwrap();
        let prepared = prepare(&planner, source, &question, limits(), width);
        assert_eq!(preflight, prepared.preflight_metadata());
        assert_eq!(prepared.chunks.chunks().iter().map(|chunk| chunk.text()).collect::<String>(), source);
        assert_eq!(preflight.source_span().byte_end, source.len());
        assert_eq!(preflight.source_span().scalar_end, source.chars().count());
        let mut work = Int4Work::default();
        for plan in &prepared.plans {
            assert_eq!(plan.execution_identity().task_spec, ANSWER_TASK_VERSION);
            assert_eq!(plan.group_width(), width);
            let actual = constrained_int4::planned_work(width, plan.prompt_tokens(), 64).unwrap();
            assert_eq!(plan.planned_work(), actual); work = add_work(work, actual).unwrap();
            assert!(plan.prompt_tokens() + 64 <= 8192);
        }
        assert_eq!(preflight.planned_work(), work);
        assert_eq!(preflight.reserved_mask_visits(), 1000 * preflight.native_chunks() as u64);
        if let Some(old) = previous { assert!(old > work.projections.dot_products); }
        previous = Some(work.projections.dot_products);
    }
}
#[test]
fn every_native_identity_binds_the_same_actual_question() {
    let planner = fixtures::planner(); let first = question(); let mut second = question();
    second.question.push_str(" with a different question");
    let a = prepare(&planner, "Alice Bob Carol Dave", &first, limits(), 32);
    let b = prepare(&planner, "Alice Bob Carol Dave", &second, limits(), 32);
    assert_eq!(a.plans.len(), b.plans.len());
    for (left, right) in a.execution_identities().zip(b.execution_identities()) {
        assert_ne!(left.prompt_digest, right.prompt_digest); assert_ne!(left.taskir_digest, right.taskir_digest);
    }
}
#[test]
fn complete_source_keeps_unicode_occurrences_and_whitespace_without_fake_native_runs() {
    let planner = fixtures::planner(); let source = "ééé                        ééé";
    let prepared = prepare(&planner, source, &question(), limits(), 32);
    let expected = prepared.preflight_metadata(); let driver = script(&prepared); let calls = Rc::clone(&driver.calls);
    assert!(expected.whitespace_chunks() > 0);
    let run = execute(prepared, driver).unwrap(); expected.verify_completed(&run).unwrap();
    assert_eq!(calls.get(), expected.native_chunks()); assert_eq!(run.outcome, QuestionOutcome::OneAnswerText);
    assert_eq!(run.verification_used.matches, 6); assert_eq!(run.mask_node_visit_charge, 20 * calls.get() as u64);
    for chunk in run.mapped.root().value().chunks() {
        if chunk.status == QuestionChunkStatus::WhitespaceOnly {
            assert_eq!(chunk.model_work, Int4Work::default()); assert_eq!(chunk.mask_node_visit_charge, 0);
            assert!(chunk.answer.is_none()); assert!(chunk.citations.is_empty());
        }
        for citation in &chunk.citations { for span in &citation.spans {
            assert_eq!(&source[span.byte_start..span.byte_end], citation.quote);
            assert_eq!(source[..span.byte_start].chars().count(), span.scalar_start);
            assert_eq!(source[..span.byte_end].chars().count(), span.scalar_end);
        } }
    }
}
#[test]
fn distinct_answers_and_model_abstentions_survive_without_global_selection() {
    let planner = fixtures::planner(); let source = "Alice Bob Carol Dave Eve Frank";
    let prepared = prepare(&planner, source, &question(), limits(), 32); let mut driver = script(&prepared);
    assert!(driver.outputs.len() > 2);
    driver.outputs[0].answer = Some("first answer".to_owned());
    driver.outputs[1].answer = Some("second answer".to_owned());
    let last = driver.outputs.back_mut().unwrap();
    last.status = AnswerStatus::Abstained; last.answerable = false; last.answer = None; last.citations.clear();
    let native = driver.outputs.len(); let run = execute(prepared, driver).unwrap();
    assert_eq!(run.outcome, QuestionOutcome::MultipleAnswerTexts); assert!(run.distinct_answer_texts >= 2);
    assert_eq!(run.abstained_chunks, 1); assert_eq!(run.answered_chunks + run.abstained_chunks, native);
    let answers = run.mapped.root().value().chunks().filter_map(|chunk| chunk.answer.as_deref()).collect::<Vec<_>>();
    assert!(answers.contains(&"first answer")); assert!(answers.contains(&"second answer"));
}
#[test]
fn final_identity_actual_model_group_and_full_resident_kv_are_checked_before_first_run() {
    let planner = fixtures::planner();
    for axis in 0..7 {
        let mut prepared = prepare(&planner, "Alice Bob Carol Dave", &question(), limits(), 32);
        let mut admitted = prepared.execution_identities().cloned().collect::<Vec<_>>();
        let mut driver = script(&prepared); let calls = Rc::clone(&driver.calls);
        match axis {
            0 => admitted.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"wrong final question"),
            1 => { admitted.pop(); },
            2 => driver.fail_model_at = Some(prepared.plans.len()),
            3 => prepared.plans.last_mut().unwrap().budget.max_kv_bytes -= 1,
            4 => driver.width = 16, 5 => driver.capacity += 1, _ => driver.dirty_after = Some(0),
        }
        assert!(prepared.execute_with_driver(&admitted, driver).is_err()); assert_eq!(calls.get(), 0);
    }
}
#[test]
fn wrong_passage_question_only_quote_and_incomplete_unicode_occurrences_are_rejected() {
    let planner = fixtures::planner();
    for axis in 0..5 {
        let prepared = prepare(&planner, "éééééé", &question(), limits(), 32); let mut driver = script(&prepared);
        let citation = &mut driver.outputs[0].citations[0];
        match axis {
            0 => citation.quote = "private-question-token".to_owned(),
            1 => citation.spans[0].passage_id = "question".to_owned(),
            2 => citation.spans[0].span.scalar_end += 1,
            3 => { citation.spans.pop(); }, _ => citation.occurrence = SourceOccurrence::Anchored,
        }
        assert!(execute(prepared, driver).is_err());
    }
}
#[test]
fn every_native_work_axis_receipt_group_masks_eos_and_profile_fail_closed() {
    let planner = fixtures::planner();
    for axis in 0..10 {
        let prepared = prepare(&planner, "Alice Bob Carol Dave", &question(), limits(), 32); let mut driver = script(&prepared);
        if axis < 5 { driver.corrupt_axis = Some(axis); } else {
            match axis {
                5 => driver.receipt_width = Some(16),
                6 => driver.outputs[0].mask_node_visit_charge = 1001,
                7 => { driver.outputs[0].generated_token_ids.pop(); },
                8 => driver.outputs[0].numerics_profile = "strict-int8-v1".to_owned(),
                _ => { let eos = *driver.outputs[0].generated_token_ids.last().unwrap(); driver.outputs[0].generated_token_ids[0] = eos; },
            }
        }
        let calls = Rc::clone(&driver.calls); assert!(execute(prepared, driver).is_err()); assert_eq!(calls.get(), 1);
    }
}
#[test]
fn cumulative_verification_fields_matches_and_scan_steps_are_never_renewed() {
    let planner = fixtures::planner(); let source = "éééééééééééé";
    let prepared = prepare(&planner, source, &question(), limits(), 32); let driver = script(&prepared);
    let used = execute(prepared, driver).unwrap().verification_used;
    for axis in 0..3 {
        let mut request = question();
        match axis { 0 => request.verification.max_fields = used.fields - 1,
            1 => request.verification.max_matches = used.matches - 1,
            _ => request.verification.max_scan_steps = used.scan_steps - 1 }
        let prepared = prepare(&planner, source, &request, limits(), 32); let driver = script(&prepared);
        assert!(execute(prepared, driver).is_err());
    }
}
#[test]
fn every_whole_document_work_axis_and_mask_reservation_refuse_one_unit_short() {
    let planner = fixtures::planner(); let source = "Alice Bob Carol Dave"; let request = question();
    let expected = prepare(&planner, source, &request, limits(), 32).preflight_metadata();
    let id = identity(&planner, 32); let budget = fixtures::budget();
    for axis in 0..6 {
        let mut mapping = limits(); mapping.max_model_work = expected.planned_work();
        mapping.max_mask_visits = expected.reserved_mask_visits();
        if axis < 5 { change_axis(&mut mapping.max_model_work, axis, -1); } else { mapping.max_mask_visits -= 1; }
        assert!(matches!(planner.plan_int4_question_with_control(source, &request, budget,
            &PlanContext::new(&id, budget).unwrap(), SourcePlanningLimits::default(), mapping, 32, &mut fixtures::Continue),
            Err(Int4SourceMapError::WorkLimit)));
    }
}
#[test]
fn empty_wholly_blank_and_invalid_question_inputs_produce_no_document_success() {
    let planner = fixtures::planner(); let id = identity(&planner, 32); let budget = fixtures::budget();
    for source in ["", " \t\r\n\u{2003} "] {
        assert!(matches!(planner.plan_int4_question_with_control(source, &question(), budget,
            &PlanContext::new(&id, budget).unwrap(), SourcePlanningLimits::default(), limits(), 32, &mut fixtures::Continue),
            Err(Int4SourceMapError::EmptySource)));
    }
    let mut request = question(); request.question = " \r\n ".to_owned(); assert!(request.validate().is_err());
    request = question(); request.verification.max_matches = 0; assert!(request.validate().is_err());
}
#[test]
fn malformed_abstention_dirty_session_and_later_native_failure_cannot_return_partial_source() {
    let planner = fixtures::planner();
    for axis in 0..3 {
        let prepared = prepare(&planner, "Alice Bob Carol Dave", &question(), limits(), 32); let mut driver = script(&prepared);
        match axis { 0 => driver.outputs[0].status = AnswerStatus::Abstained,
            1 => driver.dirty_after = Some(1), _ => driver.fail_run_at = Some(2) }
        let error = execute(prepared, driver).err().unwrap();
        if axis == 2 { assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); }
    }
}
#[test]
fn reduction_fan_in_preserves_all_answers_and_final_cancellation_preserves_typed_cause() {
    let planner = fixtures::planner(); let source = "Alice Bob Carol Dave Eve Frank Grace";
    let mut baseline = None;
    for fan_in in [2, 3, 8] {
        let mut mapping = limits(); mapping.reduction.reduce_fan_in = fan_in;
        let prepared = prepare(&planner, source, &question(), mapping, 32); let driver = script(&prepared);
        let bytes = canonjson::canonical_bytes(execute(prepared, driver).unwrap().mapped.root().value()).unwrap();
        if let Some(previous) = &baseline { assert_eq!(previous, &bytes); } else { baseline = Some(bytes); }
    }
    let prepared = prepare(&planner, source, &question(), limits(), 32); let driver = script(&prepared);
    let checkpoints = Rc::clone(&driver.checkpoints); execute(prepared, driver).unwrap();
    let prepared = prepare(&planner, source, &question(), limits(), 32); let mut driver = script(&prepared);
    driver.cancel_at = Some(checkpoints.get());
    assert_eq!(execute(prepared, driver).err().unwrap().cancellation(), Some(DecodeCancellationKind::Deadline));
}
#[test]
fn serialized_wrapper_is_bounded_and_omits_private_question_prompts_and_native_ids() {
    let planner = fixtures::planner(); let source = "Alice Bob Carol Dave";
    let prepared = prepare(&planner, source, &question(), limits(), 32); let driver = script(&prepared);
    let run = execute(prepared, driver).unwrap(); let bytes = canonjson::canonical_bytes(&run).unwrap();
    let json = std::str::from_utf8(&bytes).unwrap();
    for private in ["private-question-token", "generated_token_ids", "prompt_digest", "taskir_digest"] {
        assert!(!json.contains(private));
    }
    let mut mapping = limits(); mapping.reduction.max_result_bytes = bytes.len() - 1;
    let prepared = prepare(&planner, source, &question(), mapping, 32); let driver = script(&prepared);
    assert!(execute(prepared, driver).is_err());
}
#[test]
fn completed_receipt_rechecks_group_work_masks_and_verification_counts() {
    let planner = fixtures::planner(); let source = "Alice Bob Carol Dave";
    for axis in 0..5 {
        let prepared = prepare(&planner, source, &question(), limits(), 32);
        let expected = prepared.preflight_metadata(); let driver = script(&prepared); let mut run = execute(prepared, driver).unwrap();
        match axis { 0 => run.group_width = 16, 1 => run.model_work.attention_pairs += 1,
            2 => run.mask_node_visit_charge += 1, 3 => run.verification_used.fields += 1,
            _ => run.verification_used.matches += 1 }
        assert!(expected.verify_completed(&run).is_err());
    }
}
