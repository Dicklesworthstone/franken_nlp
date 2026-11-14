//! Complete-question planning and sealed scripted results, without model weights.
use super::*;
use super::tests as qa;
use super::super::tests as fixtures;
use crate::tasks::answer::PassageOccurrence;
use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding};

const SOURCE: &str = "The quick brown fox jumps over the lazy dog. 上海 <think> <|im_start|> ▁";
fn policy() -> UntrustedEncoding {
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 })
}
fn allowance() -> SafeBpeWorkLimits {
    SafeBpeWorkLimits { max_input_bytes: 65_536, max_output_tokens: 65_536,
        max_pair_checks: 1 << 26, max_pair_bytes: 1 << 30 }
}
fn compact(work: SafeBpeWorkLimits) -> SourceTaskPlanner {
    fixtures::planner().with_map_document_encoding(policy(), work).unwrap()
}
fn mapping() -> Int4SourceMapLimits {
    let mut limits = qa::limits();
    limits.chunks.max_chunk_bytes = 4096; limits.chunks.max_chunk_tokens = 4096;
    limits
}
fn identity(planner: &SourceTaskPlanner) -> ExecutionIdentity {
    let mut identity = fixtures::identity(planner, &SourceMapTask::Ner(NerOptions::default()), 32);
    identity.task_spec = ANSWER_TASK_VERSION.to_owned(); identity
}
fn plan<'s, C: DecodeStepControl>(planner: &SourceTaskPlanner, source: &'s str, question: &SourceQuestion,
    budget: TaskBudget, limits: Int4SourceMapLimits, control: &mut C)
    -> Result<PreparedInt4Question<'s>, Int4SourceMapError> {
    planner.plan_int4_question_with_control(source, question, budget,
        &PlanContext::new(&identity(planner), budget).unwrap(), SourcePlanningLimits::default(), limits, 32, control)
}
fn used(planner: &SourceTaskPlanner) -> SafeBpeWork {
    planner.map_document_encoding_work().unwrap().unwrap()
}

#[test]
fn compact_complete_question_preflight_matches_native_plans_and_original_evidence_coordinates() {
    let question = qa::question(); let budget = fixtures::budget();
    let byte = fixtures::planner();
    let original = qa::prepare(&byte, SOURCE, &question, mapping(), 32);
    let compact = compact(allowance());
    let expected = compact.preflight_int4_question_with_control(SOURCE, &question, budget,
        &PlanContext::new(&identity(&compact), budget).unwrap(), SourcePlanningLimits::default(), mapping(), 32,
        &mut fixtures::Continue).unwrap();
    let preflight_work = used(&compact);
    let prepared = qa::prepare(&compact, SOURCE, &question, mapping(), 32);
    assert_eq!(expected, prepared.preflight_metadata());
    assert_eq!(prepared.chunks.chunks().iter().map(SourceChunk::text).collect::<String>(), SOURCE);
    assert_eq!(expected.source_span(), original.preflight_metadata().source_span());
    assert!(expected.planned_work().forward_positions < original.preflight_metadata().planned_work().forward_positions);
    assert_ne!(prepared.execution_identities().next().unwrap().prompt_digest,
        original.execution_identities().next().unwrap().prompt_digest);
    // Both paths encode the sizing question, the source probe and the entire
    // actual question/manifest/source prompt. Hosted replanning cannot reset it.
    assert_eq!(used(&compact), preflight_work.checked_add(preflight_work).unwrap());
    let passages = [AnswerPassage { id: PASSAGE_ID.to_owned(), text: SOURCE.to_owned() }];
    let bytes = AnswerContext::encode(&byte.encoder, &question.question, &passages,
        AnswerInputLimits { max_passages: 1, max_input_bytes: 4096, max_input_tokens: 8192 }).unwrap();
    assert_eq!(preflight_work.input_bytes(), question.question.len() + SOURCE.len() + bytes.document().total_input_bytes());
    assert!(preflight_work.output_tokens() < preflight_work.input_bytes());
    assert!(preflight_work.pair_checks() > 0 && preflight_work.pair_bytes() > 0);
    let mut driver = qa::script(&prepared);
    let byte_start = SOURCE.find("上海").unwrap(); let byte_end = byte_start + "上海".len();
    let span = VerifiedSourceSpan { byte_start, byte_end,
        scalar_start: SOURCE[..byte_start].chars().count(), scalar_end: SOURCE[..byte_end].chars().count() };
    driver.outputs[0].citations = vec![PassageCitation { quote: "上海".to_owned(),
        occurrence: SourceOccurrence::Anchored, spans: vec![PassageOccurrence { passage_id: PASSAGE_ID.to_owned(), span }] }];
    let run = qa::execute(prepared, driver).unwrap();
    expected.verify_completed(&run).unwrap();
    for chunk in run.mapped.root().value().chunks() {
        for citation in &chunk.citations { for span in &citation.spans {
            assert_eq!(&SOURCE[span.byte_start..span.byte_end], citation.quote);
            assert_eq!(SOURCE[..span.byte_start].chars().count(), span.scalar_start);
            assert_eq!(SOURCE[..span.byte_end].chars().count(), span.scalar_end);
        } }
    }
}

#[test]
fn every_cumulative_encoding_axis_admits_exact_work_and_refuses_one_unit_short_without_refunds() {
    let question = qa::question(); let budget = fixtures::budget();
    let baseline = compact(allowance());
    let _ = qa::prepare(&baseline, SOURCE, &question, mapping(), 32);
    let consumed = used(&baseline);
    let exact = SafeBpeWorkLimits { max_input_bytes: consumed.input_bytes() as u64,
        max_output_tokens: consumed.output_tokens() as u64,
        max_pair_checks: consumed.pair_checks(), max_pair_bytes: consumed.pair_bytes() };
    let admitted = compact(exact);
    let _ = qa::prepare(&admitted, SOURCE, &question, mapping(), 32);
    assert_eq!(used(&admitted), consumed);
    assert!(plan(&admitted, SOURCE, &question, budget, mapping(), &mut fixtures::Continue).is_err());
    assert_eq!(used(&admitted), consumed);
    for axis in 0..4 {
        let mut short = exact;
        match axis { 0 => short.max_input_bytes -= 1, 1 => short.max_output_tokens -= 1,
            2 => short.max_pair_checks -= 1, _ => short.max_pair_bytes -= 1 }
        let refused = compact(short);
        assert!(plan(&refused, SOURCE, &question, budget, mapping(), &mut fixtures::Continue).is_err());
        let failed = used(&refused);
        assert!(failed.input_bytes() > 0, "failed complete prompts retain earlier encoding work");
        assert!(failed.input_bytes() <= consumed.input_bytes());
        assert!(failed.output_tokens() <= consumed.output_tokens());
        assert!(failed.pair_checks() <= consumed.pair_checks());
        assert!(failed.pair_bytes() <= consumed.pair_bytes());
        assert!(plan(&refused, SOURCE, &question, budget, mapping(), &mut fixtures::Continue).is_err());
        let repeated = used(&refused);
        assert!(repeated.input_bytes() >= failed.input_bytes() && repeated.output_tokens() >= failed.output_tokens());
        assert!(repeated.pair_checks() >= failed.pair_checks() && repeated.pair_bytes() >= failed.pair_bytes());
    }
}

#[test]
fn question_byte_length_may_exceed_token_budget_only_when_actual_compact_prompt_fits() {
    let mut question = qa::question(); question.question = " the".repeat(600);
    let mut budget = fixtures::budget(); budget.max_input_tokens = 2048;
    assert!(question.question.len() > budget.max_input_tokens as usize);
    let compact = compact(allowance());
    let prepared = plan(&compact, "Alice", &question, budget, mapping(), &mut fixtures::Continue).unwrap();
    assert!(prepared.plans.iter().all(|p| p.prompt_tokens() <= budget.max_input_tokens as usize));
    assert!(plan(&fixtures::planner(), "Alice", &question, budget, mapping(), &mut fixtures::Continue).is_err());
}

#[test]
fn rejected_chunk_probes_also_consume_the_question_operations_encoding_allowance() {
    let question = qa::question(); let compact = compact(allowance());
    let mut limits = mapping(); limits.chunks.max_chunk_tokens = 4;
    let prepared = qa::prepare(&compact, SOURCE, &question, limits, 32);
    assert!(prepared.chunks.tokenizer_calls() > prepared.chunks.chunks().len());
    assert!(used(&compact).input_bytes() > SOURCE.len() * 2 + question.question.len());
    assert_eq!(prepared.chunks.chunks().iter().map(SourceChunk::text).collect::<String>(), SOURCE);
}

#[test]
fn compact_question_and_synthesis_require_one_whole_operation_allowance() {
    let question = qa::question(); let budget = fixtures::budget();
    let unbudgeted = fixtures::planner().with_document_encoding(policy()).unwrap();
    assert!(plan(&unbudgeted, SOURCE, &question, budget, mapping(), &mut fixtures::Continue).is_err());
    let compact = compact(allowance());
    let request = synthesis::SourceQuestionSynthesis { question,
        limits: synthesis::QuestionSynthesisLimits { max_evidence_passages: 8, max_evidence_bytes: 1024 } };
    let id = identity(&compact); let context = PlanContext::new(&id, budget).unwrap();
    let expected = compact.preflight_int4_question_synthesis_with_control(SOURCE, &request, budget, &context,
        SourcePlanningLimits::default(), mapping(), 32, &mut fixtures::Continue).unwrap();
    let first = used(&compact);
    let prepared = compact.plan_int4_question_synthesis_with_control(SOURCE, &request, budget, &context,
        SourcePlanningLimits::default(), mapping(), 32, &mut fixtures::Continue).unwrap();
    assert_eq!(expected, prepared.preflight_metadata());
    assert!(first.input_bytes() > 0);
    assert_eq!(used(&compact), first.checked_add(first).unwrap());
    let unbudgeted_id = identity(&unbudgeted);
    assert!(unbudgeted.plan_int4_question_synthesis_with_control(SOURCE, &request, budget,
        &PlanContext::new(&unbudgeted_id, budget).unwrap(), SourcePlanningLimits::default(), mapping(), 32,
        &mut fixtures::Continue).is_err());
}

struct StopInsideEncoding<'a> {
    planner: &'a SourceTaskPlanner,
    cause: DecodeCancellationKind,
    wait_for_question: bool,
    armed: bool,
    busy_polls: usize,
}
impl DecodeStepControl for StopInsideEncoding<'_> {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        match self.planner.map_document_encoding_work() {
            Ok(Some(work)) if work.input_bytes() > 0 => self.armed = true,
            // The cumulative ledger is held only while an actual encoding is
            // running. Stop inside it, after source bytes have been charged.
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
fn every_cancellation_cause_survives_inside_question_encoding_and_source_chunk_probes() {
    use DecodeCancellationKind::*;
    for wait_for_question in [false, true] {
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let compact = compact(allowance());
            let mut question = qa::question(); question.question = " the".repeat(200);
            let mut control = StopInsideEncoding { planner: &compact, cause, wait_for_question,
                armed: false, busy_polls: 0 };
            let error = plan(&compact, SOURCE, &question, fixtures::budget(), mapping(), &mut control).err().unwrap();
            assert_eq!(error.cancellation(), Some(cause));
            assert_eq!(control.busy_polls, 8);
            assert!(used(&compact).input_bytes() > 0);
            if wait_for_question { assert!(used(&compact).input_bytes() > question.question.len()); }
        }
    }
}

#[test]
fn sealed_question_results_reject_equal_length_source_or_question_substitutions_and_payload_edits() {
    let planner = fixtures::planner();
    let mut first_question = qa::question(); first_question.question = "Who?".to_owned();
    let mut second_question = qa::question(); second_question.question = "Why?".to_owned();
    let first = qa::prepare(&planner, "Alice Bob", &first_question, qa::limits(), 32);
    let expected = first.preflight_metadata();
    let driver = qa::script(&first); let mut first = qa::execute(first, driver).unwrap();
    expected.verify_completed(&first).unwrap();
    for (source, question) in [("Alice Eve", &first_question), ("Alice Bob", &second_question)] {
        let changed = qa::prepare(&planner, source, question, qa::limits(), 32);
        let other = changed.preflight_metadata();
        assert_eq!(expected.source_span(), other.source_span());
        assert_eq!(expected.planned_work(), other.planned_work());
        let driver = qa::script(&changed); let changed = qa::execute(changed, driver).unwrap();
        other.verify_completed(&changed).unwrap();
        assert!(expected.verify_completed(&changed).is_err());
        assert!(other.verify_completed(&first).is_err());
    }
    let public = serde_json::to_value(&first).unwrap();
    assert!(public.get("request_binding").is_none() && public.get("result_binding").is_none());
    assert!(!format!("{expected:?}").contains("request_binding"));
    let bytes = serde_json::to_vec(&first).unwrap().len() as u64;
    assert_eq!(binding::result(&first, bytes).unwrap(), first.result_binding);
    assert!(binding::result(&first, bytes - 1).is_err());
    // This accounting change stays below the public verification ceiling;
    // exact completion binding must still refuse it.
    first.verification_used.scan_steps += 1;
    assert!(expected.verify_completed(&first).is_err());
}

#[test]
fn identical_question_and_native_work_cannot_authorize_different_verification_or_encoding_caps() {
    let planner = compact(allowance()); let question = qa::question();
    let original = qa::prepare(&planner, SOURCE, &question, mapping(), 32);
    let expected = original.preflight_metadata();
    let mut limits = mapping(); limits.verification.max_scan_steps -= 1;
    let changed = qa::prepare(&planner, SOURCE, &question, limits, 32);
    assert_eq!(expected.planned_work(), changed.preflight_metadata().planned_work());
    let driver = qa::script(&changed); let changed = qa::execute(changed, driver).unwrap();
    assert!(expected.verify_completed(&changed).is_err());
    for axis in 0..4 {
        let mut cap = allowance();
        match axis { 0 => cap.max_input_bytes -= 1, 1 => cap.max_output_tokens -= 1,
            2 => cap.max_pair_checks -= 1, _ => cap.max_pair_bytes -= 1 }
        let other = compact(cap);
        let changed = qa::prepare(&other, SOURCE, &question, mapping(), 32);
        assert_eq!(expected.planned_work(), changed.preflight_metadata().planned_work());
        let driver = qa::script(&changed); let changed = qa::execute(changed, driver).unwrap();
        assert!(expected.verify_completed(&changed).is_err());
    }
}
