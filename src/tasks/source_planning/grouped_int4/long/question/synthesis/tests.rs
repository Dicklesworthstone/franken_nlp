//! Source-only pinned planning and private synthetic receipt fixtures. No native
//! model execution, answer quality, confidence, parity, or performance evidence.
use super::*;
use crate::{native_engine::lmhead::NANBEIGE_VOCAB_SIZE, tasks::answer::PassageOccurrence,
    tasks::source_planning::grouped_int4::long::tests as maps};
use std::collections::VecDeque;

pub(super) struct Control { calls: usize, stop: usize }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.calls >= self.stop).then_some(DecodeCancellationKind::PollQuota)
    }
}
pub(super) fn continuing() -> Control { Control { calls: 0, stop: usize::MAX } }
pub(super) fn request() -> SourceQuestionSynthesis {
    SourceQuestionSynthesis { question: SourceQuestion { question: "Who is mentioned?".to_owned(),
        options: AnswerOptions::default(), verification: GroundingBudget::default() },
        limits: QuestionSynthesisLimits { max_evidence_passages: 32, max_evidence_bytes: 4096 } }
}
pub(super) fn mapping() -> Int4SourceMapLimits {
    let mut m = maps::limits(); m.chunks.max_chunk_bytes = 6; m.chunks.max_chunk_tokens = 6; m
}
fn identity(p: &SourceTaskPlanner, width: usize) -> ExecutionIdentity {
    let mut id = maps::identity(p, &SourceMapTask::Ner(NerOptions::default()), width);
    id.task_spec = ANSWER_TASK_VERSION.to_owned(); id
}
pub(super) fn prepare<'s>(p: &SourceTaskPlanner, text: &'s str, request: &SourceQuestionSynthesis,
    mapping: Int4SourceMapLimits, width: usize, control: &mut Control)
    -> Result<PreparedInt4QuestionSynthesis<'s>, Int4SourceMapError> {
    let id = identity(p, width); let budget = maps::budget();
    p.plan_int4_question_synthesis_with_control(text, request, budget, &PlanContext::new(&id, budget).unwrap(),
        SourcePlanningLimits::default(), mapping, width, control)
}
pub(super) fn raw(plan: &PreparedInt4SourceTask, passages: &[(&str, &str)], quote: Option<&str>, text: &str) -> AnswerResult {
    let options = plan.extraction.options();
    let ordinary = (0..NANBEIGE_VOCAB_SIZE as u32)
        .find(|id| *id != options.eos_token_id && !options.excluded_token_ids.contains(id)).unwrap();
    let citations = if let Some(quote) = quote {
        let mut spans = Vec::new();
        for &(id, source) in passages {
            match scan_occurrences(source, quote, &mut GroundingBudget::default()) {
                Ok(found) => spans.extend(found.into_iter().map(|span| PassageOccurrence { passage_id: id.to_owned(), span })),
                Err(FieldGroundingError::Absent) => (), Err(error) => panic!("{error}"),
            }
        }
        vec![PassageCitation { quote: quote.to_owned(), occurrence: if spans.len() == 1 {
            SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }]
    } else { Vec::new() };
    let work = constrained_int4::planned_work(plan.group_width(), plan.prompt_tokens(), 2).unwrap();
    AnswerResult { schema_version: 1, task_spec_version: ANSWER_TASK_VERSION.to_owned(), numerics_profile: STRICT_INT4_PROFILE.to_owned(),
        status: if quote.is_some() { AnswerStatus::Answered } else { AnswerStatus::Abstained }, answerable: quote.is_some(),
        answer: quote.map(|_| text.to_owned()), citations, calibration: AnswerCalibration::Uncalibrated,
        citation_guarantee: CitationGuarantee::StructuralSourceMembership, semantic_support: SummarySemanticSupport::NotAssessed,
        score_space: ScoreSpace::NotComputed, untrusted_fields: ["answer".to_owned(), "citations".to_owned()],
        generated_token_ids: vec![ordinary, options.eos_token_id], forward_positions: work.forward_positions,
        projected_logits: work.projected_logits, mask_node_visit_charge: 20 }
}
struct Script { outputs: VecDeque<AnswerResult>, width: usize }
impl SourceDriver for Script {
    fn capacity(&self) -> usize { 8192 }
    fn group_width(&self) -> usize { self.width }
    fn clean(&self) -> bool { true }
    fn check_model(&self, _: &ExecutionIdentity) -> Result<(), Int4SourceError> { Ok(()) }
    fn checkpoint(&mut self) -> Result<(), Int4SourceError> { Ok(()) }
    fn run(&mut self, plan: &PreparedInt4SourceTask, _: &ExecutionIdentity) -> Result<Int4SourceTaskRun, Int4SourceError> {
        let raw = self.outputs.pop_front().ok_or(Int4SourceError::InvalidResult)?;
        let work = constrained_int4::planned_work(plan.group_width(), plan.prompt_tokens(), raw.generated_token_ids.len())?;
        Ok(Int4SourceTaskRun { schema_version: 1, execution: INT4_SOURCE_EXECUTION.to_owned(), group_width: self.width,
            result: SourceTaskResult::Answer(raw), model_work: work })
    }
}
// mode0=answered, mode1=all maps abstain, mode2=final pass abstains.
pub(super) fn synthetic(mode: usize) -> (Int4QuestionSynthesisPreflight, Int4QuestionSynthesisRun) {
    let p = maps::planner(); let source = "ééé                        ééé"; let request = request();
    let prepared = prepare(&p, source, &request, mapping(), 32, &mut continuing()).unwrap();
    let expected = prepared.expected;
    let admitted = prepared.execution_identities().cloned().collect::<Vec<_>>();
    let outputs = prepared.map.chunks.chunks().iter().zip(&prepared.map.plan_indices).filter_map(|(chunk, index)| {
        index.map(|index| raw(&prepared.map.plans[index], &[(PASSAGE_ID, chunk.text())],
            if mode == 1 { None } else { Some("é") }, &format!("map alternative {index}")))
    }).collect();
    let discovery = prepared.map.execute_with_driver(&admitted, Script { outputs, width: 32 }).unwrap();
    let mut remaining = remaining_verification(expected.discovery.verification, discovery.verification_used).unwrap();
    let evidence = evidence::collect(source, discovery.mapped.root().value(), request.limits,
        &mut remaining, &mut continuing()).unwrap();
    let (answer, prompt, tokens, planned, work, masks) = if mode == 1 {
        (SynthesisAnswer { status: SynthesisStatus::NoEvidenceCollected, answer: None, citations: Vec::new() },
            0, 0, Int4Work::default(), Int4Work::default(), 0)
    } else {
        let context = PlanContext::new(&prepared.identity, prepared.budget).unwrap();
        let req = SourceTaskRequest::Answer { question: prepared.question.clone(), passages: evidence.copy_passages().unwrap(),
            options: request.question.options, budget: prepared.budget };
        let plan = p.plan_int4_with_control(&req, &context, prepared.planning, 32, &mut continuing()).unwrap();
        let passages = evidence.passages.iter().map(|p| (p.id.as_str(), p.text.as_str())).collect::<Vec<_>>();
        let result = raw(&plan, &passages, if mode == 2 { None } else { Some("é") }, "final answer");
        let work = constrained_int4::planned_work(32, plan.prompt_tokens(), result.generated_token_ids.len()).unwrap();
        let native = Int4SourceTaskRun { schema_version: 1, execution: INT4_SOURCE_EXECUTION.to_owned(), group_width: 32,
            result: SourceTaskResult::Answer(result), model_work: work };
        check_run(&plan, &native, mapping().mask_visits_per_chunk).unwrap();
        let SourceTaskResult::Answer(result) = native.result else { panic!() };
        let tokens = result.generated_token_ids.len(); let masks = result.mask_node_visit_charge;
        let answer = evidence::final_answer(source, result, &evidence, request.question.options, &mut remaining, &mut continuing()).unwrap();
        (answer, plan.prompt_tokens(), tokens, plan.planned_work(), work, masks)
    };
    let run = finish(discovery, answer, evidence.passages.len(), evidence.bytes, prompt, tokens, planned, work, masks,
        remaining, expected, 16 << 20, &mut continuing()).unwrap();
    (expected, run)
}
fn decrement(work: &mut Int4Work, axis: usize) {
    match axis { 0 => work.forward_positions -= 1, 1 => work.projected_logits -= 1, 2 => work.attention_pairs -= 1,
        3 => work.projections.dot_products -= 1, _ => work.projections.multiply_accumulates -= 1 }
}

#[test]
fn pinned_preflight_matches_actual_compilation_for_each_group_partition() {
    let p = maps::planner(); let request = request(); let source = "é Alice Bob 上海"; let mut previous = None;
    for width in [16, 32, 127] {
        let prepared = prepare(&p, source, &request, mapping(), width, &mut continuing()).unwrap();
        let id = identity(&p, width); let budget = maps::budget();
        let expected = p.preflight_int4_question_synthesis_with_control(source, &request, budget,
            &PlanContext::new(&id, budget).unwrap(), SourcePlanningLimits::default(), mapping(), width, &mut continuing()).unwrap();
        assert_eq!(expected, prepared.expected); assert_eq!(prepared.group_width(), width);
        assert_eq!(expected.reserved_mask_visits().unwrap(), expected.discovery.reserved_mask_visits() + 1000);
        assert_eq!(expected.reserved_model_work().unwrap(), add_work(expected.discovery.planned_work(), expected.synthesis_reserve).unwrap());
        if let Some(old) = previous { assert!(old > expected.synthesis_reserve.projections.dot_products); }
        previous = Some(expected.synthesis_reserve.projections.dot_products);
    }
}
#[test]
fn final_reserve_is_not_renewed_and_each_work_axis_masks_refuse_one_unit_short() {
    let p = maps::planner(); let request = request(); let text = "Alice Bob";
    let expected = prepare(&p, text, &request, mapping(), 32, &mut continuing()).unwrap().expected;
    let mut exact = mapping(); exact.max_model_work = expected.reserved_model_work().unwrap();
    exact.max_mask_visits = expected.reserved_mask_visits().unwrap();
    prepare(&p, text, &request, exact, 32, &mut continuing()).unwrap();
    for axis in 0..6 {
        let mut cap = exact;
        if axis < 5 { decrement(&mut cap.max_model_work, axis); } else { cap.max_mask_visits -= 1; }
        assert!(prepare(&p, text, &request, cap, 32, &mut continuing()).is_err(), "{axis}");
    }
}
#[test]
fn whole_verification_deducts_map_fields_matches_and_scan_steps_before_synthesis() {
    let total = GroundingBudget { max_fields: 10, max_matches: 20, max_scan_steps: 30 };
    let used = SourceMapVerification { fields: 2, matches: 3, scan_steps: 4 };
    assert_eq!(remaining_verification(total, used).unwrap(), GroundingBudget { max_fields: 8, max_matches: 17, max_scan_steps: 26 });
    for axis in 0..3 {
        let mut cap = total;
        match axis { 0 => cap.max_fields = 1, 1 => cap.max_matches = 2, _ => cap.max_scan_steps = 3 }
        assert!(remaining_verification(cap, used).is_err());
    }
}
#[test]
fn changed_question_changes_every_actual_discovery_identity() {
    let p = maps::planner(); let a = request(); let mut b = request(); b.question.question = "Which person appears?".to_owned();
    let left = prepare(&p, "Alice Bob", &a, mapping(), 32, &mut continuing()).unwrap();
    let right = prepare(&p, "Alice Bob", &b, mapping(), 32, &mut continuing()).unwrap();
    for (a, b) in left.execution_identities().zip(right.execution_identities()) { assert_ne!(a.prompt_digest, b.prompt_digest); }
}
#[test]
fn invalid_evidence_capacity_profile_or_empty_source_cannot_prepare() {
    let p = maps::planner();
    for axis in 0..3 {
        let mut request = request();
        match axis { 0 => request.limits.max_evidence_passages = 0, 1 => request.limits.max_evidence_bytes = 0,
            _ => request.limits.max_evidence_passages = SourcePlanningLimits::default().max_passages + 1 }
        assert!(prepare(&p, "Alice", &request, mapping(), 32, &mut continuing()).is_err());
    }
    for text in ["", "  \r\n  "] { assert!(prepare(&p, text, &request(), mapping(), 32, &mut continuing()).is_err()); }
    let mut id = identity(&p, 32); id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 };
    assert!(p.plan_int4_question_synthesis_with_control("Alice", &request(), maps::budget(),
        &PlanContext::new(&id, maps::budget()).unwrap(), SourcePlanningLimits::default(), mapping(), 32, &mut continuing()).is_err());
}
#[test]
fn first_middle_last_planning_cancellation_remains_typed() {
    let p = maps::planner(); let mut control = continuing();
    prepare(&p, "Alice Bob", &request(), mapping(), 32, &mut control).unwrap();
    let last = control.calls;
    for stop in [1, last / 2, last] {
        let error = prepare(&p, "Alice Bob", &request(), mapping(), 32, &mut Control { calls: 0, stop }).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::PollQuota));
    }
}
#[test]
fn completed_answer_preserves_conflicting_maps_whitespace_and_original_unicode_occurrences() {
    let (expected, run) = synthetic(0); expected.verify_completed(&run).unwrap();
    assert_eq!(run.synthesis.status, SynthesisStatus::Answered); assert_eq!(run.discovery.distinct_answer_texts, 2);
    assert!(run.discovery.whitespace_chunks > 0); assert_eq!(run.synthesis.citations[0].spans.len(), 6);
    assert_eq!(run.synthesis.citations[0].spans.last().unwrap().byte_start, 34);
    assert!(run.model_work.forward_positions < run.reserved_model_work.forward_positions);
    assert!(run.verification_used.fields > run.discovery.verification_used.fields);
    let json = canonjson::canonical_string(&run).unwrap();
    assert!(!json.contains("generated_token_ids")); assert!(!json.contains("prompt_digest"));
}
#[test]
fn no_evidence_skips_final_native_work_without_refunding_reserve_or_hiding_maps() {
    let (expected, mut run) = synthetic(1); expected.verify_completed(&run).unwrap();
    assert_eq!(run.synthesis.status, SynthesisStatus::NoEvidenceCollected); assert_eq!(run.discovery.abstained_chunks, 2);
    assert_eq!(run.synthesis_generated_tokens, 0); assert_eq!(run.synthesis_model_work, Int4Work::default());
    assert!(run.synthesis_reserved_work.forward_positions > 0);
    run.synthesis_prompt_tokens = 1; assert!(expected.verify_completed(&run).is_err());
}
#[test]
fn final_abstention_has_positive_native_work_and_is_distinct_from_missing_evidence() {
    let (expected, mut run) = synthetic(2); expected.verify_completed(&run).unwrap();
    assert_eq!(run.synthesis.status, SynthesisStatus::Abstained); assert_eq!(run.discovery.answered_chunks, 2);
    assert!(run.evidence_passages > 0); assert!(run.synthesis_model_work.forward_positions > 0);
    run.synthesis.status = SynthesisStatus::NoEvidenceCollected;
    assert!(expected.verify_completed(&run).is_err());
}
#[test]
fn each_final_work_axis_is_recomputed_even_when_aggregate_tampering_matches() {
    for axis in 0..5 {
        let (expected, mut run) = synthetic(0); decrement(&mut run.synthesis_model_work, axis);
        run.model_work = add_work(run.discovery.model_work, run.synthesis_model_work).unwrap();
        assert!(expected.verify_completed(&run).is_err(), "{axis}");
    }
}
#[test]
fn group_token_counts_masks_grounding_and_nested_receipt_tampering_is_refused() {
    for axis in 0..7 {
        let (expected, mut run) = synthetic(0);
        match axis { 0 => run.group_width = 16, 1 => run.synthesis_prompt_tokens += 1,
            2 => run.synthesis_generated_tokens += 1, 3 => run.mask_node_visit_charge += 1,
            4 => run.verification_used.matches = 0, 5 => run.discovery.group_width = 16,
            _ => run.evidence_passages = 0 }
        assert!(expected.verify_completed(&run).is_err(), "{axis}");
    }
}
#[test]
fn invalid_answer_text_and_original_span_bounds_cannot_complete() {
    for axis in 0..3 {
        let (expected, mut run) = synthetic(0);
        match axis { 0 => run.synthesis.answer = Some(" ".to_owned()), 1 => run.synthesis.citations[0].occurrence = SourceOccurrence::Anchored,
            _ => run.synthesis.citations[0].spans[0].scalar_end = expected.discovery.source_span().scalar_end + 1 }
        assert!(expected.verify_completed(&run).is_err());
    }
}
#[test]
fn final_byte_limit_and_final_checkpoint_prevent_partial_publication() {
    for late_cancel in [false, true] {
        let (expected, run) = synthetic(0);
        let remaining = remaining_verification(expected.discovery.verification, run.verification_used).unwrap();
        let mut control = Control { calls: 0, stop: if late_cancel { 2 } else { usize::MAX } };
        let error = finish(run.discovery, run.synthesis, run.evidence_passages, run.evidence_bytes, run.synthesis_prompt_tokens,
            run.synthesis_generated_tokens, run.synthesis_planned_work, run.synthesis_model_work, run.synthesis_mask_node_visits,
            remaining, expected, if late_cancel { 16 << 20 } else { 1 }, &mut control).err().unwrap();
        if late_cancel { assert_eq!(error.cancellation(), Some(DecodeCancellationKind::PollQuota)); }
        else { assert!(matches!(error, Int4SourceMapError::Source(_))); }
    }
}
