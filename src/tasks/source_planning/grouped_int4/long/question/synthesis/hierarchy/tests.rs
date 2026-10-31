//! Pinned planning and private scripted reduction fixtures. These are not
//! native-model execution, answer quality, calibration or parity evidence.
use super::*;
use crate::{tasks::answer::PassageOccurrence,
    tasks::source_planning::grouped_int4::long::tests as maps};
use std::collections::VecDeque;
mod grouped;

pub(super) struct Control { calls: usize, stop: usize }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.calls >= self.stop).then_some(DecodeCancellationKind::Deadline)
    }
}
pub(super) fn go() -> Control { Control { calls: 0, stop: usize::MAX } }
fn request() -> SourceQuestionSynthesis {
    SourceQuestionSynthesis { question: SourceQuestion { question: "Who is mentioned?".to_owned(),
        options: AnswerOptions::default(), verification: GroundingBudget::default() },
        limits: QuestionSynthesisLimits { max_evidence_passages: 32, max_evidence_bytes: 4096 } }
}
pub(super) fn limits() -> QuestionHierarchyLimits { QuestionHierarchyLimits { max_passes: 4, ..QuestionHierarchyLimits::default() } }
fn identity(p: &SourceTaskPlanner, width: usize) -> ExecutionIdentity {
    let mut id = maps::identity(p, &SourceMapTask::Ner(NerOptions::default()), width);
    id.task_spec = ANSWER_TASK_VERSION.to_owned(); id
}
pub(super) fn prepare<'s>(p: &SourceTaskPlanner, source: &'s str, mapping: Int4SourceMapLimits,
    hierarchy: QuestionHierarchyLimits, width: usize, control: &mut Control)
    -> Result<PreparedInt4QuestionHierarchy<'s>, Int4SourceMapError> {
    let id = identity(p, width); let budget = maps::budget();
    p.plan_int4_question_hierarchy_with_control(source, &request(), hierarchy, budget, &PlanContext::new(&id, budget).unwrap(),
        SourcePlanningLimits::default(), mapping, width, control)
}
fn raw(width: usize, prompt: usize, eos: u32, ordinary: u32, passages: &[AnswerPassage], quote: Option<&str>, answer: &str) -> AnswerResult {
    let mut citations = Vec::new();
    if let Some(quote) = quote {
        let mut spans = Vec::new();
        for passage in passages {
            match scan_occurrences(&passage.text, quote, &mut GroundingBudget::default()) {
                Ok(found) => spans.extend(found.into_iter().map(|span| PassageOccurrence { passage_id: passage.id.clone(), span })),
                Err(FieldGroundingError::Absent) => (), Err(error) => panic!("{error}"),
            }
        }
        citations.push(PassageCitation { quote: quote.to_owned(),
            occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans });
    }
    let work = constrained_int4::planned_work(width, prompt, 2).unwrap();
    AnswerResult { schema_version: 1, task_spec_version: ANSWER_TASK_VERSION.to_owned(), numerics_profile: STRICT_INT4_PROFILE.to_owned(),
        status: if quote.is_some() { AnswerStatus::Answered } else { AnswerStatus::Abstained }, answerable: quote.is_some(),
        answer: quote.map(|_| answer.to_owned()), citations, calibration: AnswerCalibration::Uncalibrated,
        citation_guarantee: CitationGuarantee::StructuralSourceMembership, semantic_support: SummarySemanticSupport::NotAssessed,
        score_space: ScoreSpace::NotComputed, untrusted_fields: ["answer".to_owned(), "citations".to_owned()],
        generated_token_ids: vec![ordinary, eos], forward_positions: work.forward_positions,
        projected_logits: work.projected_logits, mask_node_visit_charge: 1 }
}
struct MapScript { outputs: VecDeque<AnswerResult> }
impl SourceDriver for MapScript {
    fn capacity(&self) -> usize { 8192 }
    fn group_width(&self) -> usize { 32 }
    fn clean(&self) -> bool { true }
    fn check_model(&self, _: &ExecutionIdentity) -> Result<(), Int4SourceError> { Ok(()) }
    fn checkpoint(&mut self) -> Result<(), Int4SourceError> { Ok(()) }
    fn run(&mut self, plan: &PreparedInt4SourceTask, _: &ExecutionIdentity) -> Result<Int4SourceTaskRun, Int4SourceError> {
        let raw = self.outputs.pop_front().ok_or(Int4SourceError::InvalidResult)?;
        let work = constrained_int4::planned_work(plan.group_width(), plan.prompt_tokens(), raw.generated_token_ids.len())?;
        Ok(Int4SourceTaskRun { schema_version: 1, execution: INT4_SOURCE_EXECUTION.to_owned(), group_width: 32,
            result: SourceTaskResult::Answer(raw), model_work: work })
    }
}
pub(super) const SOURCE: &str = "éabc----βdef";
fn setup(empty: bool) -> (Int4QuestionHierarchyPreflight, Int4QuestionRun, evidence::Collection, GroundingBudget) {
    let p = maps::planner(); let prepared = prepare(&p, SOURCE, maps::limits(), limits(), 32, &mut go()).unwrap();
    let mut expected = prepared.expected;
    // A private narrow grouping seam: one full quote fits; both original
    // quotes exceed it; the two selected one-scalar quotes fit exactly.
    expected.capacity.prompt = expected.capacity.scaffold + 247;
    let admitted = prepared.execution_identities().cloned().collect::<Vec<_>>();
    let mut outputs = VecDeque::new();
    assert_eq!(prepared.base.map.chunks.chunks().len(), 2);
    for ((chunk, index), quote) in prepared.base.map.chunks.chunks().iter().zip(&prepared.base.map.plan_indices).zip(["éabc", "βdef"]) {
        let plan = &prepared.base.map.plans[index.unwrap()]; let options = plan.extraction.options();
        let ordinary = (0..NANBEIGE_VOCAB_SIZE as u32)
            .find(|id| *id != options.eos_token_id && !options.excluded_token_ids.contains(id)).unwrap();
        outputs.push_back(raw(32, plan.prompt_tokens(), options.eos_token_id, ordinary,
            &[AnswerPassage { id: PASSAGE_ID.to_owned(), text: chunk.text().to_owned() }],
            if empty { None } else { Some(quote) }, &format!("MAP_ASSERTION_{}", chunk.id())));
    }
    let discovery = prepared.base.map.execute_with_driver(&admitted, MapScript { outputs }).unwrap();
    let mut remaining = remaining_verification(expected.base.discovery.verification, discovery.verification_used).unwrap();
    let evidence = evidence::collect(SOURCE, discovery.mapped.root().value(), expected.base.limits, &mut remaining, &mut go()).unwrap();
    (expected, discovery, evidence, remaining)
}
pub(super) struct Script {
    expected: Int4QuestionHierarchyPreflight, question: String,
    inputs: Vec<Vec<AnswerPassage>>, preparations: Vec<Vec<AnswerPassage>>, events: Vec<&'static str>,
    full: bool, empty: bool, final_abstain: bool, fail_prepare: Option<usize>, fail_run: Option<usize>,
    zero_tokens: bool, count_error: bool, wrong_witness: bool,
}
pub(super) fn script(expected: Int4QuestionHierarchyPreflight) -> Script {
    Script { expected, question: request().question.question, inputs: vec![], preparations: vec![], events: vec![],
        full: false, empty: false, final_abstain: false, fail_prepare: None, fail_run: None,
        zero_tokens: false, count_error: false, wrong_witness: false }
}
impl Script {
    fn binding(&self, passages: &[AnswerPassage], prompt: usize) -> Sha256Digest {
        Sha256Digest::of_bytes(&canonjson::canonical_bytes(&(&self.question, passages, prompt)).unwrap())
    }
}
impl execution::Driver for Script {
    type Witness = (Sha256Digest, usize);
    fn count<C: DecodeStepControl>(&mut self, _: &[AnswerPassage], bytes: usize, _: &mut C)
        -> Result<Option<usize>, Int4SourceMapError> {
        if self.count_error { return Err(Int4SourceMapError::Allocation); }
        let prompt = self.expected.capacity.scaffold + if self.zero_tokens { 0 } else { bytes };
        Ok((prompt <= self.expected.capacity.prompt).then_some(prompt))
    }
    fn prepare<C: DecodeStepControl>(&mut self, passages: &[AnswerPassage], prompt: usize, _: &mut C)
        -> Result<Self::Witness, Int4SourceMapError> {
        self.preparations.push(passages.to_vec()); self.events.push("prepare");
        if self.fail_prepare == Some(self.preparations.len()) { return Err(Int4SourceMapError::Admission); }
        Ok((if self.wrong_witness { Sha256Digest::of_bytes(b"wrong question") } else { self.binding(passages, prompt) }, prompt))
    }
    fn run<C: DecodeStepControl>(&mut self, passages: &[AnswerPassage], witness: &Self::Witness, _: &mut C)
        -> Result<(Int4SourceTaskRun, Int4Work), Int4SourceMapError> {
        if witness.0 != self.binding(passages, witness.1) { return Err(Int4SourceMapError::Admission); }
        self.inputs.push(passages.to_vec()); self.events.push("run");
        if self.fail_run == Some(self.inputs.len()) { return Err(Int4SourceError::Cancelled(DecodeCancellationKind::Deadline).into()); }
        let quote = if self.full { passages[0].text.clone() } else { passages[0].text.chars().next().unwrap().to_string() };
        let abstain = self.empty || (self.final_abstain && self.inputs.len() == 3);
        let raw = raw(self.expected.group_width(), witness.1, self.expected.capacity.eos, 1, passages,
            if abstain { None } else { Some(&quote) }, &format!("generated-{quote}"));
        let work = constrained_int4::planned_work(self.expected.group_width(), witness.1, raw.generated_token_ids.len()).unwrap();
        Ok((Int4SourceTaskRun { schema_version: 1, execution: INT4_SOURCE_EXECUTION.to_owned(),
            group_width: self.expected.group_width(), result: SourceTaskResult::Answer(raw), model_work: work },
            constrained_int4::planned_work(self.expected.group_width(), witness.1, self.expected.base.max_output_tokens).unwrap()))
    }
}
// mode0=answered, mode1=no initial evidence, mode2=intermediate exhaustion,
// mode3=final model abstention with a real final native pass.
pub(super) fn complete(mode: usize) -> (Int4QuestionHierarchyPreflight, Int4QuestionHierarchyRun) {
    let (expected, discovery, collection, mut remaining) = setup(mode == 1); let initial = EvidenceSize::of(&collection);
    let mut driver = script(expected); driver.empty = mode == 2; driver.final_abstain = mode == 3;
    let reduced = execution::reduce(SOURCE, collection, expected, 16 << 20, &mut remaining, &mut driver, &mut go()).unwrap();
    let run = finish(discovery, initial, reduced, remaining, expected, 16 << 20, &mut go()).unwrap();
    (expected, run)
}
fn decrement(work: &mut Int4Work, axis: usize) {
    match axis { 0 => work.forward_positions -= 1, 1 => work.projected_logits -= 1,
        2 => work.attention_pairs -= 1, 3 => work.projections.dot_products -= 1, _ => work.projections.multiply_accumulates -= 1 }
}

#[test]
fn pinned_preflight_equals_prepared_hierarchy_for_each_actual_group_partition() {
    let p = maps::planner(); let request = request(); let budget = maps::budget(); let mut previous = None;
    for width in [16, 32, 127] {
        let prepared = prepare(&p, SOURCE, maps::limits(), limits(), width, &mut go()).unwrap();
        let id = identity(&p, width);
        let expected = p.preflight_int4_question_hierarchy_with_control(SOURCE, &request, limits(), budget,
            &PlanContext::new(&id, budget).unwrap(), SourcePlanningLimits::default(), maps::limits(), width, &mut go()).unwrap();
        assert_eq!(expected, prepared.preflight_metadata()); assert_eq!(prepared.group_width(), width);
        assert_eq!(expected.discovery().chunk_count(), 2); assert_eq!(expected.reserved_mask_visits(), 6000);
        if let Some(old) = previous { assert!(old > expected.reserved_model_work().projections.dot_products); }
        previous = Some(expected.reserved_model_work().projections.dot_products);
    }
}
#[test]
fn every_permitted_pass_is_reserved_on_all_five_work_axes_and_masks() {
    let p = maps::planner(); let prepared = prepare(&p, SOURCE, maps::limits(), limits(), 32, &mut go()).unwrap();
    let expected = prepared.expected; let mut work = expected.discovery().planned_work();
    for _ in 0..4 { work = add_work(work, expected.base.synthesis_reserve).unwrap(); }
    assert_eq!(work, expected.reserved_model_work());
    let mut exact = maps::limits(); exact.max_model_work = work; exact.max_mask_visits = expected.reserved_mask_visits();
    prepare(&p, SOURCE, exact, limits(), 32, &mut go()).unwrap();
    for axis in 0..6 {
        let mut short = exact;
        if axis < 5 { decrement(&mut short.max_model_work, axis); } else { short.max_mask_visits -= 1; }
        assert!(prepare(&p, SOURCE, short, limits(), 32, &mut go()).is_err(), "{axis}");
    }
}
#[test]
fn complete_question_manifest_geometry_matches_actual_pinned_context_and_compiled_prompt() {
    let p = maps::planner(); let request = request();
    let expected = prepare(&p, SOURCE, maps::limits(), limits(), 32, &mut go()).unwrap().expected;
    let passages = vec![AnswerPassage { id: "evidence_0".to_owned(), text: "éabc".to_owned() },
        AnswerPassage { id: "evidence_1".to_owned(), text: "βdef".to_owned() }];
    let bytes = transport::context_bytes(request.question.question.len(), passages.iter().map(|p| (p.id.as_str(), p.text.as_str()))).unwrap();
    assert_eq!(bytes, 255);
    let encoded = AnswerContext::encode(p.source_encoder(), &request.question.question, &passages,
        AnswerInputLimits { max_passages: 32, max_input_bytes: 4096, max_input_tokens: 8192 }).unwrap();
    assert_eq!(encoded.document().total_token_count(), bytes);
    assert!(AnswerContext::encode(p.source_encoder(), &request.question.question, &passages,
        AnswerInputLimits { max_passages: 32, max_input_bytes: 4096, max_input_tokens: bytes - 1 }).is_err());
    let id = identity(&p, 32); let budget = maps::budget();
    let plan = p.plan_int4_with_control(&SourceTaskRequest::Answer { question: request.question.question,
        passages, options: request.question.options, budget }, &PlanContext::new(&id, budget).unwrap(),
        SourcePlanningLimits::default(), 32, &mut go()).unwrap();
    assert_eq!(plan.prompt_tokens(), expected.capacity.scaffold + bytes);
}
#[test]
fn multilevel_reduction_retains_original_question_and_never_feeds_generated_answers() {
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = script(expected);
    let reduced = execution::reduce(SOURCE, collection, expected, 16 << 20, &mut remaining, &mut driver, &mut go()).unwrap();
    let inputs = driver.inputs.iter().map(|ps| ps.iter().map(|p| p.text.as_str()).collect::<Vec<_>>()).collect::<Vec<_>>();
    assert_eq!(inputs, [vec!["éabc"], vec!["βdef"], vec!["é", "β"]]);
    assert_eq!(driver.question, "Who is mentioned?");
    assert_eq!(driver.events, ["prepare", "prepare", "run", "run", "prepare", "run"]);
    assert_eq!(reduced.levels.len(), 2); assert_eq!(reduced.final_pass, Some(2));
    assert_eq!(reduced.status, QuestionHierarchyStatus::Answered);
    assert_eq!(reduced.tokenizer, HierarchyTokenizerWork { calls: 5, bytes: 970 });
    assert_eq!(reduced.passes[2].answer.citations[0].spans[0], VerifiedSourceSpan { byte_start: 0, byte_end: 2, scalar_start: 0, scalar_end: 1 });
}
#[test]
fn complete_level_last_group_admission_failure_prevents_every_native_call() {
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = script(expected); driver.fail_prepare = Some(2);
    assert!(matches!(execution::reduce(SOURCE, collection, expected, 16 << 20,
        &mut remaining, &mut driver, &mut go()), Err(Int4SourceMapError::Admission)));
    assert_eq!(driver.preparations.len(), 2); assert!(driver.inputs.is_empty());
}
#[test]
fn changed_private_witness_prevents_native_work_after_complete_level_preparation() {
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = script(expected); driver.wrong_witness = true;
    assert!(matches!(execution::reduce(SOURCE, collection, expected, 16 << 20,
        &mut remaining, &mut driver, &mut go()), Err(Int4SourceMapError::Admission)));
    assert_eq!(driver.preparations.len(), 2); assert!(driver.inputs.is_empty());
}
#[test]
fn unchanged_evidence_refuses_nonprogress_without_a_final_native_pass() {
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = script(expected); driver.full = true;
    assert!(execution::reduce(SOURCE, collection, expected, 16 << 20, &mut remaining, &mut driver, &mut go()).is_err());
    assert_eq!(driver.inputs.len(), 2);
}
#[test]
fn shared_levels_passes_and_complete_context_tokenizer_work_never_renew() {
    for axis in 0..4 {
        let (mut expected, _, collection, mut remaining) = setup(false);
        match axis { 0 => expected.limits.max_levels = 1, 1 => expected.limits.max_passes = 2,
            2 => expected.limits.max_tokenizer_calls = 4, _ => expected.limits.max_tokenizer_bytes = 969 }
        let mut driver = script(expected);
        assert!(execution::reduce(SOURCE, collection, expected, 16 << 20, &mut remaining, &mut driver, &mut go()).is_err());
        assert_eq!(driver.inputs.len(), if axis < 2 { 0 } else { 2 });
    }
}
#[test]
fn atomic_quotation_zero_tokens_and_unrelated_counter_errors_refuse_without_native_work() {
    for axis in 0..3 {
        let (mut expected, _, collection, mut remaining) = setup(false);
        if axis == 0 { expected.capacity.prompt = expected.capacity.scaffold + 156; }
        let mut driver = script(expected); driver.zero_tokens = axis == 1; driver.count_error = axis == 2;
        let error = execution::reduce(SOURCE, collection, expected, 16 << 20, &mut remaining, &mut driver, &mut go()).err().unwrap();
        if axis == 2 { assert!(matches!(error, Int4SourceMapError::Allocation)); }
        assert!(driver.inputs.is_empty()); assert!(driver.preparations.is_empty());
    }
}
#[test]
fn no_initial_evidence_intermediate_exhaustion_and_final_abstention_remain_distinct() {
    for (mode, status, passes, final_pass) in [(1, QuestionHierarchyStatus::NoEvidenceCollected, 0, None),
        (2, QuestionHierarchyStatus::EvidenceExhausted, 2, None), (3, QuestionHierarchyStatus::Abstained, 3, Some(2))] {
        let (expected, run) = complete(mode); expected.verify_completed(&run).unwrap();
        assert_eq!(run.status, status); assert_eq!(run.passes.len(), passes); assert_eq!(run.final_pass, final_pass);
        assert_eq!(run.answer().is_some(), mode == 3); assert_eq!(run.discovery.native_chunks, 2);
        assert_eq!(run.discovery.abstained_chunks, if mode == 1 { 2 } else { 0 });
        if mode != 1 { assert!(run.passes.iter().any(|pass| pass.answer.status == SynthesisStatus::Abstained)); }
    }
}
#[test]
fn completed_receipt_keeps_all_maps_native_passes_final_index_and_cumulative_verification() {
    let (expected, run) = complete(0); expected.verify_completed(&run).unwrap();
    assert_eq!(run.discovery.distinct_answer_texts, 2); assert_eq!(run.discovery.mapped.root().value().chunks().len(), 2);
    assert_eq!(run.passes.len(), 3); assert_eq!(run.final_pass, Some(2));
    assert_eq!(run.answer().unwrap().answer.as_deref(), Some("generated-é"));
    assert_eq!(run.initial_evidence, EvidenceSize { passages: 2, bytes: 10 });
    assert_eq!(run.levels[0].next_evidence, Some(EvidenceSize { passages: 2, bytes: 4 }));
    assert!(run.verification_used.fields > run.discovery.verification_used.fields);
    assert!(run.verification_used.matches > run.discovery.verification_used.matches);
    assert!(run.verification_used.scan_steps > run.discovery.verification_used.scan_steps);
    assert!(run.model_work.forward_positions < run.reserved_model_work.forward_positions);
    let json = canonjson::canonical_string(&run).unwrap();
    assert!(json.contains("generated_token_ids")); assert!(!json.contains("prompt_digest"));
}
#[test]
fn each_native_work_axis_group_prompt_eos_and_original_bounds_are_independently_checked() {
    for axis in 0..10 {
        let (expected, mut run) = complete(0); let pass = &mut run.passes[2];
        if axis < 5 { decrement(&mut pass.native.model_work, axis); }
        else if axis == 5 { pass.native.group_width = 16; }
        else if axis == 6 { pass.prompt_tokens += 1; }
        else if axis == 7 {
            let SourceTaskResult::Answer(raw) = &mut pass.native.result else { panic!() }; raw.generated_token_ids.pop();
        } else if axis == 8 { pass.answer.citations[0].spans[0].byte_end = SOURCE.len() + 1; }
        else { pass.answer.answer = Some("different lifted assertion".to_owned()); }
        assert!(receipt::verify_pass(&expected, pass).is_err(), "{axis}");
    }
}
#[test]
fn complete_frontier_coverage_context_manifest_geometry_and_final_index_cannot_be_rewritten() {
    for axis in 0..8 {
        let (expected, mut run) = complete(0);
        match axis { 0 => run.passes[1].input_passages.start = 0, 1 => run.passes[0].group = 1,
            2 => run.passes[0].input_bytes += 1, 3 => run.passes[0].context_bytes += 1,
            4 => run.levels[0].next_evidence.as_mut().unwrap().bytes += 1, 5 => run.levels[1].passes.start = 1,
            6 => run.levels[1].input.passages += 1, _ => run.final_pass = Some(1) }
        assert!(expected.verify_completed(&run).is_err(), "{axis}");
    }
}
#[test]
fn local_passage_ids_scalar_offsets_duplicates_and_separator_membership_are_checked() {
    for axis in 0..4 {
        let (expected, mut run) = complete(0);
        let SourceTaskResult::Answer(raw) = &mut run.passes[2].native.result else { panic!() };
        let citation = &mut raw.citations[0];
        match axis { 0 => citation.spans[0].passage_id = "question".to_owned(),
            1 => citation.spans[0].span.scalar_end += 1,
            2 => { citation.spans.push(citation.spans[0].clone()); citation.occurrence = SourceOccurrence::Ambiguous; },
            _ => { citation.spans[0].span.byte_start = 1; citation.spans[0].span.byte_end = 3; } }
        assert!(expected.verify_completed(&run).is_err(), "{axis}");
    }
}
#[test]
fn aggregate_native_work_masks_verification_initial_evidence_and_status_remain_consistent() {
    for axis in 0..9 {
        let (expected, mut run) = complete(0);
        match axis { 0 => run.group_width = 16, 1 => run.discovery.group_width = 16,
            2 => run.model_work.projections.dot_products += 1, 3 => run.mask_node_visit_charge += 1,
            4 => run.verification_used.scan_steps = 0, 5 => run.initial_evidence.passages += 1,
            6 => run.tokenizer_work.calls = 0, 7 => run.tokenizer_work.bytes = 0,
            _ => run.status = QuestionHierarchyStatus::EvidenceExhausted }
        assert!(expected.verify_completed(&run).is_err(), "{axis}");
    }
}
#[test]
fn all_three_verification_budgets_and_output_limit_apply_across_every_level() {
    let (expected, _, collection, start) = setup(false); let mut remaining = start;
    execution::reduce(SOURCE, collection, expected, 16 << 20, &mut remaining, &mut script(expected), &mut go()).unwrap();
    let used = SourceMapVerification::used(start, remaining).unwrap();
    for axis in 0..4 {
        let (expected, _, collection, mut remaining) = setup(false);
        match axis { 0 => remaining.max_fields = used.fields - 1, 1 => remaining.max_matches = used.matches - 1,
            2 => remaining.max_scan_steps = used.scan_steps - 1, _ => () }
        assert!(execution::reduce(SOURCE, collection, expected, if axis == 3 { 1 } else { 16 << 20 },
            &mut remaining, &mut script(expected), &mut go()).is_err(), "{axis}");
    }
}
#[test]
fn later_native_failure_and_last_reduction_checkpoint_publish_no_partial_success() {
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = script(expected); driver.fail_run = Some(2);
    let error = execution::reduce(SOURCE, collection, expected, 16 << 20, &mut remaining, &mut driver, &mut go()).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(driver.inputs.len(), 2);
    let (expected, _, collection, mut remaining) = setup(false); let mut count = go();
    execution::reduce(SOURCE, collection, expected, 16 << 20, &mut remaining, &mut script(expected), &mut count).unwrap();
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = script(expected);
    let error = execution::reduce(SOURCE, collection, expected, 16 << 20,
        &mut remaining, &mut driver, &mut Control { calls: 0, stop: count.calls }).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(driver.inputs.len(), 3);
}
#[test]
fn evidence_windows_preserve_ids_original_unicode_origins_and_exact_byte_count() {
    let (_, _, collection, _) = setup(false); let window = collection.window(1..2, &mut go()).unwrap();
    assert_eq!(window.passages.len(), 1); assert_eq!(window.passages[0].id, "evidence_1");
    assert_eq!(window.passages[0].text, "βdef"); assert_eq!(window.bytes, 5);
    assert_eq!(window.origins[0][0], VerifiedSourceSpan { byte_start: 9, byte_end: 14, scalar_start: 8, scalar_end: 12 });
    assert!(collection.window(1..3, &mut go()).is_err()); assert!(collection.window(1..1, &mut go()).is_err());
}
#[test]
fn transport_checks_every_duplicate_origin_before_dedup_and_charges_each_copy() {
    let (expected, _, _, _) = setup(false);
    let citation = SourceCitation { quote: "é".to_owned(), occurrence: SourceOccurrence::Anchored,
        spans: vec![VerifiedSourceSpan { byte_start: 0, byte_end: 2, scalar_start: 0, scalar_end: 1 }] };
    let mut answer = SynthesisAnswer { status: SynthesisStatus::Answered, answer: Some("not evidence".to_owned()),
        citations: vec![citation.clone(), citation] };
    let mut budget = GroundingBudget { max_fields: 2, max_matches: 2, max_scan_steps: 4 };
    let mut collection = evidence::Collection::empty();
    collection.append_verified(SOURCE, &answer, expected, &mut budget, &mut go()).unwrap();
    assert_eq!(collection.passages.len(), 1); assert_eq!(collection.passages[0].text, "é");
    assert_eq!(budget, GroundingBudget { max_fields: 0, max_matches: 0, max_scan_steps: 0 });
    answer.citations[1].spans[0] = VerifiedSourceSpan { byte_start: 9, byte_end: 11, scalar_start: 8, scalar_end: 9 };
    assert!(evidence::Collection::empty().append_verified(SOURCE, &answer, expected,
        &mut GroundingBudget::default(), &mut go()).is_err());
}
#[test]
fn final_publication_size_and_last_checkpoint_apply_after_complete_reduction() {
    for late in [false, true] {
        let (expected, discovery, collection, mut remaining) = setup(false); let initial = EvidenceSize::of(&collection);
        let reduced = execution::reduce(SOURCE, collection, expected, 16 << 20,
            &mut remaining, &mut script(expected), &mut go()).unwrap();
        let error = finish(discovery, initial, reduced, remaining, expected, if late { 16 << 20 } else { 1 },
            &mut Control { calls: 0, stop: if late { 2 } else { usize::MAX } }).err().unwrap();
        if late { assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); }
        else { assert!(matches!(error, Int4SourceMapError::Source(_))); }
    }
}
#[test]
fn actual_compiled_witness_binds_question_and_manifest_even_with_identical_quote_bytes() {
    let p = maps::planner(); let id = identity(&p, 32); let budget = maps::budget();
    let build = |question: &str, passage_id: &str| {
        let request = SourceTaskRequest::Answer { question: question.to_owned(),
            passages: vec![AnswerPassage { id: passage_id.to_owned(), text: "éabc".to_owned() }],
            options: AnswerOptions::default(), budget };
        let plan = p.plan_int4_with_control(&request, &PlanContext::new(&id, budget).unwrap(),
            SourcePlanningLimits::default(), 32, &mut go()).unwrap();
        execution::Witness::of(&plan).unwrap()
    };
    assert!(build("Who?", "evidence_0") != build("Why?", "evidence_0"));
    assert!(build("Who?", "evidence_0") != build("Who?", "evidence_1"));
}
#[test]
fn invalid_hierarchy_limits_and_first_preparation_cancellation_fail_closed() {
    let p = maps::planner();
    for axis in 0..4 {
        let mut bad = limits();
        match axis { 0 => bad.max_levels = 0, 1 => bad.max_passes = 257,
            2 => bad.max_tokenizer_calls = 0, _ => bad.max_tokenizer_bytes = 0 }
        assert!(prepare(&p, SOURCE, maps::limits(), bad, 32, &mut go()).is_err());
    }
    let error = prepare(&p, SOURCE, maps::limits(), limits(), 32, &mut Control { calls: 0, stop: 1 }).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
}
