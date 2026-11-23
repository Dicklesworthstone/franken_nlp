//! Pinned planning and PRIVATE scripted receipts. These are regression
//! definitions, not model inference, answer quality or performance evidence.
use super::*;
use crate::{native_engine::strict_int8::STRICT_INT8_EXECUTION, tasks::answer::PassageOccurrence,
    tokenizer::pinned_controls, validation::grounded_fields::FieldGroundingError};
use std::collections::VecDeque;

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn planner() -> SourceTaskPlanner {
    let controls = pinned_controls::pinned().unwrap();
    let eos = controls.template_controls().entries().iter().find(|entry| entry.special && entry.surface == IM_END).unwrap().id;
    SourceTaskPlanner::pinned(controls.template_controls(), eos).unwrap()
}
fn identity(planner: &SourceTaskPlanner) -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"private int8 answer hierarchy fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: digest,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: digest,
        tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(), task_spec: ANSWER_TASK_VERSION.to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "none".to_owned(), schema_digest: digest,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn mapping() -> Int8SourceMapLimits {
    Int8SourceMapLimits { chunks: ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 8, max_chunk_tokens: 8,
        context_tokens: 8192, reserved_tokens: 64, max_chunks: 256, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { map_batch_chunks: 2, reduce_fan_in: 2,
            max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
        max_model_work: Int8Work { forward_positions: u64::MAX, projected_logits: u64::MAX, attention_pairs: u64::MAX,
            projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 512_000 }
}
fn request() -> SourceQuestionSynthesis {
    SourceQuestionSynthesis { question: SourceQuestion { question: "Who is named?".to_owned(),
        options: AnswerOptions::default(), verification: GroundingBudget::default() },
        limits: QuestionSynthesisLimits { max_evidence_passages: 32, max_evidence_bytes: 8192 } }
}
fn hierarchy() -> QuestionHierarchyLimits { QuestionHierarchyLimits { max_levels: 4, max_passes: 8, ..Default::default() } }
fn prepare<'s>(planner: &SourceTaskPlanner, source: &'s str, request: &SourceQuestionSynthesis)
    -> PreparedInt8QuestionHierarchy<'s> {
    let identity = identity(planner);
    planner.plan_int8_question_hierarchy_with_control(source, request, hierarchy(), budget(),
        &PlanContext::new(&identity, budget()).unwrap(), SourcePlanningLimits::default(), mapping(), &mut Continue).unwrap()
}

#[test]
fn lightweight_preflight_matches_actual_discovery_and_reserves_all_hierarchy_calls() {
    let planner = planner(); let request = request(); let identity = identity(&planner);
    let source = "Alice é 上海 <tool_call>\r\n".repeat(5);
    let prepared = prepare(&planner, &source, &request); let expected = prepared.preflight_metadata();
    let preflight = planner.preflight_int8_question_hierarchy_with_control(&source, &request, hierarchy(), budget(),
        &PlanContext::new(&identity, budget()).unwrap(), SourcePlanningLimits::default(), mapping(), &mut Continue).unwrap();
    assert_eq!(expected, preflight); assert!(preflight.discovery().chunk_count() > 1);
    assert_eq!(prepared.execution_identities().len(), preflight.discovery().native_chunks());
    let mut work = preflight.discovery().planned_work();
    for _ in 0..hierarchy().max_passes { work = add_work(work, preflight.base.synthesis_reserve).unwrap(); }
    assert_eq!(preflight.reserved_model_work(), work);
    assert_eq!(preflight.reserved_mask_visits(), preflight.discovery().reserved_mask_visits()
        + hierarchy().max_passes as u64 * mapping().mask_visits_per_chunk);
}
#[test]
fn every_whole_hierarchy_axis_refuses_one_unit_below_the_complete_reservation() {
    let planner = planner(); let request = request(); let identity = identity(&planner);
    let context = PlanContext::new(&identity, budget()).unwrap();
    let expected = planner.preflight_int8_question_hierarchy_with_control("Alice", &request, hierarchy(), budget(),
        &context, SourcePlanningLimits::default(), mapping(), &mut Continue).unwrap();
    let mut exact = mapping(); exact.max_model_work = expected.reserved_model_work();
    exact.max_mask_visits = expected.reserved_mask_visits();
    planner.preflight_int8_question_hierarchy_with_control("Alice", &request, hierarchy(), budget(),
        &context, SourcePlanningLimits::default(), exact, &mut Continue).unwrap();
    for axis in 0..6 {
        let mut bad = exact;
        match axis { 0 => bad.max_model_work.forward_positions -= 1, 1 => bad.max_model_work.projected_logits -= 1,
            2 => bad.max_model_work.attention_pairs -= 1, 3 => bad.max_model_work.projections.dot_products -= 1,
            4 => bad.max_model_work.projections.multiply_accumulates -= 1, _ => bad.max_mask_visits -= 1 }
        assert!(planner.preflight_int8_question_hierarchy_with_control("Alice", &request, hierarchy(), budget(),
            &context, SourcePlanningLimits::default(), bad, &mut Continue).is_err(), "{axis}");
    }
}

const SOURCE: &str = "éabc----βdef";
fn raw(prompt: usize, eos: u32, ordinary: u32, passages: &[AnswerPassage], quote: Option<&str>, answer: &str) -> AnswerResult {
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
    let work = constrained_int8::planned_work(prompt, 2).unwrap();
    AnswerResult { schema_version: 1, task_spec_version: ANSWER_TASK_VERSION.to_owned(), numerics_profile: STRICT_INT8_PROFILE.to_owned(),
        status: if quote.is_some() { AnswerStatus::Answered } else { AnswerStatus::Abstained }, answerable: quote.is_some(),
        answer: quote.map(|_| answer.to_owned()), citations, calibration: AnswerCalibration::Uncalibrated,
        citation_guarantee: CitationGuarantee::StructuralSourceMembership, semantic_support: SummarySemanticSupport::NotAssessed,
        score_space: ScoreSpace::NotComputed, untrusted_fields: ["answer".to_owned(), "citations".to_owned()],
        generated_token_ids: vec![ordinary, eos], forward_positions: work.forward_positions,
        projected_logits: work.projected_logits, mask_node_visit_charge: 1 }
}
struct MapScript { outputs: VecDeque<AnswerResult> }
impl SourceDriver for MapScript {
    fn checkpoint(&mut self) -> Result<(), Int8SourceError> { Ok(()) }
    fn run(&mut self, plan: &PreparedInt8SourceTask, _: &ExecutionIdentity) -> Result<Int8SourceTaskRun, Int8SourceError> {
        let raw = self.outputs.pop_front().ok_or(Int8SourceError::InvalidResult)?;
        let work = constrained_int8::planned_work(plan.prompt_tokens(), raw.generated_token_ids.len())?;
        Ok(Int8SourceTaskRun { schema_version: 1, execution: INT8_SOURCE_EXECUTION.to_owned(),
            result: SourceTaskResult::Answer(raw), model_work: work })
    }
}
fn setup(empty: bool) -> (Int8QuestionHierarchyPreflight, Int8QuestionRun, evidence::Collection, GroundingBudget) {
    let planner = planner(); let prepared = prepare(&planner, SOURCE, &request());
    let mut expected = prepared.expected;
    // Narrow only the private reduction fixture: each original quote fits,
    // their combined context does not, and the two selected scalars fit exactly.
    let next = [AnswerPassage { id: "evidence_0".to_owned(), text: "é".to_owned() },
        AnswerPassage { id: "evidence_1".to_owned(), text: "β".to_owned() }];
    expected.capacity.prompt = expected.capacity.scaffold + transport::context_bytes(expected.capacity.question_bytes,
        next.iter().map(|passage| (passage.id.as_str(), passage.text.as_str())), &mut Continue).unwrap();
    let admitted = prepared.execution_identities().cloned().collect::<Vec<_>>();
    let mut outputs = VecDeque::new();
    assert_eq!(prepared.base.map.chunks.chunks().len(), 2);
    for ((chunk, index), quote) in prepared.base.map.chunks.chunks().iter().zip(&prepared.base.map.plan_indices).zip(["éabc", "βdef"]) {
        let plan = &prepared.base.map.plans[index.unwrap()]; let options = plan.extraction.options();
        let ordinary = (0..NANBEIGE_VOCAB_SIZE as u32)
            .find(|id| *id != options.eos_token_id && !options.excluded_token_ids.contains(id)).unwrap();
        outputs.push_back(raw(plan.prompt_tokens(), options.eos_token_id, ordinary,
            &[AnswerPassage { id: PASSAGE_ID.to_owned(), text: chunk.text().to_owned() }],
            if empty { None } else { Some(quote) }, &format!("MAP_ASSERTION_{}", chunk.id())));
    }
    let discovery = prepared.base.map.execute_with_driver(&admitted, MapScript { outputs }).unwrap();
    let mut remaining = remaining_verification(&discovery, expected.base.discovery.verification, &mut Continue).unwrap();
    let evidence = evidence::collect(SOURCE, discovery.mapped.root().value(), expected.base.limits, &mut remaining, &mut Continue).unwrap();
    (expected, discovery, evidence, remaining)
}
struct Script {
    expected: Int8QuestionHierarchyPreflight, question: String,
    inputs: Vec<Vec<AnswerPassage>>, preparations: Vec<Vec<AnswerPassage>>, events: Vec<&'static str>,
    full: bool, empty: bool, final_abstain: bool, fail_prepare: Option<usize>, fail_run: Option<usize>,
    zero_tokens: bool, count_error: bool, wrong_witness: bool,
}
fn script(expected: Int8QuestionHierarchyPreflight) -> Script {
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
    fn count<C: DecodeStepControl>(&mut self, _: &[AnswerPassage], bytes: usize, control: &mut C)
        -> Result<Option<usize>, Int8SourceMapError> {
        checkpoint(control)?;
        if self.count_error { return Err(Int8SourceMapError::Allocation); }
        let prompt = self.expected.capacity.scaffold + if self.zero_tokens { 0 } else { bytes };
        Ok((prompt <= self.expected.capacity.prompt).then_some(prompt))
    }
    fn prepare<C: DecodeStepControl>(&mut self, passages: &[AnswerPassage], prompt: usize, control: &mut C)
        -> Result<Self::Witness, Int8SourceMapError> {
        checkpoint(control)?;
        self.preparations.push(passages.to_vec()); self.events.push("prepare");
        if self.fail_prepare == Some(self.preparations.len()) { return Err(Int8SourceMapError::Admission); }
        Ok((if self.wrong_witness { Sha256Digest::of_bytes(b"wrong question") } else { self.binding(passages, prompt) }, prompt))
    }
    fn run<C: DecodeStepControl>(&mut self, passages: &[AnswerPassage], witness: &Self::Witness, control: &mut C)
        -> Result<(Int8SourceTaskRun, Int8Work), Int8SourceMapError> {
        checkpoint(control)?;
        if witness.0 != self.binding(passages, witness.1) { return Err(Int8SourceMapError::Admission); }
        self.inputs.push(passages.to_vec()); self.events.push("run");
        if self.fail_run == Some(self.inputs.len()) { return Err(Int8SourceError::Cancelled(DecodeCancellationKind::Deadline).into()); }
        let quote = if self.full { passages[0].text.clone() } else { passages[0].text.chars().next().unwrap().to_string() };
        let abstain = self.empty || (self.final_abstain && self.inputs.len() == 3);
        let raw = raw(witness.1, self.expected.capacity.eos, 1, passages,
            if abstain { None } else { Some(&quote) }, &format!("GENERATED_ASSERTION_{quote}"));
        let work = constrained_int8::planned_work(witness.1, raw.generated_token_ids.len()).unwrap();
        Ok((Int8SourceTaskRun { schema_version: 1, execution: INT8_SOURCE_EXECUTION.to_owned(),
            result: SourceTaskResult::Answer(raw), model_work: work },
            constrained_int8::planned_work(witness.1, self.expected.output_tokens).unwrap()))
    }
}
// 0=answer, 1=no initial evidence, 2=intermediate exhaustion, 3=final abstention.
fn complete(mode: usize) -> (Int8QuestionHierarchyPreflight, Int8QuestionHierarchyRun) {
    let (expected, discovery, collection, mut remaining) = setup(mode == 1); let initial = EvidenceSize::of(&collection);
    let mut driver = script(expected); driver.empty = mode == 2; driver.final_abstain = mode == 3;
    let reduced = execution::reduce(SOURCE, collection, expected, expected.max_result_bytes, &mut remaining, &mut driver, &mut Continue).unwrap();
    let run = finish(discovery, initial, reduced, remaining, expected, &mut Continue).unwrap();
    (expected, run)
}

#[test]
fn multilevel_reduction_preserves_original_question_and_never_feeds_generated_answers() {
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = script(expected);
    let reduced = execution::reduce(SOURCE, collection, expected, expected.max_result_bytes, &mut remaining, &mut driver, &mut Continue).unwrap();
    let inputs = driver.inputs.iter().map(|passages| passages.iter().map(|p| p.text.as_str()).collect::<Vec<_>>()).collect::<Vec<_>>();
    assert_eq!(inputs, [vec!["éabc"], vec!["βdef"], vec!["é", "β"]]);
    assert_eq!(driver.question, request().question.question);
    assert_eq!(driver.events, ["prepare", "prepare", "run", "run", "prepare", "run"]);
    assert_eq!(reduced.levels.len(), 2); assert_eq!(reduced.final_pass, Some(2));
    assert_eq!(reduced.status, QuestionHierarchyStatus::Answered); assert_eq!(reduced.tokenizer.calls, 5);
    assert_eq!(reduced.passes[2].answer.citations[0].spans[0],
        VerifiedSourceSpan { byte_start: 0, byte_end: 2, scalar_start: 0, scalar_end: 1 });
}
#[test]
fn all_group_plans_and_witnesses_must_be_admitted_before_the_first_native_call() {
    for wrong_witness in [false, true] {
        let (expected, _, collection, mut remaining) = setup(false); let mut driver = script(expected);
        if wrong_witness { driver.wrong_witness = true; } else { driver.fail_prepare = Some(2); }
        assert!(matches!(execution::reduce(SOURCE, collection, expected, expected.max_result_bytes,
            &mut remaining, &mut driver, &mut Continue), Err(Int8SourceMapError::Admission)));
        assert_eq!(driver.preparations.len(), 2); assert!(driver.inputs.is_empty());
    }
}
#[test]
fn nonprogress_atomic_quotation_and_counter_errors_cannot_become_partial_success() {
    for axis in 0..4 {
        let (mut expected, _, collection, mut remaining) = setup(false);
        if axis == 1 { expected.capacity.prompt = expected.capacity.scaffold + 1; }
        let mut driver = script(expected); driver.full = axis == 0; driver.zero_tokens = axis == 2; driver.count_error = axis == 3;
        let error = execution::reduce(SOURCE, collection, expected, expected.max_result_bytes,
            &mut remaining, &mut driver, &mut Continue).err().unwrap();
        if axis == 3 { assert!(matches!(error, Int8SourceMapError::Allocation)); }
        assert_eq!(driver.inputs.len(), if axis == 0 { 2 } else { 0 });
    }
}
#[test]
fn no_evidence_exhausted_selection_and_final_abstention_are_distinct_completed_states() {
    for (mode, status, passes, final_pass) in [(1, QuestionHierarchyStatus::NoEvidenceCollected, 0, None),
        (2, QuestionHierarchyStatus::EvidenceExhausted, 2, None), (3, QuestionHierarchyStatus::Abstained, 3, Some(2))] {
        let (expected, run) = complete(mode); expected.verify_completed(&run).unwrap();
        assert_eq!(run.status, status); assert_eq!(run.passes.len(), passes); assert_eq!(run.final_pass, final_pass);
        assert_eq!(run.answer().is_some(), mode == 3); assert_eq!(run.discovery.native_chunks, 2);
        assert_eq!(run.discovery.abstained_chunks, if mode == 1 { 2 } else { 0 });
    }
}
#[test]
fn complete_result_retains_all_discovery_alternatives_and_every_native_pass() {
    let (expected, run) = complete(0); expected.verify_completed(&run).unwrap();
    assert_eq!(run.discovery.distinct_answer_texts, 2);
    assert_eq!(run.discovery.mapped.root().value().chunks().len(), 2);
    assert_eq!(run.passes.len(), 3); assert_eq!(run.final_pass, Some(2));
    assert_eq!(run.answer().unwrap().answer.as_deref(), Some("GENERATED_ASSERTION_é"));
    assert_eq!(run.initial_evidence, EvidenceSize { passages: 2, bytes: 10 });
    assert_eq!(run.levels[0].next_evidence, Some(EvidenceSize { passages: 2, bytes: 4 }));
    assert!(run.verification_used.scan_steps > run.discovery.verification_scan_steps);
    assert!(run.model_work.forward_positions < run.reserved_model_work.forward_positions);
    let json = canonjson::canonical_string(&run).unwrap();
    assert!(json.contains("generated_token_ids"));
    for private in ["request_binding", "result_binding", "prompt_digest", "logical_model_digest", "Who is named?"] {
        assert!(!json.contains(private)); assert!(!format!("{expected:?}").contains(private));
    }
}
#[test]
fn hierarchy_and_verification_limits_apply_to_the_entire_reduction() {
    let (expected, _, collection, before) = setup(false); let mut remaining = before;
    let reduced = execution::reduce(SOURCE, collection, expected, expected.max_result_bytes,
        &mut remaining, &mut script(expected), &mut Continue).unwrap();
    let used = QuestionVerificationWork { fields: before.max_fields - remaining.max_fields,
        matches: before.max_matches - remaining.max_matches, scan_steps: before.max_scan_steps - remaining.max_scan_steps };
    for axis in 0..8 {
        let (mut expected, _, collection, mut remaining) = setup(false);
        match axis { 0 => expected.limits.max_levels = 1, 1 => expected.limits.max_passes = 2,
            2 => expected.limits.max_tokenizer_calls = reduced.tokenizer.calls - 1,
            3 => expected.limits.max_tokenizer_bytes = reduced.tokenizer.bytes - 1,
            4 => remaining.max_fields = used.fields - 1, 5 => remaining.max_matches = used.matches - 1,
            6 => remaining.max_scan_steps = used.scan_steps - 1, _ => expected.max_result_bytes = 1 }
        let mut driver = script(expected);
        assert!(execution::reduce(SOURCE, collection, expected, expected.max_result_bytes,
            &mut remaining, &mut driver, &mut Continue).is_err(), "{axis}");
        if axis < 2 { assert!(driver.inputs.is_empty()); }
    }
}
#[test]
fn same_geometry_requests_and_coordinated_native_payload_edits_cannot_cross_verify() {
    let (expected, run) = complete(0); let planner = planner();
    let foreign = prepare(&planner, "éxyz----βuvw", &request()).preflight_metadata();
    assert_eq!(expected.discovery().source_span(), foreign.discovery().source_span());
    assert!(foreign.verify_completed(&run).is_err());
    let mut changed_question = request(); changed_question.question.question = "Why is named?".to_owned();
    assert!(prepare(&planner, SOURCE, &changed_question).preflight_metadata().verify_completed(&run).is_err());
    for axis in 0..7 {
        let (expected, mut run) = complete(0);
        match axis {
            0 | 1 => {
                let index = if axis == 0 { 0 } else { run.final_pass.unwrap() };
                let pass = &mut run.passes[index];
                let SourceTaskResult::Answer(raw) = &mut pass.native.result else { panic!("answer") };
                raw.answer = Some("coordinated replacement".to_owned()); pass.answer.answer = raw.answer.clone();
            }
            2 => run.passes[1].input_passages.start = 0,
            3 => run.passes[0].context_bytes += 1,
            4 => run.levels[1].input.passages += 1,
            5 => run.final_pass = Some(1),
            _ => run.verification_used.fields += 1,
        }
        assert!(expected.verify_completed(&run).is_err(), "{axis}");
    }
}
#[test]
fn complete_context_and_real_plan_witnesses_bind_question_and_passage_manifest() {
    let planner = planner(); let identity = identity(&planner); let request = request();
    let expected = prepare(&planner, SOURCE, &request).preflight_metadata();
    let passages = vec![AnswerPassage { id: "evidence_0".to_owned(), text: "éabc".to_owned() },
        AnswerPassage { id: "evidence_1".to_owned(), text: "βdef".to_owned() }];
    let bytes = transport::context_bytes(request.question.question.len(),
        passages.iter().map(|p| (p.id.as_str(), p.text.as_str())), &mut Continue).unwrap();
    let encoded = AnswerContext::encode(planner.source_encoder(), &request.question.question, &passages,
        AnswerInputLimits { max_passages: 32, max_input_bytes: 8192, max_input_tokens: 8192 }).unwrap();
    assert_eq!(encoded.document().total_token_count(), bytes);
    let prompt = execution::count_context(&planner, &request.question.question, &passages, bytes,
        AnswerInputLimits { max_passages: 32, max_input_bytes: 8192, max_input_tokens: 8192 }, expected.capacity, &mut Continue).unwrap().unwrap();
    let actual = planner.plan_int8_with_control(&SourceTaskRequest::Answer { question: request.question.question,
        passages, options: request.question.options, budget: budget() }, &PlanContext::new(&identity, budget()).unwrap(),
        SourcePlanningLimits::default(), &mut Continue).unwrap();
    assert_eq!(actual.prompt_tokens(), prompt); assert_eq!(prompt, expected.capacity.scaffold + bytes);
    let build = |question: &str, passage_id: &str| {
        let request = SourceTaskRequest::Answer { question: question.to_owned(),
            passages: vec![AnswerPassage { id: passage_id.to_owned(), text: "éabc".to_owned() }],
            options: AnswerOptions::default(), budget: budget() };
        let plan = planner.plan_int8_with_control(&request, &PlanContext::new(&identity, budget()).unwrap(),
            SourcePlanningLimits::default(), &mut Continue).unwrap();
        execution::Witness::of(&plan).unwrap()
    };
    assert!(build("Who?", "evidence_0") != build("Why?", "evidence_0"));
    assert!(build("Who?", "evidence_0") != build("Who?", "evidence_1"));
}
struct StopAt { calls: usize, stop: usize, cause: DecodeCancellationKind }
impl DecodeStepControl for StopAt {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.calls == self.stop).then_some(self.cause)
    }
}
#[test]
fn late_native_failure_and_every_terminal_cancellation_cause_never_publish_partial_results() {
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = script(expected); driver.fail_run = Some(2);
    let error = execution::reduce(SOURCE, collection, expected, expected.max_result_bytes,
        &mut remaining, &mut driver, &mut Continue).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(driver.inputs.len(), 2);
    use DecodeCancellationKind::*;
    let (expected, run) = complete(0);
    let mut count = StopAt { calls: 0, stop: usize::MAX, cause: User };
    expected.verify_completed_with_control(&run, &mut count).unwrap();
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for stop in [1, count.calls / 2, count.calls] {
            let mut control = StopAt { calls: 0, stop, cause };
            assert_eq!(expected.verify_completed_with_control(&run, &mut control).unwrap_err().cancellation(), Some(cause));
            assert_eq!(control.calls, stop);
        }
    }
}
#[test]
fn final_output_limit_counts_every_native_and_lifted_pass_after_reduction() {
    let (_, run) = complete(0); let bytes = canonjson::canonical_bytes(&run).unwrap().len();
    for cap in [bytes - 1, bytes] {
        let (mut expected, discovery, collection, mut remaining) = setup(false); let initial = EvidenceSize::of(&collection);
        let reduced = execution::reduce(SOURCE, collection, expected, expected.max_result_bytes,
            &mut remaining, &mut script(expected), &mut Continue).unwrap();
        expected.max_result_bytes = cap as u64;
        let finished = finish(discovery, initial, reduced, remaining, expected, &mut Continue);
        if cap < bytes { assert!(finished.is_err()); }
        else { assert_eq!(canonjson::canonical_bytes(&finished.unwrap()).unwrap().len(), bytes); }
    }
}
