//! Model-free pinned plans and PRIVATE evidence/corruption fixtures. These do
//! not establish neural success, answer quality, calibrated abstention or speed.
use super::*;
use crate::{native_engine::strict_int8::STRICT_INT8_EXECUTION,
    tasks::answer::PassageOccurrence, tokenizer::pinned_controls};
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
struct Stop;
impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) } }
fn planner() -> SourceTaskPlanner {
    let controls = pinned_controls::pinned().unwrap();
    let eos = controls.template_controls().entries().iter().find(|e| e.special && e.surface == IM_END).unwrap().id;
    SourceTaskPlanner::pinned(controls.template_controls(), eos).unwrap()
}
fn identity(p: &SourceTaskPlanner) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"private synthesis test fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: ANSWER_TASK_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn limits() -> Int8SourceMapLimits {
    Int8SourceMapLimits { chunks: ChunkLimits { max_input_bytes: 8192, max_chunk_bytes: 64, max_chunk_tokens: 64,
        context_tokens: 8192, reserved_tokens: 64, max_chunks: 256, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { map_batch_chunks: 2, reduce_fan_in: 2, max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
        max_model_work: Int8Work { forward_positions: u64::MAX, projected_logits: u64::MAX, attention_pairs: u64::MAX,
            projections: ProjectionWork { dot_products: u64::MAX, multiply_accumulates: u64::MAX } },
        mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 257_000 }
}
fn request() -> SourceQuestionSynthesis {
    SourceQuestionSynthesis { question: SourceQuestion { question: "Who is named?".to_owned(),
        options: AnswerOptions::default(), verification: GroundingBudget::default() },
        limits: QuestionSynthesisLimits { max_evidence_passages: 32, max_evidence_bytes: 8192 } }
}
fn prepare<'s>(p: &SourceTaskPlanner, text: &'s str, request: &SourceQuestionSynthesis) -> PreparedInt8QuestionSynthesis<'s> {
    let id = identity(p);
    p.plan_int8_question_synthesis_with_control(text, request, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), limits(), &mut Continue).unwrap()
}
fn value(parts: &[(&str, Option<&str>)]) -> (String, QuestionValue) {
    let mut source = String::new(); let mut chunks = Vec::new(); let mut scalar = 0;
    for (id, &(text, quote)) in parts.iter().enumerate() {
        let byte = source.len(); source.push_str(text); let end = scalar + text.chars().count();
        let citations = quote.into_iter().map(|quote| {
            let mut spans = scan_occurrences(text, quote, &mut GroundingBudget::default()).unwrap();
            for s in &mut spans { s.byte_start += byte; s.byte_end += byte; s.scalar_start += scalar; s.scalar_end += scalar; }
            SourceCitation { quote: quote.to_owned(), occurrence: occurrence(spans.len()), spans }
        }).collect();
        chunks.push(Arc::new(QuestionChunk { chunk_id: id,
            source_span: VerifiedSourceSpan { byte_start: byte, byte_end: source.len(), scalar_start: scalar, scalar_end: end },
            status: if quote.is_some() { QuestionChunkStatus::Answered } else if has_text(text) {
                QuestionChunkStatus::Abstained } else { QuestionChunkStatus::WhitespaceOnly },
            answer: quote.map(|_| "MODEL ANSWER MUST NOT BECOME EVIDENCE".to_owned()), citations,
            model_work: Int8Work::default(), mask_node_visit_charge: 0 }));
        scalar = end;
    }
    (source, QuestionValue { chunks })
}
fn occurrence(count: usize) -> SourceOccurrence { if count == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous } }
fn raw(evidence: &evidence::Collection, quote: &str) -> AnswerResult {
    let mut spans = Vec::new();
    for p in &evidence.passages {
        for span in scan_occurrences(&p.text, quote, &mut GroundingBudget::default()).unwrap() {
            spans.push(PassageOccurrence { passage_id: p.id.clone(), span });
        }
    }
    AnswerResult { schema_version: 1, task_spec_version: ANSWER_TASK_VERSION.to_owned(), numerics_profile: STRICT_INT8_PROFILE.to_owned(),
        status: AnswerStatus::Answered, answerable: true, answer: Some("answer".to_owned()),
        citations: vec![PassageCitation { quote: quote.to_owned(), occurrence: occurrence(spans.len()), spans }],
        calibration: AnswerCalibration::Uncalibrated, citation_guarantee: CitationGuarantee::StructuralSourceMembership,
        semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
        untrusted_fields: ["answer".to_owned(), "citations".to_owned()], generated_token_ids: vec![1, 0],
        forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 20 }
}
fn collection(source: &str, value: &QuestionValue) -> evidence::Collection {
    evidence::collect(source, value, request().limits, &mut GroundingBudget::default(), &mut Continue).unwrap()
}
#[test]
fn pinned_preflight_matches_the_actual_map_and_reserves_a_final_context_before_discovery() {
    let p = planner(); let q = request(); let text = "é Alice and Bob"; let id = identity(&p);
    let prepared = prepare(&p, text, &q);
    let preflight = p.preflight_int8_question_synthesis_with_control(text, &q, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), limits(), &mut Continue).unwrap();
    assert_eq!(preflight, prepared.preflight_metadata());
    assert_eq!(preflight.reserved_mask_visits().unwrap(), preflight.discovery().reserved_mask_visits() + limits().mask_visits_per_chunk);
    let (remainder, reserve) = reserve(q.limits, budget(), SourcePlanningLimits::default(), limits()).unwrap();
    assert_eq!(add_work(remainder.max_model_work, reserve).unwrap(), limits().max_model_work);
    assert_eq!(reserve, preflight.synthesis_reserved_work());
    assert!(reserve.forward_positions > 0);
}
#[test]
fn every_native_axis_and_the_final_mask_reservation_fail_before_discovery_when_unfunded() {
    let q = request(); let (_, required) = reserve(q.limits, budget(), SourcePlanningLimits::default(), limits()).unwrap();
    for i in 0..5 {
        let mut l = limits(); l.max_model_work = required;
        let axis = match i { 0 => &mut l.max_model_work.forward_positions, 1 => &mut l.max_model_work.projected_logits,
            2 => &mut l.max_model_work.attention_pairs, 3 => &mut l.max_model_work.projections.dot_products,
            _ => &mut l.max_model_work.projections.multiply_accumulates };
        *axis -= 1;
        assert!(matches!(reserve(q.limits, budget(), SourcePlanningLimits::default(), l), Err(Int8SourceMapError::WorkLimit)));
    }
    let p = planner(); let id = identity(&p); let mut l = limits(); l.max_mask_visits = l.mask_visits_per_chunk;
    assert!(p.preflight_int8_question_synthesis_with_control("Alice", &q, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), l, &mut Continue).is_err());
}
#[test]
fn only_verbatim_quotes_enter_the_final_request_and_every_candidate_answer_survives() {
    let (source, values) = value(&[("é Alice", Some("Alice")), (" Bob", Some("Bob"))]);
    let evidence = collection(&source, &values); let passages = evidence.copy_passages().unwrap();
    assert_eq!(passages.len(), 2); assert_eq!(passages[0].text, "Alice"); assert_eq!(passages[1].text, "Bob");
    assert!(!canonjson::canonical_string(&passages).unwrap().contains("MODEL ANSWER"));
    assert!(values.chunks().all(|c| c.answer.as_deref() == Some("MODEL ANSWER MUST NOT BECOME EVIDENCE")));
    let p = planner(); let id = identity(&p);
    let plan = p.plan_int8_with_control(&SourceTaskRequest::Answer { question: "Who is named?".to_owned(), passages,
        options: AnswerOptions::default(), budget: budget() }, &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), &mut Continue).unwrap();
    let (_, reserve) = reserve(request().limits, budget(), SourcePlanningLimits::default(), limits()).unwrap();
    assert!(within(plan.planned_work(), reserve));
}
#[test]
fn repeated_unicode_subquotes_lift_all_evidence_origins_but_not_unexamined_equal_text() {
    let (source, values) = value(&[("éAlice éAlice", Some("éAlice")), (" Alice", None)]);
    let evidence = collection(&source, &values);
    let answer = evidence::final_answer(&source, raw(&evidence, "Alice"), &evidence, AnswerOptions::default(),
        &mut GroundingBudget::default(), &mut Continue).unwrap();
    let c = &answer.citations[0]; assert_eq!(c.occurrence, SourceOccurrence::Ambiguous); assert_eq!(c.spans.len(), 2);
    assert_eq!(c.spans[0].byte_start, 2); assert_eq!(c.spans[0].scalar_start, 1);
    assert_eq!(c.spans[1].byte_start, 10); assert_eq!(c.spans[1].scalar_start, 8);
    for s in &c.spans { assert_eq!(&source[s.byte_start..s.byte_end], "Alice"); }
}
#[test]
fn overlapping_evidence_does_not_manufacture_multiple_original_occurrences() {
    let (source, mut values) = value(&[("é Alice", Some("é Alice"))]);
    Arc::get_mut(&mut values.chunks[0]).unwrap().citations.push(SourceCitation { quote: "Alice".to_owned(),
        occurrence: SourceOccurrence::Anchored, spans: vec![VerifiedSourceSpan { byte_start: 3, byte_end: 8, scalar_start: 2, scalar_end: 7 }] });
    let evidence = collection(&source, &values);
    let native = raw(&evidence, "Alice"); assert_eq!(native.citations[0].spans.len(), 2);
    let answer = evidence::final_answer(&source, native, &evidence, AnswerOptions::default(), &mut GroundingBudget::default(), &mut Continue).unwrap();
    assert_eq!(answer.citations[0].spans.len(), 1); assert_eq!(answer.citations[0].occurrence, SourceOccurrence::Anchored);
}
#[test]
fn duplicate_quotes_are_verified_before_deduplication() {
    let (source, mut values) = value(&[("Alice", Some("Alice"))]);
    let chunk = Arc::get_mut(&mut values.chunks[0]).unwrap(); chunk.citations.push(chunk.citations[0].clone());
    assert_eq!(collection(&source, &values).passages.len(), 1);
    Arc::get_mut(&mut values.chunks[0]).unwrap().citations[1].spans[0].scalar_end += 1;
    assert!(evidence::collect(&source, &values, request().limits, &mut GroundingBudget::default(), &mut Continue).is_err());
}
#[test]
fn passage_byte_and_occurrence_caps_reject_complete_evidence_instead_of_top_k() {
    let (source, values) = value(&[("Alice", Some("Alice")), (" Bob", Some("Bob"))]);
    for l in [QuestionSynthesisLimits { max_evidence_passages: 1, ..request().limits },
        QuestionSynthesisLimits { max_evidence_bytes: 7, ..request().limits }] {
        assert!(matches!(evidence::collect(&source, &values, l, &mut GroundingBudget::default(), &mut Continue), Err(Int8SourceMapError::WorkLimit)));
    }
    let (source, values) = value(&[("éAlice éAlice", Some("éAlice"))]); let evidence = collection(&source, &values);
    let mut small = GroundingBudget::default(); small.max_matches = 2; // One local match PLUS two original occurrences.
    assert!(matches!(evidence::final_answer(&source, raw(&evidence, "Alice"), &evidence, AnswerOptions::default(),
        &mut small, &mut Continue), Err(Int8SourceMapError::WorkLimit)));
}
#[test]
fn final_citations_cannot_cross_synthetic_joins_omit_occurrences_or_substitute_passage_ids() {
    let (source, values) = value(&[("Alice", Some("Alice")), (" Alice", Some("Alice"))]); let evidence = collection(&source, &values);
    for axis in 0..5 {
        let mut native = raw(&evidence, "Alice");
        match axis { 0 => native.citations[0].spans[1].passage_id = "question".to_owned(),
            1 => { native.citations[0].spans.pop(); }, 2 => native.citations[0].spans[0].span.scalar_start += 1,
            3 => native.citations[0].occurrence = SourceOccurrence::Anchored,
            _ => native.citations[0].quote = "Alice\n\nAlice".to_owned() }
        assert!(evidence::final_answer(&source, native, &evidence, AnswerOptions::default(), &mut GroundingBudget::default(), &mut Continue).is_err());
    }
}
#[test]
fn final_abstention_is_distinct_from_no_evidence_and_cannot_hide_answer_text() {
    let (source, values) = value(&[("Alice", Some("Alice"))]); let evidence = collection(&source, &values);
    let mut native = raw(&evidence, "Alice"); native.status = AnswerStatus::Abstained; native.answerable = false;
    native.answer = None; native.citations.clear();
    let answer = evidence::final_answer(&source, native.clone(), &evidence, AnswerOptions::default(), &mut GroundingBudget::default(), &mut Continue).unwrap();
    assert_eq!(answer.status, SynthesisStatus::Abstained);
    native.answer = Some("hidden answer".to_owned());
    assert!(evidence::final_answer(&source, native, &evidence, AnswerOptions::default(), &mut GroundingBudget::default(), &mut Continue).is_err());
}
#[test]
fn malformed_late_geometry_and_cancellation_cannot_yield_partial_evidence() {
    let (source, mut values) = value(&[("Alice", Some("Alice")), (" Bob", Some("Bob"))]);
    Arc::get_mut(&mut values.chunks[1]).unwrap().source_span.scalar_start += 1;
    assert!(evidence::collect(&source, &values, request().limits, &mut GroundingBudget::default(), &mut Continue).is_err());
    let (source, values) = value(&[("Alice", Some("Alice"))]);
    let error = evidence::collect(&source, &values, request().limits, &mut GroundingBudget::default(), &mut Stop).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    let evidence = collection(&source, &values);
    let error = evidence::final_answer(&source, raw(&evidence, "Alice"), &evidence, AnswerOptions::default(),
        &mut GroundingBudget::default(), &mut Stop).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    assert!(!format!("{error:?}").contains("Alice"));
}
struct Abstain;
impl SourceDriver for Abstain {
    fn checkpoint(&mut self) -> Result<(), Int8SourceError> { Ok(()) }
    fn run(&mut self, plan: &PreparedInt8SourceTask, _: &ExecutionIdentity) -> Result<Int8SourceTaskRun, Int8SourceError> {
        let work = constrained_int8::planned_work(plan.prompt_tokens(), 2).unwrap();
        let raw = AnswerResult { schema_version: 1, task_spec_version: ANSWER_TASK_VERSION.to_owned(), numerics_profile: STRICT_INT8_PROFILE.to_owned(),
            status: AnswerStatus::Abstained, answerable: false, answer: None, citations: Vec::new(),
            calibration: AnswerCalibration::Uncalibrated, citation_guarantee: CitationGuarantee::StructuralSourceMembership,
            semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
            untrusted_fields: ["answer".to_owned(), "citations".to_owned()], generated_token_ids: vec![1, 0],
            forward_positions: work.forward_positions, projected_logits: work.projected_logits, mask_node_visit_charge: 20 };
        Ok(Int8SourceTaskRun { schema_version: 1, execution: INT8_SOURCE_EXECUTION.to_owned(), result: SourceTaskResult::Answer(raw), model_work: work })
    }
}
#[test]
fn no_evidence_skips_the_final_model_without_refunding_its_reservation_or_hiding_discovery() {
    let p = planner(); let q = request(); let text = "Alice"; let prepared = prepare(&p, text, &q);
    let expected = prepared.expected; let admitted: Vec<_> = prepared.execution_identities().cloned().collect();
    let discovery = prepared.map.execute_with_driver(&admitted, Abstain).unwrap();
    let mut remaining = remaining_verification(&discovery, q.question.verification).unwrap();
    let evidence = evidence::collect(text, discovery.mapped.root().value(), q.limits, &mut remaining, &mut Continue).unwrap();
    assert!(evidence.passages.is_empty());
    let answer = SynthesisAnswer { status: SynthesisStatus::NoEvidenceCollected, answer: None, citations: Vec::new() };
    let mut run = finish(discovery, answer, 0, 0, Int8Work::default(), Int8Work::default(), 0, remaining,
        expected, 1 << 20, &mut Continue).unwrap();
    assert_eq!(run.model_work, run.discovery.model_work); assert_eq!(run.discovery.abstained_chunks, 1);
    assert!(run.synthesis_reserved_work.forward_positions > 0); expected.verify_completed(&run).unwrap();
    let json = canonjson::canonical_string(&run).unwrap(); assert!(!json.contains("generated_token_ids")); assert!(!json.contains("prompt_digest"));
    run.synthesis_model_work.forward_positions = 1; assert!(expected.verify_completed(&run).is_err());
    run.synthesis_model_work.forward_positions = 0;
    assert!(extract_int8::check_size(&run, 1).is_err());
}
#[test]
fn changing_the_original_question_changes_all_discovery_identities_and_bad_profiles_fail_closed() {
    let p = planner(); let a = request(); let mut b = request(); b.question.question = "Which person appears?".to_owned();
    let left = prepare(&p, "Alice", &a); let right = prepare(&p, "Alice", &b);
    assert_ne!(left.execution_identities().next().unwrap().prompt_digest, right.execution_identities().next().unwrap().prompt_digest);
    let mut id = identity(&p); id.numerics_profile = NumericsProfile::HfBf16Eager;
    assert!(p.plan_int8_question_synthesis_with_control("Alice", &a, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), limits(), &mut Continue).is_err());
}
