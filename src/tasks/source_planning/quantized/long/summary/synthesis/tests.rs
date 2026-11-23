//! Pinned planning and private completion regressions; no native/model-quality evidence.
use super::*;
use crate::{native_engine::strict_int8::STRICT_INT8_EXECUTION, tokenizer::pinned_controls};

struct Control { calls: usize, stop: usize }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.calls >= self.stop).then_some(DecodeCancellationKind::PollQuota)
    }
}
fn continuing() -> Control { Control { calls: 0, stop: usize::MAX } }
fn planner() -> SourceTaskPlanner {
    let registry = pinned_controls::pinned().unwrap(); let c = registry.template_controls();
    let eos = c.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    SourceTaskPlanner::pinned(c, eos).unwrap()
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn identity(p: &SourceTaskPlanner) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"summary-synthesis-pinned-planning-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(),
        task_spec: SUMMARIZE_TASK_VERSION.to_owned(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn mapping() -> Int8SourceMapLimits {
    Int8SourceMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 6, max_chunk_tokens: 6,
            context_tokens: 8192, reserved_tokens: 64, max_chunks: 64, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
        max_model_work: Int8Work::for_sequence(0, 32_768, 32_768 * 166_144).unwrap(),
        mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 64_000,
    }
}
fn request() -> SourceSummarySynthesis {
    SourceSummarySynthesis { map_options: SummaryOptions::default(), synthesis_options: SummaryOptions::default(),
        limits: SummarySynthesisLimits { max_evidence_segments: 64, max_evidence_bytes: 4096,
            verification: GroundingBudget::default() } }
}
fn prepare<'s>(p: &SourceTaskPlanner, text: &'s str, request: SourceSummarySynthesis,
    mapping: Int8SourceMapLimits, control: &mut Control) -> Result<PreparedInt8SummarySynthesis<'s>, Int8SourceMapError> {
    let id = identity(p);
    p.plan_int8_summary_synthesis_with_control(text, request, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), mapping, control)
}
#[test]
fn all_actual_unicode_chunks_and_synthesis_are_reserved_before_execution() {
    let p = planner(); let text = "éAéAéAéA";
    let prepared = prepare(&p, text, request(), mapping(), &mut continuing()).unwrap();
    let expected = prepared.preflight_metadata();
    assert_eq!(expected.chunk_count(), 2);
    assert_eq!(expected.source_span().byte_end, text.len());
    assert_eq!(expected.source_span().scalar_end, text.chars().count());
    assert_eq!(expected.reserved_model_work().unwrap(), add_work(prepared.map.work, expected.synthesis_reserve).unwrap());
    assert_eq!(expected.reserved_mask_visits().unwrap(), 3000);
    assert_eq!(prepared.execution_identities().len(), 2);
    for id in prepared.execution_identities() { assert_eq!(id.task_spec, SUMMARIZE_TASK_VERSION); }
}
#[test]
fn every_whole_invocation_axis_includes_final_synthesis_even_at_exact_boundary() {
    let p = planner(); let source = "éAéAéAéA";
    let expected = prepare(&p, source, request(), mapping(), &mut continuing()).unwrap().preflight_metadata();
    let mut exact = mapping(); exact.max_model_work = expected.reserved_model_work().unwrap();
    exact.max_mask_visits = expected.reserved_mask_visits().unwrap();
    prepare(&p, source, request(), exact, &mut continuing()).unwrap();
    for axis in 0..6 {
        let mut limited = exact;
        match axis { 0 => limited.max_model_work.forward_positions -= 1,
            1 => limited.max_model_work.projected_logits -= 1, 2 => limited.max_model_work.attention_pairs -= 1,
            3 => limited.max_model_work.projections.dot_products -= 1,
            4 => limited.max_model_work.projections.multiply_accumulates -= 1, _ => limited.max_mask_visits -= 1 }
        assert!(prepare(&p, source, request(), limited, &mut continuing()).is_err(), "{axis}");
    }
}
#[test]
fn different_final_options_cannot_change_map_prompts_or_evade_synthesis_bounds() {
    let p = planner(); let a = prepare(&p, "source", request(), mapping(), &mut continuing()).unwrap();
    let mut req = request(); req.synthesis_options.max_bullets = 1;
    let b = prepare(&p, "source", req, mapping(), &mut continuing()).unwrap();
    assert!(a.execution_identities().eq(b.execution_identities()));
    assert_ne!(a.expected.options, b.expected.options);
    assert_eq!(a.expected.reserved_model_work().unwrap(), b.expected.reserved_model_work().unwrap());
    req.synthesis_options.max_citations_per_bullet = 0;
    assert!(prepare(&p, "source", req, mapping(), &mut continuing()).is_err());
}
#[test]
fn empty_source_invalid_verification_and_foreign_profile_are_refused() {
    let p = planner();
    assert!(prepare(&p, "", request(), mapping(), &mut continuing()).is_err());
    for axis in 0..5 {
        let mut r = request();
        match axis { 0 => r.limits.max_evidence_segments = 0, 1 => r.limits.max_evidence_bytes = 0,
            2 => r.limits.verification.max_fields = 0, 3 => r.limits.verification.max_matches = 0,
            _ => r.limits.verification.max_scan_steps = 0 }
        assert!(prepare(&p, "source", r, mapping(), &mut continuing()).is_err());
    }
    let mut id = identity(&p); id.numerics_profile = NumericsProfile::HfBf16Eager;
    assert!(p.plan_int8_summary_synthesis_with_control("source", request(), budget(),
        &PlanContext::new(&id, budget()).unwrap(), SourcePlanningLimits::default(), mapping(), &mut continuing()).is_err());
}
#[test]
fn cancellation_survives_every_pinned_preparation_checkpoint_including_the_last() {
    let p = planner(); let mut control = continuing();
    prepare(&p, "éAéA", request(), mapping(), &mut control).unwrap();
    let last = control.calls; assert!(last > 2);
    for stop in [1, last / 2, last] {
        let mut control = Control { calls: 0, stop };
        let error = prepare(&p, "éAéA", request(), mapping(), &mut control).err().unwrap();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::PollQuota));
        assert_eq!(control.calls, stop);
    }
}
#[test]
fn maximum_context_reserve_is_subtracted_without_saturating_any_axis() {
    let mut m = mapping(); m.max_model_work = Int8Work::default();
    assert!(reserve(budget(), SourcePlanningLimits::default(), m).is_err());
    let mut m = mapping(); m.max_mask_visits = m.mask_visits_per_chunk - 1;
    assert!(reserve(budget(), SourcePlanningLimits::default(), m).is_err());
}

struct EmptySummary { token: u32 }
impl SourceDriver for EmptySummary {
    fn checkpoint(&mut self) -> Result<(), Int8SourceError> { Ok(()) }
    fn run(&mut self, plan: &PreparedInt8SourceTask, _: &ExecutionIdentity) -> Result<Int8SourceTaskRun, Int8SourceError> {
        let work = constrained_int8::planned_work(plan.prompt_tokens(), 2).unwrap();
        let raw = SummaryResult { schema_version: 1, task_spec_version: SUMMARIZE_TASK_VERSION.to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), citation_guarantee: CitationGuarantee::StructuralSourceMembership,
            semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
            bullets: Vec::new(), generated_token_ids: vec![self.token, 0],
            forward_positions: work.forward_positions, projected_logits: work.projected_logits, mask_node_visit_charge: 20 };
        Ok(Int8SourceTaskRun { schema_version: 1, execution: INT8_SOURCE_EXECUTION.to_owned(),
            result: SourceTaskResult::Summarize(raw), model_work: work })
    }
}
fn no_evidence_completion(p: &SourceTaskPlanner, text: &str, token: u32)
    -> (Int8SummarySynthesisPreflight, Int8SummarySynthesisRun) {
    let prepared = prepare(p, text, request(), mapping(), &mut continuing()).unwrap();
    let expected = prepared.expected;
    let admitted: Vec<_> = prepared.execution_identities().cloned().collect();
    let discovery = prepared.map.execute_with_driver(&admitted, EmptySummary { token }).unwrap();
    let total = request().limits.verification; let mut remaining = total;
    let collection = evidence::collect(text, discovery.mapped.root().value(), request().map_options,
        request().limits, &mut remaining, &mut continuing()).unwrap();
    assert_eq!(collection.segment_count(), 0);
    let work = discovery.model_work; let masks = discovery.mask_node_visit_charge;
    let run = Int8SummarySynthesisRun { schema_version: 1, execution: INT8_SUMMARY_SYNTHESIS_EXECUTION,
        numerics_profile: STRICT_INT8_PROFILE, semantics: SEMANTICS, status: SummarySynthesisStatus::NoEvidenceCollected,
        evidence_segments: 0, evidence_bytes: 0, synthesis_reserved_work: expected.synthesis_reserve,
        synthesis_planned_work: Int8Work::default(), synthesis_model_work: Int8Work::default(),
        reserved_model_work: expected.reserved_model_work().unwrap(), model_work: work,
        reserved_mask_node_visits: expected.reserved_mask_visits().unwrap(), synthesis_mask_node_visits: 0,
        mask_node_visit_charge: masks, verification_used: SummaryVerificationWork {
            fields: total.max_fields - remaining.max_fields, matches: total.max_matches - remaining.max_matches,
            scan_steps: total.max_scan_steps - remaining.max_scan_steps },
        citation_guarantee: CitationGuarantee::StructuralSourceMembership, semantic_support: SummarySemanticSupport::NotAssessed,
        untrusted_fields: ["bullets", "synthesis_native", "discovery.mapped.root.value"],
        bullets: Vec::new(), synthesis_native: None, discovery,
        request_binding: expected.request_binding, result_binding: [0; 32] };
    (expected, seal_completed(run, expected, &mut continuing()).unwrap())
}
#[test]
fn summary_completion_rejects_equal_geometry_from_another_source_or_allowance() {
    let p = planner(); let (expected, run) = no_evidence_completion(&p, "Alice", 1);
    expected.verify_completed(&run).unwrap();
    let mut bytes = request(); bytes.limits.max_evidence_bytes -= 1;
    let mut matches = request(); matches.limits.verification.max_matches -= 1;
    let mut masks = mapping(); masks.max_mask_visits -= 1;
    for (text, request, mapping) in [("Carol", request(), mapping()), ("Alice", bytes, mapping()),
        ("Alice", matches, mapping()), ("Alice", request(), masks)] {
        let other = prepare(&p, text, request, mapping, &mut continuing()).unwrap().preflight_metadata();
        assert_eq!(other.source_span(), expected.source_span());
        assert_eq!(other.reserved_model_work().unwrap(), expected.reserved_model_work().unwrap());
        assert_eq!(other.reserved_mask_visits().unwrap(), expected.reserved_mask_visits().unwrap());
        assert!(other.verify_completed(&run).is_err());
    }
    for map_options in [false, true] {
        let mut changed = request();
        if map_options { changed.map_options.max_bullets = 1; } else { changed.synthesis_options.max_bullets = 1; }
        let other = prepare(&p, "Alice", changed, mapping(), &mut continuing()).unwrap().preflight_metadata();
        assert!(other.verify_completed(&run).is_err());
    }
}
#[test]
fn summary_completion_binds_verification_metadata_and_retained_native_tokens() {
    let p = planner(); let (expected, mut run) = no_evidence_completion(&p, "Alice", 1);
    run.verification_used.fields += 1;
    assert!(expected.verify_completed(&run).is_err());
    run.verification_used.fields -= 1;
    expected.verify_completed(&run).unwrap();
    let (_, other) = no_evidence_completion(&p, "Alice", 2);
    assert_eq!(run.discovery.model_work, other.discovery.model_work);
    assert_eq!(run.discovery.mask_node_visit_charge, other.discovery.mask_node_visit_charge);
    run.discovery = other.discovery;
    assert!(expected.verify_completed(&run).is_err());
}
#[test]
fn summary_completion_verification_preserves_all_terminal_causes() {
    struct CancelAt { cause: DecodeCancellationKind, stop: usize, polls: usize }
    impl DecodeStepControl for CancelAt {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.polls += 1; (self.polls == self.stop).then_some(self.cause)
        }
    }
    use DecodeCancellationKind::*;
    let p = planner(); let (expected, run) = no_evidence_completion(&p, "Alice", 1);
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for stop in [1, 3] {
            let mut control = CancelAt { cause, stop, polls: 0 };
            assert_eq!(expected.verify_completed_with_control(&run, &mut control).unwrap_err().cancellation(), Some(cause));
            assert_eq!(control.polls, stop);
        }
    }
}
#[test]
fn summary_completion_hashes_are_private_and_public_output_has_an_exact_byte_ceiling() {
    let p = planner(); let (mut expected, run) = no_evidence_completion(&p, "Alice", 1);
    let json = crate::canonjson::canonical_string(&run).unwrap();
    let debug = format!("{expected:?}");
    for private in ["request_binding", "result_binding", "Alice"] {
        assert!(!json.contains(private)); assert!(!debug.contains(private));
    }
    expected.max_result_bytes = json.len() as u64;
    expected.verify_completed(&run).unwrap();
    expected.max_result_bytes -= 1;
    assert!(expected.verify_completed(&run).is_err());
}

const COMPACT_SOURCE: &str = "The quick brown fox jumps over the lazy dog. 上海 <think> <|im_start|> ▁\r\n";
pub(super) fn compact_policy() -> crate::tokenizer::untrusted::UntrustedEncoding {
    use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 })
}
pub(super) fn encoding_allowance() -> crate::tokenizer::untrusted::SafeBpeWorkLimits {
    crate::tokenizer::untrusted::SafeBpeWorkLimits { max_input_bytes: 65_536, max_output_tokens: 65_536,
        max_pair_checks: 1 << 26, max_pair_bytes: 1 << 30 }
}
fn compact(work: crate::tokenizer::untrusted::SafeBpeWorkLimits) -> SourceTaskPlanner {
    planner().with_map_document_encoding(compact_policy(), work).unwrap()
}
fn compact_mapping() -> Int8SourceMapLimits {
    let mut limits = mapping();
    limits.chunks.max_chunk_bytes = 4096; limits.chunks.max_chunk_tokens = 4096;
    limits
}
pub(super) fn encoding_used(p: &SourceTaskPlanner) -> crate::tokenizer::untrusted::SafeBpeWork {
    p.map_document_encoding_work().unwrap().unwrap()
}
pub(super) fn quoted_run(plan: &PreparedInt8SourceTask, document: &str, quotes: &[&str]) -> Int8SourceTaskRun {
    let work = constrained_int8::planned_work(plan.prompt_tokens(), 2).unwrap();
    let bullets = quotes.iter().map(|quote| {
        let spans = scan_occurrences(document, quote, &mut GroundingBudget::default()).unwrap();
        CitedBullet { text: "untrusted generated assertion".to_owned(), citations: vec![SourceCitation {
            quote: (*quote).to_owned(),
            occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }] }
    }).collect();
    let raw = SummaryResult { schema_version: 1, task_spec_version: SUMMARIZE_TASK_VERSION.to_owned(),
        numerics_profile: STRICT_INT8_PROFILE.to_owned(), citation_guarantee: CitationGuarantee::StructuralSourceMembership,
        semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
        bullets, generated_token_ids: vec![1, 0], forward_positions: work.forward_positions,
        projected_logits: work.projected_logits, mask_node_visit_charge: 20 };
    Int8SourceTaskRun { schema_version: 1, execution: INT8_SOURCE_EXECUTION.to_owned(),
        result: SourceTaskResult::Summarize(raw), model_work: work }
}
pub(super) struct QuotedSummary<'a> { pub document: &'a str, pub quotes: &'a [&'a str] }
impl SourceDriver for QuotedSummary<'_> {
    fn checkpoint(&mut self) -> Result<(), Int8SourceError> { Ok(()) }
    fn run(&mut self, plan: &PreparedInt8SourceTask, _: &ExecutionIdentity) -> Result<Int8SourceTaskRun, Int8SourceError> {
        Ok(quoted_run(plan, self.document, self.quotes))
    }
}
fn quoted_discovery(p: &SourceTaskPlanner, empty: bool) -> (SummarySynthesisTail<'static>, Int8SourceMapRun) {
    let prepared = prepare(p, COMPACT_SOURCE, request(), compact_mapping(), &mut continuing()).unwrap();
    assert_eq!(prepared.expected.chunk_count(), 1);
    let admitted: Vec<_> = prepared.execution_identities().cloned().collect();
    let (map, tail) = prepared.into_parts();
    let discovery = map.execute_with_driver(&admitted, QuotedSummary { document: COMPACT_SOURCE,
        quotes: if empty { &[] } else { &[COMPACT_SOURCE] } }).unwrap();
    (tail, discovery)
}
fn quoted_completion(p: &SourceTaskPlanner) -> (Int8SummarySynthesisPreflight, Int8SummarySynthesisRun) {
    let (tail, discovery) = quoted_discovery(p, false); let expected = tail.expected;
    let run = tail.execute(discovery, |plan, _| Ok(quoted_run(plan, COMPACT_SOURCE, &[COMPACT_SOURCE])),
        &mut continuing()).unwrap();
    (expected, run)
}
#[test]
fn compact_summary_reuses_discovery_encoding_budget_and_preserves_original_evidence() {
    let (byte_expected, byte) = quoted_completion(&planner());
    let p = compact(encoding_allowance()); let (expected, run) = quoted_completion(&p);
    expected.verify_completed(&run).unwrap(); byte_expected.verify_completed(&byte).unwrap();
    assert_eq!(run.bullets, byte.bullets);
    assert_eq!(run.evidence_segments, 1); assert_eq!(run.evidence_bytes, COMPACT_SOURCE.len());
    assert_eq!(run.bullets[0].citations[0].quote, COMPACT_SOURCE);
    assert_eq!(run.bullets[0].citations[0].spans[0], VerifiedSourceSpan {
        byte_start: 0, byte_end: COMPACT_SOURCE.len(), scalar_start: 0, scalar_end: COMPACT_SOURCE.chars().count() });
    assert!(run.discovery.model_work.forward_positions < byte.discovery.model_work.forward_positions);
    assert!(run.synthesis_model_work.forward_positions < byte.synthesis_model_work.forward_positions);
    // The source fit probe, discovery plan and derived evidence plan all spend
    // the same operation; no generated assertion is ever encoded as evidence.
    let used = encoding_used(&p);
    assert_eq!(used.input_bytes(), 3 * COMPACT_SOURCE.len());
    assert!(used.output_tokens() < used.input_bytes());
    assert!(used.pair_checks() > 0); assert!(used.pair_bytes() > 0);
    assert!(byte_expected.verify_completed(&run).is_err());
    assert!(expected.verify_completed(&byte).is_err());
    let byte_json = serde_json::to_value(&byte).unwrap(); let compact_json = serde_json::to_value(&run).unwrap();
    assert!(byte_json.as_object().unwrap().keys().eq(compact_json.as_object().unwrap().keys()));

    let mut cap = encoding_allowance(); cap.max_input_bytes = (2 * COMPACT_SOURCE.len()) as u64;
    let p = compact(cap); let (tail, discovery) = quoted_discovery(&p, true);
    let before = encoding_used(&p);
    let run = tail.execute(discovery, |_, _| panic!("empty evidence must not run synthesis"), &mut continuing()).unwrap();
    assert_eq!(run.status, SummarySynthesisStatus::NoEvidenceCollected);
    assert_eq!(encoding_used(&p), before);
}
#[test]
fn every_compact_summary_axis_is_nonrenewable_through_dynamic_synthesis() {
    let baseline = compact(encoding_allowance()); quoted_completion(&baseline);
    let used = encoding_used(&baseline);
    for axis in 0..4 {
        let mut cap = encoding_allowance();
        match axis { 0 => cap.max_input_bytes = used.input_bytes() as u64 - 1,
            1 => cap.max_output_tokens = used.output_tokens() as u64 - 1,
            2 => cap.max_pair_checks = used.pair_checks() - 1, _ => cap.max_pair_bytes = used.pair_bytes() - 1 }
        let p = compact(cap); let fresh = compact(cap);
        let prepared = prepare(&p, COMPACT_SOURCE, request(), compact_mapping(), &mut continuing()).unwrap();
        // An execution-time planner with identical policy has no authority to
        // replace the original operation's already-spent allowance.
        prepared.check_planner(&fresh).unwrap();
        let admitted: Vec<_> = prepared.execution_identities().cloned().collect();
        let (map, tail) = prepared.into_parts();
        let discovery = map.execute_with_driver(&admitted, QuotedSummary {
            document: COMPACT_SOURCE, quotes: &[COMPACT_SOURCE] }).unwrap();
        let before = encoding_used(&p); let mut native_calls = 0;
        assert!(tail.execute(discovery, |_, _| { native_calls += 1; Err(invalid()) }, &mut continuing()).is_err(), "{axis}");
        assert_eq!(native_calls, 0); assert_eq!(encoding_used(&fresh).input_bytes(), 0);
        let after = encoding_used(&p);
        assert!(after.input_bytes() >= before.input_bytes()); assert!(after.output_tokens() >= before.output_tokens());
        assert!(after.pair_checks() >= before.pair_checks()); assert!(after.pair_bytes() >= before.pair_bytes());
    }
}
#[test]
fn compact_summary_binds_each_limit_and_rejects_a_per_call_only_policy() {
    use crate::tokenizer::untrusted::UntrustedEncoding;
    let p = planner().with_document_encoding(compact_policy()).unwrap();
    assert!(prepare(&p, COMPACT_SOURCE, request(), compact_mapping(), &mut continuing()).is_err());
    let (expected, run) = quoted_completion(&compact(encoding_allowance()));
    for axis in 0..8 {
        let UntrustedEncoding::SafeBpeV1(mut call) = compact_policy() else { unreachable!() };
        let mut whole = encoding_allowance();
        match axis { 0 => call.max_input_bytes += 1, 1 => call.max_output_tokens += 1,
            2 => call.max_pair_checks += 1, 3 => call.max_pair_bytes += 1,
            4 => whole.max_input_bytes += 1, 5 => whole.max_output_tokens += 1,
            6 => whole.max_pair_checks += 1, _ => whole.max_pair_bytes += 1 }
        let p = planner().with_map_document_encoding(UntrustedEncoding::SafeBpeV1(call), whole).unwrap();
        let other = prepare(&p, COMPACT_SOURCE, request(), compact_mapping(), &mut continuing()).unwrap().preflight_metadata();
        assert_eq!(other.reserved_model_work().unwrap(), expected.reserved_model_work().unwrap());
        assert_ne!(other, expected); assert!(other.verify_completed(&run).is_err(), "{axis}");
    }
}
pub(super) struct StopInsideEncoding<'a> {
    pub planner: &'a SourceTaskPlanner, pub cause: DecodeCancellationKind, pub busy_polls: usize,
}
impl DecodeStepControl for StopInsideEncoding<'_> {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if self.planner.map_document_encoding_work().is_err() {
            self.busy_polls += 1;
            if self.busy_polls == 8 { return Some(self.cause); }
        }
        None
    }
}
#[test]
fn dynamic_summary_encoding_retains_all_terminal_causes_and_spent_work() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let p = compact(encoding_allowance()); let (tail, discovery) = quoted_discovery(&p, false);
        let before = encoding_used(&p); let mut control = StopInsideEncoding { planner: &p, cause, busy_polls: 0 };
        let error = tail.execute(discovery, |_, _| panic!("cancelled encoding must not run inference"), &mut control).err().unwrap();
        assert_eq!(error.cancellation(), Some(cause)); assert_eq!(control.busy_polls, 8);
        assert!(encoding_used(&p).input_bytes() > before.input_bytes());
    }
}
