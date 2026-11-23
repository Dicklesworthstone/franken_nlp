//! Model-free pinned planning and private scripted scheduling regressions.
//! No executed-test, neural quality or full-context equivalence evidence.
use super::*;
use crate::{native_engine::strict_int8::STRICT_INT8_EXECUTION, tokenizer::pinned_controls};
use super::super::tests::{compact_policy, encoding_allowance, encoding_used, quoted_run, QuotedSummary, StopInsideEncoding};
struct Control { calls: usize, stop: usize }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.calls >= self.stop).then_some(DecodeCancellationKind::Deadline)
    }
}
fn go() -> Control { Control { calls: 0, stop: usize::MAX } }
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
    let d = Sha256Digest::of_bytes(b"hierarchy-planning-fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(), task_spec: SUMMARIZE_TASK_VERSION.to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn request() -> SourceSummarySynthesis {
    SourceSummarySynthesis { map_options: SummaryOptions::default(), synthesis_options: SummaryOptions::default(),
        limits: SummarySynthesisLimits { max_evidence_segments: 64, max_evidence_bytes: 4096, verification: GroundingBudget::default() } }
}
fn limits() -> SummaryHierarchyLimits { SummaryHierarchyLimits { max_passes: 4, ..SummaryHierarchyLimits::default() } }
fn mapping() -> Int8SourceMapLimits {
    Int8SourceMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 6, max_chunk_tokens: 6,
        context_tokens: 8192, reserved_tokens: 64, max_chunks: 64, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
        max_model_work: Int8Work::for_sequence(0, 65_536, 65_536 * 166_144).unwrap(),
        mask_limits: MaskWorkLimits::default(), mask_visits_per_chunk: 1000, max_mask_visits: 64_000 }
}
fn prepare<'s>(p: &SourceTaskPlanner, text: &'s str, m: Int8SourceMapLimits) -> PreparedInt8SummaryHierarchy<'s> {
    let id = identity(p);
    p.plan_int8_summary_hierarchy_with_control(text, request(), limits(), budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), m, &mut go()).unwrap()
}
#[test]
fn reserves_every_pass_on_all_five_axes_and_masks_before_any_execution() {
    let p = planner(); let text = "éAéAéAéA";
    let prepared = prepare(&p, text, mapping()); let expected = prepared.preflight_metadata();
    assert_eq!(expected.chunk_count(), 2); assert_eq!(expected.reserved_mask_visits(), 6000);
    let single = &prepared.base;
    let mut work = single.expected.discovery_work;
    for _ in 0..4 { work = add_work(work, single.expected.synthesis_reserve).unwrap(); }
    assert_eq!(work, expected.reserved_model_work());
    let mut exact = mapping(); exact.max_model_work = work; exact.max_mask_visits = expected.reserved_mask_visits();
    prepare(&p, text, exact);
    for axis in 0..6 {
        let mut short = exact;
        match axis { 0 => short.max_model_work.forward_positions -= 1, 1 => short.max_model_work.projected_logits -= 1,
            2 => short.max_model_work.attention_pairs -= 1, 3 => short.max_model_work.projections.dot_products -= 1,
            4 => short.max_model_work.projections.multiply_accumulates -= 1, _ => short.max_mask_visits -= 1 }
        let id = identity(&p);
        assert!(p.plan_int8_summary_hierarchy_with_control(text, request(), limits(), budget(),
            &PlanContext::new(&id, budget()).unwrap(), SourcePlanningLimits::default(), short, &mut go()).is_err(), "{axis}");
    }
}
const SOURCE: &str = "éabc----βdef";
fn setup() -> (Int8SummaryHierarchyPreflight, evidence::Collection) {
    let p = planner(); let mut expected = prepare(&p, SOURCE, mapping()).preflight_metadata();
    // Get a REAL pinned scaffold with exactly six available source tokens.
    // The private scheduler script below deliberately counts bytes for fixtures.
    let mut b = budget(); b.max_input_tokens = (expected.capacity.scaffold_tokens() + 6).try_into().unwrap();
    let id = identity(&p);
    expected.capacity = p.int8_map_capacity_with_control(&SourceMapTask::Summarize(request().synthesis_options), b,
        &PlanContext::new(&id, b).unwrap(), SourcePlanningLimits::default(), &mut go()).unwrap();
    assert_eq!(expected.capacity.max_source_tokens(), 6);
    let mut collection = evidence::Collection::empty();
    let mut remaining = GroundingBudget::default();
    for quote in ["éabc", "βdef"] {
        let spans = scan_occurrences(SOURCE, quote, &mut GroundingBudget::default()).unwrap();
        let bullet = CitedBullet { text: "not-source-facts".to_owned(), citations: vec![SourceCitation {
            quote: quote.to_owned(), occurrence: SourceOccurrence::Anchored, spans }] };
        collection.append_verified(SOURCE, &[bullet], request().synthesis_options, request().limits, &mut remaining, &mut go()).unwrap();
    }
    (expected, collection)
}
struct Script { expected: Int8SummaryHierarchyPreflight, inputs: Vec<String>, full: bool, empty: bool }
impl execution::Driver for Script {
    fn count<C: DecodeStepControl>(&mut self, text: &str, _: &mut C) -> Result<usize, Int8SourceMapError> { Ok(text.len()) }
    fn run<C: DecodeStepControl>(&mut self, text: &str, _: &mut C) -> Result<(Int8SourceTaskRun, Int8Work), Int8SourceMapError> {
        self.inputs.push(text.to_owned());
        let quote = if self.full { text.to_owned() } else { text.chars().next().unwrap().to_string() };
        let spans = scan_occurrences(text, &quote, &mut GroundingBudget::default()).unwrap();
        let prompt = self.expected.capacity.scaffold_tokens() + text.len();
        let work = constrained_int8::planned_work(prompt, 2).unwrap();
        let raw = SummaryResult { schema_version: 1, task_spec_version: SUMMARIZE_TASK_VERSION.to_owned(), numerics_profile: STRICT_INT8_PROFILE.to_owned(),
            citation_guarantee: CitationGuarantee::StructuralSourceMembership, semantic_support: SummarySemanticSupport::NotAssessed,
            score_space: ScoreSpace::NotComputed, bullets: if self.empty { vec![] } else { vec![CitedBullet {
                text: format!("generated-{quote}"), citations: vec![SourceCitation { quote,
                    occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }] }] },
            generated_token_ids: vec![1, 0], forward_positions: work.forward_positions, projected_logits: work.projected_logits, mask_node_visit_charge: 1 };
        Ok((Int8SourceTaskRun { schema_version: 1, execution: INT8_SOURCE_EXECUTION.to_owned(), model_work: work,
            result: SourceTaskResult::Summarize(raw) }, constrained_int8::planned_work(prompt, self.expected.output_tokens).unwrap()))
    }
}
fn script(expected: Int8SummaryHierarchyPreflight) -> Script { Script { expected, inputs: Vec::new(), full: false, empty: false } }
#[test]
fn multi_context_quotes_reduce_to_one_summary_without_feeding_back_generated_assertions() {
    let (expected, collection) = setup(); let mut driver = script(expected);
    let reduced = execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
        &mut GroundingBudget::default(), &mut driver, &mut go()).unwrap();
    assert_eq!(driver.inputs, ["éabc", "βdef", "é\n\nβ"]);
    assert_eq!(reduced.levels.len(), 2); assert_eq!(reduced.final_pass, Some(2));
    assert_eq!(reduced.status, SummarySynthesisStatus::Synthesized);
    assert_eq!(reduced.passes[2].bullets[0].citations[0].spans[0], VerifiedSourceSpan {
        byte_start: 0, byte_end: 2, scalar_start: 0, scalar_end: 1 });
    for pass in &reduced.passes { receipt::verify_pass(&expected, pass, &mut go()).unwrap(); }
    assert_eq!(reduced.tokenizer, HierarchyTokenizerWork { calls: 5, bytes: 30 });
}
#[test]
fn nonprogress_fails_without_a_final_call_or_partial_success() {
    let (expected, collection) = setup(); let mut driver = script(expected); driver.full = true;
    assert!(execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
        &mut GroundingBudget::default(), &mut driver, &mut go()).is_err());
    assert_eq!(driver.inputs.len(), 2);
}
#[test]
fn depth_pass_and_aggregate_tokenizer_limits_never_renew() {
    for axis in 0..4 {
        let (mut expected, collection) = setup();
        match axis { 0 => expected.limits.max_levels = 1, 1 => expected.limits.max_passes = 2,
            2 => expected.limits.max_tokenizer_calls = 4, _ => expected.limits.max_tokenizer_bytes = 29 }
        let mut driver = script(expected);
        assert!(execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
            &mut GroundingBudget::default(), &mut driver, &mut go()).is_err());
        assert_eq!(driver.inputs.len(), if axis < 2 { 0 } else { 2 });
    }
}
#[test]
fn exhaustion_is_typed_and_all_empty_group_outputs_remain_visible() {
    let (expected, collection) = setup(); let mut driver = script(expected); driver.empty = true;
    let result = execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
        &mut GroundingBudget::default(), &mut driver, &mut go()).unwrap();
    assert_eq!(result.status, SummarySynthesisStatus::NoBulletsProduced); assert_eq!(result.final_pass, None);
    assert_eq!(result.passes.len(), 2); assert_eq!(result.levels[0].next_evidence, Some(EvidenceSize { segments: 0, bytes: 0 }));
}
#[test]
fn final_cancellation_retains_kind_after_all_native_and_verification_work() {
    let (expected, collection) = setup(); let mut driver = script(expected); let mut control = go();
    execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
        &mut GroundingBudget::default(), &mut driver, &mut control).unwrap();
    let (expected, collection) = setup(); let mut driver = script(expected); let mut cancel = Control { calls: 0, stop: control.calls };
    let error = execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
        &mut GroundingBudget::default(), &mut driver, &mut cancel).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(driver.inputs.len(), 3);
}
#[test]
fn every_completed_pass_checks_native_work_and_original_source_bounds() {
    for axis in 0..6 {
        let (expected, collection) = setup(); let mut driver = script(expected);
        let mut result = execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
            &mut GroundingBudget::default(), &mut driver, &mut go()).unwrap();
        let pass = &mut result.passes[2];
        match axis { 0 => pass.native.model_work.forward_positions += 1, 1 => pass.native.model_work.projected_logits += 1,
            2 => pass.native.model_work.attention_pairs += 1, 3 => pass.native.model_work.projections.dot_products += 1,
            4 => pass.native.model_work.projections.multiply_accumulates += 1,
            _ => pass.bullets[0].citations[0].spans[0].byte_end = SOURCE.len() + 1 }
        assert!(receipt::verify_pass(&expected, pass, &mut go()).is_err(), "{axis}");
    }
}
#[test]
fn verification_work_and_complete_output_caps_apply_across_levels() {
    let (expected, collection) = setup(); let mut driver = script(expected); let mut remaining = GroundingBudget::default();
    execution::reduce(SOURCE, collection, request(), expected, 4 << 20, &mut remaining, &mut driver, &mut go()).unwrap();
    let used = GroundingBudget::default().max_scan_steps - remaining.max_scan_steps;
    for output_cap in [1, 4 << 20] {
        let (expected, collection) = setup(); let mut driver = script(expected);
        let mut remaining = GroundingBudget { max_scan_steps: used - 1, ..GroundingBudget::default() };
        assert!(execution::reduce(SOURCE, collection, request(), expected, output_cap,
            &mut remaining, &mut driver, &mut go()).is_err());
    }
}

struct MapSummary<'a> { source: &'a str, empty: bool }
impl SourceDriver for MapSummary<'_> {
    fn checkpoint(&mut self) -> Result<(), Int8SourceError> { Ok(()) }
    fn run(&mut self, plan: &PreparedInt8SourceTask, _: &ExecutionIdentity) -> Result<Int8SourceTaskRun, Int8SourceError> {
        let work = constrained_int8::planned_work(plan.prompt_tokens(), 2).unwrap();
        let bullets = if self.empty { Vec::new() } else {
            vec![CitedBullet { text: "discovery candidate".to_owned(), citations: vec![SourceCitation {
                quote: self.source.to_owned(), occurrence: SourceOccurrence::Anchored,
                spans: scan_occurrences(self.source, self.source, &mut GroundingBudget::default()).unwrap() }] }]
        };
        let raw = SummaryResult { schema_version: 1, task_spec_version: SUMMARIZE_TASK_VERSION.to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), citation_guarantee: CitationGuarantee::StructuralSourceMembership,
            semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
            bullets, generated_token_ids: vec![1, 0], forward_positions: work.forward_positions,
            projected_logits: work.projected_logits, mask_node_visit_charge: 20 };
        Ok(Int8SourceTaskRun { schema_version: 1, execution: INT8_SOURCE_EXECUTION.to_owned(),
            result: SourceTaskResult::Summarize(raw), model_work: work })
    }
}
fn completion(p: &SourceTaskPlanner, text: &str, empty: bool)
    -> (Int8SummaryHierarchyPreflight, Int8SummaryHierarchyRun) {
    let prepared = prepare(p, text, mapping()); let expected = prepared.expected;
    assert_eq!(expected.chunk_count(), 1);
    let admitted: Vec<_> = prepared.execution_identities().cloned().collect();
    let discovery = prepared.base.map.execute_with_driver(&admitted, MapSummary { source: text, empty }).unwrap();
    let mut remaining = request().limits.verification;
    let collection = evidence::collect(text, discovery.mapped.root().value(), request().map_options,
        request().limits, &mut remaining, &mut go()).unwrap();
    let initial = EvidenceSize::of(&collection);
    let reduced = execution::reduce(text, collection, request(), expected, mapping().reduction.max_result_bytes as u64,
        &mut remaining, &mut script(expected), &mut go()).unwrap();
    (expected, finish(discovery, initial, reduced, remaining, expected, &mut go()).unwrap())
}
#[test]
fn completed_hierarchy_is_bound_to_source_options_and_the_original_global_allowance() {
    let p = planner(); let (expected, run) = completion(&p, "Alice", false);
    expected.verify_completed(&run).unwrap();
    let foreign = prepare(&p, "Carol", mapping()).preflight_metadata();
    assert_eq!(foreign.source_span(), expected.source_span());
    assert_eq!(foreign.reserved_model_work(), expected.reserved_model_work());
    assert_eq!(foreign.reserved_mask_visits(), expected.reserved_mask_visits());
    assert!(foreign.verify_completed(&run).is_err());
    let mut smaller = mapping(); smaller.max_mask_visits -= 1;
    let other = prepare(&p, "Alice", smaller).preflight_metadata();
    assert_eq!(other.reserved_mask_visits(), expected.reserved_mask_visits());
    assert!(other.verify_completed(&run).is_err());

    let (_, empty) = completion(&p, "Alice", true);
    let id = identity(&p); let context = PlanContext::new(&id, budget()).unwrap();
    for change_options in [false, true] {
        let mut changed = request();
        if change_options { changed.synthesis_options.max_bullets = 1; }
        else { changed.limits.verification.max_matches -= 1; }
        let other = p.plan_int8_summary_hierarchy_with_control("Alice", changed, limits(), budget(),
            &context, SourcePlanningLimits::default(), mapping(), &mut go()).unwrap().preflight_metadata();
        assert!(other.verify_completed(&empty).is_err());
    }
}
#[test]
fn hierarchy_seal_rejects_changed_final_text_quotes_and_tokens_that_pass_structural_checks() {
    let p = planner();
    for axis in 0..3 {
        let (expected, mut run) = completion(&p, "Alice", false);
        let pass = &mut run.passes[0];
        let SourceTaskResult::Summarize(raw) = &mut pass.native.result else { panic!("summary fixture"); };
        match axis {
            0 => {
                raw.bullets[0].text = "rewritten".to_owned();
                pass.bullets[0].text = raw.bullets[0].text.clone();
            }
            1 => {
                let span = VerifiedSourceSpan { byte_start: 1, byte_end: 2, scalar_start: 1, scalar_end: 2 };
                raw.bullets[0].citations[0].quote = "l".to_owned();
                raw.bullets[0].citations[0].spans = vec![span];
                pass.bullets[0].citations[0].quote = "l".to_owned();
                pass.bullets[0].citations[0].spans = vec![span];
            }
            _ => raw.generated_token_ids[0] = 2,
        }
        // The old header, geometry, native-work and lineage checks all accept
        // these edits. Exact completion binding must still reject them.
        receipt::verify(&expected, &run, &mut go()).unwrap();
        assert!(expected.verify_completed(&run).is_err(), "{axis}");
    }
}
#[test]
fn hierarchy_completion_preserves_cancellation_during_binding_and_lineage_verification() {
    struct CancelAt { cause: DecodeCancellationKind, stop: usize, polls: usize }
    impl DecodeStepControl for CancelAt {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.polls += 1; (self.polls == self.stop).then_some(self.cause)
        }
    }
    use DecodeCancellationKind::*;
    let p = planner(); let (expected, run) = completion(&p, "Alice", false);
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for stop in [1, 3, 7] {
            let mut control = CancelAt { cause, stop, polls: 0 };
            assert_eq!(expected.verify_completed_with_control(&run, &mut control).unwrap_err().cancellation(), Some(cause));
            assert_eq!(control.polls, stop);
        }
    }
}
#[test]
fn hierarchy_private_hashes_do_not_change_public_output_or_its_exact_byte_budget() {
    let p = planner(); let (mut expected, run) = completion(&p, "Alice", true);
    let json = crate::canonjson::canonical_string(&run).unwrap(); let debug = format!("{expected:?}");
    for private in ["request_binding", "result_binding", "Alice"] {
        assert!(!json.contains(private)); assert!(!debug.contains(private));
    }
    expected.max_result_bytes = json.len() as u64;
    expected.verify_completed(&run).unwrap();
    expected.max_result_bytes -= 1;
    assert!(expected.verify_completed(&run).is_err());
}

const FIRST_QUOTE: &str = "The quick brown fox jumps over the lazy dog.";
const SECOND_QUOTE: &str = "A blue bird sings softly near the window.";
const COMPACT_DOCUMENT: &str = "The quick brown fox jumps over the lazy dog. -- A blue bird sings softly near the window.";

struct CompactScript<'a> {
    planner: &'a SourceTaskPlanner, identity: &'a ExecutionIdentity, inputs: Vec<String>,
}
impl execution::Driver for CompactScript<'_> {
    fn count<C: DecodeStepControl>(&mut self, text: &str, control: &mut C) -> Result<usize, Int8SourceMapError> {
        execution::count_source(self.planner, SourcePlanningLimits::default(), text, control)
    }
    fn run<C: DecodeStepControl>(&mut self, text: &str, control: &mut C)
        -> Result<(Int8SourceTaskRun, Int8Work), Int8SourceMapError> {
        let request = SourceTaskRequest::Summarize { document: text.to_owned(), options: request().synthesis_options, budget: budget() };
        let context = PlanContext::new(self.identity, budget()).unwrap();
        let plan = self.planner.plan_int8_with_control(&request, &context, SourcePlanningLimits::default(), control)?;
        self.inputs.push(text.to_owned());
        let quote = text.chars().next().unwrap().to_string();
        Ok((quoted_run(&plan, text, &[&quote]), plan.planned_work()))
    }
}
fn compact_planner(work: crate::tokenizer::untrusted::SafeBpeWorkLimits) -> SourceTaskPlanner {
    planner().with_map_document_encoding(compact_policy(), work).unwrap()
}
fn compact_completion(p: &SourceTaskPlanner)
    -> (Int8SummaryHierarchyPreflight, Result<Int8SummaryHierarchyRun, Int8SourceMapError>, Vec<String>, usize) {
    // Measure the real compact joined texts on an independent fixture counter,
    // then narrow the private scheduler capacity to force two reduction levels.
    // Actual discovery, fit probes and node plans all use the original operation.
    let counter = compact_planner(encoding_allowance());
    let count = |text: &str| execution::count_source(&counter, SourcePlanningLimits::default(), text, &mut go()).unwrap();
    let cap = count(FIRST_QUOTE).max(count(SECOND_QUOTE));
    assert!(count(&format!("{FIRST_QUOTE}\n\n{SECOND_QUOTE}")) > cap);
    let mut mapping = mapping(); mapping.chunks.max_chunk_bytes = 4096; mapping.chunks.max_chunk_tokens = 4096;
    let prepared = prepare(p, COMPACT_DOCUMENT, mapping); let mut expected = prepared.expected;
    assert_eq!(expected.chunk_count(), 1);
    let mut smaller = budget(); smaller.max_input_tokens = (expected.capacity.scaffold_tokens() + cap).try_into().unwrap();
    let id = identity(p);
    expected.capacity = p.int8_map_capacity_with_control(&SourceMapTask::Summarize(request().synthesis_options), smaller,
        &PlanContext::new(&id, smaller).unwrap(), SourcePlanningLimits::default(), &mut go()).unwrap();
    assert_eq!(expected.capacity.max_source_tokens(), cap);
    let admitted: Vec<_> = prepared.execution_identities().cloned().collect();
    let (map, tail) = prepared.base.into_parts();
    let discovery = map.execute_with_driver(&admitted, QuotedSummary {
        document: COMPACT_DOCUMENT, quotes: &[FIRST_QUOTE, SECOND_QUOTE] }).unwrap();
    expected.discovery.discovery.verify_completed(&discovery).unwrap();
    let mut remaining = request().limits.verification;
    let collection = evidence::collect(COMPACT_DOCUMENT, discovery.mapped.root().value(), request().map_options,
        request().limits, &mut remaining, &mut go()).unwrap();
    let initial = EvidenceSize::of(&collection); let before = encoding_used(p).input_bytes();
    let mut driver = CompactScript { planner: &tail.operation, identity: &tail.identity, inputs: Vec::new() };
    let result = execution::reduce(COMPACT_DOCUMENT, collection, request(), expected, mapping.reduction.max_result_bytes as u64,
        &mut remaining, &mut driver, &mut go())
        .and_then(|reduced| finish(discovery, initial, reduced, remaining, expected, &mut go()));
    (expected, result, driver.inputs, before)
}
#[test]
fn compact_hierarchy_debits_rejected_joins_and_all_nodes_on_the_original_operation() {
    let p = compact_planner(encoding_allowance()); let (expected, result, inputs, before) = compact_completion(&p);
    let run = result.unwrap(); expected.verify_completed(&run).unwrap();
    assert_eq!(inputs, [FIRST_QUOTE, SECOND_QUOTE, "T\n\nA"]);
    assert_eq!(run.levels.len(), 2); assert_eq!(run.final_pass, Some(2));
    assert_eq!(run.status, SummarySynthesisStatus::Synthesized);
    assert_eq!(run.passes[2].bullets[0].citations[0].quote, "T");
    assert_eq!(run.passes[2].bullets[0].citations[0].spans[0], VerifiedSourceSpan {
        byte_start: 0, byte_end: 1, scalar_start: 0, scalar_end: 1 });
    assert_eq!(run.tokenizer_work.calls, 5);
    assert_eq!(run.tokenizer_work.bytes,
        (FIRST_QUOTE.len() * 2 + SECOND_QUOTE.len() * 2 + 2 + 1 + "T\n\nA".len()) as u64);
    let native_bytes: usize = run.passes.iter().map(|pass| pass.input_bytes).sum();
    assert_eq!(before, COMPACT_DOCUMENT.len() * 2);
    assert_eq!(encoding_used(&p).input_bytes(), before + run.tokenizer_work.bytes as usize + native_bytes);
    assert!(run.passes[..2].iter().all(|pass| pass.input_tokens < pass.input_bytes));
    for pass in &run.passes {
        assert!(pass.input_tokens <= pass.input_bytes);
        receipt::verify_pass(&expected, pass, &mut go()).unwrap();
    }
}
#[test]
fn all_compact_work_limits_remain_spent_across_hierarchy_levels() {
    let baseline = compact_planner(encoding_allowance()); let (_, result, _, _) = compact_completion(&baseline);
    result.unwrap(); let used = encoding_used(&baseline);
    for axis in 0..4 {
        let mut cap = encoding_allowance();
        match axis { 0 => cap.max_input_bytes = used.input_bytes() as u64 - 1,
            1 => cap.max_output_tokens = used.output_tokens() as u64 - 1,
            2 => cap.max_pair_checks = used.pair_checks() - 1, _ => cap.max_pair_bytes = used.pair_bytes() - 1 }
        let p = compact_planner(cap); let (_, result, calls, before) = compact_completion(&p);
        assert!(result.is_err(), "{axis}"); assert!(calls.len() < 3, "{axis}");
        assert!(encoding_used(&p).input_bytes() >= before);
    }
}
#[test]
fn compact_hierarchy_fit_probes_preserve_every_cancellation_kind_and_never_refund() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let p = compact_planner(encoding_allowance()); let retained = p.retain_map_document_operation().unwrap();
        let mut control = StopInsideEncoding { planner: &p, cause, busy_polls: 0 };
        let error = execution::count_source(&retained, SourcePlanningLimits::default(), COMPACT_DOCUMENT, &mut control).unwrap_err();
        assert_eq!(error.cancellation(), Some(cause)); assert_eq!(control.busy_polls, 8);
        assert_eq!(encoding_used(&p).input_bytes(), COMPACT_DOCUMENT.len());
        let before = encoding_used(&p);
        execution::count_source(&retained, SourcePlanningLimits::default(), COMPACT_DOCUMENT, &mut go()).unwrap();
        assert_eq!(encoding_used(&p).input_bytes(), before.input_bytes() + COMPACT_DOCUMENT.len());
        assert!(encoding_used(&p).pair_checks() >= before.pair_checks());
    }
}
