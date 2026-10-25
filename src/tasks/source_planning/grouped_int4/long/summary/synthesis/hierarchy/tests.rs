//! Private planning/scheduling/evidence fixtures; no native model execution or
//! quality evidence. The scripted tokenizer counts bytes only inside this seam.
use super::*;
use crate::tasks::source_planning::grouped_int4::long::tests as maps;

struct Control { calls: usize, stop: usize }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1; (self.calls >= self.stop).then_some(DecodeCancellationKind::Deadline)
    }
}
fn go() -> Control { Control { calls: 0, stop: usize::MAX } }
fn request() -> SourceSummarySynthesis {
    SourceSummarySynthesis { map_options: SummaryOptions::default(), synthesis_options: SummaryOptions::default(),
        limits: SummarySynthesisLimits { max_evidence_segments: 64, max_evidence_bytes: 4096, verification: GroundingBudget::default() } }
}
fn limits() -> SummaryHierarchyLimits { SummaryHierarchyLimits { max_passes: 4, ..SummaryHierarchyLimits::default() } }
fn prepare<'s>(p: &SourceTaskPlanner, text: &'s str, mapping: Int4SourceMapLimits,
    hierarchy: SummaryHierarchyLimits, width: usize) -> Result<PreparedInt4SummaryHierarchy<'s>, Int4SourceMapError> {
    let id = maps::identity(p, &SourceMapTask::Summarize(request().map_options), width); let budget = maps::budget();
    p.plan_int4_summary_hierarchy_with_control(text, request(), hierarchy, budget, &PlanContext::new(&id, budget).unwrap(),
        SourcePlanningLimits::default(), mapping, width, &mut go())
}
fn capacity(expected: &mut Int4SummaryHierarchyPreflight, available: usize) {
    let p = maps::planner(); let id = maps::identity(&p, &SourceMapTask::Summarize(request().synthesis_options), expected.group_width());
    let mut budget = maps::budget(); budget.max_input_tokens = (expected.capacity.scaffold_tokens() + available).try_into().unwrap();
    expected.capacity = p.int4_map_capacity_with_control(&SourceMapTask::Summarize(request().synthesis_options), budget,
        &PlanContext::new(&id, budget).unwrap(), SourcePlanningLimits::default(), expected.group_width(), &mut go()).unwrap();
}
const SOURCE: &str = "éabc----βdef";
fn setup(empty: bool) -> (Int4SummaryHierarchyPreflight, Int4SourceMapRun, evidence::Collection, GroundingBudget) {
    let p = maps::planner(); let prepared = prepare(&p, SOURCE, maps::limits(), limits(), 32).unwrap();
    let mut expected = prepared.expected; capacity(&mut expected, 6);
    let admitted = maps::admitted(&prepared.base.map); let mut driver = maps::script(&prepared.base.map);
    assert_eq!(driver.outputs.len(), 2);
    for ((output, chunk), quote) in driver.outputs.iter_mut().zip(prepared.base.map.chunks.chunks()).zip(["éabc", "βdef"]) {
        let SourceTaskResult::Summarize(raw) = output else { panic!() };
        let spans = scan_occurrences(chunk.text(), quote, &mut GroundingBudget::default()).unwrap();
        raw.bullets = if empty { vec![] } else { vec![CitedBullet { text: "MAP_ASSERTION_MUST_NOT_ENTER_NEXT_PASS".to_owned(),
            citations: vec![SourceCitation { quote: quote.to_owned(), occurrence: SourceOccurrence::Anchored, spans }] }] };
    }
    let discovery = prepared.base.map.execute_with_driver(&admitted, driver).unwrap();
    let mut remaining = remaining_verification(request().limits.verification, discovery.verification_used).unwrap();
    let evidence = evidence::collect(SOURCE, discovery.mapped.root().value(), request().map_options,
        request().limits, &mut remaining, &mut go()).unwrap();
    (expected, discovery, evidence, remaining)
}
struct Script {
    expected: Int4SummaryHierarchyPreflight, inputs: Vec<String>, preparations: Vec<String>, events: Vec<&'static str>,
    full: bool, empty: bool, fail_prepare: Option<usize>, fail_run: Option<usize>, zero_tokens: bool,
}
impl execution::Driver for Script {
    type Witness = Sha256Digest;
    fn count<C: DecodeStepControl>(&mut self, text: &str, _: &mut C) -> Result<usize, Int4SourceMapError> {
        Ok(if self.zero_tokens { 0 } else { text.len() })
    }
    fn prepare<C: DecodeStepControl>(&mut self, text: &str, _: &mut C) -> Result<Sha256Digest, Int4SourceMapError> {
        self.preparations.push(text.to_owned()); self.events.push("prepare");
        if self.fail_prepare == Some(self.preparations.len()) { return Err(Int4SourceMapError::Admission); }
        Ok(Sha256Digest::of_bytes(text.as_bytes()))
    }
    fn run<C: DecodeStepControl>(&mut self, text: &str, witness: &Sha256Digest, _: &mut C)
        -> Result<(Int4SourceTaskRun, Int4Work), Int4SourceMapError> {
        if *witness != Sha256Digest::of_bytes(text.as_bytes()) { return Err(Int4SourceMapError::Admission); }
        self.inputs.push(text.to_owned()); self.events.push("run");
        if self.fail_run == Some(self.inputs.len()) { return Err(Int4SourceError::Cancelled(DecodeCancellationKind::Deadline).into()); }
        let quote = if self.full { text.to_owned() } else { text.chars().next().unwrap().to_string() };
        let spans = scan_occurrences(text, &quote, &mut GroundingBudget::default()).unwrap();
        let prompt = self.expected.capacity.scaffold_tokens() + text.len(); let width = self.expected.group_width();
        let work = constrained_int4::planned_work(width, prompt, 2).unwrap();
        let raw = SummaryResult { schema_version: 1, task_spec_version: SUMMARIZE_TASK_VERSION.to_owned(), numerics_profile: STRICT_INT4_PROFILE.to_owned(),
            citation_guarantee: CitationGuarantee::StructuralSourceMembership, semantic_support: SummarySemanticSupport::NotAssessed,
            score_space: ScoreSpace::NotComputed, bullets: if self.empty { vec![] } else { vec![CitedBullet {
                text: format!("generated-{quote}"), citations: vec![SourceCitation { quote,
                    occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }] }] },
            generated_token_ids: vec![0, self.expected.discovery.eos_token_id], forward_positions: work.forward_positions,
            projected_logits: work.projected_logits, mask_node_visit_charge: 1 };
        Ok((Int4SourceTaskRun { schema_version: 1, execution: INT4_SOURCE_EXECUTION.to_owned(), group_width: width,
            model_work: work, result: SourceTaskResult::Summarize(raw) },
            constrained_int4::planned_work(width, prompt, self.expected.discovery.max_output_tokens).unwrap()))
    }
}
fn script(expected: Int4SummaryHierarchyPreflight) -> Script {
    Script { expected, inputs: vec![], preparations: vec![], events: vec![], full: false, empty: false,
        fail_prepare: None, fail_run: None, zero_tokens: false }
}
fn complete(empty: bool) -> (Int4SummaryHierarchyPreflight, Int4SummaryHierarchyRun) {
    let (expected, discovery, collection, mut remaining) = setup(empty); let initial = EvidenceSize::of(&collection);
    let reduced = execution::reduce(SOURCE, collection, request(), expected, 4 << 20, &mut remaining,
        &mut script(expected), &mut go()).unwrap();
    let run = finish(discovery, initial, reduced, remaining, expected, 4 << 20, &mut go()).unwrap();
    (expected, run)
}
fn decrement(work: &mut Int4Work, axis: usize) {
    match axis { 0 => work.forward_positions -= 1, 1 => work.projected_logits -= 1,
        2 => work.attention_pairs -= 1, 3 => work.projections.dot_products -= 1, _ => work.projections.multiply_accumulates -= 1 }
}

#[test]
fn every_permitted_pass_is_reserved_on_all_five_axes_and_masks_before_discovery() {
    let p = maps::planner(); let prepared = prepare(&p, SOURCE, maps::limits(), limits(), 32).unwrap(); let expected = prepared.expected;
    assert_eq!(expected.chunk_count(), 2); assert_eq!(expected.reserved_mask_visits(), 6000);
    let mut work = prepared.base.expected.discovery.planned_work();
    for _ in 0..4 { work = add_work(work, prepared.base.expected.synthesis_reserve).unwrap(); }
    assert_eq!(work, expected.reserved_model_work());
    let mut exact = maps::limits(); exact.max_model_work = work; exact.max_mask_visits = expected.reserved_mask_visits();
    prepare(&p, SOURCE, exact, limits(), 32).unwrap();
    for axis in 0..6 {
        let mut short = exact;
        if axis < 5 { decrement(&mut short.max_model_work, axis); } else { short.max_mask_visits -= 1; }
        assert!(prepare(&p, SOURCE, short, limits(), 32).is_err(), "{axis}");
    }
}
#[test]
fn each_actual_group_partition_changes_reserved_dot_work_without_relabeling_int8() {
    let p = maps::planner(); let mut previous = None;
    for width in [16, 32, 127] {
        let prepared = prepare(&p, SOURCE, maps::limits(), limits(), width).unwrap();
        assert_eq!(prepared.group_width(), width);
        if let Some(old) = previous { assert!(old > prepared.expected.reserved_work.projections.dot_products); }
        previous = Some(prepared.expected.reserved_work.projections.dot_products);
        for id in prepared.execution_identities() { assert_eq!(id.numerics_profile, NumericsProfile::StrictQuantized { version: 2 }); }
    }
}
#[test]
fn multi_context_frontier_reaches_one_final_summary_without_feeding_generated_assertions() {
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = script(expected);
    let reduced = execution::reduce(SOURCE, collection, request(), expected, 4 << 20, &mut remaining, &mut driver, &mut go()).unwrap();
    assert_eq!(driver.inputs, ["éabc", "βdef", "é\n\nβ"]);
    assert_eq!(driver.events, ["prepare", "prepare", "run", "run", "prepare", "run"]);
    assert_eq!(reduced.levels.len(), 2); assert_eq!(reduced.final_pass, Some(2));
    assert_eq!(reduced.status, SummarySynthesisStatus::Synthesized);
    assert_eq!(reduced.passes[2].bullets[0].citations[0].spans[0], VerifiedSourceSpan { byte_start: 0, byte_end: 2, scalar_start: 0, scalar_end: 1 });
    assert_eq!(reduced.tokenizer, HierarchyTokenizerWork { calls: 5, bytes: 30 });
}
#[test]
fn last_group_admission_failure_prevents_every_native_call_in_that_level() {
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = script(expected); driver.fail_prepare = Some(2);
    assert!(matches!(execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
        &mut remaining, &mut driver, &mut go()), Err(Int4SourceMapError::Admission)));
    assert_eq!(driver.preparations.len(), 2); assert!(driver.inputs.is_empty());
}
#[test]
fn unchanged_quote_frontier_refuses_nonprogress_without_a_final_call() {
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = script(expected); driver.full = true;
    assert!(execution::reduce(SOURCE, collection, request(), expected, 4 << 20, &mut remaining, &mut driver, &mut go()).is_err());
    assert_eq!(driver.inputs.len(), 2);
}
#[test]
fn depth_pass_tokenizer_call_and_byte_limits_never_renew_between_levels() {
    for axis in 0..4 {
        let (mut expected, _, collection, mut remaining) = setup(false);
        match axis { 0 => expected.limits.max_levels = 1, 1 => expected.limits.max_passes = 2,
            2 => expected.limits.max_tokenizer_calls = 4, _ => expected.limits.max_tokenizer_bytes = 29 }
        let mut driver = script(expected);
        assert!(execution::reduce(SOURCE, collection, request(), expected, 4 << 20, &mut remaining, &mut driver, &mut go()).is_err());
        assert_eq!(driver.inputs.len(), if axis < 2 { 0 } else { 2 });
    }
}
#[test]
fn atomic_oversize_quote_and_zero_token_counter_refuse_before_native_calls() {
    for zero in [false, true] {
        let (mut expected, _, collection, mut remaining) = setup(false);
        if !zero { capacity(&mut expected, 4); }
        let mut driver = script(expected); driver.zero_tokens = zero;
        assert!(execution::reduce(SOURCE, collection, request(), expected, 4 << 20, &mut remaining, &mut driver, &mut go()).is_err());
        assert!(driver.inputs.is_empty()); assert!(driver.preparations.is_empty());
    }
}
#[test]
fn all_empty_group_outputs_remain_visible_and_exhaustion_is_distinct_from_no_initial_evidence() {
    let (expected, discovery, collection, mut remaining) = setup(false); let initial = EvidenceSize::of(&collection);
    let mut driver = script(expected); driver.empty = true;
    let result = execution::reduce(SOURCE, collection, request(), expected, 4 << 20, &mut remaining, &mut driver, &mut go()).unwrap();
    assert_eq!(result.status, SummarySynthesisStatus::NoBulletsProduced); assert_eq!(result.final_pass, None);
    assert_eq!(result.passes.len(), 2); assert_eq!(result.levels[0].next_evidence, Some(EvidenceSize { segments: 0, bytes: 0 }));
    let run = finish(discovery, initial, result, remaining, expected, 4 << 20, &mut go()).unwrap();
    assert!(run.bullets().is_empty()); assert_eq!(run.passes.len(), 2);
    let (_, empty) = complete(true);
    assert_eq!(empty.status, SummarySynthesisStatus::NoEvidenceCollected); assert!(empty.passes.is_empty());
    assert!(empty.reserved_model_work.forward_positions > empty.model_work.forward_positions);
}
#[test]
fn complete_receipt_retains_discovery_every_pass_final_index_and_cumulative_verification() {
    let (expected, run) = complete(false); expected.verify_completed(&run).unwrap();
    assert_eq!(run.discovery.mapped.root().value().len(), 2); assert_eq!(run.passes.len(), 3);
    assert_eq!(run.final_pass, Some(2)); assert_eq!(run.bullets()[0].text, "generated-é");
    assert!(run.verification_used.fields > run.discovery.verification_used.fields);
    assert!(run.verification_used.matches > run.discovery.verification_used.matches);
    assert!(run.verification_used.scan_steps > run.discovery.verification_used.scan_steps);
    assert_eq!(run.initial_evidence, EvidenceSize { segments: 2, bytes: 12 });
    assert_eq!(run.levels[0].next_evidence, Some(EvidenceSize { segments: 2, bytes: 6 }));
}
#[test]
fn all_five_pass_work_axes_group_eos_and_original_bounds_are_independently_checked() {
    for axis in 0..9 {
        let (expected, mut run) = complete(false); let pass = &mut run.passes[2];
        if axis < 5 { decrement(&mut pass.native.model_work, axis); }
        else if axis == 5 { pass.native.group_width = 16; }
        else if axis == 6 { pass.input_tokens += 1; }
        else if axis == 7 {
            let SourceTaskResult::Summarize(raw) = &mut pass.native.result else { panic!() }; raw.generated_token_ids.pop();
        } else { pass.bullets[0].citations[0].spans[0].byte_end = SOURCE.len() + 1; }
        assert!(receipt::verify_pass(&expected, pass).is_err(), "{axis}");
    }
}
#[test]
fn level_coverage_gaps_reordering_or_invented_frontier_geometry_are_rejected() {
    for axis in 0..7 {
        let (expected, mut run) = complete(false);
        match axis { 0 => run.passes[1].input_segments.start = 0, 1 => run.passes[0].group = 1,
            2 => run.passes[0].input_bytes += 1, 3 => run.levels[0].next_evidence.as_mut().unwrap().bytes += 1,
            4 => run.levels[1].passes.start = 1, 5 => run.levels[1].input.segments += 1,
            _ => run.final_pass = Some(1) }
        assert!(expected.verify_completed(&run).is_err(), "{axis}");
    }
}
#[test]
fn completed_total_work_masks_verification_and_initial_evidence_cannot_be_rewritten() {
    for axis in 0..8 {
        let (expected, mut run) = complete(false);
        match axis { 0 => run.group_width = 16, 1 => run.discovery.group_width = 16,
            2 => run.model_work.projections.dot_products += 1, 3 => run.mask_node_visit_charge += 1,
            4 => run.verification_used.scan_steps = 0, 5 => run.initial_evidence.segments += 1,
            6 => run.tokenizer_work.calls = 0, _ => run.tokenizer_work.bytes = 0 }
        assert!(expected.verify_completed(&run).is_err(), "{axis}");
    }
}
#[test]
fn later_native_failure_and_final_cancellation_publish_no_partial_success() {
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = script(expected); driver.fail_run = Some(2);
    let error = execution::reduce(SOURCE, collection, request(), expected, 4 << 20, &mut remaining, &mut driver, &mut go()).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(driver.inputs.len(), 2);
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = script(expected); let mut count = go();
    execution::reduce(SOURCE, collection, request(), expected, 4 << 20, &mut remaining, &mut driver, &mut count).unwrap();
    let (expected, _, collection, mut remaining) = setup(false); let mut driver = script(expected);
    let error = execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
        &mut remaining, &mut driver, &mut Control { calls: 0, stop: count.calls }).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(driver.inputs.len(), 3);
}
#[test]
fn all_three_verification_axes_and_complete_output_caps_span_every_level() {
    let (expected, _, collection, start) = setup(false); let mut remaining = start;
    execution::reduce(SOURCE, collection, request(), expected, 4 << 20, &mut remaining, &mut script(expected), &mut go()).unwrap();
    let used = SourceMapVerification::used(start, remaining).unwrap();
    for axis in 0..4 {
        let (expected, _, collection, mut remaining) = setup(false);
        match axis { 0 => remaining.max_fields = used.fields - 1, 1 => remaining.max_matches = used.matches - 1,
            2 => remaining.max_scan_steps = used.scan_steps - 1, _ => () }
        assert!(execution::reduce(SOURCE, collection, request(), expected, if axis == 3 { 1 } else { 4 << 20 },
            &mut remaining, &mut script(expected), &mut go()).is_err(), "{axis}");
    }
}
#[test]
fn evidence_windows_rebase_byte_and_scalar_extents_without_changing_origins() {
    let (_, _, collection, _) = setup(false);
    let window = collection.window(1..2, &mut go()).unwrap();
    assert_eq!(window.text, "βdef"); assert_eq!(window.segments[0].extent,
        VerifiedSourceSpan { byte_start: 0, byte_end: 5, scalar_start: 0, scalar_end: 4 });
    assert_eq!(window.segments[0].origins[0], VerifiedSourceSpan { byte_start: 9, byte_end: 14, scalar_start: 8, scalar_end: 12 });
    assert!(collection.window(1..3, &mut go()).is_err()); assert!(collection.window(1..1, &mut go()).is_err());
}
#[test]
fn transport_verifies_duplicate_origins_before_dedup_and_charges_each_copy() {
    let citation = SourceCitation { quote: "é".to_owned(), occurrence: SourceOccurrence::Anchored,
        spans: vec![VerifiedSourceSpan { byte_start: 0, byte_end: 2, scalar_start: 0, scalar_end: 1 }] };
    let mut bullets = vec![CitedBullet { text: "not evidence".to_owned(), citations: vec![citation.clone(), citation] }];
    let mut budget = GroundingBudget { max_fields: 2, max_matches: 2, max_scan_steps: 4 };
    let mut collection = evidence::Collection::empty();
    collection.append_verified(SOURCE, &bullets, request().synthesis_options, request().limits, &mut budget, &mut go()).unwrap();
    assert_eq!(collection.segment_count(), 1); assert_eq!(collection.text, "é");
    assert_eq!(budget, GroundingBudget { max_fields: 0, max_matches: 0, max_scan_steps: 0 });
    bullets[0].citations[1].spans[0].byte_start = 9; bullets[0].citations[1].spans[0].byte_end = 11;
    assert!(evidence::Collection::empty().append_verified(SOURCE, &bullets, request().synthesis_options,
        request().limits, &mut GroundingBudget::default(), &mut go()).is_err());
}
#[test]
fn final_publication_limit_and_last_checkpoint_apply_after_complete_reduction() {
    for late in [false, true] {
        let (expected, discovery, collection, mut remaining) = setup(false); let initial = EvidenceSize::of(&collection);
        let reduced = execution::reduce(SOURCE, collection, request(), expected, 4 << 20,
            &mut remaining, &mut script(expected), &mut go()).unwrap();
        let error = finish(discovery, initial, reduced, remaining, expected, if late { 4 << 20 } else { 1 },
            &mut Control { calls: 0, stop: if late { 2 } else { usize::MAX } }).err().unwrap();
        if late { assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); }
        else { assert!(matches!(error, Int4SourceMapError::Source(_))); }
    }
}
#[test]
fn hierarchy_limits_and_first_preparation_cancellation_fail_closed() {
    let p = maps::planner();
    for axis in 0..4 {
        let mut bad = limits();
        match axis { 0 => bad.max_levels = 0, 1 => bad.max_passes = 257, 2 => bad.max_tokenizer_calls = 0,
            _ => bad.max_tokenizer_bytes = 0 }
        assert!(prepare(&p, SOURCE, maps::limits(), bad, 32).is_err());
    }
    let id = maps::identity(&p, &SourceMapTask::Summarize(request().map_options), 32); let budget = maps::budget();
    let error = p.plan_int4_summary_hierarchy_with_control(SOURCE, request(), limits(), budget,
        &PlanContext::new(&id, budget).unwrap(), SourcePlanningLimits::default(), maps::limits(), 32,
        &mut Control { calls: 0, stop: 1 }).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
}
