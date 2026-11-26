//! Real pinned planning and private synthetic-driver contract tests.
//! These are not model inference, task-quality, or throughput evidence.
use super::*;
use std::{cell::Cell, rc::Rc};
use crate::{
    native_engine::strict_int8::STRICT_INT8_EXECUTION,
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace, ner::NerResult},
    tokenizer::specials::ArchivedControlRegistries,
};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn planner() -> SourceTaskPlanner {
    let t = EmbeddedTokenizer::pinned().unwrap();
    let entries: Vec<_> = [IM_START, IM_END, THINK_START, THINK_END].iter().map(|&surface| {
        let ids = t.tokenizer().encode_ids_with_options(surface, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
        assert_eq!(ids.len(), 1);
        serde_json::json!({"id":ids[0],"special":surface == IM_START || surface == IM_END,"surface":surface})
    }).collect();
    let special: Vec<_> = entries.iter().filter(|e| e["special"] == true).cloned().collect();
    let registry = ArchivedControlRegistries::from_archived_json(
        &serde_json::json!({"schema_version":1,"registry":"TokenizerSpecialIds","entries":special}).to_string(),
        &serde_json::json!({"schema_version":1,"registry":"TemplateControlIds","entries":entries}).to_string()).unwrap();
    SourceTaskPlanner::pinned(registry.template_controls(), t.eos_token_id().unwrap()).unwrap()
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 8192, max_output_tokens: 64, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: 1 << 31 }
}
fn identity(p: &SourceTaskPlanner, task: &SourceMapTask) -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic source-map fixture");
    ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(), logical_model_digest: d,
        artifact_format: "fixture".to_owned(), quant_recipe: "int8-fixture".to_owned(), packing_set_digest: d,
        tokenizer_digest: p.tokenizer_digest(), template_digest: *p.template_digest(),
        task_spec: task.request(String::new(), budget()).task().spec().identity(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None }
}
fn unlimited() -> Int8Work {
    let mut w = Int8Work::default();
    w.forward_positions = u64::MAX; w.projected_logits = u64::MAX; w.attention_pairs = u64::MAX;
    w.projections.dot_products = u64::MAX; w.projections.multiply_accumulates = u64::MAX; w
}
fn limits() -> Int8SourceMapLimits {
    Int8SourceMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 8, max_chunk_tokens: 8,
            context_tokens: 8192, reserved_tokens: 1024, max_chunks: 256, max_tokenizer_calls: 1024 },
        reduction: ExecutionLimits { map_batch_chunks: 2, reduce_fan_in: 2,
            max_value_bytes: 4 << 20, ..ExecutionLimits::default() },
        max_model_work: unlimited(), mask_limits: MaskWorkLimits::default(),
        mask_visits_per_chunk: 1000, max_mask_visits: 256_000,
    }
}
fn prepare<'s>(p: &SourceTaskPlanner, source: &'s str, l: Int8SourceMapLimits) -> PreparedInt8SourceMap<'s> {
    let task = SourceMapTask::Ner(NerOptions::default()); let id = identity(p, &task);
    p.plan_int8_map_with_control(source, &task, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), l, &mut Continue).unwrap()
}
fn admitted(p: &PreparedInt8SourceMap<'_>) -> Vec<ExecutionIdentity> { p.execution_identities().cloned().collect() }
fn driver() -> (FakeDriver, Rc<Cell<usize>>) {
    let calls = Rc::new(Cell::new(0));
    (FakeDriver { calls: Rc::clone(&calls), fail_at: None, corrupt: false, token: 1 }, calls)
}
struct FakeDriver { calls: Rc<Cell<usize>>, fail_at: Option<usize>, corrupt: bool, token: u32 }
impl SourceDriver for FakeDriver {
    fn checkpoint(&mut self) -> Result<(), Int8SourceError> { Ok(()) }
    fn run(&mut self, plan: &PreparedInt8SourceTask, _: &ExecutionIdentity) -> Result<Int8SourceTaskRun, Int8SourceError> {
        let call = self.calls.get() + 1; self.calls.set(call);
        if self.fail_at == Some(call) { return Err(Int8SourceError::Cancelled(DecodeCancellationKind::Deadline)); }
        let work = constrained_int8::planned_work(plan.prompt_tokens(), 2).unwrap();
        let result = SourceTaskResult::Ner(NerResult { schema_version: 1, task_spec_version: NER_TASK_VERSION.to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), score_space: ScoreSpace::NotComputed,
            grounding: ExtractionGrounding::SourceMembership, entities: vec![], generated_token_ids: vec![self.token, 0],
            forward_positions: work.forward_positions, projected_logits: work.projected_logits, mask_node_visit_charge: 20 });
        let mut model_work = work;
        if self.corrupt { model_work.projections.multiply_accumulates -= 1; }
        Ok(Int8SourceTaskRun { schema_version: 1, execution: INT8_SOURCE_EXECUTION.to_owned(), result, model_work })
    }
}

#[test]
fn real_planning_binds_all_three_task_families_and_the_complete_work() {
    let p = planner();
    for task in [SourceMapTask::Ner(NerOptions::default()), SourceMapTask::Keyphrases(KeyphraseOptions::default()),
        SourceMapTask::Summarize(SummaryOptions::default())] {
        let id = identity(&p, &task);
        let prepared = p.plan_int8_map_with_control("Alice Bob Carol Dave", &task, budget(),
            &PlanContext::new(&id, budget()).unwrap(), SourcePlanningLimits::default(), limits(), &mut Continue).unwrap();
        assert!(prepared.chunk_count() > 1);
        let mut total = Int8Work::default();
        for plan in &prepared.plans {
            assert_eq!(plan.execution_identity().task_spec, id.task_spec);
            assert_eq!(plan.planned_work(), constrained_int8::planned_work(plan.prompt_tokens(), 64).unwrap());
            total = add_work(total, plan.planned_work()).unwrap();
        }
        assert_eq!(prepared.planned_work(), total);
        assert_eq!(prepared.reserved_mask_visits(), 1000 * prepared.chunk_count() as u64);
        assert_eq!(prepared.chunks.chunks().iter().map(|c| c.text()).collect::<String>(), "Alice Bob Carol Dave");
    }
}
#[test]
fn full_commitment_is_checked_before_any_native_call() {
    let p = planner(); let prepared = prepare(&p, "Alice Bob Carol Dave", limits());
    let mut identities = admitted(&prepared);
    identities.last_mut().unwrap().prompt_digest = Sha256Digest::of_bytes(b"tampered last chunk");
    let (driver, calls) = driver();
    assert!(prepared.execute_with_driver(&identities, driver).is_err()); assert_eq!(calls.get(), 0);
}
#[test]
fn missing_last_identity_is_not_a_partial_admission() {
    let p = planner(); let prepared = prepare(&p, "Alice Bob Carol Dave", limits());
    let mut identities = admitted(&prepared); identities.pop();
    let (driver, calls) = driver();
    assert!(matches!(prepared.execute_with_driver(&identities, driver), Err(Int8SourceMapError::Admission)));
    assert_eq!(calls.get(), 0);
}
#[test]
fn aggregate_model_and_mask_ceilings_refuse_before_execution() {
    let p = planner(); let task = SourceMapTask::Ner(NerOptions::default()); let id = identity(&p, &task);
    for mask in [false, true] {
        let mut l = limits();
        if mask { l.max_mask_visits = l.mask_visits_per_chunk; } else { l.max_model_work.forward_positions = 1; }
        assert!(matches!(p.plan_int8_map_with_control("Alice Bob Carol Dave", &task, budget(),
            &PlanContext::new(&id, budget()).unwrap(), SourcePlanningLimits::default(), l, &mut Continue),
            Err(Int8SourceMapError::WorkLimit)));
    }
}
#[test]
fn native_mapping_and_multilevel_merge_preserve_every_chunk_in_source_order() {
    let p = planner(); let prepared = prepare(&p, "Alice Bob Carol Dave Eve Frank Grace", limits());
    let count = prepared.chunk_count(); let identities = admitted(&prepared); let (driver, calls) = driver();
    let result = prepared.execute_with_driver(&identities, driver).unwrap();
    assert_eq!(calls.get(), count);
    assert_eq!(result.mapped.root().value().chunks().map(|c| c.chunk_id).collect::<Vec<_>>(), (0..count).collect::<Vec<_>>());
    assert!(result.mapped.reduction_levels() > 1);
    assert_eq!(result.mask_node_visit_charge, 20 * count as u64);
    assert_eq!(result.reserved_mask_node_visits, 1000 * count as u64);
    assert!(within(result.model_work, result.planned_model_work));
    assert!(result.mapped.warnings().contains(&mapreduce::ReductionWarning::SingleContextEquivalenceNotEstablished));
}
#[test]
fn cancellation_of_a_later_chunk_discards_the_partial_document() {
    let p = planner(); let prepared = prepare(&p, "Alice Bob Carol Dave", limits()); let ids = admitted(&prepared);
    let (mut driver, calls) = driver(); driver.fail_at = Some(2);
    let error = prepared.execute_with_driver(&ids, driver).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline)); assert_eq!(calls.get(), 2);
}
#[test]
fn completion_binds_exact_source_options_model_and_unused_allowances() {
    let p = planner(); let prepared = prepare(&p, "Alice", limits());
    let expected = prepared.preflight_metadata(); let ids = admitted(&prepared);
    let result = prepared.execute_with_driver(&ids, driver().0).unwrap();
    expected.verify_completed(&result).unwrap();
    for axis in 0..8 {
        let mut source = "Alice"; let mut options = NerOptions::default();
        let mut mapping = limits(); let mut planning = SourcePlanningLimits::default();
        match axis { 0 => source = "Bobby", 1 => options.max_entities -= 1,
            3 => mapping.max_model_work.forward_positions -= 1,
            4 => mapping.max_mask_visits -= 1, 5 => mapping.chunks.max_chunks -= 1,
            6 => mapping.reduction.reduce_fan_in += 1, 7 => planning.max_input_bytes -= 1,
            _ => {} }
        let task = SourceMapTask::Ner(options); let mut id = identity(&p, &task);
        if axis == 2 { id.logical_model_digest = Sha256Digest::of_bytes(b"another model"); }
        let foreign = p.preflight_int8_map_with_control(source, &task, budget(),
            &PlanContext::new(&id, budget()).unwrap(), planning, mapping, &mut Continue).unwrap();
        assert_eq!(foreign.source_span(), expected.source_span());
        if axis != 1 { assert_eq!(foreign.planned_work(), expected.planned_work()); }
        assert_ne!(foreign, expected);
        assert!(foreign.verify_completed(&result).is_err());
    }
    let foreign = prepare(&p, "Bobby", limits()); let ids = admitted(&foreign);
    let foreign = foreign.execute_with_driver(&ids, driver().0).unwrap();
    assert_eq!(crate::canonjson::canonical_bytes(&result).unwrap(), crate::canonjson::canonical_bytes(&foreign).unwrap());
    assert!(expected.verify_completed(&foreign).is_err(), "same visible output cannot substitute a foreign source");
    let json = serde_json::to_string(&result).unwrap(); let debug = format!("{expected:?}");
    for private in ["request_binding", "result_binding", "Alice"] {
        assert!(!json.contains(private)); assert!(!debug.contains(private));
    }
}
#[test]
fn completion_detects_payload_substitution_even_when_all_work_and_spans_match() {
    let p = planner(); let prepared = prepare(&p, "Alice Bob Carol Dave", limits());
    let expected = prepared.preflight_metadata(); let ids = admitted(&prepared);
    let mut result = prepared.execute_with_driver(&ids, driver().0).unwrap();
    let prepared = prepare(&p, "Alice Bob Carol Dave", limits()); let ids = admitted(&prepared);
    let (mut alternate, _) = driver(); alternate.token = 2;
    let alternate = prepared.execute_with_driver(&ids, alternate).unwrap();
    expected.verify_completed(&alternate).unwrap();
    assert_eq!(result.model_work, alternate.model_work);
    assert_eq!(result.mask_node_visit_charge, alternate.mask_node_visit_charge);
    result.mapped = alternate.mapped;
    assert!(expected.verify_completed(&result).is_err());
}
#[test]
fn completion_check_preserves_all_cancellation_causes_through_final_serialization() {
    struct Stop { calls: usize, stop_at: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.calls == self.stop_at).then_some(self.cause)
        }
    }
    use DecodeCancellationKind::*;
    let p = planner(); let prepared = prepare(&p, "Alice Bob Carol Dave", limits());
    let expected = prepared.preflight_metadata(); let ids = admitted(&prepared);
    let result = prepared.execute_with_driver(&ids, driver().0).unwrap();
    let mut count = Stop { calls: 0, stop_at: usize::MAX, cause: User };
    expected.verify_completed_with_control(&result, &mut count).unwrap();
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for stop_at in [1, 2, count.calls] {
            let error = expected.verify_completed_with_control(&result,
                &mut Stop { calls: 0, stop_at, cause }).unwrap_err();
            assert_eq!(error.cancellation(), Some(cause));
        }
    }
}
#[test]
fn corrupt_native_mac_accounting_stops_before_the_next_chunk() {
    let p = planner(); let prepared = prepare(&p, "Alice Bob Carol Dave", limits()); let ids = admitted(&prepared);
    let (mut driver, calls) = driver(); driver.corrupt = true;
    assert!(prepared.execute_with_driver(&ids, driver).is_err()); assert_eq!(calls.get(), 1);
}
#[test]
fn reduction_tree_cost_is_admitted_before_mapping() {
    let p = planner(); let mut l = limits(); l.reduction.max_task_calls = 1;
    let prepared = prepare(&p, "Alice Bob Carol Dave Eve Frank Grace", l); let ids = admitted(&prepared);
    let (driver, calls) = driver();
    assert!(matches!(prepared.execute_with_driver(&ids, driver),
        Err(Int8SourceMapError::Reduction(ExecutionError::WorkBudget))));
    assert_eq!(calls.get(), 0);
}
#[test]
fn value_budget_failure_is_not_retried_or_returned_as_partial_success() {
    let p = planner(); let mut l = limits(); l.reduction.map_batch_chunks = 1; l.reduction.max_value_bytes = 1;
    let prepared = prepare(&p, "Alice Bob Carol Dave", l); let ids = admitted(&prepared); let (driver, calls) = driver();
    assert!(matches!(prepared.execute_with_driver(&ids, driver),
        Err(Int8SourceMapError::Reduction(ExecutionError::ValueBudget))));
    assert_eq!(calls.get(), 1);
}
#[test]
fn complete_outer_envelope_is_bounded() {
    let p = planner(); let mut l = limits(); l.reduction.max_result_bytes = 1;
    let prepared = prepare(&p, "Alice Bob Carol Dave", l); let ids = admitted(&prepared); let (driver, _) = driver();
    assert!(prepared.execute_with_driver(&ids, driver).is_err());
}
#[test]
fn cancellation_during_partitioning_retains_its_exact_cause() {
    struct Stop { calls: usize }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.calls == 3).then_some(DecodeCancellationKind::Deadline)
        }
    }
    let p = planner(); let task = SourceMapTask::Ner(NerOptions::default()); let id = identity(&p, &task);
    let error = p.plan_int8_map_with_control("Alice", &task, budget(), &PlanContext::new(&id, budget()).unwrap(),
        SourcePlanningLimits::default(), limits(), &mut Stop { calls: 0 }).err().unwrap();
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
}

fn compact_policy() -> crate::tokenizer::untrusted::UntrustedEncoding {
    use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
    UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 })
}
fn compact_work() -> crate::tokenizer::untrusted::SafeBpeWorkLimits {
    crate::tokenizer::untrusted::SafeBpeWorkLimits { max_input_bytes: 65_536, max_output_tokens: 65_536,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 }
}
fn compact_limits() -> Int8SourceMapLimits {
    let mut limit = limits(); limit.chunks.max_chunk_bytes = 4096; limit.chunks.max_chunk_tokens = 4096; limit
}
const COMPACT_SOURCE: &str = "The quick brown fox jumps over the lazy dog. 上海 <think> <|im_start|> ▁";

#[test]
fn compact_maps_preserve_exact_source_and_reduce_native_prompt_work_for_all_three_tasks() {
    for task in [SourceMapTask::Ner(NerOptions::default()), SourceMapTask::Keyphrases(KeyphraseOptions::default()),
        SourceMapTask::Summarize(SummaryOptions::default())] {
        let byte = planner(); let byte_id = identity(&byte, &task);
        let original = byte.plan_int8_map_with_control(COMPACT_SOURCE, &task, budget(),
            &PlanContext::new(&byte_id, budget()).unwrap(), SourcePlanningLimits::default(),
            compact_limits(), &mut Continue).unwrap();
        let compact = planner().with_map_document_encoding(compact_policy(), compact_work()).unwrap();
        let id = identity(&compact, &task);
        let plan = compact.plan_int8_map_with_control(COMPACT_SOURCE, &task, budget(),
            &PlanContext::new(&id, budget()).unwrap(), SourcePlanningLimits::default(),
            compact_limits(), &mut Continue).unwrap();
        assert_eq!(plan.chunk_count(), original.chunk_count());
        assert_eq!(plan.chunks.chunks().iter().map(SourceChunk::text).collect::<String>(), COMPACT_SOURCE);
        assert_eq!(plan.chunks.chunks()[0].span(), original.chunks.chunks()[0].span());
        assert!(plan.planned_work().forward_positions < original.planned_work().forward_positions);
        assert_ne!(plan.execution_identities().next().unwrap().template_digest,
            original.execution_identities().next().unwrap().template_digest);
        let used = compact.map_document_encoding_work().unwrap().unwrap();
        assert_eq!(used.input_bytes(), 2 * COMPACT_SOURCE.len(), "sizing probe and real native plan");
        assert!(used.output_tokens() < used.input_bytes());
        assert!(used.pair_checks() > 0 && used.pair_bytes() > 0);
        assert!(byte.map_document_encoding_work().unwrap().is_none());
    }
}

#[test]
fn compact_token_fit_probes_preserve_multichunk_unicode_and_spend_rejected_work() {
    let source = format!("{COMPACT_SOURCE}\r\n").repeat(3);
    let compact = planner().with_map_document_encoding(compact_policy(), compact_work()).unwrap();
    let mut mapping = compact_limits();
    mapping.chunks.max_chunk_bytes = 64;
    mapping.chunks.max_chunk_tokens = 8;
    let prepared = prepare(&compact, &source, mapping);
    assert!(prepared.chunk_count() > 1);
    assert_eq!(prepared.chunks.chunks().iter().map(SourceChunk::text).collect::<String>(), source);
    let (mut byte_end, mut scalar_end) = (0, 0);
    for chunk in prepared.chunks.chunks() {
        let span = chunk.span();
        assert_eq!((span.byte_start, span.scalar_start), (byte_end, scalar_end));
        assert_eq!(&source[span.byte_start..span.byte_end], chunk.text());
        assert!(chunk.tokens() <= 8);
        assert!(!(source.as_bytes().get(span.byte_end.wrapping_sub(1)) == Some(&b'\r')
            && source.as_bytes().get(span.byte_end) == Some(&b'\n')));
        byte_end = span.byte_end; scalar_end = span.scalar_end;
    }
    assert_eq!((byte_end, scalar_end), (source.len(), source.chars().count()));
    assert!(compact.map_document_encoding_work().unwrap().unwrap().input_bytes() > 2 * source.len(),
        "discarded token-fit probes remain charged beside accepted probes and actual plans");
}

#[test]
fn compact_probes_plans_and_repeated_preparation_never_reset_the_operation_allowance() {
    let mut exact = compact_work(); exact.max_input_bytes = 2 * COMPACT_SOURCE.len() as u64;
    let compact = planner().with_map_document_encoding(compact_policy(), exact).unwrap();
    let first = prepare(&compact, COMPACT_SOURCE, compact_limits());
    assert_eq!(first.chunk_count(), 1);
    let used = compact.map_document_encoding_work().unwrap().unwrap();
    let task = SourceMapTask::Ner(NerOptions::default()); let id = identity(&compact, &task);
    assert!(compact.plan_int8_map_with_control(COMPACT_SOURCE, &task, budget(),
        &PlanContext::new(&id, budget()).unwrap(), SourcePlanningLimits::default(),
        compact_limits(), &mut Continue).is_err());
    assert_eq!(compact.map_document_encoding_work().unwrap().unwrap(), used);

    exact.max_input_bytes -= 1;
    let short = planner().with_map_document_encoding(compact_policy(), exact).unwrap();
    let id = identity(&short, &task);
    assert!(short.plan_int8_map_with_control(COMPACT_SOURCE, &task, budget(),
        &PlanContext::new(&id, budget()).unwrap(), SourcePlanningLimits::default(),
        compact_limits(), &mut Continue).is_err());
    assert_eq!(short.map_document_encoding_work().unwrap().unwrap().input_bytes(), COMPACT_SOURCE.len());
}

#[test]
fn compact_maps_require_cumulative_admission_and_bind_all_four_operation_ceilings() {
    let task = SourceMapTask::Ner(NerOptions::default());
    let one = planner().with_document_encoding(compact_policy()).unwrap(); let id = identity(&one, &task);
    let context = PlanContext::new(&id, budget()).unwrap();
    assert!(one.plan_int8_map_with_control(COMPACT_SOURCE, &task, budget(), &context,
        SourcePlanningLimits::default(), compact_limits(), &mut Continue).is_err());
    assert!(one.int8_map_capacity_with_control(&task, budget(), &context,
        SourcePlanningLimits::default(), &mut Continue).is_err());
    let original = planner().with_map_document_encoding(compact_policy(), compact_work()).unwrap();
    let original = prepare(&original, COMPACT_SOURCE, compact_limits());
    for axis in 0..4 {
        let mut work = compact_work();
        match axis { 0 => work.max_input_bytes -= 1, 1 => work.max_output_tokens -= 1,
            2 => work.max_pair_checks -= 1, _ => work.max_pair_bytes -= 1 }
        let next = planner().with_map_document_encoding(compact_policy(), work).unwrap();
        let next = prepare(&next, COMPACT_SOURCE, compact_limits());
        assert_eq!(next.planned_work(), original.planned_work());
        assert_ne!(next.execution_identities().next().unwrap().template_digest,
            original.execution_identities().next().unwrap().template_digest);
    }
}

#[test]
fn cancellation_inside_compact_probes_retains_every_original_cause_and_spent_work() {
    struct Stop { calls: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1; (self.calls >= 7).then_some(self.cause)
        }
    }
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let compact = planner().with_map_document_encoding(compact_policy(), compact_work()).unwrap();
        let task = SourceMapTask::Ner(NerOptions::default()); let id = identity(&compact, &task);
        let error = compact.plan_int8_map_with_control(&COMPACT_SOURCE.repeat(8), &task, budget(),
            &PlanContext::new(&id, budget()).unwrap(), SourcePlanningLimits::default(),
            compact_limits(), &mut Stop { calls: 0, cause }).err().unwrap();
        assert_eq!(error.cancellation(), Some(cause));
        let used = compact.map_document_encoding_work().unwrap().unwrap();
        assert!(used.input_bytes() <= 8 * COMPACT_SOURCE.len());
    }
}
fn small_chunks(source: &str) -> ChunkPlan<'_> {
    let mut l = limits().chunks; l.max_chunk_bytes = 4; l.max_chunk_tokens = 4;
    ChunkPlan::build(source, l, |s| Ok(s.len())).unwrap()
}
fn span(a: usize, b: usize, c: usize, d: usize) -> VerifiedSourceSpan {
    VerifiedSourceSpan { byte_start: a, byte_end: b, scalar_start: c, scalar_end: d }
}
#[test]
fn unicode_and_crlf_coordinates_lift_without_normalization() {
    let plan = small_chunks("é\r\n猫x"); assert_eq!(plan.chunks().len(), 2);
    let spans = lift_spans(&plan, &plan.chunks()[1], "猫", &[span(0, 3, 0, 1)], &mut || Ok(())).unwrap();
    assert_eq!(spans, vec![span(4, 7, 3, 4)]);
    assert_eq!(plan.chunks()[0].text(), "é\r\n");
}
#[test]
fn every_repeated_occurrence_is_retained_without_selecting_one() {
    let plan = small_chunks("a a");
    let expected = vec![span(0, 1, 0, 1), span(2, 3, 2, 3)];
    assert_eq!(lift_spans(&plan, &plan.chunks()[0], "a", &expected, &mut || Ok(())).unwrap(), expected);
}
#[test]
fn cross_chunk_and_interior_utf8_spans_are_refused() {
    let plan = small_chunks("abcdEFGH");
    assert!(lift_spans(&plan, &plan.chunks()[0], "dE", &[span(3, 5, 3, 5)], &mut || Ok(())).is_err());
    let plan = small_chunks("猫x");
    assert!(lift_spans(&plan, &plan.chunks()[0], "猫", &[span(1, 3, 0, 1)], &mut || Ok(())).is_err());
}
#[test]
fn scalar_mismatch_empty_quote_and_missing_occurrence_fail_closed() {
    let plan = small_chunks("猫x"); let chunk = &plan.chunks()[0];
    assert!(lift_spans(&plan, chunk, "猫", &[span(0, 3, 0, 3)], &mut || Ok(())).is_err());
    assert!(lift_spans(&plan, chunk, "", &[span(0, 0, 0, 0)], &mut || Ok(())).is_err());
    assert!(lift_spans(&plan, chunk, "猫", &[], &mut || Ok(())).is_err());
}
#[test]
fn coordinate_projection_checks_cancellation_between_occurrences() {
    let plan = small_chunks("a a"); let mut calls = 0;
    let result = lift_spans(&plan, &plan.chunks()[0], "a", &[span(0, 1, 0, 1), span(2, 3, 2, 3)], &mut || {
        calls += 1;
        if calls == 2 { Err(Int8SourceError::Cancelled(DecodeCancellationKind::Deadline)) } else { Ok(()) }
    });
    assert_eq!(result.err().unwrap().cancellation(), Some(DecodeCancellationKind::Deadline));
}
#[test]
fn all_five_native_counters_use_checked_whole_document_arithmetic() {
    let a = unlimited(); let mut one = Int8Work::default();
    for axis in 0..5 {
        match axis { 0 => one.forward_positions = 1, 1 => one.projected_logits = 1,
            2 => one.attention_pairs = 1, 3 => one.projections.dot_products = 1,
            _ => one.projections.multiply_accumulates = 1 }
        assert!(add_work(a, one).is_none()); one = Int8Work::default();
    }
    assert_eq!(add_work(a, Int8Work::default()), Some(a));
}
#[test]
fn task_schema_excludes_qa_and_caller_executable_identity() {
    assert!(serde_json::from_str::<SourceMapTask>(r#"{"task":"answer","options":{}}"#).is_err());
    assert!(serde_json::from_str::<SourceMapTask>(r#"{"task":"ner","options":{"types":["person"],"max_entities":2,"max_mention_scalars":8},"identity":{}}"#).is_err());
}
#[test]
fn empty_source_and_unbounded_chunk_counts_are_refused() {
    let p = planner(); let task = SourceMapTask::Ner(NerOptions::default()); let id = identity(&p, &task);
    let ctx = PlanContext::new(&id, budget()).unwrap();
    assert!(matches!(p.plan_int8_map_with_control("", &task, budget(), &ctx,
        SourcePlanningLimits::default(), limits(), &mut Continue), Err(Int8SourceMapError::EmptySource)));
    let mut l = limits(); l.chunks.max_chunks = 257;
    assert!(matches!(p.plan_int8_map_with_control("Alice", &task, budget(), &ctx,
        SourcePlanningLimits::default(), l, &mut Continue), Err(Int8SourceMapError::InvalidLimits)));
}

#[test]
fn long_source_coordinate_scans_preserve_all_native_cancellation_causes() {
    let source = format!("{}Z", "é😀".repeat(4096));
    let chunks = ChunkPlan::build(&source, ChunkLimits { max_input_bytes: source.len(),
        max_chunk_bytes: source.len(), max_chunk_tokens: 1, ..ChunkLimits::default() }, |_| Ok(1)).unwrap();
    let chunk = &chunks.chunks()[0];
    let local = VerifiedSourceSpan { byte_start: source.len() - 1, byte_end: source.len(),
        scalar_start: 8192, scalar_end: 8193 };
    let valid = lift_spans(&chunks, chunk, "Z", &[local], &mut || Ok(())).unwrap();
    assert_eq!(valid, vec![local]);
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let mut polls = 0;
        let error = lift_spans(&chunks, chunk, "Z", &[local], &mut || {
            polls += 1;
            if polls == 4 { Err(Int8SourceError::Cancelled(cause)) } else { Ok(()) }
        }).unwrap_err();
        assert_eq!(error.cancellation(), Some(cause));
        assert_eq!(polls, 4);
    }
}

#[test]
fn controlled_span_lifting_still_rejects_wrong_scalar_coordinates() {
    let chunks = ChunkPlan::build("éA😀Z", ChunkLimits { max_chunk_bytes: 16,
        max_chunk_tokens: 16, ..ChunkLimits::default() }, |_| Ok(4)).unwrap();
    let chunk = &chunks.chunks()[0];
    let correct = VerifiedSourceSpan { byte_start: 7, byte_end: 8, scalar_start: 3, scalar_end: 4 };
    assert_eq!(lift_spans(&chunks, chunk, "Z", &[correct], &mut || Ok(())).unwrap(), vec![correct]);
    for span in [VerifiedSourceSpan { scalar_start: 2, ..correct },
        VerifiedSourceSpan { scalar_end: 5, ..correct },
        VerifiedSourceSpan { byte_start: 4, ..correct }] {
        assert!(matches!(lift_spans(&chunks, chunk, "Z", &[span], &mut || Ok(())),
            Err(Int8SourceError::InvalidResult)));
    }
}

#[test]
fn packed_map_schedules_leave_the_complete_request_and_work_commitment_unchanged() {
    let planner = planner();
    let source = "Alice Bob Carol Dave Eve Frank Grace é上海";
    let plan = prepare(&planner, source, limits());
    let binding = plan.preflight_metadata();
    let identities = admitted(&plan);
    let work = plan.planned_work();
    let masks = plan.reserved_mask_visits();
    let prompts: Vec<_> = plan.plans.iter().map(PreparedInt8SourceTask::prompt_tokens).collect();
    // Real pinned plans contain full morsels and a tail in this matrix. No
    // fabricated logits or mocked native result supplies a numerical claim.
    assert!(prompts.iter().all(|&tokens| tokens > 7));
    assert!(prompts.iter().any(|&tokens| tokens % 7 != 0));
    for rows in [1, 3, 7, 64] {
        let prefill = Int8PrefillLimits { max_batch_rows: rows,
            max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() };
        check_prefill(prefill).unwrap();
        assert_eq!(plan.preflight_metadata(), binding);
        assert_eq!(admitted(&plan), identities);
        assert_eq!(plan.planned_work(), work);
        assert_eq!(plan.reserved_mask_visits(), masks);
    }
    for rows in [0, 65, usize::MAX] {
        assert!(matches!(check_prefill(Int8PrefillLimits {
            max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX,
        }), Err(Int8SourceMapError::InvalidLimits)));
    }
    let bytes = Int8PrefillLimits::required_extra_scratch_bytes(7).unwrap();
    assert!(matches!(check_prefill(Int8PrefillLimits {
        max_batch_rows: 7, max_extra_scratch_bytes: bytes - 1,
    }), Err(Int8SourceMapError::InvalidLimits)));
    assert_eq!(plan.preflight_metadata(), binding);
}

#[test]
fn entity_reduction_authenticates_the_map_and_preserves_its_work_and_cancellation() {
    use entity_reduction::Int8EntityLimits;
    let p = planner();
    let prepared = prepare(&p, "Alice Bob Carol Dave", limits());
    let expected = prepared.preflight_metadata();
    let bound = Int8EntityLimits { max_result_bytes: expected.max_result_bytes as usize,
        ..Int8EntityLimits::default() };
    prepared.check_entities(bound).unwrap();
    let ids = admitted(&prepared);
    let mut run = prepared.execute_with_driver(&ids, driver().0).unwrap();
    let reduced = expected.reduce_entities_with_control(&run, bound, &mut Continue).unwrap();
    assert_eq!(reduced.mapped_chunks, expected.chunk_count());
    assert_eq!(reduced.source_span, expected.source_span());
    assert_eq!(reduced.model_work, run.model_work);
    assert!(reduced.entities.is_empty());
    let tiny = Int8EntityLimits { max_result_bytes: 1, ..bound };
    assert!(expected.reduce_entities_with_control(&run, tiny, &mut Continue).is_err());
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::User) }
    }
    assert_eq!(expected.reduce_entities_with_control(&run, bound, &mut Stop).err().unwrap().cancellation(),
        Some(DecodeCancellationKind::User));
    run.semantics = "substituted";
    assert!(expected.reduce_entities_with_control(&run, bound, &mut Continue).is_err());
}
