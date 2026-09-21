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
    (FakeDriver { calls: Rc::clone(&calls), fail_at: None, corrupt: false }, calls)
}
struct FakeDriver { calls: Rc<Cell<usize>>, fail_at: Option<usize>, corrupt: bool }
impl SourceDriver for FakeDriver {
    fn checkpoint(&mut self) -> Result<(), Int8SourceError> { Ok(()) }
    fn run(&mut self, plan: &PreparedInt8SourceTask, _: &ExecutionIdentity) -> Result<Int8SourceTaskRun, Int8SourceError> {
        let call = self.calls.get() + 1; self.calls.set(call);
        if self.fail_at == Some(call) { return Err(Int8SourceError::Cancelled(DecodeCancellationKind::Deadline)); }
        let work = constrained_int8::planned_work(plan.prompt_tokens(), 2).unwrap();
        let result = SourceTaskResult::Ner(NerResult { schema_version: 1, task_spec_version: NER_TASK_VERSION.to_owned(),
            numerics_profile: STRICT_INT8_PROFILE.to_owned(), score_space: ScoreSpace::NotComputed,
            grounding: ExtractionGrounding::SourceMembership, entities: vec![], generated_token_ids: vec![1, 0],
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
