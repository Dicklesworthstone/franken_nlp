//! Real pinned tokenization/planning and capacity arithmetic, never neural runs.
use super::*;
use crate::{execution_identity::ExecutionIdentity,
    tokenizer::pinned_controls, native_engine::strict_int8::STRICT_INT8_EXECUTION};

struct Control { calls: usize, stop_at: usize }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.calls += 1;
        (self.calls >= self.stop_at).then_some(DecodeCancellationKind::Deadline)
    }
}
fn go() -> Control { Control { calls: 0, stop_at: usize::MAX } }
fn fixture(task: BuiltInTask) -> (SourceTaskPlanner, ExecutionIdentity, TaskBudget) {
    let controls = pinned_controls::pinned().unwrap();
    let planner = SourceTaskPlanner::pinned(controls.template_controls(), 166_101).unwrap();
    let d = Sha256Digest::of_bytes(b"capacity-only-not-a-model-receipt");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: d, artifact_format: "fixture".to_owned(), quant_recipe: "fixture".to_owned(),
        packing_set_digest: d, tokenizer_digest: planner.tokenizer_digest(), template_digest: *planner.template_digest(),
        task_spec: task.spec().identity(), taskir_digest: d, prompt_digest: d,
        grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "fixture".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    let budget = TaskBudget { max_input_tokens: 8192, max_output_tokens: 512,
        max_output_bytes: 1 << 20, max_grammar_states: 4096, max_kv_bytes: 1 << 31 };
    (planner, identity, budget)
}
fn request(task: &SourceMapTask, document: String, budget: TaskBudget) -> SourceTaskRequest {
    match task { SourceMapTask::Ner(options) => SourceTaskRequest::Ner { document, options: options.clone(), budget },
        SourceMapTask::Keyphrases(options) => SourceTaskRequest::Keyphrases { document, options: *options, budget },
        SourceMapTask::Summarize(options) => SourceTaskRequest::Summarize { document, options: *options, budget } }
}
#[test]
fn prompt_and_context_ceilings_are_independent_and_output_is_reserved_in_full() {
    let p = price(100, 500, 200, 1000).unwrap();
    assert_eq!(p.max_source_tokens(), 400);
    assert_eq!(p.reserved_tokens(), 300);
    assert_eq!(price(100, 900, 200, 500).unwrap().max_source_tokens(), 200);
    assert_eq!(price(100, 101, 200, 301).unwrap().max_source_tokens(), 1);
}
#[test]
fn zero_room_and_overflow_never_wrap_into_capacity() {
    for (s, p, o, c) in [(100, 100, 200, 1000), (100, 500, 200, 300),
        (101, 100, 1, 1000), (usize::MAX, usize::MAX, 1, usize::MAX),
        (0, 100, 1, 1000), (100, 500, 0, 1000)] {
        assert!(price(s, p, o, c).is_err());
    }
}
#[test]
fn chunk_constraints_never_widen_any_caller_ceiling() {
    let cap = price(100, 500, 200, 1000).unwrap();
    let original = ChunkLimits { context_tokens: 600, reserved_tokens: 350,
        max_chunk_tokens: 250, ..ChunkLimits::default() };
    let bounded = cap.constrain_chunks(original).unwrap();
    assert_eq!(bounded, original);
    let wide = ChunkLimits { context_tokens: 8192, reserved_tokens: 1, max_chunk_tokens: 4096, ..original };
    let bounded = cap.constrain_chunks(wide).unwrap();
    assert_eq!(bounded.context_tokens, 1000);
    assert_eq!(bounded.reserved_tokens, 300);
    assert_eq!(bounded.max_chunk_tokens, 400);
    assert_eq!(bounded.max_chunk_bytes, wide.max_chunk_bytes);
    assert_eq!(bounded.max_chunks, wide.max_chunks);
    assert_eq!(bounded.max_tokenizer_calls, wide.max_tokenizer_calls);
    assert!(cap.constrain_chunks(ChunkLimits { context_tokens: 299, ..wide }).is_err());
}
#[test]
fn exact_scaffold_matches_real_compilation_for_all_three_tasks() {
    for (kind, task) in [(BuiltInTask::Ner, SourceMapTask::Ner(NerOptions::default())),
        (BuiltInTask::Keyphrases, SourceMapTask::Keyphrases(KeyphraseOptions::default())),
        (BuiltInTask::Summarize, SourceMapTask::Summarize(SummaryOptions::default()))] {
        let (p, id, b) = fixture(kind); let ctx = PlanContext::new(&id, b).unwrap();
        let limits = SourcePlanningLimits::default();
        let cap = p.int8_map_capacity_with_control(&task, b, &ctx, limits, &mut go()).unwrap();
        let text = "é <tool_call> 上海\r\n😀";
        let plan = p.plan_int8_with_control(&request(&task, text.to_owned(), b), &ctx, limits, &mut go()).unwrap();
        let encoded = p.source_encoder().encode(text, limits.max_input_bytes, b.max_input_tokens as usize).unwrap();
        assert_eq!(plan.prompt_tokens(), cap.scaffold_tokens() + encoded.total_token_count());
        assert_eq!(cap.reserved_tokens(), cap.scaffold_tokens() + b.max_output_tokens as usize);
    }
}
#[test]
fn exact_boundary_fits_and_one_extra_source_token_refuses() {
    let (p, id, b) = fixture(BuiltInTask::Ner); let ctx = PlanContext::new(&id, b).unwrap();
    let task = SourceMapTask::Ner(NerOptions::default()); let limits = SourcePlanningLimits::default();
    let cap = p.int8_map_capacity_with_control(&task, b, &ctx, limits, &mut go()).unwrap();
    let text = "a".repeat(cap.max_source_tokens());
    let plan = p.plan_int8_with_control(&request(&task, text.clone(), b), &ctx, limits, &mut go()).unwrap();
    assert_eq!(plan.prompt_tokens() + b.max_output_tokens as usize, limits.max_context_tokens);
    assert!(p.plan_int8_with_control(&request(&task, text + "a", b), &ctx, limits, &mut go()).is_err());
}
#[test]
fn changed_schema_options_are_priced_by_the_same_compiler() {
    let (p, id, b) = fixture(BuiltInTask::Ner); let ctx = PlanContext::new(&id, b).unwrap();
    let limits = SourcePlanningLimits::default();
    for count in [1, 8, 64] {
        let task = SourceMapTask::Ner(NerOptions { max_entities: count, ..NerOptions::default() });
        let cap = p.int8_map_capacity_with_control(&task, b, &ctx, limits, &mut go()).unwrap();
        let plan = p.plan_int8_with_control(&request(&task, "Alice".to_owned(), b), &ctx, limits, &mut go()).unwrap();
        assert_eq!(plan.prompt_tokens(), cap.scaffold_tokens() + 5);
    }
}
#[test]
fn wrong_task_profile_assets_and_oversized_schema_refuse() {
    let (p, id, b) = fixture(BuiltInTask::Ner); let task = SourceMapTask::Ner(NerOptions::default());
    for axis in 0..5 {
        let mut bad = id.clone();
        match axis { 0 => bad.task_spec = "summarize-v1".to_owned(),
            1 => bad.numerics_profile = NumericsProfile::HfBf16Eager,
            2 => bad.backend_semantic_version = "other".to_owned(),
            3 => bad.template_digest = Sha256Digest::of_bytes(b"other"),
            _ => bad.tokenizer_digest = Sha256Digest::of_bytes(b"other") }
        assert!(p.int8_map_capacity_with_control(&task, b, &PlanContext::new(&bad, b).unwrap(),
            SourcePlanningLimits::default(), &mut go()).is_err());
    }
    let mut limits = SourcePlanningLimits::default(); limits.compiler.max_schema_bytes = 1;
    assert!(p.int8_map_capacity_with_control(&task, b, &PlanContext::new(&id, b).unwrap(), limits, &mut go()).is_err());
}
#[test]
fn cancellation_before_or_after_sizing_returns_no_capacity() {
    let (p, id, b) = fixture(BuiltInTask::Ner); let task = SourceMapTask::Ner(NerOptions::default());
    let ctx = PlanContext::new(&id, b).unwrap(); let limits = SourcePlanningLimits::default();
    let mut count = go(); p.int8_map_capacity_with_control(&task, b, &ctx, limits, &mut count).unwrap();
    for stop_at in 1..=count.calls {
        let error = p.int8_map_capacity_with_control(&task, b, &ctx, limits,
            &mut Control { calls: 0, stop_at }).unwrap_err();
        assert_eq!(error.cancellation(), Some(DecodeCancellationKind::Deadline));
    }
}
