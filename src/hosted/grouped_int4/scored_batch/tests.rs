//! Admission fixtures only; synthetic identities never authorize model execution.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    batch::{BatchLimits, scored_int4::Int4ScoredPlanFacts},
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT, lmhead::NANBEIGE_VOCAB_SIZE,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE, strict_int4::STRICT_INT4_EXECUTION},
    tasks::{ir::TaskBudget, classify::grouped_int4::Int4ClassificationPlanner,
        sentiment::grouped_int4::Int4SentimentPlanner, judge::grouped_int4::Int4JudgePlanner},
};

fn limits() -> CorpusLimits {
    CorpusLimits { native: NativeLimits { context_tokens: 32, allocator_reserve_bytes: 1 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000,
            cleanup_reserve_bytes: 65_536 } }, transport: BatchLimits::default(),
        preparation_reserve_bytes: 128 << 20, io_reserve_bytes: 65_536 }
}
fn work(width: usize) -> Int4Work {
    let head = Int4Work::for_sequence(width, 0, 24, 3 * NANBEIGE_VOCAB_SIZE).unwrap();
    head.checked_add(head).unwrap().checked_add(head).unwrap().checked_add(head).unwrap()
}
fn work_limits(width: usize) -> Int4ScoredBatchLimits {
    let item = work(width);
    Int4ScoredBatchLimits { max_item_work: item, max_model_work: item.checked_add(item).unwrap() }
}
fn identity(task: &str, width: usize) -> ExecutionIdentity {
    let digest = Sha256Digest::of_bytes(b"scored-host-admission-fixture-only");
    let packing = Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
        INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", digest, width,
    )).unwrap());
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: digest, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={width}"), packing_set_digest: packing,
        tokenizer_digest: digest, template_digest: digest, task_spec: task.to_owned(),
        taskir_digest: digest, prompt_digest: digest, grammar_compiler_version: "none".to_owned(),
        schema_digest: digest, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "complete-candidate-eos".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest, decision_policy_digest: digest,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn request(id: &ExecutionIdentity, width: usize) -> Int4ScoredAdmission<'_> {
    let kv = Int4MemoryRequirement::for_context(32, width).unwrap().kv_bytes;
    Int4ScoredAdmission { facts: Int4ScoredPlanFacts { identity: id, model_work: work(width),
        group_width: width, head_count: 4, required_context: 24,
        task_budget: TaskBudget { max_input_tokens: 32, max_output_tokens: 16,
            max_output_bytes: 8192, max_grammar_states: 256, max_kv_bytes: kv },
        max_result_bytes: 8192 }, kv_reservation_bytes: kv }
}
fn check(task: &str, request: &Int4ScoredAdmission<'_>) -> Result<(), BatchItemFailure> {
    check_request(32, 32, Int4MemoryRequirement::for_context(32, 32).unwrap().kv_bytes,
        work(32), task, 8192, request)
}

#[test]
fn each_group_partition_prices_one_full_native_context_and_owned_stream() {
    for width in [1, 3, 16, 32, 127, 128, MAX_MODEL_K] {
        let limits = limits();
        let required = requirements(width, work_limits(width), limits).unwrap();
        let expected = Int4MemoryRequirement::for_context(32, width).unwrap();
        assert_eq!(required.native, expected);
        assert_eq!(required.workspace_bytes,
            expected.rope_bytes + expected.scratch_payload_bound + limits.native.allocator_reserve_bytes);
        assert_eq!(required.input_bytes, limits.reservation_bytes().unwrap());
    }
}

#[test]
fn complete_multihead_work_can_exceed_reused_context_capacity() {
    for task in ["classify-v1", "sentiment-v1", "judge-v1"] {
        let id = identity(task, 32); let request = request(&id, 32);
        assert!(request.facts.model_work.forward_positions > limits().native.context_tokens as u64);
        check(task, &request).unwrap();
    }
}

#[test]
fn conditional_sentiment_rows_are_not_forced_into_a_full_vocab_denominator() {
    let id = identity("sentiment-v1", 32); let mut request = request(&id, 32);
    let head = Int4Work::for_sequence(32, 0, 24, 5).unwrap();
    request.facts.model_work = head.checked_add(head).unwrap();
    request.facts.head_count = 2;
    assert_ne!(request.facts.model_work.projected_logits % NANBEIGE_VOCAB_SIZE as u64, 0);
    check("sentiment-v1", &request).unwrap();
}

#[test]
fn every_per_item_work_axis_is_checked_independently() {
    let id = identity("classify-v1", 32); let request = request(&id, 32);
    for axis in 0..5 {
        let mut cap = work(32);
        match axis { 0 => cap.forward_positions -= 1, 1 => cap.projected_logits -= 1,
            2 => cap.attention_pairs -= 1, 3 => cap.projections.dot_products -= 1,
            _ => cap.projections.multiply_accumulates -= 1 }
        let error = check_request(32, 32, request.kv_reservation_bytes, cap, "classify-v1", 8192, &request).unwrap_err();
        assert!(error.stop); assert_eq!(error.fault.code, BatchCode::InvalidExecution);
    }
}

#[test]
fn invalid_whole_run_limits_never_reach_model_or_stream_io() {
    for axis in 0..6 {
        let mut work = work_limits(32);
        match axis { 0 => work.max_item_work.forward_positions = 0, 1 => work.max_item_work.projected_logits = 0,
            2 => work.max_item_work.attention_pairs = 0, 3 => work.max_item_work.projections.dot_products = 0,
            4 => work.max_item_work.projections.multiply_accumulates = 0,
            _ => work.max_model_work.forward_positions = work.max_item_work.forward_positions - 1 }
        assert!(requirements(32, work, limits()).is_err(), "axis {axis}");
    }
    for axis in 0..8 {
        let mut limits = limits();
        match axis { 0 => limits.native.context_tokens = 0, 1 => limits.native.context_tokens = 8193,
            2 => limits.native.allocator_reserve_bytes = 0, 3 => limits.native.allocator_reserve_bytes = u64::MAX,
            4 => limits.preparation_reserve_bytes = 0, 5 => limits.io_reserve_bytes = 0,
            6 => limits.preparation_reserve_bytes = u64::MAX, _ => limits.transport.max_json_depth = 65 }
        assert!(requirements(32, work_limits(32), limits).is_err(), "axis {axis}");
    }
}

#[test]
fn model_profile_group_and_fixed_task_are_never_repaired() {
    for axis in 0..8 {
        let mut id = identity("judge-v1", 32);
        match axis { 0 => id.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            1 => id.backend_semantic_version = "other".to_owned(), 2 => id.kv_dtype = "int8".to_owned(),
            3 => id.artifact_format = "fnlpq-v1".to_owned(), 4 => id.thinking_mode = ThinkingMode::Enabled,
            5 => id.tool_mode = ToolMode::Json, 6 => id.task_spec = "classify-v1".to_owned(),
            _ => id.packing_set_digest = Sha256Digest::of_bytes(b"other-group") }
        let failure = check("judge-v1", &request(&id, 32)).unwrap_err();
        assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::Admission);
    }
    let id = identity("judge-v1", 128);
    assert!(check("judge-v1", &request(&id, 128)).unwrap_err().stop);
}

#[test]
fn actual_full_kv_must_match_the_host_and_fit_each_task_budget() {
    let id = identity("sentiment-v1", 32);
    for axis in 0..5 {
        let mut request = request(&id, 32);
        match axis { 0 => request.kv_reservation_bytes -= 1, 1 => request.kv_reservation_bytes += 1,
            2 => request.facts.task_budget.max_kv_bytes -= 1, 3 => request.facts.required_context = 0,
            _ => request.facts.required_context = 33 }
        let failure = check("sentiment-v1", &request).unwrap_err();
        assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::Admission);
    }
}

#[test]
fn empty_head_bundles_and_unpriced_output_are_refused() {
    let id = identity("classify-v1", 32);
    let mut item = request(&id, 32); item.facts.head_count = 0;
    assert_eq!(check("classify-v1", &item).unwrap_err().fault.code, BatchCode::InvalidExecution);
    for cap in [0, 8193, u64::MAX] {
        let mut item = request(&id, 32); item.facts.max_result_bytes = cap;
        let failure = check("classify-v1", &item).unwrap_err();
        assert!(!failure.stop); assert_eq!(failure.fault.code, BatchCode::OutputLineLimit);
    }
}

#[test]
fn io_and_preparation_owners_are_both_in_the_whole_stream_reservation() {
    let base = requirements(32, work_limits(32), limits()).unwrap();
    let mut bigger = limits(); bigger.io_reserve_bytes += 4096; bigger.preparation_reserve_bytes += 8192;
    let enlarged = requirements(32, work_limits(32), bigger).unwrap();
    assert_eq!(enlarged.input_bytes - base.input_bytes, 4096 + 8192);
    assert_eq!(enlarged.native, base.native); assert_eq!(enlarged.workspace_bytes, base.workspace_bytes);
}

#[test]
fn all_three_closed_planners_and_owned_io_fit_the_existing_host_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Int4ScoredBatchPlanner<Int4ClassificationPlanner>>();
    send::<Int4ScoredBatchPlanner<Int4SentimentPlanner>>();
    send::<Int4ScoredBatchPlanner<Int4JudgePlanner>>();
    let _classify = NlpEngine::batch_int4_scored::<Int4ClassificationPlanner, std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _sentiment = NlpEngine::batch_int4_scored::<Int4SentimentPlanner, std::io::Cursor<Vec<u8>>, Vec<u8>>;
    let _judge = NlpEngine::batch_int4_scored::<Int4JudgePlanner, std::io::Cursor<Vec<u8>>, Vec<u8>>;
}
