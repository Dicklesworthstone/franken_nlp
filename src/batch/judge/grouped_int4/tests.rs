//! Pure corpus authority and accounting regressions. They do not substitute
//! fabricated judgments for the separately qualified native map executor.
use super::*;
use crate::{canonjson, native_engine::{decode::DecodeCancellationKind,
    lmhead::NANBEIGE_VOCAB_SIZE, portable_int8::ProjectionWork},
    tasks::{mapreduce::ChunkLimits, judge::FaithfulnessAggregationPolicy}};

fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 768, max_output_tokens: 16, max_output_bytes: 65_536,
        max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(1024, 32).unwrap().kv_bytes * 32 }
}
fn work(units: u64) -> Int4Work {
    Int4Work { forward_positions: 40 * units, projected_logits: NANBEIGE_VOCAB_SIZE as u64 * 8 * units,
        attention_pairs: 60 * units,
        projections: ProjectionWork { dot_products: 70 * units, multiply_accumulates: 80 * units } }
}
fn mapping() -> Int4FaithfulnessMapLimits {
    Int4FaithfulnessMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 8,
        max_chunk_tokens: 768, context_tokens: 1024, reserved_tokens: 16,
        max_chunks: 64, max_tokenizer_calls: 4096 }, max_model_work: work(4),
        max_retained_result_bytes: 1024 * 1024, max_result_bytes: 1024 * 1024,
        aggregation: FaithfulnessAggregationPolicy::default() }
}
fn policy() -> FaithfulnessPolicy {
    FaithfulnessPolicy { minimum_candidate_weight_ppm: 700_000, minimum_margin_milli: 200,
        evidence_window_bytes: 16, max_evidence_windows: 31, max_evidence_spans: 31 }
}
fn document() -> BatchDocument<Int4FaithfulnessDocumentArgs> {
    BatchDocument { id: "transport-only".to_owned(), text: " é\r\n第二节 𐐀 <think> exact tail ".to_owned(),
        task_args: Some(Int4FaithfulnessDocumentArgs { claim: " exact\r\n主张 𐐀 ".to_owned() }) }
}

#[test]
fn original_source_and_repeated_claim_keep_exact_bytes_and_fixed_policy() {
    let original = document();
    let source = original.text.clone();
    let claim = original.task_args.as_ref().unwrap().claim.clone();
    let JudgeRequest::Faithfulness { source: actual, claim: actual_claim, policy: actual_policy, budget }
        = build_request(original, policy(), task(), mapping()).unwrap() else { panic!("wrong judge mode"); };
    assert_eq!(actual, source); assert_eq!(actual_claim, claim);
    assert_eq!(actual_policy, policy()); assert_eq!(budget, task());
    assert!(actual.len() > mapping().chunks.max_chunk_bytes);
}

#[test]
fn strict_wire_args_cannot_replace_policy_budget_or_compiled_heads() {
    let good = canonjson::canonical_string(&Int4FaithfulnessDocumentArgs { claim: "exact".to_owned() }).unwrap();
    assert!(serde_json::from_str::<Int4FaithfulnessDocumentArgs>(&good).is_ok());
    for wire in ["{}", "[]", "null", r#"{"claim":null}"#, r#"{"claim":"one","claim":"two"}"#] {
        assert!(serde_json::from_str::<Int4FaithfulnessDocumentArgs>(wire).is_err());
    }
    for key in ["source", "policy", "budget", "mode", "scores", "execution", "heads", "cohort_chunks"] {
        let mut value = serde_json::json!({ "claim": "exact" }); value[key] = serde_json::json!({});
        assert!(serde_json::from_value::<Int4FaithfulnessDocumentArgs>(value).is_err(), "{key}");
    }
    let row: BatchDocument<Int4FaithfulnessDocumentArgs> = serde_json::from_str(
        r#"{"id":"one","text":"source","task_args":null}"#).unwrap();
    assert_eq!(build_request(row, policy(), task(), mapping()).err().unwrap().fault.code, BatchCode::Planning);
}

#[test]
fn source_plus_claim_has_one_exact_utf8_byte_ceiling_without_tail_truncation() {
    let row = document();
    let bytes = row.text.len() + row.task_args.as_ref().unwrap().claim.len();
    let mut limits = mapping(); limits.chunks.max_input_bytes = bytes;
    build_request(row, policy(), task(), limits).unwrap();
    limits.chunks.max_input_bytes -= 1;
    let failure = build_request(document(), policy(), task(), limits).err().unwrap();
    assert_eq!(failure.fault.code, BatchCode::DocumentLimit); assert!(!failure.stop);
    for source_empty in [true, false] {
        let mut row = document();
        if source_empty { row.text.clear(); } else { row.task_args.as_mut().unwrap().claim.clear(); }
        assert_eq!(build_request(row, policy(), task(), mapping()).err().unwrap().fault.code, BatchCode::DocumentLimit);
    }
}

#[test]
fn every_whole_source_work_axis_must_fit_the_nonrenewable_invocation() {
    validate_limits(task(), JudgeLimits::default(), mapping(), work(8)).unwrap();
    assert!(mapping().max_result_bytes > task().max_output_bytes);
    for axis in 0..5 {
        let mut ceiling = work(4);
        match axis { 0 => ceiling.forward_positions -= 1, 1 => ceiling.projected_logits -= 1,
            2 => ceiling.attention_pairs -= 1, 3 => ceiling.projections.dot_products -= 1,
            _ => ceiling.projections.multiply_accumulates -= 1 }
        assert!(validate_limits(task(), JudgeLimits::default(), mapping(), ceiling).is_err());
        assert_eq!(subtract(ceiling, work(4)).err().unwrap().fault.code, BatchCode::WorkLimit);
    }
    let mut limits = mapping(); limits.max_model_work.projections.multiply_accumulates = 0;
    assert!(validate_limits(task(), JudgeLimits::default(), limits, work(8)).is_err());
    limits = mapping(); limits.chunks.max_input_bytes = 0;
    assert!(validate_limits(task(), JudgeLimits::default(), limits, work(8)).is_err());
    let mut planning = JudgeLimits::default(); planning.max_total_candidate_tokens = 0;
    assert!(validate_limits(task(), planning, mapping(), work(8)).is_err());
}

#[test]
fn ragged_cohort_width_is_not_falsely_limited_to_two_source_chunks() {
    for chunks in [1, 2, 3, 22, 64] { validate_cohort_chunks(chunks, mapping()).unwrap(); }
    for chunks in [0, 65, usize::MAX] { assert!(validate_cohort_chunks(chunks, mapping()).is_err()); }
    let mut limits = mapping(); limits.chunks.max_chunks = 2;
    assert!(validate_cohort_chunks(3, limits).is_err());
    let memory = Int4FaithfulnessMapMemoryLimits { context_tokens: 1024,
        preparation_reserve_bytes: 128 * 1024 * 1024, allocator_reserve_bytes: 1024 * 1024 };
    let quote = memory.cohort_requirements(32, task(), JudgeLimits::default(), mapping(), 32, 22).unwrap();
    let row = Int4MemoryRequirement::for_context(1024, 32).unwrap();
    assert_eq!(quote.memory.native.kv_bytes, row.kv_bytes * 64);
    assert_eq!(quote.memory.max_output_bytes, mapping().max_result_bytes + COHORT_WRAPPER_BYTES);
    // Exact prepared windows still refuse more than 64 native heads; this
    // quote is the maximum admitted physical peak, not a schedule rewrite.
}

#[test]
fn explicit_head_cap_must_hold_the_entire_fixed_policy_without_pruning() {
    for windows in 1..=31 {
        let mut policy = policy();
        policy.max_evidence_windows = windows; policy.max_evidence_spans = windows;
        let required = if windows == 1 { 1 } else { windows as usize + 1 };
        for cap in 1..=MAX_FAITHFULNESS_MAP_HEADS {
            assert_eq!(validate_head_cap(policy, cap).is_ok(), cap >= required,
                "{windows} evidence windows under a {cap}-head declaration");
        }
        for invalid in [0, 33, usize::MAX] { assert!(validate_head_cap(policy, invalid).is_err()); }
    }
    let mut invalid = policy(); invalid.max_evidence_spans = 0;
    assert!(validate_head_cap(invalid, 32).is_err());
    // Resource declarations and the selected window remain independent:
    // 64 chunks can be valid with one head each, while native preflight must
    // still refuse an actual later window containing more than 64 heads.
    validate_cohort_chunks(64, mapping()).unwrap();
}

#[test]
fn physical_summary_counts_full_vocabulary_projections_and_all_ragged_windows() {
    backend::verify_schedule(7, 3, 3, 9, 8, work(1)).unwrap();
    for (chunks, width, windows, steps, projections) in [(0, 3, 0, 1, 1), (7, 0, 3, 9, 8),
        (7, 3, 2, 9, 8), (7, 3, 3, 0, 8), (7, 3, 3, 9, 0), (7, 3, 3, 41, 8), (7, 3, 3, 9, 9)] {
        assert!(backend::verify_schedule(chunks, width, windows, steps, projections, work(1)).is_err());
    }
    let mut corrupt = work(1); corrupt.projected_logits += 1;
    assert!(backend::verify_schedule(7, 3, 3, 9, 8, corrupt).is_err());
}

#[test]
fn flush_epochs_never_renew_any_native_work_axis() {
    let mut run = RunState::new(work(2));
    for (sequence, epoch) in [(1, 1), (3, 2)] {
        run.enter().unwrap();
        run.charge(work(1), BatchRequestContext { request_seq: sequence, epoch, input_line: sequence, byte_offset: 0 }).unwrap();
        run.finish_execution(Ok(()), true).unwrap();
    }
    assert_eq!(run.remaining(), Int4Work::default());
    run.enter().unwrap();
    let failure = run.charge(work(1), BatchRequestContext { request_seq: 4, epoch: 3, input_line: 4, byte_offset: 0 }).err().unwrap();
    assert_eq!(failure.fault.code, BatchCode::WorkLimit); assert!(failure.stop);
    assert!(run.finish_execution::<()>(Err(failure), true).is_err()); assert!(run.is_unavailable());
}

#[test]
fn all_cancellation_causes_survive_nested_native_or_aggregation_errors() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost, ParentCancelled,
        ResourceUnavailable, Shutdown, LinkedExit] {
        for error in [Int4FaithfulnessMapError::Judge(Int4JudgeError::Native(StrictInt4Error::Cancelled(cause))),
            Int4FaithfulnessMapError::Aggregation(FaithfulnessAggregationError::Cancelled(cause))] {
            let mut run = RunState::new(work(2)); run.enter().unwrap();
            run.charge(work(1), BatchRequestContext { request_seq: 1, epoch: 1, input_line: 1, byte_offset: 0 }).unwrap();
            let failure = run.finish_execution::<()>(Err(map_failure(error, true)), false).err().unwrap();
            assert_eq!(failure.fault.cancellation, Some(cause)); assert!(failure.stop);
            assert_eq!(run.remaining(), work(1)); assert!(run.enter().is_err());
        }
    }
}

#[test]
fn planning_refusals_recover_only_before_an_execution_attempt() {
    for executing in [false, true] {
        let failure = map_failure(Int4JudgeError::Task(JudgeError::Limit("private source policy")).into(), executing);
        assert_eq!(failure.fault.code, BatchCode::Planning); assert_eq!(failure.stop, executing);
        for (error, code) in [(JudgeError::AllocationRefused, BatchCode::Allocation),
            (JudgeError::Serialization, BatchCode::Serialization)] {
            let failure = map_failure(Int4JudgeError::Task(error).into(), executing);
            assert_eq!(failure.fault.code, code); assert!(failure.stop);
            assert!(!failure.fault.to_string().contains("private"));
        }
    }
    fn send<T: Send + 'static>() {}
    send::<Int4FaithfulnessDocumentBatchPlanner>(); send::<PreparedInt4FaithfulnessDocument>();
    send::<Int4FaithfulnessDocumentRun>();
}
