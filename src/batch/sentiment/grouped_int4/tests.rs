//! Pure source/authority/accounting contracts. Actual native equivalence is
//! covered by the separately armed CLI fixture, not by fabricated model scores.
use super::*;
use crate::{canonjson, native_engine::{decode::DecodeCancellationKind, portable_int8::ProjectionWork},
    tasks::{mapreduce::ChunkLimits, sentiment::SentimentAggregationPolicy}};

fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 768, max_output_tokens: 16, max_output_bytes: 65_536,
        max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(1024, 32).unwrap().kv_bytes * 4 }
}
fn work(units: u64) -> Int4Work {
    Int4Work { forward_positions: 40 * units, projected_logits: 50 * units, attention_pairs: 60 * units,
        projections: ProjectionWork { dot_products: 70 * units, multiply_accumulates: 80 * units } }
}
fn mapping() -> Int4SentimentMapLimits {
    Int4SentimentMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 8,
        max_chunk_tokens: 768, context_tokens: 1024, reserved_tokens: 16,
        max_chunks: 64, max_tokenizer_calls: 4096 }, max_model_work: work(4),
        max_retained_result_bytes: 1024 * 1024, max_result_bytes: 1024 * 1024,
        aggregation: SentimentAggregationPolicy::default() }
}
fn args() -> SentimentBatchArgs {
    SentimentBatchArgs { axes: vec![SentimentAxis::Dominance, SentimentAxis::Valence], budget: task() }
}
fn document(task_args: Option<SentimentBatchArgs>) -> BatchDocument<SentimentBatchArgs> {
    BatchDocument { id: "transport-key".to_owned(), text: "é\r\n第二节 𐐀 <think> exact tail".to_owned(), task_args }
}

#[test]
fn original_document_and_complete_override_survive_without_prefix_truncation() {
    let defaults = Some(args());
    let source = document(None).text;
    let request = build_request(document(None), &defaults, mapping(), task(), 4, None).unwrap();
    assert_eq!(request.document, source);
    assert!(request.document.len() > mapping().chunks.max_chunk_bytes);
    assert_eq!(request.axes, vec![SentimentAxis::Valence, SentimentAxis::Dominance]);
    let mut replacement = args(); replacement.axes = vec![SentimentAxis::Approach];
    replacement.budget.max_output_bytes /= 2;
    let actual = build_request(document(Some(replacement.clone())), &defaults, mapping(), task(), 4, None).unwrap();
    assert_eq!(actual.document, source); assert_eq!(actual.axes, replacement.axes);
    assert_eq!(actual.budget, replacement.budget);
    assert_eq!(defaults.unwrap().axes, vec![SentimentAxis::Dominance, SentimentAxis::Valence]);
    assert_eq!(build_request(document(None), &None, mapping(), task(), 4, None).err().unwrap().fault.code, BatchCode::Planning);
}

#[test]
fn incomplete_or_invalid_axis_authority_never_merges_into_defaults() {
    for axis in 0..5 {
        let mut expanded = args();
        match axis { 0 => expanded.budget.max_input_tokens += 1, 1 => expanded.budget.max_output_tokens += 1,
            2 => expanded.budget.max_output_bytes += 1, 3 => expanded.budget.max_grammar_states += 1,
            _ => expanded.budget.max_kv_bytes += 1 }
        let error = build_request(document(Some(expanded)), &Some(args()), mapping(), task(), 4, None).err().unwrap();
        assert_eq!(error.fault.code, BatchCode::Planning); assert!(!error.stop);
    }
    for axes in [vec![], vec![SentimentAxis::Valence, SentimentAxis::Valence],
        vec![SentimentAxis::Valence; 5]] {
        let mut bad = args(); bad.axes = axes;
        assert!(validate_args(&bad, task()).is_err());
    }
    let defaults = Some(SentimentBatchArgs { axes: vec![SentimentAxis::Valence], budget: task() });
    build_request(document(None), &defaults, mapping(), task(), 1, Some(64)).unwrap();
    let error = build_request(document(Some(args())), &defaults, mapping(), task(), 1, Some(64)).err().unwrap();
    assert_eq!(error.fault.code, BatchCode::Planning, "a small default never authorizes a wider override");
}

#[test]
fn strict_wire_arguments_refuse_policy_replacement_unknown_axes_and_duplicate_keys() {
    let good = canonjson::canonical_string(&args()).unwrap();
    assert!(serde_json::from_str::<SentimentBatchArgs>(&good).is_ok());
    for wire in ["{}".to_owned(), "[]".to_owned(), r#"{"axes":["valence"]}"#.to_owned(),
        good.replace("valence", "unregistered-axis"),
        good.replacen('{', r#"{"policy":{},"#, 1),
        good.replacen('{', r#"{"axes":[],"#, 1)] {
        assert!(serde_json::from_str::<SentimentBatchArgs>(&wire).is_err());
    }
    let row: BatchDocument<SentimentBatchArgs> = serde_json::from_str(
        r#"{"id":"original","text":"source","task_args":null}"#).unwrap();
    assert!(row.task_args.is_none());
}

#[test]
fn empty_and_oversized_originals_are_rejected_whole() {
    for text in [String::new(), "x".repeat(mapping().chunks.max_input_bytes + 1)] {
        let mut row = document(None); row.text = text;
        let error = build_request(row, &Some(args()), mapping(), task(), 4, None).err().unwrap();
        assert_eq!(error.fault.code, BatchCode::DocumentLimit); assert!(!error.stop);
    }
}

#[test]
fn declared_and_actual_chunk_axis_geometry_is_exact_and_never_clamped() {
    for (axes, chunks) in [(1, 64), (2, 32), (3, 21), (4, 16)] {
        validate_geometry(axes, Some(chunks)).unwrap();
    }
    for (axes, chunks) in [(0, 1), (5, 1), (4, 17), (2, 33), (1, 0), (4, usize::MAX)] {
        assert!(validate_geometry(axes, Some(chunks)).is_err());
    }
    for chunks in [0, 65, usize::MAX] { assert!(validate_cohort_chunks(chunks, mapping()).is_err()); }
    let mut limits = mapping(); limits.chunks.max_chunks = 2;
    assert!(validate_cohort_chunks(3, limits).is_err());
}

#[test]
fn every_work_axis_is_nonrenewable_and_aggregate_output_has_its_own_limit() {
    let planning = SentimentLimits::default();
    validate_limits(task(), planning, mapping(), work(8)).unwrap();
    assert!(mapping().max_result_bytes > task().max_output_bytes);
    for axis in 0..5 {
        let mut cap = work(4);
        match axis { 0 => cap.forward_positions -= 1, 1 => cap.projected_logits -= 1,
            2 => cap.attention_pairs -= 1, 3 => cap.projections.dot_products -= 1,
            _ => cap.projections.multiply_accumulates -= 1 }
        assert!(validate_limits(task(), planning, mapping(), cap).is_err());
        assert_eq!(subtract(cap, work(4)).err().unwrap().fault.code, BatchCode::WorkLimit);
    }
    let mut limits = mapping(); limits.max_model_work.projections.multiply_accumulates = 0;
    assert!(validate_limits(task(), planning, limits, work(8)).is_err());
    let mut planning = planning; planning.max_total_nodes = 0;
    assert!(validate_limits(task(), planning, mapping(), work(8)).is_err());
}

#[test]
fn task_kv_prices_one_complete_axis_bundle_and_not_the_cohort_window() {
    let row = Int4MemoryRequirement::for_context(1024, 32).unwrap().kv_bytes;
    let mut budget = task(); budget.max_kv_bytes = row * 2;
    backend::check_task_kv(1024, 32, 2, budget).unwrap();
    assert!(backend::check_task_kv(1024, 32, 3, budget).is_err());
    budget.max_kv_bytes -= 1;
    assert!(backend::check_task_kv(1024, 32, 2, budget).is_err());
    assert!(budget.max_kv_bytes < row * 64, "physical chunk-cohort KV has a separate owner");
}

#[test]
fn physical_cohort_summary_requires_all_ragged_windows_and_bounded_work() {
    backend::verify_schedule(7, 3, 3, 9, 8, work(1)).unwrap();
    for (chunks, window, windows, steps, groups) in [(0, 3, 0, 1, 1), (7, 0, 3, 9, 8),
        (7, 3, 2, 9, 8), (7, 3, 3, 0, 8), (7, 3, 3, 9, 0), (7, 3, 3, 41, 8), (7, 3, 3, 9, 51)] {
        assert!(backend::verify_schedule(chunks, window, windows, steps, groups, work(1)).is_err());
    }
}

#[test]
fn all_cancellation_causes_survive_native_poison_and_consume_the_attempt() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost, ParentCancelled,
        ResourceUnavailable, Shutdown, LinkedExit] {
        let error = Int4SentimentMapError::Sentiment(Int4SentimentError::Native(StrictInt4Error::Cancelled(cause)));
        let failure = map_failure(error, true);
        let mut run = RunState::new(work(2)); run.enter().unwrap();
        run.charge(work(1), BatchRequestContext { request_seq: 3, epoch: 2, input_line: 4, byte_offset: 128 }).unwrap();
        let failure = run.finish_execution::<()>(Err(failure), false).err().unwrap();
        assert_eq!(failure.fault.cancellation, Some(cause)); assert!(failure.stop);
        assert_eq!(run.remaining(), work(1)); assert!(run.is_unavailable()); assert!(run.enter().is_err());
    }
}

#[test]
fn execution_refusals_stop_even_after_preparation_would_be_recoverable() {
    for error in [Int4SentimentMapError::Work, Int4SentimentMapError::Output,
        Int4SentimentMapError::EmptySource, Int4SentimentMapError::Limits] {
        assert!(map_failure(error, true).stop);
    }
    assert!(!map_failure(Int4SentimentMapError::Work, false).stop);
    assert!(!map_failure(Int4SentimentMapError::Output, false).stop);
    fn send<T: Send + 'static>() {}
    send::<Int4SentimentDocumentBatchPlanner>(); send::<PreparedInt4SentimentDocument>();
    send::<Int4SentimentDocumentRun>();
}

#[test]
fn bounded_planning_refusals_recover_but_nested_storage_failures_keep_their_codes() {
    for executing in [false, true] {
        let bounded = Int4SentimentError::Planning(SentimentPlanningError::Task(
            SentimentError::LimitExceeded("private task planning bound")));
        let failure = map_failure(bounded.into(), executing);
        assert_eq!(failure.fault.code, BatchCode::Planning);
        assert_eq!(failure.stop, executing);
        for (inner, code) in [(SentimentError::AllocationRefused, BatchCode::Allocation),
            (SentimentError::Serialization, BatchCode::Serialization)] {
            let error = Int4SentimentError::Planning(SentimentPlanningError::Task(inner));
            let failure = map_failure(error.into(), executing);
            assert_eq!(failure.fault.code, code); assert!(failure.stop);
            assert!(!failure.fault.to_string().contains("private"));
        }
    }
}
