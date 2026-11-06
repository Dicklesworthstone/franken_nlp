//! Pure ownership/accounting regressions. Native corpus/map equivalence is
//! exercised separately by the explicitly armed CLI model fixture.
use super::*;
use crate::{canonjson, native_engine::decode::DecodeCancellationKind,
    tasks::{classify::{ClassificationAggregationPolicy, ClassificationPolicy, ClassificationPlanningError},
        mapreduce::ChunkLimits}};

fn work(units: u64) -> Int4Work {
    Int4Work { forward_positions: 40 * units, projected_logits: 50 * units, attention_pairs: 60 * units,
        projections: ProjectionWork { dot_products: 70 * units, multiply_accumulates: 80 * units } }
}
fn task() -> TaskBudget {
    TaskBudget { max_input_tokens: 768, max_output_tokens: 16, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(1024, 32).unwrap().kv_bytes }
}
fn planning() -> ClassificationLimits {
    ClassificationLimits { max_labels: 4, max_input_bytes: 4096, max_total_label_bytes: 1024,
        max_context_tokens: 1024, ..Default::default() }
}
fn mapping() -> Int4ClassificationMapLimits {
    Int4ClassificationMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 8,
        max_chunk_tokens: 768, context_tokens: 1024, reserved_tokens: 16, max_chunks: 64, max_tokenizer_calls: 4096 },
        max_model_work: work(4), max_retained_result_bytes: 1024 * 1024, max_result_bytes: 1024 * 1024,
        aggregation: ClassificationAggregationPolicy::default() }
}
fn args() -> ClassificationBatchArgs {
    ClassificationBatchArgs { labels: ["alpha", "beta"].map(|id| ClassificationLabel {
        id: id.to_owned(), description: format!("Private description of {id}") }).to_vec(),
        mode: ClassificationMode::Exclusive, policy: ClassificationPolicy::default(), budget: task() }
}
fn context(seq: u64, epoch: u64) -> BatchRequestContext {
    BatchRequestContext { request_seq: seq, epoch, input_line: seq, byte_offset: 0 }
}
fn document(task_args: Option<ClassificationBatchArgs>) -> BatchDocument<ClassificationBatchArgs> {
    BatchDocument { id: "not-a-semantic-key".to_owned(), text: "é\r\n𐐀 <think> exact tail".to_owned(), task_args }
}
fn lower(work: &mut Int4Work, axis: usize) {
    match axis { 0 => work.forward_positions -= 1, 1 => work.projected_logits -= 1,
        2 => work.attention_pairs -= 1, 3 => work.projections.dot_products -= 1,
        _ => work.projections.multiply_accumulates -= 1 }
}

#[test]
fn original_source_is_owned_exactly_and_overrides_replace_every_default_field() {
    let defaults = Some(args());
    let source = document(None).text;
    let from_default = build_request(document(None), &defaults, mapping(), task(), planning()).unwrap();
    assert_eq!(from_default.document, source);
    assert!(from_default.document.len() > mapping().chunks.max_chunk_bytes);
    assert_eq!(from_default.labels.len(), 2);
    let mut replacement = args(); replacement.labels.truncate(1);
    replacement.labels[0].id = "replacement-only".to_owned();
    replacement.mode = ClassificationMode::MultiLabel;
    replacement.policy.minimum_margin_ppm = 500_000;
    replacement.budget.max_output_bytes /= 2;
    let actual = build_request(document(Some(replacement.clone())), &defaults, mapping(), task(), planning()).unwrap();
    assert_eq!(actual.document, source);
    assert!(actual.labels == replacement.labels);
    assert_eq!(actual.mode, replacement.mode); assert_eq!(actual.policy, replacement.policy);
    assert_eq!(actual.budget, replacement.budget);
    assert_eq!(defaults.as_ref().unwrap().labels.len(), 2);
    assert_eq!(build_request(document(None), &None, mapping(), task(), planning()).err().unwrap().fault.code, BatchCode::Planning);
}

#[test]
fn empty_or_oversized_originals_are_refused_instead_of_using_a_context_prefix() {
    let defaults = Some(args());
    for source in [String::new(), "x".repeat(mapping().chunks.max_input_bytes + 1)] {
        let mut row = document(None); row.text = source;
        let failure = build_request(row, &defaults, mapping(), task(), planning()).err().unwrap();
        assert_eq!(failure.fault.code, BatchCode::DocumentLimit); assert!(!failure.stop);
    }
}

#[test]
fn a_record_can_shrink_but_never_expand_any_task_authority_axis() {
    for axis in 0..5 {
        let mut request = args();
        match axis { 0 => request.budget.max_input_tokens += 1, 1 => request.budget.max_output_tokens += 1,
            2 => request.budget.max_output_bytes += 1, 3 => request.budget.max_grammar_states += 1,
            _ => request.budget.max_kv_bytes += 1 }
        let failure = build_request(document(Some(request)), &Some(args()), mapping(), task(), planning()).err().unwrap();
        assert_eq!(failure.fault.code, BatchCode::Planning); assert!(!failure.stop);
    }
    for fault in 0..6 {
        let mut request = args();
        match fault { 0 => request.labels.clear(), 1 => request.labels[1].id = request.labels[0].id.clone(),
            2 => request.labels[0].id = "\nprivate".to_owned(), 3 => request.policy.minimum_margin_ppm = 1_000_001,
            4 => request.labels[0].description = "x".repeat(planning().max_total_label_bytes + 1),
            _ => { request.labels.pop(); } }
        assert!(validate_args(&request, task(), planning()).is_err(), "fault {fault}");
    }
}

#[test]
fn duplicate_unknown_and_partial_task_objects_never_become_merged_defaults() {
    let encoded = canonjson::canonical_string(&args()).unwrap();
    let good = format!(r#"{{"id":"x","text":"source","task_args":{encoded}}}"#);
    let value = canonjson::parse_str(&good).unwrap();
    assert!(serde_json::from_value::<BatchDocument<ClassificationBatchArgs>>(value).is_ok());
    for task in [r#"{}"#, r#"{"labels":[]}"#] {
        let raw = format!(r#"{{"id":"x","text":"source","task_args":{task}}}"#);
        assert!(serde_json::from_value::<BatchDocument<ClassificationBatchArgs>>(canonjson::parse_str(&raw).unwrap()).is_err());
    }
    let mut value = canonjson::parse_str(&good).unwrap();
    value["task_args"]["model_identity"] = serde_json::json!("caller cannot replace the model");
    assert!(serde_json::from_value::<BatchDocument<ClassificationBatchArgs>>(value).is_err());
    assert!(canonjson::parse_str(r#"{"id":"x","text":"first","text":"second"}"#).is_err());
    for raw in [r#"{"id":"x","text":"source"}"#, r#"{"id":"x","text":"source","task_args":null}"#] {
        let row = serde_json::from_value::<BatchDocument<ClassificationBatchArgs>>(canonjson::parse_str(raw).unwrap()).unwrap();
        assert!(row.task_args.is_none());
        assert_eq!(build_request(row, &Some(args()), mapping(), task(), planning()).unwrap().labels.len(), 2);
    }
}

#[test]
fn item_ceiling_and_nonrenewable_invocation_ceiling_cover_every_native_axis() {
    validate_limits(task(), planning(), mapping(), work(8)).unwrap();
    assert!(mapping().max_result_bytes > task().max_output_bytes,
        "a complete aggregate must not be confused with one chunk result");
    for axis in 0..5 {
        let mut cap = work(4); lower(&mut cap, axis);
        assert!(validate_limits(task(), planning(), mapping(), cap).is_err(), "axis {axis}");
        let before = cap;
        assert_eq!(subtract(cap, work(4)).unwrap_err().fault.code, BatchCode::WorkLimit);
        assert_eq!(cap, before, "a failing later axis must not partially change earlier counters");
    }
    let mut bad = mapping(); bad.chunks.max_input_bytes = planning().max_input_bytes + 1;
    assert!(validate_limits(task(), planning(), bad, work(8)).is_err());
    let mut bad = mapping(); bad.chunks.context_tokens = planning().max_context_tokens + 1;
    assert!(validate_limits(task(), planning(), bad, work(8)).is_err());
}

#[test]
fn successful_documents_across_flush_epochs_never_renew_the_native_budget() {
    let mut run = RunState::new(work(2));
    for (sequence, epoch) in [(1, 1), (3, 2)] {
        run.enter().unwrap();
        run.finish_preparation(Ok(()), true).unwrap();
        run.enter().unwrap(); run.charge(work(1), context(sequence, epoch)).unwrap();
        run.finish_execution(Ok(()), true).unwrap();
    }
    assert_eq!(run.remaining, work(0));
    run.enter().unwrap();
    let error = subtract(run.remaining, work(1)).unwrap_err();
    assert_eq!(run.finish_preparation::<()>(Err(error), true).unwrap_err().fault.code, BatchCode::WorkLimit);
    assert_eq!(run.state, State::Ready); assert_eq!(run.remaining, work(0));
}

#[test]
fn execution_refusal_and_unwind_keep_the_entire_attempt_debited_and_stop_reuse() {
    for clean in [true, false] {
        let mut run = RunState::new(work(2)); run.enter().unwrap();
        run.charge(work(1), context(1, 1)).unwrap();
        let failure = run.finish_execution::<()>(Err(BatchItemFailure::reject(BatchCode::Admission)), clean).unwrap_err();
        assert!(failure.stop); assert_eq!(failure.fault.code, BatchCode::Admission);
        assert_eq!(run.remaining, work(1)); assert_eq!(run.state, State::Failed); assert!(run.enter().is_err());
    }
    let mut run = RunState::new(work(2));
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run.enter().unwrap(); run.charge(work(1), context(1, 1)).unwrap();
        panic!("private lifecycle fixture");
    }));
    assert!(panic.is_err()); assert_eq!(run.remaining, work(1)); assert!(run.enter().is_err());
}

#[test]
fn every_cancellation_cause_survives_poisoned_native_settlement() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost, ParentCancelled,
        ResourceUnavailable, Shutdown, LinkedExit] {
        let error = Int4ClassificationError::Planning(ClassificationPlanningError::Cancelled(cause));
        let failure = map_failure(error.into(), true);
        let mut run = RunState::new(work(2)); run.enter().unwrap(); run.charge(work(1), context(1, 1)).unwrap();
        let settled = run.finish_execution::<()>(Err(failure), false).unwrap_err();
        assert!(settled.stop); assert_eq!(settled.fault, BatchFault::cancelled(cause));
        assert_eq!(run.remaining, work(1)); assert_eq!(run.state, State::Failed);
    }
}

#[test]
fn stale_delivery_context_and_dirty_success_cannot_reenter_a_finished_native_attempt() {
    let mut run = RunState::new(work(3)); run.enter().unwrap(); run.charge(work(1), context(2, 1)).unwrap();
    run.finish_execution(Ok(()), true).unwrap();
    run.enter().unwrap();
    let error = run.charge(work(1), context(2, 2)).unwrap_err();
    assert_eq!(error.fault.code, BatchCode::InvalidExecution);
    assert_eq!(run.remaining, work(2));
    assert!(run.finish_execution::<()>(Err(error), true).unwrap_err().stop);
    let mut dirty = RunState::new(work(1)); dirty.enter().unwrap();
    assert_eq!(dirty.finish_execution(Ok(()), false).unwrap_err().fault.code, BatchCode::InvalidExecution);
    assert_eq!(dirty.state, State::Failed);
}

#[test]
fn preparation_and_publication_owners_fit_the_existing_physical_host_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Int4ClassificationDocumentBatchPlanner>();
    send::<PreparedInt4ClassificationDocument>();
    send::<Int4ClassificationMapRun>();
}
