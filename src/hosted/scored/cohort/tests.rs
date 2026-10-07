//! Pure admission/ownership type checks; no fake resident model or runtime.
use super::*;
fn case() -> (Facts, NativeLimits, Vec<usize>) {
    let contexts = vec![32, 64, 48];
    let work = contexts.iter().try_fold(Int8Work::default(), |sum, &n|
        sum.checked_add(Int8Work::for_sequence(0, n, 1000).unwrap())).unwrap();
    let kv = Int8MemoryRequirement::for_cohort(&contexts).unwrap().kv_bytes;
    (Facts { task: TaskBudget { max_input_tokens: 4096, max_output_tokens: 16,
        max_output_bytes: 65536, max_grammar_states: 4096, max_kv_bytes: kv },
        work, heads: 3, output_bytes: 65536 }, NativeLimits { context_tokens: 64, allocator_reserve_bytes: 1 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 1000, cleanup_reserve_bytes: 65536 } }, contexts)
}
#[test]
fn simultaneous_kv_is_a_sum_not_largest_head_or_total_forward_context() {
    let (facts, native, contexts) = case();
    validate_work(facts, native, 1 << 20, facts.work).unwrap();
    let required = geometry(facts, native, &contexts).unwrap();
    assert_eq!(required.kv_bytes, contexts.iter().map(|&n|
        Int8MemoryRequirement::for_context(n).unwrap().kv_bytes).sum::<u64>());
    assert!(facts.work.forward_positions > native.context_tokens as u64);
    let mut lower = facts; lower.task.max_kv_bytes -= 1;
    assert!(geometry(lower, native, &contexts).is_err());
    lower.task.max_kv_bytes = Int8MemoryRequirement::for_context(64).unwrap().kv_bytes;
    assert!(geometry(lower, native, &contexts).is_err());
}
#[test]
fn every_aggregate_work_axis_refuses_before_dispatch() {
    let (facts, native, _) = case();
    for axis in 0..5 {
        let mut cap = facts.work;
        match axis { 0 => cap.forward_positions -= 1, 1 => cap.projected_logits -= 1,
            2 => cap.attention_pairs -= 1, 3 => cap.projections.dot_products -= 1,
            _ => cap.projections.multiply_accumulates -= 1 }
        assert!(validate_work(facts, native, 1 << 20, cap).is_err());
    }
}
#[test]
fn missing_head_wrong_context_and_incomplete_output_allowance_are_not_repaired() {
    let (facts, native, contexts) = case();
    assert!(geometry(facts, native, &contexts[..2]).is_err());
    for bad in [0, 65, usize::MAX] {
        let mut contexts = contexts.clone(); contexts[2] = bad;
        assert!(geometry(facts, native, &contexts).is_err());
    }
    for width in [0, 65] {
        let mut bad = facts; bad.heads = width;
        assert!(validate_work(bad, native, 1 << 20, facts.work).is_err());
    }
    let mut bad = facts; bad.output_bytes += 1;
    assert!(validate_work(bad, native, 1 << 20, facts.work).is_err());
}
#[test]
fn prompt_cursor_and_head_staging_add_to_instead_of_replace_retained_preparation() {
    let (facts, native, _) = case();
    let extra = transient_bytes(facts).unwrap();
    assert_eq!(extra, facts.work.forward_positions * 64 + facts.output_bytes * 8 + facts.heads as u64 * 65536);
    assert_eq!(sum(&[1 << 20, extra]).unwrap(), (1 << 20) + extra);
    assert!(validate_work(facts, native, 0, facts.work).is_err());
    let mut bad = facts; bad.work.forward_positions = u64::MAX;
    assert!(transient_bytes(bad).is_err());
    bad = facts; bad.output_bytes = u64::MAX;
    assert!(transient_bytes(bad).is_err());
}
#[test]
fn cohort_plan_packages_and_outputs_cross_the_real_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Input<PreparedInt8Classification>>(); send::<Input<PreparedInt8Sentiment>>(); send::<Input<PreparedInt8Judge>>();
    send::<Int8ScoredCohort<Int8ClassificationRun>>(); send::<Int8ScoredCohort<Int8SentimentRun>>(); send::<Int8ScoredCohort<Int8JudgeRun>>();
}
#[test]
fn task_cancellation_causes_survive_the_closed_host_dispatch_mapping() {
    let cause = crate::native_engine::decode::DecodeCancellationKind::Deadline;
    let classification = HostedError::Classification(Int8ClassificationError::Native(StrictInt8Error::Cancelled(cause)));
    let sentiment = HostedError::Sentiment(Int8SentimentError::Native(StrictInt8Error::Cancelled(cause)));
    let judge = HostedError::Judge(Int8JudgeError::Native(StrictInt8Error::Cancelled(cause)));
    for error in [classification, sentiment, judge] {
        assert!(error.source().is_some()); assert_eq!(format!("{error:?}"), error.to_string());
    }
}
