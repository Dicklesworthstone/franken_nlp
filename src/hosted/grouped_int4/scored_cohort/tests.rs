//! Pure aggregate admission and owned-boundary checks, not model inference.
use super::*;
use std::time::Duration;
fn case(width: usize) -> (Facts, NativeLimits, Vec<usize>) {
    let contexts = vec![32, 64, 48];
    let work = contexts.iter().try_fold(Int4Work::default(), |sum, &count|
        sum.checked_add(Int4Work::for_sequence(width, 0, count, 1000).unwrap())).unwrap();
    let kv = Int4MemoryRequirement::for_cohort(&contexts, width).unwrap().kv_bytes;
    (Facts { task: TaskBudget { max_input_tokens: 4096, max_output_tokens: 16,
        max_output_bytes: 65536, max_grammar_states: 4096, max_kv_bytes: kv }, work,
        heads: 3, group_width: width, output_bytes: 65536 }, NativeLimits { context_tokens: 64,
        allocator_reserve_bytes: 1 << 20, run: RunLimits { max_elapsed: Duration::from_secs(30),
            max_checkpoints: 1000, cleanup_reserve_bytes: 65536 } }, contexts)
}
#[test]
fn simultaneous_kv_is_the_sum_for_every_supported_group_partition() {
    for width in [1, 3, 32, 128, MAX_MODEL_K] {
        let (facts, native, contexts) = case(width);
        validate_work(facts, native, 1 << 20, facts.work).unwrap();
        let required = geometry(facts, native, &contexts).unwrap();
        assert_eq!(required.kv_bytes, contexts.iter().map(|&count|
            Int4MemoryRequirement::for_context(count, width).unwrap().kv_bytes).sum::<u64>());
        assert!(facts.work.forward_positions > native.context_tokens as u64);
        let mut lower = facts; lower.task.max_kv_bytes -= 1;
        assert!(geometry(lower, native, &contexts).is_err());
        lower.task.max_kv_bytes = Int4MemoryRequirement::for_context(64, width).unwrap().kv_bytes;
        assert!(geometry(lower, native, &contexts).is_err());
    }
}
#[test]
fn every_aggregate_work_axis_refuses_before_dispatch() {
    let (facts, native, _) = case(3);
    for axis in 0..5 {
        let mut cap = facts.work;
        match axis { 0 => cap.forward_positions -= 1, 1 => cap.projected_logits -= 1,
            2 => cap.attention_pairs -= 1, 3 => cap.projections.dot_products -= 1,
            _ => cap.projections.multiply_accumulates -= 1 }
        assert!(validate_work(facts, native, 1 << 20, cap).is_err());
    }
}
#[test]
fn missing_head_bad_context_or_output_ceiling_is_not_repaired() {
    let (facts, native, contexts) = case(128);
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
fn transient_quote_is_shared_with_cli_and_added_to_retained_preparation() {
    let (facts, native, _) = case(32);
    let extra = transient_bytes(facts).unwrap();
    assert_eq!(extra, required_task_workspace_bytes(facts.work, facts.heads, facts.output_bytes).unwrap());
    assert_eq!(extra, facts.work.forward_positions * 64 + facts.output_bytes * 8 + facts.heads as u64 * 65536);
    assert_eq!(sum(&[1 << 20, extra]).unwrap(), (1 << 20) + extra);
    assert!(validate_work(facts, native, 0, facts.work).is_err());
    let mut bad = facts; bad.work.forward_positions = u64::MAX;
    assert!(transient_bytes(bad).is_err());
}
#[test]
fn plan_inputs_outputs_and_entrypoints_cross_the_owned_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Input<PreparedInt4Classification>>(); send::<Input<PreparedInt4Sentiment>>(); send::<Input<PreparedInt4Judge>>();
    send::<Int4ScoredCohort<Int4ClassificationRun>>(); send::<Int4ScoredCohort<Int4SentimentRun>>(); send::<Int4ScoredCohort<Int4JudgeRun>>();
    let _classification = NlpEngine::execute_int4_classify_cohort;
    let _sentiment = NlpEngine::execute_int4_sentiment_cohort;
    let _judge = NlpEngine::execute_int4_judge_cohort;
}
#[test]
fn typed_cancellation_survives_all_three_host_error_mappings() {
    let cause = crate::native_engine::decode::DecodeCancellationKind::Deadline;
    let classification = classification_error(Int4ClassificationError::Native(StrictInt4Error::Cancelled(cause)));
    let sentiment = sentiment_error(Int4SentimentError::Native(StrictInt4Error::Cancelled(cause)));
    let judge = judge_error(Int4JudgeError::Native(StrictInt4Error::Cancelled(cause)));
    for error in [classification, sentiment, judge] {
        assert!(error.source().is_some()); assert_eq!(format!("{error:?}"), error.to_string());
    }
}
