//! Complete-bundle admission fixtures; no native model execution.
use super::*;
use crate::{native_engine::{decode::DecodeCancellationKind, lmhead::NANBEIGE_VOCAB_SIZE},
    tasks::sentiment::grouped_int4::Int4SentimentPlanner};
fn fixture(width: usize) -> (Int4SentimentHostLimits, TaskBudget, Int4Work) {
    let work = Int4Work::for_sequence(width, 0, 64, 4 * NANBEIGE_VOCAB_SIZE).unwrap();
    let native = NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65_536,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000, cleanup_reserve_bytes: 65_536 } };
    (Int4SentimentHostLimits { native, preparation_reserve_bytes: 1 << 20, max_model_work: work },
        TaskBudget { max_input_tokens: 128, max_output_tokens: 16, max_output_bytes: 65_536,
            max_grammar_states: 4096,
            max_kv_bytes: Int4MemoryRequirement::for_context(native.context_tokens, width).unwrap().kv_bytes }, work)
}
#[test]
fn every_group_partition_prices_the_complete_native_allocation() {
    for width in [1, 3, 16, 32, 128, 10752] {
        let (limits, task, work) = fixture(width);
        let (required, scratch) = requirements(limits, task, 64, width, work).unwrap();
        assert_eq!(required, Int4MemoryRequirement::for_context(limits.native.context_tokens, width).unwrap());
        assert_eq!(required.kv_bytes, task.max_kv_bytes);
        assert_eq!(scratch, required.rope_bytes + required.scratch_payload_bound + limits.native.allocator_reserve_bytes);
    }
}
#[test]
fn all_five_work_axes_are_nonrenewable_across_dimensions() {
    for axis in 0..5 {
        let (mut limits, task, work) = fixture(32);
        match axis { 0 => limits.max_model_work.forward_positions -= 1,
            1 => limits.max_model_work.projected_logits -= 1, 2 => limits.max_model_work.attention_pairs -= 1,
            3 => limits.max_model_work.projections.dot_products -= 1,
            _ => limits.max_model_work.projections.multiply_accumulates -= 1 }
        assert!(requirements(limits, task, 64, 32, work).is_err(), "axis {axis}");
    }
}
#[test]
fn largest_head_sets_capacity_while_complete_bundle_spends_work() {
    let (mut limits, task, work) = fixture(32);
    let bundle = work.checked_add(work).unwrap().checked_add(work).unwrap();
    assert!(bundle.forward_positions > limits.native.context_tokens as u64);
    assert!(requirements(limits, task, 64, 32, bundle).is_err());
    limits.max_model_work = bundle; requirements(limits, task, 64, 32, bundle).unwrap();
    for context in [0, 129] { assert!(requirements(limits, task, context, 32, bundle).is_err()); }
    let mut insufficient = task; insufficient.max_kv_bytes -= 1;
    assert!(requirements(limits, insufficient, 64, 32, bundle).is_err());
}
#[test]
fn invalid_preparation_partition_or_overflow_refuses_before_execution() {
    for axis in 0..4 {
        let (mut limits, task, work) = fixture(128);
        match axis { 0 => limits.preparation_reserve_bytes = 0, 1 => limits.native.allocator_reserve_bytes = 0,
            2 => limits.native.allocator_reserve_bytes = u64::MAX, _ => limits.native.run.max_checkpoints = 0 }
        assert!(requirements(limits, task, 64, 128, work).is_err(), "axis {axis}");
    }
    let (limits, task, work) = fixture(128);
    for width in [0, 10753, usize::MAX] { assert!(requirements(limits, task, 64, width, work).is_err()); }
}
#[test]
fn typed_cancellation_survives_redacted_diagnostics() {
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::CostBudget] {
        let error = sentiment_error(Int4SentimentError::Native(StrictInt4Error::Cancelled(cause)));
        let HostedError::Int4(Int4HostError::Sentiment(inner)) = &error else { panic!("wrong stage") };
        assert_eq!(inner.cancellation(), Some(cause)); assert_eq!(error.to_string(), format!("{error:?}"));
        assert!(error.source().is_some());
    }
}
#[test]
fn sealed_plans_results_and_limits_cross_the_owned_host_boundary() {
    fn send<T: Send + 'static>() {}
    send::<PreparedInt4Sentiment>(); send::<Int4SentimentPlanner>(); send::<Int4SentimentRun>(); send::<Int4SentimentHostLimits>();
    let _host = NlpEngine::execute_int4_sentiment;
}
