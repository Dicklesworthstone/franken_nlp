use super::*;
use crate::native_engine::{constrained::JsonDecodeError,
    constrained_int4::{Int4JsonError, planned_work}, decode::DecodeCancellationKind};

const PREPARATION: u64 = 600 * 1024 * 1024;

fn fixture(samples: usize) -> (SampledSentimentHostLimits, TaskBudget, Int4Work) {
    let native = NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65_536,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000,
            cleanup_reserve_bytes: 65_536 } };
    let one = planned_work(128, 32, 16).unwrap();
    let work = (0..samples).try_fold(Int4Work::default(), |work, _| work.checked_add(one)).unwrap();
    let task = TaskBudget { max_input_tokens: 32, max_output_tokens: 16 * samples as u32,
        max_output_bytes: 65_536, max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(128, 128).unwrap().kv_bytes };
    let limits = SampledSentimentHostLimits { native, preparation_reserve_bytes: PREPARATION,
        max_model_work: work, max_sampler_bytes: json_sampling_workspace_bytes().unwrap(),
        mask_limits: MaskWorkLimits { max_trie_node_visits: 2_000_000,
            checkpoint_interval_nodes: 256 }, max_mask_node_visits: 32_000_000_000 };
    (limits, task, work)
}

#[test]
fn samples_share_one_native_capacity_and_sampler_reservation() {
    let mut previous = None;
    for samples in [1, 4, 32] {
        let (limits, task, work) = fixture(samples);
        let (native, scratch) = requirements(limits, task, 47, 128, work, PREPARATION).unwrap();
        assert_eq!(scratch, native.rope_bytes + native.scratch_payload_bound
            + limits.native.allocator_reserve_bytes + json_sampling_workspace_bytes().unwrap());
        assert_eq!(native.kv_bytes, task.max_kv_bytes);
        if let Some(expected) = previous { assert_eq!((native, scratch), expected); }
        previous = Some((native, scratch));
        assert_eq!(work, limits.max_model_work);
    }
}

#[test]
fn every_complete_model_work_axis_is_admitted_before_native_allocation() {
    let (baseline, task, work) = fixture(4);
    for axis in 0..5 {
        let mut limits = baseline;
        match axis {
            0 => limits.max_model_work.forward_positions -= 1,
            1 => limits.max_model_work.projected_logits -= 1,
            2 => limits.max_model_work.attention_pairs -= 1,
            3 => limits.max_model_work.projections.dot_products -= 1,
            _ => limits.max_model_work.projections.multiply_accumulates -= 1,
        }
        assert!(requirements(limits, task, 47, 128, work, PREPARATION).is_err());
    }
}

#[test]
fn preparation_sampler_kv_and_mask_refusals_precede_native_allocation() {
    let (baseline, task, work) = fixture(4);
    for axis in 0..9 {
        let mut limits = baseline;
        let mut task = task;
        match axis {
            0 => limits.preparation_reserve_bytes -= 1,
            1 => limits.max_sampler_bytes -= 1,
            2 => task.max_kv_bytes -= 1,
            3 => limits.native.context_tokens = 46,
            4 => limits.native.allocator_reserve_bytes = 0,
            5 => limits.mask_limits.max_trie_node_visits = 0,
            6 => limits.mask_limits.checkpoint_interval_nodes = 0,
            7 => limits.max_mask_node_visits = 0,
            _ => limits.native.run.max_elapsed = Duration::ZERO,
        }
        assert!(requirements(limits, task, 47, 128, work, PREPARATION).is_err());
    }
    assert!(requirements(baseline, task, 0, 128, work, PREPARATION).is_err());
    assert!(requirements(baseline, task, 47, 128, work, 0).is_err());
    assert!(requirements(SampledSentimentHostLimits { native: NativeLimits {
        allocator_reserve_bytes: u64::MAX, ..baseline.native }, ..baseline },
        task, 47, 128, work, PREPARATION).is_err());
}

#[test]
fn task_errors_retain_cancellation_without_disclosing_nested_text() {
    let error = task_error(SampledSentimentError::Native(Int4JsonError::Decode(
        JsonDecodeError::InvalidRequest("PRIVATE_DOCUMENT_OR_SEED"))));
    assert!(!format!("{error:?} {error}").contains("PRIVATE_DOCUMENT_OR_SEED"));
    assert!(error.source().is_some());
    let error = task_error(SampledSentimentError::Cancelled(DecodeCancellationKind::ParentCancelled));
    let HostedError::Int4(Int4HostError::SampledSentiment(error)) = error else {
        panic!("sampled task error was flattened");
    };
    assert_eq!(error.cancellation(), Some(DecodeCancellationKind::ParentCancelled));
}
