//! Admission and ownership fixtures; no native model allocation or execution.
use super::*;
use crate::{native_engine::{decode::DecodeCancellationKind, lmhead::NANBEIGE_VOCAB_SIZE},
    tasks::judge::grouped_int4::Int4JudgePlanner};

fn fixture(width: usize) -> (Int4JudgeHostLimits, TaskBudget, Int4Work) {
    let work = Int4Work::for_sequence(width, 0, 64, 4 * NANBEIGE_VOCAB_SIZE).unwrap();
    let native = NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65_536,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000, cleanup_reserve_bytes: 65_536 } };
    (Int4JudgeHostLimits { native, preparation_reserve_bytes: 1 << 20, max_model_work: work },
        TaskBudget { max_input_tokens: 128, max_output_tokens: 16, max_output_bytes: 65_536,
            max_grammar_states: 4096,
            max_kv_bytes: Int4MemoryRequirement::for_context(native.context_tokens, width).unwrap().kv_bytes }, work)
}

#[test]
fn every_group_partition_charges_the_entire_native_capacity() {
    for width in [1, 3, 16, 32, 128, 10752] {
        let (limits, task, work) = fixture(width);
        let (required, scratch) = requirements(limits, task, 64, width, work).unwrap();
        assert_eq!(required, Int4MemoryRequirement::for_context(limits.native.context_tokens, width).unwrap());
        assert_eq!(required.kv_bytes, task.max_kv_bytes);
        assert_eq!(scratch, required.rope_bytes + required.scratch_payload_bound + limits.native.allocator_reserve_bytes);
        let mut prefix_only = task;
        prefix_only.max_kv_bytes = Int4MemoryRequirement::for_context(64, width).unwrap().kv_bytes;
        assert!(requirements(limits, prefix_only, 64, width, work).is_err());
    }
}

#[test]
fn both_orders_all_criteria_and_all_evidence_share_five_work_ceilings() {
    let (mut limits, task, head) = fixture(32);
    let complete = head.checked_add(head).unwrap().checked_add(head).unwrap();
    limits.max_model_work = complete;
    requirements(limits, task, 64, 32, complete).unwrap();
    for axis in 0..5 {
        let mut short = limits;
        match axis {
            0 => short.max_model_work.forward_positions -= 1,
            1 => short.max_model_work.projected_logits -= 1,
            2 => short.max_model_work.attention_pairs -= 1,
            3 => short.max_model_work.projections.dot_products -= 1,
            _ => short.max_model_work.projections.multiply_accumulates -= 1,
        }
        assert!(requirements(short, task, 64, 32, complete).is_err(), "axis {axis}");
    }
}

#[test]
fn reusable_context_is_distinct_from_total_judgment_work() {
    let (mut limits, task, head) = fixture(32);
    let complete = head.checked_add(head).unwrap().checked_add(head).unwrap();
    assert!(complete.forward_positions > limits.native.context_tokens as u64);
    assert!(requirements(limits, task, 64, 32, complete).is_err());
    limits.max_model_work = complete;
    requirements(limits, task, 64, 32, complete).unwrap();
    for context in [0, 129] {
        assert!(requirements(limits, task, context, 32, complete).is_err());
    }
    let mut insufficient = task; insufficient.max_kv_bytes -= 1;
    assert!(requirements(limits, insufficient, 64, 32, complete).is_err());
}

#[test]
fn malformed_budgets_partitions_and_overflow_refuse_before_dispatch() {
    for axis in 0..5 {
        let (mut limits, mut task, work) = fixture(128);
        match axis {
            0 => limits.preparation_reserve_bytes = 0,
            1 => limits.native.allocator_reserve_bytes = 0,
            2 => limits.native.allocator_reserve_bytes = u64::MAX,
            3 => limits.native.run.max_checkpoints = 0,
            _ => task.max_output_bytes = 0,
        }
        assert!(requirements(limits, task, 64, 128, work).is_err(), "axis {axis}");
    }
    let (limits, task, work) = fixture(128);
    for width in [0, 10753, usize::MAX] {
        assert!(requirements(limits, task, 64, width, work).is_err());
    }
}

#[test]
fn cancellation_remains_typed_through_redacted_host_diagnostics() {
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::CostBudget] {
        let error = judge_error(Int4JudgeError::Native(StrictInt4Error::Cancelled(cause)));
        let HostedError::Int4(Int4HostError::Judge(inner)) = &error else { panic!("wrong stage"); };
        assert_eq!(inner.cancellation(), Some(cause));
        assert_eq!(error.to_string(), format!("{error:?}"));
        assert!(error.source().is_some());
    }
}

#[test]
fn sealed_plan_and_complete_result_can_cross_the_owned_host_boundary() {
    fn send<T: Send + 'static>() {}
    send::<PreparedInt4Judge>(); send::<Int4JudgePlanner>();
    send::<Int4JudgeRun>(); send::<Int4JudgeHostLimits>();
    let _host = NlpEngine::execute_int4_judge;
    let _grouped = NlpEngine::execute_int4_judge_layer_major;
}
#[test]
fn every_judgment_head_reuses_one_admitted_prompt_workspace() {
    let (mut limits, task, head) = fixture(32);
    let complete = head.checked_add(head).unwrap().checked_add(head).unwrap();
    limits.max_model_work = complete;
    let (required, base) = requirements(limits, task, 64, 32, complete).unwrap();
    for rows in [1, 4, 64] {
        let extra = Int4PrefillLimits::required_extra_scratch_bytes(rows).unwrap();
        let prefill = Int4PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: extra };
        assert_eq!(super::super::prefill::with_extra_workspace(base, Some(prefill)).unwrap(), base + extra);
        assert_eq!(required.kv_bytes, task.max_kv_bytes);
    }
}
