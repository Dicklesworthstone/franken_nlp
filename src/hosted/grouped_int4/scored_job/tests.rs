//! Pure admission/API fixtures; no fabricated resident model or native claims.
use super::*;
use crate::{
    jobs::{JobWork, population::PopulationReadLimits},
    native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4Work},
    tasks::{classify::grouped_int4::Int4ClassificationPlanner,
        sentiment::grouped_int4::Int4SentimentPlanner,
        judge::grouped_int4::Int4JudgePlanner},
};
use std::time::Duration;

fn fixture(width: usize) -> (TaskBudget, Int4ScoredBatchLimits, JobLimits, JobHostLimits) {
    let native = Int4MemoryRequirement::for_context(32, width).unwrap();
    let head = Int4Work::for_sequence(width, 0, 24, 3 * NANBEIGE_VOCAB_SIZE).unwrap();
    let item = head.checked_add(head).unwrap().checked_add(head).unwrap();
    let work = Int4ScoredBatchLimits { max_item_work: item,
        max_model_work: item.checked_add(item).unwrap() };
    let task = TaskBudget { max_input_tokens: 32, max_output_tokens: 16,
        max_output_bytes: 8192, max_grammar_states: 256, max_kv_bytes: native.kv_bytes };
    let job = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 8192, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20,
        max_attempts: 4, max_work: JobWork::from_int4(work.max_model_work, 0) };
    let host = JobHostLimits {
        native: NativeLimits { context_tokens: 32, allocator_reserve_bytes: 1 << 20,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000,
                cleanup_reserve_bytes: 65536 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
        preparation_reserve_bytes: 128 << 20, io_reserve_bytes: 131072,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
    };
    (task, work, job, host)
}

#[test]
fn full_population_and_group_specific_native_storage_are_charged_together() {
    for width in [1, 3, 32, 127, 128, MAX_MODEL_K] {
        let (task, work, job, host) = fixture(width);
        let needed = requirements(width, task, work, job, host).unwrap();
        assert_eq!(needed.input_bytes, host.required_buffer_bytes(job).unwrap());
        assert_eq!(needed.native, Int4MemoryRequirement::for_context(32, width).unwrap());
        assert_eq!(needed.native.kv_bytes, task.max_kv_bytes);
        assert_eq!(needed.workspace_bytes, needed.native.rope_bytes
            + needed.native.scratch_payload_bound + host.native.allocator_reserve_bytes);
    }
}

#[test]
fn multihead_work_is_independent_of_one_reused_kv_context() {
    let (task, work, job, host) = fixture(128);
    assert!(work.max_item_work.forward_positions > host.native.context_tokens as u64);
    assert!(work.max_model_work.forward_positions > work.max_item_work.forward_positions);
    requirements(128, task, work, job, host).unwrap();
}

#[test]
fn every_lifetime_counter_must_match_and_scoring_never_grants_masks() {
    let (task, work, job, host) = fixture(128);
    for axis in 0..6 {
        let mut wrong = job;
        match axis {
            0 => wrong.max_work.model.forward_positions += 1,
            1 => wrong.max_work.model.projected_logits += 1,
            2 => wrong.max_work.model.attention_pairs += 1,
            3 => wrong.max_work.model.projections.dot_products += 1,
            4 => wrong.max_work.model.projections.multiply_accumulates += 1,
            _ => wrong.max_work.mask_node_visits = 1,
        }
        assert!(requirements(128, task, work, wrong, host).is_err(), "axis {axis}");
    }
}

#[test]
fn item_and_attempt_counts_never_multiply_compute_authority() {
    let (task, work, mut job, host) = fixture(128);
    let before = requirements(128, task, work, job, host).unwrap();
    let immutable = job.max_work;
    job.max_items = 100;
    job.max_attempts = 100;
    let after = requirements(128, task, work, job, host).unwrap();
    assert_eq!(job.max_work, immutable);
    assert_eq!(before.native, after.native);
    assert_eq!(after.input_bytes - before.input_bytes, 98 * 1024);
}

#[test]
fn incomplete_kv_or_result_capacity_refuses_before_population_access() {
    let (task, work, job, host) = fixture(128);
    for axis in 0..4 {
        let mut wrong = task;
        match axis { 0 => wrong.max_kv_bytes = 0, 1 => wrong.max_kv_bytes -= 1,
            2 => wrong.max_output_bytes = 0, _ => wrong.max_output_bytes = job.max_result_bytes as u64 + 1 }
        assert!(requirements(128, wrong, work, job, host).is_err(), "axis {axis}");
    }
    // A larger explicit task ceiling is not a larger native allocation.
    let mut ceiling = task;
    ceiling.max_kv_bytes += 1;
    assert_eq!(requirements(128, ceiling, work, job, host).unwrap().native.kv_bytes, task.max_kv_bytes);
}

#[test]
fn impossible_context_and_missing_or_overflowing_host_storage_are_refused() {
    let (task, work, job, host) = fixture(128);
    for axis in 0..11 {
        let mut wrong = host;
        match axis {
            0 => wrong.native.context_tokens = 0,
            1 => wrong.native.context_tokens = 8193,
            2 => wrong.native.allocator_reserve_bytes = 0,
            3 => wrong.native.allocator_reserve_bytes = u64::MAX,
            4 => wrong.preparation_reserve_bytes = 0,
            5 => wrong.io_reserve_bytes = 0,
            6 => wrong.journal_reserve_bytes = 0,
            7 => wrong.serialization_reserve_bytes = 0,
            8 => wrong.preparation_reserve_bytes = u64::MAX,
            9 => wrong.transport.max_stream_bytes = 0,
            _ => wrong.transport.max_lines = 0,
        }
        assert!(requirements(128, task, work, job, wrong).is_err(), "axis {axis}");
    }
    assert!(requirements(0, task, work, job, host).is_err());
}

#[test]
fn per_item_work_cannot_exceed_any_frozen_lifetime_axis() {
    let (task, mut work, mut job, host) = fixture(128);
    work.max_model_work = work.max_item_work;
    for axis in 0..5 {
        let mut wrong = work;
        match axis {
            0 => wrong.max_model_work.forward_positions -= 1,
            1 => wrong.max_model_work.projected_logits -= 1,
            2 => wrong.max_model_work.attention_pairs -= 1,
            3 => wrong.max_model_work.projections.dot_products -= 1,
            _ => wrong.max_model_work.projections.multiply_accumulates -= 1,
        }
        job.max_work = JobWork::from_int4(wrong.max_model_work, 0);
        assert!(requirements(128, task, wrong, job, host).is_err(), "axis {axis}");
    }
}

#[test]
fn all_three_real_task_factories_cross_the_existing_owned_host_boundary() {
    fn send<T: Send + 'static>() {}
    fn binding<P: Int4ScoredJobTask>() {
        send::<Int4ScoredJobPlanner<P>>();
        send::<JobInput<P, std::io::Cursor<Vec<u8>>>>();
        let _entrypoint = NlpEngine::job_int4_scored::<P, std::io::Cursor<Vec<u8>>>;
    }
    binding::<Int4ClassificationPlanner>();
    binding::<Int4SentimentPlanner>();
    binding::<Int4JudgePlanner>();
}
