//! Pure ownership/admission regressions; no model, tokenizer or native run.
use super::*;
use crate::{native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4Work},
    tasks::{mapreduce::ChunkLimits, sentiment::SentimentAggregationPolicy}};

fn fixture() -> (Int4SentimentMapMemoryLimits, TaskBudget, SentimentLimits, Int4SentimentMapLimits) {
    let context_tokens = 256;
    let row = Int4MemoryRequirement::for_context(context_tokens, 32).unwrap();
    let task = TaskBudget { max_input_tokens: 128, max_output_tokens: 16,
        max_output_bytes: 65_536, max_grammar_states: 4096, max_kv_bytes: row.kv_bytes * 4 };
    let memory = Int4SentimentMapMemoryLimits { context_tokens,
        preparation_reserve_bytes: MIN_SENTIMENT_MAP_PREPARATION_BYTES,
        allocator_reserve_bytes: 65_536, max_axes: 4, cohort_chunks: None };
    let mapping = Int4SentimentMapLimits { chunks: ChunkLimits {
        max_input_bytes: 4096, max_chunk_bytes: 512, max_chunk_tokens: 128,
        context_tokens, reserved_tokens: 16, max_chunks: 8, max_tokenizer_calls: 1024 },
        max_model_work: Int4Work::for_sequence(32, 0, 1024, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
        max_retained_result_bytes: 8 * 1024 * 1024, max_result_bytes: 1024 * 1024,
        aggregation: SentimentAggregationPolicy::default() };
    (memory, task, SentimentLimits::default(), mapping)
}

#[test]
fn serial_reuses_one_cache_while_cohorts_sum_every_independent_axis() {
    let (mut memory, task, planning, mapping) = fixture();
    for width in [1, 3, 16, 32, 128, 10752] {
        let serial = memory.requirements(width, task, planning, mapping).unwrap();
        let row = Int4MemoryRequirement::for_context(memory.context_tokens, width).unwrap();
        assert_eq!(serial.native, row);
        assert_eq!(serial.workspace_bytes, row.rope_bytes + row.scratch_payload_bound + memory.allocator_reserve_bytes);
        memory.cohort_chunks = Some(4);
        let cohort = memory.requirements(width, task, planning, mapping).unwrap();
        assert_eq!(cohort.native,
            Int4MemoryRequirement::for_cohort(&[memory.context_tokens; 16], width).unwrap());
        assert_eq!(cohort.native.kv_bytes, row.kv_bytes * 16);
        assert_eq!(cohort.native.rope_bytes, row.rope_bytes);
        assert!(cohort.native.scratch_payload_bound > row.scratch_payload_bound);
        assert_eq!(cohort.max_cohort_output_bytes, task.max_output_bytes * 4);
        assert!(cohort.input_bytes > serial.input_bytes);
        memory.cohort_chunks = None;
    }
}

#[test]
fn unknown_axes_quote_the_largest_valid_geometry_without_authorizing_clamping() {
    let (mut memory, task, planning, mut mapping) = fixture();
    mapping.chunks.max_chunks = 128;
    memory.cohort_chunks = Some(64);
    let required = memory.requirements(32, task, planning, mapping).unwrap();
    assert_eq!(required.native,
        Int4MemoryRequirement::for_cohort(&[memory.context_tokens; 64], 32).unwrap());
    memory.cohort_chunks = Some(65);
    assert!(memory.requirements(32, task, planning, mapping).is_err());
    memory.cohort_chunks = Some(4);
    mapping.chunks.max_chunks = 3;
    assert!(memory.requirements(32, task, planning, mapping).is_err());
}

#[test]
fn complete_frontier_and_current_window_are_separate_from_final_output() {
    let (memory, task, planning, mut mapping) = fixture();
    mapping.max_retained_result_bytes = task.max_output_bytes;
    let one = memory.requirements(32, task, planning, mapping).unwrap();
    mapping.max_retained_result_bytes *= 2;
    let two = memory.requirements(32, task, planning, mapping).unwrap();
    assert_eq!(two.input_bytes - one.input_bytes, task.max_output_bytes * 8);
    assert_eq!(two.output_reservation_bytes, one.output_reservation_bytes);
    mapping.max_result_bytes *= 2;
    let larger_output = memory.requirements(32, task, planning, mapping).unwrap();
    assert_eq!(larger_output.input_bytes, two.input_bytes);
    assert_eq!(larger_output.max_output_bytes, mapping.max_result_bytes + WRAPPER_BYTES);
    assert_eq!(larger_output.output_reservation_bytes,
        4 * larger_output.max_output_bytes + WRAPPER_BYTES);
    assert_eq!(one.task_workspace_bytes, required_task_workspace_bytes(mapping.max_model_work,
        memory.max_axes, task.max_output_bytes).unwrap());
    assert!(one.input_bytes > one.task_workspace_bytes + memory.preparation_reserve_bytes);
}

#[test]
fn source_and_every_retained_plan_are_priced_before_construction() {
    let (memory, task, planning, mut mapping) = fixture();
    let before = memory.requirements(32, task, planning, mapping).unwrap();
    mapping.chunks.max_input_bytes *= 2;
    let source = memory.requirements(32, task, planning, mapping).unwrap();
    assert_eq!(source.input_bytes - before.input_bytes, 4 * 4096);
    assert_eq!(source.native, before.native);
    mapping.chunks.max_chunks *= 2;
    let plans = memory.requirements(32, task, planning, mapping).unwrap();
    assert!(plans.input_bytes > source.input_bytes);
    assert_eq!(plans.task_workspace_bytes, source.task_workspace_bytes);
    // A failed next plan is still admitted even after accepted plans consume
    // the complete forward allowance; the bound is not just the work cap.
    let plan_tokens = mapping.max_model_work.forward_positions
        + u64::from(task.max_input_tokens) * memory.max_axes as u64;
    assert!(plans.input_bytes >= plan_tokens * PLAN_BYTES_PER_TOKEN);
}

#[test]
fn fixed_per_chunk_task_authority_is_not_silently_widened_by_the_schedule() {
    let (mut memory, mut task, planning, mapping) = fixture();
    task.max_kv_bytes -= 1;
    assert!(matches!(memory.requirements(32, task, planning, mapping), Err(Int4SentimentMapError::Admission)));
    memory.cohort_chunks = Some(2);
    assert!(matches!(memory.requirements(32, task, planning, mapping), Err(Int4SentimentMapError::Admission)));
    memory.max_axes = 1;
    assert!(memory.requirements(32, task, planning, mapping).is_ok());
}

#[test]
fn invalid_axes_work_limits_and_overflow_refuse_pure_admission() {
    for axis in 0..9 {
        let (mut memory, mut task, mut planning, mut mapping) = fixture();
        match axis {
            0 => memory.max_axes = 0, 1 => memory.max_axes = 5,
            2 => memory.preparation_reserve_bytes = MIN_SENTIMENT_MAP_PREPARATION_BYTES - 1,
            3 => memory.allocator_reserve_bytes = 0, 4 => memory.preparation_reserve_bytes = u64::MAX,
            5 => mapping.max_model_work.forward_positions = 0,
            6 => planning.per_axis.max_nodes = 0, 7 => task.max_input_tokens = 0,
            _ => memory.context_tokens = mapping.chunks.context_tokens - 1,
        }
        assert!(memory.requirements(32, task, planning, mapping).is_err(), "axis {axis}");
    }
    let (mut memory, mut task, mut planning, mapping) = fixture();
    task.max_output_bytes = MAX_WINDOW_BYTES;
    planning.max_output_bytes = MAX_WINDOW_BYTES;
    memory.cohort_chunks = Some(2);
    assert!(memory.requirements(32, task, planning, mapping).is_err());
    let (memory, task, planning, mapping) = fixture();
    for width in [0, 10753, usize::MAX] {
        assert!(memory.requirements(width, task, planning, mapping).is_err());
    }
}

#[test]
fn compact_workspace_is_admitted_once_beside_plans_in_every_schedule() {
    use crate::tokenizer::untrusted::SafeBpeLimits;
    let policy = SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 100_000, max_pair_bytes: 1_000_000 };
    let encoding = UntrustedEncoding::SafeBpeV1(policy);
    let encoder_bytes = policy.requirements().unwrap().workspace_bytes;
    let (mut memory, task, planning, mapping) = fixture();
    for axes in [1, 4] {
        memory.max_axes = axes;
        for cohort in [None, Some(1), Some(4)] {
            memory.cohort_chunks = cohort;
            let byte = memory.requirements(32, task, planning, mapping).unwrap();
            assert_eq!(byte.with_input_encoding(&UntrustedEncoding::ByteFallback).unwrap(), byte);
            let compact = byte.with_input_encoding(&encoding).unwrap();
            assert_eq!(compact.input_bytes, byte.input_bytes + encoder_bytes);
            assert_eq!(compact.native, byte.native);
            assert_eq!(compact.task_workspace_bytes, byte.task_workspace_bytes);
            assert_eq!(compact.workspace_bytes, byte.workspace_bytes);
            assert_eq!(compact.output_reservation_bytes, byte.output_reservation_bytes);
        }
    }
    let quote = memory.requirements(32, task, planning, mapping).unwrap();
    for axis in 0..4 {
        let mut invalid = policy;
        match axis { 0 => invalid.max_input_bytes = 0, 1 => invalid.max_output_tokens = 0,
            2 => invalid.max_pair_checks = 0, _ => invalid.max_pair_bytes = 0 }
        assert!(quote.with_input_encoding(&UntrustedEncoding::SafeBpeV1(invalid)).is_err());
    }
    let mut overflow = quote; overflow.input_bytes = u64::MAX;
    assert!(overflow.with_input_encoding(&encoding).is_err());
    overflow = quote; overflow.output_reservation_bytes = u64::MAX - quote.input_bytes;
    assert!(overflow.with_input_encoding(&encoding).is_err());
}
