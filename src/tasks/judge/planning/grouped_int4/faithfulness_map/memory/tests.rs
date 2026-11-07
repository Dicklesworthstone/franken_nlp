//! Pure admission quotes: no tokenizer, weights or engine are constructed.
use super::*;
use crate::{native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4Work},
    tasks::{mapreduce::ChunkLimits, judge::FaithfulnessAggregationPolicy}};

fn fixture() -> (Int4FaithfulnessMapMemoryLimits, TaskBudget, JudgeLimits, Int4FaithfulnessMapLimits) {
    let memory = Int4FaithfulnessMapMemoryLimits { context_tokens: 512,
        preparation_reserve_bytes: MIN_FAITHFULNESS_MAP_PREPARATION_BYTES, allocator_reserve_bytes: 65536 };
    let task = TaskBudget { max_input_tokens: 512, max_output_tokens: 16, max_output_bytes: 65536,
        max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(512, 32).unwrap().kv_bytes * 32 };
    let mapping = Int4FaithfulnessMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 128,
        max_chunk_tokens: 256, context_tokens: 512, reserved_tokens: 16, max_chunks: 64, max_tokenizer_calls: 1024 },
        max_model_work: Int4Work::for_sequence(32, 0, 1024, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
        max_retained_result_bytes: 1024 * 1024, max_result_bytes: 1024 * 1024,
        aggregation: FaithfulnessAggregationPolicy::default() };
    (memory, task, JudgeLimits::default(), mapping)
}
#[test]
fn serial_evidence_and_chunks_share_one_cache_and_keep_complete_output_owned() {
    let (memory, task, planning, mapping) = fixture();
    for width in [1, 3, 32, 128, 10752] {
        let quote = memory.requirements(width, task, planning, mapping).unwrap();
        let native = Int4MemoryRequirement::for_context(memory.context_tokens, width).unwrap();
        assert_eq!(quote.native, native);
        assert_eq!(quote.workspace_bytes, native.rope_bytes + native.scratch_payload_bound + memory.allocator_reserve_bytes);
        assert!(quote.input_bytes > memory.preparation_reserve_bytes);
        assert_eq!(quote.output_reservation_bytes, 8 * mapping.max_result_bytes + 4096);
    }
}
#[test]
fn all_retained_plans_source_claim_and_result_frontiers_have_checked_owners() {
    let (memory, task, planning, mut mapping) = fixture();
    mapping.max_retained_result_bytes = task.max_output_bytes;
    let first = memory.requirements(32, task, planning, mapping).unwrap();
    mapping.max_retained_result_bytes *= 2;
    let retained = memory.requirements(32, task, planning, mapping).unwrap();
    assert_eq!(retained.input_bytes - first.input_bytes, 8 * task.max_output_bytes);
    mapping.chunks.max_input_bytes *= 2;
    let source = memory.requirements(32, task, planning, mapping).unwrap();
    assert!(source.input_bytes > retained.input_bytes); assert_eq!(source.native, retained.native);
    mapping.chunks.max_chunks *= 2;
    assert!(memory.requirements(32, task, planning, mapping).unwrap().input_bytes > source.input_bytes);
}
#[test]
fn invalid_caps_and_overflow_fail_without_io_or_allocation() {
    for axis in 0..8 {
        let (mut memory, mut task, mut planning, mut mapping) = fixture();
        match axis { 0 => task.max_kv_bytes = 1, 1 => memory.preparation_reserve_bytes -= 1,
            2 => memory.allocator_reserve_bytes = 0, 3 => memory.preparation_reserve_bytes = u64::MAX,
            4 => planning.max_total_prompt_tokens = 0, 5 => mapping.max_model_work.projected_logits = 0,
            6 => memory.context_tokens = mapping.chunks.context_tokens - 1,
            _ => mapping.aggregation.minimum_classified_coverage_ppm = 1_000_001 }
        assert!(memory.requirements(32, task, planning, mapping).is_err(), "axis {axis}");
    }
}
#[test]
fn ragged_cohort_quote_caps_allocation_at_64_without_shrinking_requested_schedule() {
    let (memory, task, planning, mapping) = fixture();
    let serial = memory.requirements(32, task, planning, mapping).unwrap();
    for chunks in [1, 2, 3, 16, 64] {
        let quote = memory.cohort_requirements(32, task, planning, mapping, 32, chunks).unwrap();
        let heads = (32 * chunks).min(64);
        assert_eq!(quote.memory.native, Int4MemoryRequirement::for_cohort(&vec![512; heads], 32).unwrap());
        assert!(quote.memory.input_bytes >= serial.input_bytes + quote.task_workspace_bytes);
        assert_eq!(quote.max_cohort_output_bytes, task.max_output_bytes * chunks as u64 + COHORT_WRAPPER_BYTES);
        assert_eq!(quote.memory.max_output_bytes, mapping.max_result_bytes + COHORT_WRAPPER_BYTES);
        assert_eq!(memory.requirements(32, task, planning, mapping).unwrap(), serial);
    }
    let single_heads = memory.cohort_requirements(32, task, planning, mapping, 1, 16).unwrap();
    assert_eq!(single_heads.memory.native.kv_bytes, serial.native.kv_bytes * 16);
}
#[test]
fn per_chunk_task_kv_and_complete_window_output_are_independent_hard_caps() {
    let (memory, mut task, planning, mapping) = fixture();
    task.max_kv_bytes -= 1;
    assert!(matches!(memory.cohort_requirements(32, task, planning, mapping, 32, 2),
        Err(Int4FaithfulnessMapError::Admission)));
    task.max_kv_bytes += 1;
    for (heads, chunks) in [(0, 1), (33, 1), (1, 0), (1, 65), (usize::MAX, 1), (1, usize::MAX)] {
        assert!(memory.cohort_requirements(32, task, planning, mapping, heads, chunks).is_err());
    }
    task.max_output_bytes = 1_048_576;
    assert!(memory.cohort_requirements(32, task, planning, mapping, 32, 64).is_err());
}

#[test]
fn explicit_head_cap_reduces_plan_ownership_without_shrinking_complete_results() {
    let (memory, task, planning, mapping) = fixture();
    let legacy = memory.requirements(32, task, planning, mapping).unwrap();
    assert_eq!(legacy, memory.requirements_with_max_heads(32, task, planning, mapping, 32).unwrap());
    let mut previous = None;
    for heads in [1, 3, 8, 16, 32] {
        let current = memory.requirements_with_max_heads(32, task, planning, mapping, heads).unwrap();
        assert_eq!(current.native, legacy.native, "serial execution keeps one physical row");
        assert_eq!(current.workspace_bytes, legacy.workspace_bytes);
        assert_eq!(current.max_output_bytes, legacy.max_output_bytes);
        assert_eq!(current.output_reservation_bytes, legacy.output_reservation_bytes);
        assert!(current.input_bytes > memory.preparation_reserve_bytes);
        if let Some(previous) = previous { assert!(current.input_bytes > previous); }
        previous = Some(current.input_bytes);
    }
    for heads in [0, 33, usize::MAX] {
        assert!(memory.requirements_with_max_heads(32, task, planning, mapping, heads).is_err());
    }
}

#[test]
fn small_declared_evidence_sets_enable_cohorts_with_their_own_task_kv_ceiling() {
    let (memory, mut task, planning, mapping) = fixture();
    let row = Int4MemoryRequirement::for_context(memory.context_tokens, 32).unwrap().kv_bytes;
    task.max_kv_bytes = row;
    assert!(memory.cohort_requirements(32, task, planning, mapping, 32, 3).is_err());
    let one = memory.cohort_requirements(32, task, planning, mapping, 1, 3).unwrap();
    assert_eq!(one.memory.native.kv_bytes, 3 * row);
    let serial = memory.requirements_with_max_heads(32, task, planning, mapping, 1).unwrap();
    assert!(one.memory.input_bytes >= serial.input_bytes + one.task_workspace_bytes);
    task.max_kv_bytes = 3 * row;
    let three = memory.cohort_requirements(32, task, planning, mapping, 3, 3).unwrap();
    assert_eq!(three.memory.native.kv_bytes, 9 * row);
    assert!(three.memory.input_bytes > one.memory.input_bytes);
    assert_eq!(three.memory.max_output_bytes, one.memory.max_output_bytes);
    assert_eq!(three.memory.output_reservation_bytes, one.memory.output_reservation_bytes);
    task.max_kv_bytes -= 1;
    assert!(memory.cohort_requirements(32, task, planning, mapping, 3, 3).is_err());
    // The maximum admissible physical window remains64, not a silently
    // shortened chunk schedule. Actual execution still checks every window.
    task.max_kv_bytes += 1;
    let many = memory.cohort_requirements(32, task, planning, mapping, 3, 64).unwrap();
    assert_eq!(many.memory.native.kv_bytes, 64 * row);
}
