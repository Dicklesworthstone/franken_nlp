//! Pure admission regressions; no tokenizer, model or runtime is constructed.
use super::*;
use crate::{native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4Work},
    tasks::{mapreduce::ChunkLimits, classify::ClassificationAggregationPolicy}};

fn fixture() -> (Int4ClassificationMapMemoryLimits, TaskBudget, ClassificationLimits, Int4ClassificationMapLimits) {
    let memory = Int4ClassificationMapMemoryLimits { context_tokens: 512,
        preparation_reserve_bytes: MIN_CLASSIFICATION_MAP_PREPARATION_BYTES, allocator_reserve_bytes: 65_536 };
    let task = TaskBudget { max_input_tokens: 256, max_output_tokens: 16, max_output_bytes: 65_536,
        max_grammar_states: 4096, max_kv_bytes: Int4MemoryRequirement::for_context(512, 32).unwrap().kv_bytes };
    let planning = ClassificationLimits { max_input_bytes: 8192, max_labels: 16, ..Default::default() };
    let mapping = Int4ClassificationMapLimits { chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 128,
        max_chunk_tokens: 256, context_tokens: 512, reserved_tokens: 16, max_chunks: 8, max_tokenizer_calls: 1024 },
        max_model_work: Int4Work::for_sequence(32, 0, 1024, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
        max_retained_result_bytes: 1024 * 1024, max_result_bytes: 1024 * 1024,
        aggregation: ClassificationAggregationPolicy::default() };
    (memory, task, planning, mapping)
}
#[test]
fn sequential_labels_and_chunks_share_exactly_one_native_cache() {
    let (memory, task, planning, mapping) = fixture();
    for width in [1, 3, 32, 128, 10752] {
        let quote = memory.requirements(width, task, planning, mapping).unwrap();
        let row = Int4MemoryRequirement::for_context(memory.context_tokens, width).unwrap();
        assert_eq!(quote.native, row);
        assert_eq!(quote.workspace_bytes, row.rope_bytes + row.scratch_payload_bound + memory.allocator_reserve_bytes);
        assert!(quote.input_bytes > memory.preparation_reserve_bytes);
        assert_eq!(quote.max_output_bytes, mapping.max_result_bytes);
        assert_eq!(quote.output_reservation_bytes, 8 * mapping.max_result_bytes + 4096);
    }
}
#[test]
fn source_all_plans_and_retained_results_have_distinct_checked_owners() {
    let (memory, task, planning, mut mapping) = fixture();
    mapping.max_retained_result_bytes = task.max_output_bytes;
    let first = memory.requirements(32, task, planning, mapping).unwrap();
    mapping.max_retained_result_bytes *= 2;
    let results = memory.requirements(32, task, planning, mapping).unwrap();
    assert_eq!(results.input_bytes - first.input_bytes, 8 * task.max_output_bytes);
    assert_eq!(results.output_reservation_bytes, first.output_reservation_bytes);
    mapping.chunks.max_input_bytes *= 2;
    let source = memory.requirements(32, task, planning, mapping).unwrap();
    assert_eq!(source.input_bytes - results.input_bytes, 4 * 4096);
    mapping.chunks.max_chunks *= 2;
    let plans = memory.requirements(32, task, planning, mapping).unwrap();
    assert!(plans.input_bytes > source.input_bytes); assert_eq!(plans.native, source.native);
    let more = ClassificationLimits { max_labels: 32, ..planning };
    assert!(memory.requirements(32, task, more, mapping).unwrap().input_bytes > plans.input_bytes);
}
#[test]
fn undersized_kv_incoherent_source_limits_and_overflow_fail_pure_admission() {
    for axis in 0..10 {
        let (mut memory, mut task, mut planning, mut mapping) = fixture();
        match axis { 0 => task.max_kv_bytes -= 1, 1 => memory.preparation_reserve_bytes -= 1,
            2 => memory.allocator_reserve_bytes = 0, 3 => memory.preparation_reserve_bytes = u64::MAX,
            4 => mapping.chunks.max_input_bytes = planning.max_input_bytes + 1,
            5 => mapping.max_model_work.projections.dot_products = 0,
            6 => planning.scoring.max_nodes = usize::MAX,
            7 => memory.context_tokens = mapping.chunks.context_tokens - 1,
            8 => planning.max_context_tokens = memory.context_tokens - 1,
            _ => mapping.aggregation.minimum_classified_coverage_ppm = 1_000_001 }
        assert!(memory.requirements(32, task, planning, mapping).is_err(), "axis {axis}");
    }
}

#[test]
fn cohorts_charge_independent_kv_shared_rope_and_separate_live_window_storage() {
    let (memory, mut task, mut planning, mapping) = fixture(); planning.max_labels = 4;
    task.max_kv_bytes *= planning.max_labels as u64;
    for width in [1, 3, 32, 128, 10752] {
        let serial = memory.requirements(width, task, planning, mapping).unwrap();
        let grouped = memory.cohort_requirements(width, task, planning, mapping, 4, 2).unwrap();
        assert_eq!(grouped.memory.native.kv_bytes, serial.native.kv_bytes * 8);
        assert_eq!(grouped.memory.native.rope_bytes, serial.native.rope_bytes);
        assert!(grouped.memory.native.scratch_payload_bound > serial.native.scratch_payload_bound);
        assert_eq!(grouped.memory.input_bytes, serial.input_bytes + grouped.task_workspace_bytes
            + 8 * (grouped.max_cohort_output_bytes - task.max_output_bytes));
        assert_eq!(grouped.memory.workspace_bytes, grouped.memory.native.rope_bytes + grouped.memory.native.scratch_payload_bound
            + memory.allocator_reserve_bytes);
        assert_eq!(grouped.max_cohort_output_bytes, task.max_output_bytes * 2 + COHORT_WRAPPER_BYTES);
        assert_eq!(grouped.memory.max_output_bytes, mapping.max_result_bytes + COHORT_WRAPPER_BYTES);
        assert_eq!(grouped.memory.output_reservation_bytes, serial.output_reservation_bytes + 8 * COHORT_WRAPPER_BYTES);
    }
}

#[test]
fn unknown_label_mode_prices_the_largest_valid_declared_chunk_head_product() {
    let (memory, mut task, mut planning, mut mapping) = fixture();
    planning.max_labels = 128; mapping.chunks.max_chunks = 128;
    task.max_kv_bytes *= planning.max_labels as u64;
    let row = Int4MemoryRequirement::for_context(memory.context_tokens, 32).unwrap();
    let three = memory.cohort_requirements(32, task, planning, mapping, 21, 3).unwrap();
    assert_eq!(three.memory.native.kv_bytes, row.kv_bytes * 63);
    let sixty_four = memory.cohort_requirements(32, task, planning, mapping, 1, 64).unwrap();
    assert_eq!(sixty_four.memory.native.kv_bytes, row.kv_bytes * 64);
    // Callers quote the largest valid unknown shape explicitly. This never
    // authorizes more actual heads or silently shrinks the requested window.
    assert!(memory.cohort_requirements(32, task, planning, mapping, 22, 3).is_err());
    for count in [0, 65, usize::MAX] {
        assert!(memory.cohort_requirements(32, task, planning, mapping, 1, count).is_err());
    }
    mapping.chunks.max_chunks = 2;
    assert!(memory.cohort_requirements(32, task, planning, mapping, 1, 3).is_err());
}

#[test]
fn window_document_and_retained_result_caps_have_distinct_owners() {
    let (memory, mut task, planning, mut mapping) = fixture();
    let before = memory.cohort_requirements(32, task, planning, mapping, 1, 2).unwrap();
    mapping.max_result_bytes *= 2;
    let document = memory.cohort_requirements(32, task, planning, mapping, 1, 2).unwrap();
    assert_eq!(document.memory.input_bytes, before.memory.input_bytes);
    assert_eq!(document.task_workspace_bytes, before.task_workspace_bytes);
    assert_eq!(document.max_cohort_output_bytes, before.max_cohort_output_bytes);
    assert!(document.memory.output_reservation_bytes > before.memory.output_reservation_bytes);
    mapping.max_retained_result_bytes = task.max_output_bytes;
    let frontier = memory.cohort_requirements(32, task, planning, mapping, 1, 2).unwrap();
    assert!(frontier.memory.input_bytes < document.memory.input_bytes);
    assert_eq!(frontier.memory.output_reservation_bytes, document.memory.output_reservation_bytes);
    task.max_output_bytes = super::super::MAX_BYTES;
    assert!(memory.cohort_requirements(32, task, planning, mapping, 1, 2).is_err());
}

#[test]
fn cohort_workspace_overflow_and_invalid_declarations_fail_without_allocation() {
    for axis in 0..6 {
        let (mut memory, mut task, mut planning, mut mapping) = fixture();
        match axis { 0 => mapping.max_model_work.forward_positions = u64::MAX,
            1 => task.max_output_bytes = u64::MAX, 2 => memory.preparation_reserve_bytes = u64::MAX,
            3 => planning.max_labels = 0, 4 => mapping.chunks.max_chunks = 1,
            _ => task.max_kv_bytes = 1 }
        assert!(memory.cohort_requirements(32, task, planning, mapping, 1, 2).is_err(), "axis {axis}");
    }
}

#[test]
fn cohort_quote_prices_all_chunk_heads_and_keeps_the_serial_quote_unchanged() {
    let (memory, mut task, planning, mapping) = fixture();
    for heads in [1, 3, 8] {
        let row = Int4MemoryRequirement::for_context(memory.context_tokens, 32).unwrap();
        task.max_kv_bytes = row.kv_bytes * heads as u64;
        let serial = memory.requirements(32, task, planning, mapping).unwrap();
        for chunks in [1, 2, 8] {
            let quote = memory.cohort_requirements(32, task, planning, mapping, heads, chunks).unwrap();
            let contexts = vec![memory.context_tokens; heads * chunks];
            assert_eq!(quote.memory.native, Int4MemoryRequirement::for_cohort(&contexts, 32).unwrap());
            assert_eq!(quote.memory.native.kv_bytes, row.kv_bytes * (heads * chunks) as u64);
            assert_eq!(quote.memory.workspace_bytes, quote.memory.native.rope_bytes
                + quote.memory.native.scratch_payload_bound + memory.allocator_reserve_bytes);
            assert!(quote.memory.input_bytes >= serial.input_bytes + quote.task_workspace_bytes);
            assert!(quote.max_cohort_output_bytes > chunks as u64 * task.max_output_bytes);
            assert_eq!(quote.memory.max_output_bytes, mapping.max_result_bytes + COHORT_WRAPPER_BYTES);
            assert_eq!(quote.memory.output_reservation_bytes, 8 * quote.memory.max_output_bytes + 4096);
            assert_eq!(memory.requirements(32, task, planning, mapping).unwrap(), serial);
        }
    }
}

#[test]
fn cohort_quote_refuses_per_document_kv_and_oversized_windows_without_clamping() {
    let (memory, mut task, planning, mapping) = fixture();
    let one = task.max_kv_bytes;
    task.max_kv_bytes = one * 3;
    assert!(memory.cohort_requirements(32, task, planning, mapping, 3, 2).is_ok());
    task.max_kv_bytes -= 1;
    assert!(matches!(memory.cohort_requirements(32, task, planning, mapping, 3, 2),
        Err(Int4ClassificationMapError::Admission)));
    task.max_kv_bytes = u64::MAX;
    for (heads, chunks) in [(0, 1), (1, 0), (planning.max_labels + 1, 1), (16, 5),
        (1, mapping.chunks.max_chunks + 1), (usize::MAX, 1), (1, usize::MAX)] {
        assert!(matches!(memory.cohort_requirements(32, task, planning, mapping, heads, chunks),
            Err(Int4ClassificationMapError::Limits)), "heads={heads} chunks={chunks}");
    }
}

#[test]
fn cohort_workspace_and_current_result_frontier_have_checked_finite_bounds() {
    let (memory, task, planning, mut mapping) = fixture();
    let one = memory.cohort_requirements(32, task, planning, mapping, 1, 1).unwrap();
    let two = memory.cohort_requirements(32, task, planning, mapping, 1, 2).unwrap();
    assert!(two.task_workspace_bytes > one.task_workspace_bytes);
    assert!(two.memory.input_bytes > one.memory.input_bytes);
    assert_eq!(two.memory.output_reservation_bytes, one.memory.output_reservation_bytes);
    let mut excessive = task; excessive.max_output_bytes = 64 * 1024 * 1024;
    assert!(memory.cohort_requirements(32, excessive, planning, mapping, 1, 1).is_err());
    mapping.max_model_work.forward_positions = u64::MAX;
    assert!(memory.cohort_requirements(32, task, planning, mapping, 1, 1).is_err());
}
