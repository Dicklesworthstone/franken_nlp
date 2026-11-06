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
