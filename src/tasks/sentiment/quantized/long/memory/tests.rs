//! Pure ownership/admission regressions; no tokenizer, model or native run.
use super::*;
use crate::{
    native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int8::Int8Work},
    tasks::{mapreduce::ChunkLimits, sentiment::SentimentAggregationPolicy},
    tokenizer::untrusted::SafeBpeLimits,
};

fn fixture() -> (
    Int8SentimentMapMemoryLimits,
    TaskBudget,
    SentimentLimits,
    Int8SentimentMapLimits,
) {
    let context_tokens = 256;
    let row = Int8MemoryRequirement::for_context(context_tokens).unwrap();
    let task = TaskBudget {
        max_input_tokens: 128,
        max_output_tokens: 16,
        max_output_bytes: 65_536,
        max_grammar_states: 4096,
        max_kv_bytes: row.kv_bytes * 4,
    };
    let memory = Int8SentimentMapMemoryLimits {
        context_tokens,
        preparation_reserve_bytes: MIN_SENTIMENT_MAP_PREPARATION_BYTES,
        allocator_reserve_bytes: 65_536,
        max_axes: 4,
    };
    let mapping = Int8SentimentMapLimits {
        chunks: ChunkLimits {
            max_input_bytes: 4096,
            max_chunk_bytes: 512,
            max_chunk_tokens: 128,
            context_tokens,
            reserved_tokens: 16,
            max_chunks: 8,
            max_tokenizer_calls: 1024,
        },
        max_model_work: Int8Work::for_sequence(0, 1024, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
        max_retained_result_bytes: 8 * 1024 * 1024,
        max_result_bytes: 1024 * 1024,
        aggregation: SentimentAggregationPolicy::default(),
    };
    (memory, task, SentimentLimits::default(), mapping)
}

#[test]
fn every_axis_and_chunk_reuses_one_physical_cache() {
    let (mut memory, task, planning, mut mapping) = fixture();
    let row = Int8MemoryRequirement::for_context(memory.context_tokens).unwrap();
    let mut previous_input = 0;
    for axes in 1..=4 {
        memory.max_axes = axes;
        let required = memory.requirements(task, planning, mapping).unwrap();
        assert_eq!(required.native, row);
        assert_eq!(
            required.workspace_bytes,
            row.rope_bytes + row.scratch_payload_bound + memory.allocator_reserve_bytes
        );
        assert!(required.input_bytes > previous_input);
        previous_input = required.input_bytes;
    }
    mapping.chunks.max_chunks *= 2;
    let more_chunks = memory.requirements(task, planning, mapping).unwrap();
    assert_eq!(more_chunks.native, row);
    assert!(more_chunks.input_bytes > previous_input);
}

#[test]
fn per_chunk_task_authority_still_covers_all_declared_axes() {
    let (mut memory, mut task, planning, mapping) = fixture();
    for axes in 1..=4 {
        memory.max_axes = axes;
        let row = Int8MemoryRequirement::for_context(memory.context_tokens).unwrap();
        task.max_kv_bytes = row.kv_bytes * axes as u64;
        assert!(memory.requirements(task, planning, mapping).is_ok());
        task.max_kv_bytes -= 1;
        assert!(matches!(
            memory.requirements(task, planning, mapping),
            Err(Int8SentimentMapError::Admission)
        ));
    }
}

#[test]
fn complete_frontier_and_next_chunk_are_separate_from_final_publication() {
    let (memory, task, planning, mut mapping) = fixture();
    mapping.max_retained_result_bytes = task.max_output_bytes;
    let one = memory.requirements(task, planning, mapping).unwrap();
    mapping.max_retained_result_bytes *= 2;
    let two = memory.requirements(task, planning, mapping).unwrap();
    assert_eq!(two.input_bytes - one.input_bytes, task.max_output_bytes * 8);
    assert_eq!(two.output_reservation_bytes, one.output_reservation_bytes);
    // A smaller accepted-result frontier cannot discard the next full chunk
    // that exists before the cumulative-output cap can refuse it.
    mapping.max_retained_result_bytes = 1;
    let tiny_frontier = memory.requirements(task, planning, mapping).unwrap();
    assert!(tiny_frontier.input_bytes >= task.max_output_bytes * 8);
    assert_eq!(tiny_frontier.task_workspace_bytes, one.task_workspace_bytes);
    mapping.max_result_bytes *= 2;
    let larger_output = memory.requirements(task, planning, mapping).unwrap();
    assert_eq!(larger_output.input_bytes, tiny_frontier.input_bytes);
    assert_eq!(
        larger_output.max_output_bytes,
        mapping.max_result_bytes + WRAPPER_BYTES
    );
    assert_eq!(
        larger_output.output_reservation_bytes,
        4 * larger_output.max_output_bytes + WRAPPER_BYTES
    );
}

#[test]
fn whole_source_and_one_rejected_plan_remain_admitted() {
    let (memory, task, planning, mut mapping) = fixture();
    let before = memory.requirements(task, planning, mapping).unwrap();
    mapping.chunks.max_input_bytes *= 2;
    let source = memory.requirements(task, planning, mapping).unwrap();
    assert_eq!(source.input_bytes - before.input_bytes, 4 * 4096);
    assert_eq!(source.native, before.native);
    mapping.chunks.max_chunks *= 2;
    let plans = memory.requirements(task, planning, mapping).unwrap();
    assert!(plans.input_bytes > source.input_bytes);
    assert_eq!(plans.task_workspace_bytes, source.task_workspace_bytes);
    let accepted_plus_next = mapping.max_model_work.forward_positions
        + u64::from(task.max_input_tokens) * memory.max_axes as u64;
    assert!(plans.input_bytes >= accepted_plus_next * PLAN_BYTES_PER_TOKEN);
}

#[test]
fn invalid_geometry_limits_and_each_work_counter_refuse_pure_admission() {
    for axis in 0..16 {
        let (mut memory, mut task, mut planning, mut mapping) = fixture();
        match axis {
            0 => memory.max_axes = 0,
            1 => memory.max_axes = 5,
            2 => memory.preparation_reserve_bytes = MIN_SENTIMENT_MAP_PREPARATION_BYTES - 1,
            3 => memory.allocator_reserve_bytes = 0,
            4 => memory.context_tokens = 0,
            5 => memory.context_tokens = mapping.chunks.context_tokens - 1,
            6 => mapping.chunks.max_input_bytes = 0,
            7 => mapping.max_model_work.forward_positions = 0,
            8 => mapping.max_model_work.projected_logits = 0,
            9 => mapping.max_model_work.attention_pairs = 0,
            10 => mapping.max_model_work.projections.dot_products = 0,
            11 => mapping.max_model_work.projections.multiply_accumulates = 0,
            12 => planning.per_axis.max_nodes = 0,
            13 => task.max_input_tokens = 0,
            14 => mapping.max_retained_result_bytes = 0,
            _ => mapping.max_result_bytes = 0,
        }
        assert!(
            memory.requirements(task, planning, mapping).is_err(),
            "axis {axis}"
        );
    }
    let (memory, mut task, mut planning, mapping) = fixture();
    task.max_output_bytes = MAX_CHUNK_OUTPUT_BYTES + 1;
    planning.max_output_bytes = MAX_CHUNK_OUTPUT_BYTES + 1;
    assert!(memory.requirements(task, planning, mapping).is_err());
    for context in [0, usize::MAX] {
        assert!(
            Int8SentimentMapMemoryLimits {
                context_tokens: context,
                ..memory
            }
            .requirements(task, planning, mapping)
            .is_err()
        );
    }
}

#[test]
fn overflow_cannot_wrap_an_input_scratch_or_aggregate_reservation() {
    for axis in 0..3 {
        let (mut memory, task, planning, mut mapping) = fixture();
        match axis {
            0 => memory.preparation_reserve_bytes = u64::MAX,
            1 => memory.allocator_reserve_bytes = u64::MAX,
            _ => mapping.max_model_work.forward_positions = u64::MAX,
        }
        assert!(
            memory.requirements(task, planning, mapping).is_err(),
            "axis {axis}"
        );
    }
    let (mut memory, task, planning, mapping) = fixture();
    let quote = memory.requirements(task, planning, mapping).unwrap();
    // Every subtotal fits by itself, but their process-wide sum cannot.
    memory.preparation_reserve_bytes =
        u64::MAX - (quote.input_bytes - memory.preparation_reserve_bytes);
    assert!(memory.requirements(task, planning, mapping).is_err());
}

#[test]
fn compact_encoding_adds_one_workspace_independent_of_axis_and_chunk_counts() {
    let policy = SafeBpeLimits {
        max_input_bytes: 8192,
        max_output_tokens: 4096,
        max_pair_checks: 100_000,
        max_pair_bytes: 1_000_000,
    };
    let encoding = UntrustedEncoding::SafeBpeV1(policy);
    let encoder_bytes = policy.requirements().unwrap().workspace_bytes;
    let (mut memory, task, planning, mut mapping) = fixture();
    for axes in 1..=4 {
        memory.max_axes = axes;
        for chunks in [1, 8, 64] {
            mapping.chunks.max_chunks = chunks;
            let byte = memory.requirements(task, planning, mapping).unwrap();
            assert_eq!(
                byte.with_input_encoding(&UntrustedEncoding::ByteFallback)
                    .unwrap(),
                byte
            );
            let compact = byte.with_input_encoding(&encoding).unwrap();
            assert_eq!(compact.input_bytes, byte.input_bytes + encoder_bytes);
            assert_eq!(compact.native, byte.native);
            assert_eq!(compact.task_workspace_bytes, byte.task_workspace_bytes);
            assert_eq!(compact.workspace_bytes, byte.workspace_bytes);
            assert_eq!(
                compact.output_reservation_bytes,
                byte.output_reservation_bytes
            );
        }
    }
    let quote = memory.requirements(task, planning, mapping).unwrap();
    for axis in 0..4 {
        let mut invalid = policy;
        match axis {
            0 => invalid.max_input_bytes = 0,
            1 => invalid.max_output_tokens = 0,
            2 => invalid.max_pair_checks = 0,
            _ => invalid.max_pair_bytes = 0,
        }
        assert!(
            quote
                .with_input_encoding(&UntrustedEncoding::SafeBpeV1(invalid))
                .is_err()
        );
    }
    let mut overflow = quote;
    overflow.input_bytes = u64::MAX;
    assert!(overflow.with_input_encoding(&encoding).is_err());
    overflow = quote;
    overflow.output_reservation_bytes = u64::MAX - quote.input_bytes;
    assert!(overflow.with_input_encoding(&encoding).is_err());
}
