//! Pure durable-host contracts, without fake model execution or quality claims.
use super::*;
use std::time::Duration;
use crate::{jobs::population::PopulationReadLimits,
    tasks::{mapreduce::ChunkLimits, judge::{FaithfulnessAggregationPolicy,
        grouped_int4::faithfulness_map::MAX_FAITHFULNESS_MAP_HEADS}}};

fn fixture() -> (TaskBudget, JudgeLimits, Int4FaithfulnessMapLimits, JobLimits, JobHostLimits) {
    let task = TaskBudget { max_input_tokens: 128, max_output_tokens: 16,
        max_output_bytes: 65_536, max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(256, 32).unwrap().kv_bytes * 32 };
    let mapping = Int4FaithfulnessMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 512,
            max_chunk_tokens: 128, context_tokens: 256, reserved_tokens: 16,
            max_chunks: 64, max_tokenizer_calls: 1024 },
        max_model_work: Int4Work::for_sequence(32, 0, 1024, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
        max_retained_result_bytes: 8 << 20, max_result_bytes: 1 << 20,
        aggregation: FaithfulnessAggregationPolicy::default(),
    };
    let job = JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65_536, max_result_bytes: (1 << 20) + 4096,
        max_spool_bytes: 8 << 20, max_materialized_bytes: 8 << 20,
        max_journal_bytes: 8 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(mapping.max_model_work, 0) };
    let host = JobHostLimits {
        native: NativeLimits { context_tokens: 256, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 10000,
                cleanup_reserve_bytes: 65_536 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
        preparation_reserve_bytes: 128 << 20, io_reserve_bytes: 131_072,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
    };
    (task, JudgeLimits::default(), mapping, job, host)
}

#[test]
fn population_original_claims_and_replanned_document_frontier_coexist() {
    let (task, planning, mapping, job, host) = fixture();
    for chunks in [None, Some(1), Some(2), Some(3), Some(64)] {
        let needed = requirements(32, task, planning, mapping, job, host, 32, chunks).unwrap();
        let map = match chunks {
            None => map_memory(host).requirements(32, task, planning, mapping).unwrap(),
            Some(chunks) => map_memory(host).cohort_requirements(32, task, planning, mapping,
                MAX_FAITHFULNESS_MAP_HEADS, chunks).unwrap().memory,
        };
        assert_eq!(needed.native, map.native);
        assert_eq!(needed.workspace_bytes, map.workspace_bytes);
        assert_eq!(needed.output_reservation_bytes, map.output_reservation_bytes);
        assert_eq!(needed.max_output_bytes, map.max_output_bytes);
        assert_eq!(needed.input_bytes, host.required_buffer_bytes(job).unwrap()
            + map.input_bytes - host.preparation_reserve_bytes
            + mapping.chunks.max_chunks as u64 * FAITHFULNESS_DOCUMENT_CHUNK_METADATA_BYTES);
        let mut larger = job; larger.max_snapshot_bytes += 4096;
        let larger = requirements(32, task, planning, mapping, larger, host, 32, chunks).unwrap();
        assert_eq!(larger.input_bytes - needed.input_bytes, 8192);
        assert_eq!(larger.native, needed.native);
    }
}

#[test]
fn every_native_axis_fits_the_persisted_nonrenewable_work_allowance() {
    let (task, planning, mapping, job, host) = fixture();
    for axis in 0..6 {
        let mut limited = job;
        match axis { 0 => limited.max_work.model.forward_positions -= 1,
            1 => limited.max_work.model.projected_logits -= 1,
            2 => limited.max_work.model.attention_pairs -= 1,
            3 => limited.max_work.model.projections.dot_products -= 1,
            4 => limited.max_work.model.projections.multiply_accumulates -= 1,
            _ => limited.max_work.mask_node_visits = 1 }
        assert!(requirements(32, task, planning, mapping, limited, host, 32, None).is_err(), "axis {axis}");
    }
    let quote = requirements(32, task, planning, mapping, job, host, 32, None).unwrap();
    let mut more_attempts = job; more_attempts.max_attempts = 100;
    assert_eq!(quote, requirements(32, task, planning, mapping, more_attempts, host, 32, None).unwrap());
}

#[test]
fn ragged_cohorts_price_at_most_sixty_four_active_heads_without_changing_schedule() {
    let (task, planning, mapping, job, host) = fixture();
    let row = Int4MemoryRequirement::for_context(256, 32).unwrap();
    for chunks in [3, 16, 64] {
        let needed = requirements(32, task, planning, mapping, job, host, 32, Some(chunks)).unwrap();
        assert_eq!(needed.native.kv_bytes, row.kv_bytes * 64);
    }
    for chunks in [0, 65, usize::MAX] {
        assert!(requirements(32, task, planning, mapping, job, host, 32, Some(chunks)).is_err());
    }
    let mut one_row = task; one_row.max_kv_bytes = row.kv_bytes;
    requirements(32, one_row, planning, mapping, job, host, 32, None).unwrap();
    assert!(requirements(32, one_row, planning, mapping, job, host, 32, Some(1)).is_err());
}

#[test]
fn declared_head_cap_narrows_job_frontier_and_native_peak_without_pruning_outputs() {
    let (task, planning, mapping, job, host) = fixture();
    let row = Int4MemoryRequirement::for_context(host.native.context_tokens, 32).unwrap().kv_bytes;
    for chunks in [None, Some(1), Some(3), Some(64)] {
        let legacy = requirements(32, task, planning, mapping, job, host, 32, chunks).unwrap();
        let mut bounded = task; bounded.max_kv_bytes = row;
        let capped = requirements(32, bounded, planning, mapping, job, host, 1, chunks).unwrap();
        assert_eq!(capped.native.kv_bytes, row * chunks.unwrap_or(1) as u64);
        assert!(capped.input_bytes < legacy.input_bytes);
        assert_eq!(capped.output_reservation_bytes, legacy.output_reservation_bytes);
        assert_eq!(capped.max_output_bytes, legacy.max_output_bytes);
        assert!(capped.input_bytes > host.required_buffer_bytes(job).unwrap());
    }
    for cap in [0, 33, usize::MAX] {
        assert!(requirements(32, task, planning, mapping, job, host, cap, None).is_err());
    }
}

#[test]
fn stored_result_and_original_input_caps_match_the_authenticated_recipe() {
    let (task, planning, mapping, mut job, host) = fixture();
    job.max_result_bytes = mapping.max_result_bytes as usize;
    requirements(32, task, planning, mapping, job, host, 32, None).unwrap();
    assert!(requirements(32, task, planning, mapping, job, host, 32, Some(2)).is_err());
    job.max_result_bytes += 4096;
    let quote = requirements(32, task, planning, mapping, job, host, 32, Some(2)).unwrap();
    assert_eq!(quote.max_output_bytes, job.max_result_bytes as u64);
    assert!(quote.output_reservation_bytes > quote.max_output_bytes);
    job.max_result_bytes -= 1;
    assert!(requirements(32, task, planning, mapping, job, host, 32, Some(2)).is_err());
    let mut job = fixture().3;
    job.max_input_bytes_per_item = mapping.chunks.max_input_bytes - 1;
    assert!(requirements(32, task, planning, mapping, job, host, 32, None).is_err());
}

#[test]
fn invalid_host_capacity_and_overflow_refuse_before_private_population_io() {
    let (task, planning, mapping, job, host) = fixture();
    for field in 0..11 {
        let mut host = host;
        match field { 0 => host.native.context_tokens = 255, 1 => host.native.context_tokens = 257,
            2 => host.native.allocator_reserve_bytes = 0, 3 => host.preparation_reserve_bytes = (128 << 20) - 1,
            4 => host.preparation_reserve_bytes = u64::MAX, 5 => host.io_reserve_bytes = u64::MAX,
            6 => host.journal_reserve_bytes = 0, 7 => host.serialization_reserve_bytes = 0,
            8 => host.native.run.max_checkpoints = 1, 9 => host.transport.max_stream_bytes = 0,
            _ => host.native.allocator_reserve_bytes = u64::MAX }
        assert!(requirements(32, task, planning, mapping, job, host, 32, None).is_err(), "field {field}");
    }
    for width in [0, 10753, usize::MAX] {
        assert!(requirements(width, task, planning, mapping, job, host, 32, None).is_err());
    }
}

#[test]
fn owned_reader_and_private_recipe_share_the_existing_job_boundary() {
    let _host = NlpEngine::job_int4_faithfulness_document::<std::io::Cursor<Vec<u8>>>;
    fn send<T: Send + 'static>() {}
    send::<JobInput<std::io::Cursor<Vec<u8>>>>();
}
