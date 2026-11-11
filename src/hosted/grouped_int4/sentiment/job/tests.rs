//! Model-free durable-host refusal and ownership contracts. These fixtures
//! cannot manufacture native weights or confer model-quality evidence.
use super::*;
use std::time::Duration;
use crate::{jobs::population::PopulationReadLimits,
    native_engine::lmhead::NANBEIGE_VOCAB_SIZE,
    tasks::{mapreduce::ChunkLimits, sentiment::SentimentAggregationPolicy}};

fn fixture() -> (TaskBudget, SentimentLimits, Int4SentimentMapLimits, JobLimits, JobHostLimits) {
    let task = TaskBudget { max_input_tokens: 128, max_output_tokens: 16,
        max_output_bytes: 65_536, max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(256, 32).unwrap().kv_bytes * 4 };
    let mapping = Int4SentimentMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 512,
            max_chunk_tokens: 128, context_tokens: 256, reserved_tokens: 16,
            max_chunks: 64, max_tokenizer_calls: 1024 },
        max_model_work: Int4Work::for_sequence(32, 0, 1024, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
        max_retained_result_bytes: 8 << 20, max_result_bytes: 1 << 20,
        aggregation: SentimentAggregationPolicy::default(),
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
    (task, SentimentLimits::default(), mapping, job, host)
}

#[test]
fn complete_population_and_replanned_documents_coexist_in_the_reservation() {
    let (task, planning, mapping, job, host) = fixture();
    for (axes, chunks) in [(1, None), (4, None), (4, Some(1)), (4, Some(16)), (1, Some(64))] {
        let needed = requirements(32, task, planning, mapping, job, host, axes, chunks, JobInputMemory::CompletePopulation).unwrap();
        let map = map_memory(host, axes, chunks).requirements(32, task, planning, mapping).unwrap();
        assert_eq!(needed.native, map.native);
        assert_eq!(needed.output_reservation_bytes, map.output_reservation_bytes);
        assert!(needed.input_bytes > host.required_buffer_bytes(job).unwrap());
        assert_eq!(needed.input_bytes, host.required_buffer_bytes(job).unwrap()
            + map.input_bytes - host.preparation_reserve_bytes
            + mapping.chunks.max_chunks as u64 * SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES);
        // Native windows never replace the original population commitment.
        let mut larger = job;
        larger.max_snapshot_bytes += 4096;
        let larger = requirements(32, task, planning, mapping, larger, host, axes, chunks, JobInputMemory::CompletePopulation).unwrap();
        assert_eq!(larger.input_bytes - needed.input_bytes, 8192);
        assert_eq!(larger.native, needed.native);
    }
}

#[test]
fn each_native_work_axis_must_fit_the_persisted_lifetime_allowance() {
    let (task, planning, mapping, job, host) = fixture();
    for axis in 0..6 {
        let mut limited = job;
        match axis {
            0 => limited.max_work.model.forward_positions -= 1,
            1 => limited.max_work.model.projected_logits -= 1,
            2 => limited.max_work.model.attention_pairs -= 1,
            3 => limited.max_work.model.projections.dot_products -= 1,
            4 => limited.max_work.model.projections.multiply_accumulates -= 1,
            _ => limited.max_work.mask_node_visits = 1,
        }
        assert!(requirements(32, task, planning, mapping, limited, host, 4, None, JobInputMemory::CompletePopulation).is_err(), "axis {axis}");
    }
    let quote = requirements(32, task, planning, mapping, job, host, 4, None, JobInputMemory::CompletePopulation).unwrap();
    let mut more_attempts = job;
    more_attempts.max_attempts = 100;
    assert_eq!(quote, requirements(32, task, planning, mapping, more_attempts, host, 4, None, JobInputMemory::CompletePopulation).unwrap());
}

#[test]
fn oversized_axis_windows_refuse_without_clamping_the_fixed_recipe() {
    let (task, planning, mapping, job, host) = fixture();
    for (axes, chunks) in [(0, None), (5, None), (4, Some(0)), (4, Some(17)),
        (2, Some(33)), (1, Some(65)), (1, Some(usize::MAX))] {
        assert!(requirements(32, task, planning, mapping, job, host, axes, chunks, JobInputMemory::CompletePopulation).is_err());
    }
    let needed = requirements(32, task, planning, mapping, job, host, 2, Some(3), JobInputMemory::CompletePopulation).unwrap();
    let row = Int4MemoryRequirement::for_context(256, 32).unwrap();
    assert_eq!(needed.native.kv_bytes, row.kv_bytes * 6);
    let mut missing_axis = task;
    missing_axis.max_kv_bytes = row.kv_bytes * 2 - 1;
    assert!(requirements(32, missing_axis, planning, mapping, job, host, 2, Some(3), JobInputMemory::CompletePopulation).is_err());
}

#[test]
fn stored_document_cap_accounts_for_actual_cohort_wrapper() {
    let (task, planning, mapping, mut job, host) = fixture();
    job.max_result_bytes = mapping.max_result_bytes as usize;
    requirements(32, task, planning, mapping, job, host, 4, None, JobInputMemory::CompletePopulation).unwrap();
    assert!(requirements(32, task, planning, mapping, job, host, 4, Some(2), JobInputMemory::CompletePopulation).is_err());
    job.max_result_bytes += 4096;
    let quote = requirements(32, task, planning, mapping, job, host, 4, Some(2), JobInputMemory::CompletePopulation).unwrap();
    assert_eq!(quote.max_output_bytes, job.max_result_bytes as u64);
    assert!(quote.output_reservation_bytes > quote.max_output_bytes);
    job.max_result_bytes -= 1;
    assert!(requirements(32, task, planning, mapping, job, host, 4, Some(2), JobInputMemory::CompletePopulation).is_err());
}

#[test]
fn host_shortfalls_and_arithmetic_overflow_fail_before_population_io() {
    let (task, planning, mapping, job, host) = fixture();
    for field in 0..11 {
        let mut host = host;
        match field {
            0 => host.native.context_tokens = 255,
            1 => host.native.context_tokens = 257,
            2 => host.native.allocator_reserve_bytes = 0,
            3 => host.preparation_reserve_bytes = (128 << 20) - 1,
            4 => host.preparation_reserve_bytes = u64::MAX,
            5 => host.io_reserve_bytes = u64::MAX,
            6 => host.journal_reserve_bytes = 0,
            7 => host.serialization_reserve_bytes = 0,
            8 => host.native.run.max_checkpoints = 1,
            9 => host.transport.max_stream_bytes = 0,
            _ => host.native.allocator_reserve_bytes = u64::MAX,
        }
        assert!(requirements(32, task, planning, mapping, job, host, 4, None, JobInputMemory::CompletePopulation).is_err(), "field {field}");
    }
    for width in [0, 10753, usize::MAX] {
        assert!(requirements(width, task, planning, mapping, job, host, 4, None, JobInputMemory::CompletePopulation).is_err());
    }
}

#[test]
fn owned_reader_and_private_recipe_travel_through_the_hosted_job_boundary() {
    let _host = NlpEngine::job_int4_sentiment_document::<std::io::Cursor<Vec<u8>>>;
    fn send<T: Send + 'static>() {}
    send::<JobInput<std::io::Cursor<Vec<u8>>>>();
}

#[cfg(feature = "asupersync-runtime")]
mod native;

#[test]
fn retained_replay_keeps_one_compact_workspace_without_restoring_population_payloads() {
    use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
    let policy = SafeBpeLimits { max_input_bytes: 262_144, max_output_tokens: 131_072,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 };
    let encoding = UntrustedEncoding::SafeBpeV1(policy);
    let workspace = policy.requirements().unwrap().workspace_bytes;
    let (task, planning, mapping, job, host) = fixture();
    for cohorts in [None, Some(2)] {
        let complete = requirements(32, task, planning, mapping, job, host, 4, cohorts,
            JobInputMemory::CompletePopulation).unwrap().with_input_encoding(&encoding).unwrap();
        let retained = requirements(32, task, planning, mapping, job, host, 4, cohorts,
            JobInputMemory::RetainedRecord).unwrap();
        let compact = retained.with_input_encoding(&encoding).unwrap();
        assert_eq!(compact.input_bytes - retained.input_bytes, workspace);
        assert_eq!(complete.input_bytes - compact.input_bytes, 2 * job.max_snapshot_bytes);
        assert_eq!(complete.native, compact.native);
        assert_eq!(compact.native, retained.native);
        assert_eq!(compact.output_reservation_bytes, retained.output_reservation_bytes);
    }
}
