//! Complete retained-job admission and API contracts, without model execution.
use super::*;
use crate::hosted::corpus::jobs::JobOpenMode;
use crate::jobs::{InputRetentionLimits, JobWork, TailPolicy, population::PopulationReadLimits};
use crate::tasks::redact::corpus::LongRedactionBatchConfig;
use std::io::Cursor;

fn job_limits() -> JobLimits {
    let work = crate::native_engine::constrained_int8::planned_work(128, 16).unwrap();
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 65536, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork { model: work.checked_add(work).unwrap(), mask_node_visits: 4000 } }
}
fn limits() -> JobHostLimits {
    JobHostLimits { native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 64 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000,
            cleanup_reserve_bytes: 65_536 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 1000 },
        preparation_reserve_bytes: 32 << 20, io_reserve_bytes: 128 << 10,
        journal_reserve_bytes: 16 << 20, serialization_reserve_bytes: 8 << 20 }
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}

#[test]
fn retained_quote_includes_population_recipe_edit_secret_prompt_and_workers_simultaneously() {
    let limits = limits(); let job = job_limits(); let native = requirements(limits.native).unwrap();
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let serial = quote(limits, job, input, native, 101, 3073, None, None).unwrap();
        assert_eq!(serial.input_bytes, limits.required_buffer_bytes_for_input(job, input).unwrap()
            + 101 + 3073 + 4 * RECIPE_BYTES as u64);
        assert_eq!(serial.scratch_bytes, native.rope_bytes + native.scratch_payload_bound
            + limits.native.allocator_reserve_bytes);
        assert_eq!(serial.worker_bytes, 0); assert_eq!(serial.max_batch_rows, 1);
        for width in [2, 7, 64] { for rows in [1, 3, 64] {
            let schedule = prefill(rows);
            let actual = quote(limits, job, input, native, 101, 3073, Some(schedule), Some(width)).unwrap();
            assert_eq!(actual.native, serial.native);
            assert_eq!(actual.input_bytes, serial.input_bytes);
            assert_eq!(actual.scratch_bytes, serial.scratch_bytes + schedule.validate().unwrap());
            assert_eq!(actual.worker_bytes, dispatch::int8_prefill_team_workspace_bytes(width, rows).unwrap());
            assert_eq!(actual.max_batch_rows, rows);
            assert_eq!(actual.total_bytes().unwrap(), serial.total_bytes().unwrap()
                + schedule.validate().unwrap() + actual.worker_bytes);
        } }
    }
}

#[test]
fn create_and_resume_price_the_actual_population_owner_without_schedule_discounts() {
    let limits = limits(); let job = job_limits(); let native = requirements(limits.native).unwrap();
    let retention = InputRetentionLimits { max_spool_bytes: 1 << 20 };
    for schedule in [None, Some(prefill(7))] { for width in [None, Some(4)] {
        let original = quote(limits, job, JobInputMemory::CompletePopulation, native, 43, 59, schedule, width).unwrap();
        for mode in [JobOpenMode::Create, JobOpenMode::Resume(TailPolicy::Refuse)] {
            assert_eq!(quote(limits, job, mode.input_memory(), native, 43, 59, schedule, width).unwrap(), original);
        }
        for mode in [JobOpenMode::CreateRetainingInputs(retention),
            JobOpenMode::ResumeRetained { tail: TailPolicy::DiscardUncommitted, retention }] {
            let retained = quote(limits, job, mode.input_memory(), native, 43, 59, schedule, width).unwrap();
            assert_eq!(original.input_bytes - retained.input_bytes, 2 * job.max_snapshot_bytes);
            assert_eq!(original.total_bytes().unwrap() - retained.total_bytes().unwrap(), 2 * job.max_snapshot_bytes);
            assert_eq!(original.scratch_bytes, retained.scratch_bytes);
            assert_eq!(original.worker_bytes, retained.worker_bytes);
        }
        let mut larger = limits;
        larger.preparation_reserve_bytes += 19; larger.io_reserve_bytes += 31;
        larger.journal_reserve_bytes += 61; larger.serialization_reserve_bytes += 71;
        assert_eq!(quote(larger, job, JobInputMemory::CompletePopulation, native, 43 + 83, 59 + 97,
            schedule, width).unwrap().total_bytes().unwrap(), original.total_bytes().unwrap()
            + 19 + 31 + 61 + 71 + 83 + 97);
    } }
}

#[test]
fn malformed_job_schedule_and_every_extra_memory_axis_fail_before_population_io() {
    let limits = limits(); let job = job_limits(); let native = requirements(limits.native).unwrap();
    let input = JobInputMemory::RetainedRecord;
    for width in [0, 1, 65, usize::MAX] {
        assert!(quote(limits, job, input, native, 1, 1, Some(prefill(4)), Some(width)).is_err());
    }
    for rows in [0, 65, usize::MAX] {
        let invalid = Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX };
        assert!(quote(limits, job, input, native, 1, 1, Some(invalid), Some(4)).is_err());
    }
    let mut short = prefill(4); short.max_extra_scratch_bytes -= 1;
    assert!(quote(limits, job, input, native, 1, 1, Some(short), Some(4)).is_err());
    assert!(quote(limits, job, input, native, u64::MAX, 1, None, None).is_err());
    assert!(quote(limits, job, input, native, 1, u64::MAX, None, None).is_err());
    let mut overflow = limits; overflow.native.allocator_reserve_bytes = u64::MAX;
    assert!(quote(overflow, job, input, native, 1, 1, Some(prefill(4)), Some(4)).is_err());
    let mut invalid = limits; invalid.journal_reserve_bytes = 0;
    assert!(quote(invalid, job, input, native, 1, 1, None, None).is_err());
    let mut invalid = job; invalid.max_attempts = 0;
    assert!(quote(limits, invalid, input, native, 1, 1, None, None).is_err());
}

#[test]
fn both_job_owners_and_schedule_entrypoints_keep_owned_readers_across_the_runtime_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Execution<RedactionCorpusConfig, Cursor<Vec<u8>>>>();
    send::<Execution<RedactionCorpusConfig<LongRedactionBatchConfig>, Cursor<Vec<u8>>>>();
    let _short = NlpEngine::job_int8_redact_layer_major::<Cursor<Vec<u8>>>;
    let _short_team = NlpEngine::job_int8_redact_with_threads::<Cursor<Vec<u8>>>;
    let _short_both = NlpEngine::job_int8_redact_layer_major_with_threads::<Cursor<Vec<u8>>>;
    let _long = NlpEngine::job_int8_redact_document_layer_major::<Cursor<Vec<u8>>>;
    let _long_team = NlpEngine::job_int8_redact_document_with_threads::<Cursor<Vec<u8>>>;
    let _long_both = NlpEngine::job_int8_redact_document_layer_major_with_threads::<Cursor<Vec<u8>>>;
}

mod model;
