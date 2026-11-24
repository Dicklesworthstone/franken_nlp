//! Complete admission geometry without native or durability proof claims.
use super::*;
use crate::jobs::JobWork;

fn limits() -> JobHostLimits {
    JobHostLimits { native: NativeLimits { context_tokens: 32, allocator_reserve_bytes: 1 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 10000,
            cleanup_reserve_bytes: 65536 } },
        transport: PopulationReadLimits { max_stream_bytes: 128 << 20, max_lines: 32 },
        preparation_reserve_bytes: 8 << 20, io_reserve_bytes: 65536,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 2 << 20 }
}
fn job() -> JobLimits {
    JobLimits { max_items: 4, max_id_bytes: 128, max_input_bytes_per_item: 65536,
        max_snapshot_bytes: 64 << 20, max_result_bytes: 65536, max_spool_bytes: 16 << 20,
        max_materialized_bytes: 16 << 20, max_journal_bytes: 1 << 20, max_attempts: 16,
        max_work: JobWork { model: Int8Work::for_sequence(0, 8192, 100_000_000).unwrap(),
            mask_node_visits: 0 } }
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}

#[test]
fn retained_job_quote_keeps_population_journal_native_prompt_and_workers_simultaneously() {
    let limits = limits(); let native = requirements(limits.native).unwrap();
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let serial = quote(limits, job(), input, native, None, None).unwrap();
        assert_eq!(serial.input_bytes, limits.required_buffer_bytes_for_input(job(), input).unwrap());
        assert_eq!(serial.worker_bytes, 0);
        assert_eq!(serial.scratch_bytes, native.rope_bytes + native.scratch_payload_bound
            + limits.native.allocator_reserve_bytes);
        for width in [2, 4, 64] { for rows in [1, 4, 64] {
            let prefill = prefill(rows);
            let scheduled = quote(limits, job(), input, native, Some(prefill), Some(width)).unwrap();
            assert_eq!(scheduled.input_bytes, serial.input_bytes);
            assert_eq!(scheduled.scratch_bytes, serial.scratch_bytes + prefill.validate().unwrap());
            assert_eq!(scheduled.worker_bytes, dispatch::int8_prefill_team_workspace_bytes(width, rows).unwrap());
            assert_eq!(scheduled.total_bytes().unwrap(), serial.total_bytes().unwrap()
                + prefill.validate().unwrap() + scheduled.worker_bytes);
        } }
    }
}

#[test]
fn schedule_cannot_discount_either_input_owner_or_durable_staging() {
    let original = limits(); let native = requirements(original.native).unwrap();
    let a = quote(original, job(), JobInputMemory::CompletePopulation, native, None, None).unwrap();
    let b = quote(original, job(), JobInputMemory::RetainedRecord, native, None, None).unwrap();
    assert!(b.input_bytes < a.input_bytes);
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let before = quote(original, job(), input, native, Some(prefill(4)), Some(4)).unwrap();
        let mut larger = original;
        larger.preparation_reserve_bytes += 11; larger.io_reserve_bytes += 17;
        larger.journal_reserve_bytes += 23; larger.serialization_reserve_bytes += 31;
        let after = quote(larger, job(), input, native, Some(prefill(4)), Some(4)).unwrap();
        assert_eq!(after.total_bytes().unwrap(), before.total_bytes().unwrap() + 11 + 17 + 23 + 31);
        assert_eq!(after.native, before.native); assert_eq!(after.worker_bytes, before.worker_bytes);
    }
}

#[test]
fn invalid_width_prompt_geometry_or_arithmetic_refuses_before_population_ingestion() {
    let original = limits(); let native = requirements(original.native).unwrap();
    for width in [0, 1, 65, usize::MAX] {
        assert!(quote(original, job(), JobInputMemory::CompletePopulation,
            native, Some(prefill(4)), Some(width)).is_err());
    }
    for rows in [0, 65, usize::MAX] {
        assert!(quote(original, job(), JobInputMemory::CompletePopulation, native,
            Some(Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX }), Some(4)).is_err());
    }
    let mut short = prefill(4); short.max_extra_scratch_bytes -= 1;
    assert!(quote(original, job(), JobInputMemory::RetainedRecord, native, Some(short), Some(4)).is_err());
    let mut overflow = original; overflow.native.allocator_reserve_bytes = u64::MAX;
    assert!(quote(overflow, job(), JobInputMemory::CompletePopulation, native, None, Some(4)).is_err());
    let mut overflow = original; overflow.journal_reserve_bytes = u64::MAX;
    assert!(quote(overflow, job(), JobInputMemory::RetainedRecord, native, None, Some(4)).is_err());
}

mod native;
