//! Exact admission/transport seam tests, without invented neural-success claims.
use super::*;
use crate::batch::{BatchDocument, BatchWork};
use crate::native_engine::decode::DecodeCancellationKind;
use std::io::Cursor;

fn limits() -> CorpusLimits {
    CorpusLimits { native: NativeLimits { context_tokens: 32, allocator_reserve_bytes: 1024 * 1024,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 10000,
            cleanup_reserve_bytes: 65536 } },
        transport: BatchLimits::default(), preparation_reserve_bytes: 8 * 1024 * 1024,
        io_reserve_bytes: 64 * 1024 }
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}

#[test]
fn combined_quote_keeps_native_input_encoder_coordinator_and_children_simultaneously() {
    let limits = limits(); let native = requirements(limits.native).unwrap();
    let serial = quote(limits, native, 17, None, None).unwrap();
    assert_eq!(serial.input_bytes, limits.reservation_bytes().unwrap() + 17);
    assert_eq!(serial.scratch_bytes, native.rope_bytes + native.scratch_payload_bound
        + limits.native.allocator_reserve_bytes);
    assert_eq!(serial.worker_bytes, 0); assert_eq!(serial.max_batch_rows, 1);
    for width in [2, 7, 64] {
        for rows in [1, 3, 64] {
            let schedule = prefill(rows);
            let threaded = quote(limits, native, 17, Some(schedule), Some(width)).unwrap();
            assert_eq!(threaded.native, serial.native);
            assert_eq!(threaded.input_bytes, serial.input_bytes);
            assert_eq!(threaded.scratch_bytes, serial.scratch_bytes + schedule.validate().unwrap());
            assert_eq!(threaded.worker_bytes,
                dispatch::int8_prefill_team_workspace_bytes(width, rows).unwrap());
            assert_eq!(threaded.max_batch_rows, rows);
            assert_eq!(threaded.total_bytes().unwrap(), serial.total_bytes().unwrap()
                + schedule.validate().unwrap() + threaded.worker_bytes);
        }
    }
}

#[test]
fn tokenizer_and_external_stream_storage_cannot_be_discounted_by_scheduling() {
    let original = limits(); let native = requirements(original.native).unwrap();
    for schedule in [None, Some(prefill(4))] {
        for width in [None, Some(4)] {
            let baseline = quote(original, native, 0, schedule, width).unwrap().total_bytes().unwrap();
            let mut larger = original; larger.preparation_reserve_bytes += 19;
            larger.io_reserve_bytes += 31;
            assert_eq!(quote(larger, native, 43, schedule, width).unwrap().total_bytes().unwrap(),
                baseline + 19 + 31 + 43);
        }
    }
}

#[test]
fn malformed_or_underfunded_prompt_geometry_and_team_width_refuse_before_stream_io() {
    let limits = limits(); let native = requirements(limits.native).unwrap();
    for width in [0, 1, 65, usize::MAX] {
        assert!(quote(limits, native, 0, Some(prefill(4)), Some(width)).is_err());
    }
    for rows in [0, 65, usize::MAX] {
        let invalid = Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX };
        assert!(quote(limits, native, 0, Some(invalid), Some(4)).is_err());
    }
    let mut too_small = prefill(4); too_small.max_extra_scratch_bytes -= 1;
    assert!(quote(limits, native, 0, Some(too_small), Some(4)).is_err());
    assert!(quote(limits, native, u64::MAX, Some(prefill(4)), Some(4)).is_err());
    let mut overflow = limits; overflow.native.allocator_reserve_bytes = u64::MAX;
    assert!(quote(overflow, native, 0, Some(prefill(4)), Some(4)).is_err());
}

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
#[derive(Default)]
struct Echo { calls: usize }
impl BatchProcessor for Echo {
    type Args = (); type Prepared = String; type Output = String;
    fn prepare(&mut self, document: BatchDocument<()>) -> Result<String, BatchItemFailure> { Ok(document.text) }
    fn planned_work(&self, _: &String) -> BatchWork {
        BatchWork { forward_positions: 2, projected_logits: 3 }
    }
    fn execute<C: DecodeStepControl>(&mut self, value: String, _: &mut C)
        -> Result<String, BatchItemFailure> {
        self.calls += 1;
        if value == "reject" { Err(BatchItemFailure::reject(BatchCode::Execution)) } else { Ok(value) }
    }
}

#[test]
fn shared_scored_transport_preserves_records_and_flush_work_and_defers_only_terminal_frame() {
    let input = b"{\"id\":\"a\",\"text\":\"reject\"}\n{\"flush\":true}\n{\"id\":\"a\",\"text\":\"ok\"}\n{\"id\":\"b\",\"text\":\"over-budget\"}";
    let limits = BatchLimits { max_work: BatchWork { forward_positions: 4, projected_logits: 6 },
        ..BatchLimits::default() };
    let mut expected = Vec::new(); let mut serial = Echo::default();
    let Completion::Published(summary) = execute_stream(&mut Cursor::new(input), &mut expected,
        &mut serial, limits, false, &mut Continue).unwrap() else { panic!("serial footer was withheld"); };
    let mut output = Vec::new(); let mut threaded = Echo::default();
    let Completion::Deferred(completion) = execute_stream(&mut Cursor::new(input), &mut output,
        &mut threaded, limits, true, &mut Continue).unwrap() else { panic!("success preceded physical join"); };
    assert!(!std::str::from_utf8(&output).unwrap().contains("run_complete"));
    assert_eq!(serial.calls, 2); assert_eq!(threaded.calls, serial.calls);
    assert_eq!(summary.reserved_work, limits.max_work);
    assert_eq!(summary.succeeded, 1); assert_eq!(summary.failed, 2);
    assert_eq!(completion.publish(&mut output, &mut Continue).unwrap(), summary);
    assert_eq!(output, expected, "scheduling must not replace ordered per-item semantics");
}

mod native;

mod structured;
