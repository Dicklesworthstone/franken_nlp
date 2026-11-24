//! Resource and terminal-publication contracts; no synthetic neural evidence.
use super::*;
use crate::batch::{BatchDocument, BatchProcessor, BatchWork};
use crate::native_engine::decode::{DecodeCancellationKind, DecodeStepControl};
use crate::tasks::redact::corpus::LongRedactionBatchConfig;
use std::io::Cursor;

fn limits() -> CorpusLimits {
    CorpusLimits { native: NativeLimits { context_tokens: 32, allocator_reserve_bytes: 1 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 10000,
            cleanup_reserve_bytes: 65536 } },
        transport: BatchLimits::default(), preparation_reserve_bytes: 8 << 20,
        io_reserve_bytes: 64 << 10 }
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}

#[test]
fn live_quote_prices_map_edit_secret_io_native_prompt_and_workers_together() {
    let limits = limits(); let native = requirements(limits.native).unwrap();
    let serial = quote(limits, native, 101, 3073, None, None).unwrap();
    assert_eq!(serial.input_bytes, limits.reservation_bytes().unwrap() + 101 + 3073);
    assert_eq!(serial.scratch_bytes, native.rope_bytes + native.scratch_payload_bound
        + limits.native.allocator_reserve_bytes);
    assert_eq!(serial.worker_bytes, 0); assert_eq!(serial.max_batch_rows, 1);
    for width in [2, 7, 64] { for rows in [1, 3, 64] {
        let schedule = prefill(rows);
        let actual = quote(limits, native, 101, 3073, Some(schedule), Some(width)).unwrap();
        assert_eq!(actual.native, serial.native);
        assert_eq!(actual.input_bytes, serial.input_bytes);
        assert_eq!(actual.scratch_bytes, serial.scratch_bytes + schedule.validate().unwrap());
        assert_eq!(actual.worker_bytes, dispatch::int8_prefill_team_workspace_bytes(width, rows).unwrap());
        assert_eq!(actual.max_batch_rows, rows);
        assert_eq!(actual.total_bytes().unwrap(), serial.total_bytes().unwrap()
            + schedule.validate().unwrap() + actual.worker_bytes);
    } }
}

#[test]
fn prefill_cannot_discount_transferred_preparation_edit_frontier_or_secret_capacity() {
    let original = limits(); let native = requirements(original.native).unwrap();
    for schedule in [None, Some(prefill(4))] { for width in [None, Some(4)] {
        let baseline = quote(original, native, 0, 0, schedule, width).unwrap().total_bytes().unwrap();
        let mut larger = original; larger.preparation_reserve_bytes += 19; larger.io_reserve_bytes += 31;
        assert_eq!(quote(larger, native, 43, 59, schedule, width).unwrap().total_bytes().unwrap(),
            baseline + 19 + 31 + 43 + 59);
    } }
}

#[test]
fn invalid_geometry_and_all_memory_overflow_axes_fail_closed() {
    let limits = limits(); let native = requirements(limits.native).unwrap();
    for width in [0, 1, 65, usize::MAX] {
        assert!(quote(limits, native, 1, 1, Some(prefill(4)), Some(width)).is_err());
    }
    for rows in [0, 65, usize::MAX] {
        let invalid = Int8PrefillLimits { max_batch_rows: rows, max_extra_scratch_bytes: u64::MAX };
        assert!(quote(limits, native, 1, 1, Some(invalid), Some(4)).is_err());
    }
    let mut too_small = prefill(4); too_small.max_extra_scratch_bytes -= 1;
    assert!(quote(limits, native, 1, 1, Some(too_small), Some(4)).is_err());
    assert!(quote(limits, native, u64::MAX, 1, None, None).is_err());
    assert!(quote(limits, native, 1, u64::MAX, None, None).is_err());
    let mut overflow = limits; overflow.native.allocator_reserve_bytes = u64::MAX;
    assert!(quote(overflow, native, 1, 1, Some(prefill(4)), Some(4)).is_err());
}

#[test]
fn both_stream_owners_and_all_public_schedule_entries_have_owned_io() {
    fn send<T: Send + 'static>() {}
    send::<Execution<RedactionCorpusConfig, Cursor<Vec<u8>>, Vec<u8>>>();
    send::<Execution<RedactionCorpusConfig<LongRedactionBatchConfig>, Cursor<Vec<u8>>, Vec<u8>>>();
    let _short = NlpEngine::batch_int8_redact_layer_major::<Cursor<Vec<u8>>, Vec<u8>>;
    let _short_team = NlpEngine::batch_int8_redact_with_threads::<Cursor<Vec<u8>>, Vec<u8>>;
    let _short_both = NlpEngine::batch_int8_redact_layer_major_with_threads::<Cursor<Vec<u8>>, Vec<u8>>;
    let _long = NlpEngine::batch_int8_redact_document_layer_major::<Cursor<Vec<u8>>, Vec<u8>>;
    let _long_team = NlpEngine::batch_int8_redact_document_with_threads::<Cursor<Vec<u8>>, Vec<u8>>;
    let _long_both = NlpEngine::batch_int8_redact_document_layer_major_with_threads::<Cursor<Vec<u8>>, Vec<u8>>;
}

struct Control(Option<DecodeCancellationKind>);
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { self.0 }
}
struct TransportOnly;
impl BatchProcessor for TransportOnly {
    type Args = (); type Prepared = (); type Output = ();
    fn prepare(&mut self, _: BatchDocument<()>) -> Result<(), BatchItemFailure> { Ok(()) }
    fn planned_work(&self, _: &()) -> BatchWork { BatchWork { forward_positions: 1, projected_logits: 1 } }
    fn execute<C: DecodeStepControl>(&mut self, _: (), _: &mut C) -> Result<(), BatchItemFailure> { Ok(()) }
}

#[test]
fn late_cancellation_after_records_never_publishes_or_retries_terminal_success() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let mut output = Vec::new();
        let mut reader = Cursor::new(b"{\"id\":\"a\",\"text\":\"private\"}\n{\"flush\":true}\n");
        let Completion::Deferred(completion) = super::super::super::scored::execute_stream(
            &mut reader, &mut output, &mut TransportOnly, BatchLimits::default(), true,
            &mut Control(None)).unwrap() else { panic!("terminal success must await host drain"); };
        assert!(!std::str::from_utf8(&output).unwrap().contains("run_complete"));
        let before = output.clone();
        let error = completion.publish(&mut output, &mut Control(Some(cause))).unwrap_err();
        assert_eq!(error.fault.cancellation, Some(cause));
        assert_eq!(output, before, "late stop cannot append a success or error footer");
    }
}

mod model;
