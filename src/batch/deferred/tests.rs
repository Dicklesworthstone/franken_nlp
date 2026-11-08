//! Transport fixtures only; no synthetic result is model-parity evidence.
use super::*;
use std::io::{self, Cursor};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
#[derive(Default)]
struct Echo { calls: usize }
impl BatchProcessor for Echo {
    type Args = (); type Prepared = String; type Output = String;
    fn prepare(&mut self, document: BatchDocument<()>) -> Result<String, BatchItemFailure> { Ok(document.text) }
    fn planned_work(&self, _: &String) -> BatchWork { BatchWork { forward_positions: 2, projected_logits: 3 } }
    fn execute<C: DecodeStepControl>(&mut self, text: String, _: &mut C) -> Result<String, BatchItemFailure> {
        self.calls += 1;
        if text == "fail" { Err(BatchItemFailure::reject(BatchCode::Execution)) } else { Ok(text) }
    }
}
const INPUT: &[u8] = b"{\"id\":\"a\",\"text\":\"fail\"}\n{\"flush\":true}\n{\"id\":\"a\",\"text\":\"second\"}\n{\"id\":\"b\",\"text\":\"over-budget\"}";

#[test]
fn deferred_footer_preserves_exact_records_epochs_and_nonrenewable_work() {
    let limits = BatchLimits { max_work: BatchWork { forward_positions: 4, projected_logits: 6 },
        ..BatchLimits::default() };
    let mut expected = Vec::new(); let mut serial = Echo::default();
    let summary = run_ndjson(&mut Cursor::new(INPUT), &mut expected, &mut serial, limits, &mut Continue).unwrap();
    let mut output = Vec::new(); let mut processor = Echo::default();
    let deferred = run_ndjson_deferred(&mut Cursor::new(INPUT), &mut output, &mut processor, limits, &mut Continue).unwrap();
    let provisional = std::str::from_utf8(&output).unwrap();
    assert!(!provisional.contains("run_complete"));
    assert!(provisional.contains("work_limit"));
    assert_eq!(processor.calls, 2); assert_eq!(processor.calls, serial.calls);
    assert_eq!(deferred.publish(&mut output, &mut Continue).unwrap(), summary);
    assert_eq!(summary.failed, 2); assert_eq!(summary.succeeded, 1); assert_eq!(summary.flushes, 2);
    assert_eq!(summary.reserved_work, limits.max_work);
    assert_eq!(output, expected, "only terminal publication timing may change");
}

#[test]
fn post_drain_cancellation_never_publishes_or_retries_success() {
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let mut output = Vec::new();
        let deferred = run_ndjson_deferred(&mut Cursor::new(INPUT), &mut output,
            &mut Echo::default(), BatchLimits::default(), &mut Continue).unwrap();
        let prefix = output.clone();
        let error = deferred.publish(&mut output, &mut Stop(cause)).unwrap_err();
        assert_eq!(error.fault, BatchFault::cancelled(cause));
        assert_eq!(error.summary.reserved_work.forward_positions, 6);
        assert_eq!(output, prefix);
        assert!(!std::str::from_utf8(&output).unwrap().contains("run_complete"));
    }
}

#[test]
fn deferred_transport_keeps_cumulative_output_refusal_and_terminal_error_bytes() {
    let input: Vec<u8> = (0..40).flat_map(|index|
        format!("{{\"id\":\"{index}\",\"text\":\"bounded\"}}\n").into_bytes()).collect();
    let limits = BatchLimits { max_output_bytes: 4096, ..BatchLimits::default() };
    let mut expected = Vec::new(); let mut serial = Echo::default();
    let error = run_ndjson(&mut Cursor::new(&input), &mut expected, &mut serial, limits, &mut Continue).unwrap_err();
    let mut output = Vec::new(); let mut processor = Echo::default();
    let deferred = run_ndjson_deferred(&mut Cursor::new(&input), &mut output, &mut processor, limits, &mut Continue);
    let Err(actual) = deferred else { panic!("deferred completion reset output authority"); };
    assert_eq!(actual, error); assert_eq!(error.fault.code, BatchCode::OutputLimit);
    assert_eq!(processor.calls, serial.calls); assert_eq!(output, expected);
    assert!(output.len() <= limits.max_output_bytes as usize);
}

#[test]
fn partial_terminal_write_and_failed_flush_are_terminal_without_error_append() {
    struct Writer { bytes: Vec<u8>, armed: bool, fail_flush: bool, attempts: usize, flushes: usize }
    impl Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.armed {
                self.attempts += 1;
                if !self.fail_flush {
                    self.bytes.extend_from_slice(&bytes[..bytes.len().min(7)]);
                    return Err(io::Error::other("fixture partial write"));
                }
            }
            self.bytes.extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            if self.armed { self.flushes += 1; return Err(io::Error::other("fixture flush")); }
            Ok(())
        }
    }
    for fail_flush in [false, true] {
        let mut writer = Writer { bytes: Vec::new(), armed: false, fail_flush, attempts: 0, flushes: 0 };
        let deferred = run_ndjson_deferred(&mut Cursor::new(b""), &mut writer,
            &mut Echo::default(), BatchLimits::default(), &mut Continue).unwrap();
        let prefix = writer.bytes.len(); writer.armed = true;
        let error = deferred.publish(&mut writer, &mut Continue).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::OutputIo);
        assert_eq!(writer.attempts, 1); assert_eq!(writer.flushes, usize::from(fail_flush));
        assert!(!std::str::from_utf8(&writer.bytes).unwrap().contains("run_error"));
        if !fail_flush { assert_eq!(writer.bytes.len(), prefix + 7); }
    }
}
