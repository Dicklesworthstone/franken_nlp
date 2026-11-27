//! Delivery/resource-lifetime regression cases with synthetic I/O only.
use super::*;
use std::{cell::{Cell, RefCell}, io::{self, Cursor}, rc::Rc};

#[test]
fn nonfinite_values_are_rejected_instead_of_becoming_json_null() {
    let mut output = Vec::new(); let mut sink = Sink::new(&mut output, BatchLimits::default());
    let value = f64::NAN; let mut event = Event::new("doc", 1, 1); event.result = Some(&value);
    assert_eq!(sink.emit(&event, false).unwrap_err().code, BatchCode::Serialization);
    assert!(sink.writer.is_empty()); assert!(!sink.poisoned());
}
#[test]
fn oversized_output_never_writes_a_prefix_of_the_document_record() {
    let mut output = Vec::new();
    let mut sink = Sink::new(&mut output, BatchLimits { max_output_line_bytes: 2048, ..BatchLimits::default() });
    let value = "private".repeat(400); let mut event = Event::new("doc", 1, 1); event.result = Some(&value);
    assert_eq!(sink.emit(&event, false).unwrap_err().code, BatchCode::OutputLineLimit);
    assert!(sink.writer.is_empty());
}
#[test]
fn total_output_exhaustion_preserves_room_for_a_terminal_error() {
    let mut output = Vec::new(); let limits = BatchLimits { max_output_bytes: 4096, ..BatchLimits::default() };
    let mut sink = Sink::new(&mut output, limits); let mut seq = 0;
    loop {
        seq += 1;
        match sink.emit(&Event::<()>::new("doc", 1, seq), false) {
            Ok(()) => {}, Err(e) => { assert_eq!(e.code, BatchCode::OutputLimit); break; }
        }
    }
    let mut error = Event::<()>::new("run_error", 1, seq); error.error = Some(BatchCode::OutputLimit.into());
    sink.emit(&error, true).unwrap(); drop(sink);
    assert!(output.len() <= 4096);
    let last = std::str::from_utf8(&output).unwrap().lines().last().unwrap();
    assert_eq!(canonjson::parse_str(last).unwrap()["event"], "run_error");
}
#[test]
fn flush_failure_poisons_output_without_retrying_a_fully_written_record() {
    struct Writer { calls: usize }
    impl Write for Writer {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> { self.calls += 1; Ok(b.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("private output failure")) }
    }
    let mut writer = Writer { calls: 0 }; let mut sink = Sink::new(&mut writer, BatchLimits::default());
    let event = Event::<()>::new("run_start", 1, 0);
    assert_eq!(sink.emit(&event, false).unwrap_err().code, BatchCode::OutputIo);
    let calls = sink.writer.calls;
    assert_eq!(sink.emit(&event, true).unwrap_err().code, BatchCode::OutputIo); assert_eq!(sink.writer.calls, calls);
}
#[test]
fn admission_guard_survives_serialization_write_and_flush_but_not_next_item() {
    struct Guard(Rc<Cell<bool>>);
    impl Drop for Guard { fn drop(&mut self) { self.0.set(false); } }
    struct Processor { live: Rc<Cell<bool>> }
    impl BatchProcessor for Processor {
        type Args = (); type Prepared = String; type Output = GuardedOutput<String, Guard>;
        fn prepare(&mut self, doc: BatchDocument<()>) -> Result<String, BatchItemFailure> {
            assert!(!self.live.get(), "prior output guard must drop before the next item"); Ok(doc.text)
        }
        fn planned_work(&self, _: &String) -> BatchWork { BatchWork::default() }
        fn execute<C: DecodeStepControl>(&mut self, value: String, _: &mut C) -> Result<Self::Output, BatchItemFailure> {
            self.live.set(true); Ok(GuardedOutput::new(value, Guard(self.live.clone())))
        }
    }
    struct Writer { live: Rc<Cell<bool>>, flushes: Rc<RefCell<Vec<bool>>> }
    impl Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { self.flushes.borrow_mut().push(self.live.get()); Ok(()) }
    }
    struct Continue;
    impl DecodeStepControl for Continue {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
    }
    let live = Rc::new(Cell::new(false)); let flushes = Rc::new(RefCell::new(Vec::new()));
    let mut processor = Processor { live: live.clone() }; let mut writer = Writer { live: live.clone(), flushes: flushes.clone() };
    let input = b"{\"id\":\"a\",\"text\":\"a\"}\n{\"id\":\"b\",\"text\":\"b\"}\n";
    run_ndjson(&mut Cursor::new(input), &mut writer, &mut processor, BatchLimits::default(), &mut Continue).unwrap();
    assert_eq!(*flushes.borrow(), vec![false, true, true, false, false]); assert!(!live.get());
}

const CANCELLATION_CAUSES: [DecodeCancellationKind; 11] = [
    DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
    DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
    DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
    DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit,
];

#[derive(Default)]
struct PublicationState {
    cause: Cell<Option<DecodeCancellationKind>>, serializations: Cell<usize>,
    live_results: Cell<usize>, live_guard: Cell<bool>, executions: Cell<usize>,
}
struct PublicationGuard(Rc<PublicationState>);
impl Drop for PublicationGuard {
    fn drop(&mut self) {
        assert_eq!(self.0.live_results.get(), 0, "all unpublished results must drop before admission");
        self.0.live_guard.set(false);
    }
}
struct CancellingPayload { state: Rc<PublicationState>, pass: usize, cause: DecodeCancellationKind }
impl Serialize for CancellingPayload {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        assert!(self.state.live_guard.get(), "serialization must retain admission");
        let pass = self.state.serializations.get() + 1;
        self.state.serializations.set(pass);
        if pass == self.pass { self.state.cause.set(Some(self.cause)); }
        serializer.serialize_str("private completed result")
    }
}
impl Drop for CancellingPayload {
    fn drop(&mut self) { self.state.live_results.set(self.state.live_results.get() - 1); }
}
struct PublicationControl(Rc<PublicationState>);
impl DecodeStepControl for PublicationControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if self.0.cause.get().is_some() && self.0.live_results.get() > 0 {
            assert!(self.0.live_guard.get(), "cancellation refusal must retain the whole result reservation");
        }
        self.0.cause.get()
    }
}
struct PublicationProcessor { state: Rc<PublicationState>, pass: usize, cause: DecodeCancellationKind }
impl PublicationProcessor {
    fn guard(&self) -> PublicationGuard {
        assert!(!self.state.live_guard.replace(true));
        self.state.executions.set(self.state.executions.get() + 1);
        PublicationGuard(Rc::clone(&self.state))
    }
    fn payload(&self) -> CancellingPayload {
        self.state.live_results.set(self.state.live_results.get() + 1);
        CancellingPayload { state: Rc::clone(&self.state), pass: self.pass, cause: self.cause }
    }
}
impl BatchProcessor for PublicationProcessor {
    type Args = (); type Prepared = (); type Output = GuardedOutput<CancellingPayload, PublicationGuard>;
    fn prepare(&mut self, _: BatchDocument<()>) -> Result<(), BatchItemFailure> { Ok(()) }
    fn planned_work(&self, _: &()) -> BatchWork { BatchWork { forward_positions: 1, projected_logits: 1 } }
    fn execute<C: DecodeStepControl>(&mut self, _: (), _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        let guard = self.guard(); Ok(GuardedOutput::new(self.payload(), guard))
    }
}
impl grouped::GroupedBatchProcessor for PublicationProcessor {
    type Args = (); type Prepared = (); type Output = CancellingPayload; type Guard = PublicationGuard;
    fn max_group_size(&self) -> usize { 2 }
    fn prepare(&mut self, _: BatchDocument<()>) -> Result<(), BatchItemFailure> { Ok(()) }
    fn planned_work(&self, _: &()) -> Result<BatchWork, BatchItemFailure> {
        Ok(BatchWork { forward_positions: 1, projected_logits: 1 })
    }
    fn execute_group<C: DecodeStepControl>(&mut self, requests: Vec<grouped::GroupedRequest<()>>, _: &mut C)
        -> Result<grouped::GroupedOutput<Self::Output, Self::Guard>, BatchFault> {
        let guard = self.guard();
        let rows = requests.into_iter().map(|request| grouped::GroupedRow {
            request_seq: request.context.request_seq, result: Ok(self.payload()),
        }).collect();
        Ok(grouped::GroupedOutput::new(rows, guard))
    }
}

#[test]
fn cancellation_in_any_serialization_pass_refuses_serial_and_grouped_document_publication() {
    // Sizing, finite-value validation and canonical tree construction each
    // invoke Serialize. A deadline can expire in any of them.
    for cause in CANCELLATION_CAUSES {
        for pass in 1..=3 {
            for grouped in [false, true] {
                let state = Rc::new(PublicationState::default());
                let mut processor = PublicationProcessor { state: Rc::clone(&state), pass, cause };
                let mut control = PublicationControl(Rc::clone(&state));
                let bytes = b"{\"id\":\"a\",\"text\":\"a\"}\n{\"id\":\"b\",\"text\":\"b\"}\n{\"id\":\"c\",\"text\":\"c\"}\n";
                let mut reader = Cursor::new(bytes); let mut writer = Vec::new();
                let result = if grouped {
                    grouped::run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(),
                        grouped::GroupLimits { max_records: 2 }, &mut control)
                } else {
                    run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(), &mut control)
                };
                let error = result.unwrap_err();
                assert_eq!(error.fault, BatchFault::cancelled(cause));
                assert_eq!(error.summary.succeeded, 0);
                assert_eq!(error.summary.reserved_work.forward_positions, if grouped { 2 } else { 1 });
                assert_eq!(state.executions.get(), 1); assert_eq!(state.live_results.get(), 0);
                assert!(!state.live_guard.get()); assert!(reader.position() < bytes.len() as u64);
                let events: Vec<serde_json::Value> = std::str::from_utf8(&writer).unwrap().lines()
                    .map(|line| serde_json::from_str(line).unwrap()).collect();
                assert_eq!(events.iter().map(|event| event["event"].as_str().unwrap()).collect::<Vec<_>>(),
                    vec!["run_start", "run_error"]);
                assert_eq!(events[1]["error"], serde_json::to_value(BatchFault::cancelled(cause)).unwrap());
                assert!(!std::str::from_utf8(&writer).unwrap().contains("private completed result"));
            }
        }
    }
}

#[test]
fn cancellation_during_eof_flush_never_acknowledges_or_appends_after_delivery() {
    struct Writer { output: Vec<u8>, state: Rc<PublicationState>, cause: DecodeCancellationKind, eof: bool }
    impl Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            let event: serde_json::Value = serde_json::from_slice(bytes).unwrap();
            self.eof = event["event"] == "flush" && event["eof"] == true;
            self.output.extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            if self.eof { self.state.cause.set(Some(self.cause)); }
            Ok(())
        }
    }
    for cause in CANCELLATION_CAUSES {
        for grouped in [false, true] {
            let state = Rc::new(PublicationState::default());
            let mut processor = PublicationProcessor { state: Rc::clone(&state), pass: usize::MAX, cause };
            let mut control = PublicationControl(Rc::clone(&state));
            let mut writer = Writer { output: Vec::new(), state, cause, eof: false };
            let mut reader = Cursor::new(b"");
            let result = if grouped {
                grouped::run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(),
                    grouped::GroupLimits::default(), &mut control)
            } else {
                run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(), &mut control)
            };
            let error = result.unwrap_err();
            assert_eq!(error.fault, BatchFault::cancelled(cause));
            assert_eq!(error.summary.flushes, 0);
            let events: Vec<serde_json::Value> = std::str::from_utf8(&writer.output).unwrap().lines()
                .map(|line| serde_json::from_str(line).unwrap()).collect();
            assert_eq!(events.iter().map(|event| event["event"].as_str().unwrap()).collect::<Vec<_>>(),
                vec!["run_start", "flush"]);
        }
    }
}


#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeliveryStage { Write, Flush }

/// The transport may observe the caller's stop and fail in the same callback.
/// Error strings are private fixture data and must never replace the cause.
struct DeliveryWriter {
    bytes: Vec<u8>, state: Rc<PublicationState>, target: &'static str,
    stage: DeliveryStage, fail: bool, cause: Option<DecodeCancellationKind>,
    armed: bool, writes: usize, flushes: usize,
}
impl DeliveryWriter {
    fn callback(&mut self, stage: DeliveryStage) -> io::Result<()> {
        if self.armed && self.stage == stage {
            self.state.cause.set(self.cause);
            if self.fail { return Err(io::Error::other("private transport failure")); }
        }
        Ok(())
    }
}
impl Write for DeliveryWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let event: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        self.armed = event["event"] == self.target;
        if self.armed {
            self.writes += 1;
            if self.target == "doc" {
                assert!(self.state.live_guard.get(), "delivery must retain result admission");
            }
        }
        self.callback(DeliveryStage::Write)?;
        self.bytes.extend_from_slice(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.armed { self.flushes += 1; }
        self.callback(DeliveryStage::Flush)
    }
}

#[test]
fn transport_callbacks_preserve_terminal_causes_and_never_acknowledge_refused_documents() {
    for cause in [None].into_iter().chain(CANCELLATION_CAUSES.into_iter().map(Some)) {
        for stage in [DeliveryStage::Write, DeliveryStage::Flush] {
            for fail in [false, true] {
                for target in ["doc", "run_complete"] {
                    for grouped in [false, true] {
                        let state = Rc::new(PublicationState::default());
                        let mut processor = PublicationProcessor { state: Rc::clone(&state),
                            pass: usize::MAX, cause: DecodeCancellationKind::User };
                        let mut control = PublicationControl(Rc::clone(&state));
                        let mut writer = DeliveryWriter { bytes: Vec::new(), state: Rc::clone(&state),
                            target, stage, fail, cause, armed: false, writes: 0, flushes: 0 };
                        let input = b"{\"id\":\"a\",\"text\":\"a\"}\n{\"id\":\"b\",\"text\":\"b\"}\n{\"id\":\"c\",\"text\":\"c\"}\n";
                        let mut reader = Cursor::new(input);
                        let result = if grouped {
                            grouped::run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(),
                                grouped::GroupLimits { max_records: 2 }, &mut control)
                        } else {
                            run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(), &mut control)
                        };
                        assert_eq!(state.live_results.get(), 0);
                        assert!(!state.live_guard.get());
                        let events: Vec<serde_json::Value> = std::str::from_utf8(&writer.bytes).unwrap().lines()
                            .map(|line| serde_json::from_str(line).unwrap()).collect();
                        assert!(!events.iter().any(|event| event["event"] == "run_error"),
                            "never append an error after uncertain transport or a delivered success footer");
                        if cause.is_none() && !fail {
                            let summary = result.unwrap();
                            assert_eq!(summary.succeeded, 3); assert_eq!(summary.failed, 0);
                            assert_eq!(summary.reserved_work.forward_positions, 3);
                            assert_eq!(events.last().unwrap()["event"], "run_complete");
                            assert_eq!(writer.writes, if target == "doc" { 3 } else { 1 });
                            assert_eq!(writer.flushes, writer.writes);
                            continue;
                        }
                        let error = result.unwrap_err();
                        assert_eq!(error.fault, cause.map(BatchFault::cancelled)
                            .unwrap_or_else(|| BatchCode::OutputIo.into()));
                        assert_eq!(error.summary.failed, 0);
                        assert_eq!(writer.writes, 1, "a refused record must never be retried");
                        assert_eq!(writer.flushes, usize::from(stage == DeliveryStage::Flush));
                        let published = usize::from(stage == DeliveryStage::Flush || !fail);
                        assert_eq!(events.iter().filter(|event| event["event"] == target).count(), published);
                        if target == "doc" {
                            assert_eq!(error.summary.succeeded, 0, "no acknowledgement after callback refusal");
                            assert_eq!(error.summary.reserved_work.forward_positions, if grouped { 2 } else { 1 });
                            assert_eq!(state.executions.get(), 1);
                            assert!(reader.position() < input.len() as u64);
                            assert!(!events.iter().any(|event| event["event"] == "run_complete"));
                        } else {
                            assert_eq!(error.summary.succeeded, 3);
                            assert_eq!(error.summary.reserved_work.forward_positions, 3);
                            assert_eq!(error.summary.flushes, 1);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn post_write_refusal_seals_sink_without_advancing_its_output_acknowledgement() {
    for stage in [DeliveryStage::Write, DeliveryStage::Flush] {
        for fail in [false, true] {
            let state = Rc::new(PublicationState::default());
            let mut writer = DeliveryWriter { bytes: Vec::new(), state: Rc::clone(&state),
                target: "run_complete", stage, fail, cause: Some(DecodeCancellationKind::Deadline),
                armed: false, writes: 0, flushes: 0 };
            let mut control = PublicationControl(state);
            let mut sink = Sink::new(&mut writer, BatchLimits::default());
            sink.emit(&Event::<()>::new("run_start", 1, 0), false).unwrap();
            let acknowledged = sink.emitted;
            let event = Event::<()>::new("run_complete", 1, 0);
            let error = sink.emit_with_checkpoint(&event, true, || checkpoint(&mut control)).unwrap_err();
            assert_eq!(error, BatchFault::cancelled(DecodeCancellationKind::Deadline));
            assert!(sink.poisoned()); assert_eq!(sink.emitted, acknowledged);
            let bytes = sink.writer.bytes.len();
            assert_eq!(sink.emit(&Event::<()>::new("run_error", 1, 0), true).unwrap_err().code, BatchCode::OutputIo);
            assert_eq!(sink.writer.bytes.len(), bytes); assert_eq!(sink.writer.writes, 1);
        }
    }
}

#[test]
fn read_callbacks_preserve_cancellation_before_errors_eof_or_admission() {
    struct Reader {
        bytes: Cursor<Vec<u8>>, state: Rc<PublicationState>, cause: DecodeCancellationKind,
        fail: bool, calls: usize,
    }
    impl io::Read for Reader {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> { io::Read::read(&mut self.bytes, bytes) }
    }
    impl io::BufRead for Reader {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            self.calls += 1;
            self.state.cause.set(Some(self.cause));
            if self.fail { Err(io::Error::other("private read failure")) }
            else { io::BufRead::fill_buf(&mut self.bytes) }
        }
        fn consume(&mut self, amount: usize) { io::BufRead::consume(&mut self.bytes, amount); }
    }
    for cause in CANCELLATION_CAUSES {
        for (fail, input) in [(true, b"".as_slice()), (false, b"".as_slice()),
            (false, b"{\"id\":\"a\",\"text\":\"private\"}\n".as_slice())] {
            for grouped in [false, true] {
                let state = Rc::new(PublicationState::default());
                let mut processor = PublicationProcessor { state: Rc::clone(&state), pass: usize::MAX, cause };
                let mut control = PublicationControl(Rc::clone(&state));
                let mut reader = Reader { bytes: Cursor::new(input.to_vec()), state: Rc::clone(&state), cause, fail, calls: 0 };
                let mut output = Vec::new();
                let result = if grouped {
                    grouped::run_ndjson(&mut reader, &mut output, &mut processor, BatchLimits::default(),
                        grouped::GroupLimits::default(), &mut control)
                } else {
                    run_ndjson(&mut reader, &mut output, &mut processor, BatchLimits::default(), &mut control)
                };
                let error = result.unwrap_err();
                assert_eq!(error.fault, BatchFault::cancelled(cause));
                assert_eq!(error.summary, BatchSummary::default());
                assert_eq!(reader.calls, 1); assert_eq!(reader.bytes.position(), 0);
                assert_eq!(state.executions.get(), 0);
                let events: Vec<serde_json::Value> = std::str::from_utf8(&output).unwrap().lines()
                    .map(|line| serde_json::from_str(line).unwrap()).collect();
                assert_eq!(events.iter().map(|event| event["event"].as_str().unwrap()).collect::<Vec<_>>(),
                    vec!["run_start", "run_error"]);
                assert_eq!(events[1]["error"], serde_json::to_value(BatchFault::cancelled(cause)).unwrap());
            }
        }
    }
}

struct FailureProcessor {
    state: Rc<PublicationState>, fault: BatchFault, later: Option<DecodeCancellationKind>,
    in_prepare: bool, prepared: usize,
}
impl BatchProcessor for FailureProcessor {
    type Args = (); type Prepared = String; type Output = String;
    fn prepare(&mut self, document: BatchDocument<()>) -> Result<String, BatchItemFailure> {
        self.prepared += 1;
        if self.in_prepare {
            self.state.cause.set(self.later);
            Err(BatchItemFailure::reject(self.fault))
        } else { Ok(document.text) }
    }
    fn planned_work(&self, _: &String) -> BatchWork { BatchWork { forward_positions: 1, projected_logits: 1 } }
    fn execute<C: DecodeStepControl>(&mut self, _: String, _: &mut C) -> Result<String, BatchItemFailure> {
        self.state.executions.set(self.state.executions.get() + 1);
        self.state.cause.set(self.later);
        Err(BatchItemFailure::reject(self.fault))
    }
}
impl grouped::GroupedBatchProcessor for FailureProcessor {
    type Args = (); type Prepared = String; type Output = String; type Guard = ();
    fn max_group_size(&self) -> usize { 1 }
    fn prepare(&mut self, document: BatchDocument<()>) -> Result<String, BatchItemFailure> {
        <Self as BatchProcessor>::prepare(self, document)
    }
    fn planned_work(&self, value: &String) -> Result<BatchWork, BatchItemFailure> {
        Ok(<Self as BatchProcessor>::planned_work(self, value))
    }
    fn execute_group<C: DecodeStepControl>(&mut self, requests: Vec<grouped::GroupedRequest<String>>, control: &mut C)
        -> Result<grouped::GroupedOutput<String, ()>, BatchFault> {
        let rows = requests.into_iter().map(|request| grouped::GroupedRow {
            request_seq: request.context.request_seq,
            result: <Self as BatchProcessor>::execute(self, request.prepared, control),
        }).collect();
        Ok(grouped::GroupedOutput::new(rows, ()))
    }
}

#[test]
fn failed_error_delivery_never_replaces_an_already_observed_cancellation() {
    for cause in CANCELLATION_CAUSES {
        for target in ["doc_error", "run_error"] {
            for stage in [DeliveryStage::Write, DeliveryStage::Flush] {
                for grouped in [false, true] {
                    let state = Rc::new(PublicationState::default());
                    let fault = BatchFault::cancelled(cause);
                    let mut processor = FailureProcessor { state: Rc::clone(&state), fault,
                        later: Some(DecodeCancellationKind::Deadline), in_prepare: false, prepared: 0 };
                    let mut control = PublicationControl(Rc::clone(&state));
                    let mut writer = DeliveryWriter { bytes: Vec::new(), state: Rc::clone(&state),
                        target, stage, fail: true, cause: Some(DecodeCancellationKind::Shutdown),
                        armed: false, writes: 0, flushes: 0 };
                    let first = b"{\"id\":\"a\",\"text\":\"a\"}\n";
                    let mut bytes = first.to_vec(); bytes.extend_from_slice(b"{\"id\":\"b\",\"text\":\"b\"}\n");
                    let mut reader = Cursor::new(bytes);
                    let result = if grouped {
                        grouped::run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(),
                            grouped::GroupLimits::default(), &mut control)
                    } else {
                        run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(), &mut control)
                    };
                    let error = result.unwrap_err();
                    assert_eq!(error.fault, fault, "original stop survives later control and sink failures");
                    assert_eq!(reader.position(), first.len() as u64);
                    assert_eq!(processor.prepared, 1); assert_eq!(state.executions.get(), 1);
                    assert_eq!(error.summary.succeeded, 0);
                    assert_eq!(error.summary.failed, u64::from(target == "run_error"));
                    assert_eq!(error.summary.reserved_work.forward_positions, 1);
                    assert_eq!(writer.writes, 1, "reporting failures are never retried");
                    assert_eq!(writer.flushes, usize::from(stage == DeliveryStage::Flush));
                    let output = std::str::from_utf8(&writer.bytes).unwrap();
                    assert!(!output.contains("run_complete")); assert!(!output.contains("private transport failure"));
                    if target == "doc_error" { assert!(!output.contains("run_error")); }
                }
            }
        }
    }
}

#[test]
fn failed_planning_or_execution_observes_the_original_control_before_reporting() {
    for cause in CANCELLATION_CAUSES {
        for in_prepare in [false, true] {
            for grouped in [false, true] {
                let state = Rc::new(PublicationState::default());
                let mut processor = FailureProcessor { state: Rc::clone(&state), fault: BatchCode::Planning.into(),
                    later: Some(cause), in_prepare, prepared: 0 };
                let mut control = PublicationControl(Rc::clone(&state));
                let first = b"{\"id\":\"a\",\"text\":\"a\"}\n";
                let mut bytes = first.to_vec(); bytes.extend_from_slice(b"{\"id\":\"b\",\"text\":\"b\"}\n");
                let mut reader = Cursor::new(bytes); let mut output = Vec::new();
                let result = if grouped {
                    grouped::run_ndjson(&mut reader, &mut output, &mut processor, BatchLimits::default(),
                        grouped::GroupLimits::default(), &mut control)
                } else {
                    run_ndjson(&mut reader, &mut output, &mut processor, BatchLimits::default(), &mut control)
                };
                let error = result.unwrap_err();
                assert_eq!(error.fault, BatchFault::cancelled(cause));
                assert_eq!(reader.position(), first.len() as u64);
                assert_eq!(processor.prepared, 1); assert_eq!(state.executions.get(), usize::from(!in_prepare));
                assert_eq!(error.summary.succeeded, 0);
                assert_eq!(error.summary.reserved_work.forward_positions, u64::from(!in_prepare));
                let events: Vec<serde_json::Value> = std::str::from_utf8(&output).unwrap().lines()
                    .map(|line| serde_json::from_str(line).unwrap()).collect();
                assert_eq!(events.last().unwrap()["event"], "run_error");
                for event in events.iter().filter(|event| event.get("error").is_some()) {
                    assert_eq!(event["error"], serde_json::to_value(BatchFault::cancelled(cause)).unwrap());
                }
            }
        }
    }
}

#[test]
fn first_cancellation_during_terminal_error_delivery_is_observed_after_the_callback() {
    for cause in CANCELLATION_CAUSES {
        for stage in [DeliveryStage::Write, DeliveryStage::Flush] {
            for fail in [false, true] {
                for grouped in [false, true] {
                    let state = Rc::new(PublicationState::default());
                    let mut processor = FailureProcessor { state: Rc::clone(&state),
                        fault: BatchCode::InputIo.into(), later: None, in_prepare: true, prepared: 0 };
                    let mut control = PublicationControl(Rc::clone(&state));
                    let mut writer = DeliveryWriter { bytes: Vec::new(), state: Rc::clone(&state),
                        target: "run_error", stage, fail, cause: Some(cause), armed: false, writes: 0, flushes: 0 };
                    let first = b"{\"id\":\"a\",\"text\":\"a\"}\n";
                    let mut bytes = first.to_vec(); bytes.extend_from_slice(b"{\"id\":\"b\",\"text\":\"b\"}\n");
                    let mut reader = Cursor::new(bytes);
                    let result = if grouped {
                        grouped::run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(),
                            grouped::GroupLimits::default(), &mut control)
                    } else {
                        run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(), &mut control)
                    };
                    assert_eq!(result.unwrap_err().fault, BatchFault::cancelled(cause));
                    assert_eq!(reader.position(), first.len() as u64);
                    assert_eq!(processor.prepared, 1); assert_eq!(state.executions.get(), 0);
                    assert_eq!(writer.writes, 1);
                    assert_eq!(writer.flushes, usize::from(stage == DeliveryStage::Flush || !fail));
                    assert!(!std::str::from_utf8(&writer.bytes).unwrap().contains("run_complete"));
                }
            }
        }
    }
}
