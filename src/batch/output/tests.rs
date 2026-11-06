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
fn cancellation_at_eof_flush_emits_error_instead_of_success_footer() {
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
            assert_eq!(error.summary.flushes, 1);
            let events: Vec<serde_json::Value> = std::str::from_utf8(&writer.output).unwrap().lines()
                .map(|line| serde_json::from_str(line).unwrap()).collect();
            assert_eq!(events.iter().map(|event| event["event"].as_str().unwrap()).collect::<Vec<_>>(),
                vec!["run_start", "flush", "run_error"]);
        }
    }
}
