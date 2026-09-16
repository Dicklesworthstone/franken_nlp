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
