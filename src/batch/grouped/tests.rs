//! Transport execution tests with a deterministic grouped processor. These do
//! not load weights, fabricate production admission, or certify native parity.
use super::*;
use std::{cell::{Cell, RefCell}, io::Cursor, rc::Rc};
use serde_json::Value;

#[derive(Default)]
struct State {
    leases: Cell<usize>, payloads: Cell<usize>, groups: RefCell<Vec<Vec<u64>>>, cancel: Cell<bool>,
}
struct Lease(Rc<State>);
impl Drop for Lease {
    fn drop(&mut self) {
        assert_eq!(self.0.payloads.get(), 0, "results must drop BEFORE their cohort reservation");
        self.0.leases.set(self.0.leases.get() - 1);
    }
}
struct Payload { text: String, state: Rc<State> }
impl Serialize for Payload {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> { self.text.serialize(s) }
}
impl Drop for Payload {
    fn drop(&mut self) { self.state.payloads.set(self.state.payloads.get() - 1); }
}
struct Processor {
    state: Rc<State>, fail_row: Option<u64>, fatal_row: Option<u64>, wrong_route: bool,
    huge_first: bool, cancel_after_execute: bool,
}
impl Processor {
    fn new(state: Rc<State>) -> Self {
        Self { state, fail_row: None, fatal_row: None, wrong_route: false, huge_first: false, cancel_after_execute: false }
    }
}
impl GroupedBatchProcessor for Processor {
    type Args = (); type Prepared = String; type Output = Payload; type Guard = Lease;
    fn max_group_size(&self) -> usize { 4 }
    fn prepare(&mut self, document: BatchDocument<()>) -> Result<String, BatchItemFailure> {
        match document.text.as_str() {
            "reject" => Err(BatchItemFailure::reject(BatchCode::Planning)),
            "fatal" => Err(BatchItemFailure::fatal(BatchCode::Admission)),
            _ => Ok(document.text),
        }
    }
    fn planned_work(&self, _: &String) -> Result<BatchWork, BatchItemFailure> {
        Ok(BatchWork { forward_positions: 1, projected_logits: 10 })
    }
    fn execute_group<C: DecodeStepControl>(&mut self, input: Vec<GroupedRequest<String>>, _: &mut C)
        -> Result<GroupedOutput<Payload, Lease>, BatchFault> {
        self.state.groups.borrow_mut().push(input.iter().map(|r| r.context.request_seq).collect());
        assert_eq!(self.state.leases.get(), 0, "the previous cohort must drain first");
        self.state.leases.set(1); let guard = Lease(Rc::clone(&self.state));
        let mut rows = Vec::new();
        for request in input {
            let seq = request.context.request_seq;
            let result = if self.fail_row == Some(seq) { Err(BatchItemFailure::reject(BatchCode::Execution)) }
                else if self.fatal_row == Some(seq) { Err(BatchItemFailure::fatal(BatchCode::Execution)) }
                else {
                    self.state.payloads.set(self.state.payloads.get() + 1);
                    Ok(Payload { text: if self.huge_first && seq == 1 { "x".repeat(5000) } else { request.prepared },
                        state: Rc::clone(&self.state) })
                };
            rows.push(GroupedRow { request_seq: seq + u64::from(self.wrong_route), result });
        }
        if self.cancel_after_execute { self.state.cancel.set(true); }
        Ok(GroupedOutput::new(rows, guard))
    }
}
struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
struct Control(Rc<State>);
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.0.cancel.get().then_some(DecodeCancellationKind::Deadline)
    }
}
struct Writer {
    bytes: Vec<u8>, state: Rc<State>, doc_pending: bool, fail_doc_flush: bool, panic_doc: bool,
}
impl Writer {
    fn new(state: Rc<State>) -> Self {
        Self { bytes: Vec::new(), state, doc_pending: false, fail_doc_flush: false, panic_doc: false }
    }
}
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let event: Value = serde_json::from_slice(bytes).unwrap();
        self.doc_pending = event["event"] == "doc";
        if self.doc_pending {
            assert_eq!(self.state.leases.get(), 1, "lease must survive serialization and writing");
            assert!(self.state.payloads.get() > 0);
            assert!(!self.panic_doc, "synthetic writer panic");
        }
        self.bytes.extend_from_slice(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        if self.doc_pending {
            assert_eq!(self.state.leases.get(), 1, "lease must survive the physical flush");
            if self.fail_doc_flush { return Err(std::io::Error::other("synthetic flush failure")); }
        }
        self.doc_pending = false; Ok(())
    }
}
fn record(id: &str, text: &str) -> String { format!("{}\n", serde_json::json!({"id":id,"text":text})) }
fn records(count: usize) -> String { (0..count).map(|i| record(&format!("id-{i}"), &format!("row-{i}"))).collect() }
fn events(bytes: &[u8]) -> Vec<Value> {
    std::str::from_utf8(bytes).unwrap().lines().map(|line| serde_json::from_str(line).unwrap()).collect()
}
fn document_events(events: &[Value]) -> Vec<&Value> {
    events.iter().filter(|e| e["event"] == "doc" || e["event"] == "doc_error").collect()
}
fn execute(input: &str, processor: &mut Processor, width: usize, limits: BatchLimits) -> (BatchSummary, Vec<Value>) {
    let mut writer = Writer::new(Rc::clone(&processor.state));
    let summary = run_ndjson(&mut Cursor::new(input.as_bytes()), &mut writer, processor, limits,
        GroupLimits { max_records: width }, &mut Continue).unwrap();
    (summary, events(&writer.bytes))
}
#[test]
fn one_group_call_per_window_and_original_order_across_short_final_window() {
    for width in 1..=4 {
        let state = Rc::new(State::default()); let mut processor = Processor::new(Rc::clone(&state));
        let (summary, output) = execute(&records(7), &mut processor, width, BatchLimits::default());
        assert_eq!(summary.succeeded, 7); assert_eq!(summary.failed, 0); assert_eq!(summary.flushes, 1);
        let expected: Vec<Vec<u64>> = (1..=7).collect::<Vec<_>>().chunks(width).map(<[_]>::to_vec).collect();
        assert_eq!(*state.groups.borrow(), expected);
        assert!(output.iter().all(|e| e["execution"] == GROUPED_EXECUTION && e["protocol"] == BATCH_PROTOCOL));
        assert_eq!(document_events(&output).iter().map(|e| e["request_seq"].as_u64().unwrap()).collect::<Vec<_>>(), (1..=7).collect::<Vec<_>>());
        assert_eq!(state.leases.get(), 0); assert_eq!(state.payloads.get(), 0);
    }
}
#[test]
fn parse_errors_keep_their_position_and_do_not_enter_the_native_group() {
    let state = Rc::new(State::default()); let mut processor = Processor::new(Rc::clone(&state));
    let input = format!("{}secret malformed request\n{}", record("a", "first"), record("b", "last"));
    let (summary, output) = execute(&input, &mut processor, 3, BatchLimits::default());
    assert_eq!(*state.groups.borrow(), vec![vec![1, 3]]);
    let docs = document_events(&output); assert_eq!(docs.len(), 3);
    assert_eq!(docs[0]["result"], "first"); assert_eq!(docs[1]["event"], "doc_error");
    assert_eq!(docs[1]["request_seq"], 2); assert_eq!(docs[1]["input_line"], 2); assert_eq!(docs[2]["result"], "last");
    assert!(!serde_json::to_string(&output).unwrap().contains("secret malformed"));
    assert_eq!(summary.failed, 1);
}
#[test]
fn flush_drains_before_epoch_reset_and_duplicate_ids_survive_rejections() {
    let state = Rc::new(State::default()); let mut processor = Processor::new(Rc::clone(&state));
    let input = format!("{}{}{{\"flush\":true}}\n{}{}", record("a", "reject"), record("a", "duplicate"), record("a", "new epoch"), record("b", "other"));
    let (summary, output) = execute(&input, &mut processor, 4, BatchLimits::default());
    assert_eq!(*state.groups.borrow(), vec![vec![4, 5]]);
    let docs = document_events(&output);
    assert_eq!(docs[0]["error"]["code"], "planning"); assert_eq!(docs[1]["error"]["code"], "duplicate_id");
    assert_eq!(docs[2]["epoch"], 2); assert_eq!(docs[2]["request_seq"], 4);
    let explicit = output.iter().position(|e| e["event"] == "flush" && e["eof"] == false).unwrap();
    let next_doc = output.iter().position(|e| e["event"] == "doc").unwrap(); assert!(explicit < next_doc);
    assert_eq!(summary.requests, 5); assert_eq!(summary.failed, 2); assert_eq!(summary.flushes, 2);
}
#[test]
fn aggregate_work_is_reserved_before_execution_and_never_refunded() {
    let state = Rc::new(State::default()); let mut processor = Processor::new(Rc::clone(&state)); processor.fail_row = Some(2);
    let limits = BatchLimits { max_work: BatchWork { forward_positions: 2, projected_logits: 20 }, ..BatchLimits::default() };
    let (summary, output) = execute(&records(3), &mut processor, 3, limits);
    assert_eq!(*state.groups.borrow(), vec![vec![1, 2]]); assert_eq!(summary.reserved_work, limits.max_work);
    assert_eq!(summary.succeeded, 1); assert_eq!(summary.failed, 2);
    let docs = document_events(&output); assert_eq!(docs[1]["error"]["code"], "execution");
    assert_eq!(docs[1]["reserved_work"]["forward_positions"], 1); assert_eq!(docs[2]["error"]["code"], "work_limit");
}
#[test]
fn oversized_complete_result_is_rejected_without_releasing_sibling_reservation() {
    let state = Rc::new(State::default()); let mut processor = Processor::new(Rc::clone(&state)); processor.huge_first = true;
    let (summary, output) = execute(&records(2), &mut processor, 2,
        BatchLimits { max_output_line_bytes: 2048, ..BatchLimits::default() });
    assert_eq!(summary.succeeded, 1); assert_eq!(summary.failed, 1);
    let docs = document_events(&output); assert_eq!(docs[0]["error"]["code"], "output_line_limit"); assert_eq!(docs[1]["event"], "doc");
    assert_eq!(state.leases.get(), 0);
}
#[test]
fn failed_flush_poisoning_stops_read_ahead_and_frees_results_before_guard() {
    let state = Rc::new(State::default()); let mut processor = Processor::new(Rc::clone(&state));
    let input = records(6); let first_window_bytes = records(2).len() as u64;
    let mut reader = Cursor::new(input.as_bytes()); let mut writer = Writer::new(Rc::clone(&state)); writer.fail_doc_flush = true;
    let error = run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(),
        GroupLimits { max_records: 2 }, &mut Continue).unwrap_err();
    assert_eq!(error.fault.code, BatchCode::OutputIo); assert_eq!(reader.position(), first_window_bytes);
    assert_eq!(*state.groups.borrow(), vec![vec![1, 2]]); assert_eq!(error.summary.succeeded, 0);
    assert!(!events(&writer.bytes).iter().any(|e| e["event"] == "run_error"));
    assert_eq!(state.leases.get(), 0); assert_eq!(state.payloads.get(), 0);
}
#[test]
fn foreign_or_reordered_result_coordinates_are_fatal_before_any_document_delivery() {
    let state = Rc::new(State::default()); let mut processor = Processor::new(Rc::clone(&state)); processor.wrong_route = true;
    let mut writer = Writer::new(Rc::clone(&state));
    let error = run_ndjson(&mut Cursor::new(records(2)), &mut writer, &mut processor, BatchLimits::default(),
        GroupLimits { max_records: 2 }, &mut Continue).unwrap_err();
    assert_eq!(error.fault.code, BatchCode::InvalidExecution); assert!(document_events(&events(&writer.bytes)).is_empty());
    assert_eq!(state.leases.get(), 0); assert_eq!(state.payloads.get(), 0);
}
#[test]
fn fatal_row_failure_prevents_later_delivery_and_next_cohort() {
    let state = Rc::new(State::default()); let mut processor = Processor::new(Rc::clone(&state)); processor.fatal_row = Some(1);
    let input = records(4); let mut reader = Cursor::new(input.as_bytes()); let mut writer = Writer::new(Rc::clone(&state));
    let error = run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(),
        GroupLimits { max_records: 2 }, &mut Continue).unwrap_err();
    assert_eq!(error.fault.code, BatchCode::Execution); assert_eq!(reader.position(), records(2).len() as u64);
    let output = events(&writer.bytes); assert_eq!(document_events(&output).len(), 1);
    assert_eq!(output.last().unwrap()["event"], "run_error"); assert_eq!(state.leases.get(), 0);
}
#[test]
fn cancellation_after_native_return_preserves_cause_without_publishing_more_results() {
    let state = Rc::new(State::default()); let mut processor = Processor::new(Rc::clone(&state)); processor.cancel_after_execute = true;
    let mut writer = Writer::new(Rc::clone(&state));
    let error = run_ndjson(&mut Cursor::new(records(4)), &mut writer, &mut processor, BatchLimits::default(),
        GroupLimits { max_records: 2 }, &mut Control(Rc::clone(&state))).unwrap_err();
    assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(error.summary.reserved_work.forward_positions, 2); assert_eq!(error.summary.succeeded, 0);
    assert!(document_events(&events(&writer.bytes)).is_empty()); assert_eq!(state.leases.get(), 0);
}
#[test]
fn fatal_input_failure_discards_queued_work_instead_of_running_after_abort() {
    let state = Rc::new(State::default()); let mut processor = Processor::new(Rc::clone(&state));
    let first = record("a", "queued"); let input = format!("{first}unbounded-second-line");
    let limits = BatchLimits { max_input_bytes: first.len() as u64 + 3, ..BatchLimits::default() };
    let mut writer = Writer::new(Rc::clone(&state));
    let error = run_ndjson(&mut Cursor::new(input), &mut writer, &mut processor, limits,
        GroupLimits { max_records: 3 }, &mut Continue).unwrap_err();
    assert_eq!(error.fault.code, BatchCode::InputLimit); assert!(state.groups.borrow().is_empty());
    assert_eq!(error.summary.reserved_work.forward_positions, 1); assert!(document_events(&events(&writer.bytes)).is_empty());
}
#[test]
fn invalid_width_is_refused_before_io_or_preparation() {
    for width in [0, 5, MAX_GROUP_RECORDS + 1] {
        let state = Rc::new(State::default()); let mut processor = Processor::new(Rc::clone(&state));
        let mut reader = Cursor::new(records(1)); let mut writer = Writer::new(Rc::clone(&state));
        let error = run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(),
            GroupLimits { max_records: width }, &mut Continue).unwrap_err();
        assert_eq!(error.fault.code, BatchCode::InvalidLimits); assert_eq!(reader.position(), 0); assert!(writer.bytes.is_empty());
    }
}
#[test]
fn panic_during_delivery_unwinds_all_cohort_results_then_the_reservation() {
    let state = Rc::new(State::default()); let mut processor = Processor::new(Rc::clone(&state));
    let mut writer = Writer::new(Rc::clone(&state)); writer.panic_doc = true;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = run_ndjson(&mut Cursor::new(records(2)), &mut writer, &mut processor, BatchLimits::default(),
            GroupLimits { max_records: 2 }, &mut Continue);
    }));
    assert!(result.is_err()); assert_eq!(state.leases.get(), 0); assert_eq!(state.payloads.get(), 0);
}
