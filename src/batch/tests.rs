//! Synthetic transport/provider regressions, not native-model qualification.
use super::*;
use std::{cell::Cell, io::{self, BufReader, Cursor}, rc::Rc};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Args { fail: bool }
struct Processor { calls: usize }
impl BatchProcessor for Processor {
    type Args = Args; type Prepared = BatchDocument<Args>; type Output = String;
    fn prepare(&mut self, doc: Self::Prepared) -> Result<Self::Prepared, BatchItemFailure> { Ok(doc) }
    fn planned_work(&self, _: &Self::Prepared) -> BatchWork { BatchWork { forward_positions: 2, projected_logits: 3 } }
    fn execute<C: DecodeStepControl>(&mut self, doc: Self::Prepared, _: &mut C) -> Result<String, BatchItemFailure> {
        self.calls += 1;
        if doc.task_args.is_some_and(|a| a.fail) { Err(BatchItemFailure::reject(BatchCode::Execution)) } else { Ok(doc.text) }
    }
}
struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
fn run(input: &[u8], limits: BatchLimits) -> (Result<BatchSummary, BatchRunError>, Vec<serde_json::Value>, usize) {
    let mut input = BufReader::with_capacity(3, Cursor::new(input));
    let mut output = Vec::new(); let mut processor = Processor { calls: 0 };
    let result = run_ndjson(&mut input, &mut output, &mut processor, limits, &mut Continue);
    let lines = std::str::from_utf8(&output).unwrap().lines().map(|line| canonjson::parse_str(line).unwrap()).collect();
    (result, lines, processor.calls)
}
#[test]
fn framing_preserves_unicode_and_last_unterminated_record() {
    let input = "\n\r\n{\"id\":\"one\",\"text\":\"é\\n上海\"}\r\n{\"id\":\"two\",\"text\":\"tail\"}";
    let (result, events, calls) = run(input.as_bytes(), BatchLimits::default());
    let summary = result.unwrap(); assert_eq!(calls, 2); assert_eq!(summary.requests, 2); assert_eq!(summary.input_lines, 4);
    assert_eq!(events[1]["request_seq"], 1); assert_eq!(events[1]["input_line"], 3);
    assert_eq!(events[1]["result"], "é\n上海"); assert_eq!(events.last().unwrap()["event"], "run_complete");
}
#[test]
fn duplicate_ids_remain_reserved_after_failure_until_acknowledged_flush() {
    let input = b"{\"id\":\"x\",\"text\":\"a\",\"task_args\":{\"fail\":true}}\n{\"id\":\"x\",\"text\":\"b\"}\n{\"flush\":true}\n{\"id\":\"x\",\"text\":\"c\"}\n";
    let (result, events, calls) = run(input, BatchLimits::default());
    let summary = result.unwrap(); assert_eq!(calls, 2); assert_eq!(summary.failed, 2); assert_eq!(summary.succeeded, 1);
    assert_eq!(events[2]["error"]["code"], "duplicate_id"); assert_eq!(events[4]["epoch"], 2);
    assert_eq!(events[4]["request_seq"], 4); assert_eq!(summary.reserved_work.forward_positions, 4);
}
#[test]
fn malformed_utf8_duplicate_keys_and_unknown_fields_get_one_error_each() {
    let input = b"\xff\n{\"id\":\"x\",\"id\":\"y\",\"text\":\"private\"}\n{\"id\":\"z\",\"text\":\"private\",\"tools\":[]}\n \n";
    let (result, events, calls) = run(input, BatchLimits::default());
    assert_eq!(result.unwrap().failed, 4); assert_eq!(calls, 0);
    for (i, event) in events.iter().filter(|e| e["event"] == "doc_error").enumerate() {
        assert_eq!(event["request_seq"], i + 1); assert!(event.get("caller_id").is_none());
        assert!(!event.to_string().contains("private"));
    }
}
#[test]
fn oversized_record_is_drained_once_and_next_record_is_not_lost() {
    let mut input = vec![b'x'; 1000]; input.extend_from_slice(b"\n{\"id\":\"ok\",\"text\":\"yes\"}\n");
    let limits = BatchLimits { max_line_bytes: 64, max_document_bytes: 32, ..BatchLimits::default() };
    let (result, events, calls) = run(&input, limits);
    assert_eq!(result.unwrap().failed, 1); assert_eq!(calls, 1);
    assert_eq!(events[1]["error"]["code"], "line_limit"); assert_eq!(events[2]["request_seq"], 2);
}
#[test]
fn endless_oversized_input_hits_total_byte_budget_without_model_work() {
    let limits = BatchLimits { max_line_bytes: 8, max_document_bytes: 8, max_input_bytes: 64, ..BatchLimits::default() };
    let (result, events, calls) = run(&[b'x'; 1000], limits);
    assert_eq!(result.unwrap_err().fault.code, BatchCode::InputLimit); assert_eq!(calls, 0);
    assert_eq!(events.last().unwrap()["event"], "run_error");
}
#[test]
fn epoch_id_cap_is_not_renewed_by_failed_documents() {
    let input = b"{\"id\":\"a\",\"text\":\"a\"}\n{\"id\":\"b\",\"text\":\"b\"}\n{\"flush\":true}\n{\"id\":\"b\",\"text\":\"b\"}\n";
    let limits = BatchLimits { max_epoch_ids: 1, ..BatchLimits::default() };
    let (result, events, calls) = run(input, limits);
    assert_eq!(result.unwrap().failed, 1); assert_eq!(calls, 2); assert_eq!(events[2]["error"]["code"], "epoch_id_limit");
}
#[test]
fn failed_attempt_spends_work_and_later_request_cannot_renew_it() {
    let input = b"{\"id\":\"a\",\"text\":\"a\",\"task_args\":{\"fail\":true}}\n{\"id\":\"b\",\"text\":\"b\"}\n";
    let limits = BatchLimits { max_work: BatchWork { forward_positions: 2, projected_logits: 3 }, ..BatchLimits::default() };
    let (result, events, calls) = run(input, limits);
    assert_eq!(calls, 1); assert_eq!(result.unwrap().failed, 2); assert_eq!(events[2]["error"]["code"], "work_limit");
}
#[test]
fn writer_failure_stops_reads_and_no_terminal_is_appended_to_partial_line() {
    struct Fails { bytes: Vec<u8>, flushes: usize }
    impl Write for Fails {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.flushes == 1 {
                self.bytes.extend_from_slice(&bytes[..bytes.len().min(7)]);
                return Err(io::Error::other("private sink detail"));
            }
            self.bytes.extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { self.flushes += 1; Ok(()) }
    }
    let first = b"{\"id\":\"a\",\"text\":\"a\"}\n"; let mut input = first.to_vec(); input.extend_from_slice(first);
    let mut reader = Cursor::new(input); let mut writer = Fails { bytes: Vec::new(), flushes: 0 }; let mut processor = Processor { calls: 0 };
    let error = run_ndjson(&mut reader, &mut writer, &mut processor, BatchLimits::default(), &mut Continue).unwrap_err();
    assert_eq!(error.fault.code, BatchCode::OutputIo); assert_eq!(processor.calls, 1); assert_eq!(reader.position(), first.len() as u64);
    assert!(!String::from_utf8(writer.bytes).unwrap().contains("run_error"));
}
#[test]
fn cancellation_before_admission_is_terminal_and_keeps_the_original_cause() {
    struct Stop;
    impl DecodeStepControl for Stop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
        fn prefill_checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(DecodeCancellationKind::Deadline) }
    }
    let mut input = Cursor::new(b"{\"id\":\"a\",\"text\":\"a\"}"); let mut output = Vec::new(); let mut processor = Processor { calls: 0 };
    let error = run_ndjson(&mut input, &mut output, &mut processor, BatchLimits::default(), &mut Stop).unwrap_err();
    assert_eq!(error.fault.cancellation, Some(DecodeCancellationKind::Deadline)); assert_eq!(input.position(), 0); assert_eq!(processor.calls, 0);
}
#[test]
fn ordered_replay_is_byte_identical_without_volatile_metadata() {
    let input = b"{\"id\":\"a\",\"text\":\"a\"}\n{\"flush\":true}\n";
    let (_, first, _) = run(input, BatchLimits::default()); let (_, second, _) = run(input, BatchLimits::default());
    assert_eq!(first, second);
    for event in first { let bytes = canonjson::canonical_string(&event).unwrap(); assert!(!bytes.contains("digest")); assert!(!bytes.contains("timestamp")); }
}

const CANCELLATION_KINDS: [DecodeCancellationKind; 11] = [
    DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
    DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota,
    DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
    DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
    DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown,
    DecodeCancellationKind::LinkedExit,
];
const ITEM_WORK: BatchWork = BatchWork { forward_positions: 2, projected_logits: 3 };
const FIRST_ITEM: &[u8] = b"{\"id\":\"first\",\"text\":\"first\"}\n";
fn two_items() -> Vec<u8> {
    let mut bytes = FIRST_ITEM.to_vec();
    bytes.extend_from_slice(b"{\"id\":\"second\",\"text\":\"second\"}\n"); bytes
}
fn decoded_events(output: &[u8]) -> Vec<serde_json::Value> {
    std::str::from_utf8(output).unwrap().lines().map(|line| canonjson::parse_str(line).unwrap()).collect()
}

struct FailingProcessor {
    failure: BatchItemFailure, in_prepare: bool, prepared: usize, executed: usize,
}
impl BatchProcessor for FailingProcessor {
    type Args = (); type Prepared = String; type Output = String;
    fn prepare(&mut self, document: BatchDocument<()>) -> Result<String, BatchItemFailure> {
        self.prepared += 1;
        if self.in_prepare && self.prepared == 1 { Err(self.failure) } else { Ok(document.text) }
    }
    fn planned_work(&self, _: &String) -> BatchWork { ITEM_WORK }
    fn execute<C: DecodeStepControl>(&mut self, text: String, _: &mut C) -> Result<String, BatchItemFailure> {
        self.executed += 1;
        if !self.in_prepare && self.executed == 1 { Err(self.failure) } else { Ok(text) }
    }
}
fn assert_terminal_processor_failure(failure: BatchItemFailure, in_prepare: bool) {
    let mut reader = Cursor::new(two_items()); let mut output = Vec::new();
    let mut processor = FailingProcessor { failure, in_prepare, prepared: 0, executed: 0 };
    let error = run_ndjson(&mut reader, &mut output, &mut processor, BatchLimits::default(), &mut Continue).unwrap_err();
    assert_eq!(error.fault, failure.fault);
    assert_eq!(reader.position(), FIRST_ITEM.len() as u64, "a fatal processor failure must stop input admission");
    assert_eq!(processor.prepared, 1); assert_eq!(processor.executed, usize::from(!in_prepare));
    assert_eq!(error.summary.requests, 1); assert_eq!(error.summary.failed, 1); assert_eq!(error.summary.succeeded, 0);
    assert_eq!(error.summary.reserved_work, if in_prepare { BatchWork::default() } else { ITEM_WORK });
    let events = decoded_events(&output);
    assert_eq!(events.iter().map(|event| event["event"].as_str().unwrap()).collect::<Vec<_>>(),
        vec!["run_start", "doc_error", "run_error"]);
    for event in &events[1..] {
        assert_eq!(event["error"], serde_json::to_value(failure.fault).unwrap());
        assert_eq!(event["request_seq"], 1);
    }
}

#[test]
fn typed_cancellation_and_broken_state_cannot_be_made_recoverable_by_the_processor() {
    let mut faults = CANCELLATION_KINDS.map(BatchFault::cancelled).to_vec();
    faults.extend([BatchCode::Cancelled.into(), BatchCode::InputIo.into(),
        BatchCode::OutputIo.into(), BatchCode::InvalidExecution.into(),
        BatchFault { code: BatchCode::Planning, cancellation: Some(DecodeCancellationKind::ParentCancelled) }]);
    for fault in faults {
        for in_prepare in [true, false] {
            assert_terminal_processor_failure(BatchItemFailure::reject(fault), in_prepare);
        }
    }
}

#[test]
fn explicit_host_fatality_still_stops_otherwise_recoverable_categories() {
    for code in [BatchCode::Planning, BatchCode::Admission, BatchCode::WorkLimit, BatchCode::Execution] {
        for in_prepare in [true, false] {
            assert_terminal_processor_failure(BatchItemFailure::fatal(code), in_prepare);
        }
    }
}

#[test]
fn recoverable_planning_admission_and_work_failures_continue_without_refunding_attempts() {
    for code in [BatchCode::Planning, BatchCode::Admission, BatchCode::WorkLimit] {
        for in_prepare in [true, false] {
            let input = two_items(); let mut reader = Cursor::new(&input); let mut output = Vec::new();
            let mut processor = FailingProcessor {
                failure: BatchItemFailure::reject(code), in_prepare, prepared: 0, executed: 0,
            };
            let summary = run_ndjson(&mut reader, &mut output, &mut processor, BatchLimits::default(), &mut Continue).unwrap();
            assert_eq!(reader.position(), input.len() as u64);
            assert_eq!(processor.prepared, 2); assert_eq!(processor.executed, if in_prepare { 1 } else { 2 });
            assert_eq!(summary.failed, 1); assert_eq!(summary.succeeded, 1);
            assert_eq!(summary.reserved_work, if in_prepare { ITEM_WORK } else { ITEM_WORK.add(ITEM_WORK).unwrap() });
            let events = decoded_events(&output);
            assert_eq!(events[1]["event"], "doc_error"); assert_eq!(events[2]["event"], "doc");
            assert_eq!(events[2]["result"], "second"); assert_eq!(events.last().unwrap()["event"], "run_complete");
        }
    }
}

#[derive(Default)]
struct PublicationState {
    cancellation: Cell<Option<DecodeCancellationKind>>, payload_live: Cell<bool>, guard_live: Cell<bool>,
    checked_before_publication: Cell<bool>, serialized: Cell<bool>,
}
struct CompletedPayload(Rc<PublicationState>);
impl Serialize for CompletedPayload {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialized.set(true); serializer.serialize_str("completed private result")
    }
}
impl Drop for CompletedPayload {
    fn drop(&mut self) {
        assert!(self.0.guard_live.get(), "result storage must remain admitted until it is dropped");
        self.0.payload_live.set(false);
    }
}
struct PublicationGuard(Rc<PublicationState>);
impl Drop for PublicationGuard {
    fn drop(&mut self) {
        assert!(!self.0.payload_live.get(), "result storage must drop before its reservation");
        self.0.guard_live.set(false);
    }
}
struct CancelOnReturn { state: Rc<PublicationState>, cause: DecodeCancellationKind, calls: usize }
impl BatchProcessor for CancelOnReturn {
    type Args = (); type Prepared = (); type Output = output::GuardedOutput<CompletedPayload, PublicationGuard>;
    fn prepare(&mut self, _: BatchDocument<()>) -> Result<(), BatchItemFailure> { Ok(()) }
    fn planned_work(&self, _: &()) -> BatchWork { ITEM_WORK }
    fn execute<C: DecodeStepControl>(&mut self, _: (), _: &mut C) -> Result<Self::Output, BatchItemFailure> {
        self.calls += 1;
        self.state.payload_live.set(true); self.state.guard_live.set(true);
        self.state.cancellation.set(Some(self.cause));
        Ok(output::GuardedOutput::new(CompletedPayload(Rc::clone(&self.state)), PublicationGuard(Rc::clone(&self.state))))
    }
}
struct PublicationControl(Rc<PublicationState>);
impl DecodeStepControl for PublicationControl {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if self.0.cancellation.get().is_some() {
            assert!(self.0.payload_live.get()); assert!(self.0.guard_live.get());
            assert!(!self.0.serialized.get(), "cancellation must precede successful result serialization");
            self.0.checked_before_publication.set(true);
        }
        self.0.cancellation.get()
    }
}

#[test]
fn cancellation_after_execution_keeps_the_guard_until_refusal_and_never_publishes_success() {
    for cause in CANCELLATION_KINDS {
        let state = Rc::new(PublicationState::default());
        let mut processor = CancelOnReturn { state: Rc::clone(&state), cause, calls: 0 };
        let mut control = PublicationControl(Rc::clone(&state));
        let mut reader = Cursor::new(two_items()); let mut output = Vec::new();
        let error = run_ndjson(&mut reader, &mut output, &mut processor, BatchLimits::default(), &mut control).unwrap_err();
        assert_eq!(error.fault, BatchFault::cancelled(cause));
        assert_eq!(reader.position(), FIRST_ITEM.len() as u64); assert_eq!(processor.calls, 1);
        assert_eq!(error.summary.requests, 1); assert_eq!(error.summary.succeeded, 0); assert_eq!(error.summary.failed, 0);
        assert_eq!(error.summary.reserved_work, ITEM_WORK);
        assert!(state.checked_before_publication.get()); assert!(!state.serialized.get());
        assert!(!state.payload_live.get()); assert!(!state.guard_live.get());
        let events = decoded_events(&output);
        assert_eq!(events.iter().map(|event| event["event"].as_str().unwrap()).collect::<Vec<_>>(), vec!["run_start", "run_error"]);
        assert_eq!(events[1]["error"], serde_json::to_value(BatchFault::cancelled(cause)).unwrap());
    }
}
