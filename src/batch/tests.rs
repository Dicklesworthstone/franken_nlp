//! Synthetic transport/provider regressions, not native-model qualification.
use super::*;
use std::io::{self, BufReader, Cursor};

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
