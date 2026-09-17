use super::*;
use std::{cell::Cell, io::{self, BufReader, Cursor}, rc::Rc};
use crate::native_engine::decode::DecodeCancellationKind;

struct Control(Rc<Cell<bool>>);
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { self.0.get().then_some(DecodeCancellationKind::Deadline) }
}
struct Guard(Rc<Cell<usize>>);
impl Guard { fn new(alive: &Rc<Cell<usize>>) -> Self { alive.set(alive.get() + 1); Self(Rc::clone(alive)) } }
impl Drop for Guard { fn drop(&mut self) { self.0.set(self.0.get() - 1); } }
#[derive(Serialize)]
struct Payload { ids: Vec<String>, value: f64 }
struct Processor {
    alive: Rc<Cell<usize>>, cancel: Rc<Cell<bool>>, calls: usize, acknowledgements: usize, abandons: usize,
    fail_call: Option<usize>, fail_ack: bool, cancel_after: bool, nan: bool, pending: bool, failed: bool,
    texts: Vec<String>,
}
impl Processor {
    fn new() -> (Self, Control) {
        let cancel = Rc::new(Cell::new(false));
        (Self { alive: Rc::new(Cell::new(0)), cancel: Rc::clone(&cancel), calls: 0, acknowledgements: 0,
            abandons: 0, fail_call: None, fail_ack: false, cancel_after: false, nan: false, pending: false, failed: false,
            texts: Vec::new() }, Control(cancel))
    }
}
impl EntitySnapshotProcessor for Processor {
    type Output = GuardedOutput<Payload, Guard>;
    type Error = &'static str;
    fn execute_snapshot<C: DecodeStepControl>(&mut self, mut documents: Vec<EntityDocument>, _: &mut C) -> Result<Self::Output, Self::Error> {
        if self.pending || self.failed { return Err("not ready"); }
        assert_eq!(self.alive.get(), 0); self.calls += 1;
        if self.fail_call == Some(self.calls) { return Err("fixture native failure"); }
        documents.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        self.texts.extend(documents.iter().map(|d| d.text.clone()));
        let ids = documents.into_iter().map(|d| d.id).collect(); self.pending = true;
        if self.cancel_after { self.cancel.set(true); }
        Ok(GuardedOutput::new(Payload { ids, value: if self.nan { f64::NAN } else { 0.0 } }, Guard::new(&self.alive)))
    }
    fn acknowledge_snapshot(&mut self) -> Result<(), Self::Error> {
        assert!(self.pending); assert_eq!(self.alive.get(), 1);
        if self.fail_ack { return Err("fixture acknowledgement failure"); }
        self.acknowledgements += 1; self.pending = false; Ok(())
    }
    fn abandon_snapshot(&mut self) { self.abandons += 1; self.failed = true; }
}
const A: &str = "{\"id\":\"a\",\"text\":\"Alice\"}\n";
const B: &str = "{\"id\":\"b\",\"text\":\"Bob\"}\n";
const FLUSH: &str = "{\"flush\":true}\n";
fn run(text: &[u8], p: &mut Processor, c: &mut Control, l: EntityStreamLimits) -> (Result<EntityStreamSummary, EntityStreamError<&'static str>>, Vec<u8>) {
    let mut output = Vec::new(); let result = run_ndjson(&mut Cursor::new(text), &mut output, p, l, c); (result, output)
}
#[test]
fn flush_and_eof_execute_complete_raw_snapshots_and_acknowledge_delivery() {
    let (mut p, mut c) = Processor::new(); let text = format!("{B}{A}{FLUSH}{A}");
    let (result, output) = run(text.as_bytes(), &mut p, &mut c, EntityStreamLimits::default());
    let summary = result.unwrap(); assert_eq!((summary.records, summary.snapshots, summary.documents), (4, 2, 3));
    assert_eq!(p.acknowledgements, 2); assert_eq!(p.alive.get(), 0);
    let lines: Vec<serde_json::Value> = output.split(|&b| b == b'\n').filter(|l| !l.is_empty()).map(|l| serde_json::from_slice(l).unwrap()).collect();
    assert_eq!(lines[0]["result"]["ids"], serde_json::json!(["a", "b"]));
    assert_eq!(lines[0]["through_request_seq"], 3); assert_eq!(lines[0]["eof"], false);
    assert_eq!(lines[1]["epoch"], 2); assert_eq!(lines[1]["eof"], true);
    assert_eq!(lines[0]["protocol"], ENTITY_STREAM_PROTOCOL);
}
#[test]
fn empty_stream_does_not_infer_but_explicit_empty_flush_does() {
    let (mut p, mut c) = Processor::new(); let (r, b) = run(b"\n\r\n", &mut p, &mut c, EntityStreamLimits::default());
    assert_eq!(r.unwrap().snapshots, 0); assert!(b.is_empty()); assert_eq!(p.calls, 0);
    let (r, _) = run(FLUSH.as_bytes(), &mut p, &mut c, EntityStreamLimits::default());
    assert_eq!(r.unwrap().documents, 0); assert_eq!(p.calls, 1); assert_eq!(p.acknowledgements, 1);
}
#[test]
fn malformed_current_snapshot_is_never_silently_reduced() {
    for bad in ["{\"id\":\"b\",\"text\":\"x\",\"mentions\":[]}", "{\"id\":\"b\",\"text\":\"x\",\"text\":\"y\"}",
        "{\"flush\":false}", "{\"flush\":true,\"extra\":1}", "null", " ", "{\"id\":\"b\",\"text\":3}", "{\"id\":\"a\",\"text\":\"duplicate\"}"] {
        let (mut p, mut c) = Processor::new(); let text = format!("{A}{bad}\n{FLUSH}");
        let (r, b) = run(text.as_bytes(), &mut p, &mut c, EntityStreamLimits::default());
        assert!(matches!(r, Err(EntityStreamError::InvalidRecord))); assert!(b.is_empty()); assert_eq!(p.calls, 0);
    }
}
#[test]
fn earlier_acknowledged_snapshot_survives_a_later_native_failure() {
    let (mut p, mut c) = Processor::new(); p.fail_call = Some(2);
    let (r, b) = run(format!("{A}{FLUSH}{B}{FLUSH}{A}").as_bytes(), &mut p, &mut c, EntityStreamLimits::default());
    assert!(matches!(r, Err(EntityStreamError::Handler(_)))); assert_eq!(p.calls, 2); assert_eq!(p.acknowledgements, 1);
    assert_eq!(p.abandons, 1); assert_eq!(b.iter().filter(|&&b| b == b'\n').count(), 1); assert_eq!(p.alive.get(), 0);
}
#[test]
fn utf8_crlf_and_eof_framing_are_independent_of_read_buffer_size() {
    let text = "{\"id\":\"é\",\"text\":\"é\\r\\n𐐀\"}\r\n{\"id\":\"b\",\"text\":\"\"}";
    let mut expected = None;
    for size in 1..=text.len() {
        let (mut p, mut c) = Processor::new(); let mut out = Vec::new();
        let mut reader = BufReader::with_capacity(size, Cursor::new(text.as_bytes()));
        let summary = run_ndjson(&mut reader, &mut out, &mut p, EntityStreamLimits::default(), &mut c).unwrap();
        assert_eq!(summary.input_bytes, text.len() as u64); assert_eq!(p.texts, vec!["".to_owned(), "é\r\n𐐀".to_owned()]);
        if let Some(bytes) = &expected { assert_eq!(&out, bytes); } else { expected = Some(out); }
    }
}
#[test]
fn input_limits_stop_before_any_snapshot_execution() {
    for axis in 0..5 {
        let (mut p, mut c) = Processor::new(); let mut l = EntityStreamLimits::default();
        match axis { 0 => l.max_line_bytes = A.len() - 2, 1 => l.max_document_bytes = 4,
            2 => l.max_corpus_bytes = 9, 3 => l.max_documents = 1, _ => l.max_records = 1 }
        let (r, b) = run(format!("{A}{B}{FLUSH}").as_bytes(), &mut p, &mut c, l);
        assert!(r.is_err()); assert!(b.is_empty()); assert_eq!(p.calls, 0);
    }
    let (mut p, mut c) = Processor::new(); let l = EntityStreamLimits { max_input_bytes: 2, ..EntityStreamLimits::default() };
    let (r, _) = run(b"\n\n\n", &mut p, &mut c, l); assert!(matches!(r, Err(EntityStreamError::InputLimit))); assert_eq!(p.calls, 0);
}
struct Writer { alive: Rc<Cell<usize>>, bytes: Vec<u8>, mode: u8, writes: usize }
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        assert_eq!(self.alive.get(), 1); self.writes += 1;
        if self.mode == 1 { return Ok(0); }
        if self.mode == 2 && self.writes > 1 { return Err(io::Error::other("fixture partial write")); }
        let n = if self.mode == 2 { bytes.len().min(3) } else { bytes.len() }; self.bytes.extend_from_slice(&bytes[..n]); Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        assert_eq!(self.alive.get(), 1);
        if self.mode == 3 { Err(io::Error::other("fixture flush")) } else { Ok(()) }
    }
}
#[test]
fn write_zero_partial_write_and_flush_errors_never_acknowledge_or_retry() {
    for mode in 1..=3 {
        let (mut p, mut c) = Processor::new(); let text = format!("{A}{FLUSH}{B}{FLUSH}");
        let mut reader = Cursor::new(text.as_bytes());
        let mut writer = Writer { alive: Rc::clone(&p.alive), bytes: Vec::new(), mode, writes: 0 };
        let r = run_ndjson(&mut reader, &mut writer, &mut p, EntityStreamLimits::default(), &mut c);
        assert!(matches!(r, Err(EntityStreamError::OutputIo))); assert_eq!(p.calls, 1);
        assert_eq!(p.acknowledgements, 0); assert_eq!(p.abandons, 1); assert_eq!(p.alive.get(), 0);
        assert_eq!(reader.position(), (A.len() + FLUSH.len()) as u64);
        if mode == 2 { assert_eq!(writer.bytes.len(), 3); }
        assert!(p.execute_snapshot(Vec::new(), &mut c).is_err());
    }
}
#[test]
fn cancelled_or_nonfinite_output_is_abandoned_without_output_bytes() {
    for nan in [false, true] {
        let (mut p, mut c) = Processor::new(); p.nan = nan; p.cancel_after = !nan;
        let (r, b) = run(A.as_bytes(), &mut p, &mut c, EntityStreamLimits::default());
        assert!(r.is_err()); assert!(b.is_empty()); assert_eq!(p.acknowledgements, 0); assert_eq!(p.abandons, 1); assert_eq!(p.alive.get(), 0);
        if !nan { assert!(matches!(r, Err(EntityStreamError::Resolution(ResolveError::Cancelled(DecodeCancellationKind::Deadline))))); }
    }
}
#[test]
fn complete_output_limits_are_checked_before_writing_any_prefix() {
    for line in [false, true] {
        let (mut p, mut c) = Processor::new(); let mut limits = EntityStreamLimits::default();
        if line { limits.max_output_line_bytes = 2; } else { limits.max_output_bytes = 2; }
        let (r, bytes) = run(A.as_bytes(), &mut p, &mut c, limits);
        assert!(r.is_err()); assert!(bytes.is_empty()); assert_eq!(p.calls, 1); assert_eq!(p.acknowledgements, 0); assert_eq!(p.abandons, 1);
    }
}
fn budget(n: u64) -> EntityStreamBudget {
    EntityStreamBudget { work: BatchWork { forward_positions: n, projected_logits: n * 10 }, mask_visits: n,
        verification: GroundingBudget { max_fields: n as usize, max_matches: n as usize, max_scan_steps: n } }
}
fn used(n: usize) -> EntityVerificationWork { EntityVerificationWork { fields: n, matches: n, scan_steps: n as u64 } }
fn ledger() -> Ledger { Ledger { remaining: budget(100), state: State::Ready } }
#[test]
fn ledger_returns_only_unspent_allowance_and_only_after_delivery_ack() {
    let mut l = ledger(); l.begin(budget(40)).unwrap(); assert_eq!(l.remaining, budget(60));
    l.complete(budget(10).work, used(5)).unwrap(); assert_eq!(l.remaining, budget(60)); assert!(l.ready().is_err());
    l.acknowledge().unwrap(); assert!(l.ready().is_ok());
    assert_eq!(l.remaining.work, budget(90).work); assert_eq!(l.remaining.mask_visits, 60);
    assert_eq!(l.remaining.verification, budget(95).verification);
}
#[test]
fn reservation_failure_is_atomic_on_every_budget_axis() {
    for axis in 0..6 {
        let mut l = ledger(); let mut charge = budget(40);
        match axis { 0 => charge.work.forward_positions = 101, 1 => charge.work.projected_logits = 1001,
            2 => charge.mask_visits = 101, 3 => charge.verification.max_fields = 101,
            4 => charge.verification.max_matches = 101, _ => charge.verification.max_scan_steps = 101 }
        assert!(l.begin(charge).is_err()); assert_eq!(l.remaining, budget(100)); assert_eq!(l.state, State::Ready);
    }
}
#[test]
fn abandoned_or_failed_native_execution_cannot_restore_budget_or_restart() {
    for completed in [false, true] {
        let mut l = ledger(); l.begin(budget(40)).unwrap();
        if completed { l.complete(budget(10).work, used(5)).unwrap(); }
        l.abandon(); assert_eq!(l.remaining, budget(60)); assert!(l.acknowledge().is_err());
        assert!(l.begin(budget(1)).is_err()); assert_eq!(l.remaining, budget(60));
    }
}
#[test]
fn impossible_usage_receipt_poisoning_cannot_mint_refunds() {
    for axis in 0..4 {
        let mut l = ledger(); l.begin(budget(40)).unwrap(); let mut work = budget(10).work; let mut v = used(5);
        match axis { 0 => work.projected_logits = 401, 1 => v.fields = 41, 2 => v.matches = 41, _ => v.scan_steps = 41 }
        assert!(l.complete(work, v).is_err()); assert_eq!(l.state, State::Failed);
        assert_eq!(l.remaining, budget(60)); assert!(l.acknowledge().is_err());
    }
}
#[test]
fn successful_then_failed_snapshots_keep_both_attempts_charged() {
    let mut l = ledger(); l.begin(budget(40)).unwrap(); l.complete(budget(10).work, used(5)).unwrap(); l.acknowledge().unwrap();
    l.begin(budget(40)).unwrap(); l.abandon();
    assert_eq!(l.remaining.work, budget(50).work); assert_eq!(l.remaining.mask_visits, 20);
    assert_eq!(l.remaining.verification, budget(55).verification);
}
#[test]
fn acknowledgement_failure_after_flush_never_starts_another_snapshot() {
    let (mut p, mut c) = Processor::new(); p.fail_ack = true;
    let text = format!("{A}{FLUSH}{B}{FLUSH}"); let mut reader = Cursor::new(text.as_bytes());
    let mut writer = Writer { alive: Rc::clone(&p.alive), bytes: Vec::new(), mode: 0, writes: 0 };
    let r = run_ndjson(&mut reader, &mut writer, &mut p, EntityStreamLimits::default(), &mut c);
    assert!(matches!(r, Err(EntityStreamError::Handler(_))));
    assert_eq!(p.calls, 1); assert_eq!(p.acknowledgements, 0); assert_eq!(p.abandons, 1); assert_eq!(p.alive.get(), 0);
    assert_eq!(reader.position(), (A.len() + FLUSH.len()) as u64);
    assert_eq!(writer.bytes.iter().filter(|&&b| b == b'\n').count(), 1);
    assert!(p.execute_snapshot(Vec::new(), &mut c).is_err());
}
