//! Retained-input framing and authority fixtures. These do not execute a
//! model, certify filesystem durability, or acknowledge a journal transaction.
use super::*;
use crate::jobs::{JobContract, JobId, MismatchField,
    tests::{Control, error, identity, key, limits}};
use crate::native_engine::decode::DecodeCancellationKind;
use std::{cell::Cell, io::{self, Cursor}, rc::Rc};

const A: &[u8] = b" \t{\"text\":\"private \\u0041lice\",\"id\":\"alpha\", \"task_args\":{\"n\":1}} \r";
const B: &[u8] = br#"{"id":"beta","text":"private Bob"}"#;
fn inputs() -> [JobInput<'static>; 2] {
    [JobInput { id: "alpha", original: A, normalized: A },
        JobInput { id: "beta", original: B, normalized: B }]
}
fn freeze(inputs: &[JobInput<'_>]) -> FrozenManifest {
    FrozenManifest::freeze(&key(), JobContract { job_id: JobId([7; 16]),
        execution: &identity(), recipe: &"retained-envelope-fixture", limits: limits() },
        inputs.iter().map(|input| JobInput { id: input.id,
            original: input.original, normalized: input.normalized }), &mut Control::default()).unwrap()
}
fn retention() -> InputRetentionLimits { InputRetentionLimits { max_spool_bytes: 1 << 20 } }
fn encoded() -> (FrozenManifest, InputReceipt, Vec<u8>) {
    let inputs = inputs(); let manifest = freeze(&inputs);
    let mut spool = Cursor::new(Vec::new());
    let receipt = write(&mut spool, &key(), &manifest, retention(), &inputs,
        &mut Control::default()).unwrap();
    (manifest, receipt, spool.into_inner())
}
fn restored(bytes: &[u8], manifest: &FrozenManifest, receipt: &InputReceipt)
    -> Result<Vec<(String, Vec<u8>)>, JobError> {
    let mut rows = Vec::new();
    read(&mut Cursor::new(bytes), &key(), manifest, receipt, &mut Control::default(),
        |id, bytes| { rows.push((id, bytes)); Ok(()) })?;
    // The partially built population is dropped on ANY error, including the
    // final receipt check, and cannot escape as an authenticated prefix.
    Ok(rows)
}

#[test]
fn exact_envelopes_preserve_whitespace_escape_spelling_and_carriage_returns() {
    let (manifest, receipt, bytes) = encoded();
    let rows = restored(&bytes, &manifest, &receipt).unwrap();
    assert_eq!(rows, vec![("alpha".to_owned(), A.to_vec()), ("beta".to_owned(), B.to_vec())]);
    assert_eq!(receipt.spool_bytes, (2 * HEADER_BYTES + A.len() + B.len()) as u64);
    assert_eq!(&bytes[HEADER_BYTES..HEADER_BYTES + A.len()], A);
    let report = receipt.report();
    assert_eq!(report.items, 2); assert_eq!(report.spool_bytes, bytes.len() as u64);
    let metadata = crate::canonjson::canonical_string(&report).unwrap();
    for private in ["alpha", "beta", "private", "commitment"] { assert!(!metadata.contains(private)); }
}

#[test]
fn complete_spool_cap_is_checked_before_the_first_write() {
    let inputs = inputs(); let manifest = freeze(&inputs);
    let required = preflight(&key(), &manifest, retention(), &inputs, &mut Control::default()).unwrap();
    let mut spool = Cursor::new(Vec::new());
    let cap = InputRetentionLimits { max_spool_bytes: required - 1 };
    assert_eq!(error(write(&mut spool, &key(), &manifest, cap, &inputs,
        &mut Control::default())), JobError::Limit);
    assert!(spool.get_ref().is_empty()); assert_eq!(spool.position(), 0);
    let cap = InputRetentionLimits { max_spool_bytes: required };
    let receipt = write(&mut spool, &key(), &manifest, cap, &inputs, &mut Control::default()).unwrap();
    assert_eq!(receipt.spool_bytes, required);
    assert_eq!(restored(spool.get_ref(), &manifest, &receipt).unwrap().len(), 2);
}

#[test]
fn invalid_retention_limits_and_existing_bytes_never_write_a_prefix() {
    let inputs = inputs(); let manifest = freeze(&inputs);
    for max_spool_bytes in [0, u64::MAX] {
        let mut spool = Cursor::new(Vec::new());
        assert_eq!(error(write(&mut spool, &key(), &manifest, InputRetentionLimits { max_spool_bytes },
            &inputs, &mut Control::default())), JobError::InvalidLimits);
        assert!(spool.get_ref().is_empty());
    }
    let mut spool = Cursor::new(b"previous partial input".to_vec());
    assert_eq!(error(write(&mut spool, &key(), &manifest, retention(), &inputs,
        &mut Control::default())), JobError::AlreadyExists);
    assert_eq!(spool.into_inner(), b"previous partial input");
}

#[test]
fn malformed_or_id_mismatched_envelopes_cannot_gain_a_retention_receipt() {
    for bytes in [b"".as_slice(), b"plain input", b"\xff", br#"{"flush":true}"#,
        br#"{"id":"alpha","text":3}"#, br#"{"id":"beta","text":"wrong embedded ID"}"#,
        br#"{"id":"alpha","id":"alpha","text":"duplicate"}"#,
        br#"{"id":"alpha","text":"extra","unknown":1}"#,
        br#"{"id":"alpha","text":"x","task_args":{"n":1,"n":2}}"#] {
        let inputs = [JobInput { id: "alpha", original: bytes, normalized: bytes }];
        // Generic manifests permit opaque bytes. Retention's stricter replay
        // format must therefore validate the envelope before writing anything.
        let manifest = freeze(&inputs); let mut spool = Cursor::new(Vec::new());
        assert_eq!(error(write(&mut spool, &key(), &manifest, retention(), &inputs,
            &mut Control::default())), JobError::InvalidInput);
        assert!(spool.get_ref().is_empty());
    }
}

#[test]
fn normalized_and_original_population_mismatches_fail_before_any_write() {
    let normalized = [JobInput { id: "alpha", original: A, normalized: B }];
    let manifest = freeze(&normalized); let mut spool = Cursor::new(Vec::new());
    assert_eq!(error(write(&mut spool, &key(), &manifest, retention(), &normalized,
        &mut Control::default())), JobError::InvalidInput);
    assert!(spool.get_ref().is_empty());

    let baseline = inputs(); let manifest = freeze(&baseline);
    let changed = br#"{"id":"alpha","text":"different private bytes"}"#;
    let substituted = [JobInput { id: "alpha", original: changed, normalized: changed },
        JobInput { id: "beta", original: B, normalized: B }];
    assert_eq!(error(write(&mut spool, &key(), &manifest, retention(), &substituted,
        &mut Control::default())), JobError::Mismatch(MismatchField::Population));
    assert!(spool.get_ref().is_empty());
    assert_eq!(error(write(&mut spool, &key(), &manifest, retention(), &baseline[..1],
        &mut Control::default())), JobError::InvalidInput);
    assert!(spool.get_ref().is_empty());
}

#[test]
fn every_header_and_payload_byte_is_authenticated() {
    let (manifest, receipt, original) = encoded();
    for offset in 0..original.len() {
        let mut bytes = original.clone(); bytes[offset] ^= 1;
        assert!(restored(&bytes, &manifest, &receipt).is_err(), "corruption at byte {offset}");
    }
}

#[test]
fn truncated_reordered_and_extra_frames_never_become_a_population() {
    let (manifest, receipt, original) = encoded();
    for end in 0..original.len() {
        assert_eq!(error(restored(&original[..end], &manifest, &receipt)), JobError::Corrupt);
    }
    let split = HEADER_BYTES + A.len();
    let reordered = [&original[split..], &original[..split]].concat();
    assert!(restored(&reordered, &manifest, &receipt).is_err());
    for extra in [b"orphan".as_slice(), &original[..split]] {
        let mut bytes = original.clone(); bytes.extend_from_slice(extra);
        assert_eq!(error(restored(&bytes, &manifest, &receipt)), JobError::Corrupt);
    }
}

#[test]
fn receipt_version_count_size_original_cap_and_final_chain_are_bound() {
    let (manifest, receipt, bytes) = encoded();
    for field in 0..6 {
        let mut changed = receipt.clone();
        match field {
            0 => changed.schema_version += 1,
            1 => changed.items += 1,
            2 => changed.spool_bytes += 1,
            3 => changed.max_spool_bytes += 1,
            4 => changed.max_spool_bytes = changed.spool_bytes - 1,
            _ => changed.commitment.0[0] ^= 1,
        }
        assert!(restored(&bytes, &manifest, &changed).is_err());
    }
    let mut final_mismatch = receipt.clone(); final_mismatch.commitment.0[0] ^= 1;
    let mut visited = 0;
    assert_eq!(error(read(&mut Cursor::new(&bytes), &key(), &manifest, &final_mismatch,
        &mut Control::default(), |_, _| { visited += 1; Ok(()) })), JobError::Authentication);
    assert_eq!(visited, 2);
    assert!(restored(&bytes, &manifest, &final_mismatch).is_err());
}

#[test]
fn frames_cannot_be_replayed_under_another_key_or_frozen_contract() {
    let (manifest, receipt, bytes) = encoded();
    assert_eq!(error(read(&mut Cursor::new(&bytes), &JobSecret::from_bytes([18; 32]),
        &manifest, &receipt, &mut Control::default(), |_, _| Ok(()))), JobError::Authentication);
    for field in 0..5 {
        let mut changed = freeze(&inputs());
        match field {
            0 => changed.binding.job.0[0] ^= 1,
            1 => changed.binding.population.0[0] ^= 1,
            2 => changed.binding.execution.0[0] ^= 1,
            3 => changed.binding.recipe.0[0] ^= 1,
            _ => changed.binding.limits.0[0] ^= 1,
        }
        assert_eq!(error(restored(&bytes, &changed, &receipt)), JobError::Authentication);
    }
}

struct PayloadGuard { cursor: Cursor<Vec<u8>>, allowed_until: u64 }
impl Read for PayloadGuard {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        assert!(self.cursor.position().saturating_add(bytes.len() as u64) <= self.allowed_until,
            "out-of-budget payload read occurred");
        self.cursor.read(bytes)
    }
}
impl Seek for PayloadGuard {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> { self.cursor.seek(position) }
}

#[test]
fn declared_per_record_and_cumulative_snapshot_limits_precede_payload_reads() {
    let (manifest, receipt, bytes) = encoded();
    let mut oversized = bytes.clone(); oversized[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut reader = PayloadGuard { cursor: Cursor::new(oversized), allowed_until: HEADER_BYTES as u64 };
    assert_eq!(error(read(&mut reader, &key(), &manifest, &receipt, &mut Control::default(),
        |_, _| panic!("oversized frame must not escape"))), JobError::Corrupt);

    let mut narrow = freeze(&inputs()); narrow.limits.max_input_bytes_per_item = A.len() - 1;
    let mut reader = PayloadGuard { cursor: Cursor::new(bytes.clone()), allowed_until: HEADER_BYTES as u64 };
    assert_eq!(error(read(&mut reader, &key(), &narrow, &receipt, &mut Control::default(),
        |_, _| panic!("oversized record must not escape"))), JobError::Corrupt);

    let mut narrow = freeze(&inputs());
    narrow.limits.max_snapshot_bytes = (2 * (A.len() + B.len()) + "alpha".len() - 1) as u64;
    let mut reader = PayloadGuard { cursor: Cursor::new(bytes), allowed_until: (2 * HEADER_BYTES + A.len()) as u64 };
    let mut visited = 0;
    assert_eq!(error(read(&mut reader, &key(), &narrow, &receipt, &mut Control::default(),
        |_, _| { visited += 1; Ok(()) })), JobError::Limit);
    assert_eq!(visited, 1);
}

struct ObservedIo { cursor: Cursor<Vec<u8>>, processed: Rc<Cell<usize>> }
impl Read for ObservedIo {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let count = self.cursor.read(bytes)?;
        self.processed.set(self.processed.get() + count); Ok(count)
    }
}
impl Write for ObservedIo {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let count = self.cursor.write(bytes)?;
        self.processed.set(self.processed.get() + count); Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> { self.cursor.flush() }
}
impl Seek for ObservedIo {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> { self.cursor.seek(position) }
}
struct CancelAfterIo(Rc<Cell<usize>>);
impl DecodeStepControl for CancelAfterIo {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        (self.0.get() >= 8192).then_some(DecodeCancellationKind::Deadline)
    }
}

#[test]
fn cancellation_during_one_large_record_stops_both_read_and_write() {
    let envelope = format!("{{\"id\":\"alpha\",\"text\":\"{}\"}}", "x".repeat(40000));
    let inputs = [JobInput { id: "alpha", original: envelope.as_bytes(), normalized: envelope.as_bytes() }];
    let manifest = freeze(&inputs); let mut full = Cursor::new(Vec::new());
    let receipt = write(&mut full, &key(), &manifest, retention(), &inputs, &mut Control::default()).unwrap();

    let processed = Rc::new(Cell::new(0));
    let mut writer = ObservedIo { cursor: Cursor::new(Vec::new()), processed: Rc::clone(&processed) };
    assert_eq!(error(write(&mut writer, &key(), &manifest, retention(), &inputs,
        &mut CancelAfterIo(processed))), JobError::Cancelled(DecodeCancellationKind::Deadline));
    assert!(writer.cursor.get_ref().len() > HEADER_BYTES);
    assert!(writer.cursor.get_ref().len() < full.get_ref().len());
    assert_eq!(error(restored(writer.cursor.get_ref(), &manifest, &receipt)), JobError::Corrupt);

    let processed = Rc::new(Cell::new(0));
    let mut reader = ObservedIo { cursor: full, processed: Rc::clone(&processed) };
    assert_eq!(error(read(&mut reader, &key(), &manifest, &receipt, &mut CancelAfterIo(processed),
        |_, _| panic!("cancelled record must not escape"))), JobError::Cancelled(DecodeCancellationKind::Deadline));
    assert!(reader.cursor.position() > HEADER_BYTES as u64);
    assert!(reader.cursor.position() < receipt.spool_bytes);
}

struct ShortWriter { cursor: Cursor<Vec<u8>>, remaining: usize }
impl Write for ShortWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.remaining == 0 { return Err(io::Error::other("private storage failure")); }
        let count = self.cursor.write(&bytes[..bytes.len().min(self.remaining)])?;
        self.remaining -= count; Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> { self.cursor.flush() }
}
impl Seek for ShortWriter {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> { self.cursor.seek(position) }
}

#[test]
fn partial_writes_and_visitor_failures_never_return_successful_authority() {
    let inputs = inputs(); let (manifest, receipt, bytes) = encoded();
    let mut writer = ShortWriter { cursor: Cursor::new(Vec::new()), remaining: HEADER_BYTES + 3 };
    assert_eq!(error(write(&mut writer, &key(), &manifest, retention(), &inputs,
        &mut Control::default())), JobError::Io);
    assert_eq!(writer.cursor.get_ref().len(), HEADER_BYTES + 3);
    assert!(restored(writer.cursor.get_ref(), &manifest, &receipt).is_err());
    let mut visited = 0;
    assert_eq!(error(read(&mut Cursor::new(bytes), &key(), &manifest, &receipt,
        &mut Control::default(), |_, _| { visited += 1; Err(JobError::Allocation) })), JobError::Allocation);
    assert_eq!(visited, 1);
}
