//! Parser/population fixtures only, not native inference or IO qualification.
use super::*;
use crate::jobs::tests::{Control, error, identity, key, limits};
use crate::native_engine::decode::DecodeCancellationKind;
use std::io::{self, BufReader, Cursor, Read};

const A: &[u8] = br#"{"id":"alpha","text":"private Alice","task_args":{"n":1}}"#;
const B: &[u8] = br#"{"id":"beta","text":"private Bob"}"#;
fn transport() -> PopulationReadLimits { PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 128 } }
fn stream() -> Vec<u8> { [b"\n".as_slice(), A, b"\r\n\r\n", B].concat() }
fn read(bytes: &[u8]) -> Result<JobPopulation, JobError> {
    JobPopulation::read_ndjson(&mut Cursor::new(bytes), &key(), JobId([7;16]), limits(), transport(), &mut Control::default())
}

#[test]
fn chunk_boundaries_do_not_change_exact_envelopes_or_commitments() {
    let bytes = stream(); let mut baseline = None;
    for capacity in [1, 2, 7, 8192] {
        let mut reader = BufReader::with_capacity(capacity, Cursor::new(&bytes));
        let population = JobPopulation::read_ndjson(&mut reader, &key(), JobId([7;16]), limits(), transport(), &mut Control::default()).unwrap();
        let stats = population.stats();
        assert_eq!(stats.items, 2); assert_eq!(stats.input_lines, 4); assert_eq!(stats.input_bytes, bytes.len() as u64);
        let inputs = population.borrowed_inputs().unwrap();
        assert_eq!(inputs[0].id, "alpha"); assert_eq!(inputs[0].original, [A, b"\r"].concat());
        assert_eq!(inputs[1].original, B);
        assert!(std::ptr::eq(inputs[0].original.as_ptr(), inputs[0].normalized.as_ptr()));
        assert_eq!(stats.snapshot_bytes, ("alpha".len() + "beta".len() + 2 * (A.len() + 1 + B.len())) as u64);
        let manifest = population.freeze(&key(), JobContract { job_id: JobId([7;16]), execution: &identity(),
            recipe: &"population-fixture", limits: limits() }, &mut Control::default()).unwrap();
        if let Some(previous) = baseline { assert!(manifest.population_commitment().matches(previous)); }
        else { baseline = Some(manifest.population_commitment()); }
    }
}
#[test]
fn optional_final_lf_does_not_change_the_record() {
    let without = read(A).unwrap(); let bytes = [A, b"\n"].concat(); let with = read(&bytes).unwrap();
    assert_eq!(without.inputs().next().unwrap().original, with.inputs().next().unwrap().original);
    assert_eq!(without.stats().input_lines, 1); assert_eq!(with.stats().input_lines, 1);
}
#[test]
fn duplicate_ids_after_blank_lines_refuse_the_complete_population() {
    let bytes = [A, b"\n\n", B, b"\n\r\n", A].concat();
    assert_eq!(error(read(&bytes)), JobError::DuplicateId);
}
#[test]
fn malformed_later_records_never_return_a_prefix() {
    for bad in [br#"{"id":"x","id":"y","text":"x"}"#.as_slice(),
        br#"{"id":"x","text":"x","task_args":{"a":1,"a":2}}"#,
        br#"{"flush":true}"#, br#"{"id":"x","text":4}"#,
        br#"{"id":"x","text":"x","extra":1}"#, br#"{"id":"","text":"x"}"#,
        b" ", b"{", b"\xff"] {
        let bytes = [A, b"\n", bad].concat();
        assert_eq!(error(read(&bytes)), JobError::InvalidInput);
    }
}
#[test]
fn empty_and_blank_only_inputs_are_not_jobs() {
    for bytes in [b"".as_slice(), b"\n", b"\r\n\n"] {
        assert_eq!(error(read(bytes)), JobError::InvalidInput);
    }
}
#[test]
fn complete_population_and_logical_snapshot_caps_fail_closed() {
    for axis in 0..3 {
        let mut cap = limits();
        match axis { 0 => cap.max_items = 1, 1 => cap.max_snapshot_bytes = 1, _ => cap.max_attempts = 1 }
        let failure = error(JobPopulation::read_ndjson(&mut Cursor::new(stream()), &key(), JobId([7;16]),
            cap, transport(), &mut Control::default()));
        assert_eq!(failure, if axis == 2 { JobError::InvalidLimits } else { JobError::Limit });
    }
}
#[test]
fn record_bound_includes_cr_but_excludes_lf() {
    let mut cap = limits(); cap.max_input_bytes_per_item = A.len();
    for (suffix, success) in [(b"\n".as_slice(), true), (b"\r\n".as_slice(), false)] {
        let bytes = [A, suffix].concat();
        let result = JobPopulation::read_ndjson(&mut Cursor::new(bytes), &key(), JobId([7;16]), cap, transport(), &mut Control::default());
        if success { assert!(result.is_ok()); } else { assert_eq!(error(result), JobError::Limit); }
    }
}
#[test]
fn stream_and_line_caps_include_ignored_blank_lines_and_delimiters() {
    let bytes = stream();
    for cap in [PopulationReadLimits { max_stream_bytes: bytes.len() as u64 - 1, max_lines: 128 },
        PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 3 }] {
        assert_eq!(error(JobPopulation::read_ndjson(&mut Cursor::new(&bytes), &key(), JobId([7;16]),
            limits(), cap, &mut Control::default())), JobError::Limit);
    }
    assert!(JobPopulation::read_ndjson(&mut Cursor::new(&bytes), &key(), JobId([7;16]), limits(),
        PopulationReadLimits { max_stream_bytes: bytes.len() as u64, max_lines: 4 }, &mut Control::default()).is_ok());
}
struct Broken;
impl Read for Broken { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { Err(io::Error::other("private IO")) } }
impl BufRead for Broken {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { Err(io::Error::other("private IO")) }
    fn consume(&mut self, _: usize) {}
}
#[test]
fn reader_errors_and_pre_cancelled_calls_are_typed_and_content_free() {
    assert_eq!(error(JobPopulation::read_ndjson(&mut Broken, &key(), JobId([7;16]), limits(), transport(), &mut Control::default())), JobError::Io);
    let mut control = Control { calls: 0, stop_at: Some(1) };
    assert_eq!(error(JobPopulation::read_ndjson(&mut Broken, &key(), JobId([7;16]), limits(), transport(), &mut control)),
        JobError::Cancelled(DecodeCancellationKind::Deadline));
}
#[test]
fn long_unterminated_record_checkpoints_before_reading_the_whole_line() {
    let bytes = vec![b' '; 40000]; let mut reader = Cursor::new(&bytes);
    let mut control = Control { calls: 0, stop_at: Some(4) };
    assert_eq!(error(JobPopulation::read_ndjson(&mut reader, &key(), JobId([7;16]), limits(), transport(), &mut control)),
        JobError::Cancelled(DecodeCancellationKind::Deadline));
    assert!(reader.position() < bytes.len() as u64);
}
#[test]
fn transport_limits_must_be_finite_and_nonzero() {
    for cap in [PopulationReadLimits { max_stream_bytes: 0, max_lines: 1 },
        PopulationReadLimits { max_stream_bytes: 1, max_lines: 0 },
        PopulationReadLimits { max_stream_bytes: u64::MAX, max_lines: 1 },
        PopulationReadLimits { max_stream_bytes: 1, max_lines: u64::MAX }] {
        assert_eq!(cap.validate(), Err(JobError::InvalidLimits));
    }
}

#[test]
fn streamed_records_freeze_the_same_exact_population_without_borrowing_payloads() {
    let bytes = stream();
    let population = read(&bytes).unwrap();
    let expected = population.freeze(&key(), JobContract { job_id: JobId([7;16]),
        execution: &identity(), recipe: &"streaming-fixture", limits: limits() },
        &mut Control::default()).unwrap();
    for capacity in [1, 2, 7, 8192] {
        let source = BufReader::with_capacity(capacity, Cursor::new(&bytes));
        let mut reader = PopulationReader::new(source, limits(), transport(), &mut Control::default()).unwrap();
        let mut builder = crate::jobs::manifest::ManifestBuilder::new(&key(), JobContract {
            job_id: JobId([7;16]), execution: &identity(), recipe: &"streaming-fixture", limits: limits(),
        }, &mut Control::default()).unwrap();
        while let Some(record) = reader.next_record(&mut Control::default()).unwrap() {
            let input = record.input();
            assert!(std::ptr::eq(input.original.as_ptr(), input.normalized.as_ptr()));
            builder.push(&key(), &input, &mut Control::default()).unwrap();
            // Each owned record is dropped here; only keyed bindings survive.
        }
        let actual = builder.finish(&key()).unwrap();
        assert_eq!(reader.stats(), population.stats());
        assert_eq!(actual.binding.compare(&expected.binding), Ok(()));
        assert_eq!(actual.item_count(), 2);
    }
}

#[test]
fn streaming_reader_consumes_only_one_record_and_never_reuses_a_failed_prefix() {
    let bytes = [A, b"\n", B, b"\n{"].concat();
    let mut reader = PopulationReader::new(Cursor::new(&bytes), limits(), transport(), &mut Control::default()).unwrap();
    let first = reader.next_record(&mut Control::default()).unwrap().unwrap();
    assert_eq!(first.input().original, A);
    assert_eq!(reader.reader.position(), A.len() as u64 + 1);
    drop(first);
    let second = reader.next_record(&mut Control::default()).unwrap().unwrap();
    assert_eq!(second.input().original, B);
    assert_eq!(reader.reader.position(), (A.len() + B.len() + 2) as u64);
    drop(second);
    assert_eq!(error(reader.next_record(&mut Control::default())), JobError::InvalidInput);
    let position = reader.reader.position();
    assert_eq!(error(reader.next_record(&mut Control::default())), JobError::Poisoned);
    assert_eq!(reader.reader.position(), position);
}

#[test]
fn streaming_reader_caps_and_eof_validation_never_authorize_an_incomplete_population() {
    let bytes = [B, b"\n", A].concat();
    for axis in 0..6 {
        let mut cap = limits(); let mut wire = transport();
        match axis {
            0 => cap.max_items = 1,
            1 => cap.max_snapshot_bytes = ("beta".len() + 2 * B.len()) as u64,
            2 => cap.max_input_bytes_per_item = B.len(),
            3 => wire.max_stream_bytes = bytes.len() as u64 - 1,
            4 => wire.max_lines = 1,
            _ => cap.max_attempts = 1,
        }
        let mut reader = PopulationReader::new(Cursor::new(&bytes), cap, wire, &mut Control::default()).unwrap();
        assert!(reader.next_record(&mut Control::default()).unwrap().is_some());
        if axis == 5 { assert!(reader.next_record(&mut Control::default()).unwrap().is_some()); }
        let expected = if axis == 5 { JobError::InvalidLimits } else { JobError::Limit };
        assert_eq!(error(reader.next_record(&mut Control::default())), expected);
        let position = reader.reader.position();
        assert_eq!(error(reader.next_record(&mut Control::default())), JobError::Poisoned);
        assert_eq!(reader.reader.position(), position);
    }
    let mut empty = PopulationReader::new(Cursor::new(b"\r\n\n"), limits(), transport(), &mut Control::default()).unwrap();
    assert_eq!(error(empty.next_record(&mut Control::default())), JobError::InvalidInput);
    assert_eq!(error(empty.next_record(&mut Control::default())), JobError::Poisoned);
    let mut broken = PopulationReader::new(Broken, limits(), transport(), &mut Control::default()).unwrap();
    assert_eq!(error(broken.next_record(&mut Control::default())), JobError::Io);
    assert_eq!(error(broken.next_record(&mut Control::default())), JobError::Poisoned);
}

#[test]
fn streaming_reader_keeps_every_original_cancellation_kind_and_stops_consuming() {
    struct Cancel(DecodeCancellationKind);
    impl DecodeStepControl for Cancel {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) }
    }
    for kind in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::Deadline, DecodeCancellationKind::PollQuota,
        DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
        DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown,
        DecodeCancellationKind::LinkedExit] {
        let bytes = [A, b"\n", B].concat();
        let mut reader = PopulationReader::new(Cursor::new(&bytes), limits(), transport(), &mut Control::default()).unwrap();
        assert!(reader.next_record(&mut Control::default()).unwrap().is_some());
        let position = reader.reader.position();
        assert_eq!(error(reader.next_record(&mut Cancel(kind))), JobError::Cancelled(kind));
        assert_eq!(reader.reader.position(), position);
        assert_eq!(error(reader.next_record(&mut Control::default())), JobError::Poisoned);
        assert_eq!(reader.reader.position(), position);
    }
}
