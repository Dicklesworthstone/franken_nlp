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
