//! One-record initialization uses the existing retained format and runner.
use super::*;
use std::io::{BufRead, Cursor, Read};

fn transport() -> PopulationReadLimits {
    PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 1000 }
}
fn ndjson(first: &[u8], second: &[u8]) -> Vec<u8> {
    let mut bytes = first.to_vec(); bytes.push(b'\n'); bytes.extend_from_slice(second); bytes
}

struct ObservedReader {
    reader: Cursor<Vec<u8>>, events: Rc<Events>, eof: Rc<Cell<bool>>, dropped: Rc<Cell<bool>>,
}
impl Read for ObservedReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> { self.reader.read(output) }
}
impl BufRead for ObservedReader {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        assert!(self.events.prepared.borrow().is_empty());
        assert!(self.events.executed.borrow().is_empty());
        let bytes = self.reader.fill_buf()?;
        if bytes.is_empty() { self.eof.set(true); }
        Ok(bytes)
    }
    fn consume(&mut self, bytes: usize) { self.reader.consume(bytes); }
}
impl Drop for ObservedReader { fn drop(&mut self) { self.dropped.set(true); } }

#[test]
fn streamed_creation_matches_v1_bytes_and_resumes_after_original_reader_is_dropped() {
    let streamed = Root::new(); let legacy = Root::new(); let mut control = Control::default();
    let a = b" {\"id\":\"alpha\", \"text\":\"Alice private source\", \"task_args\":{\"v\":1}}\r";
    let b = inputs()[1].original;
    let original = [JobInput { id: "alpha", original: a, normalized: a },
        JobInput { id: "beta", original: b, normalized: b }];
    let mut baseline = JobRunner::create_with_retained_inputs(&legacy.0, key(), JobId([7;16]),
        limits(), retention(), &original, Processor::new(Rc::new(Events::default())), &mut control).unwrap();
    baseline.run(&mut control).unwrap(); baseline.materialize_ordered(&mut control).unwrap();

    // Empty LF/CRLF transport lines are ignored; nonempty CR and whitespace
    // remain exact envelope bytes. The final record has no newline terminator.
    let mut bytes = b"\n\r\n".to_vec(); bytes.extend(ndjson(a, b));
    let events = Rc::new(Events::default()); let eof = Rc::new(Cell::new(false));
    let dropped = Rc::new(Cell::new(false));
    let reader = ObservedReader { reader: Cursor::new(bytes), events: events.clone(),
        eof: eof.clone(), dropped: dropped.clone() };
    let mut run = JobRunner::create_with_retained_ndjson(&streamed.0, key(), JobId([7;16]),
        limits(), retention(), reader, transport(), Processor::new(events.clone()), &mut control).unwrap();
    assert!(eof.get()); assert!(dropped.get());
    assert!(events.prepared.borrow().is_empty()); assert!(events.executed.borrow().is_empty());
    match &run.inputs { Inputs::Retained(cursor) => assert_eq!(cursor.ordinal(), 0),
        Inputs::Borrowed(_) => panic!("streamed creation must own its replay cursor") }
    assert_eq!(fs::read(streamed.0.join("inputs.spool")).unwrap(),
        fs::read(legacy.0.join("inputs.spool")).unwrap());
    assert_eq!(run.step(&mut control).unwrap().unwrap().committed, 1); drop(run);

    let mut resumed = JobRunner::resume_with_retained_inputs(&streamed.0, key(), JobId([7;16]),
        limits(), retention(), Processor::new(events.clone()), TailPolicy::Refuse, &mut control).unwrap();
    let progress = resumed.run(&mut control).unwrap(); resumed.materialize_ordered(&mut control).unwrap();
    assert_eq!(progress.attempts, 2); assert_eq!(progress.reserved_work, work().checked_add(work()).unwrap());
    assert_eq!(*events.prepared.borrow(), vec!["alpha", "beta"]);
    assert_eq!(*events.executed.borrow(), vec![1, 2]);
    assert_eq!(fs::read(streamed.0.join("materialized.ndjson")).unwrap(),
        fs::read(legacy.0.join("materialized.ndjson")).unwrap());
}

#[test]
fn later_typed_argument_or_duplicate_failure_prevents_all_preparation_and_execution() {
    let original = inputs();
    for (second, expected) in [
        (br#"{"id":"beta","text":"valid envelope","task_args":{"v":"wrong type"}}"#.as_slice(),
            JobRunError::InvalidEnvelope),
        (original[0].original, JobRunError::Storage(JobError::DuplicateId)),
    ] {
        let root = Root::new(); let events = Rc::new(Events::default()); let mut control = Control::default();
        assert_eq!(err(JobRunner::create_with_retained_ndjson(&root.0, key(), JobId([7;16]), limits(),
            retention(), Cursor::new(ndjson(original[0].original, second)), transport(),
            Processor::new(events.clone()), &mut control)), expected);
        assert!(events.prepared.borrow().is_empty()); assert!(events.executed.borrow().is_empty());
        assert!(!root.0.join("materialized.ndjson").exists());
        assert!(crate::jobs::StoredJob::open(&root.0, key(), JobId([7;16]), limits(), &mut control).is_err());
    }
}

#[test]
fn streamed_runner_preserves_transport_and_logical_population_ceilings() {
    let original = inputs(); let bytes = ndjson(original[0].original, original[1].original);
    for axis in 0..3 {
        let root = Root::new(); let events = Rc::new(Events::default());
        let mut job = limits(); let mut wire = transport();
        match axis {
            0 => wire.max_stream_bytes = bytes.len() as u64 - 1,
            1 => wire.max_lines = 1,
            _ => job.max_snapshot_bytes = original.iter().map(|input|
                (input.id.len() + input.original.len() + input.normalized.len()) as u64).sum::<u64>() - 1,
        }
        assert_eq!(err(JobRunner::create_with_retained_ndjson(&root.0, key(), JobId([7;16]), job,
            retention(), Cursor::new(bytes.clone()), wire, Processor::new(events.clone()),
            &mut Control::default())), JobRunError::Storage(JobError::Limit));
        assert!(events.prepared.borrow().is_empty()); assert!(events.executed.borrow().is_empty());
    }
}
