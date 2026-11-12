//! Storage-format and failure-boundary fixtures, not native execution or crash qualification.
use super::*;
use crate::jobs::{JobContract, tests::{Control, error, identity, key, limits}};
use crate::native_engine::decode::DecodeCancellationKind;
use std::{fs::{self, OpenOptions}, io::{self, BufReader, Cursor},
    os::unix::fs::{DirBuilderExt, MetadataExt}, path::PathBuf,
    sync::atomic::{AtomicU64, Ordering}};

const A: &[u8] = b" \t{\"text\":\"private \\u0041lice\",\"id\":\"alpha\", \"task_args\":{\"v\":1}} \r";
const B: &[u8] = br#"{"id":"beta","text":"Bob private source","task_args":{"v":2}}"#;
const RECIPE: &str = "streamed exact-envelope fixture";
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("fnlp-job-stream-create-{}-{}",
            std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap(); Self(path)
    }
}
impl Drop for Root { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
fn population() -> [JobInput<'static>; 2] {
    [JobInput { id: "alpha", original: A, normalized: A },
        JobInput { id: "beta", original: B, normalized: B }]
}
fn source() -> Vec<u8> { [b"\n".as_slice(), A, b"\n\r\n", B].concat() }
fn retention() -> InputRetentionLimits { InputRetentionLimits { max_spool_bytes: 1 << 20 } }
fn transport() -> PopulationReadLimits { PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 16 } }
fn fresh(cap: JobLimits) -> ManifestBuilder {
    ManifestBuilder::new(&key(), JobContract { job_id: JobId([7; 16]), execution: &identity(),
        recipe: &RECIPE, limits: cap }, &mut Control::default()).unwrap()
}
fn manifest() -> FrozenManifest {
    let mut builder = fresh(limits());
    for input in population() { builder.push(&key(), &input, &mut Control::default()).unwrap(); }
    builder.finish(&key()).unwrap()
}
fn create<R: BufRead>(root: &Path, reader: R, retention: InputRetentionLimits)
    -> Result<(OwnedJob, RetainedInputCursor<File>), JobError> {
    OwnedJob::create_with_retained_ndjson(root, key(), limits(), retention, reader, transport(),
        fresh(limits()), &mut Control::default(), |_, _| Ok(()))
}
fn assert_uncommitted(root: &Path) {
    let before = fs::read(root.join("inputs.spool")).unwrap();
    assert!(StoredJob::open(root, key(), JobId([7; 16]), limits(), &mut Control::default()).is_err());
    assert!(OwnedJob::resume(root, key(), manifest(), TailPolicy::DiscardUncommitted,
        &mut Control::default()).is_err());
    assert_eq!(fs::read(root.join("inputs.spool")).unwrap(), before);
    assert_eq!(fs::metadata(root.join("results.spool")).unwrap().len(), 0);
    // A failed initialization remains an explicit orphan, never an implicit retry.
    assert_eq!(error(create(root, Cursor::new(source()), retention())), JobError::AlreadyExists);
    assert_eq!(fs::read(root.join("inputs.spool")).unwrap(), before);
}
fn staged() -> Vec<u8> {
    let mut bytes = Vec::new();
    for (ordinal, input) in population().iter().enumerate() {
        stage_input(&mut bytes, ordinal as u64, input, &mut Control::default()).unwrap();
    }
    bytes
}

#[test]
fn streamed_creation_keeps_exact_v1_spool_receipt_and_initial_header() {
    let baseline = Root::new();
    let legacy = OwnedJob::create_with_retained_inputs(&baseline.0, key(), manifest(), retention(),
        &population(), &mut Control::default()).unwrap();
    let expected_spool = fs::read(baseline.0.join("inputs.spool")).unwrap();
    let expected_header = crate::jobs::manifest::bounded_json(&legacy.header, 16 * 1024).unwrap();
    for capacity in [1, 2, 7, 8192] {
        let root = Root::new();
        let reader = BufReader::with_capacity(capacity, Cursor::new(source()));
        let (job, mut cursor) = create(&root.0, reader, retention()).unwrap();
        assert_eq!(fs::read(root.0.join("inputs.spool")).unwrap(), expected_spool);
        assert_eq!(crate::jobs::manifest::bounded_json(&job.header, 16 * 1024).unwrap(), expected_header);
        assert!(job.header.retained_inputs == legacy.header.retained_inputs);
        job.manifest.binding.compare(&legacy.manifest.binding).unwrap();
        assert_eq!(cursor.ordinal(), 0);
        for expected in population() {
            let input = cursor.read_next(&job.key, &job.manifest, &mut Control::default()).unwrap().unwrap();
            assert_eq!(input.id, expected.id);
            assert_eq!(input.original, expected.original); assert_eq!(input.normalized, expected.normalized);
        }
        assert!(cursor.read_next(&job.key, &job.manifest, &mut Control::default()).unwrap().is_none());
        drop(cursor); drop(job);
    }
}

#[derive(Debug, Eq, PartialEq)]
enum ValidationFailure { Storage(JobError), Typed }
impl From<JobError> for ValidationFailure { fn from(error: JobError) -> Self { Self::Storage(error) } }

#[test]
fn late_typed_failure_and_duplicate_id_never_publish_a_prefix_job() {
    let root = Root::new(); let mut visited = 0;
    let result = OwnedJob::create_with_retained_ndjson(&root.0, key(), limits(), retention(),
        Cursor::new(source()), transport(), fresh(limits()), &mut Control::default(),
        |_, _| -> Result<(), ValidationFailure> {
            visited += 1;
            if visited == 2 { Err(ValidationFailure::Typed) } else { Ok(()) }
        });
    assert!(matches!(result, Err(ValidationFailure::Typed))); assert_eq!(visited, 2);
    assert_eq!(fs::metadata(root.0.join("inputs.spool")).unwrap().len(), (retained::HEADER_BYTES + A.len()) as u64);
    assert_uncommitted(&root.0);

    let root = Root::new(); let duplicate = [A, b"\n", A].concat();
    assert_eq!(error(create(&root.0, Cursor::new(duplicate), retention())), JobError::DuplicateId);
    assert_uncommitted(&root.0);
}

#[test]
fn staged_payload_drift_cannot_replace_the_original_stream_manifest() {
    let root = Root::new(); let mut visited = 0;
    let result = OwnedJob::create_with_retained_ndjson(&root.0, key(), limits(), retention(),
        Cursor::new(source()), transport(), fresh(limits()), &mut Control::default(),
        |_, _| -> Result<(), JobError> {
            visited += 1;
            if visited == 2 {
                // Keep valid JSON, the same ID and byte length. Resealing from
                // disk alone would silently turn this drift into new authority.
                let position = A.windows(b"private".len()).position(|word| word == b"private").unwrap();
                let mut file = OpenOptions::new().write(true).open(root.0.join("inputs.spool")).unwrap();
                file.seek(SeekFrom::Start((retained::HEADER_BYTES + position) as u64)).unwrap();
                file.write_all(b"P").unwrap();
            }
            Ok(())
        });
    assert_eq!(error(result), JobError::Mismatch(MismatchField::Population));
    assert_uncommitted(&root.0);
}

#[test]
fn framed_disk_cap_is_checked_before_any_byte_of_the_next_record() {
    let exact = (2 * retained::HEADER_BYTES + A.len() + B.len()) as u64;
    let root = Root::new();
    let (job, cursor) = create(&root.0, Cursor::new(source()), InputRetentionLimits { max_spool_bytes: exact }).unwrap();
    assert_eq!(job.header.retained_inputs.as_ref().unwrap().spool_bytes, exact);
    drop(cursor); drop(job);
    let root = Root::new();
    assert_eq!(error(create(&root.0, Cursor::new(source()),
        InputRetentionLimits { max_spool_bytes: exact - 1 })), JobError::Limit);
    let bytes = fs::read(root.0.join("inputs.spool")).unwrap();
    assert_eq!(bytes.len(), retained::HEADER_BYTES + A.len());
    assert_eq!(&bytes[retained::HEADER_BYTES..], A);
    assert!(bytes[24..retained::HEADER_BYTES].iter().all(|&byte| byte == 0));
    assert_uncommitted(&root.0);
}

#[test]
fn staging_and_authenticated_manifest_must_share_the_same_logical_limits() {
    let root = Root::new(); let mut changed = limits(); changed.max_attempts += 1;
    let result = OwnedJob::create_with_retained_ndjson(&root.0, key(), limits(), retention(),
        Cursor::new(source()), transport(), fresh(changed), &mut Control::default(),
        |_, _| -> Result<(), JobError> { Ok(()) });
    assert_eq!(error(result), JobError::Mismatch(MismatchField::Limits));
    assert_uncommitted(&root.0);
}

#[test]
fn unjournaled_sealed_spools_are_orphans_and_committed_lost_ack_is_recoverable() {
    for point in [CreateFault::InputSpoolWritten, CreateFault::InputSpoolSynced, CreateFault::HeaderCommitted] {
        let root = Root::new();
        let result = OwnedJob::create_with_retained_ndjson_inner(&root.0, key(), limits(), retention(),
            Cursor::new(source()), transport(), fresh(limits()), &mut Control::default(),
            |_, _| -> Result<(), JobError> { Ok(()) }, Some(point));
        assert_eq!(error(result), JobError::Io);
        if point == CreateFault::HeaderCommitted {
            let mut stored = StoredJob::open(&root.0, key(), JobId([7; 16]), limits(), &mut Control::default()).unwrap();
            let report = stored.verify(&mut Control::default()).unwrap();
            assert_eq!((report.items, report.committed, report.attempts), (2, 0, 0));
            assert_eq!(report.retained_inputs.unwrap().items, 2); drop(stored);
            let (job, cursor) = OwnedJob::resume_with_retained_inputs(&root.0, key(), JobId([7; 16]), limits(),
                retention(), TailPolicy::Refuse, fresh(limits()), &mut Control::default(),
                |_, _| -> Result<(), JobError> { Ok(()) }).unwrap();
            assert_eq!(cursor.ordinal(), 0); assert!(!job.is_poisoned());
            assert_eq!(job.progress().attempts, 0); drop(cursor); drop(job);
        } else { assert_uncommitted(&root.0); }
    }
}

#[test]
fn one_exclusive_owner_spans_ingestion_header_publication_and_returned_cursor() {
    let root = Root::new(); let mut visited = 0;
    let (job, cursor) = OwnedJob::create_with_retained_ndjson(&root.0, key(), limits(), retention(),
        Cursor::new(source()), transport(), fresh(limits()), &mut Control::default(),
        |_, _| -> Result<(), JobError> {
            assert_eq!(error(StoredJob::open(&root.0, key(), JobId([7; 16]), limits(), &mut Control::default())), JobError::Busy);
            visited += 1; Ok(())
        }).unwrap();
    assert_eq!(visited, 2); assert_eq!(cursor.ordinal(), 0);
    for name in ["job.lock", "journal.fsqlite", "results.spool", "inputs.spool"] {
        assert_eq!(fs::metadata(root.0.join(name)).unwrap().mode() & 0o7777, 0o600);
    }
    assert_eq!(error(StoredJob::open(&root.0, key(), JobId([7; 16]), limits(), &mut Control::default())), JobError::Busy);
    drop(cursor);
    assert_eq!(error(StoredJob::open(&root.0, key(), JobId([7; 16]), limits(), &mut Control::default())), JobError::Busy);
    drop(job);
    drop(StoredJob::open(&root.0, key(), JobId([7; 16]), limits(), &mut Control::default()).unwrap());
}

#[test]
fn staged_header_payload_order_and_physical_extent_are_not_trusted() {
    let original = staged(); let split = retained::HEADER_BYTES + A.len();
    for mutation in 0..9 {
        let mut bytes = original.clone();
        match mutation {
            0 => bytes[0] ^= 1,
            1 => bytes[8] ^= 1,
            2 => bytes[16..24].copy_from_slice(&u64::MAX.to_le_bytes()),
            3 => bytes[24] = 1,
            4 => bytes[retained::HEADER_BYTES + 4] ^= 1,
            5 => bytes.truncate(bytes.len() - 1),
            6 => bytes.push(0),
            7 => { bytes = [bytes[split..].to_vec(), bytes[..split].to_vec()].concat(); },
            _ => bytes[split + 16..split + 24].fill(0),
        }
        let mut file = Cursor::new(bytes);
        assert!(seal_inputs(&mut file, &key(), &manifest(), retention(), original.len() as u64,
            &mut Control::default()).is_err(), "mutation {mutation}");
    }
}

struct CancelAt { polls: u64, stop_at: u64, cause: DecodeCancellationKind }
impl DecodeStepControl for CancelAt {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        self.polls += 1; (self.polls >= self.stop_at).then_some(self.cause)
    }
}
fn causes() -> [DecodeCancellationKind; 11] {
    use DecodeCancellationKind::*;
    [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit]
}

#[test]
fn every_sealing_checkpoint_preserves_the_original_typed_cancellation() {
    let original = staged(); let manifest = manifest();
    let mut successful = Control::default();
    seal_inputs(&mut Cursor::new(original.clone()), &key(), &manifest, retention(), original.len() as u64,
        &mut successful).unwrap();
    for cause in causes() {
        for stop_at in 1..=successful.calls {
            let mut control = CancelAt { polls: 0, stop_at, cause };
            assert_eq!(error(seal_inputs(&mut Cursor::new(original.clone()), &key(), &manifest,
                retention(), original.len() as u64, &mut control)), JobError::Cancelled(cause));
        }
    }
}

#[test]
fn malformed_declared_length_is_refused_before_reading_any_payload() {
    struct HeaderOnly(Cursor<Vec<u8>>);
    impl Read for HeaderOnly {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            assert!(self.0.position() < retained::HEADER_BYTES as u64, "unadmitted staged payload read");
            self.0.read(buffer)
        }
    }
    impl Write for HeaderOnly {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("sealed an unadmitted frame") }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    impl Seek for HeaderOnly {
        fn seek(&mut self, from: SeekFrom) -> io::Result<u64> { self.0.seek(from) }
    }
    let mut bytes = staged(); let length = bytes.len() as u64;
    bytes[16..24].copy_from_slice(&(limits().max_input_bytes_per_item as u64 + 1).to_le_bytes());
    let mut file = HeaderOnly(Cursor::new(bytes));
    assert_eq!(error(seal_inputs(&mut file, &key(), &manifest(), retention(), length,
        &mut Control::default())), JobError::Corrupt);
}
