//! Model-free source IO and cancellation contracts; no native-model evidence.
use super::*;
use std::{
    fs::{self, OpenOptions},
    io::{self, Cursor, Write},
    sync::atomic::{AtomicU64, Ordering},
};

fn causes() -> [DecodeCancellationKind; 11] {
    use DecodeCancellationKind::*;
    [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit]
}

struct Metered {
    bytes: Cursor<Vec<u8>>,
    requests: Vec<usize>,
    max_read: usize,
    interrupt: bool,
}
impl Metered {
    fn new(bytes: Vec<u8>) -> Self {
        Self { bytes: Cursor::new(bytes), requests: Vec::new(),
            max_read: usize::MAX, interrupt: false }
    }
}
impl Read for Metered {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        self.requests.push(output.len());
        if std::mem::take(&mut self.interrupt) { return Err(io::ErrorKind::Interrupted.into()); }
        let len = output.len().min(self.max_read);
        self.bytes.read(&mut output[..len])
    }
}
fn source(bytes: &[u8]) -> SourceDigest {
    SourceDigest::new("fixture.safetensors", bytes.len() as u64, digest_hex(bytes)).unwrap()
}

#[test]
fn source_hash_keeps_exact_identity_and_fixed_read_quanta() {
    let bytes: Vec<_> = (0..2 * SOURCE_VERIFICATION_CHUNK_BYTES + 17).map(|i| (i % 251) as u8).collect();
    let expected = source(&bytes);
    let mut reader = Metered::new(bytes);
    let mut polls = 0;
    let receipt = verify_source_reader(&expected, &mut reader, &mut || { polls += 1; None }).unwrap();
    assert_eq!(receipt.bytes, expected.expected_bytes());
    assert_eq!(receipt.sha256, expected.expected_sha256());
    assert_eq!(reader.requests, [SOURCE_VERIFICATION_CHUNK_BYTES, SOURCE_VERIFICATION_CHUNK_BYTES, 17, 1]);
    assert_eq!(polls, reader.requests.len() + 1);

    let bytes = vec![93; 1031];
    let expected = source(&bytes);
    let mut reader = Metered::new(bytes);
    reader.max_read = 13; reader.interrupt = true;
    let receipt = verify_source_reader(&expected, &mut reader, &mut || None).unwrap();
    assert_eq!(receipt.sha256, expected.expected_sha256());
    assert!(reader.requests.iter().all(|size| *size <= SOURCE_VERIFICATION_CHUNK_BYTES));
}

#[test]
fn every_cause_returns_immediately_at_each_hash_boundary_including_completion() {
    let bytes = vec![7; 2 * SOURCE_VERIFICATION_CHUNK_BYTES + 17];
    let expected = source(&bytes);
    for cause in causes() {
        // Three chunks, the EOF probe, then the pre-publication checkpoint.
        for stop in 1..=5 {
            let mut reader = Metered::new(bytes.clone());
            let mut polls = 0;
            let result = verify_source_reader(&expected, &mut reader, &mut || {
                polls += 1;
                // Return the cause only once: a swallowed stop cannot be hidden
                // by a second callback returning the same value.
                (polls == stop).then_some(cause)
            });
            assert_eq!(result.unwrap_err(), SafetensorsError::Cancelled(cause));
            assert_eq!(polls, stop);
            assert_eq!(reader.requests.len(), stop - 1);
            assert!(reader.bytes.position() <= expected.expected_bytes());
        }
    }
}

#[test]
fn pinned_length_bounds_short_and_continuously_growing_readers() {
    let bytes = vec![5; SOURCE_VERIFICATION_CHUNK_BYTES + 3];
    let expected = source(&bytes);
    let mut short = Metered::new(bytes[..bytes.len() - 1].to_vec());
    assert!(matches!(verify_source_reader(&expected, &mut short, &mut || None),
        Err(SafetensorsError::SourceLength { expected: n, actual, .. })
            if n == expected.expected_bytes() && actual + 1 == n));
    struct Growing { read: usize }
    impl Read for Growing {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            assert!(output.len() <= SOURCE_VERIFICATION_CHUNK_BYTES);
            self.read += output.len();
            assert!(self.read <= SOURCE_VERIFICATION_CHUNK_BYTES + 4,
                "a length-pinned reader must never chase appended data");
            output.fill(5); Ok(output.len())
        }
    }
    let mut growing = Growing { read: 0 };
    assert!(matches!(verify_source_reader(&expected, &mut growing, &mut || None),
        Err(SafetensorsError::SourceLength { expected: n, actual, .. })
            if n == expected.expected_bytes() && actual == n + 1));
    assert_eq!(growing.read, expected.expected_bytes() as usize + 1);
    let mut wrong = Metered::new(vec![6; bytes.len()]);
    assert!(matches!(verify_source_reader(&expected, &mut wrong, &mut || None),
        Err(SafetensorsError::SourceDigest { .. })));
}

#[test]
fn interrupted_reads_and_metadata_reads_spend_the_same_cancellation_control() {
    struct Interrupted(usize);
    impl Read for Interrupted {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            self.0 += 1; Err(io::ErrorKind::Interrupted.into())
        }
    }
    for cause in causes() {
        let mut reader = Interrupted(0); let mut polls = 0;
        let result = verify_source_reader(&source(b"x"), &mut reader, &mut || {
            polls += 1; (polls == 4).then_some(cause)
        });
        assert_eq!(result.unwrap_err(), SafetensorsError::Cancelled(cause));
        assert_eq!(reader.0, 3); assert_eq!(polls, 4);

        let mut reader = Metered::new(vec![19; 2 * SOURCE_VERIFICATION_CHUNK_BYTES + 1]);
        let mut output = vec![0; 2 * SOURCE_VERIFICATION_CHUNK_BYTES + 1];
        let mut polls = 0;
        let result = read_exact_with_checkpoint(&mut reader, &mut output, "fixture", "read", &mut || {
            polls += 1; (polls == 2).then_some(cause)
        });
        assert_eq!(result.unwrap_err(), SafetensorsError::Cancelled(cause));
        assert_eq!(reader.requests, [SOURCE_VERIFICATION_CHUNK_BYTES]);
        assert!(output[..SOURCE_VERIFICATION_CHUNK_BYTES].iter().all(|b| *b == 19));
        assert!(output[SOURCE_VERIFICATION_CHUNK_BYTES..].iter().all(|b| *b == 0));
    }
}

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);
fn fixture() -> (PathBuf, SourceClosure, Vec<u8>) {
    let root = std::env::temp_dir().join(format!("fnlp-source-verify-{}-{}",
        std::process::id(), NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)));
    fs::create_dir(&root).unwrap();
    let payload = vec![1; 2 * SOURCE_VERIFICATION_CHUNK_BYTES + 18];
    let header = serde_json::to_vec(&serde_json::json!({
        "x": { "dtype": "BF16", "shape": [payload.len() / 2], "data_offsets": [0, payload.len()] }
    })).unwrap();
    let mut shard = (header.len() as u64).to_le_bytes().to_vec();
    shard.extend_from_slice(&header); shard.extend_from_slice(&payload);
    let index = br#"{"weight_map":{"x":"fixture.safetensors"}}"#.to_vec();
    for (name, bytes) in [("index.json", &index), ("fixture.safetensors", &shard)] {
        let mut file = OpenOptions::new().write(true).create_new(true).open(root.join(name)).unwrap();
        file.write_all(bytes).unwrap();
    }
    let closure = SourceClosure::new(
        SourceDigest::new("index.json", index.len() as u64, digest_hex(&index)).unwrap(),
        vec![source(&shard)],
    ).unwrap();
    (root, closure, payload)
}

#[test]
fn initial_and_retained_verification_use_control_without_exposing_partial_indexes() {
    let (root, closure, payload) = fixture();
    let mut total = 0;
    let index = SafetensorsRangeIndex::open_with_options_and_checkpoint(
        &root, closure.clone(), RangeIndexOptions::default(), || { total += 1; None },
    ).unwrap();
    assert_eq!(index.read_range("x", RowPanel::WholeTensor).unwrap(), payload);
    for cause in causes() {
        for stop in [1, total / 2, total] {
            let mut polls = 0;
            let result = SafetensorsRangeIndex::open_with_options_and_checkpoint(
                &root, closure.clone(), RangeIndexOptions::default(), || {
                    polls += 1; (polls == stop).then_some(cause)
                },
            );
            assert!(matches!(result, Err(SafetensorsError::Cancelled(observed)) if observed == cause));
            assert_eq!(polls, stop);
        }
    }
    let mut retained_polls = 0;
    index.revalidate_retained_shards_with_checkpoint(|| { retained_polls += 1; None }).unwrap();
    for cause in causes() {
        for stop in 1..=retained_polls {
            let mut polls = 0;
            assert_eq!(index.revalidate_retained_shards_with_checkpoint(|| {
                polls += 1; (polls == stop).then_some(cause)
            }).unwrap_err(), SafetensorsError::Cancelled(cause));
            assert_eq!(polls, stop);
            // A stopped hash releases its lock; the next pass rewinds the
            // original held descriptor and must still prove its whole digest.
            index.revalidate_retained_shards().unwrap();
        }
    }
    let mut writer = OpenOptions::new().write(true).open(root.join("fixture.safetensors")).unwrap();
    writer.seek(SeekFrom::End(-1)).unwrap(); writer.write_all(&[2]).unwrap();
    assert!(matches!(index.revalidate_retained_shards(), Err(SafetensorsError::SourceDigest { .. })));
    assert!(matches!(SafetensorsRangeIndex::open(&root, closure), Err(SafetensorsError::SourceDigest { .. })));
}

#[test]
fn cancellation_and_nonregular_sources_are_refused_before_data_open() {
    let absent = Path::new("/fnlp-source-verification-does-not-exist");
    for cause in causes() {
        assert!(matches!(SafetensorsRangeIndex::open_pinned_nanbeige42_with_checkpoint(absent, || Some(cause)),
            Err(SafetensorsError::Cancelled(observed)) if observed == cause));
    }
    let (root, closure, _) = fixture();
    let bogus = SourceDigest::new("directory", 0, digest_hex(b"")).unwrap();
    assert!(matches!(open_source_file(&root, &bogus), Err(SafetensorsError::Io { operation: "regular-file check", .. })));
    #[cfg(unix)]
    {
        assert!(matches!(open_source_file(Path::new("/dev/null"), &bogus),
            Err(SafetensorsError::Io { operation: "regular-file check", .. })));
        let link = root.join("link");
        std::os::unix::fs::symlink(root.join("fixture.safetensors"), &link).unwrap();
        assert!(matches!(open_source_file(&link, &closure.shards()[0]),
            Err(SafetensorsError::Io { operation: "regular-file check", .. })));
    }
}
