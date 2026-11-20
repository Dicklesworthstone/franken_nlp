//! Source preparation controls remain independent of canonical artifact authority.
use super::*;
use std::{
    fs::OpenOptions,
    io::{self, Cursor, Write},
    sync::atomic::{AtomicU64, Ordering},
};

fn causes() -> [DecodeCancellationKind; 11] {
    use DecodeCancellationKind::*;
    [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit]
}
struct Metered {
    inner: Cursor<Vec<u8>>, requests: Vec<usize>, interrupt: bool,
}
impl Metered {
    fn new(bytes: Vec<u8>) -> Self {
        Self { inner: Cursor::new(bytes), requests: Vec::new(), interrupt: false }
    }
}
impl Read for Metered {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        self.requests.push(output.len());
        if std::mem::take(&mut self.interrupt) { return Err(io::ErrorKind::Interrupted.into()); }
        self.inner.read(output)
    }
}

#[test]
fn closure_hash_has_exact_bytes_and_returns_every_single_shot_cancellation() {
    let bytes: Vec<_> = (0..2 * SOURCE_VERIFICATION_CHUNK_BYTES + 9).map(|i| (i % 239) as u8).collect();
    let path = Path::new("fixture");
    let mut reader = Metered::new(bytes.clone());
    let mut polls = 0;
    let hash = stream_sha256(&mut reader, path, bytes.len() as u64, &mut || {
        polls += 1; None
    }).unwrap();
    assert_eq!(hash, sha256_hex(&bytes));
    assert_eq!(reader.requests, [SOURCE_VERIFICATION_CHUNK_BYTES, SOURCE_VERIFICATION_CHUNK_BYTES, 9, 1]);
    assert_eq!(polls, 5);
    for cause in causes() {
        for stop in 1..=polls {
            let mut reader = Metered::new(bytes.clone()); let mut calls = 0;
            let error = stream_sha256(&mut reader, path, bytes.len() as u64, &mut || {
                calls += 1; (calls == stop).then_some(cause)
            }).unwrap_err();
            assert_eq!(error.cancellation_kind(), Some(cause));
            assert_eq!(calls, stop); assert_eq!(reader.requests.len(), stop - 1);
        }
    }
    let mut reader = Metered::new(bytes.clone()); reader.interrupt = true;
    assert_eq!(stream_sha256(&mut reader, path, bytes.len() as u64, &mut || None).unwrap(), hash);
}

#[test]
fn declared_source_length_refuses_short_or_growing_inputs_without_chasing_eof() {
    let path = Path::new("fixture");
    let mut short = Cursor::new(b"abc");
    assert!(matches!(stream_sha256(&mut short, path, 4, &mut || None),
        Err(ConverterError::SourceLength { expected: 4, actual: 3, .. })));
    let mut grown = Metered::new(vec![0; 2 * SOURCE_VERIFICATION_CHUNK_BYTES]);
    assert!(matches!(stream_sha256(&mut grown, path, 7, &mut || None),
        Err(ConverterError::SourceLength { expected: 7, actual: 8, .. })));
    assert_eq!(grown.requests, [7, 1]); assert_eq!(grown.inner.position(), 8);
}

#[test]
fn manifest_read_cap_survives_growth_and_checks_cancellation_between_chunks() {
    let path = Path::new("fixture-manifest");
    let mut full = Metered::new(vec![b' '; MAX_SOURCE_MANIFEST_BYTES]);
    let bytes = read_source_manifest_bytes(&mut full, path, &mut || None).unwrap();
    assert_eq!(bytes.len(), MAX_SOURCE_MANIFEST_BYTES);
    assert_eq!(full.requests.last(), Some(&1));
    assert!(full.requests.iter().all(|size| *size <= SOURCE_VERIFICATION_CHUNK_BYTES));
    let mut grown = Metered::new(vec![b' '; MAX_SOURCE_MANIFEST_BYTES + SOURCE_VERIFICATION_CHUNK_BYTES]);
    assert!(matches!(read_source_manifest_bytes(&mut grown, path, &mut || None),
        Err(ConverterError::ManifestTooLarge { observed, cap, .. })
            if observed == MAX_SOURCE_MANIFEST_BYTES as u64 + 1 && cap == MAX_SOURCE_MANIFEST_BYTES as u64));
    assert_eq!(grown.inner.position(), MAX_SOURCE_MANIFEST_BYTES as u64 + 1);
    for cause in causes() {
        let mut reader = Metered::new(vec![0; 3 * SOURCE_VERIFICATION_CHUNK_BYTES]);
        let mut calls = 0;
        let error = read_source_manifest_bytes(&mut reader, path, &mut || {
            calls += 1; (calls == 2).then_some(cause)
        }).unwrap_err();
        assert_eq!(error.cancellation_kind(), Some(cause));
        assert_eq!(calls, 2); assert_eq!(reader.requests, [SOURCE_VERIFICATION_CHUNK_BYTES]);
    }
    let mut interrupted = Metered::new(b"bounded".to_vec()); interrupted.interrupt = true;
    assert_eq!(read_source_manifest_bytes(&mut interrupted, path, &mut || None).unwrap(), b"bounded");
}

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);
fn root() -> PathBuf {
    let path = std::env::temp_dir().join(format!("fnlp-source-controls-{}-{}",
        std::process::id(), NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)));
    fs::create_dir(&path).unwrap(); path
}
fn write_new(path: &Path, bytes: &[u8]) -> File {
    let mut file = OpenOptions::new().read(true).write(true).create_new(true).open(path).unwrap();
    file.write_all(bytes).unwrap(); file
}
fn closure_fixture() -> (PathBuf, ConversionSourceManifest) {
    let root = root(); let mut files = Vec::new(); let mut total = 0;
    for i in 0..10 {
        let name = if i == 0 { "config.json".to_owned() } else { format!("source-{i}.bin") };
        let bytes = vec![i as u8; if i == 0 { 2 * SOURCE_VERIFICATION_CHUNK_BYTES + 11 } else { 7 }];
        let _file = write_new(&root.join(&name), &bytes);
        total += bytes.len();
        files.push(serde_json::json!({ "name": name, "bytes": bytes.len(), "sha256": sha256_hex(&bytes) }));
    }
    let bytes = serde_json::to_string(&serde_json::json!({
        "schema_version": 1, "model": "Nanbeige4.2-3B", "revision": PINNED_REVISION,
        "observed_state": "synthetic-non-authoritative", "closure_total_bytes": total,
        "logical_safetensors_payload_bytes": 0, "safetensors_container_header_bytes": 0,
        "files": files,
    })).unwrap();
    (root, ConversionSourceManifest::parse(&bytes).unwrap())
}

#[test]
fn whole_closure_and_retained_metadata_share_control_and_leave_original_files_unchanged() {
    let (root, manifest) = closure_fixture();
    let baseline = verify_source_closure(&root, manifest.clone(), true).unwrap();
    let mut total = 0;
    assert_eq!(verify_source_closure_with_checkpoint(&root, manifest.clone(), true, || {
        total += 1; None
    }).unwrap().files, baseline.files);
    for cause in causes() {
        for stop in [1, total / 2, total] {
            let mut calls = 0;
            let error = verify_source_closure_with_checkpoint(&root, manifest.clone(), true, || {
                calls += 1; (calls == stop).then_some(cause)
            }).unwrap_err();
            assert_eq!(error.cancellation_kind(), Some(cause)); assert_eq!(calls, stop);
        }
        let mut calls = 0;
        let error = read_materialized_source_member(&root, &baseline, "config.json", &mut || {
            calls += 1; (calls == 3).then_some(cause)
        }).unwrap_err();
        assert_eq!(error.cancellation_kind(), Some(cause)); assert_eq!(calls, 3);
        assert_eq!(prepare_nanbeige42_input_with_checkpoint(&root.join("absent"), manifest.clone(), true,
            DEFAULT_PANEL_BYTES, || Some(cause)).err().unwrap().cancellation_kind(), Some(cause));
    }
    assert_eq!(verify_source_closure(&root, manifest, true).unwrap().files, baseline.files);
}

#[test]
fn both_manifest_loaders_preserve_pins_and_refuse_nonregular_or_changed_inputs() {
    let pinned = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/truth-pack/nanbeige4.2-3b.source.json");
    assert_eq!(ConversionSourceManifest::load_pinned(&pinned).unwrap(),
        ConversionSourceManifest::load_pinned_with_checkpoint(&pinned, || None).unwrap());
    assert_eq!(ConversionSourceManifest::load(&pinned).unwrap(),
        ConversionSourceManifest::load_with_checkpoint(&pinned, || None).unwrap());
    let root = root();
    for cause in causes() {
        assert_eq!(ConversionSourceManifest::load_pinned_with_checkpoint(root.join("absent"), || Some(cause))
            .unwrap_err().cancellation_kind(), Some(cause));
        assert_eq!(ConversionSourceManifest::load_with_checkpoint(root.join("absent"), || Some(cause))
            .unwrap_err().cancellation_kind(), Some(cause));
    }
    assert!(matches!(ConversionSourceManifest::load(&root), Err(ConverterError::SourceNotRegular { .. })));
    assert!(matches!(ConversionSourceManifest::load_pinned(&root), Err(ConverterError::SourceNotRegular { .. })));
    let path = root.join("growing.json");
    let file = write_new(&path, b"x");
    let mut calls = 0;
    let error = read_source_manifest(&path, &mut || {
        calls += 1;
        // Growth occurs after the held descriptor's initial length check.
        if calls == 3 { file.set_len(MAX_SOURCE_MANIFEST_BYTES as u64 + 1).unwrap(); }
        None
    }).unwrap_err();
    assert!(matches!(error, ConverterError::ManifestTooLarge { .. }));
    file.set_len(1).unwrap(); let mut calls = 0;
    assert!(matches!(read_source_manifest(&path, &mut || {
        calls += 1; if calls == 3 { file.set_len(0).unwrap(); } None
    }), Err(ConverterError::Io { operation: "source manifest length check", .. })));
    #[cfg(unix)]
    {
        assert!(matches!(ConversionSourceManifest::load("/dev/null"), Err(ConverterError::SourceNotRegular { .. })));
        let link = root.join("manifest-link"); std::os::unix::fs::symlink(&pinned, &link).unwrap();
        assert!(matches!(ConversionSourceManifest::load_pinned(link), Err(ConverterError::SourceNotRegular { .. })));
    }
}

#[test]
fn direct_and_nested_source_errors_preserve_all_original_cancellation_causes() {
    for cause in causes() {
        assert_eq!(ConverterError::Cancelled(cause).cancellation_kind(), Some(cause));
        assert_eq!(ConverterError::Safetensors(SafetensorsError::Cancelled(cause)).cancellation_kind(), Some(cause));
    }
    assert_eq!(ConverterError::Arithmetic { invariant: "fixture" }.cancellation_kind(), None);
}
