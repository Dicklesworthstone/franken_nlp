use super::*;
use std::{
    cell::Cell,
    future::Future,
    sync::{Arc, atomic::{AtomicU64, Ordering}},
    task::{Context, Poll, Waker},
    time::Duration,
};
use crate::artifact::manifest::{
    ReleaseArtifact, ReleaseCompatibility, ReleaseDigests, ReleaseLifecycle,
    ReleaseSourceClosure,
};
use super::super::charge_body_bytes;

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

fn stage(bytes: &[u8]) -> (PathBuf, File) {
    let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "fnlp-direct-artifact-{}-{sequence}.stage", std::process::id(),
    ));
    let mut file = open_staging(&path, false).unwrap();
    file.write_all(bytes).unwrap();
    (path, file)
}

// Only the private streaming helpers accept this scaled multipart inventory.
// Public acquisition still authenticates and validates the real fixed-size
// manifest policy before inspecting or creating staging.
fn manifest() -> ReleaseManifest {
    let parts = [b"abc".as_slice(), b"DEFG", b"hijkl"];
    ReleaseManifest {
        manifest_schema: String::new(), model_id: String::new(),
        source_revision: String::new(), artifact_id: String::new(), release_tag: String::new(),
        artifact: ReleaseArtifact {
            logical_name: "fixture.fnlpq".to_owned(), bytes: 12,
            sha256: hex_lower(&Sha256::digest(b"abcDEFGhijkl")),
        },
        parts: parts.iter().enumerate().map(|(index, bytes)| ReleasePart {
            id: index as u32, name: format!("fixture.part{index:02}"), bytes: bytes.len() as u64,
            sha256: hex_lower(&Sha256::digest(bytes)),
            mirrors: vec!["https://first.invalid/part".to_owned(), "https://second.invalid/part".to_owned()],
        }).collect(),
        source: ReleaseSourceClosure { files: Vec::new(), source_root_sha256: String::new() },
        compatibility: ReleaseCompatibility {
            recipe_id: String::new(), converter_id: String::new(),
            fnlpq_format: String::new(), pull_api: String::new(),
        },
        digests: ReleaseDigests {
            tokenizer_sha256: String::new(), template_sha256: String::new(),
            census_sha256: String::new(), logical_model_sha256: String::new(),
            packing_set_sha256: String::new(),
        },
        license_bundle_sha256: String::new(), packing_policy: String::new(),
        lifecycle: ReleaseLifecycle {
            state: "active".to_owned(), superseded_by: None, revocation_reason: None,
        },
    }
}

#[test]
fn every_multipart_resume_boundary_authenticates_complete_parts_and_hashes_the_tail() {
    let manifest = manifest();
    let bytes = b"abcDEFGhijkl";
    for (retained, complete, verified) in [(0, 0, 0), (1, 0, 0), (3, 1, 3),
        (4, 1, 3), (7, 2, 7), (8, 2, 7), (12, 3, 12)] {
        let (path, mut file) = stage(&bytes[..retained]);
        let prefix = authenticate_prefix(&mut file, &manifest, retained as u64,
            &mut [0_u8; 2], &mut |_| Ok(())).unwrap();
        assert_eq!(prefix.complete_parts, complete);
        assert_eq!(prefix.verified_bytes, verified as u64);
        assert_eq!(hex_lower(&prefix.partial_hasher.finalize()),
            hex_lower(&Sha256::digest(&bytes[verified..retained])));
        assert_eq!(file.stream_position().unwrap(), retained as u64);
        assert_eq!(fs::read(path).unwrap(), bytes[..retained]);
    }
}

#[test]
fn corrupt_complete_prefix_and_oversized_staging_refuse_without_modification() {
    let manifest = manifest();
    let (path, mut file) = stage(b"abXDE");
    assert!(matches!(authenticate_prefix(&mut file, &manifest, 5,
        &mut [0_u8; 2], &mut |_| Ok(())), Err(NetworkPullError::Digest { part: 0, .. })));
    assert_eq!(fs::read(&path).unwrap(), b"abXDE");
    let (path, mut file) = stage(b"abcDEFGhijkl!");
    assert!(matches!(authenticate_prefix(&mut file, &manifest, 13,
        &mut [0_u8; 2], &mut |_| Ok(())), Err(NetworkPullError::Manifest(PullError::ResumeLength {
            expected_max: 12, observed: 13,
        }))));
    assert_eq!(fs::read(path).unwrap(), b"abcDEFGhijkl!");
}

#[test]
fn corruption_in_a_later_complete_part_is_found_before_any_append() {
    let manifest = manifest();
    let (path, mut file) = stage(b"abcDEXGh");
    assert!(matches!(authenticate_prefix(&mut file, &manifest, 8,
        &mut [0_u8; 2], &mut |_| Ok(())), Err(NetworkPullError::Digest { part: 1, .. })));
    assert_eq!(fs::read(path).unwrap(), b"abcDEXGh");
}

#[test]
fn mirror_digest_failure_restores_absolute_offset_and_next_mirror_can_finish() {
    let manifest = manifest();
    let part = &manifest.parts[1];
    let (path, mut file) = stage(b"abcDE");
    let prefix = authenticate_prefix(&mut file, &manifest, 5,
        &mut [0_u8; 2], &mut |_| Ok(())).unwrap();
    let mut downloaded = 0;
    let mut hasher = prefix.partial_hasher.clone();
    file.write_all(b"XX").unwrap(); hasher.update(b"XX");
    charge_body_bytes(&mut downloaded, 2).unwrap();
    assert!(matches!(finish_artifact_attempt(1, 0, part, 3, 2, 3,
        &mut file, &hasher, Ok(2)), Err(NetworkPullError::Digest { part: 1, .. })));
    assert_eq!(file.stream_position().unwrap(), 5);
    assert_eq!(fs::read(&path).unwrap(), b"abcDE", "earlier part must survive rollback");
    let end = artifact_range_end(1, 3, 2, 4, 12, Some(4), downloaded).unwrap();
    assert_eq!(end, 3);
    let mut hasher = prefix.partial_hasher;
    file.write_all(b"FG").unwrap(); hasher.update(b"FG");
    charge_body_bytes(&mut downloaded, 2).unwrap();
    assert_eq!(finish_artifact_attempt(1, 1, part, 3, 2, end,
        &mut file, &hasher, Ok(2)).unwrap(), 2);
    assert_eq!(downloaded, 4);
    assert_eq!(fs::read(path).unwrap(), b"abcDEFG");
}

#[test]
fn failed_mirror_bytes_reduce_the_next_range_and_budget_reports_absolute_retained_bytes() {
    let manifest = manifest();
    let part = &manifest.parts[1];
    let (path, mut file) = stage(b"abcD");
    let prefix = authenticate_prefix(&mut file, &manifest, 4,
        &mut [0_u8; 2], &mut |_| Ok(())).unwrap();
    let mut downloaded = 0;
    let mut bad = prefix.partial_hasher.clone();
    file.write_all(b"XXX").unwrap(); bad.update(b"XXX");
    charge_body_bytes(&mut downloaded, 3).unwrap();
    assert!(finish_artifact_attempt(1, 0, part, 3, 1, 3,
        &mut file, &bad, Ok(3)).is_err());
    let end = artifact_range_end(1, 3, 1, 4, 12, Some(4), downloaded).unwrap();
    assert_eq!(end, 1, "discarded bytes cannot be spent again");
    let mut hasher = prefix.partial_hasher;
    file.write_all(b"E").unwrap(); hasher.update(b"E");
    charge_body_bytes(&mut downloaded, 1).unwrap();
    assert_eq!(finish_artifact_attempt(1, 1, part, 3, 1, end,
        &mut file, &hasher, Ok(1)).unwrap(), 1);
    assert_eq!(fs::read(path).unwrap(), b"abcDE");
    assert!(matches!(artifact_range_end(1, 3, 2, 4, 12, Some(4), downloaded),
        Err(NetworkPullError::DownloadBudgetExceeded {
            limit: 4, downloaded_bytes: 4, retained_bytes: 5, total_bytes: 12, part: 1,
        })));
}

#[test]
fn corrupt_unverified_tail_cannot_pass_when_the_part_completes() {
    let manifest = manifest();
    let (path, mut file) = stage(b"abcDX");
    let prefix = authenticate_prefix(&mut file, &manifest, 5,
        &mut [0_u8; 2], &mut |_| Ok(())).unwrap();
    assert_eq!(prefix.verified_bytes, 3, "the partial tail is not authenticated");
    let mut hasher = prefix.partial_hasher;
    file.write_all(b"FG").unwrap(); hasher.update(b"FG");
    assert!(matches!(finish_artifact_attempt(1, 0, &manifest.parts[1], 3, 2, 3,
        &mut file, &hasher, Ok(2)), Err(NetworkPullError::Digest { part: 1, .. })));
    assert_eq!(fs::read(path).unwrap(), b"abcDX", "refuse rather than repair retained input");
}

#[test]
fn prefix_cancellation_keeps_its_exact_cause_and_retained_bytes() {
    let manifest = manifest();
    let (path, mut file) = stage(b"abcDEFGh");
    let mut reads = 0;
    let result = authenticate_prefix(&mut file, &manifest, 8, &mut [0_u8; 1], &mut |part| {
        if part == Some(1) { reads += 1; }
        if reads == 3 {
            Err(NetworkPullError::Cancelled { cause: Some(CancelKind::CostBudget), part })
        } else { Ok(()) }
    });
    assert!(matches!(result, Err(NetworkPullError::Cancelled {
        cause: Some(CancelKind::CostBudget), part: Some(1),
    })));
    assert_eq!(fs::read(path).unwrap(), b"abcDEFGh");
}

struct Dropped<'a>(&'a Cell<bool>);
impl Drop for Dropped<'_> {
    fn drop(&mut self) { self.0.set(true); }
}

#[test]
fn timeout_and_cancellation_drop_the_transfer_then_restore_the_artifact_boundary() {
    for cancel in [false, true] {
        let manifest = manifest();
        let (path, mut file) = stage(b"abcDE");
        let mut hasher = Sha256::new(); hasher.update(b"DE");
        let mut downloaded = 0;
        let dropped = Cell::new(false);
        let clock = Arc::new(asupersync::time::VirtualClock::starting_at(
            asupersync::types::Time::from_secs(17),
        ));
        let driver = asupersync::time::TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
            .with_timer_driver(driver).build().unwrap();
        let result = runtime.block_on(async {
            let cx = Cx::current().unwrap();
            let mut task = Context::from_waker(Waker::noop());
            let mut attempt = std::pin::pin!(drive_attempt(&cx, 1, 0, Duration::from_millis(250), async {
                let _guard = Dropped(&dropped);
                file.write_all(b"F").unwrap(); hasher.update(b"F");
                charge_body_bytes(&mut downloaded, 1).unwrap();
                std::future::pending::<Result<u64, NetworkPullError>>().await
            }));
            assert!(attempt.as_mut().poll(&mut task).is_pending());
            if cancel { cx.cancel_with(CancelKind::User, None); }
            clock.advance(if cancel { 100_000_000 } else { 250_000_001 });
            match attempt.as_mut().poll(&mut task) {
                Poll::Ready(result) => result,
                Poll::Pending => panic!("owned deadline or cancellation timer must end the attempt"),
            }
        });
        assert!(dropped.get());
        let error = finish_artifact_attempt(1, 0, &manifest.parts[1], 3, 2, 3,
            &mut file, &hasher, result).unwrap_err();
        if cancel {
            assert!(matches!(error, NetworkPullError::Cancelled {
                cause: Some(CancelKind::User), part: Some(1),
            }));
        } else {
            assert!(matches!(error, NetworkPullError::RequestTimeout { part: 1, mirror: 0 }));
        }
        assert_eq!(file.stream_position().unwrap(), 5);
        assert_eq!(fs::read(path).unwrap(), b"abcDE");
        assert_eq!(downloaded, 1, "a failed attempt cannot refund bytes");
    }
}

#[test]
fn mode_path_conflicts_are_rejected_without_touching_staging() {
    let mut request = NetworkArtifactRequest {
        manifest: PathBuf::from("/caller/release.manifest.json"),
        expected_release_manifest_sha256: "a".repeat(64),
        staging_output: PathBuf::from("/caller/model.stage"),
        output: PathBuf::from("/caller/model.fnlpq"),
    };
    assert!(validate_request(&request).is_ok());
    request.output = request.staging_output.clone();
    assert!(validate_request(&request).is_err());
    request.output = PathBuf::from("/different/model.fnlpq");
    assert!(validate_request(&request).is_err());
    request.output = request.manifest.clone();
    assert!(validate_request(&request).is_err());
}
