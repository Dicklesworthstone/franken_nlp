//! Storage-management fixtures, not model, replay, or power-loss evidence.
use super::*;
use crate::jobs::tests::{Control, error, input, key, limits, manifest, work};
use crate::native_engine::decode::DecodeCancellationKind;
use std::{fs::{self, File, OpenOptions}, io::{Seek, SeekFrom, Write},
    os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt}, path::PathBuf,
    sync::atomic::{AtomicU64, Ordering}};

static NEXT: AtomicU64 = AtomicU64::new(0);
fn root() -> PathBuf {
    let path = std::env::temp_dir().join(format!("fnlp-stored-job-{}-{}",
        std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    fs::DirBuilder::new().mode(0o700).create(&path).unwrap();
    path
}
fn create(root: &Path) -> OwnedJob {
    OwnedJob::create(root, key(), manifest(), &mut Control::default()).unwrap()
}
fn open(root: &Path) -> StoredJob {
    StoredJob::open(root, key(), JobId([7; 16]), limits(), &mut Control::default()).unwrap()
}
fn commit(job: &mut OwnedJob, ordinal: usize) {
    job.begin(&input(ordinal), work(), &mut Control::default()).unwrap()
        .commit(&serde_json::json!({"private_result": ordinal}), &mut Control::default()).unwrap();
}
fn complete(root: &Path) {
    let mut job = create(root);
    commit(&mut job, 0); commit(&mut job, 1);
}
fn write_private(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}

#[test]
fn committed_state_can_be_inspected_without_any_original_bytes() {
    let dir = root(); let mut job = create(&dir); commit(&mut job, 0); drop(job);
    let before = fs::read(dir.join("results.spool")).unwrap();
    let stored = open(&dir); let report = stored.status().unwrap();
    assert_eq!((report.items, report.committed, report.attempts), (2, 1, 1));
    assert_eq!(report.reserved_work, work()); assert_eq!(report.uncommitted_spool_bytes, 0);
    assert_eq!(report.verification_scope, "authenticated-stored-state-not-input-replay-v1");
    assert_eq!(fs::read(dir.join("results.spool")).unwrap(), before);
    let json = crate::canonjson::canonical_string(&report).unwrap();
    for private in ["private_result", "private Alice", "alpha", "recipe", "commitment"] {
        assert!(!json.contains(private));
    }
}
#[test]
fn no_model_materialization_matches_original_owner_bytes_and_never_charges_work() {
    let a = root(); let b = root(); complete(&a); complete(&b);
    let mut original = OwnedJob::resume(&a, key(), manifest(), TailPolicy::Refuse, &mut Control::default()).unwrap();
    original.materialize_ordered(&mut Control::default()).unwrap(); drop(original);
    let mut stored = open(&b); let before = stored.status().unwrap();
    let after = stored.materialize_ordered(&mut Control::default()).unwrap();
    assert!(after.materialized); assert!(!after.unacknowledged_output_present);
    assert_eq!(after.attempts, before.attempts); assert_eq!(after.reserved_work, before.reserved_work);
    assert_eq!(fs::read(a.join("materialized.ndjson")).unwrap(), fs::read(b.join("materialized.ndjson")).unwrap());
    assert_eq!(stored.materialize_ordered(&mut Control::default()).unwrap(), after);
}
#[test]
fn pending_and_failed_attempts_are_reported_without_retry_or_refund() {
    let dir = root(); let mut job = create(&dir);
    drop(job.begin(&input(0), work(), &mut Control::default()).unwrap()); drop(job);
    let mut stored = open(&dir);
    let report = stored.verify(&mut Control::default()).unwrap();
    assert_eq!((report.committed, report.attempts), (0, 1)); assert_eq!(report.reserved_work, work());
    assert_eq!(error(stored.materialize_ordered(&mut Control::default())), JobError::Incomplete);
    assert!(!dir.join("materialized.ndjson").exists());
}
#[test]
fn wrong_secret_expected_job_and_limits_cannot_open_the_management_view() {
    let dir = root(); complete(&dir);
    assert!(StoredJob::open(&dir, JobSecret::from_bytes([33; 32]), JobId([7; 16]), limits(), &mut Control::default()).is_err());
    assert_eq!(error(StoredJob::open(&dir, key(), JobId([8; 16]), limits(), &mut Control::default())),
        JobError::Mismatch(MismatchField::Job));
    let mut changed = limits(); changed.max_attempts += 1;
    assert_eq!(error(StoredJob::open(&dir, key(), JobId([7; 16]), changed, &mut Control::default())),
        JobError::Mismatch(MismatchField::Limits));
}
#[test]
fn original_execution_resume_still_requires_the_original_population() {
    let dir = root(); complete(&dir); drop(open(&dir));
    let replacement = JobInput { original: b"not original", ..input(0) };
    let wrong = FrozenManifest::freeze(&key(), super::super::super::JobContract {
        job_id: JobId([7; 16]), execution: &crate::jobs::tests::identity(),
        recipe: &"fixed item-local recipe; effective-seed=7", limits: limits(),
    }, [replacement, input(1)], &mut Control::default()).unwrap();
    assert_eq!(error(OwnedJob::resume(&dir, key(), wrong, TailPolicy::Refuse, &mut Control::default())),
        JobError::Mismatch(MismatchField::Population));
}
#[test]
fn uncommitted_tail_is_visible_in_status_and_never_promoted_or_truncated() {
    let dir = root(); complete(&dir);
    let mut spool = OpenOptions::new().append(true).open(dir.join("results.spool")).unwrap();
    spool.write_all(b"uncommitted private tail").unwrap(); spool.sync_all().unwrap(); drop(spool);
    let before = fs::read(dir.join("results.spool")).unwrap();
    let mut stored = open(&dir);
    assert_eq!(stored.status().unwrap().uncommitted_spool_bytes, 24);
    assert_eq!(error(stored.verify(&mut Control::default())), JobError::UncommittedTail);
    assert_eq!(error(stored.status()), JobError::Poisoned); drop(stored);
    let mut stored = open(&dir);
    assert_eq!(error(stored.materialize_ordered(&mut Control::default())), JobError::UncommittedTail);
    assert_eq!(fs::read(dir.join("results.spool")).unwrap(), before);
    assert!(!dir.join("materialized.ndjson").exists());
}
#[test]
fn uncommitted_stages_are_reported_and_left_for_explicit_original_recovery() {
    let dir = root(); complete(&dir);
    let stage = dir.join(".fnlp-redact-1234-7.part"); write_private(&stage, b"staged private bytes");
    let mut stored = open(&dir); assert!(stored.status().unwrap().staged_output_present);
    assert_eq!(error(stored.materialize_ordered(&mut Control::default())), JobError::UncommittedTail);
    assert_eq!(fs::read(stage).unwrap(), b"staged private bytes");
    assert!(!dir.join("materialized.ndjson").exists());
}
#[test]
fn corrupt_committed_result_blocks_even_status_and_never_repairs_tail() {
    let dir = root(); complete(&dir);
    let mut spool = OpenOptions::new().write(true).open(dir.join("results.spool")).unwrap();
    spool.seek(SeekFrom::Start(frame::HEADER_BYTES as u64)).unwrap(); spool.write_all(b"X").unwrap();
    spool.sync_all().unwrap(); drop(spool);
    let before = fs::read(dir.join("results.spool")).unwrap();
    assert!(StoredJob::open(&dir, key(), JobId([7; 16]), limits(), &mut Control::default()).is_err());
    assert_eq!(fs::read(dir.join("results.spool")).unwrap(), before);
}
#[test]
fn independently_authenticated_rows_cannot_change_the_frozen_population_root() {
    let dir = root(); let job = create(&dir);
    let mut row = job.item(1).unwrap(); row.binding.original = key().commit(b"different", &[b"input"]);
    job.journal.transaction(|j| j.write(Table::Item, 1, &job.key, &row, false)).unwrap();
    job.files.sync_database().unwrap(); drop(job);
    assert_eq!(error(StoredJob::open(&dir, key(), JobId([7; 16]), limits(), &mut Control::default())), JobError::Corrupt);
}
#[test]
fn wrong_ordinals_and_duplicate_keyed_ids_cannot_rehydrate_a_manifest() {
    for duplicate in [false, true] {
        let dir = root(); let job = create(&dir); let mut row = job.item(1).unwrap();
        if duplicate { row.binding.id = job.item(0).unwrap().binding.id; } else { row.binding.ordinal = 0; }
        job.journal.transaction(|j| j.write(Table::Item, 1, &job.key, &row, false)).unwrap();
        job.files.sync_database().unwrap(); drop(job);
        assert_eq!(error(StoredJob::open(&dir, key(), JobId([7; 16]), limits(), &mut Control::default())), JobError::Corrupt);
    }
}
#[test]
fn unacknowledged_publication_is_verified_before_reconciliation_without_rerunning() {
    let dir = root(); let mut job = create(&dir); commit(&mut job, 0); commit(&mut job, 1);
    job.fail_at = Some(Fault::Published);
    assert_eq!(error(job.materialize_ordered(&mut Control::default())), JobError::Io); drop(job);
    let before = fs::read(dir.join("materialized.ndjson")).unwrap();
    let mut stored = open(&dir); let report = stored.status().unwrap();
    assert!(report.unacknowledged_output_present); assert!(!report.materialized);
    assert!(!stored.verify(&mut Control::default()).unwrap().materialized);
    let after = stored.materialize_ordered(&mut Control::default()).unwrap();
    assert!(after.materialized); assert_eq!(after.attempts, 2); assert_eq!(after.reserved_work, report.reserved_work);
    assert_eq!(fs::read(dir.join("materialized.ndjson")).unwrap(), before);
}
#[test]
fn unrelated_existing_output_is_never_overwritten_or_reported_as_verified() {
    let dir = root(); complete(&dir); let destination = dir.join("materialized.ndjson");
    write_private(&destination, b"unrelated output");
    let mut stored = open(&dir); assert!(stored.status().unwrap().unacknowledged_output_present);
    assert!(stored.verify(&mut Control::default()).is_err()); drop(stored);
    assert!(open(&dir).materialize_ordered(&mut Control::default()).is_err());
    assert_eq!(fs::read(destination).unwrap(), b"unrelated output");
}
#[test]
fn declared_materialized_output_is_checked_when_the_view_opens() {
    let dir = root(); complete(&dir); open(&dir).materialize_ordered(&mut Control::default()).unwrap();
    write_private(&dir.join("materialized.ndjson"), b"corrupted");
    assert!(StoredJob::open(&dir, key(), JobId([7; 16]), limits(), &mut Control::default()).is_err());
}
#[test]
fn management_keeps_the_same_exclusive_lock_as_execution() {
    let dir = root(); complete(&dir); let stored = open(&dir);
    assert_eq!(error(OwnedJob::resume(&dir, key(), manifest(), TailPolicy::Refuse, &mut Control::default())), JobError::Busy);
    drop(stored); drop(OwnedJob::resume(&dir, key(), manifest(), TailPolicy::Refuse, &mut Control::default()).unwrap());
}
#[test]
fn cancellation_cannot_publish_or_leave_a_usable_management_session() {
    let dir = root(); complete(&dir);
    assert_eq!(error(StoredJob::open(&dir, key(), JobId([7; 16]), limits(), &mut Control { calls: 0, stop_at: Some(1) })),
        JobError::Cancelled(crate::native_engine::decode::DecodeCancellationKind::Deadline));
    let mut stored = open(&dir);
    assert!(stored.materialize_ordered(&mut Control { calls: 0, stop_at: Some(1) }).is_err());
    assert_eq!(error(stored.status()), JobError::Poisoned);
    assert!(!dir.join("materialized.ndjson").exists());
}

const CANCELLATION_CAUSES: [DecodeCancellationKind; 11] = [
    DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
    DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
    DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
    DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit,
];

#[test]
fn management_preserves_the_first_cancellation_without_polling_a_replacement() {
    struct FirstStop { cause: DecodeCancellationKind, calls: usize }
    impl DecodeStepControl for FirstStop {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            Some(if self.calls == 1 { self.cause } else { DecodeCancellationKind::User })
        }
    }
    for cause in CANCELLATION_CAUSES {
        let dir = root(); complete(&dir);
        let mut control = FirstStop { cause, calls: 0 };
        assert_eq!(error(StoredJob::open(&dir, key(), JobId([7; 16]), limits(), &mut control)),
            JobError::Cancelled(cause));
        assert_eq!(control.calls, 1);
        for materialize in [false, true] {
            let mut stored = open(&dir); let before = stored.status().unwrap();
            let mut control = FirstStop { cause, calls: 0 };
            let result = if materialize { stored.materialize_ordered(&mut control) }
                else { stored.verify(&mut control) };
            assert_eq!(error(result), JobError::Cancelled(cause));
            assert_eq!(control.calls, 1);
            assert_eq!(error(stored.status()), JobError::Poisoned);
            assert_eq!(stored.report, before);
            assert!(!dir.join("materialized.ndjson").exists());
        }
    }
}

#[test]
fn opening_checks_terminal_cancellation_after_failed_metadata_authentication() {
    struct StopAfterAdmission { cause: DecodeCancellationKind, calls: usize }
    impl DecodeStepControl for StopAfterAdmission {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            if self.calls == 1 { None } else { Some(self.cause) }
        }
    }
    for cause in CANCELLATION_CAUSES {
        let dir = root(); complete(&dir);
        let before = open(&dir).status().unwrap();
        let journal = fs::read(dir.join("journal.fsqlite")).unwrap();
        let spool = fs::read(dir.join("results.spool")).unwrap();
        let mut control = StopAfterAdmission { cause, calls: 0 };
        // No row checkpoint can follow a header MAC failure. The second
        // checkpoint is the terminal boundary after that synchronous read.
        assert_eq!(error(StoredJob::open(&dir, JobSecret::from_bytes([33; 32]),
            JobId([7; 16]), limits(), &mut control)), JobError::Cancelled(cause));
        assert_eq!(control.calls, 2);
        assert_eq!(fs::read(dir.join("journal.fsqlite")).unwrap(), journal);
        assert_eq!(fs::read(dir.join("results.spool")).unwrap(), spool);
        assert_eq!(open(&dir).status().unwrap(), before);
        assert_eq!(error(StoredJob::open(&dir, JobSecret::from_bytes([33; 32]),
            JobId([7; 16]), limits(), &mut Control::default())), JobError::Authentication);
    }
}

/// A cloned held descriptor shares the actual frame cursor. Output verification
/// can be armed only after the result prefix was read and then stopped after
/// the first materialized row was compared, independently of checkpoint counts.
struct StopAfterRead {
    spool: File, end: u64, cause: DecodeCancellationKind,
    output_comparison: bool, armed: bool, observed: bool,
}
impl StopAfterRead {
    fn new(job: &mut OwnedJob, cause: DecodeCancellationKind, output_comparison: bool) -> Self {
        job.files.spool.seek(SeekFrom::Start(0)).unwrap();
        Self { spool: job.files.spool.try_clone().unwrap(), end: job.header.spool_end,
            cause, output_comparison, armed: false, observed: false }
    }
}
impl DecodeStepControl for StopAfterRead {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if self.armed {
            // A corrupted first output row is read only after the complete
            // prefix scan; its source frame leaves the cursor below the end.
            assert!(self.spool.stream_position().unwrap() < self.end);
            self.observed = true; return Some(self.cause);
        }
        if self.spool.stream_position().unwrap() == self.end {
            if self.output_comparison { self.armed = true; return None; }
            self.observed = true; return Some(self.cause);
        }
        None
    }
}

#[test]
fn management_cancellation_precedes_corrupt_committed_frames_without_repair() {
    for cause in CANCELLATION_CAUSES {
        for materialize in [false, true] {
            let dir = root(); complete(&dir);
            let mut stored = open(&dir); let before = stored.status().unwrap();
            let pointer = stored.job.item(1).unwrap().pointer.unwrap();
            stored.job.files.spool.seek(SeekFrom::Start(pointer.offset + frame::HEADER_BYTES as u64)).unwrap();
            stored.job.files.spool.write_all(b"X").unwrap();
            stored.job.files.spool.sync_all().unwrap();
            let bytes = fs::read(dir.join("results.spool")).unwrap();
            let mut control = StopAfterRead::new(&mut stored.job, cause, false);
            let result = if materialize { stored.materialize_ordered(&mut control) }
                else { stored.verify(&mut control) };
            assert_eq!(error(result), JobError::Cancelled(cause));
            assert!(control.observed);
            assert_eq!(error(stored.status()), JobError::Poisoned);
            assert_eq!(stored.report, before);
            assert_eq!(fs::read(dir.join("results.spool")).unwrap(), bytes);
            assert!(!dir.join("materialized.ndjson").exists());
            drop(control); drop(stored);
            assert_eq!(error(StoredJob::open(&dir, key(), JobId([7; 16]), limits(),
                &mut Control::default())), JobError::Authentication);
        }
    }
}

#[test]
fn management_cancellation_precedes_corrupt_materialized_bytes_without_replacement() {
    for cause in CANCELLATION_CAUSES {
        for materialize in [false, true] {
            let dir = root(); complete(&dir);
            let mut stored = open(&dir);
            let before = stored.materialize_ordered(&mut Control::default()).unwrap();
            let destination = dir.join("materialized.ndjson");
            let inode = fs::metadata(&destination).unwrap().ino();
            let mut corrupted = fs::read(&destination).unwrap();
            corrupted[0] = b'X';
            write_private(&destination, &corrupted);
            let mut control = StopAfterRead::new(&mut stored.job, cause, true);
            let result = if materialize { stored.materialize_ordered(&mut control) }
                else { stored.verify(&mut control) };
            assert_eq!(error(result), JobError::Cancelled(cause));
            assert!(control.armed && control.observed);
            assert_eq!(error(stored.status()), JobError::Poisoned);
            assert_eq!(stored.report, before);
            assert_eq!(fs::read(&destination).unwrap(), corrupted);
            assert_eq!(fs::metadata(&destination).unwrap().ino(), inode);
            drop(control); drop(stored);
            assert_eq!(error(StoredJob::open(&dir, key(), JobId([7; 16]), limits(),
                &mut Control::default())), JobError::Corrupt);
        }
    }
}

#[test]
fn cancelled_final_management_report_keeps_the_committed_file_and_work() {
    struct StopDuringReport {
        journal: PathBuf, before: Vec<u8>, destination: PathBuf,
        cause: DecodeCancellationKind, committed_checks: usize, observed: bool,
    }
    impl DecodeStepControl for StopDuringReport {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            if fs::read(&self.journal).unwrap() == self.before { return None; }
            self.committed_checks += 1;
            match self.committed_checks {
                // The owned materializer's final check accepts its commit.
                1 => None,
                // Refuse the final management metadata scan, after publication
                // committed. No private output bytes or journal rows change.
                2 => {
                    fs::set_permissions(&self.destination, fs::Permissions::from_mode(0o640)).unwrap();
                    None
                }
                _ => { self.observed = true; Some(self.cause) }
            }
        }
    }
    for cause in CANCELLATION_CAUSES {
        let dir = root(); complete(&dir);
        let mut stored = open(&dir); let before = stored.status().unwrap();
        let journal = dir.join("journal.fsqlite");
        let destination = dir.join("materialized.ndjson");
        let mut control = StopDuringReport { before: fs::read(&journal).unwrap(),
            journal, destination: destination.clone(), cause, committed_checks: 0, observed: false };
        assert_eq!(error(stored.materialize_ordered(&mut control)), JobError::Cancelled(cause));
        assert!(control.observed); assert_eq!(control.committed_checks, 3);
        assert_eq!(error(stored.status()), JobError::Poisoned);
        assert_eq!(stored.report, before);
        assert!(stored.job.progress().materialized);
        assert_eq!(stored.job.progress().attempts, before.attempts);
        assert_eq!(stored.job.progress().reserved_work, before.reserved_work);
        let output = fs::read(&destination).unwrap();
        let inode = fs::metadata(&destination).unwrap().ino();
        assert_eq!(output, b"{\"private_result\":0}\n{\"private_result\":1}\n");
        // Repair only the fixture's permission mutation, then require a fresh
        // authenticated owner. The poisoned instance cannot acknowledge it.
        fs::set_permissions(&destination, fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(error(stored.status()), JobError::Poisoned);
        drop(stored);
        let mut restored = open(&dir);
        let after = restored.materialize_ordered(&mut Control::default()).unwrap();
        assert!(after.materialized); assert!(!after.unacknowledged_output_present);
        assert_eq!(after.attempts, before.attempts);
        assert_eq!(after.reserved_work, before.reserved_work);
        assert_eq!(fs::read(&destination).unwrap(), output);
        assert_eq!(fs::metadata(&destination).unwrap().ino(), inode);
    }
}
