//! Real fsqlite/filesystem restart tests with injected boundary failures.
//! These are storage fixtures, never native model or platform-ratification evidence.
use super::*;
use crate::jobs::tests::{Control, error, input, key, limits, manifest, manifest_with, work};
use std::{fs::{self, OpenOptions}, io::{Seek, SeekFrom, Write},
    os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt, symlink},
    path::PathBuf, sync::atomic::{AtomicU64, Ordering}};
static NEXT: AtomicU64 = AtomicU64::new(0);
fn root() -> PathBuf {
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!("fnlp-owned-job-{}-{n}", std::process::id()));
    fs::DirBuilder::new().mode(0o700).create(&root).unwrap(); root
}
fn create(root: &Path) -> OwnedJob { OwnedJob::create(root, key(), manifest(), &mut Control::default()).unwrap() }
fn resume(root: &Path, tail: TailPolicy) -> OwnedJob {
    OwnedJob::resume(root, key(), manifest(), tail, &mut Control::default()).unwrap()
}
fn commit(job: &mut OwnedJob, ordinal: usize) {
    let attempt = job.begin(&input(ordinal), work(), &mut Control::default()).unwrap();
    assert_eq!(attempt.ordinal(), ordinal as u64);
    attempt.commit(&serde_json::json!({"result":ordinal}), &mut Control::default()).unwrap();
}
fn append_tail(root: &Path, bytes: &[u8]) {
    let mut spool = OpenOptions::new().append(true).open(root.join("results.spool")).unwrap();
    spool.write_all(bytes).unwrap(); spool.sync_all().unwrap();
}
#[test]
fn exclusive_lock_and_private_files_hold_until_database_closes() {
    let dir = root(); let job = create(&dir);
    for name in ["journal.fsqlite", "results.spool", "job.lock"] {
        assert_eq!(fs::metadata(dir.join(name)).unwrap().mode() & 0o7777, 0o600);
    }
    assert_eq!(error(OwnedJob::resume(&dir, key(), manifest(), TailPolicy::Refuse, &mut Control::default())), JobError::Busy);
    drop(job); let _job = resume(&dir, TailPolicy::Refuse);
}
#[test]
fn original_or_normalized_input_drift_never_debits_or_executes() {
    let dir = root(); let mut job = create(&dir);
    let wrong = JobInput { original: b"different input", ..input(0) };
    assert_eq!(error(job.begin(&wrong, work(), &mut Control::default())), JobError::Mismatch(super::super::MismatchField::Population));
    assert_eq!(job.progress().attempts, 0); assert!(!job.is_poisoned());
}
#[test]
fn interrupted_attempt_stays_charged_and_reopens_with_new_attempt_number() {
    let dir = root(); let mut job = create(&dir);
    drop(job.begin(&input(0), work(), &mut Control::default()).unwrap());
    assert!(job.is_poisoned()); drop(job);
    let mut job = resume(&dir, TailPolicy::Refuse);
    assert_eq!(job.progress().attempts, 1); assert_eq!(job.progress().reserved_work, work());
    let attempt = job.begin(&input(0), work(), &mut Control::default()).unwrap();
    assert_eq!(attempt.number(), 2); attempt.commit(&"done", &mut Control::default()).unwrap();
    assert_eq!(job.progress().attempts, 2); assert_eq!(job.progress().reserved_work, work().checked_add(work()).unwrap());
}
#[test]
fn every_precommit_fault_preserves_debit_and_never_promotes_spool_bytes() {
    for fault in [Fault::Admitted, Fault::Running, Fault::SpoolWritten, Fault::SpoolSynced] {
        let dir = root(); let mut job = create(&dir); job.fail_at = Some(fault);
        let result = match job.begin(&input(0), work(), &mut Control::default()) {
            Ok(attempt) => attempt.commit(&"uncommitted", &mut Control::default()).map(|_| ()),
            Err(error) => Err(error),
        };
        assert_eq!(error(result), JobError::Io); assert!(job.is_poisoned()); drop(job);
        let mut job = resume(&dir, TailPolicy::DiscardUncommitted);
        assert_eq!(job.progress().committed, 0); assert_eq!(job.progress().attempts, 1);
        assert_eq!(job.progress().reserved_work, work());
        assert_eq!(fs::metadata(dir.join("results.spool")).unwrap().len(), 0);
        commit(&mut job, 0); assert_eq!(job.progress().attempts, 2);
    }
}
#[test]
fn lost_ack_after_journal_commit_does_not_reexecute_or_recharge() {
    let dir = root(); let mut job = create(&dir); job.fail_at = Some(Fault::ResultCommitted);
    let attempt = job.begin(&input(0), work(), &mut Control::default()).unwrap();
    assert_eq!(error(attempt.commit(&"committed", &mut Control::default())), JobError::Io);
    drop(job);
    let mut job = resume(&dir, TailPolicy::Refuse);
    assert_eq!(job.progress().committed, 1); assert_eq!(job.progress().attempts, 1);
    assert_eq!(job.read_committed(0, &mut Control::default()).unwrap(), b"\"committed\"");
    assert!(job.begin(&input(0), work(), &mut Control::default()).is_err());
    commit(&mut job, 1); assert_eq!(job.progress().attempts, 2);
}
#[test]
fn resume_refuses_tails_unless_discard_was_explicitly_requested() {
    let dir = root(); let mut job = create(&dir); commit(&mut job, 0);
    let committed_end = job.progress().spool_bytes; drop(job);
    append_tail(&dir, b"FNLPJOB1 complete-looking-or-torn-orphan");
    let before = fs::metadata(dir.join("results.spool")).unwrap().len();
    assert_eq!(error(OwnedJob::resume(&dir, key(), manifest(), TailPolicy::Refuse, &mut Control::default())), JobError::UncommittedTail);
    assert_eq!(fs::metadata(dir.join("results.spool")).unwrap().len(), before);
    let job = resume(&dir, TailPolicy::DiscardUncommitted);
    assert_eq!(job.progress().committed, 1); assert_eq!(fs::metadata(dir.join("results.spool")).unwrap().len(), committed_end);
}
#[test]
fn corrupt_committed_prefix_is_never_repaired_by_truncating_the_tail() {
    let dir = root(); let mut job = create(&dir); commit(&mut job, 0); drop(job);
    let mut spool = OpenOptions::new().write(true).open(dir.join("results.spool")).unwrap();
    spool.seek(SeekFrom::Start(frame::HEADER_BYTES as u64)).unwrap(); spool.write_all(b"X").unwrap(); spool.sync_all().unwrap();
    append_tail(&dir, b"tail");
    let bytes = fs::read(dir.join("results.spool")).unwrap();
    assert!(OwnedJob::resume(&dir, key(), manifest(), TailPolicy::DiscardUncommitted, &mut Control::default()).is_err());
    assert_eq!(fs::read(dir.join("results.spool")).unwrap(), bytes);
}
#[test]
fn wrong_secret_and_changed_manifest_cannot_repair_or_adopt_a_job() {
    let dir = root(); drop(create(&dir)); append_tail(&dir, b"uncommitted");
    let before = fs::read(dir.join("results.spool")).unwrap();
    assert!(OwnedJob::resume(&dir, JobSecret::from_bytes([99; 32]), manifest(), TailPolicy::DiscardUncommitted, &mut Control::default()).is_err());
    let mut cap = limits(); cap.max_attempts += 1;
    assert!(OwnedJob::resume(&dir, key(), manifest_with(cap), TailPolicy::DiscardUncommitted, &mut Control::default()).is_err());
    assert_eq!(fs::read(dir.join("results.spool")).unwrap(), before);
}
#[test]
fn attempt_and_model_mask_budgets_do_not_renew_on_resume() {
    for budget in 0..2 {
        let dir = root(); let mut cap = limits();
        if budget == 0 { cap.max_attempts = 2; } else { cap.max_work = work().checked_add(work()).unwrap(); }
        let mut job = OwnedJob::create(&dir, key(), manifest_with(cap), &mut Control::default()).unwrap();
        for _ in 0..2 {
            drop(job.begin(&input(0), work(), &mut Control::default()).unwrap()); drop(job);
            job = OwnedJob::resume(&dir, key(), manifest_with(cap), TailPolicy::Refuse, &mut Control::default()).unwrap();
        }
        assert_eq!(error(job.begin(&input(0), work(), &mut Control::default())), JobError::WorkLimit);
        assert_eq!(job.progress().attempts, 2);
    }
}
#[test]
fn canonical_materialization_matches_uninterrupted_and_resumed_execution() {
    let a = root(); let b = root();
    let mut job = create(&a); commit(&mut job, 0); commit(&mut job, 1);
    job.materialize_ordered(&mut Control::default()).unwrap();
    let mut job = create(&b); commit(&mut job, 0); drop(job);
    let mut job = resume(&b, TailPolicy::Refuse); commit(&mut job, 1);
    job.materialize_ordered(&mut Control::default()).unwrap();
    assert_eq!(fs::read(a.join("materialized.ndjson")).unwrap(), fs::read(b.join("materialized.ndjson")).unwrap());
    assert!(job.verify(&mut Control::default()).unwrap().materialized);
    drop(job); assert!(resume(&b, TailPolicy::Refuse).progress().materialized);
}
#[test]
fn crash_after_publication_adopts_only_exact_journal_derived_output() {
    let dir = root(); let mut job = create(&dir); commit(&mut job, 0); commit(&mut job, 1);
    job.fail_at = Some(Fault::Published);
    assert_eq!(error(job.materialize_ordered(&mut Control::default())), JobError::Io); drop(job);
    let original = fs::read(dir.join("materialized.ndjson")).unwrap();
    let mut job = resume(&dir, TailPolicy::Refuse);
    assert!(!job.progress().materialized); job.materialize_ordered(&mut Control::default()).unwrap();
    assert_eq!(fs::read(dir.join("materialized.ndjson")).unwrap(), original);
}
#[test]
fn foreign_output_is_never_replaced_and_partial_jobs_cannot_publish() {
    let dir = root(); let mut job = create(&dir); commit(&mut job, 0);
    assert_eq!(error(job.materialize_ordered(&mut Control::default())), JobError::Incomplete);
    commit(&mut job, 1);
    fs::write(dir.join("materialized.ndjson"), b"keep me").unwrap();
    fs::set_permissions(dir.join("materialized.ndjson"), fs::Permissions::from_mode(0o600)).unwrap();
    assert!(job.materialize_ordered(&mut Control::default()).is_err());
    assert_eq!(fs::read(dir.join("materialized.ndjson")).unwrap(), b"keep me");
}
#[test]
fn symlink_and_hardlink_spool_substitution_is_refused() {
    let dir = root(); drop(create(&dir));
    fs::rename(dir.join("results.spool"), dir.join("saved")).unwrap();
    symlink("saved", dir.join("results.spool")).unwrap();
    assert!(OwnedJob::resume(&dir, key(), manifest(), TailPolicy::Refuse, &mut Control::default()).is_err());
    fs::remove_file(dir.join("results.spool")).unwrap();
    fs::hard_link(dir.join("saved"), dir.join("results.spool")).unwrap();
    assert!(OwnedJob::resume(&dir, key(), manifest(), TailPolicy::Refuse, &mut Control::default()).is_err());
}
#[test]
fn cancellation_after_admission_preserves_the_typed_cause_and_debit() {
    let dir = root(); let mut job = create(&dir);
    assert_eq!(error(job.begin(&input(0), work(), &mut Control { calls: 0, stop_at: Some(2) })),
        JobError::Cancelled(crate::native_engine::decode::DecodeCancellationKind::Deadline));
    assert!(job.is_poisoned()); drop(job);
    assert_eq!(resume(&dir, TailPolicy::Refuse).progress().reserved_work, work());
}
#[test]
fn spool_length_caps_and_serialization_failure_never_commit_a_result() {
    let dir = root(); let mut job = create(&dir);
    let attempt = job.begin(&input(0), work(), &mut Control::default()).unwrap();
    assert_eq!(error(attempt.commit(&f64::INFINITY, &mut Control::default())), JobError::Serialization);
    drop(job); let job = resume(&dir, TailPolicy::Refuse);
    assert_eq!(job.progress().committed, 0); assert_eq!(job.progress().attempts, 1);
    drop(job);
    let dir = root(); let mut cap = limits(); cap.max_spool_bytes = frame::HEADER_BYTES as u64 + 1;
    let mut job = OwnedJob::create(&dir, key(), manifest_with(cap), &mut Control::default()).unwrap();
    let attempt = job.begin(&input(0), work(), &mut Control::default()).unwrap();
    assert_eq!(error(attempt.commit(&"too large", &mut Control::default())), JobError::Limit);
    drop(job);
    let job = OwnedJob::resume(&dir, key(), manifest_with(cap), TailPolicy::Refuse, &mut Control::default()).unwrap();
    assert_eq!(job.progress().committed, 0); assert_eq!(job.progress().spool_bytes, 0);
}
#[test]
fn authenticated_journal_refuses_injected_authority_json() {
    let dir = root(); drop(create(&dir));
    let db = fsqlite::Connection::open(dir.join("journal.fsqlite").to_str().unwrap()).unwrap();
    db.execute_batch("UPDATE job_header SET body = '{}' WHERE ordinal = 0").unwrap(); drop(db);
    assert!(OwnedJob::resume(&dir, key(), manifest(), TailPolicy::DiscardUncommitted, &mut Control::default()).is_err());
}

#[test]
fn interrupted_materialization_stage_requires_explicit_authenticated_cleanup() {
    let dir = root(); let mut job = create(&dir); commit(&mut job, 0); commit(&mut job, 1); drop(job);
    let stage = dir.join(".fnlp-redact-123-0.part");
    fs::write(&stage, b"partial output").unwrap();
    fs::set_permissions(&stage, fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(dir.join("unrelated.part"), b"keep").unwrap();
    assert_eq!(error(OwnedJob::resume(&dir, key(), manifest(), TailPolicy::Refuse, &mut Control::default())), JobError::UncommittedTail);
    assert!(stage.exists());
    let mut job = resume(&dir, TailPolicy::DiscardUncommitted);
    assert!(!stage.exists()); assert_eq!(fs::read(dir.join("unrelated.part")).unwrap(), b"keep");
    job.materialize_ordered(&mut Control::default()).unwrap();
}
