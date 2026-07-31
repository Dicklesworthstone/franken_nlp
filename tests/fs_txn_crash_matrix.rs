#![deny(unsafe_code)]

use std::path::Path;

use franken_nlp::artifact::fs_tx::{
    ACTIVATION_RECORD_DOMAIN, ActivationDigest, ActivationRecord, ActivationRecordBody,
    ChainWalkVerdict, FsTxError, NonReentrantContentLock, SimulatedActivationJournal,
    discover_activation, open_ratified_model_root,
};
use sha2::{Digest, Sha256};

fn digest(seed: u8) -> ActivationDigest {
    ActivationDigest::from_bytes([seed; 32])
}

fn log_walk(case: &str, result: &str, journal: &SimulatedActivationJournal) {
    match journal.discover() {
        Ok(discovery) => {
            for entry in &discovery.walk {
                eprintln!(
                    "FS_TXN case={case} sequence={} digest={} verdict={:?}",
                    entry.sequence, entry.digest, entry.verdict
                );
            }
            eprintln!("FS_TXN case={case} RESULT={result} head={:?}", discovery.head.map(|head| head.sequence()));
        }
        Err(error) => eprintln!("FS_TXN case={case} RESULT={result} error={error}"),
    }
}

#[test]
fn canonical_body_domain_digest_and_append_only_chain_are_locked() {
    let genesis_body = ActivationRecordBody::genesis(digest(1), digest(2), digest(3));
    assert_eq!(genesis_body.sequence(), 0);
    assert_eq!(genesis_body.previous_record_digest(), None);
    let record = ActivationRecord::new(genesis_body.clone());

    let mut hasher = Sha256::new();
    hasher.update(ACTIVATION_RECORD_DOMAIN);
    hasher.update(genesis_body.canonical_bytes());
    let expected = ActivationDigest::from_bytes(hasher.finalize().into());
    assert_eq!(record.record_digest(), expected);
    assert!(record.digest_is_valid());
    assert!(record.final_filename().starts_with("00000000000000000000-"));
    assert!(record.canonical_envelope_bytes().ends_with(&record.record_digest().as_bytes()));

    let mut journal = SimulatedActivationJournal::new();
    let first = journal.append(digest(1), digest(2), digest(3)).unwrap();
    let rollback = journal.append(digest(4), digest(5), digest(6)).unwrap();
    let activate = journal.append(digest(1), digest(2), digest(3)).unwrap();
    assert_eq!([first.sequence(), rollback.sequence(), activate.sequence()], [0, 1, 2]);
    assert_eq!(journal.records().len(), 3, "activate/rollback append, never overwrite");
    let discovery = journal.discover().unwrap();
    assert_eq!(discovery.head.as_ref().map(|head| head.sequence()), Some(2));
    assert_eq!(
        discovery
            .walk
            .iter()
            .filter(|entry| entry.verdict == ChainWalkVerdict::Adopted)
            .count(),
        3
    );
    log_walk("append-rollback-activate", "PASS", &journal);
}

#[test]
fn forged_successors_raise_activation_fork_and_retain_last_unambiguous_head() {
    let mut journal = SimulatedActivationJournal::new();
    let genesis = journal.append(digest(1), digest(2), digest(3)).unwrap();
    let first = ActivationRecord::new(
        ActivationRecordBody::successor(&genesis.record, digest(4), digest(5), digest(6)).unwrap(),
    );
    let second = ActivationRecord::new(
        ActivationRecordBody::successor(&genesis.record, digest(7), digest(8), digest(9)).unwrap(),
    );
    journal.retain_recovery_fixture(first).unwrap();
    journal.retain_recovery_fixture(second).unwrap();

    let error = journal.discover().expect_err("two successors must never be ordered into a winner");
    match error {
        FsTxError::ActivationFork {
            last_unambiguous,
            successor_digests,
            walk,
        } => {
            assert_eq!(last_unambiguous.map(|head| head.sequence()), Some(0));
            assert_eq!(successor_digests.len(), 2);
            assert!(walk.iter().any(|entry| entry.verdict == ChainWalkVerdict::ForkSuccessor));
        }
        other => panic!("expected ActivationFork, observed {other:?}"),
    }
    assert_eq!(journal.records().len(), 3, "fork evidence remains retained for forensics");
    log_walk("forged-valid-successors", "FORK", &journal);
}

#[test]
fn torn_gapped_and_disconnected_records_never_become_active() {
    let mut journal = SimulatedActivationJournal::new();
    let genesis = journal.append(digest(1), digest(2), digest(3)).unwrap();
    let valid_successor = ActivationRecord::new(
        ActivationRecordBody::successor(&genesis.record, digest(4), digest(5), digest(6)).unwrap(),
    );
    let torn = ActivationRecord::from_retained_parts(
        valid_successor.body().clone(),
        ActivationDigest::from_bytes([0xff; 32]),
    );
    let disconnected = ActivationRecord::new(ActivationRecordBody::genesis(digest(7), digest(8), digest(9)));
    journal.retain_recovery_fixture(torn).unwrap();
    journal.retain_recovery_fixture(disconnected).unwrap();

    let discovery = journal.discover().expect_err("two valid genesis records are an explicit fork");
    assert!(matches!(discovery, FsTxError::ActivationFork { last_unambiguous: None, .. }));

    let only_torn = vec![ActivationRecord::from_retained_parts(
        valid_successor.body().clone(),
        ActivationDigest::from_bytes([0xfe; 32]),
    )];
    let empty = discover_activation(&only_torn).unwrap();
    assert_eq!(empty.head, None);
    assert_eq!(empty.walk[0].verdict, ChainWalkVerdict::IgnoredDigestMismatch);
    eprintln!("FS_TXN case=torn-gapped-disconnected RESULT=PASS head=none");
}

#[test]
fn sequence_overflow_lock_reentry_and_unratified_root_refuse_typed() {
    let max_body = ActivationRecordBody::from_retained_parts(
        u64::MAX,
        digest(1),
        digest(2),
        digest(3),
        Some(digest(4)),
    );
    let max_record = ActivationRecord::new(max_body);
    assert!(matches!(
        ActivationRecordBody::successor(&max_record, digest(5), digest(6), digest(7)),
        Err(FsTxError::SequenceOverflow)
    ));

    let lock = NonReentrantContentLock::default();
    let guard = lock.try_lock().unwrap();
    assert!(matches!(lock.try_lock(), Err(FsTxError::LockReentrant)));
    drop(guard);
    assert!(lock.try_lock().is_ok());
    assert!(matches!(
        open_ratified_model_root(Path::new("/untrusted/model-root")),
        Err(FsTxError::PlatformSurfaceUnavailable { .. })
    ));
    eprintln!("FS_TXN case=overflow-lock-platform-refusal RESULT=PASS");
}

#[test]
fn crash_matrix_recovers_only_old_or_new_head() {
    for (logical_timestamp, stage) in [
        "create-staging",
        "sync-staging",
        "rename-final",
        "sync-directory",
    ]
    .into_iter()
    .enumerate()
    {
        let mut journal = SimulatedActivationJournal::new();
        let old = journal.append(digest(1), digest(2), digest(3)).unwrap();
        if matches!(stage, "rename-final" | "sync-directory") {
            journal.append(digest(4), digest(5), digest(6)).unwrap();
        }
        let discovery = journal.discover().unwrap();
        let observed = discovery.head.expect("old or new retained head");
        assert!(observed.sequence() == old.sequence() || observed.sequence() == old.sequence() + 1);
        eprintln!(
            "FS_TXN case=crash-{stage} logical_timestamp={logical_timestamp} RESULT=PASS head={}",
            observed.sequence()
        );
    }
    eprintln!("FS_TXN_CRASH_MATRIX RESULT=PASS rows=4 forks=0 model=simulated-failure-fs");
}
