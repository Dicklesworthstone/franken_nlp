#![deny(unsafe_code)]

use std::path::Path;

use franken_nlp::artifact::fs_tx::{
    discover_activation, open_ratified_model_root, ActivationDigest, ActivationRecord,
    ActivationRecordBody, ChainWalkVerdict, FsTxError, NonReentrantContentLock,
    SimulatedActivationJournal, ACTIVATION_RECORD_DOMAIN,
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
            eprintln!(
                "FS_TXN case={case} RESULT={result} head={:?}",
                discovery.head.map(|head| head.sequence())
            );
        }
        Err(FsTxError::ActivationFork { walk, .. }) => {
            for entry in &walk {
                eprintln!(
                    "FS_TXN case={case} sequence={} digest={} verdict={:?}",
                    entry.sequence, entry.digest, entry.verdict
                );
            }
            eprintln!("FS_TXN case={case} RESULT={result} error=ActivationFork");
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
    assert!(record
        .canonical_envelope_bytes()
        .ends_with(&record.record_digest().as_bytes()));

    let mut journal = SimulatedActivationJournal::new();
    let first = journal.append(digest(1), digest(2), digest(3)).unwrap();
    let rollback = journal.append(digest(4), digest(5), digest(6)).unwrap();
    let activate = journal.append(digest(1), digest(2), digest(3)).unwrap();
    assert_eq!(
        [first.sequence(), rollback.sequence(), activate.sequence()],
        [0, 1, 2]
    );
    assert_eq!(
        journal.records().len(),
        3,
        "activate/rollback append, never overwrite"
    );
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
fn canonical_envelopes_round_trip_and_refuse_wire_mutations() {
    let genesis = ActivationRecord::new(ActivationRecordBody::genesis(
        digest(1),
        digest(2),
        digest(3),
    ));
    let successor = ActivationRecord::new(
        ActivationRecordBody::successor(&genesis, digest(4), digest(5), digest(6)).unwrap(),
    );

    for record in [&genesis, &successor] {
        let envelope = record.canonical_envelope_bytes();
        assert_eq!(
            ActivationRecord::parse_canonical_envelope(&envelope).unwrap(),
            *record,
            "the fixed-width retained envelope must retain every body field"
        );

        let mut truncated = envelope.clone();
        truncated.pop();
        assert!(matches!(
            ActivationRecord::parse_canonical_envelope(&truncated),
            Err(FsTxError::EnvelopeLength { observed }) if observed == truncated.len()
        ));

        let mut trailing = envelope;
        trailing.push(0);
        assert!(matches!(
            ActivationRecord::parse_canonical_envelope(&trailing),
            Err(FsTxError::EnvelopeLength { observed }) if observed == trailing.len()
        ));
    }

    let mut unsupported_version = genesis.canonical_envelope_bytes();
    unsupported_version[0] = 2;
    assert!(matches!(
        ActivationRecord::parse_canonical_envelope(&unsupported_version),
        Err(FsTxError::EnvelopeVersion { observed: 2 })
    ));

    let mut invalid_previous_flag = genesis.canonical_envelope_bytes();
    invalid_previous_flag[1 + 8 + (3 * 32)] = 2;
    assert!(matches!(
        ActivationRecord::parse_canonical_envelope(&invalid_previous_flag),
        Err(FsTxError::EnvelopePreviousFlag { observed: 2 })
    ));

    let mut forged_digest = successor.canonical_envelope_bytes();
    let final_byte = forged_digest.len() - 1;
    forged_digest[final_byte] ^= 1;
    assert!(matches!(
        ActivationRecord::parse_canonical_envelope(&forged_digest),
        Err(FsTxError::EnvelopeDigestMismatch)
    ));
    eprintln!("FS_TXN case=canonical-envelope-wire-refusal RESULT=PASS rows=2");
}

#[test]
fn immutable_final_filename_binds_the_canonical_envelope_identity() {
    let record = ActivationRecord::new(ActivationRecordBody::genesis(
        digest(1),
        digest(2),
        digest(3),
    ));
    let filename = record.final_filename();
    assert!(record.validate_final_filename(&filename).is_ok());

    let wrong_sequence = format!("00000000000000000001-{}.fnlpaj", record.record_digest());
    assert!(matches!(
        record.validate_final_filename(&wrong_sequence),
        Err(FsTxError::FinalFilenameBindingMismatch { .. })
    ));

    let wrong_digest = format!("00000000000000000000-{}.fnlpaj", digest(9));
    assert!(matches!(
        record.validate_final_filename(&wrong_digest),
        Err(FsTxError::FinalFilenameBindingMismatch { .. })
    ));

    for malformed in [
        "0-not-a-digest.fnlpaj",
        "00000000000000000000-ABCDEF.fnlpaj",
        "00000000000000000000-deadbeef.fnlpaj",
        "00000000000000000000-deadbeef.fnlpaj.tmp",
        "00000000000000000000-deadbeef/escape.fnlpaj",
    ] {
        assert!(matches!(
            record.validate_final_filename(malformed),
            Err(FsTxError::FinalFilenameInvalid { .. })
        ));
    }
    eprintln!("FS_TXN case=final-filename-envelope-binding RESULT=PASS rows=7");
}

#[test]
fn simulated_retained_ingress_requires_bound_filename_and_authenticated_body() {
    let record = ActivationRecord::new(ActivationRecordBody::genesis(
        digest(1),
        digest(2),
        digest(3),
    ));
    let filename = record.final_filename();
    let envelope = record.canonical_envelope_bytes();
    let mut journal = SimulatedActivationJournal::new();

    journal
        .retain_canonical_final_envelope(&filename, &envelope)
        .unwrap();
    assert_eq!(journal.records(), [record.clone()]);
    assert!(matches!(
        journal.retain_canonical_final_envelope(&filename, &envelope),
        Err(FsTxError::FinalNameExists { filename: rejected }) if rejected == filename
    ));

    let wrong_filename = format!("00000000000000000001-{}.fnlpaj", record.record_digest());
    assert!(matches!(
        journal.retain_canonical_final_envelope(&wrong_filename, &envelope),
        Err(FsTxError::FinalFilenameBindingMismatch { .. })
    ));

    let mut tampered = envelope;
    let last = tampered.len() - 1;
    tampered[last] ^= 1;
    assert!(matches!(
        journal.retain_canonical_final_envelope(&filename, &tampered),
        Err(FsTxError::EnvelopeDigestMismatch)
    ));
    assert_eq!(journal.records(), [record]);
    eprintln!("FS_TXN case=retained-envelope-ingress RESULT=PASS rows=4");
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

    let error = journal
        .discover()
        .expect_err("two successors must never be ordered into a winner");
    match error {
        FsTxError::ActivationFork {
            last_unambiguous,
            successor_digests,
            walk,
        } => {
            assert_eq!(last_unambiguous.map(|head| head.sequence()), Some(0));
            assert_eq!(successor_digests.len(), 2);
            assert!(walk
                .iter()
                .any(|entry| entry.verdict == ChainWalkVerdict::ForkSuccessor));
        }
        other => panic!("expected ActivationFork, observed {other:?}"),
    }
    assert_eq!(
        journal.records().len(),
        3,
        "fork evidence remains retained for forensics"
    );
    log_walk("forged-valid-successors", "FORK", &journal);
}

#[test]
fn append_refuses_an_existing_fork_without_mutating_forensic_records() {
    let mut journal = SimulatedActivationJournal::new();
    let genesis = journal.append(digest(1), digest(2), digest(3)).unwrap();
    let left = ActivationRecord::new(
        ActivationRecordBody::successor(&genesis.record, digest(4), digest(5), digest(6)).unwrap(),
    );
    let right = ActivationRecord::new(
        ActivationRecordBody::successor(&genesis.record, digest(7), digest(8), digest(9)).unwrap(),
    );
    journal.retain_recovery_fixture(left).unwrap();
    journal.retain_recovery_fixture(right).unwrap();
    let retained_before = journal.records().to_vec();

    let error = journal
        .append(digest(10), digest(11), digest(12))
        .expect_err("a forked journal has no append authority");
    assert!(matches!(
        error,
        FsTxError::ActivationFork {
            last_unambiguous: Some(_),
            ..
        }
    ));
    assert_eq!(
        journal.records(),
        retained_before.as_slice(),
        "append refusal must preserve all competing retained evidence"
    );
    log_walk("append-refuses-existing-fork", "FORK", &journal);
}

#[test]
fn append_ignores_a_disconnected_record_and_starts_the_unique_genesis_chain() {
    let mut journal = SimulatedActivationJournal::new();
    let orphan = ActivationRecord::new(ActivationRecordBody::from_retained_parts(
        7,
        digest(1),
        digest(2),
        digest(3),
        Some(digest(4)),
    ));
    let orphan_digest = orphan.record_digest();
    journal.retain_recovery_fixture(orphan).unwrap();

    let genesis = journal.append(digest(5), digest(6), digest(7)).unwrap();
    assert_eq!(genesis.sequence(), 0);
    let discovery = journal.discover().unwrap();
    assert_eq!(discovery.head.as_ref().map(|head| head.digest()), Some(genesis.digest()));
    assert!(discovery.walk.iter().any(|entry| {
        entry.digest == orphan_digest && entry.verdict == ChainWalkVerdict::IgnoredDisconnected
    }));
    assert_eq!(
        journal.records().len(),
        2,
        "the orphan remains available for quarantine or forensic inspection"
    );
    log_walk("append-ignores-disconnected-record", "PASS", &journal);
}

#[test]
fn fork_walk_is_stable_across_retained_record_order() {
    let genesis = ActivationRecord::new(ActivationRecordBody::genesis(
        digest(1),
        digest(2),
        digest(3),
    ));
    let left = ActivationRecord::new(
        ActivationRecordBody::successor(&genesis, digest(4), digest(5), digest(6)).unwrap(),
    );
    let right = ActivationRecord::new(
        ActivationRecordBody::successor(&genesis, digest(7), digest(8), digest(9)).unwrap(),
    );
    let disconnected = ActivationRecord::new(ActivationRecordBody::from_retained_parts(
        7,
        digest(10),
        digest(11),
        digest(12),
        Some(genesis.record_digest()),
    ));

    let walk_for = |records: &[ActivationRecord]| match discover_activation(records) {
        Err(FsTxError::ActivationFork { walk, .. }) => walk,
        other => panic!("expected ActivationFork, observed {other:?}"),
    };
    let first = walk_for(&[
        genesis.clone(),
        right.clone(),
        disconnected.clone(),
        left.clone(),
    ]);
    let second = walk_for(&[left, disconnected, genesis, right]);
    assert_eq!(
        first, second,
        "forensic diagnostics must not depend on directory enumeration order"
    );
    assert_eq!(
        first
            .iter()
            .map(|entry| (entry.sequence, entry.digest, entry.verdict as u8))
            .collect::<Vec<_>>(),
        {
            let mut sorted = first
                .iter()
                .map(|entry| (entry.sequence, entry.digest, entry.verdict as u8))
                .collect::<Vec<_>>();
            sorted.sort();
            sorted
        },
        "fork walk must be canonically sorted"
    );
    eprintln!(
        "FS_TXN case=stable-fork-walk RESULT=PASS rows={}",
        first.len()
    );
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
    let disconnected = ActivationRecord::new(ActivationRecordBody::genesis(
        digest(7),
        digest(8),
        digest(9),
    ));
    journal.retain_recovery_fixture(torn).unwrap();
    journal.retain_recovery_fixture(disconnected).unwrap();

    let discovery = journal
        .discover()
        .expect_err("two valid genesis records are an explicit fork");
    assert!(matches!(
        discovery,
        FsTxError::ActivationFork {
            last_unambiguous: None,
            ..
        }
    ));

    let gapped = ActivationRecord::new(ActivationRecordBody::from_retained_parts(
        7,
        digest(10),
        digest(11),
        digest(12),
        Some(genesis.digest()),
    ));
    let gapped_discovery = discover_activation(&[genesis.record.clone(), gapped]).unwrap();
    assert_eq!(
        gapped_discovery.head.as_ref().map(|head| head.sequence()),
        Some(0),
        "a sequence gap is disconnected, never a promoted head"
    );
    assert!(gapped_discovery
        .walk
        .iter()
        .any(|entry| entry.verdict == ChainWalkVerdict::IgnoredDisconnected));

    let only_torn = vec![ActivationRecord::from_retained_parts(
        valid_successor.body().clone(),
        ActivationDigest::from_bytes([0xfe; 32]),
    )];
    let empty = discover_activation(&only_torn).unwrap();
    assert_eq!(empty.head, None);
    assert_eq!(
        empty.walk[0].verdict,
        ChainWalkVerdict::IgnoredDigestMismatch
    );
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
    struct CrashCase {
        stage: &'static str,
        retained_envelope: RetainedEnvelope,
        expected_head_delta: u64,
        expected_new_verdict: Option<ChainWalkVerdict>,
    }

    #[derive(Clone, Copy, Debug)]
    enum RetainedEnvelope {
        /// Staging names are never discovery candidates, even after a synced
        /// file write. The old immutable head must remain active.
        NotVisible,
        /// A completed same-filesystem rename has made the immutable final
        /// envelope discoverable, so recovery may adopt the new head even
        /// before a later directory-sync observation.
        ValidFinal,
        /// A power cut may leave bytes that resemble a final envelope but do
        /// not authenticate. Discovery must ignore them, not promote them.
        TornFinal,
    }

    let cases = [
        CrashCase {
            stage: "create-staging",
            retained_envelope: RetainedEnvelope::NotVisible,
            expected_head_delta: 0,
            expected_new_verdict: None,
        },
        CrashCase {
            stage: "sync-staging",
            retained_envelope: RetainedEnvelope::NotVisible,
            expected_head_delta: 0,
            expected_new_verdict: None,
        },
        CrashCase {
            stage: "rename-before-visibility",
            retained_envelope: RetainedEnvelope::NotVisible,
            expected_head_delta: 0,
            expected_new_verdict: None,
        },
        CrashCase {
            stage: "rename-torn-final",
            retained_envelope: RetainedEnvelope::TornFinal,
            expected_head_delta: 0,
            expected_new_verdict: Some(ChainWalkVerdict::IgnoredDigestMismatch),
        },
        CrashCase {
            stage: "rename-visible-final",
            retained_envelope: RetainedEnvelope::ValidFinal,
            expected_head_delta: 1,
            expected_new_verdict: Some(ChainWalkVerdict::Adopted),
        },
        CrashCase {
            stage: "sync-directory",
            retained_envelope: RetainedEnvelope::ValidFinal,
            expected_head_delta: 1,
            expected_new_verdict: Some(ChainWalkVerdict::Adopted),
        },
    ];

    for (logical_timestamp, case) in cases.into_iter().enumerate() {
        let mut journal = SimulatedActivationJournal::new();
        let old = journal.append(digest(1), digest(2), digest(3)).unwrap();
        let candidate = ActivationRecord::new(
            ActivationRecordBody::successor(&old.record, digest(4), digest(5), digest(6))
                .expect("the genesis record has a checked successor"),
        );
        let expected_candidate = match case.retained_envelope {
            RetainedEnvelope::NotVisible => None,
            RetainedEnvelope::ValidFinal => {
                let digest = candidate.record_digest();
                journal
                    .retain_recovery_fixture(candidate)
                    .expect("the first immutable final name is available");
                Some((digest, ChainWalkVerdict::Adopted))
            }
            RetainedEnvelope::TornFinal => {
                let torn = ActivationRecord::from_retained_parts(
                    candidate.body().clone(),
                    ActivationDigest::from_bytes([0xa5; 32]),
                );
                let digest = torn.record_digest();
                journal
                    .retain_recovery_fixture(torn)
                    .expect("the distinct torn envelope name is retained for recovery");
                Some((digest, ChainWalkVerdict::IgnoredDigestMismatch))
            }
        };

        let discovery = journal.discover().unwrap();
        let observed = discovery.head.expect("old or new retained head");
        assert_eq!(
            observed.sequence(),
            old.sequence() + case.expected_head_delta,
            "crash recovery must name the exact old/new visibility outcome"
        );
        if let Some(expected_verdict) = case.expected_new_verdict {
            let (candidate_digest, candidate_verdict) = expected_candidate
                .expect("a retained crash envelope must have a walk classification");
            assert_eq!(candidate_verdict, expected_verdict);
            assert!(
                discovery
                    .walk
                    .iter()
                    .any(|entry| entry.digest == candidate_digest && entry.verdict == expected_verdict),
                "the crash walk must classify the visible candidate: stage={} expected={expected_verdict:?} walk={:?}",
                case.stage,
                discovery.walk
            );
        } else {
            assert!(
                expected_candidate.is_none(),
                "an invisible staging record must not enter discovery"
            );
        }
        eprintln!(
            "FS_TXN case=crash-{} logical_timestamp={logical_timestamp} expected_head={} observed_head={} retained={:?} RESULT=PASS",
            case.stage,
            old.sequence() + case.expected_head_delta,
            observed.sequence(),
            case.retained_envelope,
        );
    }
    eprintln!("FS_TXN_CRASH_MATRIX RESULT=PASS rows=6 forks=0 model=simulated-failure-fs");
}
