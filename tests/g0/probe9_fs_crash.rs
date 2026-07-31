//! G0-09 kill-matrix model for model-root activation.
//!
//! The model fixes the required recovery observations before platform-specific
//! child-process kill injection is run.  It does not claim Linux, macOS, or
//! Windows durability: `ADR-G0-09-model-root-crash.md` remains BLOCKED until
//! each target-family transcript is digested and appended.

const SEED: u64 = 0x4730_3039;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ActivationStep {
    AcquireContentLock,
    CreateNewStaging,
    WriteAndSyncStaging,
    SyncParent,
    RenameIntoPlace,
    SyncModelRoot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TargetKind {
    RegularFile,
    Symlink,
    ReparsePoint,
    Device,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RecoveryObservation {
    previous_root_visible: bool,
    candidate_root_visible: bool,
    staging_visible: bool,
}

fn accepts_staging_target(kind: TargetKind, same_filesystem: bool, create_new: bool) -> bool {
    kind == TargetKind::RegularFile && same_filesystem && create_new
}

fn recover_after_kill(step: ActivationStep) -> RecoveryObservation {
    let candidate_root_visible = matches!(step, ActivationStep::RenameIntoPlace | ActivationStep::SyncModelRoot);
    RecoveryObservation {
        previous_root_visible: !candidate_root_visible,
        candidate_root_visible,
        staging_visible: false,
    }
}

fn log_case(id: &str, result: &str) {
    println!("G0_PROBE9 case={id} RESULT={result} seed={SEED}");
}

#[test]
fn kill_matrix_model_keeps_only_previous_or_candidate_root_visible() {
    for step in [
        ActivationStep::AcquireContentLock,
        ActivationStep::CreateNewStaging,
        ActivationStep::WriteAndSyncStaging,
        ActivationStep::SyncParent,
        ActivationStep::RenameIntoPlace,
        ActivationStep::SyncModelRoot,
    ] {
        let observed = recover_after_kill(step);
        assert_ne!(
            observed.previous_root_visible, observed.candidate_root_visible,
            "recovery must expose exactly one complete root"
        );
        assert!(!observed.staging_visible, "staging must never activate");
        log_case(
            match step {
                ActivationStep::AcquireContentLock => "kill-after-lock",
                ActivationStep::CreateNewStaging => "kill-after-create-new",
                ActivationStep::WriteAndSyncStaging => "kill-after-staging-sync",
                ActivationStep::SyncParent => "kill-after-parent-sync",
                ActivationStep::RenameIntoPlace => "kill-after-rename",
                ActivationStep::SyncModelRoot => "kill-after-root-sync",
            },
            "PASS",
        );
    }

    assert!(accepts_staging_target(TargetKind::RegularFile, true, true));
    for target in [TargetKind::Symlink, TargetKind::ReparsePoint, TargetKind::Device] {
        assert!(!accepts_staging_target(target, true, true));
    }
    assert!(!accepts_staging_target(TargetKind::RegularFile, false, true));
    assert!(!accepts_staging_target(TargetKind::RegularFile, true, false));
    log_case("reject-link-reparse-device-crossfs-and-nonexclusive-create", "PASS");

    println!("G0_PROBE9 RESULT=PASS cases=7 seed={SEED} authority=kill-matrix-model-only");
}
