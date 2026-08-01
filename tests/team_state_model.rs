#![deny(unsafe_code)]

//! Bounded native-team protocol model for the 7eu sealed-formation contract.
//!
//! Asupersync's deterministic lab owns async region/obligation schedules, but
//! native `scoped_cpu` children are OS threads. These tests keep the formation,
//! entry-gate, drain, panic, and completion-latch protocol independently
//! enumerable before the feature-enabled seam stress runs in the controller
//! batch.

use franken_nlp::orchestrator::{
    CapacityOneLaneError, SealedCpuTeam, SealedTeamError, SealedTeamPhase, TeamDrainReason,
    capacity_one_lane, scoped_cpu_child_cap,
};

#[test]
fn effective_team_width_reserves_the_coordinator_shard() {
    assert_eq!(scoped_cpu_child_cap(0), None);
    assert_eq!(scoped_cpu_child_cap(1), Some(0));
    assert_eq!(scoped_cpu_child_cap(4), Some(3));
}

#[test]
fn formation_seal_refuses_post_start_and_post_latch_workers() {
    let team = SealedCpuTeam::new(2);
    let first = team.form_worker().expect("first worker forms");
    assert!(matches!(
        team.seal(),
        Err(SealedTeamError::SealBeforeCompleteFormation {
            expected_children: 2,
            formed_children: 1,
        })
    ));

    let second = team.form_worker().expect("second worker forms");
    team.seal().expect("complete formation seals exactly once");
    assert!(matches!(
        team.form_worker(),
        Err(SealedTeamError::SpawnAfterSeal {
            phase: SealedTeamPhase::Sealed,
        })
    ));
    assert!(matches!(
        team.worker_started(first),
        Err(SealedTeamError::WorkerStartBeforeRelease {
            phase: SealedTeamPhase::Sealed,
        })
    ));

    team.release_workers().expect("sealed team releases entry gate");
    team.worker_started(first).expect("first worker starts after gate");
    team.worker_started(second)
        .expect("second worker starts after gate");
    team.begin_drain(TeamDrainReason::Cancelled);
    assert!(matches!(
        team.form_worker(),
        Err(SealedTeamError::SpawnAfterSeal {
            phase: SealedTeamPhase::Draining,
        })
    ));
    team.worker_exited(first).expect("first worker drains");
    team.worker_exited(second).expect("second worker drains");
    team.join().expect("latch fires after both workers exit");
    assert_eq!(team.snapshot().phase, SealedTeamPhase::Joined);
    assert!(matches!(
        team.form_worker(),
        Err(SealedTeamError::SpawnAfterSeal {
            phase: SealedTeamPhase::Joined,
        })
    ));
}

#[test]
fn panic_drain_preserves_the_latch_until_the_scope_join() {
    let team = SealedCpuTeam::new(2);
    let first = team.form_worker().expect("first worker forms");
    let second = team.form_worker().expect("second worker forms");
    team.seal().expect("team seals");
    team.release_workers().expect("entry gate opens");
    team.worker_started(first).expect("first worker starts");
    team.worker_started(second).expect("second worker starts");

    team.worker_panicked(first)
        .expect("contained child panic starts drain and exits child");
    assert_eq!(
        team.snapshot().drain_reason,
        Some(TeamDrainReason::WorkerPanicked)
    );
    assert!(matches!(
        team.join(),
        Err(SealedTeamError::JoinBeforeWorkersExit {
            expected_children: 2,
            exited_children: 1,
        })
    ));
    team.worker_exited(second)
        .expect("sibling observes drain at checkpoint");
    team.join().expect("scope join fires completion latch");

    let snapshot = team.snapshot();
    assert_eq!(snapshot.phase, SealedTeamPhase::Joined);
    assert_eq!(snapshot.exited_children, 2);
    assert_eq!(snapshot.drain_reason, Some(TeamDrainReason::WorkerPanicked));
}

#[test]
fn capacity_one_lane_refuses_a_second_in_flight_command() {
    let (sender, receiver) = capacity_one_lane();
    sender.try_send(7_u8).expect("first command occupies lane");
    assert_eq!(sender.try_send(8_u8), Err(CapacityOneLaneError::Full));
    assert_eq!(receiver.recv().expect("worker receives first command"), 7);
    sender
        .try_send(8_u8)
        .expect("drained lane accepts next command deterministically");
    assert_eq!(receiver.recv().expect("worker receives next command"), 8);
}

#[test]
fn bounded_model_enumerates_all_terminal_drain_reasons_without_late_spawn() {
    for reason in [
        TeamDrainReason::Completed,
        TeamDrainReason::Cancelled,
        TeamDrainReason::WorkerPanicked,
        TeamDrainReason::CoordinatorPanicked,
    ] {
        let team = SealedCpuTeam::new(1);
        let worker = team.form_worker().expect("one fixed child forms");
        team.seal().expect("formation seals");
        team.release_workers().expect("entry gate releases");
        team.worker_started(worker).expect("worker starts");
        team.begin_drain(reason);
        assert!(matches!(
            team.form_worker(),
            Err(SealedTeamError::SpawnAfterSeal {
                phase: SealedTeamPhase::Draining,
            })
        ));
        team.worker_exited(worker).expect("worker exits at drain boundary");
        team.join().expect("all children join before latch fires");
        assert_eq!(team.snapshot().drain_reason, Some(reason));
    }
}
