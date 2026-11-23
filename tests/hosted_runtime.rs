//! Actual process host + blocking pool; no model weights and no fake inference.
#![cfg(feature = "asupersync-runtime")]
use std::{path::PathBuf, sync::{Mutex, OnceLock}, time::Duration};
use franken_nlp::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy,
    hosted::{HostedError, LoadLimits, RunLimits, CancellationToken},
    native_engine::{artifact_bridge::{ArtifactBridgeError, ArtifactLoadBudget},
        decode::DecodeCancellationKind, strict_int8::CurrentCandidateInt8Error}};

static SERIAL: Mutex<()> = Mutex::new(());
static ENGINE: OnceLock<NlpEngine> = OnceLock::new();
fn engine() -> &'static NlpEngine {
    ENGINE.get_or_init(|| NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: 512 * 1024 * 1024,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap())
}
fn limits() -> LoadLimits {
    LoadLimits { artifact: ArtifactLoadBudget::streaming_only(1024 * 1024, 64 * 1024),
        tokenizer_and_metadata_bytes: 128 * 1024 * 1024, allocator_reserve_bytes: 32 * 1024 * 1024,
        run: RunLimits { max_elapsed: Duration::from_secs(120), max_checkpoints: 100, cleanup_reserve_bytes: 65536 } }
}
fn missing() -> PathBuf {
    let path = std::env::temp_dir().join(format!("fnlp-hosted-absent-{}-candidate.fnlpq", std::process::id()));
    assert!(!path.exists(), "test refuses to use an existing local path"); path
}
fn assert_drained(engine: &NlpEngine) {
    let memory = engine.resources().memory_snapshot();
    assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.committed_bytes, 0);
    assert_eq!(memory.outstanding_obligations, 0);
    assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(engine.resources().outstanding_closure_snapshot().active_closures, 0);
}

#[test]
fn missing_file_runs_the_real_loader_and_drains_before_returning() {
    let _serial = SERIAL.lock().unwrap(); let engine = engine();
    let error = engine.load_current_candidate_int8(missing(), limits(), CancellationToken::default()).err().unwrap();
    assert!(matches!(error, HostedError::Model(_)), "{error}");
    assert_drained(engine);
}

#[test]
fn caller_cancellation_before_launch_never_falls_through_to_file_loading() {
    let _serial = SERIAL.lock().unwrap(); let engine = engine();
    let stop = CancellationToken::default(); stop.cancel(DecodeCancellationKind::User);
    let error = engine.load_current_candidate_int8(missing(), limits(), stop).err().unwrap();
    match error { HostedError::Stopped { stop, task_error } => {
        assert_eq!(stop.kind, DecodeCancellationKind::User); assert!(task_error.is_none());
    }, other => panic!("wrong failure category: {other}") }
    assert_drained(engine);
}

#[test]
fn pull_control_is_not_reset_at_admission_or_runtime_entry_and_every_guard_drains() {
    use franken_nlp::{
        artifact::pull::http::{NetworkPartsOptions, NetworkPartsRequest, NetworkPullError,
            direct::NetworkArtifactRequest},
        hosted::{HostedNetworkPullError, NetworkPullHostOptions},
        native_engine::decode::DecodeStepControl,
    };
    use asupersync::CancelKind;
    struct StopAt { checks: usize, stop_at: usize, cause: DecodeCancellationKind }
    impl DecodeStepControl for StopAt {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.checks += 1;
            (self.checks >= self.stop_at).then_some(self.cause)
        }
    }
    let _serial = SERIAL.lock().unwrap();
    let engine = engine();
    for (cause, expected) in [
        (DecodeCancellationKind::User, CancelKind::User),
        (DecodeCancellationKind::Timeout, CancelKind::Timeout),
        (DecodeCancellationKind::Deadline, CancelKind::Deadline),
        (DecodeCancellationKind::PollQuota, CancelKind::PollQuota),
        (DecodeCancellationKind::CostBudget, CancelKind::CostBudget),
        (DecodeCancellationKind::FailFast, CancelKind::FailFast),
        (DecodeCancellationKind::RaceLost, CancelKind::RaceLost),
        (DecodeCancellationKind::ParentCancelled, CancelKind::ParentCancelled),
        (DecodeCancellationKind::ResourceUnavailable, CancelKind::ResourceUnavailable),
        (DecodeCancellationKind::Shutdown, CancelKind::Shutdown),
        (DecodeCancellationKind::LinkedExit, CancelKind::LinkedExit),
    ] {
        for stop_at in 2..=4 {
            for direct in [false, true] {
                let mut control = StopAt { checks: 0, stop_at, cause };
                // Represent prior manifest/cache/preparation work. Admission
                // must borrow this state rather than creating a fresh budget.
                assert_eq!(control.checkpoint(0), None);
                let result = if direct {
                    engine.acquire_release_artifact_candidate(NetworkArtifactRequest {
                        manifest: PathBuf::new(), expected_release_manifest_sha256: String::new(),
                        staging_output: PathBuf::new(), output: PathBuf::new(),
                    }, NetworkPartsOptions::default(), NetworkPullHostOptions::default(), &mut control)
                        .map(|_| ())
                } else {
                    engine.download_release_parts_candidate(NetworkPartsRequest {
                        manifest: PathBuf::new(), expected_release_manifest_sha256: String::new(),
                        parts_dir: PathBuf::new(),
                    }, NetworkPartsOptions::default(), NetworkPullHostOptions::default(), &mut control)
                        .map(|_| ())
                };
                assert!(matches!(result, Err(HostedNetworkPullError::Transfer(NetworkPullError::Cancelled {
                    cause: Some(observed), part: None,
                })) if observed == expected));
                assert_eq!(control.checks, stop_at);
                assert_drained(engine);
            }
        }
    }
}

#[test]
fn over_budget_load_rolls_back_the_entire_reservation_without_native_work() {
    let _serial = SERIAL.lock().unwrap(); let engine = engine(); let mut limits = limits();
    limits.artifact.max_weight_bytes = engine.resources().config().memory_ceiling_bytes;
    let error = engine.load_current_candidate_int8(missing(), limits, CancellationToken::default()).err().unwrap();
    assert!(matches!(error, HostedError::Reservation(_)));
    assert_drained(engine);
}

#[test]
fn synchronous_reentry_is_rejected_before_new_reservations_or_work() {
    let _serial = SERIAL.lock().unwrap(); let engine = engine();
    let _entry = engine.enter_sync_call().unwrap();
    let error = engine.load_current_candidate_int8(missing(), limits(), CancellationToken::default()).err().unwrap();
    assert!(matches!(error, HostedError::Reentrant));
    assert_drained(engine);
}

#[test]
fn checkpoint_budget_stops_inside_candidate_loading_and_drains_every_guard() {
    let _serial = SERIAL.lock().unwrap();
    let engine = engine();
    let mut limits = limits();
    // Dispatch consumes the first checkpoint and model loading the second.
    // The reader must then stop before the absent artifact path is inspected.
    limits.run.max_checkpoints = 2;
    let error = engine.load_current_candidate_int8(
        missing(), limits, CancellationToken::default(),
    ).err().unwrap();
    match error {
        HostedError::Stopped { stop, task_error } => {
            assert_eq!(stop.kind, DecodeCancellationKind::PollQuota);
            assert!(matches!(
                task_error.as_deref(),
                Some(HostedError::Model(CurrentCandidateInt8Error::Artifact(
                    ArtifactBridgeError::Cancelled(DecodeCancellationKind::PollQuota)
                )))
            ), "loader must retain its typed cancellation: {task_error:?}");
        }
        other => panic!("loading ignored the caller's checkpoint budget: {other}"),
    }
    assert_drained(engine);
}
