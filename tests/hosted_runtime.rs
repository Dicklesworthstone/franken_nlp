//! Actual process host + blocking pool; no model weights and no fake inference.
#![cfg(feature = "asupersync-runtime")]
use std::{path::PathBuf, sync::{Mutex, OnceLock}, time::Duration};
use franken_nlp::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy,
    hosted::{HostedError, LoadLimits, RunLimits, CancellationToken},
    native_engine::{artifact_bridge::ArtifactLoadBudget, decode::DecodeCancellationKind}};

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
