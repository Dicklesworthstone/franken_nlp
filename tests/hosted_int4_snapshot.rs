//! Model-free failure/admission regressions on the actual process-owned host.
#![cfg(feature = "asupersync-runtime")]
use std::{path::PathBuf, sync::{Mutex, OnceLock}, time::Duration};
use franken_nlp::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy,
    hosted::{HostedError, Int4HostError, RunLimits, CancellationToken},
    native_engine::{decode::DecodeCancellationKind, strict_int4::weights::snapshot::{
        Int4SnapshotBudget, Int4SnapshotPin, Int4SnapshotRequirements, Int4SnapshotError}}};
static SERIAL: Mutex<()> = Mutex::new(());
static ENGINE: OnceLock<NlpEngine> = OnceLock::new();
fn engine() -> &'static NlpEngine {
    ENGINE.get_or_init(|| NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: 6 * 1024 * 1024 * 1024,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap())
}
fn pin() -> Int4SnapshotPin { Int4SnapshotPin { group_width: 128, model_content_sha256: [1; 32] } }
fn budget() -> Int4SnapshotBudget {
    let need = Int4SnapshotRequirements::for_group_width(128).unwrap();
    Int4SnapshotBudget { max_weight_bytes: need.weight_payload_bytes, max_scratch_bytes: need.scratch_bytes,
        max_io_bytes: need.read_io_bytes().unwrap() }
}
fn run() -> RunLimits { RunLimits { max_elapsed: Duration::from_secs(120), max_checkpoints: 100, cleanup_reserve_bytes: 65536 } }
fn missing() -> PathBuf {
    let path = std::env::temp_dir().join(format!("fnlp-missing-int4-snapshot-{}", std::process::id()));
    assert!(!path.exists(), "test refuses to touch an existing snapshot"); path
}
fn drained(engine: &NlpEngine) {
    let memory = engine.resources().memory_snapshot();
    assert_eq!(memory.reserved_bytes, 0); assert_eq!(memory.committed_bytes, 0);
    assert_eq!(memory.outstanding_obligations, 0); assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(engine.resources().outstanding_closure_snapshot().active_closures, 0);
}
#[test]
fn absent_snapshot_uses_real_dispatch_and_drains_resident_and_staging_charges() {
    let _serial = SERIAL.lock().unwrap(); let engine = engine();
    let error = engine.load_int4_snapshot(missing(), pin(), budget(), 64 << 20, run(), CancellationToken::default()).err().unwrap();
    assert!(matches!(error, HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Io)))); drained(engine);
}
#[test]
fn pre_cancelled_snapshot_load_never_opens_a_file_or_retains_a_model_claim() {
    let _serial = SERIAL.lock().unwrap(); let engine = engine(); let token = CancellationToken::default();
    token.cancel(DecodeCancellationKind::User);
    let error = engine.load_int4_snapshot(missing(), pin(), budget(), 64 << 20, run(), token).err().unwrap();
    match error {
        HostedError::Stopped { stop, task_error } => { assert_eq!(stop.kind, DecodeCancellationKind::User); assert!(task_error.is_none()); }
        other => panic!("unexpected failure: {other}"),
    }
    drained(engine);
}
#[test]
fn insufficient_codec_budget_fails_before_missing_file_and_rolls_back_cleanly() {
    let _serial = SERIAL.lock().unwrap(); let engine = engine(); let mut budget = budget(); budget.max_io_bytes -= 1;
    let error = engine.load_int4_snapshot(missing(), pin(), budget, 64 << 20, run(), CancellationToken::default()).err().unwrap();
    assert!(matches!(error, HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Budget)))); drained(engine);
}
#[test]
fn excessive_resident_admission_does_not_touch_the_snapshot() {
    let _serial = SERIAL.lock().unwrap(); let engine = engine();
    let error = engine.load_int4_snapshot(missing(), pin(), budget(), engine.resources().config().memory_ceiling_bytes,
        run(), CancellationToken::default()).err().unwrap();
    assert!(matches!(error, HostedError::Reservation(_))); drained(engine);
}
#[test]
fn reentry_refuses_snapshot_loading_without_an_extra_compute_domain() {
    let _serial = SERIAL.lock().unwrap(); let engine = engine(); let _entry = engine.enter_sync_call().unwrap();
    let error = engine.load_int4_snapshot(missing(), pin(), budget(), 64 << 20, run(), CancellationToken::default()).err().unwrap();
    assert!(matches!(error, HostedError::Reentrant)); drained(engine);
}
