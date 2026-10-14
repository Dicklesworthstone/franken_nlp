//! Real host failure/admission paths, with no model weights or fake inference.
#![cfg(feature = "asupersync-runtime")]
use std::{path::PathBuf, sync::{Mutex, OnceLock}, time::Duration};
use franken_nlp::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy,
    hosted::{HostedError, Int4HostError, Int4LoadLimits, RunLimits, CancellationToken},
    native_engine::{decode::DecodeCancellationKind, strict_int4::{Int4ModelLoadRequirement, Int4ModelLoadBudget}}};

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
fn limits() -> Int4LoadLimits {
    let need = Int4ModelLoadRequirement::for_group_width(128).unwrap();
    Int4LoadLimits { model: Int4ModelLoadBudget {
        max_payload_bytes: need.peak_payload_bytes().unwrap(), max_source_values: need.source_values,
        max_groups: need.groups, max_shard_revalidation_bytes: need.shard_revalidation_bytes,
    }, source_preparation_bytes: 128 * 1024 * 1024, allocator_reserve_bytes: 64 * 1024 * 1024,
        run: RunLimits { max_elapsed: Duration::from_secs(120), max_checkpoints: 100, cleanup_reserve_bytes: 65536 } }
}
fn missing() -> (PathBuf, PathBuf) {
    let root = std::env::temp_dir().join(format!("fnlp-int4-absent-{}-source", std::process::id()));
    assert!(!root.exists(), "test refuses to touch an existing source");
    (root.clone(), root.join("source_manifest.json"))
}
fn assert_drained(engine: &NlpEngine) {
    let memory = engine.resources().memory_snapshot();
    assert_eq!(memory.reserved_bytes, 0); assert_eq!(memory.committed_bytes, 0);
    assert_eq!(memory.outstanding_obligations, 0); assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(engine.resources().outstanding_closure_snapshot().active_closures, 0);
}

#[test]
fn missing_source_manifest_uses_real_pool_and_drains_all_claims_before_returning() {
    let _serial = SERIAL.lock().unwrap(); let engine = engine(); let (root, manifest) = missing();
    let error = engine.load_source_candidate_int4(root, manifest, 128, limits(), CancellationToken::default()).err().unwrap();
    assert!(matches!(error, HostedError::Int4(Int4HostError::Source(_))), "{error}");
    assert_drained(engine);
}

#[test]
fn cancellation_before_launch_never_opens_the_missing_manifest_or_retains_model_reservations() {
    let _serial = SERIAL.lock().unwrap(); let engine = engine(); let (root, manifest) = missing();
    let token = CancellationToken::default(); token.cancel(DecodeCancellationKind::User);
    let error = engine.load_source_candidate_int4(root, manifest, 128, limits(), token).err().unwrap();
    match error { HostedError::Stopped { stop, task_error } => {
        assert_eq!(stop.kind, DecodeCancellationKind::User); assert!(task_error.is_none());
    }, other => panic!("wrong failure category: {other}") }
    assert_drained(engine);
}

#[test]
fn source_staging_admission_failure_rolls_back_the_already_reserved_weights() {
    let _serial = SERIAL.lock().unwrap(); let engine = engine(); let (root, manifest) = missing();
    let limits = Int4LoadLimits { source_preparation_bytes: engine.resources().config().memory_ceiling_bytes, ..limits() };
    let error = engine.load_source_candidate_int4(root, manifest, 128, limits, CancellationToken::default()).err().unwrap();
    assert!(matches!(error, HostedError::Reservation(_))); assert_drained(engine);
}

#[test]
fn model_work_budget_failure_precedes_all_source_and_native_work() {
    let _serial = SERIAL.lock().unwrap(); let engine = engine(); let (root, manifest) = missing();
    let mut limits = limits(); limits.model.max_groups -= 1;
    let error = engine.load_source_candidate_int4(root, manifest, 128, limits, CancellationToken::default()).err().unwrap();
    assert!(matches!(error, HostedError::Limits(_))); assert_drained(engine);
}

#[test]
fn reentry_is_refused_without_replenishing_execution_or_model_allowances() {
    let _serial = SERIAL.lock().unwrap(); let engine = engine(); let (root, manifest) = missing();
    let _entry = engine.enter_sync_call().unwrap();
    let error = engine.load_source_candidate_int4(root, manifest, 128, limits(), CancellationToken::default()).err().unwrap();
    assert!(matches!(error, HostedError::Reentrant)); assert_drained(engine);
}
