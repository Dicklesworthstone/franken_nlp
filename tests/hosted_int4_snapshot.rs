//! Model-free failure/admission regressions on the actual process-owned host.
#![cfg(feature = "asupersync-runtime")]
use std::{fs::{self, OpenOptions}, io::Write, path::PathBuf,
    sync::{Mutex, OnceLock, atomic::{AtomicU64, Ordering}}, time::Duration};
use franken_nlp::{NlpEngine, ResourceHostConfig, RuntimePreset, LeakResponsePolicy,
    hosted::{HostedError, Int4HostError, RunLimits, CancellationToken},
    native_engine::{decode::DecodeCancellationKind, strict_int4::weights::snapshot::{
        Int4SnapshotBudget, Int4SnapshotPin, Int4SnapshotRequirements, Int4SnapshotError}}};
static SERIAL: Mutex<()> = Mutex::new(());
static ENGINE: OnceLock<NlpEngine> = OnceLock::new();
static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);
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
fn fixture_dir() -> PathBuf {
    let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("fnlp-hosted-snapshot-{}-{sequence}", std::process::id()));
    fs::create_dir(&path).unwrap(); path
}
fn format_refusal(engine: &NlpEngine, path: PathBuf) {
    let error = engine.load_int4_snapshot(path, pin(), budget(), 64 << 20, run(), CancellationToken::default()).err().unwrap();
    assert!(matches!(error, HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Format))));
    drained(engine);
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
#[test]
fn malformed_snapshot_paths_leave_the_process_host_available() {
    let _serial = SERIAL.lock().unwrap(); let engine = engine(); let root = fixture_dir();
    format_refusal(engine, root.clone());
    let short = root.join("short");
    let mut file = OpenOptions::new().write(true).create_new(true).open(&short).unwrap();
    file.write_all(b"incomplete snapshot").unwrap(); drop(file);
    format_refusal(engine, short.clone());
    assert_eq!(fs::read(short).unwrap(), b"incomplete snapshot");
    #[cfg(unix)]
    {
        let link = root.join("snapshot-link");
        std::os::unix::fs::symlink(root.join("short"), &link).unwrap();
        format_refusal(engine, link.clone());
        assert!(fs::symlink_metadata(link).unwrap().file_type().is_symlink());
    }
}
#[cfg(target_os = "linux")]
#[test]
fn fifo_snapshot_is_refused_before_a_blocking_open_and_releases_host_charges() {
    use std::os::unix::fs::FileTypeExt;
    let _serial = SERIAL.lock().unwrap(); let engine = engine(); let path = fixture_dir().join("fifo");
    assert!(std::process::Command::new("mkfifo").arg(&path).status().unwrap().success());
    // Rescue the old blocking-open bug after a finite watchdog interval, so
    // this regression fails instead of hanging the entire production suite.
    // Linux opens a FIFO for read+write without waiting for another endpoint.
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let rescue_path = path.clone();
        let watchdog = scope.spawn(move || {
            match finished.recv_timeout(Duration::from_secs(5)) {
                Ok(()) => false,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    let _rescue = OpenOptions::new().read(true).write(true).open(&rescue_path).unwrap();
                    let _ = finished.recv_timeout(Duration::from_secs(5));
                    true
                }
                Err(error) => panic!("snapshot watchdog disconnected: {error}"),
            }
        });
        format_refusal(engine, path.clone());
        let _ = done.send(());
        assert!(!watchdog.join().unwrap(), "snapshot load blocked opening a FIFO before regular-file validation");
    });
    assert!(fs::symlink_metadata(&path).unwrap().file_type().is_fifo());
    // The process-owned compute domain is still usable after the rejection.
    format_refusal(engine, path.parent().unwrap().to_owned());
}
