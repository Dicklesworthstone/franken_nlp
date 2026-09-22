//! Memory/owned-lifecycle fixtures; no model or crash-qualification claims.
use super::*;
use crate::{jobs::{OwnedJob, FrozenManifest, JobContract, JobInput, JobWork},
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode}};
use std::{fs, os::unix::fs::DirBuilderExt, sync::atomic::{AtomicU64, Ordering}};
static NEXT: AtomicU64 = AtomicU64::new(0);
#[derive(Default)]
struct Control { calls: u64, stop_at: Option<u64> }
impl DecodeStepControl for Control {
    fn checkpoint(&mut self, _: usize) -> Option<crate::native_engine::decode::DecodeCancellationKind> {
        self.calls += 1;
        self.stop_at.filter(|&n| self.calls >= n).map(|_| crate::native_engine::decode::DecodeCancellationKind::Deadline)
    }
}
fn key() -> JobSecret { JobSecret::from_bytes([17; 32]) }
fn limits() -> JobLimits {
    JobLimits { max_items: 16, max_id_bytes: 128, max_input_bytes_per_item: 65536,
        max_snapshot_bytes: 1 << 20, max_result_bytes: 16384, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 32 << 20,
        max_attempts: 16, max_work: JobWork::default() }
}
fn work() -> JobWork { JobWork::default() }
fn input(ordinal: usize) -> JobInput<'static> {
    JobInput { id: if ordinal == 0 { "a" } else { "b" }, original: b"private source", normalized: b"private source" }
}
fn manifest() -> FrozenManifest {
    let d = Sha256Digest::of_bytes(b"stored-host-fixture");
    let identity = ExecutionIdentity { schema_version: 1, source_revision: "fixture".to_owned(),
        logical_model_digest: d, artifact_format: "fixture".to_owned(), quant_recipe: "fixture".to_owned(),
        packing_set_digest: d, tokenizer_digest: d, template_digest: d, task_spec: "fixture-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::DiagnosticF32, kv_dtype: "bf16".to_owned(), sampler_version: "fixture".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None, calibration_digest: d,
        decision_policy_digest: d, backend_semantic_version: "fixture-v1".to_owned(), host_class: None, compiler_identity: None };
    FrozenManifest::freeze(&key(), JobContract { job_id: JobId([7; 16]), execution: &identity,
        recipe: &"fixed-fixture", limits: limits() }, [input(0), input(1)], &mut Control::default()).unwrap()
}
fn root() -> PathBuf {
    let path = std::env::temp_dir().join(format!("fnlp-manage-host-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    fs::DirBuilder::new().mode(0o700).create(&path).unwrap(); path
}
fn host_limits() -> JobManagementLimits {
    JobManagementLimits { run: RunLimits { max_elapsed: Duration::from_secs(60),
        max_checkpoints: 100000, cleanup_reserve_bytes: 65536 },
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20, io_reserve_bytes: 65536 }
}
fn request(root: PathBuf, operation: StoredJobOperation) -> StoredJobRequest {
    StoredJobRequest { root, key: key(), job_id: JobId([7; 16]), limits: limits(), operation }
}
#[test]
fn all_management_memory_classes_and_runtime_limits_are_explicit() {
    for axis in 0..6 {
        let mut h = host_limits();
        match axis { 0 => h.journal_reserve_bytes = 0, 1 => h.serialization_reserve_bytes = 0,
            2 => h.io_reserve_bytes = 0, 3 => h.run.max_elapsed = Duration::ZERO,
            4 => h.run.max_checkpoints = 1, _ => h.run.cleanup_reserve_bytes = 0 }
        assert!(h.reservation_bytes(limits()).is_err());
    }
}
#[test]
fn accounting_uses_metadata_and_one_frame_not_original_corpus_or_weights() {
    let h = host_limits(); let mut job = limits();
    let amount = h.reservation_bytes(job).unwrap();
    assert_eq!(amount, job.max_items * 1024 + job.max_result_bytes as u64 * 4 + 2 * 1024 * 1024
        + h.journal_reserve_bytes + h.serialization_reserve_bytes + h.io_reserve_bytes);
    job.max_snapshot_bytes *= 2; job.max_input_bytes_per_item *= 2;
    assert_eq!(h.reservation_bytes(job).unwrap(), amount);
}
#[test]
fn claim_overflow_refuses_instead_of_reducing_the_reservation() {
    let mut h = host_limits(); h.journal_reserve_bytes = u64::MAX;
    assert!(h.reservation_bytes(limits()).is_err());
}
#[test]
fn all_three_owned_operations_use_the_same_authenticated_owner() {
    let dir = root(); let mut original = OwnedJob::create(&dir, key(), manifest(), &mut Control::default()).unwrap();
    for ordinal in 0..2 {
        original.begin(&input(ordinal), work(), &mut Control::default()).unwrap()
            .commit(&ordinal, &mut Control::default()).unwrap();
    }
    drop(original);
    let status = perform(request(dir.clone(), StoredJobOperation::Status), &mut Control::default()).unwrap();
    let verified = perform(request(dir.clone(), StoredJobOperation::Verify), &mut Control::default()).unwrap();
    assert_eq!(status, verified);
    let published = perform(request(dir.clone(), StoredJobOperation::MaterializeOrdered), &mut Control::default()).unwrap();
    assert!(published.materialized); assert_eq!(published.attempts, status.attempts);
    assert_eq!(fs::read(dir.join("materialized.ndjson")).unwrap(), b"0\n1\n");
}
#[test]
fn cancellation_before_open_touches_no_storage() {
    let dir = root();
    let error = perform(request(dir.clone(), StoredJobOperation::Status),
        &mut Control { calls: 0, stop_at: Some(1) }).unwrap_err();
    assert!(matches!(error, JobError::Cancelled(_)));
    assert_eq!(fs::read_dir(dir).unwrap().count(), 0);
}
#[test]
fn requests_and_guarded_metadata_are_owned_across_the_existing_dispatch_seam() {
    fn send<T: Send + 'static>() {}
    send::<StoredJobRequest>(); send::<HostedOutput<StoredJobReport>>();
    let _entry = NlpEngine::manage_owned_job;
}
