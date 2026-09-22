//! Host accounting and real owned-job lifecycle fixtures. No fake model
//! weights, no inference/physical-drain qualification claim from these tests.
use super::*;
use crate::{jobs::{JobWork, tests::{Control, identity, key, limits as job_limits, work}},
    batch::{BatchDocument, BatchProcessor, BatchRequestContext, BatchWork}};
use std::{fs, io::Cursor, os::unix::fs::DirBuilderExt, sync::{Mutex,
    atomic::{AtomicU64, Ordering}}};

fn host_limits() -> JobHostLimits {
    JobHostLimits { native: NativeLimits { context_tokens: 128, allocator_reserve_bytes: 65536,
        run: RunLimits { max_elapsed: Duration::from_secs(60), max_checkpoints: 100000, cleanup_reserve_bytes: 65536 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 1000 },
        preparation_reserve_bytes: 1 << 20, io_reserve_bytes: 65536,
        journal_reserve_bytes: 4 << 20, serialization_reserve_bytes: 1 << 20 }
}
#[test]
fn whole_population_is_priced_not_only_a_live_batch_window() {
    let cap = job_limits(); let h = host_limits(); let bytes = h.reservation_bytes(cap).unwrap();
    assert!(bytes >= 2 * cap.max_snapshot_bytes + cap.max_items * 1024
        + cap.max_input_bytes_per_item as u64 * 8 + cap.max_result_bytes as u64 * 4);
    for axis in 0..4 {
        let mut larger = cap;
        match axis { 0 => larger.max_snapshot_bytes += 1, 1 => larger.max_items += 1,
            2 => larger.max_input_bytes_per_item += 1, _ => larger.max_result_bytes += 1 }
        assert!(h.reservation_bytes(larger).unwrap() > bytes);
    }
}
#[test]
fn each_host_overhead_is_mandatory_and_adds_to_the_same_reservation() {
    let base = host_limits(); let bytes = base.reservation_bytes(job_limits()).unwrap();
    for axis in 0..4 {
        let mut h = base;
        let field = match axis { 0 => &mut h.preparation_reserve_bytes, 1 => &mut h.io_reserve_bytes,
            2 => &mut h.journal_reserve_bytes, _ => &mut h.serialization_reserve_bytes };
        *field += 7; assert_eq!(h.reservation_bytes(job_limits()).unwrap(), bytes + 7);
        let field = match axis { 0 => &mut h.preparation_reserve_bytes, 1 => &mut h.io_reserve_bytes,
            2 => &mut h.journal_reserve_bytes, _ => &mut h.serialization_reserve_bytes };
        *field = 0; assert!(h.reservation_bytes(job_limits()).is_err());
    }
}
#[test]
fn aggregate_overflow_and_invalid_transport_refuse_before_admission() {
    let mut h = host_limits(); h.journal_reserve_bytes = u64::MAX;
    assert!(h.reservation_bytes(job_limits()).is_err());
    h = host_limits(); h.transport.max_lines = 0;
    assert!(h.reservation_bytes(job_limits()).is_err());
    let mut cap = job_limits(); cap.max_snapshot_bytes = i64::MAX as u64;
    assert!(host_limits().reservation_bytes(cap).is_err());
}
#[test]
fn duration_and_cleanup_limits_are_not_refreshed_or_omitted() {
    for axis in 0..3 {
        let mut h = host_limits();
        match axis { 0 => h.native.run.max_elapsed = Duration::ZERO,
            1 => h.native.run.max_checkpoints = 1, _ => h.native.run.cleanup_reserve_bytes = 0 }
        assert!(h.reservation_bytes(job_limits()).is_err());
    }
}
#[test]
fn complete_owned_package_and_public_native_entry_are_send() {
    fn send<T: Send + 'static>() {}
    type Input = StreamInput<(Arc<SourceTaskPlanner>, Arc<ExtractionVocabulary>, SourceCorpusConfig,
        SourceJobRequest), Cursor<Vec<u8>>, ()>;
    send::<Charged<Input>>(); send::<SourceJobRequest>();
    let _entry = NlpEngine::job_int8_source::<Cursor<Vec<u8>>>;
}

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("fnlp-hosted-job-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap(); Self(path)
    }
}
impl Drop for Root { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
fn request(root: &Root, mode: JobOpenMode, materialize: bool) -> SourceJobRequest {
    SourceJobRequest { root: root.0.clone(), key: key(), job_id: JobId([7;16]), limits: job_limits(), mode, materialize }
}
fn population(changed: bool) -> JobPopulation {
    let text = if changed { "changed" } else { "original" };
    let source = format!("{{\"id\":\"alpha\",\"text\":\"{text}\"}}\n{{\"id\":\"beta\",\"text\":\"original\"}}\n");
    JobPopulation::read_ndjson(&mut Cursor::new(source), &key(), JobId([7;16]), job_limits(),
        host_limits().transport, &mut Control::default()).unwrap()
}
struct Fixture { identity: ExecutionIdentity, calls: Arc<Mutex<Vec<u64>>>, fail_second: bool }
impl Fixture {
    fn new(calls: &Arc<Mutex<Vec<u64>>>, fail_second: bool) -> Self {
        Self { identity: identity(), calls: calls.clone(), fail_second }
    }
}
impl BatchProcessor for Fixture {
    type Args = serde_json::Value;
    type Prepared = String;
    type Output = u64;
    fn prepare(&mut self, document: BatchDocument<Self::Args>) -> Result<String, BatchItemFailure> { Ok(document.id) }
    fn planned_work(&self, _: &String) -> BatchWork {
        BatchWork { forward_positions: work().model.forward_positions, projected_logits: work().model.projected_logits }
    }
    fn execute<C: DecodeStepControl>(&mut self, _: String, _: &mut C) -> Result<u64, BatchItemFailure> { panic!("missing context") }
    fn execute_with_context<C: DecodeStepControl>(&mut self, id: String, context: BatchRequestContext, _: &mut C)
        -> Result<u64, BatchItemFailure> {
        assert_eq!(id, if context.request_seq == 1 { "alpha" } else { "beta" });
        self.calls.lock().unwrap().push(context.request_seq);
        if self.fail_second && context.request_seq == 2 { return Err(BatchItemFailure::reject(BatchCode::Execution)); }
        Ok(context.request_seq)
    }
}
impl DurableBatchProcessor for Fixture {
    type Recipe = str;
    fn execution_identity(&self) -> &ExecutionIdentity { &self.identity }
    fn job_recipe(&self) -> &str { "hosted-lifecycle-fixture" }
    fn durable_work(&self, _: &String) -> Result<JobWork, BatchItemFailure> { Ok(work()) }
    fn max_result_bytes(&self, _: &String) -> u64 { 128 }
}
#[test]
fn explicit_resume_keeps_prior_results_and_failed_attempt_debits() {
    let root = Root::new(); let population = population(false); let calls = Arc::new(Mutex::new(Vec::new()));
    let mut control = Control::default();
    let error = run_population(request(&root, JobOpenMode::Create, true), &population, Fixture::new(&calls, true), &mut control).unwrap_err();
    assert_eq!(error, JobRunError::Processor(BatchCode::Execution.into()));
    assert!(!root.0.join("materialized.ndjson").exists());
    let progress = run_population(request(&root, JobOpenMode::Resume(TailPolicy::Refuse), true),
        &population, Fixture::new(&calls, false), &mut control).unwrap();
    assert_eq!(progress.committed, 2); assert_eq!(progress.attempts, 3); assert!(progress.materialized);
    assert_eq!(*calls.lock().unwrap(), vec![1, 2, 2]);
    assert_eq!(fs::read(root.0.join("materialized.ndjson")).unwrap(), b"1\n2\n");
}
#[test]
fn completion_does_not_implicitly_publish_but_resume_can_publish_without_inference() {
    let root = Root::new(); let population = population(false); let calls = Arc::new(Mutex::new(Vec::new()));
    let mut control = Control::default();
    let progress = run_population(request(&root, JobOpenMode::Create, false), &population, Fixture::new(&calls, false), &mut control).unwrap();
    assert_eq!(progress.committed, 2); assert!(!progress.materialized);
    assert!(!root.0.join("materialized.ndjson").exists());
    let progress = run_population(request(&root, JobOpenMode::Resume(TailPolicy::Refuse), true),
        &population, Fixture::new(&calls, false), &mut control).unwrap();
    assert!(progress.materialized); assert_eq!(*calls.lock().unwrap(), vec![1, 2]);
}
#[test]
fn changed_original_population_cannot_resume_or_publish() {
    let root = Root::new(); let calls = Arc::new(Mutex::new(Vec::new())); let mut control = Control::default();
    run_population(request(&root, JobOpenMode::Create, false), &population(false), Fixture::new(&calls, false), &mut control).unwrap();
    let error = run_population(request(&root, JobOpenMode::Resume(TailPolicy::DiscardUncommitted), true),
        &population(true), Fixture::new(&calls, false), &mut control).unwrap_err();
    assert_eq!(error, JobRunError::Storage(JobError::Mismatch(crate::jobs::MismatchField::Population)));
    assert!(!root.0.join("materialized.ndjson").exists()); assert_eq!(*calls.lock().unwrap(), vec![1, 2]);
}
#[test]
fn default_host_error_formatting_does_not_include_private_nested_messages() {
    let error = HostedJobError::Host(HostedError::Limits("private path or source"));
    assert!(!format!("{error:?} {error}").contains("private path"));
    let job = HostedJobError::from(JobError::DuplicateId);
    assert!(format!("{job}").contains("DuplicateId"));
}
