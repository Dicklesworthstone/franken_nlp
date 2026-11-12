//! Real admitted dispatch and cancellation, without executing a substitute judge.
use super::*;
use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    hosted::corpus::jobs::JobOpenMode, jobs::{JobId, JobSecret},
    native_engine::{decode::DecodeCancellationKind, strict_int4::weights::projection_fixture},
    tasks::judge::FaithfulnessPolicy};
use std::{io::{self, Read}, path::PathBuf, sync::atomic::{AtomicU64, AtomicUsize, Ordering}};

const PREPARATION: u64 = 128 << 20;
#[derive(Clone, Copy)]
enum Action { Refuse, Cancel(DecodeCancellationKind), Panic }
#[derive(Default)]
struct Observation { reads: AtomicUsize, drops: AtomicUsize,
    input_at_drop: AtomicU64, workers_at_read: AtomicU64 }
struct Reader { resources: Arc<EngineResources>, observed: Arc<Observation>,
    action: Action, stop: CancellationToken }
impl Drop for Reader {
    fn drop(&mut self) {
        let snapshot = self.resources.memory_snapshot();
        self.observed.input_at_drop.store(snapshot.by_class.get(&MemoryClass::JobBuffers)
            .map_or(0, |value| value.committed_bytes + value.reserved_bytes), Ordering::SeqCst);
        self.observed.drops.fetch_add(1, Ordering::SeqCst);
    }
}
impl Read for Reader {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { self.fill_buf()?; Ok(0) }
}
impl BufRead for Reader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.observed.reads.fetch_add(1, Ordering::SeqCst);
        self.observed.workers_at_read.store(self.resources.memory_snapshot().by_class
            .get(&MemoryClass::ActivationScratch).map_or(0, |value| value.committed_bytes), Ordering::SeqCst);
        match self.action {
            Action::Refuse => panic!("unadmitted faithfulness job read private input"),
            Action::Panic => panic!("injected faithfulness job reader panic"),
            Action::Cancel(cause) => self.stop.cancel(cause),
        }
        Ok(&[])
    }
    fn consume(&mut self, _: usize) {}
}
fn policy(max_heads: usize) -> FaithfulnessPolicy {
    let windows = u32::try_from(if max_heads == 1 { 1 } else { max_heads - 1 }).unwrap();
    FaithfulnessPolicy { minimum_candidate_weight_ppm: 730_000, minimum_margin_milli: 250,
        evidence_window_bytes: 16, max_evidence_windows: windows, max_evidence_spans: windows }
}
fn planner(model: &ResidentInt4, max_heads: usize, cohorts: Option<usize>)
    -> (Int4FaithfulnessDocumentJobPlanner, SourceJobRequest, JobHostLimits) {
    let (budget, planning, mapping, job, mut limits) = fixture();
    limits.native.run.max_elapsed = Duration::from_secs(60);
    limits.native.run.max_checkpoints = 100_000;
    let compiler = Int4JudgePlanner::for_model(model.model_identity(), budget, planning).unwrap();
    let planner = Int4FaithfulnessDocumentJobPlanner::new(compiler, policy(max_heads), mapping, job).unwrap()
        .with_max_heads(max_heads).unwrap();
    let planner = if let Some(chunks) = cohorts { planner.with_chunk_cohorts(chunks).unwrap() } else { planner };
    let request = SourceJobRequest { root: PathBuf::from("/nonexistent/fnlp-faithfulness-job-team-refusal"),
        key: JobSecret::from_bytes([3; 32]), job_id: JobId([4; 16]), limits: job,
        mode: JobOpenMode::Create, materialize: false };
    (planner, request, limits)
}
fn drained(host: &NlpEngine, baseline: &crate::MemorySnapshot) {
    assert_eq!(&host.resources().memory_snapshot(), baseline);
    let closures = host.resources().outstanding_closure_snapshot();
    assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
}

#[test]
#[ignore = "controller-only fresh runtime: tiny weights prove admission/drain, never native judgment or result parity"]
fn faithfulness_job_teams_preserve_typed_cancellation_and_population_ownership() {
    assert!(crate::installed_process_resources().is_none());
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: 1024 << 20,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let idle = host.resources().memory_snapshot();
    let lease = host.resources().acquire_lease();
    let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, PREPARATION).unwrap().commit().unwrap();
    let loaded = allocate(Pending::reserve(&lease, MemoryClass::Weights, 1 << 20).unwrap(),
        || Ok(projection_fixture(32).0)).unwrap();
    let model = ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease }) };
    let baseline = host.resources().memory_snapshot();
    use DecodeCancellationKind::*;
    for (width, max_heads) in [(2, 1), (4, 32)] {
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let (planner, request, limits) = planner(&model, max_heads, None);
            let charge = limits.required_buffer_bytes(request.limits).unwrap();
            let stop = CancellationToken::default(); stop.cancel(cause);
            let observed = Arc::new(Observation::default());
            let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                action: Action::Refuse, stop: stop.clone() };
            let error = host.job_int4_faithfulness_document_with_threads(&model, planner,
                request, limits, PackedInt4Kernel::scalar(), width, reader, stop).err().unwrap();
            let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                else { panic!("faithfulness pre-cancellation lost its typed cause"); };
            assert_eq!(stop.kind, cause); assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
            assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
            assert!(observed.input_at_drop.load(Ordering::SeqCst) >= PREPARATION + charge);
            drained(&host, &baseline);
        }
    }
    for (width, cohorts) in [(0, None), (1, None), (5, None), (65, None), (usize::MAX, None),
        (2, Some(1)), (4, Some(2))] {
        let (planner, request, limits) = planner(&model, 32, cohorts);
        let stop = CancellationToken::default();
        let observed = Arc::new(Observation::default());
        let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
            action: Action::Refuse, stop: stop.clone() };
        let error = host.job_int4_faithfulness_document_with_threads(&model, planner,
            request, limits, PackedInt4Kernel::scalar(), width, reader, stop).err().unwrap();
        assert!(matches!(error, HostedJobError::Host(HostedError::Limits(_))));
        assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
        assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
        drained(&host, &baseline);
    }
    {
        let (planner, request, limits) = planner(&model, 4, None);
        let required = requirements(32, planner.task_budget(), planner.planning_limits(),
            planner.document_limits(), request.limits, limits, planner.max_heads(), None,
            request.mode.input_memory()).unwrap();
        let room = sum(&[required.input_bytes, required.native.kv_bytes,
            required.workspace_bytes, int4_task_team_workspace_bytes(4, 32).unwrap() - 1]).unwrap();
        let bytes = host.resources().config().memory_ceiling_bytes
            .checked_sub(baseline.committed_bytes + baseline.reserved_bytes).unwrap().checked_sub(room).unwrap();
        let lease = host.resources().acquire_lease();
        let held = Pending::reserve(&lease, MemoryClass::JobBuffers, bytes).unwrap().commit().unwrap();
        let stop = CancellationToken::default();
        let observed = Arc::new(Observation::default());
        let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
            action: Action::Refuse, stop: stop.clone() };
        let error = host.job_int4_faithfulness_document_with_threads(&model, planner,
            request, limits, PackedInt4Kernel::scalar(), 4, reader, stop).err().unwrap();
        assert!(matches!(error, HostedJobError::Host(HostedError::Reservation(_))));
        assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
        assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
        drop(held); drop(lease);
        drained(&host, &baseline);
    }
    for action in [Action::Cancel(Deadline), Action::Panic] {
        let (planner, request, limits) = planner(&model, 4, None);
        let charge = limits.required_buffer_bytes(request.limits).unwrap();
        let stop = CancellationToken::default();
        let observed = Arc::new(Observation::default());
        let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
            action, stop: stop.clone() };
        let error = host.job_int4_faithfulness_document_with_threads(&model, planner,
            request, limits, PackedInt4Kernel::scalar(), 4, reader, stop).err().unwrap();
        match action {
            Action::Cancel(cause) => {
                let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                    else { panic!("faithfulness reader cancellation lost its typed cause"); };
                assert_eq!(stop.kind, cause);
            }
            Action::Panic => assert!(matches!(error,
                HostedJobError::Host(HostedError::Panicked | HostedError::Scope { .. }))),
            Action::Refuse => unreachable!(),
        }
        assert_eq!(observed.reads.load(Ordering::SeqCst), 1);
        assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
        assert!(observed.workers_at_read.load(Ordering::SeqCst) >= int4_task_team_workspace_bytes(4, 32).unwrap());
        assert!(observed.input_at_drop.load(Ordering::SeqCst) >= PREPARATION + charge);
        drained(&host, &baseline);
    }
    drop(model); drop(preparation);
    drained(&host, &idle);
}
