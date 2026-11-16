//! Tiny weights exercise only refusal, cancellation and physical ownership.
//! No successful native generation or durable result is synthesized here.

use super::*;
use crate::{
    LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    hosted::corpus::jobs::JobOpenMode,
    jobs::{JobId, JobSecret},
    native_engine::{decode::DecodeCancellationKind, strict_int4::weights::projection_fixture},
    tasks::sentiment::sampled::{SampledSentimentOptions, SampledSentimentPlanner},
};
use std::{io::{self, Read}, path::PathBuf, sync::atomic::{AtomicU64, AtomicUsize, Ordering}};

#[derive(Clone, Copy)]
enum Action { Refuse, Cancel(DecodeCancellationKind), Panic, Malformed }
#[derive(Default)]
struct Observation { reads: AtomicUsize, drops: AtomicUsize,
    input_at_drop: AtomicU64, native_at_drop: AtomicU64 }
struct Reader { resources: Arc<EngineResources>, observed: Arc<Observation>,
    action: Action, stop: CancellationToken }
impl Drop for Reader {
    fn drop(&mut self) {
        let snapshot = self.resources.memory_snapshot();
        self.observed.input_at_drop.store(snapshot.by_class.get(&MemoryClass::JobBuffers)
            .map_or(0, |value| value.committed_bytes + value.reserved_bytes), Ordering::SeqCst);
        self.observed.native_at_drop.store([MemoryClass::KvPages, MemoryClass::ActivationScratch]
            .iter().map(|class| snapshot.by_class.get(class)
                .map_or(0, |value| value.committed_bytes + value.reserved_bytes)).sum(), Ordering::SeqCst);
        self.observed.drops.fetch_add(1, Ordering::SeqCst);
    }
}
impl Read for Reader {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { self.fill_buf()?; Ok(0) }
}
impl BufRead for Reader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.observed.reads.fetch_add(1, Ordering::SeqCst);
        match self.action {
            Action::Refuse => panic!("unadmitted sampled job read private input"),
            Action::Panic => panic!("injected sampled job reader panic"),
            Action::Cancel(cause) => self.stop.cancel(cause),
            Action::Malformed => return Ok(b"{malformed\n"),
        }
        Ok(&[])
    }
    fn consume(&mut self, _: usize) {}
}

fn planner(model: &ResidentInt4, vocabulary: &Arc<SampledSentimentVocabulary>)
    -> (SampledSentimentJobPlanner, SourceJobRequest, JobHostLimits) {
    let (task, work, job, mut host) = fixture(32, 8);
    host.native.run.max_elapsed = Duration::from_secs(60);
    let compiler = SampledSentimentPlanner::for_model_with_encoding(model.model_identity(),
        SampledSentimentOptions { effective_seed: [23; 32], samples: 2, temperature_milli: 1000,
            top_k: None, top_p_ppm: 1_000_000, max_sample_tokens: 16, max_justification_chars: 32 },
        task, SampledSentimentLimits { max_input_bytes: 4096, max_result_bytes: 65_536,
            ..SampledSentimentLimits::default() },
        UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096, max_output_tokens: 4096,
            max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24 })).unwrap();
    let planner = SampledSentimentJobPlanner::new(compiler, Arc::clone(vocabulary), work, job).unwrap();
    assert!(planner.preparation_requirement_bytes() <= PREPARATION);
    let request = SourceJobRequest { root: PathBuf::from("/nonexistent/fnlp-sampled-job-lifecycle-refusal"),
        key: JobSecret::from_bytes([3; 32]), job_id: JobId([4; 16]), limits: job,
        mode: JobOpenMode::Create, materialize: false };
    (planner, request, host)
}

fn drained(host: &NlpEngine, baseline: &crate::MemorySnapshot) {
    assert_eq!(&host.resources().memory_snapshot(), baseline);
    let closures = host.resources().outstanding_closure_snapshot();
    assert_eq!(closures.active_closures, 0);
    assert_eq!(closures.wrapper_cancelled_closures, 0);
}

#[allow(clippy::too_many_arguments)]
fn execute(host: &NlpEngine, model: &ResidentInt4, planner: SampledSentimentJobPlanner,
    request: SourceJobRequest, limits: JobHostLimits, schedule: usize,
    reader: Reader, stop: CancellationToken) -> Result<JobProgress, HostedJobError> {
    match schedule {
        0 => host.job_int4_sampled_sentiment(model, planner, request, limits,
            PackedInt4Kernel::scalar(), reader, stop),
        1 => host.job_int4_sampled_sentiment_with_threads(model, planner, request, limits,
            PackedInt4Kernel::scalar(), 4, reader, stop),
        2 => host.job_int4_sampled_sentiment_layer_major(model, planner, request, limits,
            PackedInt4Kernel::scalar(), Int4PrefillLimits { max_batch_rows: 4,
                max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(4).unwrap() }, reader, stop),
        _ => panic!("fixture schedule"),
    }
}

#[test]
#[ignore = "controller-only fresh runtime: tiny weights prove admission/drain, never native or durable-result parity"]
fn original_cancellation_and_owned_input_charges_survive_every_physical_schedule() {
    assert!(crate::installed_process_resources().is_none());
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: 3 * 1024 * 1024 * 1024,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let idle = host.resources().memory_snapshot();
    let lease = host.resources().acquire_lease();
    let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, PREPARATION).unwrap().commit().unwrap();
    let loaded = allocate(Pending::reserve(&lease, MemoryClass::Weights, 1 << 20).unwrap(),
        || Ok(projection_fixture(32).0)).unwrap();
    let model = ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease }) };
    let vocabulary = Arc::new(SampledSentimentVocabulary::pinned().unwrap());
    let baseline = host.resources().memory_snapshot();
    use DecodeCancellationKind::*;
    for schedule in 0..3 {
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let (planner, request, limits) = planner(&model, &vocabulary);
            let charge = limits.required_buffer_bytes(request.limits).unwrap();
            let native_charge = int4_sampled_sentiment_job_requirements(32, planner.task_budget(),
                planner.native_limits(), planner.preparation_requirement_bytes(), request.limits, limits,
                request.mode.input_memory(), None, None).unwrap();
            let native_charge = native_charge.native.kv_bytes + native_charge.workspace_bytes;
            let stop = CancellationToken::default(); stop.cancel(cause);
            let observed = Arc::new(Observation::default());
            let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                action: Action::Refuse, stop: stop.clone() };
            let error = execute(&host, &model, planner, request, limits, schedule, reader, stop).err().unwrap();
            let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                else { panic!("sampled job pre-cancellation lost its typed cause"); };
            assert_eq!(stop.kind, cause);
            assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
            assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
            assert!(observed.input_at_drop.load(Ordering::SeqCst) >= PREPARATION + charge);
            assert!(observed.native_at_drop.load(Ordering::SeqCst) >= native_charge,
                "native ownership must outlive the private reader on every terminal path");
            drained(&host, &baseline);
        }
        for action in [Action::Cancel(Deadline), Action::Panic, Action::Malformed] {
            let (planner, request, limits) = planner(&model, &vocabulary);
            let charge = limits.required_buffer_bytes(request.limits).unwrap();
            let native_charge = int4_sampled_sentiment_job_requirements(32, planner.task_budget(),
                planner.native_limits(), planner.preparation_requirement_bytes(), request.limits, limits,
                request.mode.input_memory(), None, None).unwrap();
            let native_charge = native_charge.native.kv_bytes + native_charge.workspace_bytes;
            let stop = CancellationToken::default();
            let observed = Arc::new(Observation::default());
            let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                action, stop: stop.clone() };
            let error = execute(&host, &model, planner, request, limits, schedule, reader, stop).err().unwrap();
            match action {
                Action::Cancel(cause) => {
                    let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                        else { panic!("sampled job reader cancellation lost its typed cause"); };
                    assert_eq!(stop.kind, cause);
                }
                Action::Panic => assert!(matches!(error,
                    HostedJobError::Host(HostedError::Panicked | HostedError::Scope { .. }))),
                Action::Malformed => assert!(matches!(error, HostedJobError::Job(JobRunError::Storage(_)))),
                Action::Refuse => unreachable!(),
            }
            assert_eq!(observed.reads.load(Ordering::SeqCst), 1);
            assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
            assert!(observed.input_at_drop.load(Ordering::SeqCst) >= PREPARATION + charge);
            assert!(observed.native_at_drop.load(Ordering::SeqCst) >= native_charge,
                "native ownership must outlive the private reader on every terminal path");
            drained(&host, &baseline);
        }
    }
    for width in [0, 1, 5, 65, usize::MAX] {
        let (planner, request, limits) = planner(&model, &vocabulary);
        let stop = CancellationToken::default();
        let observed = Arc::new(Observation::default());
        let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
            action: Action::Refuse, stop: stop.clone() };
        let error = host.job_int4_sampled_sentiment_with_threads(&model, planner, request, limits,
            PackedInt4Kernel::scalar(), width, reader, stop).err().unwrap();
        assert!(matches!(error, HostedJobError::Host(HostedError::Limits(_))));
        assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
        assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
        drained(&host, &baseline);
    }
    drop(vocabulary);
    drop(model);
    drop(preparation);
    drained(&host, &idle);
}
