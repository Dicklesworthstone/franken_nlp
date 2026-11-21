//! Tiny weights exercise only refusal and owned physical lifetime. No native
//! result or successful durable execution is synthesized by these fixtures.
use super::*;
use crate::{
    LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    hosted::corpus::jobs::JobOpenMode,
    jobs::{JobId, JobSecret, TailPolicy},
    native_engine::{decode::DecodeCancellationKind, strict_int4::weights::projection_fixture},
};
use std::{io::{self, Read}, path::PathBuf,
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering}};

#[derive(Clone, Copy)]
enum Action { Never, Cancel(DecodeCancellationKind), Panic, Malformed }
#[derive(Default)]
struct Observation {
    reads: AtomicUsize, drops: AtomicUsize, active_at_drop: AtomicUsize,
    input_at_drop: AtomicU64, native_at_drop: AtomicU64, unwound: AtomicBool,
}
struct Reader {
    resources: Arc<EngineResources>, observed: Arc<Observation>, action: Action,
    cancellation: CancellationToken,
}
impl Read for Reader {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { self.fill_buf()?; Ok(0) }
}
impl BufRead for Reader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.observed.reads.fetch_add(1, Ordering::SeqCst);
        match self.action {
            Action::Never => panic!("unadmitted semantic job touched private input"),
            Action::Cancel(cause) => self.cancellation.cancel(cause),
            Action::Panic => panic!("injected semantic job reader panic"),
            Action::Malformed => return Ok(b"{invalid-json\n"),
        }
        Ok(&[])
    }
    fn consume(&mut self, _: usize) {}
}
impl Drop for Reader {
    fn drop(&mut self) {
        let memory = self.resources.memory_snapshot();
        let bytes = |class| memory.by_class.get(&class)
            .map_or(0, |entry| entry.committed_bytes + entry.reserved_bytes);
        self.observed.input_at_drop.store(bytes(MemoryClass::JobBuffers), Ordering::SeqCst);
        self.observed.native_at_drop.store(bytes(MemoryClass::KvPages) + bytes(MemoryClass::ActivationScratch), Ordering::SeqCst);
        self.observed.active_at_drop.store(self.resources.outstanding_closure_snapshot().active_closures,
            Ordering::SeqCst);
        self.observed.unwound.store(std::thread::panicking(), Ordering::SeqCst);
        self.observed.drops.fetch_add(1, Ordering::SeqCst);
    }
}
fn request(job: JobLimits) -> SourceJobRequest {
    SourceJobRequest { root: PathBuf::from("/nonexistent/fnlp-semantic-job-refusal"),
        key: JobSecret::from_bytes([71; 32]), job_id: JobId([72; 16]), limits: job,
        mode: JobOpenMode::Create, materialize: false }
}
fn drained(host: &NlpEngine, baseline: &crate::MemorySnapshot) {
    assert_eq!(&host.resources().memory_snapshot(), baseline);
    let closures = host.resources().outstanding_closure_snapshot();
    assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
}
#[allow(clippy::too_many_arguments)]
fn execute(host: &NlpEngine, model: &ResidentInt4, config: Int4SemanticExtractionJobConfig,
    job: JobLimits, limits: JobHostLimits, threads: usize, reader: Reader, cancellation: CancellationToken)
    -> Result<JobProgress, HostedJobError> {
    if threads == 1 {
        host.job_int4_semantic_extract(model, config, request(job), limits,
            PackedInt4Kernel::scalar(), reader, cancellation)
    } else {
        host.job_int4_semantic_extract_with_threads(model, config, request(job), limits,
            PackedInt4Kernel::scalar(), threads, reader, cancellation)
    }
}
fn observed_reader(host: &NlpEngine, action: Action, cancellation: &CancellationToken)
    -> (Reader, Arc<Observation>) {
    let observed = Arc::new(Observation::default());
    (Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
        action, cancellation: cancellation.clone() }, observed)
}

// This finalizer runs after dispatch has joined every team child. It inspects
// the actual assets, rather than inferring their lifetime from memory counters.
fn finish_retained_assets(pending: DeferredSemanticJob<Cursor<Vec<u8>>>, control: &mut RunControl)
    -> Result<Result<JobProgress, JobRunError>, HostedError> {
    let compiler = pending.input.compiler.as_ref().expect("compiled assets survive the runner");
    let compiler_lifetime = Arc::downgrade(compiler);
    let population = pending.input.population.as_ref().expect("original population survives the runner");
    let inputs = population.borrowed_inputs().unwrap();
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].id, "retained-owner");
    assert_eq!(inputs[0].original, br#"{"id":"retained-owner","text":"private source"}"#);
    assert!(!matches!(&pending.result, Ok(Ok(_))), "missing resume storage cannot complete");
    drop(inputs);
    let result = pending.finish(control);
    assert!(compiler_lifetime.upgrade().is_none(), "the retained compiler is released after finalization");
    result
}

#[test]
#[ignore = "controller-only fresh runtime: tiny weights prove semantic job cancellation, admission and drain, never inference"]
fn semantic_jobs_retain_actual_inputs_and_all_claims_through_serial_and_team_failure() {
    assert!(crate::installed_process_resources().is_none(), "requires a fresh runtime process");
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: 3 * 1024 * MIB,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let idle = host.resources().memory_snapshot();
    let lease = host.resources().acquire_lease();
    let loaded = allocate(Pending::reserve(&lease, MemoryClass::Weights, MIB).unwrap(),
        || Ok(projection_fixture(WIDTH).0)).unwrap();
    let model = ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease }) };
    let baseline = host.resources().memory_snapshot();
    use DecodeCancellationKind::*;
    for threads in [1, 4] {
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            let (config, job, limits) = fixture();
            let required = quote(&config, job, limits, JobInputMemory::CompletePopulation, None).unwrap();
            let stop = CancellationToken::default(); stop.cancel(cause);
            let (reader, observed) = observed_reader(&host, Action::Never, &stop);
            let error = execute(&host, &model, config, job, limits, threads, reader, stop).err().unwrap();
            let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                else { panic!("semantic job pre-cancellation lost its original cause"); };
            assert_eq!(stop.kind, cause);
            assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
            assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
            assert_eq!(observed.active_at_drop.load(Ordering::SeqCst), 0,
                "queued input must survive formation cancellation and physical dispatch");
            assert!(observed.input_at_drop.load(Ordering::SeqCst)
                >= required.input_bytes + required.output_reservation_bytes);
            assert!(observed.native_at_drop.load(Ordering::SeqCst)
                >= required.native.kv_bytes + required.workspace_bytes);
            drained(&host, &baseline);
        }
        for action in [Action::Cancel(Deadline), Action::Panic, Action::Malformed] {
            let (config, job, limits) = fixture();
            let required = quote(&config, job, limits, JobInputMemory::CompletePopulation, None).unwrap();
            let stop = CancellationToken::default();
            let (reader, observed) = observed_reader(&host, action, &stop);
            let error = execute(&host, &model, config, job, limits, threads, reader, stop).err().unwrap();
            match action {
                Action::Cancel(cause) => assert!(matches!(error,
                    HostedJobError::Host(HostedError::Stopped { stop, .. }) if stop.kind == cause)),
                Action::Panic => assert!(matches!(error, HostedJobError::Host(
                    HostedError::Panicked | HostedError::Scope { .. }))),
                Action::Malformed => assert!(matches!(error, HostedJobError::Job(JobRunError::Storage(_)))),
                Action::Never => unreachable!(),
            }
            assert!(observed.reads.load(Ordering::SeqCst) > 0);
            assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
            assert!(!observed.unwound.load(Ordering::SeqCst), "IO owner must stay outside native unwind");
            assert!(observed.input_at_drop.load(Ordering::SeqCst)
                >= required.input_bytes + required.output_reservation_bytes);
            assert!(observed.native_at_drop.load(Ordering::SeqCst)
                >= required.native.kv_bytes + required.workspace_bytes);
            drained(&host, &baseline);
        }
        let (mut config, job, limits) = fixture(); config.extraction.schema = "{invalid-schema}".to_owned();
        let stop = CancellationToken::default(); let (reader, observed) = observed_reader(&host, Action::Never, &stop);
        assert!(matches!(execute(&host, &model, config, job, limits, threads, reader, stop),
            Err(HostedJobError::Host(HostedError::BatchSetup(_)))));
        assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
        assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
        drained(&host, &baseline);
    }
    // Fully read an original population, then fail its explicit resume before
    // native work. Both actual compilers and plaintext records must reach the
    // post-join finalizer, even though the processor/runner has already dropped.
    for threads in [1, 4] {
        let (config, job, limits) = fixture();
        let required = quote(&config, job, limits, JobInputMemory::CompletePopulation,
            if threads == 1 { None } else { Some(threads) }).unwrap();
        let lease = host.resources().acquire_lease();
        let owners = Arc::new(JobOwners {
            _input: Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes).unwrap().commit().unwrap(),
            _kv: Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes).unwrap().commit().unwrap(),
            _scratch: Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes).unwrap().commit().unwrap(),
            _output: Pending::reserve(&lease, MemoryClass::JobBuffers, required.output_reservation_bytes).unwrap().commit().unwrap(),
            _lease: lease,
        });
        let mut request = request(job);
        request.mode = JobOpenMode::Resume(TailPolicy::Refuse);
        let execution = JobExecution { input: JobInput { config, request: Some(request),
            vocabulary: None, compiler: None, population: None,
            reader: Cursor::new(b"{\"id\":\"retained-owner\",\"text\":\"private source\"}\n".to_vec()) },
            limits, required, owners };
        let model = model.clone();
        let completed = if threads == 1 {
            dispatch::run(&host, limits.native.run, CancellationToken::default(), move |control| {
                let pending = execution.run(&model.inner.loaded.value, PackedInt4Kernel::scalar(), None, control);
                finish_retained_assets(pending, control)
            })
        } else {
            dispatch::run_int4_team_with_finalize(&host, limits.native.run, CancellationToken::default(),
                model, threads, PackedInt4Kernel::scalar(), move |weights, control, team|
                    Ok(execution.run(weights, PackedInt4Kernel::scalar(), Some(team), control)),
                finish_retained_assets)
        };
        assert!(!matches!(completed, Ok(Ok(_))));
        drained(&host, &baseline);
    }
    for checkpoints in [2, 3] {
        let (config, job, mut limits) = fixture(); limits.native.run.max_checkpoints = checkpoints;
        let stop = CancellationToken::default(); let (reader, observed) = observed_reader(&host, Action::Never, &stop);
        assert!(matches!(execute(&host, &model, config, job, limits, 4, reader, stop),
            Err(HostedJobError::Host(HostedError::Stopped { stop, .. })) if stop.kind == PollQuota));
        assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
        assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
        assert_eq!(observed.active_at_drop.load(Ordering::SeqCst), 0);
        drained(&host, &baseline);
    }
    drop(model); drained(&host, &idle);
}
