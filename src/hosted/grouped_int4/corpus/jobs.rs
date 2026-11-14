//! Authenticated resumable extraction on the charged grouped-INT4 host.
//! The existing runner owns retention, work debits and publication ordering.
use super::*;
mod source;
mod document;
mod extract_document;
mod document_native;
mod generation;
use crate::{
    hosted::corpus::jobs::{HostedJobError, JobHostLimits, SourceJobRequest, read_population, run_population},
    jobs::{JobInputMemory, JobLimits, JobProgress,
        runner::{JobRunError, extract::grouped_int4::{Int4ExtractionJobPlanner, Int4ExtractionJobProcessor}}},
    tasks::ir::TaskBudget,
};

struct JobInput<R> {
    preparation: Option<(Int4ExtractionJobPlanner, Arc<ExtractionVocabulary>, SourceJobRequest)>,
    reader: R,
}

struct ExtractionJobExecution<R> {
    input: Charged<JobInput<R>>, kv: Pending, scratch: Pending,
    limits: JobHostLimits, required: Requirements, lease: EngineLease,
}

/// Retain native reservations after their coordinator-owned buffers are gone.
/// Ordinary ingestion failures have no native allocation, but still retain the
/// admitted pending claims until the shared dispatcher has joined the team.
enum NativeJobCharges {
    Pending { _kv: Pending, _scratch: Pending },
    Committed { _kv: CommittedMemory, _scratch: CommittedMemory },
}

/// Keep input and native claims through physical join on normal and ordinary
/// job-error paths. Shared dispatch separately owns the worker/model/cleanup
/// guards through early cancellation, formation failure and panic drain.
struct ExtractionJobCompletion<R> {
    result: Result<JobProgress, JobRunError>,
    input: Charged<JobInput<R>>,
    native: NativeJobCharges,
    lease: EngineLease,
}

impl NlpEngine {
    /// Create or resume an explicitly retained INT4 extraction job using the
    /// complete original NDJSON population and one exact shared schema.
    ///
    /// The model's strict-v2/group identity and all six lifetime work counters
    /// are checked before any input or job-file access. Population ingestion,
    /// authenticated resume, native execution, spool/journal commits and optional
    /// materialization complete inside one owned blocking invocation. Stored
    /// results remain protected and no result text is written to stdout here.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_extract<R>(&self, model: &ResidentInt4, planner: Int4ExtractionJobPlanner,
        vocabulary: Arc<ExtractionVocabulary>, request: SourceJobRequest,
        limits: JobHostLimits, kernel: PackedInt4Kernel, reader: R,
        cancellation: CancellationToken) -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_extraction_job(model, planner, vocabulary, request, limits,
            kernel, None, reader, cancellation)
    }

    /// Reuse one admitted output-row team across population authentication,
    /// every pending extraction, durable acknowledgement and materialization.
    /// The width is physical scheduling only: changing it on resume neither
    /// changes the frozen recipe nor replenishes native work, masks or attempts.
    /// Widths 2..=64 require worker scratch and the installed thread envelope.
    /// Every worker joins before this call reports progress or failure.
    #[allow(clippy::too_many_arguments)]
    pub fn job_int4_extract_with_threads<R>(&self, model: &ResidentInt4,
        planner: Int4ExtractionJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, kernel: PackedInt4Kernel,
        compute_threads: usize, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        self.run_int4_extraction_job(model, planner, vocabulary, request, limits,
            kernel, Some(compute_threads), reader, cancellation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_int4_extraction_job<R>(&self, model: &ResidentInt4,
        planner: Int4ExtractionJobPlanner, vocabulary: Arc<ExtractionVocabulary>,
        request: SourceJobRequest, limits: JobHostLimits, kernel: PackedInt4Kernel,
        compute_threads: Option<usize>, reader: R, cancellation: CancellationToken)
        -> Result<JobProgress, HostedJobError>
    where R: BufRead + Send + 'static {
        match compute_threads {
            Some(width) => dispatch::preflight_int4_team(self, limits.native.run, width)?,
            None => dispatch::preflight(self, limits.native.run)?,
        }
        if !Arc::ptr_eq(self.resources(), model.resources()) {
            return Err(HostedError::ResourceDomain.into());
        }
        constrained_int4::check_model(planner.execution_identity(), model.model_identity())
            .map_err(|_| HostedError::ModelIdentity)?;
        planner.check_job_limits(request.limits).map_err(HostedError::BatchSetup)?;
        let mut required = job_requirements(model.model_identity().group_width(), planner.task_ceiling(),
            request.limits, limits, request.mode.input_memory())?;
        // The caller's preparation reserve remains independently live. The
        // hosted owner also admits the fixed encoder workspace before any
        // population read, even when retained input is replayed on demand.
        required.input_bytes = sum(&[required.input_bytes,
            planner.encoding_workspace_bytes().map_err(HostedError::BatchSetup)?])?;
        let team_scratch = compute_threads.map(|width|
            int4_task_team_workspace_bytes(width, model.model_identity().group_width())).transpose()?.unwrap_or(0);
        sum(&[required.input_bytes, required.native.kv_bytes, required.workspace_bytes, team_scratch])?;
        let lease = self.resources().acquire_lease();
        let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?,
            || Ok(JobInput { preparation: Some((planner, vocabulary, request)), reader }))?;
        let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
        let scratch = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
        let execution = ExtractionJobExecution { input, kv, scratch, limits, required, lease };
        let model = model.clone();
        let completed = match compute_threads {
            None => dispatch::run(self, limits.native.run, cancellation, move |control|
                execution.run(&model.inner.loaded.value, kernel, None, control)),
            Some(width) => dispatch::run_int4_team(self, limits.native.run, cancellation,
                model, width, kernel, move |weights, control, team|
                    execution.run(weights, kernel, Some(team), control)),
        }?;
        // Runtime cancellation/drain failure takes precedence over job return.
        // Input storage drops before its charge, after the workers have joined.
        let ExtractionJobCompletion { result, input, native, lease } = completed;
        drop(input);
        drop(native);
        drop(lease);
        result.map_err(HostedJobError::Job)
    }
}

impl<R: BufRead> ExtractionJobExecution<R> {
    fn run<'weights>(self, weights: &'weights Int4ModelWeights, kernel: PackedInt4Kernel,
        projection: Option<&'weights mut (dyn crate::native_engine::strict_int4::projection::Int4ProjectionExecutor + 'weights)>,
        control: &mut RunControl) -> Result<ExtractionJobCompletion<R>, HostedError> {
        let Self { mut input, kv, scratch, limits, required, lease } = self;
        checkpoint(control)?;
        let (planner, vocabulary, request) = input.value.preparation.take()
            .ok_or(HostedError::CompletionMissing)?;
        let native_work = planner.native_limits();
        let population = match read_population(&mut input.value.reader, &request,
            limits.transport, control) {
            Ok(population) => population,
            Err(error) => return Ok(ExtractionJobCompletion {
                result: Err(JobRunError::Storage(error)), input,
                native: NativeJobCharges::Pending { _kv: kv, _scratch: scratch }, lease,
            }),
        };
        checkpoint(control)?;
        let mut engine = allocate_native(kv, scratch, || {
            let engine = StrictInt4Engine::new(weights, limits.native.context_tokens, Int4MemoryBudget {
                max_kv_bytes: required.native.kv_bytes, max_rope_bytes: required.native.rope_bytes,
                max_scratch_payload_bytes: required.native.scratch_payload_bound,
            }).map_err(native_error)?.with_kernel(kernel);
            match projection {
                Some(team) => engine.with_projection_executor(team).map_err(native_error),
                None => Ok(engine),
            }
        })?;
        let admission = CorpusAdmission { lease: &lease, model: weights.identity(),
            kv_bytes: required.native.kv_bytes, masks: native_work.masks,
            output_bytes: request.limits.max_result_bytes as u64, selected_rows: None };
        let result = (|| {
            let processor = Int4ExtractionJobProcessor::new(planner, &mut engine.value, &vocabulary, admission)
                .map_err(JobRunError::Processor)?;
            // Authentication precedes tail repair and any native forward.
            // Output guards survive spool sync and journal acknowledgement.
            run_population(request, &population, processor, control)
        })();
        let ChargedNative { value, _kv, _scratch } = engine;
        drop(value);
        drop(population);
        drop(vocabulary);
        Ok(ExtractionJobCompletion { result, input,
            native: NativeJobCharges::Committed { _kv, _scratch }, lease })
    }
}

fn job_requirements(width: usize, task: TaskBudget, job: JobLimits, limits: JobHostLimits, input: JobInputMemory)
    -> Result<Requirements, HostedJobError> {
    let input_bytes = limits.required_buffer_bytes_for_input(job, input)?;
    if limits.native.allocator_reserve_bytes == 0 {
        return Err(HostedError::Limits("INT4 job allocator reserve").into());
    }
    let native = Int4MemoryRequirement::for_context(limits.native.context_tokens, width).map_err(native_error)?;
    if native.kv_bytes == 0 || task.max_kv_bytes != native.kv_bytes
        || task.max_output_bytes > job.max_result_bytes as u64 {
        return Err(HostedError::Limits("INT4 job complete KV and stored-result envelope").into());
    }
    let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound,
        limits.native.allocator_reserve_bytes])?;
    Ok(Requirements { native, workspace_bytes, input_bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hosted::{NativeLimits, RunLimits};
    use crate::jobs::{JobWork, population::PopulationReadLimits};
    use std::time::Duration;

    fn fixture() -> (TaskBudget, JobLimits, JobHostLimits) {
        let native = Int4MemoryRequirement::for_context(64, 128).unwrap();
        let task = TaskBudget { max_input_tokens: 48, max_output_tokens: 16,
            max_output_bytes: 4096, max_grammar_states: 4096, max_kv_bytes: native.kv_bytes };
        let job = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
            max_snapshot_bytes: 65536, max_result_bytes: 65536, max_spool_bytes: 1 << 20,
            max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
            max_work: JobWork::default() };
        let host = JobHostLimits {
            native: NativeLimits { context_tokens: 64, allocator_reserve_bytes: 65536,
                run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 1000,
                    cleanup_reserve_bytes: 65536 } },
            transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
            preparation_reserve_bytes: 1 << 20, io_reserve_bytes: 131072,
            journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
        };
        (task, job, host)
    }
    #[test]
    fn whole_population_native_workspace_and_full_context_are_priced_together() {
        let (task, job, host) = fixture();
        let needed = job_requirements(128, task, job, host, JobInputMemory::CompletePopulation).unwrap();
        assert_eq!(needed.input_bytes, host.required_buffer_bytes(job).unwrap());
        assert_eq!(needed.native.kv_bytes, task.max_kv_bytes);
        assert_eq!(needed.workspace_bytes, needed.native.rope_bytes
            + needed.native.scratch_payload_bound + host.native.allocator_reserve_bytes);
    }
    #[test]
    fn looser_or_tighter_kv_and_oversized_results_refuse_before_input() {
        let (task, job, host) = fixture();
        for kv in [0, task.max_kv_bytes - 1, task.max_kv_bytes + 1] {
            let mut wrong = task; wrong.max_kv_bytes = kv;
            assert!(job_requirements(128, wrong, job, host, JobInputMemory::CompletePopulation).is_err());
        }
        let mut wrong = task; wrong.max_output_bytes = job.max_result_bytes as u64 + 1;
        assert!(job_requirements(128, wrong, job, host, JobInputMemory::CompletePopulation).is_err());
        let mut wrong = host; wrong.native.allocator_reserve_bytes = 0;
        assert!(job_requirements(128, task, job, wrong, JobInputMemory::CompletePopulation).is_err());
    }
    #[test]
    fn group_width_and_explicit_host_reserves_remain_mandatory() {
        let (task, job, host) = fixture();
        assert!(job_requirements(0, task, job, host, JobInputMemory::CompletePopulation).is_err());
        for axis in 0..4 {
            let mut wrong = host;
            match axis {
                0 => wrong.preparation_reserve_bytes = 0,
                1 => wrong.io_reserve_bytes = 0,
                2 => wrong.journal_reserve_bytes = 0,
                _ => wrong.serialization_reserve_bytes = 0,
            }
            assert!(job_requirements(128, task, job, wrong, JobInputMemory::CompletePopulation).is_err());
        }
    }
    #[test]
    fn owned_population_factory_secret_and_vocabulary_cross_one_blocking_boundary() {
        fn send<T: Send + 'static>() {}
        send::<Int4ExtractionJobPlanner>();
        send::<JobInput<std::io::Cursor<Vec<u8>>>>();
        send::<ExtractionJobExecution<std::io::Cursor<Vec<u8>>>>();
        send::<ExtractionJobCompletion<std::io::Cursor<Vec<u8>>>>();
        let _serial = NlpEngine::job_int4_extract::<std::io::Cursor<Vec<u8>>>;
        let _threaded = NlpEngine::job_int4_extract_with_threads::<std::io::Cursor<Vec<u8>>>;
    }

    mod team {
        use super::*;
        use crate::{EngineResources, LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
            batch::extract::{ExtractionBatchArgs, ExtractionBatchGrounding},
            grammar::{CompileLimits, mask::MaskWorkLimits, runtime::SourceRuntimeLimits},
            hosted::corpus::jobs::JobOpenMode, jobs::{JobId, JobSecret},
            native_engine::{decode::DecodeCancellationKind, strict_int4::weights::projection_fixture}};
        use std::{io::{self, Read}, path::PathBuf, sync::atomic::{AtomicU64, AtomicUsize, Ordering}};

        const PREPARATION: u64 = 128 << 20;
        #[derive(Clone, Copy)]
        enum Action { Refuse, Cancel(DecodeCancellationKind), Panic, Malformed }
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
                    Action::Refuse => panic!("unadmitted extraction job read private input"),
                    Action::Panic => panic!("injected extraction job reader panic"),
                    Action::Cancel(cause) => self.stop.cancel(cause),
                    Action::Malformed => return Ok(b"{malformed\n"),
                }
                Ok(&[])
            }
            fn consume(&mut self, _: usize) {}
        }
        fn planner(model: &ResidentInt4) -> (Int4ExtractionJobPlanner, SourceJobRequest, JobHostLimits) {
            let (task, mut job, mut limits) = fixture();
            limits.native.run.max_elapsed = Duration::from_secs(60);
            limits.native.run.max_checkpoints = 100_000;
            limits.preparation_reserve_bytes = PREPARATION;
            let work = Int4ExtractionBatchLimits {
                max_model_work: Int4Work::for_sequence(32, 0, 128, 32 * NANBEIGE_VOCAB_SIZE).unwrap(),
                masks: ExtractionMaskBudget { per_mask: MaskWorkLimits {
                    max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
                    max_visits_per_item: 32_000_000, max_visits_per_run: 64_000_000 },
            };
            job.max_work = JobWork::from_int4(work.max_model_work, work.masks.max_visits_per_run);
            let defaults = ExtractionBatchArgs { schema: r#"{"type":"boolean"}"#.to_owned(),
                grounding: ExtractionBatchGrounding::Structural, budget: task };
            let planner = Int4ExtractionJobPlanner::for_model(model.model_identity(), task,
                CompileLimits::default(), SourceRuntimeLimits::default(), defaults, work).unwrap();
            let request = SourceJobRequest { root: PathBuf::from("/nonexistent/fnlp-extract-job-team-refusal"),
                key: JobSecret::from_bytes([3; 32]), job_id: JobId([4; 16]), limits: job,
                mode: JobOpenMode::Create, materialize: false };
            (planner, request, limits)
        }
        fn drained(host: &NlpEngine, baseline: &crate::MemorySnapshot) {
            assert_eq!(&host.resources().memory_snapshot(), baseline);
            let closures = host.resources().outstanding_closure_snapshot();
            assert_eq!(closures.active_closures, 0);
            assert_eq!(closures.wrapper_cancelled_closures, 0);
        }

        #[test]
        #[ignore = "controller-only fresh runtime: tiny weights prove admission and drain, never native-result parity"]
        fn extraction_workers_preserve_cancellation_and_charged_input_until_drop_then_drain() {
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
            let registry = crate::tokenizer::pinned_controls::pinned().unwrap();
            let vocabulary = Arc::new(ExtractionVocabulary::pinned(registry.template_controls()).unwrap());
            let baseline = host.resources().memory_snapshot();
            use DecodeCancellationKind::*;
            for width in [2, 4] {
                for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
                    ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
                    let (planner, request, limits) = planner(&model);
                    let charge = limits.required_buffer_bytes(request.limits).unwrap();
                    let stop = CancellationToken::default(); stop.cancel(cause);
                    let observed = Arc::new(Observation::default());
                    let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                        action: Action::Refuse, stop: stop.clone() };
                    let error = host.job_int4_extract_with_threads(&model, planner, Arc::clone(&vocabulary),
                        request, limits, PackedInt4Kernel::scalar(), width, reader, stop).err().unwrap();
                    let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                        else { panic!("extraction pre-cancellation lost its typed cause"); };
                    assert_eq!(stop.kind, cause);
                    assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
                    assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
                    assert!(observed.input_at_drop.load(Ordering::SeqCst) >= PREPARATION + charge);
                    drained(&host, &baseline);
                }
            }
            for width in [0, 1, 5, 65, usize::MAX] {
                let (planner, request, limits) = planner(&model);
                let stop = CancellationToken::default();
                let observed = Arc::new(Observation::default());
                let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                    action: Action::Refuse, stop: stop.clone() };
                let error = host.job_int4_extract_with_threads(&model, planner, Arc::clone(&vocabulary),
                    request, limits, PackedInt4Kernel::scalar(), width, reader, stop).err().unwrap();
                assert!(matches!(error, HostedJobError::Host(HostedError::Limits(_))));
                assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
                assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
                drained(&host, &baseline);
            }
            for compact in [false, true] {
                let (planner, request, limits) = planner(&model);
                let planner = if compact {
                    use crate::tokenizer::untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding};
                    planner.with_document_encoding(UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
                        max_input_bytes: 8192, max_output_tokens: 128,
                        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
                    }), Some(SafeBpeWorkLimits { max_input_bytes: 65536, max_output_tokens: 4096,
                        max_pair_checks: 1 << 22, max_pair_bytes: 1 << 26 })).unwrap()
                } else { planner };
                let mut required = job_requirements(32, planner.task_ceiling(), request.limits,
                    limits, request.mode.input_memory()).unwrap();
                required.input_bytes = sum(&[required.input_bytes, planner.encoding_workspace_bytes().unwrap()]).unwrap();
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
                let error = host.job_int4_extract_with_threads(&model, planner, Arc::clone(&vocabulary),
                    request, limits, PackedInt4Kernel::scalar(), 4, reader, stop).err().unwrap();
                assert!(matches!(error, HostedJobError::Host(HostedError::Reservation(_))));
                assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
                assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
                drop(held); drop(lease);
                drained(&host, &baseline);
            }
            for action in [Action::Cancel(Deadline), Action::Panic, Action::Malformed] {
                let (planner, request, limits) = planner(&model);
                let charge = limits.required_buffer_bytes(request.limits).unwrap();
                let stop = CancellationToken::default();
                let observed = Arc::new(Observation::default());
                let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                    action, stop: stop.clone() };
                let error = host.job_int4_extract_with_threads(&model, planner, Arc::clone(&vocabulary),
                    request, limits, PackedInt4Kernel::scalar(), 4, reader, stop).err().unwrap();
                match action {
                    Action::Cancel(cause) => {
                        let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                            else { panic!("extraction reader cancellation lost its typed cause"); };
                        assert_eq!(stop.kind, cause);
                    }
                    Action::Panic => assert!(matches!(error,
                        HostedJobError::Host(HostedError::Panicked | HostedError::Scope { .. }))),
                    Action::Malformed => assert!(matches!(error, HostedJobError::Job(JobRunError::Storage(_)))),
                    Action::Refuse => unreachable!(),
                }
                assert_eq!(observed.reads.load(Ordering::SeqCst), 1);
                assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
                assert!(observed.workers_at_read.load(Ordering::SeqCst) >= int4_task_team_workspace_bytes(4, 32).unwrap());
                assert!(observed.input_at_drop.load(Ordering::SeqCst) >= PREPARATION + charge);
                drained(&host, &baseline);
            }
            drop(vocabulary); drop(model); drop(preparation);
            drained(&host, &idle);
        }
    }
}
