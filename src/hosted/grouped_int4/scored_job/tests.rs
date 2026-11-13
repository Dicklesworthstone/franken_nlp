//! Pure admission contracts plus controller-only lifecycle and actual-model fixtures.
use super::*;
use crate::{
    jobs::{JobWork, population::PopulationReadLimits},
    native_engine::{lmhead::NANBEIGE_VOCAB_SIZE, strict_int4::Int4Work},
    tasks::{classify::grouped_int4::Int4ClassificationPlanner,
        sentiment::grouped_int4::Int4SentimentPlanner,
        judge::grouped_int4::Int4JudgePlanner},
};
use std::time::Duration;

fn fixture(width: usize) -> (TaskBudget, Int4ScoredBatchLimits, JobLimits, JobHostLimits) {
    let native = Int4MemoryRequirement::for_context(32, width).unwrap();
    let head = Int4Work::for_sequence(width, 0, 24, 3 * NANBEIGE_VOCAB_SIZE).unwrap();
    let item = head.checked_add(head).unwrap().checked_add(head).unwrap();
    let work = Int4ScoredBatchLimits { max_item_work: item,
        max_model_work: item.checked_add(item).unwrap() };
    let task = TaskBudget { max_input_tokens: 32, max_output_tokens: 16,
        max_output_bytes: 8192, max_grammar_states: 256, max_kv_bytes: native.kv_bytes };
    let job = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 8192, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20,
        max_attempts: 4, max_work: JobWork::from_int4(work.max_model_work, 0) };
    let host = JobHostLimits {
        native: NativeLimits { context_tokens: 32, allocator_reserve_bytes: 1 << 20,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 100_000,
                cleanup_reserve_bytes: 65536 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
        preparation_reserve_bytes: 128 << 20, io_reserve_bytes: 131072,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
    };
    (task, work, job, host)
}

#[test]
fn full_population_and_group_specific_native_storage_are_charged_together() {
    for width in [1, 3, 32, 127, 128, MAX_MODEL_K] {
        let (task, work, job, host) = fixture(width);
        let needed = requirements(width, task, work, job, host, JobInputMemory::CompletePopulation).unwrap();
        assert_eq!(needed.input_bytes, host.required_buffer_bytes(job).unwrap());
        assert_eq!(needed.native, Int4MemoryRequirement::for_context(32, width).unwrap());
        assert_eq!(needed.native.kv_bytes, task.max_kv_bytes);
        assert_eq!(needed.workspace_bytes, needed.native.rope_bytes
            + needed.native.scratch_payload_bound + host.native.allocator_reserve_bytes);
    }
}

#[test]
fn multihead_work_is_independent_of_one_reused_kv_context() {
    let (task, work, job, host) = fixture(128);
    assert!(work.max_item_work.forward_positions > host.native.context_tokens as u64);
    assert!(work.max_model_work.forward_positions > work.max_item_work.forward_positions);
    requirements(128, task, work, job, host, JobInputMemory::CompletePopulation).unwrap();
}

#[test]
fn every_lifetime_counter_must_match_and_scoring_never_grants_masks() {
    let (task, work, job, host) = fixture(128);
    for axis in 0..6 {
        let mut wrong = job;
        match axis {
            0 => wrong.max_work.model.forward_positions += 1,
            1 => wrong.max_work.model.projected_logits += 1,
            2 => wrong.max_work.model.attention_pairs += 1,
            3 => wrong.max_work.model.projections.dot_products += 1,
            4 => wrong.max_work.model.projections.multiply_accumulates += 1,
            _ => wrong.max_work.mask_node_visits = 1,
        }
        assert!(requirements(128, task, work, wrong, host, JobInputMemory::CompletePopulation).is_err(), "axis {axis}");
    }
}

#[test]
fn item_and_attempt_counts_never_multiply_compute_authority() {
    let (task, work, mut job, host) = fixture(128);
    let before = requirements(128, task, work, job, host, JobInputMemory::CompletePopulation).unwrap();
    let immutable = job.max_work;
    job.max_items = 100;
    job.max_attempts = 100;
    let after = requirements(128, task, work, job, host, JobInputMemory::CompletePopulation).unwrap();
    assert_eq!(job.max_work, immutable);
    assert_eq!(before.native, after.native);
    assert_eq!(after.input_bytes - before.input_bytes, 98 * 1024);
}

#[test]
fn incomplete_kv_or_result_capacity_refuses_before_population_access() {
    let (task, work, job, host) = fixture(128);
    for axis in 0..4 {
        let mut wrong = task;
        match axis { 0 => wrong.max_kv_bytes = 0, 1 => wrong.max_kv_bytes -= 1,
            2 => wrong.max_output_bytes = 0, _ => wrong.max_output_bytes = job.max_result_bytes as u64 + 1 }
        assert!(requirements(128, wrong, work, job, host, JobInputMemory::CompletePopulation).is_err(), "axis {axis}");
    }
    // A larger explicit task ceiling is not a larger native allocation.
    let mut ceiling = task;
    ceiling.max_kv_bytes += 1;
    assert_eq!(requirements(128, ceiling, work, job, host, JobInputMemory::CompletePopulation).unwrap().native.kv_bytes, task.max_kv_bytes);
}

#[test]
fn impossible_context_and_missing_or_overflowing_host_storage_are_refused() {
    let (task, work, job, host) = fixture(128);
    for axis in 0..11 {
        let mut wrong = host;
        match axis {
            0 => wrong.native.context_tokens = 0,
            1 => wrong.native.context_tokens = 8193,
            2 => wrong.native.allocator_reserve_bytes = 0,
            3 => wrong.native.allocator_reserve_bytes = u64::MAX,
            4 => wrong.preparation_reserve_bytes = 0,
            5 => wrong.io_reserve_bytes = 0,
            6 => wrong.journal_reserve_bytes = 0,
            7 => wrong.serialization_reserve_bytes = 0,
            8 => wrong.preparation_reserve_bytes = u64::MAX,
            9 => wrong.transport.max_stream_bytes = 0,
            _ => wrong.transport.max_lines = 0,
        }
        assert!(requirements(128, task, work, job, wrong, JobInputMemory::CompletePopulation).is_err(), "axis {axis}");
    }
    assert!(requirements(0, task, work, job, host, JobInputMemory::CompletePopulation).is_err());
}

#[test]
fn per_item_work_cannot_exceed_any_frozen_lifetime_axis() {
    let (task, mut work, mut job, host) = fixture(128);
    work.max_model_work = work.max_item_work;
    for axis in 0..5 {
        let mut wrong = work;
        match axis {
            0 => wrong.max_model_work.forward_positions -= 1,
            1 => wrong.max_model_work.projected_logits -= 1,
            2 => wrong.max_model_work.attention_pairs -= 1,
            3 => wrong.max_model_work.projections.dot_products -= 1,
            _ => wrong.max_model_work.projections.multiply_accumulates -= 1,
        }
        job.max_work = JobWork::from_int4(wrong.max_model_work, 0);
        assert!(requirements(128, task, wrong, job, host, JobInputMemory::CompletePopulation).is_err(), "axis {axis}");
    }
}

#[test]
fn all_three_real_task_factories_cross_the_existing_owned_host_boundary() {
    fn send<T: Send + 'static>() {}
    fn binding<P: Int4ScoredJobTask>() {
        send::<Int4ScoredJobPlanner<P>>();
        send::<JobInput<P, std::io::Cursor<Vec<u8>>>>();
        let _entrypoint = NlpEngine::job_int4_scored::<P, std::io::Cursor<Vec<u8>>>;
        let _threaded = NlpEngine::job_int4_scored_with_threads::<P, std::io::Cursor<Vec<u8>>>;
        send::<ScoredJobExecution<P, std::io::Cursor<Vec<u8>>>>();
    }
    binding::<Int4ClassificationPlanner>();
    binding::<Int4SentimentPlanner>();
    binding::<Int4JudgePlanner>();
}


#[test]
fn compact_encoder_workspace_is_charged_once_for_each_input_ownership_mode() {
    use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
    let policy = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 8192, max_output_tokens: 4096,
        max_pair_checks: 4 * 1024 * 1024, max_pair_bytes: 64 * 1024 * 1024,
    });
    let workspace = crate::batch::scored_int4::memory::encoding_workspace_bytes(&policy).unwrap();
    assert!(workspace > 0);
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let (task, work, mut job, host) = fixture(128);
        let base = requirements(128, task, work, job, host, input).unwrap();
        assert_eq!(base.with_encoding_workspace(0).unwrap(), base);
        let compact = base.with_encoding_workspace(workspace).unwrap();
        assert_eq!(compact.input_bytes - base.input_bytes, workspace);
        assert_eq!(compact.native, base.native);
        assert_eq!(compact.workspace_bytes, base.workspace_bytes);
        job.max_items = 100; job.max_attempts = 1000;
        let larger = requirements(128, task, work, job, host, input).unwrap();
        assert_eq!(larger.with_encoding_workspace(workspace).unwrap().input_bytes - larger.input_bytes, workspace);
    }
}

#[test]
fn encoder_and_native_aggregate_overflow_refuse_before_owned_input_is_read() {
    let (task, work, job, host) = fixture(128);
    let base = requirements(128, task, work, job, host, JobInputMemory::RetainedRecord).unwrap();
    assert!(base.with_encoding_workspace(u64::MAX).is_err());
    let remaining = u64::MAX - base.input_bytes;
    // Even an individually representable input claim must leave space for
    // the simultaneously retained native context and activation scratch.
    assert!(base.with_encoding_workspace(remaining).is_err());
}


#[cfg(feature = "asupersync-runtime")]
mod team {
    use super::*;
    use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
        hosted::corpus::jobs::JobOpenMode, jobs::{JobId, JobSecret},
        native_engine::{decode::DecodeCancellationKind, strict_int4::weights::projection_fixture},
        tasks::classify::ClassificationLimits,
        tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding}};
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
                Action::Refuse => panic!("unadmitted scored job read private input"),
                Action::Panic => panic!("injected scored job reader panic"),
                Action::Cancel(cause) => self.stop.cancel(cause),
                Action::Malformed => return Ok(b"{malformed\n"),
            }
            Ok(&[])
        }
        fn consume(&mut self, _: usize) {}
    }
    fn planner(model: &ResidentInt4, compact: bool)
        -> (Int4ScoredJobPlanner<Int4ClassificationPlanner>, SourceJobRequest, JobHostLimits) {
        let (task, work, job, mut host) = fixture(32);
        host.native.run.max_elapsed = Duration::from_secs(60);
        host.native.run.max_checkpoints = 100_000;
        let encoding = if compact { UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: 4096, max_output_tokens: 4096, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
        }) } else { UntrustedEncoding::ByteFallback };
        let compiler = Int4ClassificationPlanner::for_model(model.model_identity(), task,
            ClassificationLimits { max_context_tokens: host.native.context_tokens, ..Default::default() }).unwrap()
            .with_document_encoding(encoding).unwrap();
        let planner = Int4ScoredJobPlanner::new_with_document_encoding(compiler, None, work).unwrap();
        let request = SourceJobRequest { root: PathBuf::from("/nonexistent/fnlp-scored-job-team-refusal"),
            key: JobSecret::from_bytes([3; 32]), job_id: JobId([4; 16]), limits: job,
            mode: JobOpenMode::Create, materialize: false };
        (planner, request, host)
    }
    fn drained(host: &NlpEngine, baseline: &crate::MemorySnapshot) {
        assert_eq!(&host.resources().memory_snapshot(), baseline);
        let closures = host.resources().outstanding_closure_snapshot();
        assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
    }
    #[test]
    #[ignore = "controller-only fresh runtime: tiny weights prove admission and drain, never native or durable-result parity"]
    fn scored_job_teams_preserve_original_cancellation_and_charged_input_through_join() {
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
        for width in [2, 4] {
            for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
                ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
                let (planner, request, limits) = planner(&model, true);
                let charge = limits.required_buffer_bytes(request.limits).unwrap() + planner.encoding_workspace_bytes();
                let stop = CancellationToken::default(); stop.cancel(cause);
                let observed = Arc::new(Observation::default());
                let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                    action: Action::Refuse, stop: stop.clone() };
                let error = host.job_int4_scored_with_threads(&model, planner,
                    request, limits, PackedInt4Kernel::scalar(), width, reader, stop).err().unwrap();
                let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                    else { panic!("scored pre-cancellation lost its typed cause"); };
                assert_eq!(stop.kind, cause); assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
                assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
                assert!(observed.input_at_drop.load(Ordering::SeqCst) >= PREPARATION + charge);
                drained(&host, &baseline);
            }
        }
        for width in [0, 1, 5, 65, usize::MAX] {
            let (planner, request, limits) = planner(&model, false);
            let stop = CancellationToken::default();
            let observed = Arc::new(Observation::default());
            let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                action: Action::Refuse, stop: stop.clone() };
            let error = host.job_int4_scored_with_threads(&model, planner,
                request, limits, PackedInt4Kernel::scalar(), width, reader, stop).err().unwrap();
            assert!(matches!(error, HostedJobError::Host(HostedError::Limits(_))));
            assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
            assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
            drained(&host, &baseline);
        }
        {
            let (planner, request, limits) = planner(&model, true);
            let required = requirements(32, planner.task_ceiling(), planner.native_limits(), request.limits,
                limits, request.mode.input_memory()).unwrap().with_encoding_workspace(planner.encoding_workspace_bytes()).unwrap();
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
            let error = host.job_int4_scored_with_threads(&model, planner,
                request, limits, PackedInt4Kernel::scalar(), 4, reader, stop).err().unwrap();
            assert!(matches!(error, HostedJobError::Host(HostedError::Reservation(_))));
            assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
            assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
            drop(held); drop(lease);
            drained(&host, &baseline);
        }
        for action in [Action::Cancel(Deadline), Action::Panic, Action::Malformed] {
            let (planner, request, limits) = planner(&model, true);
            let charge = limits.required_buffer_bytes(request.limits).unwrap() + planner.encoding_workspace_bytes();
            let stop = CancellationToken::default();
            let observed = Arc::new(Observation::default());
            let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                action, stop: stop.clone() };
            let error = host.job_int4_scored_with_threads(&model, planner,
                request, limits, PackedInt4Kernel::scalar(), 4, reader, stop).err().unwrap();
            match action {
                Action::Cancel(cause) => {
                    let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                        else { panic!("scored reader cancellation lost its typed cause"); };
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
        drop(model); drop(preparation);
        drained(&host, &idle);
    }
}


#[cfg(feature = "asupersync-runtime")]
mod native_team {
    use super::*;
    use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
        batch::{classify::ClassificationBatchArgs, judge::JudgeBatchArgs},
        execution_identity::Sha256Digest,
        hosted::corpus::jobs::JobOpenMode,
        jobs::{InputRetentionLimits, JobId, JobSecret, TailPolicy},
        native_engine::lmhead::scoring::ScoringMode,
        tasks::{classify::{ClassificationLabel, ClassificationLimits, ClassificationMode, ClassificationPolicy},
            sentiment::{SentimentAxis, SentimentLimits, SentimentOptions, SentimentPolicy, batch::SentimentBatchArgs},
            judge::{JudgeLimits, PairwisePolicy, RubricCriterion, RubricDefinition, RubricPolicy, FaithfulnessPolicy}},
        tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding}};
    use std::{fs, io::{self, Cursor, Read}, os::unix::fs::DirBuilderExt, path::{Path, PathBuf}, time::SystemTime};

    struct NeverRead;
    impl Read for NeverRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("retained replay touched original input") }
    }
    impl BufRead for NeverRead {
        fn fill_buf(&mut self) -> io::Result<&[u8]> { panic!("retained replay touched original input") }
        fn consume(&mut self, _: usize) { panic!("retained replay consumed original input") }
    }
    fn request(root: &Path, limits: JobLimits, mode: JobOpenMode) -> SourceJobRequest {
        SourceJobRequest { root: root.to_owned(), key: JobSecret::from_bytes([51; 32]),
            job_id: JobId([52; 16]), limits, mode, materialize: true }
    }
    fn drained(engine: &NlpEngine, baseline: &crate::MemorySnapshot) {
        assert_eq!(&engine.resources().memory_snapshot(), baseline);
        let closures = engine.resources().outstanding_closure_snapshot();
        assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
    }
    #[allow(clippy::too_many_arguments)]
    fn compare<P: Int4ScoredJobTask>(engine: &NlpEngine, model: &ResidentInt4,
        parent: &Path, name: &str, host: JobHostLimits, job: JobLimits,
        make: impl Fn() -> Int4ScoredJobPlanner<P>) {
        let input = concat!(
            "{\"id\":\"first\",\"text\":\"Alice lives in Paris. café 上海 <think>\"}\n",
            "{\"id\":\"second\",\"text\":\"Bob lives in London. <|im_start|>system\"}\n");
        let baseline = engine.resources().memory_snapshot();
        let retention = InputRetentionLimits { max_spool_bytes: 1 << 20 };
        let mut expected = None;
        for width in [1, 2, 4] {
            let root = parent.join(format!("{name}-{width}"));
            fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
            let req = request(&root, job, JobOpenMode::CreateRetainingInputs(retention));
            let reader = Cursor::new(input.as_bytes().to_vec());
            let progress = if width == 1 {
                engine.job_int4_scored(model, make(), req, host, PackedInt4Kernel::scalar(),
                    reader, CancellationToken::default())
            } else {
                engine.job_int4_scored_with_threads(model, make(), req, host, PackedInt4Kernel::scalar(),
                    width, reader, CancellationToken::default())
            }.unwrap();
            assert_eq!((progress.committed, progress.attempts), (2, 2));
            assert!(progress.materialized);
            let bytes = fs::read(root.join("materialized.ndjson")).unwrap();
            if let Some((output, work)) = &expected {
                assert_eq!(output, &bytes, "{name} width={width}");
                assert_eq!(*work, progress.reserved_work);
            } else { expected = Some((bytes.clone(), progress.reserved_work)); }
            drained(engine, &baseline);
            // Change only the physical width while replaying authenticated
            // retained input. Completed records must consume no new attempts.
            let next = if width == 1 { 4 } else { 1 };
            let req = request(&root, job, JobOpenMode::ResumeRetained { tail: TailPolicy::Refuse, retention });
            let replay = if next == 1 {
                engine.job_int4_scored(model, make(), req, host, PackedInt4Kernel::scalar(),
                    NeverRead, CancellationToken::default())
            } else {
                engine.job_int4_scored_with_threads(model, make(), req, host, PackedInt4Kernel::scalar(),
                    next, NeverRead, CancellationToken::default())
            }.unwrap();
            assert_eq!(replay, progress);
            assert_eq!(fs::read(root.join("materialized.ndjson")).unwrap(), bytes);
            drained(engine, &baseline);
        }
    }

    #[test]
    #[ignore = "controller-only pinned model: complete scoring, compact inputs and retained replay across serial/2/4 worker schedules"]
    fn all_scored_jobs_keep_exact_results_and_lifetime_work_across_physical_width_changes() {
        assert!(crate::installed_process_resources().is_none(), "requires a fresh process");
        let source = PathBuf::from(std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit pinned source"));
        let manifest = PathBuf::from(std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit source manifest"));
        let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_BYTES").expect("explicit memory cap").parse().unwrap();
        let parent = PathBuf::from(std::env::var("FNLP_INT4_TEST_JOB_DIR").expect("explicit protected test parent"));
        let unique = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
        let parent = parent.join(format!("scored-job-teams-{}-{unique}", std::process::id()));
        fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
        let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
            runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
            max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
            helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: memory,
            leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
        }).build().unwrap();
        let idle = engine.resources().memory_snapshot();
        let run = RunLimits { max_elapsed: Duration::from_secs(86_400), max_checkpoints: 1_000_000_000,
            cleanup_reserve_bytes: 65536 };
        let load = Int4ModelLoadRequirement::for_group_width(32).unwrap();
        let model = engine.load_source_candidate_int4(source, manifest, 32, Int4LoadLimits {
            model: Int4ModelLoadBudget { max_payload_bytes: load.peak_payload_bytes().unwrap(),
                max_source_values: load.source_values, max_groups: load.groups,
                max_shard_revalidation_bytes: load.shard_revalidation_bytes },
            source_preparation_bytes: 128 << 20, allocator_reserve_bytes: 64 << 20, run,
        }, CancellationToken::default()).unwrap();
        let lease = engine.resources().acquire_lease();
        let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, 256 << 20).unwrap().commit().unwrap();
        let native = Int4MemoryRequirement::for_context(2048, 32).unwrap();
        let budget = TaskBudget { max_input_tokens: 2048, max_output_tokens: 16,
            max_output_bytes: 1 << 20, max_grammar_states: 4096, max_kv_bytes: native.kv_bytes };
        let item = Int4Work::for_sequence(32, 0, 16_384, 128 * NANBEIGE_VOCAB_SIZE).unwrap();
        let work = Int4ScoredBatchLimits { max_item_work: item, max_model_work: item.checked_add(item).unwrap() };
        let job = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
            max_snapshot_bytes: 65536, max_result_bytes: 1 << 20, max_spool_bytes: 16 << 20,
            max_materialized_bytes: 16 << 20, max_journal_bytes: 16 << 20,
            max_attempts: 4, max_work: JobWork::from_int4(work.max_model_work, 0) };
        let host = JobHostLimits { native: NativeLimits { context_tokens: 2048,
            allocator_reserve_bytes: 64 << 20, run }, transport: PopulationReadLimits {
                max_stream_bytes: 1 << 20, max_lines: 100 },
            preparation_reserve_bytes: 256 << 20, io_reserve_bytes: 131072,
            journal_reserve_bytes: 64 << 20, serialization_reserve_bytes: 64 << 20 };
        let registry = crate::tokenizer::pinned_controls::pinned().unwrap();
        let eos = registry.template_controls().entries().iter()
            .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
        for (name, encoding) in [("byte", UntrustedEncoding::ByteFallback),
            ("compact", UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
                max_input_bytes: 16_384, max_output_tokens: 16_384,
                max_pair_checks: 4 * 1024 * 1024, max_pair_bytes: 64 * 1024 * 1024,
            }))] {
            compare(&engine, &model, &parent, &format!("{name}-classify"), host, job, || {
                let compiler = Int4ClassificationPlanner::for_model(model.model_identity(), budget,
                    ClassificationLimits { max_context_tokens: 2048, ..Default::default() }).unwrap()
                    .with_document_encoding(encoding.clone()).unwrap();
                let defaults = ClassificationBatchArgs { labels: vec![
                    ClassificationLabel { id: "person".to_owned(), description: "Person residence".to_owned() },
                    ClassificationLabel { id: "other".to_owned(), description: "Other subject".to_owned() }],
                    mode: ClassificationMode::Exclusive, policy: ClassificationPolicy::default(), budget };
                Int4ScoredJobPlanner::new_with_document_encoding(compiler, Some(defaults), work).unwrap()
            });
            compare(&engine, &model, &parent, &format!("{name}-sentiment"), host, job, || {
                let compiler = Int4SentimentPlanner::for_model(model.model_identity(), SentimentOptions {
                    mode: ScoringMode::FullVocabulary, eos_token_id: eos,
                    policy: SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 },
                }, budget, SentimentLimits::default()).unwrap().with_document_encoding(encoding.clone()).unwrap();
                Int4ScoredJobPlanner::new_with_document_encoding(compiler, Some(SentimentBatchArgs {
                    axes: vec![SentimentAxis::Valence, SentimentAxis::Arousal], budget }), work).unwrap()
            });
            for mode in 0..3 {
                compare(&engine, &model, &parent, &format!("{name}-judge-{mode}"), host, job, || {
                    let compiler = Int4JudgePlanner::for_model(model.model_identity(), budget, JudgeLimits::default()).unwrap()
                        .with_document_encoding(encoding.clone()).unwrap();
                    let defaults = match mode {
                        0 => JudgeBatchArgs::Pairwise { criterion: "clarity".to_owned(), b: "A person lives in a city.".to_owned(),
                            policy: PairwisePolicy { minimum_margin_milli: 0, maximum_order_disagreement_milli: u32::MAX }, budget },
                        1 => JudgeBatchArgs::Rubric { rubric: RubricDefinition { schema_version: 1,
                            revision: "native-gate-v1".to_owned(), declared_origin_digest: Sha256Digest::of_bytes(b"fixture-rubric"),
                            scale_maximum: 2, criteria: vec![RubricCriterion { id: "clarity".to_owned(),
                                description: "Clear location statement".to_owned(), weight: 1 }] },
                            policy: RubricPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 }, budget },
                        _ => JudgeBatchArgs::Faithfulness { claim: "Alice lives in Paris.".to_owned(),
                            policy: FaithfulnessPolicy { minimum_candidate_weight_ppm: 0, minimum_margin_milli: u32::MAX,
                                evidence_window_bytes: 64, max_evidence_windows: 8, max_evidence_spans: 4 }, budget },
                    };
                    Int4ScoredJobPlanner::new_with_document_encoding(compiler, Some(defaults), work).unwrap()
                });
            }
        }
        drop(preparation); drop(lease); drop(model);
        drained(&engine, &idle);
        // Retain isolated owner-only job artifacts for controller inspection.
    }
}
