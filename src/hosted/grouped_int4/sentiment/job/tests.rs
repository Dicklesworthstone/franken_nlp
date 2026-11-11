//! Model-free durable-host refusal and ownership contracts. These fixtures
//! cannot manufacture native weights or confer model-quality evidence.
use super::*;
use std::time::Duration;
use crate::{jobs::population::PopulationReadLimits,
    native_engine::lmhead::NANBEIGE_VOCAB_SIZE,
    tasks::{mapreduce::ChunkLimits, sentiment::SentimentAggregationPolicy}};

fn fixture() -> (TaskBudget, SentimentLimits, Int4SentimentMapLimits, JobLimits, JobHostLimits) {
    let task = TaskBudget { max_input_tokens: 128, max_output_tokens: 16,
        max_output_bytes: 65_536, max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(256, 32).unwrap().kv_bytes * 4 };
    let mapping = Int4SentimentMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 512,
            max_chunk_tokens: 128, context_tokens: 256, reserved_tokens: 16,
            max_chunks: 64, max_tokenizer_calls: 1024 },
        max_model_work: Int4Work::for_sequence(32, 0, 1024, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
        max_retained_result_bytes: 8 << 20, max_result_bytes: 1 << 20,
        aggregation: SentimentAggregationPolicy::default(),
    };
    let job = JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65_536, max_result_bytes: (1 << 20) + 4096,
        max_spool_bytes: 8 << 20, max_materialized_bytes: 8 << 20,
        max_journal_bytes: 8 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(mapping.max_model_work, 0) };
    let host = JobHostLimits {
        native: NativeLimits { context_tokens: 256, allocator_reserve_bytes: 65_536,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 10000,
                cleanup_reserve_bytes: 65_536 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
        preparation_reserve_bytes: 128 << 20, io_reserve_bytes: 131_072,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
    };
    (task, SentimentLimits::default(), mapping, job, host)
}

#[test]
fn complete_population_and_replanned_documents_coexist_in_the_reservation() {
    let (task, planning, mapping, job, host) = fixture();
    for (axes, chunks) in [(1, None), (4, None), (4, Some(1)), (4, Some(16)), (1, Some(64))] {
        let needed = requirements(32, task, planning, mapping, job, host, axes, chunks, JobInputMemory::CompletePopulation).unwrap();
        let map = map_memory(host, axes, chunks).requirements(32, task, planning, mapping).unwrap();
        assert_eq!(needed.native, map.native);
        assert_eq!(needed.output_reservation_bytes, map.output_reservation_bytes);
        assert!(needed.input_bytes > host.required_buffer_bytes(job).unwrap());
        assert_eq!(needed.input_bytes, host.required_buffer_bytes(job).unwrap()
            + map.input_bytes - host.preparation_reserve_bytes
            + mapping.chunks.max_chunks as u64 * SENTIMENT_DOCUMENT_CHUNK_METADATA_BYTES);
        // Native windows never replace the original population commitment.
        let mut larger = job;
        larger.max_snapshot_bytes += 4096;
        let larger = requirements(32, task, planning, mapping, larger, host, axes, chunks, JobInputMemory::CompletePopulation).unwrap();
        assert_eq!(larger.input_bytes - needed.input_bytes, 8192);
        assert_eq!(larger.native, needed.native);
    }
}

#[test]
fn each_native_work_axis_must_fit_the_persisted_lifetime_allowance() {
    let (task, planning, mapping, job, host) = fixture();
    for axis in 0..6 {
        let mut limited = job;
        match axis {
            0 => limited.max_work.model.forward_positions -= 1,
            1 => limited.max_work.model.projected_logits -= 1,
            2 => limited.max_work.model.attention_pairs -= 1,
            3 => limited.max_work.model.projections.dot_products -= 1,
            4 => limited.max_work.model.projections.multiply_accumulates -= 1,
            _ => limited.max_work.mask_node_visits = 1,
        }
        assert!(requirements(32, task, planning, mapping, limited, host, 4, None, JobInputMemory::CompletePopulation).is_err(), "axis {axis}");
    }
    let quote = requirements(32, task, planning, mapping, job, host, 4, None, JobInputMemory::CompletePopulation).unwrap();
    let mut more_attempts = job;
    more_attempts.max_attempts = 100;
    assert_eq!(quote, requirements(32, task, planning, mapping, more_attempts, host, 4, None, JobInputMemory::CompletePopulation).unwrap());
}

#[test]
fn oversized_axis_windows_refuse_without_clamping_the_fixed_recipe() {
    let (task, planning, mapping, job, host) = fixture();
    for (axes, chunks) in [(0, None), (5, None), (4, Some(0)), (4, Some(17)),
        (2, Some(33)), (1, Some(65)), (1, Some(usize::MAX))] {
        assert!(requirements(32, task, planning, mapping, job, host, axes, chunks, JobInputMemory::CompletePopulation).is_err());
    }
    let needed = requirements(32, task, planning, mapping, job, host, 2, Some(3), JobInputMemory::CompletePopulation).unwrap();
    let row = Int4MemoryRequirement::for_context(256, 32).unwrap();
    assert_eq!(needed.native.kv_bytes, row.kv_bytes * 6);
    let mut missing_axis = task;
    missing_axis.max_kv_bytes = row.kv_bytes * 2 - 1;
    assert!(requirements(32, missing_axis, planning, mapping, job, host, 2, Some(3), JobInputMemory::CompletePopulation).is_err());
}

#[test]
fn stored_document_cap_accounts_for_actual_cohort_wrapper() {
    let (task, planning, mapping, mut job, host) = fixture();
    job.max_result_bytes = mapping.max_result_bytes as usize;
    requirements(32, task, planning, mapping, job, host, 4, None, JobInputMemory::CompletePopulation).unwrap();
    assert!(requirements(32, task, planning, mapping, job, host, 4, Some(2), JobInputMemory::CompletePopulation).is_err());
    job.max_result_bytes += 4096;
    let quote = requirements(32, task, planning, mapping, job, host, 4, Some(2), JobInputMemory::CompletePopulation).unwrap();
    assert_eq!(quote.max_output_bytes, job.max_result_bytes as u64);
    assert!(quote.output_reservation_bytes > quote.max_output_bytes);
    job.max_result_bytes -= 1;
    assert!(requirements(32, task, planning, mapping, job, host, 4, Some(2), JobInputMemory::CompletePopulation).is_err());
}

#[test]
fn host_shortfalls_and_arithmetic_overflow_fail_before_population_io() {
    let (task, planning, mapping, job, host) = fixture();
    for field in 0..11 {
        let mut host = host;
        match field {
            0 => host.native.context_tokens = 255,
            1 => host.native.context_tokens = 257,
            2 => host.native.allocator_reserve_bytes = 0,
            3 => host.preparation_reserve_bytes = (128 << 20) - 1,
            4 => host.preparation_reserve_bytes = u64::MAX,
            5 => host.io_reserve_bytes = u64::MAX,
            6 => host.journal_reserve_bytes = 0,
            7 => host.serialization_reserve_bytes = 0,
            8 => host.native.run.max_checkpoints = 1,
            9 => host.transport.max_stream_bytes = 0,
            _ => host.native.allocator_reserve_bytes = u64::MAX,
        }
        assert!(requirements(32, task, planning, mapping, job, host, 4, None, JobInputMemory::CompletePopulation).is_err(), "field {field}");
    }
    for width in [0, 10753, usize::MAX] {
        assert!(requirements(width, task, planning, mapping, job, host, 4, None, JobInputMemory::CompletePopulation).is_err());
    }
}

#[test]
fn owned_reader_and_private_recipe_travel_through_the_hosted_job_boundary() {
    let _host = NlpEngine::job_int4_sentiment_document::<std::io::Cursor<Vec<u8>>>;
    let _team = NlpEngine::job_int4_sentiment_document_with_threads::<std::io::Cursor<Vec<u8>>>;
    fn send<T: Send + 'static>() {}
    send::<JobInput<std::io::Cursor<Vec<u8>>>>();
    send::<JobExecution<std::io::Cursor<Vec<u8>>>>();
}

#[test]
fn durable_teams_reject_all_explicit_native_chunk_cohorts() {
    for chunks in [1, 2, 16] {
        check_team_schedule(None, Some(chunks)).unwrap();
        for threads in [2, 4, 64] {
            check_team_schedule(Some(threads), None).unwrap();
            assert!(matches!(check_team_schedule(Some(threads), Some(chunks)), Err(HostedError::Limits(_))));
        }
    }
}

#[cfg(feature = "asupersync-runtime")]
mod team {
    use super::*;
    use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
        hosted::corpus::jobs::JobOpenMode, jobs::{JobId, JobSecret},
        native_engine::{decode::DecodeCancellationKind, lmhead::scoring::ScoringMode,
            strict_int4::weights::projection_fixture},
        tasks::sentiment::{SentimentAxis, SentimentOptions, SentimentPolicy,
            batch::SentimentBatchArgs, grouped_int4::Int4SentimentPlanner},
        tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding}};
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
                Action::Refuse => panic!("unadmitted sentiment job read private input"),
                Action::Panic => panic!("injected sentiment job reader panic"),
                Action::Cancel(cause) => self.stop.cancel(cause),
            }
            Ok(&[])
        }
        fn consume(&mut self, _: usize) {}
    }
    fn options() -> SentimentOptions {
        let controls = crate::tokenizer::pinned_controls::pinned().unwrap();
        let mut ends = controls.template_controls().entries().iter()
            .filter(|entry| entry.special && entry.surface == crate::template::IM_END);
        let eos_token_id = ends.next().unwrap().id;
        assert!(ends.next().is_none());
        SentimentOptions { mode: ScoringMode::FullVocabulary, eos_token_id,
            policy: SentimentPolicy { minimum_peak_weight_ppm: 123_000, maximum_normalized_entropy_ppm: 800_000 } }
    }
    fn planner(model: &ResidentInt4, compact: bool, cohorts: Option<usize>)
        -> (Int4SentimentDocumentJobPlanner, SourceJobRequest, JobHostLimits) {
        let (mut budget, planning, mut mapping, mut job, mut limits) = fixture();
        // Fixed defaults need a real prompt-capacity envelope even though no
        // native scoring can occur in these transport/cancellation fixtures.
        budget.max_input_tokens = 768;
        budget.max_kv_bytes = Int4MemoryRequirement::for_context(1024, 32).unwrap().kv_bytes * 4;
        mapping.chunks.context_tokens = 1024;
        mapping.chunks.max_chunk_tokens = 768;
        mapping.chunks.max_chunks = 8;
        mapping.max_model_work = Int4Work::for_sequence(32, 0, 8 * 4 * 1024,
            8 * 4 * 16 * NANBEIGE_VOCAB_SIZE).unwrap();
        job.max_work = JobWork::from_int4(mapping.max_model_work, 0);
        limits.native.context_tokens = 1024;
        limits.native.run.max_elapsed = Duration::from_secs(60);
        limits.native.run.max_checkpoints = 100_000;
        let encoding = if compact { UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: 4096, max_output_tokens: 4096, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
        }) } else { UntrustedEncoding::ByteFallback };
        let compiler = Int4SentimentPlanner::for_model(model.model_identity(), options(), budget, planning).unwrap()
            .with_document_encoding(encoding).unwrap();
        let settings = SentimentBatchArgs { axes: SentimentAxis::ALL.to_vec(), budget };
        let planner = Int4SentimentDocumentJobPlanner::new(compiler, settings, mapping, job).unwrap();
        let planner = if let Some(chunks) = cohorts { planner.with_chunk_cohorts(chunks).unwrap() } else { planner };
        let request = SourceJobRequest { root: PathBuf::from("/nonexistent/fnlp-sentiment-job-team-refusal"),
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
    #[ignore = "controller-only fresh runtime: tiny weights prove admission and drain, never native or durable-result parity"]
    fn sentiment_job_teams_preserve_cancellation_and_population_ownership() {
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
                let (planner, request, limits) = planner(&model, true, None);
                let charge = limits.required_buffer_bytes(request.limits).unwrap();
                let stop = CancellationToken::default(); stop.cancel(cause);
                let observed = Arc::new(Observation::default());
                let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                    action: Action::Refuse, stop: stop.clone() };
                let error = host.job_int4_sentiment_document_with_threads(&model, planner,
                    request, limits, PackedInt4Kernel::scalar(), width, reader, stop).err().unwrap();
                let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                    else { panic!("sentiment pre-cancellation lost its typed cause"); };
                assert_eq!(stop.kind, cause); assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
                assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
                assert!(observed.input_at_drop.load(Ordering::SeqCst) >= PREPARATION + charge);
                drained(&host, &baseline);
            }
        }
        for (width, cohorts) in [(0, None), (1, None), (5, None), (65, None), (usize::MAX, None),
            (2, Some(1)), (4, Some(2))] {
            let (planner, request, limits) = planner(&model, false, cohorts);
            let stop = CancellationToken::default();
            let observed = Arc::new(Observation::default());
            let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                action: Action::Refuse, stop: stop.clone() };
            let error = host.job_int4_sentiment_document_with_threads(&model, planner,
                request, limits, PackedInt4Kernel::scalar(), width, reader, stop).err().unwrap();
            assert!(matches!(error, HostedJobError::Host(HostedError::Limits(_))));
            assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
            assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
            drained(&host, &baseline);
        }
        {
            let (planner, request, limits) = planner(&model, false, None);
            let required = requirements(32, planner.task_budget(), planner.planning_limits(),
                planner.document_limits(), request.limits, limits, planner.max_axes(), None,
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
            let error = host.job_int4_sentiment_document_with_threads(&model, planner,
                request, limits, PackedInt4Kernel::scalar(), 4, reader, stop).err().unwrap();
            assert!(matches!(error, HostedJobError::Host(HostedError::Reservation(_))));
            assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
            assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
            drop(held); drop(lease);
            drained(&host, &baseline);
        }
        for action in [Action::Cancel(Deadline), Action::Panic] {
            let (planner, request, limits) = planner(&model, true, None);
            let charge = limits.required_buffer_bytes(request.limits).unwrap();
            let stop = CancellationToken::default();
            let observed = Arc::new(Observation::default());
            let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                action, stop: stop.clone() };
            let error = host.job_int4_sentiment_document_with_threads(&model, planner,
                request, limits, PackedInt4Kernel::scalar(), 4, reader, stop).err().unwrap();
            match action {
                Action::Cancel(cause) => {
                    let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                        else { panic!("sentiment reader cancellation lost its typed cause"); };
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
}

#[cfg(feature = "asupersync-runtime")]
mod native;

#[test]
fn retained_replay_keeps_one_compact_workspace_without_restoring_population_payloads() {
    use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
    let policy = SafeBpeLimits { max_input_bytes: 262_144, max_output_tokens: 131_072,
        max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 };
    let encoding = UntrustedEncoding::SafeBpeV1(policy);
    let workspace = policy.requirements().unwrap().workspace_bytes;
    let (task, planning, mapping, job, host) = fixture();
    for cohorts in [None, Some(2)] {
        let complete = requirements(32, task, planning, mapping, job, host, 4, cohorts,
            JobInputMemory::CompletePopulation).unwrap().with_input_encoding(&encoding).unwrap();
        let retained = requirements(32, task, planning, mapping, job, host, 4, cohorts,
            JobInputMemory::RetainedRecord).unwrap();
        let compact = retained.with_input_encoding(&encoding).unwrap();
        assert_eq!(compact.input_bytes - retained.input_bytes, workspace);
        assert_eq!(complete.input_bytes - compact.input_bytes, 2 * job.max_snapshot_bytes);
        assert_eq!(complete.native, compact.native);
        assert_eq!(compact.native, retained.native);
        assert_eq!(compact.output_reservation_bytes, retained.output_reservation_bytes);
    }
}
