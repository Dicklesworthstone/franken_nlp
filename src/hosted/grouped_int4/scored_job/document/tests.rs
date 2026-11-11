//! Pure admission/accounting fixtures; synthetic identities cannot run a model.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    jobs::population::PopulationReadLimits,
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT, lmhead::NANBEIGE_VOCAB_SIZE,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::{Int4Work, STRICT_INT4_EXECUTION}},
    tasks::{classify::{ClassificationPolicy, ClassificationAggregationPolicy}, mapreduce::ChunkLimits},
};

fn fixture(width: usize) -> (ClassificationBatchArgs, ClassificationLimits,
    Int4ClassificationMapLimits, JobLimits, JobHostLimits) {
    let row = Int4MemoryRequirement::for_context(1024, width).unwrap();
    let task = TaskBudget { max_input_tokens: 768, max_output_tokens: 16,
        max_output_bytes: 65536, max_grammar_states: 4096, max_kv_bytes: row.kv_bytes * 2 };
    let settings = ClassificationBatchArgs {
        labels: ["alpha", "beta"].map(|id| ClassificationLabel {
            id: id.to_owned(), description: String::new() }).to_vec(),
        mode: ClassificationMode::Exclusive, policy: ClassificationPolicy::default(), budget: task };
    let planning = ClassificationLimits { max_input_bytes: 4096, max_labels: 4,
        max_total_label_bytes: 1024, ..Default::default() };
    let document = Int4ClassificationMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 512,
            max_chunk_tokens: 768, context_tokens: 1024, reserved_tokens: 16,
            max_chunks: 8, max_tokenizer_calls: 1024 },
        max_model_work: Int4Work::for_sequence(width, 0, 2048, 20 * NANBEIGE_VOCAB_SIZE).unwrap(),
        max_retained_result_bytes: 1 << 20, max_result_bytes: 1 << 20,
        aggregation: ClassificationAggregationPolicy::default() };
    let lifetime = document.max_model_work.checked_add(document.max_model_work).unwrap();
    let job = JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: (1 << 20) + 4096,
        max_spool_bytes: 8 << 20, max_materialized_bytes: 8 << 20,
        max_journal_bytes: 8 << 20, max_attempts: 4, max_work: JobWork::from_int4(lifetime, 0) };
    let host = JobHostLimits {
        native: NativeLimits { context_tokens: 1024, allocator_reserve_bytes: 65536,
            run: RunLimits { max_elapsed: Duration::from_secs(30), max_checkpoints: 10000,
                cleanup_reserve_bytes: 65536 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
        preparation_reserve_bytes: 128 << 20, io_reserve_bytes: 131072,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
    };
    (settings, planning, document, job, host)
}

#[test]
fn serial_job_prices_population_and_map_with_one_shared_preparation_reserve() {
    for width in [1, 3, 32, 127, 128, MAX_MODEL_K] {
        let (settings, planning, document, job, host) = fixture(width);
        let needed = requirements(width, settings.budget, planning, document, &settings, job, host, None, JobInputMemory::CompletePopulation).unwrap();
        let map = memory_limits(host).requirements(width, settings.budget, planning, document).unwrap();
        assert_eq!(needed.memory, map);
        assert_eq!(needed.input_bytes, host.required_buffer_bytes(job).unwrap()
            + map.input_bytes - host.preparation_reserve_bytes);
        assert_eq!(map.output_reservation_bytes, map.max_output_bytes * 8 + 4096);
    }
}

#[test]
fn cohorts_price_fixed_head_geometry_once_and_preserve_complete_output_wrapper() {
    let (mut settings, planning, document, job, host) = fixture(32);
    for mode in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel] {
        settings.mode = mode;
        for chunks in [1, 2, 8] {
            let quote = memory_limits(host).cohort_requirements(32, settings.budget, planning,
                document, head_count(&settings), chunks).unwrap();
            let needed = requirements(32, settings.budget, planning, document, &settings,
                job, host, Some((chunks, quote)), JobInputMemory::CompletePopulation).unwrap();
            assert_eq!(needed.memory, quote.memory);
            assert_eq!(needed.input_bytes, host.required_buffer_bytes(job).unwrap()
                + quote.memory.input_bytes - host.preparation_reserve_bytes);
            assert_eq!(quote.memory.native.kv_bytes,
                Int4MemoryRequirement::for_context(1024, 32).unwrap().kv_bytes
                    * chunks as u64 * head_count(&settings) as u64);
            assert_eq!(quote.memory.max_output_bytes, document.max_result_bytes + 4096);
            assert!(quote.task_workspace_bytes > 0);
        }
    }
}

#[test]
fn changed_cohort_geometry_refuses_but_current_host_reserves_are_requoted() {
    let (settings, planning, document, job, host) = fixture(32);
    let quote = memory_limits(host).cohort_requirements(32, settings.budget, planning, document, 1, 2).unwrap();
    for axis in 0..7 {
        let mut changed = quote;
        match axis { 0 => changed.memory.native.kv_bytes += 1,
            1 => changed.memory.native.rope_bytes += 1,
            2 => changed.memory.native.scratch_payload_bound += 1,
            3 => changed.memory.max_output_bytes += 1,
            4 => changed.memory.output_reservation_bytes += 1,
            5 => changed.task_workspace_bytes += 1,
            _ => changed.max_cohort_output_bytes += 1 }
        assert!(requirements(32, settings.budget, planning, document, &settings,
            job, host, Some((2, changed)), JobInputMemory::CompletePopulation).is_err(), "axis {axis}");
    }
    let mut wrong_context = host; wrong_context.native.context_tokens += 1;
    assert!(requirements(32, settings.budget, planning, document, &settings,
        job, wrong_context, Some((2, quote)), JobInputMemory::CompletePopulation).is_err());
    for axis in 0..2 {
        let mut changed = host;
        match axis { 0 => changed.native.allocator_reserve_bytes += 1,
            _ => changed.preparation_reserve_bytes += 1 }
        let needed = requirements(32, settings.budget, planning, document, &settings,
            job, changed, Some((2, quote)), JobInputMemory::CompletePopulation).unwrap();
        let current = memory_limits(changed).cohort_requirements(32, settings.budget,
            planning, document, 1, 2).unwrap();
        assert_eq!(needed.memory, current.memory);
    }
}

#[test]
fn every_document_work_axis_must_fit_lifetime_without_attempt_multiplication() {
    let (settings, planning, document, mut job, host) = fixture(32);
    job.max_work = JobWork::from_int4(document.max_model_work, 0);
    for axis in 0..6 {
        let mut wrong = job;
        match axis { 0 => wrong.max_work.model.forward_positions -= 1,
            1 => wrong.max_work.model.projected_logits -= 1,
            2 => wrong.max_work.model.attention_pairs -= 1,
            3 => wrong.max_work.model.projections.dot_products -= 1,
            4 => wrong.max_work.model.projections.multiply_accumulates -= 1,
            _ => wrong.max_work.mask_node_visits = 1 }
        assert!(requirements(32, settings.budget, planning, document, &settings, wrong, host, None, JobInputMemory::CompletePopulation).is_err());
    }
    let before = requirements(32, settings.budget, planning, document, &settings, job, host, None, JobInputMemory::CompletePopulation).unwrap();
    job.max_items = 40; job.max_attempts = 100;
    let after = requirements(32, settings.budget, planning, document, &settings, job, host, None, JobInputMemory::CompletePopulation).unwrap();
    assert_eq!(before.memory, after.memory);
    assert_eq!(after.input_bytes - before.input_bytes, 37 * 1024);
}

#[test]
fn result_native_and_host_storage_shortfalls_refuse_before_population_io() {
    let (settings, planning, document, job, host) = fixture(32);
    for axis in 0..8 {
        let mut host = host; let mut job = job; let mut settings = settings.clone();
        match axis { 0 => job.max_result_bytes = document.max_result_bytes as usize - 1,
            1 => settings.budget.max_kv_bytes = 1,
            2 => host.native.context_tokens = 1023,
            3 => host.native.allocator_reserve_bytes = 0,
            4 => host.preparation_reserve_bytes = (128 << 20) - 1,
            5 => host.preparation_reserve_bytes = u64::MAX,
            6 => host.journal_reserve_bytes = 0,
            _ => host.serialization_reserve_bytes = 0 }
        assert!(requirements(32, settings.budget, planning, document, &settings, job, host, None, JobInputMemory::CompletePopulation).is_err(), "axis {axis}");
    }
}

#[test]
fn private_fixed_label_spare_capacities_are_priced_without_cloning_them() {
    let (mut settings, planning, document, job, host) = fixture(32);
    let before = requirements(32, settings.budget, planning, document, &settings, job, host, None, JobInputMemory::CompletePopulation).unwrap();
    let mut labels = Vec::with_capacity(planning.max_labels + 4);
    labels.extend(settings.labels); settings.labels = labels;
    let mut id = String::with_capacity(planning.max_total_label_bytes + 4096);
    id.push_str(&settings.labels[0].id); settings.labels[0].id = id;
    let expected = spare_label_bytes(&settings, planning).unwrap();
    assert!(expected > 4096);
    let after = requirements(32, settings.budget, planning, document, &settings, job, host, None, JobInputMemory::CompletePopulation).unwrap();
    assert_eq!(after.input_bytes - before.input_bytes, expected);
}

fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"classification-document-host-admission-only");
    ExecutionIdentity { schema_version: 1, source_revision: PINNED_REVISION.to_owned(),
        logical_model_digest: d, artifact_format: INT4_SOURCE_FORMAT.to_owned(),
        quant_recipe: format!("{GROUPED_INT4_RECIPE}:group=32"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, 32_usize)).unwrap()),
        tokenizer_digest: d, template_digest: d, task_spec: "classify-v1".to_owned(), taskir_digest: d,
        prompt_digest: d, grammar_compiler_version: "none".to_owned(), schema_digest: d,
        numerics_profile: NumericsProfile::StrictQuantized { version: 2 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "complete-candidate-eos".to_owned(), thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None, calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}

#[test]
fn all_chunk_identities_and_complete_admission_dimensions_are_checked() {
    let (_, _, document, _, _) = fixture(32); let base = identity();
    let kv = Int4MemoryRequirement::for_context(1024, 32).unwrap().kv_bytes;
    for axis in 0..12 {
        let mut ids = [identity(), identity()];
        match axis { 0 => ids[1].task_spec = "sentiment-v1".to_owned(),
            1 => ids[1].tokenizer_digest = Sha256Digest::of_bytes(b"changed-tokenizer"),
            2 => ids[1].template_digest = Sha256Digest::of_bytes(b"changed-template"),
            3 => ids[1].numerics_profile = NumericsProfile::StrictQuantized { version: 1 }, _ => {} }
        let mut request = Int4ClassificationDocumentAdmission { identities: &ids,
            model_work: document.max_model_work, kv_reservation_bytes: kv,
            max_result_bytes: document.max_result_bytes, chunk_count: 2, head_count: 1 };
        match axis { 4 => request.chunk_count = 0, 5 => request.chunk_count = 3,
            6 => request.kv_reservation_bytes -= 1, 7 => request.kv_reservation_bytes += 1,
            8 => request.head_count = 2, 9 => request.model_work.forward_positions += 1,
            10 => request.max_result_bytes -= 1, 11 => request.max_result_bytes += 1, _ => {} }
        assert!(check_request(32, &base, document, kv, 1, document.max_result_bytes, &request).is_err(), "axis {axis}");
    }
    let ids = [identity(), identity()];
    let request = Int4ClassificationDocumentAdmission { identities: &ids, model_work: document.max_model_work,
        kv_reservation_bytes: kv, max_result_bytes: document.max_result_bytes, chunk_count: 2, head_count: 1 };
    assert!(check_request(32, &base, document, kv, 1, document.max_result_bytes, &request).is_ok());
}

#[test]
fn real_factory_and_transferred_reader_cross_the_owned_host_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Int4ClassificationDocumentJobPlanner>();
    send::<DocumentInput<std::io::Cursor<Vec<u8>>>>();
    send::<DocumentExecution<std::io::Cursor<Vec<u8>>>>();
    let _entrypoint = NlpEngine::job_int4_classification_document::<std::io::Cursor<Vec<u8>>>;
    let _team = NlpEngine::job_int4_classification_document_with_threads::<std::io::Cursor<Vec<u8>>>;
}

#[test]
fn durable_team_width_is_independent_of_native_chunk_cohort_authority() {
    for chunks in [1, 2, 8] {
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
        native_engine::{decode::DecodeCancellationKind, strict_int4::weights::projection_fixture},
        tasks::classify::grouped_int4::Int4ClassificationPlanner,
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
                Action::Refuse => panic!("unadmitted classification job read private input"),
                Action::Panic => panic!("injected classification job reader panic"),
                Action::Cancel(cause) => self.stop.cancel(cause),
            }
            Ok(&[])
        }
        fn consume(&mut self, _: usize) {}
    }
    fn planner(model: &ResidentInt4, compact: bool, cohorts: Option<usize>)
        -> (Int4ClassificationDocumentJobPlanner, SourceJobRequest, JobHostLimits) {
        let (settings, planning, document, job, mut limits) = fixture(32);
        limits.native.run.max_elapsed = Duration::from_secs(60);
        limits.native.run.max_checkpoints = 100_000;
        let encoding = if compact { UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: 4096, max_output_tokens: 4096, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
        }) } else { UntrustedEncoding::ByteFallback };
        let compiler = Int4ClassificationPlanner::for_model(model.model_identity(), settings.budget, planning).unwrap()
            .with_document_encoding(encoding).unwrap();
        let planner = Int4ClassificationDocumentJobPlanner::new(compiler, settings, document, job).unwrap();
        let planner = if let Some(chunks) = cohorts {
            planner.with_chunk_cohorts(chunks, memory_limits(limits)).unwrap()
        } else { planner };
        let request = SourceJobRequest { root: PathBuf::from("/nonexistent/fnlp-classification-job-team-refusal"),
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
    fn classification_job_teams_preserve_cancellation_and_population_ownership() {
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
                let error = host.job_int4_classification_document_with_threads(&model, planner,
                    request, limits, PackedInt4Kernel::scalar(), width, reader, stop).err().unwrap();
                let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                    else { panic!("classification pre-cancellation lost its typed cause"); };
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
            let error = host.job_int4_classification_document_with_threads(&model, planner,
                request, limits, PackedInt4Kernel::scalar(), width, reader, stop).err().unwrap();
            assert!(matches!(error, HostedJobError::Host(HostedError::Limits(_))));
            assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
            assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
            drained(&host, &baseline);
        }
        // Leave exactly one byte less than the complete native plus worker
        // reservation. The numerical team must refuse before private input IO.
        {
            let (planner, request, limits) = planner(&model, false, None);
            let required = requirements(32, planner.document_budget(), planner.planning_limits(),
                planner.document_limits(), planner.request(), request.limits, limits, None,
                request.mode.input_memory()).unwrap();
            let room = sum(&[required.input_bytes, required.memory.native.kv_bytes,
                required.memory.workspace_bytes, int4_task_team_workspace_bytes(4, 32).unwrap() - 1]).unwrap();
            let bytes = host.resources().config().memory_ceiling_bytes
                .checked_sub(baseline.committed_bytes + baseline.reserved_bytes).unwrap().checked_sub(room).unwrap();
            let lease = host.resources().acquire_lease();
            let held = Pending::reserve(&lease, MemoryClass::JobBuffers, bytes).unwrap().commit().unwrap();
            let stop = CancellationToken::default();
            let observed = Arc::new(Observation::default());
            let reader = Reader { resources: Arc::clone(host.resources()), observed: Arc::clone(&observed),
                action: Action::Refuse, stop: stop.clone() };
            let error = host.job_int4_classification_document_with_threads(&model, planner,
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
            let error = host.job_int4_classification_document_with_threads(&model, planner,
                request, limits, PackedInt4Kernel::scalar(), 4, reader, stop).err().unwrap();
            match action {
                Action::Cancel(cause) => {
                    let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                        else { panic!("classification reader cancellation lost its typed cause"); };
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
