//! UNRUN admission/ownership fixtures; no model inference or crash proof.
use super::*;
use crate::{
    artifact::converter::PINNED_REVISION,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    hosted::RunLimits,
    jobs::population::PopulationReadLimits,
    native_engine::{generation::grouped_int4::INT4_SOURCE_FORMAT,
        portable_int8::int4::convert::GROUPED_INT4_RECIPE,
        strict_int4::STRICT_INT4_EXECUTION},
};
use std::{io::Cursor, time::Duration};

const WIDTH: usize = 32;
fn identity() -> ExecutionIdentity {
    let d = Sha256Digest::of_bytes(b"synthetic-generation-host-admission");
    ExecutionIdentity {
        schema_version: 1, source_revision: PINNED_REVISION.to_owned(), logical_model_digest: d,
        artifact_format: INT4_SOURCE_FORMAT.to_owned(), quant_recipe: format!("{GROUPED_INT4_RECIPE}:group={WIDTH}"),
        packing_set_digest: Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&(
            INT4_SOURCE_FORMAT, "low-nibble-first-no-padding-f32le-scales-v1", d, WIDTH,
        )).unwrap()),
        tokenizer_digest: d, template_digest: d, task_spec: "generate-v1".to_owned(),
        taskir_digest: d, prompt_digest: d, grammar_compiler_version: "none".to_owned(),
        schema_digest: d, numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "fnlp-sampler-v1".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: d, decision_policy_digest: d,
        backend_semantic_version: STRICT_INT4_EXECUTION.to_owned(), host_class: None, compiler_identity: None,
    }
}
fn fixture() -> (TaskBudget, GenerationEnvelope, JobLimits, JobHostLimits) {
    let memory = Int4MemoryRequirement::for_context(64, WIDTH).unwrap();
    let work = Int4Work::for_sequence(WIDTH, 0, 64, 16 * NANBEIGE_VOCAB_SIZE).unwrap();
    let task = TaskBudget { max_input_tokens: 48, max_output_tokens: 16,
        max_output_bytes: 65536, max_grammar_states: 4096, max_kv_bytes: memory.kv_bytes };
    let job = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65536, max_result_bytes: 131072, max_spool_bytes: 1 << 20,
        max_materialized_bytes: 1 << 20, max_journal_bytes: 1 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(work, 0) };
    let envelope = GenerationEnvelope { context_tokens: 64, kv_bytes: memory.kv_bytes,
        native: Int4GenerationBudget { native: Int4RunBudget::exact(work),
            max_kv_bytes: memory.kv_bytes, max_sampler_bytes: 32 << 20 },
        lifetime: job.max_work, result_bytes: task.max_output_bytes, output_tokens: 16 };
    let host = JobHostLimits {
        native: NativeLimits { context_tokens: 64, allocator_reserve_bytes: 65536,
            run: RunLimits { max_elapsed: Duration::from_secs(10), max_checkpoints: 10000,
                cleanup_reserve_bytes: 65536 } },
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 100 },
        preparation_reserve_bytes: 1 << 20, io_reserve_bytes: 131072,
        journal_reserve_bytes: 1 << 20, serialization_reserve_bytes: 1 << 20,
    };
    (task, envelope, job, host)
}
fn request(identity: &ExecutionIdentity, envelope: GenerationEnvelope) -> Int4GenerationAdmission<'_> {
    Int4GenerationAdmission { identity,
        model_work: Int4Work::for_sequence(WIDTH, 0, 48, 8 * NANBEIGE_VOCAB_SIZE).unwrap(),
        sampler_bytes: 16 << 20, kv_reservation_bytes: envelope.kv_bytes,
        max_result_bytes: envelope.result_bytes, max_output_tokens: envelope.output_tokens }
}

#[test]
fn full_population_context_and_sampler_coexist_in_the_same_admission() {
    let (task, envelope, job, host) = fixture();
    let need = requirements(WIDTH, task, envelope, job, host, JobInputMemory::CompletePopulation).unwrap();
    assert_eq!(need.input_bytes, host.required_buffer_bytes(job).unwrap());
    assert_eq!(need.native.kv_bytes, task.max_kv_bytes);
    assert_eq!(need.workspace_bytes, need.native.rope_bytes + need.native.scratch_payload_bound
        + host.native.allocator_reserve_bytes + envelope.native.max_sampler_bytes);
    // The fixed task result and larger immutable storage cap have separate
    // accounting roles; the per-item claim must not silently widen to storage.
    assert!(envelope.result_bytes < job.max_result_bytes as u64);
}

#[test]
fn looser_or_tighter_context_and_kv_cannot_change_the_frozen_recipe() {
    for axis in 0..6 {
        let (mut task, mut envelope, job, mut host) = fixture();
        match axis { 0 => task.max_kv_bytes -= 1, 1 => task.max_kv_bytes += 1,
            2 => envelope.native.max_kv_bytes -= 1, 3 => envelope.kv_bytes += 1,
            4 => envelope.context_tokens -= 1, _ => host.native.context_tokens += 1 }
        assert!(requirements(WIDTH, task, envelope, job, host, JobInputMemory::CompletePopulation).is_err(), "axis {axis}");
    }
}

#[test]
fn missing_reserves_and_workspace_overflow_refuse_before_input() {
    for axis in 0..7 {
        let (task, mut envelope, job, mut host) = fixture();
        match axis { 0 => host.preparation_reserve_bytes = 0, 1 => host.io_reserve_bytes = 0,
            2 => host.journal_reserve_bytes = 0, 3 => host.serialization_reserve_bytes = 0,
            4 => host.native.allocator_reserve_bytes = 0, 5 => envelope.native.max_sampler_bytes = 0,
            _ => envelope.native.max_sampler_bytes = u64::MAX }
        assert!(requirements(WIDTH, task, envelope, job, host, JobInputMemory::CompletePopulation).is_err(), "axis {axis}");
    }
}

#[test]
fn every_actual_work_axis_remains_within_the_persisted_lifetime_envelope() {
    let (_, envelope, _, _) = fixture(); let base = identity();
    let req = request(&base, envelope);
    for axis in 0..5 {
        let mut tighter = envelope;
        tighter.lifetime = JobWork::from_int4(req.model_work, 0);
        match axis { 0 => tighter.lifetime.model.forward_positions -= 1,
            1 => tighter.lifetime.model.projected_logits -= 1,
            2 => tighter.lifetime.model.attention_pairs -= 1,
            3 => tighter.lifetime.model.projections.dot_products -= 1,
            _ => tighter.lifetime.model.projections.multiply_accumulates -= 1 }
        assert!(check_request(WIDTH, &base, tighter, &req).is_err(), "axis {axis}");
    }
}

#[test]
fn retained_generation_drops_only_snapshot_copies_and_keeps_sampler_and_native_floors() {
    let (task, envelope, job, host) = fixture();
    let complete = requirements(WIDTH, task, envelope, job, host, JobInputMemory::CompletePopulation).unwrap();
    let retained = requirements(WIDTH, task, envelope, job, host, JobInputMemory::RetainedRecord).unwrap();
    assert_eq!(complete.input_bytes - retained.input_bytes, 2 * job.max_snapshot_bytes);
    assert_eq!(retained.input_bytes, host.required_buffer_bytes_for_input(job, JobInputMemory::RetainedRecord).unwrap());
    assert_eq!(retained.native, complete.native);
    assert_eq!(retained.workspace_bytes, complete.workspace_bytes);
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let mut short = job; short.max_work.model.projections.multiply_accumulates -= 1;
        assert!(requirements(WIDTH, task, envelope, short, host, input).is_err());
        let mut missing = envelope; missing.native.max_sampler_bytes = 0;
        assert!(requirements(WIDTH, task, missing, job, host, input).is_err());
        let mut missing = host; missing.native.allocator_reserve_bytes = 0;
        assert!(requirements(WIDTH, task, envelope, job, missing, input).is_err());
    }
}

#[test]
fn a_different_group_partition_cannot_relabel_cheaper_work_as_the_actual_model() {
    let (_, envelope, _, _) = fixture(); let base = identity();
    let mut req = request(&base, envelope);
    assert!(check_request(WIDTH, &base, envelope, &req).is_ok());
    req.model_work = Int4Work::for_sequence(128, 0, 48, 8 * NANBEIGE_VOCAB_SIZE).unwrap();
    assert!(check_request(WIDTH, &base, envelope, &req).is_err());
    req.model_work = request(&base, envelope).model_work;
    req.model_work.attention_pairs -= 1;
    assert!(check_request(WIDTH, &base, envelope, &req).is_err());
}

#[test]
fn one_empty_cache_sequence_must_fit_the_resident_context_and_projection_schedule() {
    let (_, envelope, _, _) = fixture(); let base = identity();
    for (positions, rows) in [(0, 0), (65, NANBEIGE_VOCAB_SIZE), (1, 2 * NANBEIGE_VOCAB_SIZE), (48, 1)] {
        let mut req = request(&base, envelope);
        req.model_work = Int4Work::for_sequence(WIDTH, 0, positions, rows).unwrap();
        assert!(check_request(WIDTH, &base, envelope, &req).is_err());
    }
}

#[test]
fn per_item_prompt_and_sampling_key_change_but_fixed_task_assets_and_modes_cannot() {
    let (_, envelope, _, _) = fixture(); let base = identity();
    let mut item = base.clone();
    item.prompt_digest = Sha256Digest::of_bytes(b"different exact prompt");
    item.taskir_digest = Sha256Digest::of_bytes(b"different item task graph");
    item.decision_policy_digest = Sha256Digest::of_bytes(b"different item/sample addressed key");
    assert!(check_request(WIDTH, &base, envelope, &request(&item, envelope)).is_ok());
    for axis in 0..8 {
        let mut wrong = item.clone();
        match axis { 0 => wrong.task_spec = "chat-v1".to_owned(),
            1 => wrong.tokenizer_digest = Sha256Digest::of_bytes(b"wrong tokenizer"),
            2 => wrong.template_digest = Sha256Digest::of_bytes(b"wrong template"),
            3 => wrong.sampler_version = "different-sampler".to_owned(),
            4 => wrong.thinking_mode = ThinkingMode::Enabled, 5 => wrong.tool_mode = ToolMode::Json,
            6 => wrong.numerics_profile = NumericsProfile::StrictQuantized { version: 1 },
            _ => wrong.quant_recipe = format!("{GROUPED_INT4_RECIPE}:group=128") }
        assert!(check_request(WIDTH, &base, envelope, &request(&wrong, envelope)).is_err(), "axis {axis}");
    }
}

#[test]
fn output_and_sampler_claims_cannot_shrink_or_widen_the_fixed_admission() {
    let (_, envelope, _, _) = fixture(); let base = identity();
    for axis in 0..8 {
        let mut req = request(&base, envelope);
        match axis { 0 => req.max_result_bytes -= 1, 1 => req.max_result_bytes += 1,
            2 => req.max_output_tokens -= 1, 3 => req.max_output_tokens += 1,
            4 => req.sampler_bytes = 0, 5 => req.sampler_bytes = envelope.native.max_sampler_bytes + 1,
            6 => req.kv_reservation_bytes -= 1, _ => req.kv_reservation_bytes += 1 }
        assert!(check_request(WIDTH, &base, envelope, &req).is_err(), "axis {axis}");
    }
}

#[test]
fn owned_population_factory_and_native_result_fit_the_existing_blocking_boundary() {
    fn send<T: Send + 'static>() {}
    send::<Int4GenerationJobPlanner>();
    send::<GenerationInput<Cursor<Vec<u8>>>>();
    send::<GenerationExecution<Cursor<Vec<u8>>>>();
    send::<GenerationCompletion<Cursor<Vec<u8>>>>();
    send::<crate::tasks::chat::grouped_int4::Int4ChatResult>();
    let _entry = NlpEngine::job_int4_generation::<Cursor<Vec<u8>>>;
    let _team_entry = NlpEngine::job_int4_generation_with_threads::<Cursor<Vec<u8>>>;
}

mod team {
    use super::*;
    use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
        hosted::corpus::jobs::JobOpenMode,
        jobs::{JobError, JobId, JobSecret, runner::generation::{GenerationJobTask,
            grouped_int4::Int4GenerationJobConfig}},
        native_engine::{decode::DecodeCancellationKind,
            generation::{GenerationLimits, GenerationOptions},
            strict_int4::weights::projection_fixture},
        tasks::chat::ChatLimits,
        tokenizer::embedded::EmbeddedTokenizer};
    use std::{io::{self, BufRead, Read}, sync::atomic::{AtomicU64, AtomicUsize, Ordering}};

    const MIB: u64 = 1 << 20;
    const PREPARATION: u64 = 128 * MIB;

    #[derive(Clone, Copy)]
    enum ReadAction { Refuse, Cancel(DecodeCancellationKind), Panic, Empty }
    #[derive(Default)]
    struct Observation {
        reads: AtomicUsize, drops: AtomicUsize,
        input_bytes_at_drop: AtomicU64, worker_bytes_at_read: AtomicU64,
    }
    struct Reader {
        resources: Arc<EngineResources>, observation: Arc<Observation>,
        action: ReadAction, stop: CancellationToken,
    }
    impl Drop for Reader {
        fn drop(&mut self) {
            let snapshot = self.resources.memory_snapshot();
            let bytes = snapshot.by_class.get(&MemoryClass::JobBuffers)
                .map_or(0, |value| value.committed_bytes + value.reserved_bytes);
            self.observation.input_bytes_at_drop.store(bytes, Ordering::SeqCst);
            self.observation.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl Read for Reader {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            self.fill_buf()?; Ok(0)
        }
    }
    impl BufRead for Reader {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            self.observation.reads.fetch_add(1, Ordering::SeqCst);
            let snapshot = self.resources.memory_snapshot();
            let bytes = snapshot.by_class.get(&MemoryClass::ActivationScratch)
                .map_or(0, |value| value.committed_bytes);
            self.observation.worker_bytes_at_read.store(bytes, Ordering::SeqCst);
            match self.action {
                ReadAction::Refuse => panic!("refused job read private input"),
                ReadAction::Panic => panic!("synthetic input panic"),
                ReadAction::Cancel(cause) => self.stop.cancel(cause),
                ReadAction::Empty => (),
            }
            Ok(&[])
        }
        fn consume(&mut self, _: usize) {}
    }
    fn reader(host: &NlpEngine, action: ReadAction, stop: CancellationToken)
        -> (Reader, Arc<Observation>) {
        let observation = Arc::new(Observation::default());
        (Reader { resources: Arc::clone(host.resources()), observation: Arc::clone(&observation),
            action, stop }, observation)
    }
    fn planner(model: &ResidentInt4, task: GenerationJobTask)
        -> (Int4GenerationJobPlanner, SourceJobRequest, JobHostLimits) {
        let (budget, mut envelope, mut job, mut host) = fixture();
        let width = model.model_identity().group_width();
        let work = Int4Work::for_sequence(width, 0, 128, 16 * NANBEIGE_VOCAB_SIZE).unwrap();
        job.max_work = JobWork::from_int4(work, 0);
        envelope.native.native = Int4RunBudget::exact(work);
        host.preparation_reserve_bytes = PREPARATION;
        host.native.run.max_elapsed = Duration::from_secs(60);
        host.native.run.max_checkpoints = 100_000;
        let eos = EmbeddedTokenizer::pinned().unwrap().eos_token_id().unwrap();
        let planner = Int4GenerationJobPlanner::for_model(model.model_identity(), Int4GenerationJobConfig {
            task, generation: GenerationOptions::greedy(4, 256, eos), budget,
            planning: ChatLimits { max_messages: 4, max_message_bytes: 128, max_total_message_bytes: 256,
                generation: GenerationLimits { max_prompt_tokens: 48, max_new_tokens: 4,
                    max_output_bytes: 256, max_sampler_bytes: envelope.native.max_sampler_bytes } },
            native: envelope.native, context_tokens: 64,
        }, job).unwrap();
        let request = SourceJobRequest { root: PathBuf::from("/nonexistent/fnlp-text-job-team-refusal"),
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

    #[test]
    #[ignore = "controller-only real runtime: tiny weights test admission/ownership/drain, never model or durable-result parity"]
    fn durable_text_team_rejects_before_input_and_drains_cancelled_or_panicked_readers() {
        let host = NlpEngine::builder().resource_config(ResourceHostConfig {
            runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
            max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 2,
            helper_threads: 0, thread_ceiling: 4, memory_ceiling_bytes: 1024 * MIB,
            leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
        }).build().unwrap();
        let idle = host.resources().memory_snapshot();
        let lease = host.resources().acquire_lease();
        let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, PREPARATION).unwrap().commit().unwrap();
        let model_lease = host.resources().acquire_lease();
        let loaded = allocate(Pending::reserve(&model_lease, MemoryClass::Weights, MIB).unwrap(),
            || Ok(projection_fixture(4).0)).unwrap();
        let model = ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease: model_lease }) };
        let baseline = host.resources().memory_snapshot();
        use DecodeCancellationKind::*;
        for task in [GenerationJobTask::Generate, GenerationJobTask::Chat] {
            for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
                ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
                let (planner, request, limits) = planner(&model, task);
                let charge = limits.required_buffer_bytes(request.limits).unwrap();
                let stop = CancellationToken::default(); stop.cancel(cause);
                let (reader, observed) = reader(&host, ReadAction::Refuse, stop.clone());
                let error = host.job_int4_generation_with_threads(&model, planner, request, limits,
                    PackedInt4Kernel::scalar(), 3, reader, stop).err().unwrap();
                let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                    else { panic!("job pre-cancellation lost its typed cause"); };
                assert_eq!(stop.kind, cause);
                assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
                assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
                assert!(observed.input_bytes_at_drop.load(Ordering::SeqCst) >= PREPARATION + charge);
                drained(&host, &baseline);
            }
        }
        for width in [0, 1, 4, 65, usize::MAX] {
            let (planner, request, limits) = planner(&model, GenerationJobTask::Generate);
            let stop = CancellationToken::default();
            let (reader, observed) = reader(&host, ReadAction::Refuse, stop.clone());
            let error = host.job_int4_generation_with_threads(&model, planner, request, limits,
                PackedInt4Kernel::scalar(), width, reader, stop).err().unwrap();
            assert!(matches!(error, HostedJobError::Host(HostedError::Limits(_))));
            assert_eq!(observed.reads.load(Ordering::SeqCst), 0);
            assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
            drained(&host, &baseline);
        }
        for action in [ReadAction::Cancel(Deadline), ReadAction::Panic, ReadAction::Empty] {
            let (planner, request, limits) = planner(&model, GenerationJobTask::Chat);
            let charge = limits.required_buffer_bytes(request.limits).unwrap();
            let stop = CancellationToken::default();
            let (reader, observed) = reader(&host, action, stop.clone());
            let error = host.job_int4_generation_with_threads(&model, planner, request, limits,
                PackedInt4Kernel::scalar(), 3, reader, stop).err().unwrap();
            match action {
                ReadAction::Cancel(cause) => {
                    let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                        else { panic!("reader cancellation lost its typed cause"); };
                    assert_eq!(stop.kind, cause);
                }
                ReadAction::Panic => assert!(matches!(error,
                    HostedJobError::Host(HostedError::Panicked | HostedError::Scope { .. }))),
                ReadAction::Empty => assert!(matches!(error,
                    HostedJobError::Job(JobRunError::Storage(JobError::InvalidInput)))),
                ReadAction::Refuse => unreachable!(),
            }
            assert_eq!(observed.reads.load(Ordering::SeqCst), 1);
            assert_eq!(observed.drops.load(Ordering::SeqCst), 1);
            assert!(observed.worker_bytes_at_read.load(Ordering::SeqCst)
                >= int4_task_team_workspace_bytes(3, 4).unwrap());
            assert!(observed.input_bytes_at_drop.load(Ordering::SeqCst) >= PREPARATION + charge);
            drained(&host, &baseline);
        }
        drop(model); drop(preparation); drop(lease);
        drained(&host, &idle);
    }
}

#[test]
fn compact_encoder_workspace_coexists_with_complete_or_retained_input_and_native_storage() {
    let (task, envelope, job, host) = fixture();
    let workspace = crate::tokenizer::untrusted::SafeBpeLimits {
        max_input_bytes: 4096, max_output_tokens: 2048,
        max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
    }.requirements().unwrap().workspace_bytes;
    for input in [JobInputMemory::CompletePopulation, JobInputMemory::RetainedRecord] {
        let plain = requirements(WIDTH, task, envelope, job, host, input).unwrap();
        let selected = with_encoding_workspace(
            requirements(WIDTH, task, envelope, job, host, input).unwrap(), workspace).unwrap();
        assert_eq!(selected.input_bytes, plain.input_bytes + workspace);
        assert_eq!(selected.workspace_bytes, plain.workspace_bytes);
        assert_eq!(selected.native, plain.native);
        assert!(with_encoding_workspace(
            requirements(WIDTH, task, envelope, job, host, input).unwrap(), u64::MAX).is_err());
    }
}
