//! Controller-only stream protocol and lifetime gate with an independently
//! pinned real artifact. No synthetic model output or implicit model skip.
use super::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use crate::{canonjson, LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    native_engine::{decode::{DecodeCancellationKind, DecodeTokenEvent},
        generation::{GenerationLimits, GenerationOptions, GenerationSampling},
        simd::{config::configure_process_kernel, Int8KernelKind}, strict_int8::STRICT_INT8_EXECUTION},
    tasks::chat::{ChatLimits, ChatMessage, ChatRequest, ChatRole, GenerateRequest, quantized::Int8ChatPlanner},
    tokenizer::{pinned_controls, untrusted::{SafeBpeLimits, UntrustedEncoding}}};

const MIB: u64 = 1 << 20;
#[derive(Clone, Copy)]
enum Behavior { Collect, RefuseReserve, RefusePermit, Cancel, Panic }
#[derive(Default)]
struct Observation { dropped: AtomicUsize, callbacks: AtomicUsize, job_bytes_at_drop: AtomicU64 }
struct Sink {
    resources: Arc<EngineResources>, observation: Arc<Observation>,
    cancellation: CancellationToken, behavior: Behavior, events: Vec<DecodeTokenEvent>,
}
impl Drop for Sink {
    fn drop(&mut self) {
        let bytes = self.resources.memory_snapshot().by_class.get(&MemoryClass::JobBuffers)
            .map_or(0, |class| class.committed_bytes + class.reserved_bytes);
        self.observation.job_bytes_at_drop.store(bytes, Ordering::SeqCst);
        self.observation.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
impl DecodeEventSink for Sink {
    type Permit = usize; type Error = &'static str;
    fn reserve(&mut self, event: &DecodeTokenEvent) -> Result<usize, Self::Error> {
        self.observation.callbacks.fetch_add(1, Ordering::SeqCst);
        if matches!(self.behavior, Behavior::RefuseReserve) || self.events.len() >= 8 { return Err("stream reserve refused"); }
        Ok(event.token_index)
    }
    fn permit(&mut self, index: usize, event: DecodeTokenEvent) -> Result<(), Self::Error> {
        self.observation.callbacks.fetch_add(1, Ordering::SeqCst);
        assert_eq!(index, event.token_index); assert_eq!(index, self.events.len());
        if matches!(self.behavior, Behavior::RefusePermit) { return Err("stream permit refused"); }
        assert!(!matches!(self.behavior, Behavior::Panic), "injected stream sink panic");
        self.events.push(event);
        if matches!(self.behavior, Behavior::Cancel) { self.cancellation.cancel(DecodeCancellationKind::User); }
        Ok(())
    }
}
fn sink(host: &NlpEngine, behavior: Behavior, cancellation: CancellationToken) -> (Sink, Arc<Observation>) {
    let observation = Arc::new(Observation::default());
    (Sink { resources: Arc::clone(host.resources()), observation: Arc::clone(&observation),
        cancellation, behavior, events: Vec::new() }, observation)
}
fn limits() -> ChatStreamLimits {
    ChatStreamLimits { native: NativeLimits { context_tokens: 512, allocator_reserve_bytes: 64 * MIB,
        run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000_000,
            cleanup_reserve_bytes: 65_536 } }, max_sampler_bytes: 32 * MIB,
        preparation_reserve_bytes: 256 * MIB, sink_reserve_bytes: MIB }
}
fn task_budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 504, max_output_tokens: 8, max_output_bytes: 65_536,
        max_grammar_states: 1, max_kv_bytes: requirements(limits().native).unwrap().kv_bytes }
}
fn prefill() -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: 3,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(3).unwrap() }
}
fn prepared(facts: &ArtifactIdentity, chat: bool) -> PreparedInt8Chat {
    let absent = Sha256Digest::of_bytes(b"null");
    let packing = canonjson::canonical_bytes(&("current-candidate-portable-int8-v1", Vec::<String>::new())).unwrap();
    let identity = ExecutionIdentity { schema_version: 1, source_revision: facts.revision.clone(),
        logical_model_digest: Sha256Digest::from_hex(&facts.logical_model_sha256).unwrap(),
        artifact_format: "fnlpq-current-candidate-v1".to_owned(), quant_recipe: facts.recipe_id.clone(),
        packing_set_digest: Sha256Digest::of_bytes(&packing), tokenizer_digest: absent,
        template_digest: absent, task_spec: "uncompiled-stream".to_owned(), taskir_digest: absent,
        prompt_digest: absent, grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None };
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    let planner = Int8ChatPlanner::pinned(controls, eos, identity, task_budget(), ChatLimits {
        max_messages: 2, max_message_bytes: 256, max_total_message_bytes: 512,
        generation: GenerationLimits { max_prompt_tokens: 504, max_new_tokens: 8,
            max_output_bytes: 1024, max_sampler_bytes: limits().max_sampler_bytes },
    }).unwrap();
    let mut generation = GenerationOptions::greedy(8, 1024, eos);
    generation.capture_logprobs = true;
    if chat {
        let planner = planner.with_document_encoding(UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: 512, max_output_tokens: 504, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
        })).unwrap();
        generation.sampling = GenerationSampling::Seeded { effective_seed: [7; 32],
            temperature_milli: 600, top_k: Some(20), top_p_ppm: 950_000 };
        planner.plan_chat(&ChatRequest { item_id: "stream-team".to_owned(), sample_index: 0,
            messages: vec![ChatMessage { role: ChatRole::System, content: "Reply in English.".to_owned() },
                ChatMessage { role: ChatRole::User, content: "Say hello.".to_owned() }],
            generation, budget: task_budget() }).unwrap()
    } else {
        planner.plan_generate(&GenerateRequest { item_id: "stream-team".to_owned(), sample_index: 0,
            prompt: "Say hello in English.".to_owned(), generation, budget: task_budget() }).unwrap()
    }
}
fn drained(host: &NlpEngine, baseline: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, baseline); assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.recorded_leaks, 0); assert_eq!(memory.outstanding_obligations, 0);
    let closures = host.resources().outstanding_closure_snapshot();
    assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
}

#[test]
#[ignore = "controller-only real INT8 artifact and fresh resource-host process; stream parity/lifetime evidence pending execution"]
fn streamed_events_results_and_ownership_survive_fixed_teams_and_prompt_morsels() {
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_STREAM_TEAM_MODEL").expect("real admitted INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_STREAM_TEAM_LOGICAL_SHA256").expect("independent logical model pin required");
    let source_root = std::env::var("FNLP_INT8_STREAM_TEAM_SOURCE_ROOT_SHA256").expect("independent source root pin required");
    Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source_root).unwrap();
    configure_process_kernel(Int8KernelKind::Scalar).unwrap();
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: 16_u64 << 30,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let lease = host.resources().acquire_lease();
    let caller = Pending::reserve(&lease, MemoryClass::JobBuffers, limits().preparation_reserve_bytes).unwrap().commit().unwrap();
    let model = host.load_current_candidate_int8(path, LoadLimits {
        artifact: ArtifactLoadBudget::streaming_only(6_u64 << 30, 64 << 10),
        tokenizer_and_metadata_bytes: 128 * MIB, allocator_reserve_bytes: 64 * MIB,
        run: limits().native.run,
    }, CancellationToken::default()).unwrap();
    assert_eq!(model.artifact_identity().logical_model_sha256, logical);
    assert_eq!(model.artifact_identity().source_root_sha256, source_root);
    let baseline = host.resources().memory_snapshot().committed_bytes;
    let retained = limits().retained_bytes().unwrap();
    let result_bytes = task_budget().max_output_bytes * 4 + u64::from(task_budget().max_output_tokens) * 8 + 4096;
    for chat in [false, true] {
        let (writer, _) = sink(&host, Behavior::Collect, CancellationToken::default());
        let mut serial = host.execute_int8_chat_stream(&model, prepared(model.artifact_identity(), chat),
            1, limits(), writer, CancellationToken::default()).unwrap();
        assert!(serial.compute().is_none());
        let expected = canonjson::canonical_bytes(serial.result()).unwrap();
        let work = serial.result().model_work;
        let events = serial.with_sink(|result, writer| {
            assert_eq!(writer.events.iter().map(|event| event.token_id).collect::<Vec<_>>(), result.result.token_ids);
            assert_eq!(writer.events.iter().flat_map(|event| event.decoded_bytes.iter().copied()).collect::<Vec<_>>(),
                result.result.content.as_bytes());
            writer.events.clone()
        });
        drop(serial); drained(&host, baseline);
        for packed in [false, true] {
            let (writer, observation) = sink(&host, Behavior::Collect, CancellationToken::default());
            let plan = prepared(model.artifact_identity(), chat);
            let mut completed = if packed {
                host.execute_int8_chat_stream_layer_major_parallel(&model, plan, 1, limits(), prefill(), 4,
                    writer, CancellationToken::default())
            } else {
                host.execute_int8_chat_stream_parallel(&model, plan, 1, limits(), 2,
                    writer, CancellationToken::default())
            }.unwrap();
            assert_eq!(canonjson::canonical_bytes(completed.result()).unwrap(), expected);
            assert_eq!(completed.result().model_work, work);
            let receipt = completed.compute().unwrap();
            assert_eq!(receipt.compute_threads, if packed { 4 } else { 2 });
            assert_eq!(receipt.completed_rows, work.projections.dot_products);
            assert!(receipt.completed_worker_rows > 0);
            drained(&host, baseline + retained + result_bytes);
            completed.with_sink(|_, writer| assert_eq!(writer.events, events));
            assert_eq!(observation.dropped.load(Ordering::SeqCst), 0);
            drop(completed);
            assert_eq!(observation.dropped.load(Ordering::SeqCst), 1);
            assert!(observation.job_bytes_at_drop.load(Ordering::SeqCst) >= retained);
            drained(&host, baseline);
        }
    }
    // These callbacks happen after real native projections, not a fabricated
    // completion. None may publish success or leave a child/charge behind.
    for behavior in [Behavior::RefuseReserve, Behavior::RefusePermit, Behavior::Cancel, Behavior::Panic] {
        let cancellation = CancellationToken::default();
        let (writer, observation) = sink(&host, behavior, cancellation.clone());
        let error = host.execute_int8_chat_stream_layer_major_parallel(&model,
            prepared(model.artifact_identity(), false), 1, limits(), prefill(), 3,
            writer, cancellation).err().expect("failed sink cannot produce stream completion");
        match behavior {
            Behavior::Cancel => assert!(matches!(error, HostedError::Stopped {
                stop: RunStop { kind: DecodeCancellationKind::User, .. }, .. })),
            Behavior::Panic => assert!(matches!(error, HostedError::Panicked)),
            _ => assert!(matches!(error, HostedError::Chat(_))),
        }
        assert!(observation.callbacks.load(Ordering::SeqCst) > 0);
        assert_eq!(observation.dropped.load(Ordering::SeqCst), 1);
        assert!(observation.job_bytes_at_drop.load(Ordering::SeqCst) >= retained);
        drained(&host, baseline);
    }
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let cancellation = CancellationToken::default(); cancellation.cancel(cause);
        let (writer, observation) = sink(&host, Behavior::Collect, cancellation.clone());
        let error = host.execute_int8_chat_stream_layer_major_parallel(&model,
            prepared(model.artifact_identity(), false), 1, limits(), prefill(), 3,
            writer, cancellation).err().expect("pre-cancelled request cannot reach sink");
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
        assert_eq!(observation.callbacks.load(Ordering::SeqCst), 0);
        assert_eq!(observation.dropped.load(Ordering::SeqCst), 1);
        assert!(observation.job_bytes_at_drop.load(Ordering::SeqCst) >= retained);
        drained(&host, baseline);
    }
    drop(model); drop(caller); drop(lease); drained(&host, 0);
}
