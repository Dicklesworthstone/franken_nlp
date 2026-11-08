//! Controller-only admission, cancellation and ownership through the real
//! hosted team boundary. Tiny test weights never execute decoder inference.
use super::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    native_engine::{decode::{DecodeCancellationKind, DecodeTokenEvent}, generation::{GenerationLimits, GenerationOptions},
        kv::KV_BYTES_PER_TOKEN, strict_int4::weights::projection_fixture},
    tasks::{chat::{ChatLimits, GenerateRequest, grouped_int4::Int4ChatPlanner}, ir::TaskBudget}};

const MIB: u64 = 1 << 20;
const PREPARATION: u64 = 256 * MIB;

#[derive(Default)]
struct Observation { dropped: AtomicUsize, callbacks: AtomicUsize, bytes_at_drop: AtomicU64 }
struct Sink { resources: Arc<EngineResources>, observation: Arc<Observation> }
impl Drop for Sink {
    fn drop(&mut self) {
        let bytes = self.resources.memory_snapshot().by_class.get(&MemoryClass::JobBuffers)
            .map_or(0, |class| class.committed_bytes + class.reserved_bytes);
        self.observation.bytes_at_drop.store(bytes, Ordering::SeqCst);
        self.observation.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
impl DecodeEventSink for Sink {
    type Permit = (); type Error = &'static str;
    fn reserve(&mut self, _: &DecodeTokenEvent) -> Result<(), Self::Error> {
        self.observation.callbacks.fetch_add(1, Ordering::SeqCst);
        Err("refused stream reached a token callback")
    }
    fn permit(&mut self, _: (), _: DecodeTokenEvent) -> Result<(), Self::Error> {
        self.observation.callbacks.fetch_add(1, Ordering::SeqCst);
        Err("refused stream reached a token callback")
    }
}
fn sink(host: &NlpEngine) -> (Sink, Arc<Observation>) {
    let observation = Arc::new(Observation::default());
    (Sink { resources: Arc::clone(host.resources()), observation: Arc::clone(&observation) }, observation)
}
fn prepared(model: &ResidentInt4, limits: ChatStreamLimits) -> PreparedInt4Chat {
    let budget = TaskBudget { max_input_tokens: 128, max_output_tokens: 4, max_output_bytes: 65_536,
        max_grammar_states: 1, max_kv_bytes: limits.native.context_tokens as u64 * KV_BYTES_PER_TOKEN as u64 };
    let planner = Int4ChatPlanner::for_model(model.model_identity(), budget, ChatLimits {
        max_messages: 1, max_message_bytes: 128, max_total_message_bytes: 128,
        generation: GenerationLimits { max_prompt_tokens: 128, max_new_tokens: 4,
            max_output_bytes: 256, max_sampler_bytes: limits.max_sampler_bytes },
    }).unwrap();
    planner.plan_generate(&GenerateRequest { item_id: "refusal-fixture".to_owned(), sample_index: 0,
        prompt: "A".to_owned(), generation: GenerationOptions::greedy(4, 256, planner.eos_token_id()), budget }).unwrap()
}
fn drained(host: &NlpEngine, expected: &crate::MemorySnapshot) {
    assert_eq!(&host.resources().memory_snapshot(), expected);
    let closures = host.resources().outstanding_closure_snapshot();
    assert_eq!(closures.active_closures, 0);
    assert_eq!(closures.wrapper_cancelled_closures, 0);
}

#[test]
#[ignore = "run alone with controller-admitted real runtime; tiny test-only weights prove refusal/ownership, not decoder parity"]
fn every_pre_cancelled_stream_team_preserves_its_cause_and_drops_the_sink_under_its_charge() {
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 2,
        helper_threads: 0, thread_ceiling: 4, memory_ceiling_bytes: 1024 * MIB,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let idle = host.resources().memory_snapshot();
    let lease = host.resources().acquire_lease();
    // Pinned planning and all caller-owned private preparation are admitted
    // before construction. Each hosted call adds its own moved-input charge.
    let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, PREPARATION).unwrap().commit().unwrap();
    let model_lease = host.resources().acquire_lease();
    let loaded = allocate(Pending::reserve(&model_lease, MemoryClass::Weights, MIB).unwrap(),
        || Ok(projection_fixture(4).0)).unwrap();
    let model = ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease: model_lease }) };
    let baseline = host.resources().memory_snapshot();
    let limits = ChatStreamLimits {
        native: NativeLimits { context_tokens: 256, allocator_reserve_bytes: MIB,
            run: RunLimits { max_elapsed: Duration::from_secs(60), max_checkpoints: 100_000,
                cleanup_reserve_bytes: 65_536 } },
        max_sampler_bytes: 32 * MIB, preparation_reserve_bytes: PREPARATION, sink_reserve_bytes: MIB,
    };
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let (sink, observation) = sink(&host);
        let stop = CancellationToken::default(); stop.cancel(cause);
        let error = host.execute_int4_chat_stream_with_threads(&model, prepared(&model, limits), 1,
            limits, PackedInt4Kernel::scalar(), 3, sink, stop).err().unwrap();
        let HostedError::Stopped { stop, .. } = error else { panic!("pre-cancellation lost its typed cause"); };
        assert_eq!(stop.kind, cause);
        assert_eq!(observation.callbacks.load(Ordering::SeqCst), 0);
        assert_eq!(observation.dropped.load(Ordering::SeqCst), 1);
        assert!(observation.bytes_at_drop.load(Ordering::SeqCst) >= PREPARATION + retained_bytes(limits).unwrap());
        drained(&host, &baseline);
    }
    // Invalid widths, an installed-envelope overrun and the unsupported
    // combined schedule must fail before the task or sink can execute.
    for width in [0, 1, 4, 65, usize::MAX] {
        let (sink, observation) = sink(&host);
        let error = host.execute_int4_chat_stream_with_threads(&model, prepared(&model, limits), 1,
            limits, PackedInt4Kernel::scalar(), width, sink, CancellationToken::default()).err().unwrap();
        assert!(matches!(error, HostedError::Limits(_)));
        assert_eq!(observation.callbacks.load(Ordering::SeqCst), 0);
        assert_eq!(observation.dropped.load(Ordering::SeqCst), 1);
        drained(&host, &baseline);
    }
    let (sink, observation) = sink(&host);
    let prefill = Int4PrefillLimits { max_batch_rows: 2,
        max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(2).unwrap() };
    let error = host.run_int4_chat_stream(&model, prepared(&model, limits), 1, limits,
        Some(prefill), Some(3), PackedInt4Kernel::scalar(), sink, CancellationToken::default()).err().unwrap();
    assert!(matches!(error, HostedError::Limits(_)));
    assert_eq!(observation.callbacks.load(Ordering::SeqCst), 0);
    assert_eq!(observation.dropped.load(Ordering::SeqCst), 1);
    drained(&host, &baseline);
    for sink_bytes in [1024 * MIB, u64::MAX] {
        let (sink, observation) = self::sink(&host);
        let oversized = ChatStreamLimits { sink_reserve_bytes: sink_bytes, ..limits };
        let error = host.execute_int4_chat_stream_with_threads(&model, prepared(&model, limits), 1,
            oversized, PackedInt4Kernel::scalar(), 3, sink, CancellationToken::default()).err().unwrap();
        assert!(matches!(error, HostedError::Limits(_) | HostedError::Reservation(_)));
        assert_eq!(observation.callbacks.load(Ordering::SeqCst), 0);
        assert_eq!(observation.dropped.load(Ordering::SeqCst), 1);
        drained(&host, &baseline);
    }
    drop(model); drop(preparation); drop(lease);
    drained(&host, &idle);
}
