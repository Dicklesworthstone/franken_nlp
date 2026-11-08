//! Real runtime/ledger with test-only tiny projection weights. The callback
//! intentionally performs no model inference and cannot certify task parity.
use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    native_engine::{decode::DecodeCancellationKind,
        strict_int4::weights::projection_fixture}, tasks::judge::grouped_int4::Int4JudgeError};

const PREPARATION: u64 = 8192;
// Ordinary output_claim for a 2048-byte, 32-token complete result.
const OUTPUT: u64 = 2048 * 4 + 32 * 8 + 4096;

#[derive(Default)]
struct Observation { dropped: AtomicUsize, job_bytes_at_drop: AtomicU64 }
struct Tracked { resources: Arc<EngineResources>, observation: Arc<Observation> }
impl Drop for Tracked {
    fn drop(&mut self) {
        let snapshot = self.resources.memory_snapshot();
        let bytes = snapshot.by_class.get(&MemoryClass::JobBuffers)
            .map_or(0, |charge| charge.committed_bytes + charge.reserved_bytes);
        self.observation.job_bytes_at_drop.store(bytes, Ordering::SeqCst);
        self.observation.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
fn tracked(host: &NlpEngine) -> (Tracked, Arc<Observation>) {
    let observation = Arc::new(Observation::default());
    (Tracked { resources: Arc::clone(host.resources()), observation: Arc::clone(&observation) }, observation)
}
fn memory(native: NativeLimits) -> TaskMemory {
    let required = Int4MemoryRequirement::for_context(native.context_tokens, 4).unwrap();
    TaskMemory { required,
        scratch_bytes: required.rope_bytes + required.scratch_payload_bound + native.allocator_reserve_bytes,
        output_bytes: 2048, output_tokens: 32, preparation_bytes: Some(PREPARATION) }
}
fn drained(host: &NlpEngine, baseline: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, baseline);
    assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
}

#[test]
#[ignore = "requires controller-admitted real runtime; synthetic ownership fixture, not native task or model parity"]
fn complete_delivery_refusal_and_late_cancellation_keep_storage_charged_until_drop_and_join() {
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 2,
        helper_threads: 0, thread_ceiling: 4, memory_ceiling_bytes: 512 << 20,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let native = NativeLimits { context_tokens: 1, allocator_reserve_bytes: 65_536,
        run: RunLimits { max_elapsed: Duration::from_secs(60), max_checkpoints: 100_000,
            cleanup_reserve_bytes: 65_536 } };
    let (weights, _) = projection_fixture(4);
    let lease = host.resources().acquire_lease();
    let loaded = allocate(Pending::reserve(&lease, MemoryClass::Weights, 1 << 20).unwrap(),
        || Ok(weights)).unwrap();
    let model = ResidentInt4 { inner: Arc::new(ResidentModel { loaded, lease }) };
    let baseline = host.resources().memory_snapshot().committed_bytes;

    let (input, input_observed) = tracked(&host);
    let (output, output_observed) = tracked(&host);
    let result = execute_task(&host, &model, input, native, PackedInt4Kernel::scalar(), 3,
        memory(native), CancellationToken::default(), move |prepared, engine, _| {
            let snapshot = prepared.resources.memory_snapshot();
            assert_eq!(snapshot.by_class[&MemoryClass::JobBuffers].committed_bytes, PREPARATION);
            assert_eq!(snapshot.by_class[&MemoryClass::JobBuffers].reserved_bytes, OUTPUT);
            assert!(snapshot.by_class[&MemoryClass::ActivationScratch].committed_bytes
                >= int4_task_team_workspace_bytes(3, 4).unwrap());
            assert!(engine.kv_cache().all_slots_have_len(0));
            Ok(output)
        }).unwrap();
    assert_eq!(input_observed.dropped.load(Ordering::SeqCst), 1);
    assert!(input_observed.job_bytes_at_drop.load(Ordering::SeqCst) >= PREPARATION);
    assert_eq!(output_observed.dropped.load(Ordering::SeqCst), 0);
    assert_eq!(result.compute().compute_threads, 3);
    assert_eq!(result.compute().completed_projections, 0);
    assert!(Arc::ptr_eq(&result.result().resources, host.resources()));
    drained(&host, baseline + OUTPUT);
    drop(result);
    assert_eq!(output_observed.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(output_observed.job_bytes_at_drop.load(Ordering::SeqCst), OUTPUT);
    drained(&host, baseline);

    let (input, observed) = tracked(&host);
    let error = execute_task(&host, &model, input, native, PackedInt4Kernel::scalar(), 3,
        memory(native), CancellationToken::default(), |_, _, _| -> Result<(), HostedError> {
            Err(HostedError::Int4(Int4HostError::Judge(Int4JudgeError::Accounting)))
        }).err().unwrap();
    assert!(matches!(error, HostedError::Int4(Int4HostError::Judge(Int4JudgeError::Accounting))));
    assert_eq!(observed.dropped.load(Ordering::SeqCst), 1);
    assert!(observed.job_bytes_at_drop.load(Ordering::SeqCst) >= PREPARATION);
    drained(&host, baseline);

    let (input, input_observed) = tracked(&host);
    let (output, output_observed) = tracked(&host);
    let stop = CancellationToken::default(); let cancel = stop.clone();
    let error = execute_task(&host, &model, input, native, PackedInt4Kernel::scalar(), 3,
        memory(native), stop, move |_, _, _| {
            cancel.cancel(DecodeCancellationKind::Deadline); Ok(output)
        }).err().unwrap();
    assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind: DecodeCancellationKind::Deadline, .. }, .. }));
    assert_eq!(input_observed.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(output_observed.dropped.load(Ordering::SeqCst), 1);
    assert!(output_observed.job_bytes_at_drop.load(Ordering::SeqCst) >= OUTPUT);
    drained(&host, baseline);

    let (input, observed) = tracked(&host);
    let stop = CancellationToken::default(); stop.cancel(DecodeCancellationKind::ParentCancelled);
    let entered = Arc::new(AtomicBool::new(false)); let entry = Arc::clone(&entered);
    let error = execute_task(&host, &model, input, native, PackedInt4Kernel::scalar(), 3,
        memory(native), stop, move |_, _, _| {
            entry.store(true, Ordering::SeqCst); Ok(())
        }).err().unwrap();
    assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind: DecodeCancellationKind::ParentCancelled, .. }, .. }));
    assert!(!entered.load(Ordering::SeqCst));
    assert_eq!(observed.dropped.load(Ordering::SeqCst), 1);
    assert!(observed.job_bytes_at_drop.load(Ordering::SeqCst) >= PREPARATION);
    drained(&host, baseline);

    let mut excessive = memory(native); excessive.output_bytes = 512 << 20;
    let (input, observed) = tracked(&host);
    let error = execute_task(&host, &model, input, native, PackedInt4Kernel::scalar(), 3,
        excessive, CancellationToken::default(), |_, _, _| -> Result<(), HostedError> {
            panic!("over-budget result entered native work")
        }).err().unwrap();
    assert!(matches!(error, HostedError::Reservation(_)));
    assert_eq!(observed.dropped.load(Ordering::SeqCst), 1);
    assert!(observed.job_bytes_at_drop.load(Ordering::SeqCst) >= PREPARATION);
    drained(&host, baseline);

    let error = execute_task(&host, &model, (), native, PackedInt4Kernel::scalar(), 4,
        memory(native), CancellationToken::default(), |_, _, _| -> Result<(), HostedError> {
            panic!("unadmitted team entered native work")
        }).err().unwrap();
    assert!(matches!(error, HostedError::Limits(_)));
    drained(&host, baseline);
    drop(model);
    drained(&host, 0);
}
