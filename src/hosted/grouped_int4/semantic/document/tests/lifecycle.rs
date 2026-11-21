//! Controller-only owned-runtime refusals. Tiny weights can exercise dispatch
//! and compilation failure; they cannot authorize a successful native result.
use super::*;
use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    native_engine::strict_int4::weights::projection_fixture};

fn drained(host: &NlpEngine, baseline: &crate::MemorySnapshot) {
    assert_eq!(&host.resources().memory_snapshot(), baseline);
    let closures = host.resources().outstanding_closure_snapshot();
    assert_eq!(closures.active_closures, 0);
    assert_eq!(closures.wrapper_cancelled_closures, 0);
}
fn execute(host: &NlpEngine, model: &ResidentInt4, threads: usize, request: ExtractionBatchArgs,
    options: Int4SemanticOptions, limits: Int4SemanticObjectDocumentHostLimits,
    stop: CancellationToken) -> Result<(), HostedError> {
    let source = "The private document contains the number 7.".to_owned();
    if threads == 1 {
        host.extract_int4_semantic_object_document(model, source, request, options,
            UntrustedEncoding::ByteFallback, limits, PackedInt4Kernel::scalar(), stop).map(drop)
    } else {
        host.extract_int4_semantic_object_document_with_threads(model, source, request, options,
            UntrustedEncoding::ByteFallback, limits, PackedInt4Kernel::scalar(), threads, stop).map(drop)
    }
}

#[test]
#[ignore = "controller-only fresh runtime: tiny weights prove admission, cancellation and physical drain, never inference"]
fn complete_object_host_drains_before_returning_cancelled_or_failed_preparations() {
    assert!(crate::installed_process_resources().is_none(), "requires a fresh runtime process");
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: 2 * 1024 * MIB,
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
            let (mut request, options, limits) = fixture();
            // A malformed private schema proves that queued cancellation must
            // win before parsing/preparing the caller's source and schema.
            request.schema = "{invalid-schema}".to_owned();
            let stop = CancellationToken::default(); stop.cancel(cause);
            assert!(matches!(execute(&host, &model, threads, request, options, limits, stop),
                Err(HostedError::Stopped { stop, .. }) if stop.kind == cause));
            drained(&host, &baseline);
        }
        let (mut request, options, limits) = fixture();
        request.schema = "{invalid-schema}".to_owned();
        assert!(matches!(execute(&host, &model, threads, request, options, limits, CancellationToken::default()),
            Err(HostedError::Int4(Int4HostError::ObjectExtractionMap(_)))));
        drained(&host, &baseline);

        let (mut request, options, mut limits) = fixture();
        request.schema = "{invalid-schema}".to_owned();
        limits.preparation_reserve_bytes = 3 * 1024 * MIB;
        // Complete admission precedes private compilation. A memory refusal
        // cannot become a schema error or leave partial reservations behind.
        assert!(matches!(execute(&host, &model, threads, request, options, limits, CancellationToken::default()),
            Err(HostedError::Reservation(_))));
        drained(&host, &baseline);
    }
    for checkpoints in [2, 3] {
        let (mut request, options, mut limits) = fixture();
        request.schema = "{invalid-schema}".to_owned();
        limits.native.run.max_checkpoints = checkpoints;
        assert!(matches!(execute(&host, &model, 4, request, options, limits, CancellationToken::default()),
            Err(HostedError::Stopped { stop, .. }) if stop.kind == PollQuota));
        drained(&host, &baseline);
    }
    drop(model);
    drained(&host, &idle);
}
