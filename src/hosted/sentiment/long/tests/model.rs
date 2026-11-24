//! Explicit controller-only differential against a local current-candidate
//! artifact. Independent logical/source digests are required; no model fetch.
use super::*;
use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    native_engine::simd::config::process_kernel};

fn drained(host: &NlpEngine, committed: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, committed);
    assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
}

fn compare(host: &NlpEngine, model: &ResidentInt8, baseline: u64, encoding: UntrustedEncoding) {
    let input = planned_input(model.artifact_identity(), encoding);
    let required = input.config.requirements(input.request.budget, input.request.axes.len(),
        input.planner.document_encoding()).unwrap();
    let (prepared, identities) = prepare_map(&input, model.artifact_identity(), &mut Continue).unwrap();
    assert!(prepared.chunk_count() > 1);
    let expected = prepared.preflight_metadata();
    let work = prepared.planned_work();
    let identity_bytes = crate::canonjson::canonical_bytes(&identities).unwrap();
    drop(identities);
    drop(prepared);
    let ordinary = host.map_int8_sentiment(model, input.request.clone(),
        Arc::clone(&input.planner), input.config.clone(), CancellationToken::default()).unwrap();
    expected.verify_completed(ordinary.result()).unwrap();
    let bytes = crate::canonjson::canonical_bytes(ordinary.result()).unwrap();
    assert_eq!(ordinary.result().model_work, work);
    drained(host, baseline + required.output_reservation_bytes);
    drop(ordinary);
    drained(host, baseline);
    for width in [2, 4] {
        for schedule in [None, Some(prefill(7))] {
            let output = match schedule {
                None => host.map_int8_sentiment_with_threads(model, input.request.clone(),
                    Arc::clone(&input.planner), input.config.clone(), process_kernel(),
                    width, CancellationToken::default()),
                Some(prefill) => host.map_int8_sentiment_layer_major_with_threads(model,
                    input.request.clone(), Arc::clone(&input.planner), input.config.clone(),
                    process_kernel(), width, prefill, CancellationToken::default()),
            }.unwrap();
            expected.verify_completed(output.result()).unwrap();
            assert_eq!(crate::canonjson::canonical_bytes(output.result()).unwrap(), bytes);
            assert_eq!(output.result().model_work, work);
            assert_eq!(output.compute().compute_threads, width);
            assert_eq!(output.compute().completed_rows, work.projections.dot_products);
            assert!(output.compute().completed_worker_rows > 0);
            assert!(output.compute().completed_projections > 0);
            // Only the protected output remains charged when callers can
            // observe it; workers, KV, prompts and transferred input drained.
            drained(host, baseline + required.output_reservation_bytes);
            drop(output);
            drained(host, baseline);
            let (again, admitted) = prepare_map(&input, model.artifact_identity(), &mut Continue).unwrap();
            assert_eq!(crate::canonjson::canonical_bytes(&admitted).unwrap(), identity_bytes);
            assert_eq!(again.planned_work(), work);
        }
    }
}

#[test]
#[ignore = "controller-only pinned INT8 artifact and fresh host; model parity remains unqualified until run"]
fn document_teams_preserve_complete_outputs_and_release_every_owner_on_failure() {
    let path = std::path::PathBuf::from(std::env::var_os("FNLP_INT8_SENTIMENT_DOCUMENT_MODEL")
        .expect("admitted local INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_SENTIMENT_DOCUMENT_LOGICAL_SHA256")
        .expect("independent logical digest required");
    let source_root = std::env::var("FNLP_INT8_SENTIMENT_DOCUMENT_SOURCE_ROOT_SHA256")
        .expect("independent source digest required");
    Sha256Digest::from_hex(&logical).unwrap();
    Sha256Digest::from_hex(&source_root).unwrap();
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: 16_u64 << 30,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let lease = host.resources().acquire_lease();
    let caller_preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, 512 << 20)
        .unwrap().commit().unwrap();
    let mut native = fixture().0.native;
    native.run.max_elapsed = Duration::from_secs(3600);
    native.run.max_checkpoints = 1_000_000_000;
    let model = host.load_current_candidate_int8(path, LoadLimits {
        artifact: ArtifactLoadBudget::streaming_only(6_u64 << 30, 64 << 10),
        tokenizer_and_metadata_bytes: 128 << 20, allocator_reserve_bytes: 64 << 20,
        run: native.run,
    }, CancellationToken::default()).unwrap();
    assert_eq!(model.artifact_identity().logical_model_sha256, logical);
    assert_eq!(model.artifact_identity().source_root_sha256, source_root);
    assert_eq!(model.artifact_identity().revision, "f56ec5a9650268aa098496734743c25ea778bd2d");
    let baseline = host.resources().memory_snapshot().committed_bytes;
    for encoding in [UntrustedEncoding::ByteFallback, compact_encoding()] {
        compare(&host, &model, baseline, encoding);
    }
    let input = planned_input(model.artifact_identity(), compact_encoding());
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let cancellation = CancellationToken::default();
        cancellation.cancel(cause);
        let error = host.map_int8_sentiment_layer_major_with_threads(&model, input.request.clone(),
            Arc::clone(&input.planner), input.config.clone(), process_kernel(), 4, prefill(7),
            cancellation).err().expect("cancelled document must not publish");
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
        drained(&host, baseline);
    }
    for width in [0, 1, 5, 65, usize::MAX] {
        let error = host.map_int8_sentiment_with_threads(&model, input.request.clone(),
            Arc::clone(&input.planner), input.config.clone(), process_kernel(), width,
            CancellationToken::default()).err().expect("unadmitted width must refuse");
        assert!(matches!(error, HostedError::Limits(_)));
        drained(&host, baseline);
    }
    let mut foreign = input.config.clone();
    foreign.identity.logical_model_digest = Sha256Digest::of_bytes(b"foreign artifact");
    assert!(matches!(host.map_int8_sentiment_with_threads(&model, input.request.clone(),
        Arc::clone(&input.planner), foreign, process_kernel(), 2, CancellationToken::default()),
        Err(HostedError::ModelIdentity)));
    drained(&host, baseline);
    let mut short = input.config.clone();
    short.mapping.max_retained_result_bytes = 1;
    let error = host.map_int8_sentiment_layer_major_with_threads(&model, input.request,
        input.planner, short, process_kernel(), 4, prefill(7), CancellationToken::default())
        .err().expect("completed first chunk must exceed its retained frontier");
    assert!(matches!(error, HostedError::SentimentMap(Int8SentimentMapError::Output)));
    drained(&host, baseline);
    drop(model);
    drop(caller_preparation);
    drop(lease);
    drained(&host, 0);
}
