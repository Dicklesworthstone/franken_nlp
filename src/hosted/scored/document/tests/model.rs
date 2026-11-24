//! Controller-only complete-document differential with independently pinned
//! FNLP_INT8_CLASSIFY_MAP_TEAM_MODEL, *_LOGICAL_SHA256 and *_SOURCE_ROOT_SHA256.
//! This ignored gate never fetches weights or promotes unexecuted parity.
use super::*;
use crate::{canonjson, LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    native_engine::simd::config::process_kernel};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}

fn drained(host: &NlpEngine, baseline: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, baseline);
    assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
}

fn input(facts: &ArtifactIdentity, mode: ClassificationMode, compact: bool) -> MapInput {
    let (mut request, mut config) = fixture();
    request.document = "Helpful service. é\r\n<|im_start|> Shanghai. ".repeat(3);
    request.mode = mode;
    config.mapping.chunks.max_chunk_bytes = 48;
    config.mapping.max_model_work = Int8Work::for_sequence(0, 16384, 256 * NANBEIGE_VOCAB_SIZE).unwrap();
    config.native.allocator_reserve_bytes = 64 << 20;
    config.native.run.max_elapsed = Duration::from_secs(3600);
    config.native.run.max_checkpoints = 1_000_000_000;
    let encoding = if compact { UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 262144, max_output_tokens: 131072,
        max_pair_checks: 33554432, max_pair_bytes: 268435456,
    }) } else { UntrustedEncoding::ByteFallback };
    let mut identity = identity(&planner(encoding.clone()));
    identity.source_revision = facts.revision.clone();
    identity.logical_model_digest = Sha256Digest::from_hex(&facts.logical_model_sha256).unwrap();
    identity.quant_recipe = facts.recipe_id.clone();
    let packing = canonjson::canonical_bytes(&("current-candidate-portable-int8-v1", Vec::<String>::new())).unwrap();
    identity.packing_set_digest = Sha256Digest::of_bytes(&packing);
    MapInput { request, config, identity, encoding }
}

#[test]
#[ignore = "controller-only real INT8 artifact, independent digests and fresh host; no parity claim until executed"]
fn complete_classification_teams_preserve_every_chunk_work_seal_and_physical_drain() {
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_CLASSIFY_MAP_TEAM_MODEL")
        .expect("admitted INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_CLASSIFY_MAP_TEAM_LOGICAL_SHA256")
        .expect("independent logical pin required");
    let source_root = std::env::var("FNLP_INT8_CLASSIFY_MAP_TEAM_SOURCE_ROOT_SHA256")
        .expect("independent source pin required");
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
    let model = host.load_current_candidate_int8(path, LoadLimits {
        artifact: ArtifactLoadBudget::streaming_only(6_u64 << 30, 64 << 10),
        tokenizer_and_metadata_bytes: 128 << 20, allocator_reserve_bytes: 64 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000_000,
            cleanup_reserve_bytes: 65_536 },
    }, CancellationToken::default()).unwrap();
    assert_eq!(model.artifact_identity().logical_model_sha256, logical);
    assert_eq!(model.artifact_identity().source_root_sha256, source_root);
    assert_eq!(model.artifact_identity().revision, "f56ec5a9650268aa098496734743c25ea778bd2d");
    let baseline = host.resources().memory_snapshot().committed_bytes;

    for (mode, compact) in [(ClassificationMode::Exclusive, false), (ClassificationMode::MultiLabel, true)] {
        let fixture = input(model.artifact_identity(), mode, compact);
        let (prepared, admitted) = prepare_map(&fixture, model.artifact_identity(), &mut Continue).unwrap();
        assert!(prepared.chunk_count() > 1);
        let expected = prepared.preflight_metadata();
        let work = prepared.planned_work();
        drop(admitted); drop(prepared);
        let quote = fixture.config.requirements_with_encoding(fixture.request.budget, &fixture.encoding).unwrap();
        let serial = host.map_int8_classify_with_encoding(&model, fixture.request.clone(), fixture.identity.clone(),
            fixture.config, fixture.encoding.clone(), CancellationToken::default()).unwrap();
        expected.verify_completed_with_control(serial.result(), &mut Continue).unwrap();
        let expected_bytes = canonjson::canonical_bytes(serial.result()).unwrap();
        assert_eq!(serial.result().model_work, work);
        drop(serial); drained(&host, baseline);

        let grouped = host.map_int8_classify_layer_major_with_encoding(&model, fixture.request.clone(),
            fixture.identity.clone(), fixture.config, fixture.encoding.clone(), prefill(4),
            CancellationToken::default()).unwrap();
        expected.verify_completed_with_control(grouped.result(), &mut Continue).unwrap();
        assert_eq!(canonjson::canonical_bytes(grouped.result()).unwrap(), expected_bytes);
        drained(&host, baseline + quote.output_reservation_bytes);
        drop(grouped); drained(&host, baseline);

        for width in [2, 4] {
            for schedule in [None, Some(prefill(4))] {
                let result = match schedule {
                    None => host.map_int8_classify_with_threads(&model, fixture.request.clone(),
                        fixture.identity.clone(), fixture.config, fixture.encoding.clone(), process_kernel(), width,
                        CancellationToken::default()),
                    Some(prefill) => host.map_int8_classify_layer_major_with_threads(&model, fixture.request.clone(),
                        fixture.identity.clone(), fixture.config, fixture.encoding.clone(), process_kernel(), width,
                        prefill, CancellationToken::default()),
                }.unwrap();
                expected.verify_completed_with_control(result.result(), &mut Continue).unwrap();
                assert_eq!(canonjson::canonical_bytes(result.result()).unwrap(), expected_bytes);
                assert_eq!(result.result().model_work, work);
                assert_eq!(result.compute().compute_threads, width);
                assert_eq!(result.compute().completed_rows, work.projections.dot_products);
                assert!(result.compute().completed_projections > 0);
                assert!(result.compute().completed_worker_rows > 0);
                // Only the complete output guard survives physical join.
                drained(&host, baseline + quote.output_reservation_bytes);
                drop(result); drained(&host, baseline);
            }
        }
        for axis in 0..5 {
            let mut underfunded = fixture.config;
            underfunded.mapping.max_model_work = work;
            match axis {
                0 => underfunded.mapping.max_model_work.forward_positions -= 1,
                1 => underfunded.mapping.max_model_work.projected_logits -= 1,
                2 => underfunded.mapping.max_model_work.attention_pairs -= 1,
                3 => underfunded.mapping.max_model_work.projections.dot_products -= 1,
                _ => underfunded.mapping.max_model_work.projections.multiply_accumulates -= 1,
            }
            assert!(host.map_int8_classify_layer_major_with_threads(&model, fixture.request.clone(),
                fixture.identity.clone(), underfunded, fixture.encoding.clone(), process_kernel(), 4,
                prefill(4), CancellationToken::default()).is_err());
            drained(&host, baseline);
        }
    }
    let fixture = input(model.artifact_identity(), ClassificationMode::MultiLabel, true);
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let cancellation = CancellationToken::default(); cancellation.cancel(cause);
        let error = host.map_int8_classify_layer_major_with_threads(&model, fixture.request.clone(),
            fixture.identity.clone(), fixture.config, fixture.encoding.clone(), process_kernel(), 4,
            prefill(4), cancellation).err().expect("pre-cancelled maps must not publish");
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
        drained(&host, baseline);
    }
    for width in [0, 1, 5, 65, usize::MAX] {
        assert!(matches!(host.map_int8_classify_with_threads(&model, fixture.request.clone(),
            fixture.identity.clone(), fixture.config, fixture.encoding.clone(), process_kernel(), width,
            CancellationToken::default()), Err(HostedError::Limits(_))));
        drained(&host, baseline);
    }
    let mut foreign = fixture.identity.clone(); foreign.logical_model_digest = Sha256Digest::of_bytes(b"foreign-model");
    assert!(matches!(host.map_int8_classify_with_threads(&model, fixture.request.clone(), foreign,
        fixture.config, fixture.encoding.clone(), process_kernel(), 2, CancellationToken::default()),
        Err(HostedError::ModelIdentity)));
    drained(&host, baseline);
    drop(fixture); drop(model); drop(caller_preparation); drop(lease); drained(&host, 0);
}
