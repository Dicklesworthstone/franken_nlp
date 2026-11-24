//! Controller-only complete-source differential. Requires an explicitly
//! supplied model and independent logical/source pins; never downloads weights.
use super::*;
use crate::{canonjson, LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    native_engine::simd::config::process_kernel};

fn drained(host: &NlpEngine, baseline: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, baseline);
    assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
}

fn input(facts: &ArtifactIdentity, compact: bool) -> MapInput {
    let (request, mut config) = fixture();
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
#[ignore = "controller-only real INT8 artifact, independent digests and fresh host; unexecuted parity gate"]
fn complete_faithfulness_teams_preserve_claim_evidence_seals_work_and_physical_drain() {
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_FAITHFULNESS_MAP_TEAM_MODEL")
        .expect("explicit INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_FAITHFULNESS_MAP_TEAM_LOGICAL_SHA256")
        .expect("independent logical pin required");
    let source_root = std::env::var("FNLP_INT8_FAITHFULNESS_MAP_TEAM_SOURCE_ROOT_SHA256")
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
    let baseline = host.resources().memory_snapshot().committed_bytes;

    for compact in [false, true] {
        let fixture = input(model.artifact_identity(), compact);
        let (prepared, admitted) = prepare_map(&fixture, model.artifact_identity(), &mut Continue).unwrap();
        assert!(prepared.chunk_count() > 1);
        assert!(prepared.head_count() > prepared.chunk_count());
        let expected = prepared.preflight_metadata();
        let work = prepared.planned_work();
        drop(admitted); drop(prepared);
        let quote = fixture.config.requirements_with_encoding(task(&fixture.request), &fixture.encoding).unwrap();
        let serial = host.map_int8_faithfulness_with_encoding(&model, fixture.request.clone(), fixture.identity.clone(),
            fixture.config, fixture.encoding.clone(), CancellationToken::default()).unwrap();
        expected.verify_completed_with_control(serial.result(), &mut Continue).unwrap();
        let expected_bytes = canonjson::canonical_bytes(serial.result()).unwrap();
        assert_eq!(serial.result().model_work, work);
        drop(serial); drained(&host, baseline);

        for rows in [1, 3, 4] {
            let grouped = host.map_int8_faithfulness_layer_major_with_encoding(&model, fixture.request.clone(),
                fixture.identity.clone(), fixture.config, fixture.encoding.clone(), prefill(rows),
                CancellationToken::default()).unwrap();
            assert_eq!(canonjson::canonical_bytes(grouped.result()).unwrap(), expected_bytes);
            expected.verify_completed_with_control(grouped.result(), &mut Continue).unwrap();
            drop(grouped); drained(&host, baseline);
        }
        for width in [2, 4] {
            for schedule in [None, Some(prefill(3)), Some(prefill(4))] {
                let result = match schedule {
                    None => host.map_int8_faithfulness_with_threads(&model, fixture.request.clone(),
                        fixture.identity.clone(), fixture.config, fixture.encoding.clone(), process_kernel(), width,
                        CancellationToken::default()),
                    Some(prefill) => host.map_int8_faithfulness_layer_major_with_threads(&model, fixture.request.clone(),
                        fixture.identity.clone(), fixture.config, fixture.encoding.clone(), process_kernel(), width,
                        prefill, CancellationToken::default()),
                }.unwrap();
                expected.verify_completed_with_control(result.result(), &mut Continue).unwrap();
                assert_eq!(canonjson::canonical_bytes(result.result()).unwrap(), expected_bytes);
                assert_eq!(result.result().model_work, work);
                assert_eq!(result.compute().compute_threads, width);
                assert_eq!(result.compute().completed_rows, work.projections.dot_products);
                assert!(result.compute().completed_worker_rows > 0);
                drained(&host, baseline + quote.output_reservation_bytes);
                drop(result); drained(&host, baseline);
            }
        }
        for axis in 0..5 {
            let mut short = fixture.config; short.mapping.max_model_work = work;
            match axis {
                0 => short.mapping.max_model_work.forward_positions -= 1,
                1 => short.mapping.max_model_work.projected_logits -= 1,
                2 => short.mapping.max_model_work.attention_pairs -= 1,
                3 => short.mapping.max_model_work.projections.dot_products -= 1,
                _ => short.mapping.max_model_work.projections.multiply_accumulates -= 1,
            }
            assert!(host.map_int8_faithfulness_layer_major_with_threads(&model, fixture.request.clone(),
                fixture.identity.clone(), short, fixture.encoding.clone(), process_kernel(), 4,
                prefill(4), CancellationToken::default()).is_err());
            drained(&host, baseline);
        }
    }
    let fixture = input(model.artifact_identity(), true);
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let cancellation = CancellationToken::default(); cancellation.cancel(cause);
        let error = host.map_int8_faithfulness_layer_major_with_threads(&model, fixture.request.clone(),
            fixture.identity.clone(), fixture.config, fixture.encoding.clone(), process_kernel(), 4,
            prefill(4), cancellation).err().expect("cancelled document cannot publish success");
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
        drained(&host, baseline);
    }
    drop(fixture); drop(model); drop(caller_preparation); drop(lease); drained(&host, 0);
}
