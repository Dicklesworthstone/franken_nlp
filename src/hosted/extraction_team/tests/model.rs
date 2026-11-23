//! Controller-only real-artifact gate in a fresh process. These tests do not
//! create synthetic decoder results and are not runtime or parity evidence
//! until executed with independently pinned artifact identity values.
use super::*;
use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    native_engine::{constrained_int8::Int8JsonError, simd::config::process_kernel}};

fn drained(host: &NlpEngine, baseline: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, baseline); assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
}
fn compare(result: &Int8ExtractRun, expected: &[u8], work: Int8Work) {
    assert_eq!(canonjson::canonical_bytes(result).unwrap(), expected);
    assert_eq!(result.model_work, work);
}

#[test]
#[ignore = "controller-only real INT8 artifact and fresh resource-host process; no parity claim until executed"]
fn exact_schema_and_source_results_survive_serial_and_team_prompt_schedules() {
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_EXTRACT_TEAM_MODEL").expect("admitted INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_EXTRACT_TEAM_LOGICAL_SHA256").expect("independent logical model pin required");
    let source_root = std::env::var("FNLP_INT8_EXTRACT_TEAM_SOURCE_ROOT_SHA256").expect("independent source root pin required");
    Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source_root).unwrap();
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: 16_u64 << 30,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    // Caller preparation precedes vocabulary/planner/comparison staging and
    // remains distinct from every new hosted route's transferred-input charge.
    let lease = host.resources().acquire_lease();
    let caller_preparation = Pending::reserve(&lease, MemoryClass::JobBuffers,
        limits().preparation_reserve_bytes).unwrap().commit().unwrap();
    let model = host.load_current_candidate_int8(path, LoadLimits {
        artifact: ArtifactLoadBudget::streaming_only(6_u64 << 30, 64 << 10),
        tokenizer_and_metadata_bytes: 128 << 20, allocator_reserve_bytes: 64 << 20,
        run: limits().native.run,
    }, CancellationToken::default()).unwrap();
    assert_eq!(model.artifact_identity().logical_model_sha256, logical);
    assert_eq!(model.artifact_identity().source_root_sha256, source_root);
    assert_eq!(model.artifact_identity().revision, "f56ec5a9650268aa098496734743c25ea778bd2d");
    let vocabulary = vocabulary();
    let baseline = host.resources().memory_snapshot().committed_bytes;
    let output = task_budget().max_output_bytes * 4 + u64::from(task_budget().max_output_tokens) * 8 + 4096;
    // Compare schedules only within the same exact schema, document, input
    // encoding and full/selected head policy. No cross-policy parity claim.
    for (grounded, compact, selected) in [
        (false, false, false), (true, false, true), (false, true, true), (true, true, false),
    ] {
        let expected = host.execute_int8_extract(&model,
            prepare(model.artifact_identity(), grounded, compact, selected), Arc::clone(&vocabulary),
            limits().native, limits().mask_limits, limits().max_mask_node_visits, CancellationToken::default()).unwrap();
        if grounded {
            assert_eq!(expected.result().result.output.json, r#""Alice""#);
            assert_eq!(expected.result().result.source_fields.len(), 1);
            let evidence = &expected.result().result.source_fields[0];
            assert_eq!(evidence.spans.len(), 2);
            for span in &evidence.spans {
                assert_eq!(&DOCUMENT[span.byte_start..span.byte_end], "Alice");
                assert_eq!(DOCUMENT[..span.byte_start].chars().count(), span.scalar_start);
            }
        }
        let expected_bytes = canonjson::canonical_bytes(expected.result()).unwrap();
        let expected_work = expected.result().model_work;
        drop(expected); drained(&host, baseline);
        for schedule in [prefill(1), prefill(3)] {
            let result = host.execute_int8_extract_layer_major(&model,
                prepare(model.artifact_identity(), grounded, compact, selected), Arc::clone(&vocabulary),
                limits(), schedule, CancellationToken::default()).unwrap();
            compare(result.result(), &expected_bytes, expected_work);
            drained(&host, baseline + output); assert_eq!(Arc::strong_count(&vocabulary), 1);
            drop(result); drained(&host, baseline);
        }
        for width in [2, 4] { for schedule in [None, Some(prefill(1)), Some(prefill(3))] {
            let prepared = prepare(model.artifact_identity(), grounded, compact, selected);
            let result = match schedule {
                None => host.execute_int8_extract_with_threads(&model, prepared, Arc::clone(&vocabulary),
                    limits(), process_kernel(), width, CancellationToken::default()),
                Some(prefill) => host.execute_int8_extract_layer_major_with_threads(&model, prepared,
                    Arc::clone(&vocabulary), limits(), process_kernel(), width, prefill, CancellationToken::default()),
            }.unwrap();
            compare(result.result(), &expected_bytes, expected_work);
            assert_eq!(result.compute().compute_threads, width);
            assert_eq!(result.compute().completed_rows, expected_work.projections.dot_products);
            assert!(result.compute().completed_projections > 0);
            assert!(result.compute().completed_worker_rows > 0);
            drained(&host, baseline + output); assert_eq!(Arc::strong_count(&vocabulary), 1);
            drop(result); drained(&host, baseline);
        } }
    }

    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        for width in [1, 4] {
            let cancellation = CancellationToken::default(); cancellation.cancel(cause);
            let prepared = prepare(model.artifact_identity(), true, true, true);
            let error = if width == 1 {
                host.execute_int8_extract_layer_major(&model, prepared, Arc::clone(&vocabulary),
                    limits(), prefill(3), cancellation).err()
            } else {
                host.execute_int8_extract_layer_major_with_threads(&model, prepared, Arc::clone(&vocabulary),
                    limits(), process_kernel(), width, prefill(3), cancellation).err()
            }.expect("pre-cancelled request cannot publish");
            assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
            assert_eq!(Arc::strong_count(&vocabulary), 1); drained(&host, baseline);
        }
    }
    for width in [0, 1, 5, 65, usize::MAX] {
        let error = host.execute_int8_extract_layer_major_with_threads(&model,
            prepare(model.artifact_identity(), false, false, false), Arc::clone(&vocabulary),
            limits(), process_kernel(), width, prefill(3), CancellationToken::default())
            .err().expect("unadmitted width cannot enter work");
        assert!(matches!(error, HostedError::Limits(_)));
        assert_eq!(Arc::strong_count(&vocabulary), 1); drained(&host, baseline);
    }
    // The plan retains its original KV ceiling even though the hosted API
    // prices the complete resident context independently of that opaque plan.
    for width in [1, 2] {
        let mut short = task_budget(); short.max_kv_bytes -= 1;
        let prepared = prepare_with_budget(model.artifact_identity(), true, true, true, short);
        let error = if width == 1 {
            host.execute_int8_extract_layer_major(&model, prepared, Arc::clone(&vocabulary),
                limits(), prefill(3), CancellationToken::default()).err()
        } else {
            host.execute_int8_extract_layer_major_with_threads(&model, prepared, Arc::clone(&vocabulary),
                limits(), process_kernel(), width, prefill(3), CancellationToken::default()).err()
        }.expect("the full resident KV capacity must fit the sealed task budget");
        assert!(matches!(error, HostedError::Extraction(Int8ExtractError::Native(
            Int8JsonError::Native(StrictInt8Error::Memory)))));
        assert_eq!(Arc::strong_count(&vocabulary), 1); drained(&host, baseline);
    }
    let mut foreign = model.artifact_identity().clone(); foreign.logical_model_sha256 = "ef".repeat(32);
    assert!(matches!(host.execute_int8_extract_layer_major_with_threads(&model,
        prepare(&foreign, false, false, false), Arc::clone(&vocabulary), limits(), process_kernel(), 2, prefill(3),
        CancellationToken::default()), Err(HostedError::ModelIdentity)));
    drained(&host, baseline);
    drop(vocabulary); drop(model); drop(caller_preparation); drop(lease); drained(&host, 0);
}
