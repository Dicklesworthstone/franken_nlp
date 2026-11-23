//! Controller-only artifact gate. This is never a synthetic decoder or an
//! automatically executed build/benchmark. Run in a fresh test process with
//! independently pinned FNLP_INT8_SOURCE_TEAM_MODEL, *_LOGICAL_SHA256 and
//! *_SOURCE_ROOT_SHA256 environment values.
use super::*;
use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset, hosted::{LoadLimits, RunStop},
    native_engine::simd::config::process_kernel};

fn drained(host: &NlpEngine, baseline: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, baseline); assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
}

#[test]
#[ignore = "controller-only real INT8 artifact and fresh resource-host process; no parity claim until executed"]
fn all_four_source_tasks_keep_serial_results_work_and_guards_with_team_prompt_schedules() {
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_SOURCE_TEAM_MODEL").expect("admitted INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_SOURCE_TEAM_LOGICAL_SHA256").expect("independent logical model pin required");
    let source_root = std::env::var("FNLP_INT8_SOURCE_TEAM_SOURCE_ROOT_SHA256").expect("independent source root pin required");
    Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source_root).unwrap();
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: 16_u64 << 30,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    // Model loading owns its own artifact charge. This separate caller charge
    // precedes the vocabulary/planners and survives them through every return.
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
    // Exercise both exact input encodings and both complete head policies;
    // compare widths and prompt morsels only within the same sealed policy
    // and compiled inputs. A one-row morsel also tests the batch boundary.
    for (kind, compact, selected) in [(0, false, false), (1, false, true), (2, true, false), (3, true, true)] {
        let expected = host.execute_int8_source(&model, prepare(model.artifact_identity(), kind, compact, selected),
            Arc::clone(&vocabulary), limits(), CancellationToken::default()).unwrap();
        let expected_bytes = canonjson::canonical_bytes(expected.result()).unwrap();
        let expected_work = expected.result().model_work;
        drop(expected); drained(&host, baseline);
        for width in [2, 4] { for schedule in [None, Some(prefill(1)), Some(prefill(3))] {
            let prepared = prepare(model.artifact_identity(), kind, compact, selected);
            let result = match schedule {
                None => host.execute_int8_source_with_threads(&model, prepared,
                    Arc::clone(&vocabulary), limits(), process_kernel(), width, CancellationToken::default()),
                Some(prefill) => host.execute_int8_source_layer_major_with_threads(&model, prepared,
                    Arc::clone(&vocabulary), limits(), process_kernel(), width, prefill, CancellationToken::default()),
            }.unwrap();
            assert_eq!(canonjson::canonical_bytes(result.result()).unwrap(), expected_bytes);
            assert_eq!(result.result().model_work, expected_work);
            assert_eq!(result.compute().compute_threads, width);
            assert_eq!(result.compute().completed_rows, expected_work.projections.dot_products);
            assert!(result.compute().completed_projections > 0);
            assert!(result.compute().completed_worker_rows > 0);
            // Successful publication has already joined its team and released
            // input/native/prefill charges, while complete output stays guarded.
            let output = task_budget().max_output_bytes * 4 + u64::from(task_budget().max_output_tokens) * 8 + 4096;
            drained(&host, baseline + output);
            assert_eq!(Arc::strong_count(&vocabulary), 1);
            drop(result); drained(&host, baseline);
        } }
    }

    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let cancellation = CancellationToken::default(); cancellation.cancel(cause);
        let error = host.execute_int8_source_layer_major_with_threads(&model,
            prepare(model.artifact_identity(), 0, false, false), Arc::clone(&vocabulary), limits(),
            process_kernel(), 4, prefill(3), cancellation).err().expect("pre-cancelled request cannot publish");
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
        assert_eq!(Arc::strong_count(&vocabulary), 1); drained(&host, baseline);
    }
    for width in [0, 1, 5, 65, usize::MAX] {
        let error = host.execute_int8_source_layer_major_with_threads(&model,
            prepare(model.artifact_identity(), 0, false, false), Arc::clone(&vocabulary), limits(),
            process_kernel(), width, prefill(3), CancellationToken::default()).err().expect("unadmitted width cannot enter work");
        assert!(matches!(error, HostedError::Limits(_)));
        assert_eq!(Arc::strong_count(&vocabulary), 1); drained(&host, baseline);
    }
    for schedule in [
        Int8PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX },
        Int8PrefillLimits { max_extra_scratch_bytes: prefill(3).max_extra_scratch_bytes - 1, ..prefill(3) },
    ] {
        let error = host.execute_int8_source_layer_major_with_threads(&model,
            prepare(model.artifact_identity(), 0, false, false), Arc::clone(&vocabulary), limits(),
            process_kernel(), 2, schedule, CancellationToken::default()).err().expect("invalid prompt workspace cannot enter work");
        assert!(matches!(error, HostedError::Native(_)));
        assert_eq!(Arc::strong_count(&vocabulary), 1); drained(&host, baseline);
    }
    let mut foreign = model.artifact_identity().clone(); foreign.logical_model_sha256 = "ef".repeat(32);
    assert!(matches!(host.execute_int8_source_layer_major_with_threads(&model,
        prepare(&foreign, 0, false, false), Arc::clone(&vocabulary), limits(), process_kernel(), 2, prefill(3),
        CancellationToken::default()), Err(HostedError::ModelIdentity)));
    drained(&host, baseline);
    drop(vocabulary); drop(model); drop(caller_preparation); drop(lease); drained(&host, 0);
}
