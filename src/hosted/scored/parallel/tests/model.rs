//! Controller-only real-artifact differential, never an automatic model fetch.
//! Run in a fresh test process with independently pinned
//! FNLP_INT8_SCORED_TEAM_MODEL, *_LOGICAL_SHA256 and *_SOURCE_ROOT_SHA256.
use super::*;
use crate::{LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    native_engine::simd::config::process_kernel};

fn drained(host: &NlpEngine, baseline: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, baseline);
    assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
}

fn compare<P, F, S, T>(host: &NlpEngine, baseline: u64, prepare: F, serial: S, threaded: T)
where P: ScoredTeamPlan, P::Output: serde::Serialize,
    F: Fn() -> P,
    S: Fn(P) -> HostedOutput<P::Output>,
    T: Fn(P, usize, Option<Int8PrefillLimits>) -> HostedInt8TeamResult<P::Output> {
    let quote = prepare().quote();
    let expected = serial(prepare());
    let expected_bytes = canonjson::canonical_bytes(expected.result()).unwrap();
    let (expected_work, expected_heads) = P::accounting(expected.result());
    assert_eq!(expected_work, quote.work);
    assert_eq!(expected_heads, quote.heads);
    drop(expected);
    drained(host, baseline);
    for width in [2, 4] {
        for schedule in [None, Some(prefill(4))] {
            let result = threaded(prepare(), width, schedule);
            assert_eq!(canonjson::canonical_bytes(result.result()).unwrap(), expected_bytes);
            assert_eq!(P::accounting(result.result()), (expected_work, expected_heads));
            assert_eq!(result.compute().compute_threads, width);
            assert_eq!(result.compute().completed_rows, expected_work.projections.dot_products);
            assert!(result.compute().completed_projections > 0);
            assert!(result.compute().completed_worker_rows > 0);
            let output = quote.result_bytes * 4 + u64::from(quote.task.max_output_tokens) * 8 + 4096;
            // The output guard is the only per-request ownership remaining.
            // All native/input reservations and physical workers have drained.
            drained(host, baseline + output);
            drop(result);
            drained(host, baseline);
        }
    }
}

#[test]
#[ignore = "controller-only real INT8 artifact, independent digests and fresh host; no parity claim until executed"]
fn every_scored_task_preserves_serial_results_full_work_and_physical_drain() {
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_SCORED_TEAM_MODEL").expect("admitted INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_SCORED_TEAM_LOGICAL_SHA256").expect("independent logical pin required");
    let source_root = std::env::var("FNLP_INT8_SCORED_TEAM_SOURCE_ROOT_SHA256").expect("independent source pin required");
    Sha256Digest::from_hex(&logical).unwrap();
    Sha256Digest::from_hex(&source_root).unwrap();
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: 16_u64 << 30,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let lease = host.resources().acquire_lease();
    // The caller prices every planner and prepared value before construction.
    // Sentiment additionally transfers its existing explicit host reservation.
    let caller_preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, 512 << 20)
        .unwrap().commit().unwrap();
    let model = host.load_current_candidate_int8(path, LoadLimits {
        artifact: ArtifactLoadBudget::streaming_only(6_u64 << 30, 64 << 10),
        tokenizer_and_metadata_bytes: 128 << 20, allocator_reserve_bytes: 64 << 20,
        run: native().run,
    }, CancellationToken::default()).unwrap();
    assert_eq!(model.artifact_identity().logical_model_sha256, logical);
    assert_eq!(model.artifact_identity().source_root_sha256, source_root);
    assert_eq!(model.artifact_identity().revision, "f56ec5a9650268aa098496734743c25ea778bd2d");
    let baseline = host.resources().memory_snapshot().committed_bytes;

    for (mode, compact) in [(ClassificationMode::Exclusive, false), (ClassificationMode::MultiLabel, true)] {
        compare(&host, baseline,
            || classification(model.artifact_identity(), mode, compact),
            |plan| host.execute_int8_classify(&model, plan, native(), CancellationToken::default()).unwrap(),
            |plan, width, schedule| match schedule {
                None => host.execute_int8_classify_with_threads(&model, plan, native(),
                    process_kernel(), width, CancellationToken::default()).unwrap(),
                Some(prefill) => host.execute_int8_classify_layer_major_with_threads(&model, plan, native(),
                    process_kernel(), width, prefill, CancellationToken::default()).unwrap(),
            });
    }
    for kind in 0..3 {
        compare(&host, baseline,
            || judgment(model.artifact_identity(), kind, kind != 0),
            |plan| host.execute_int8_judge(&model, plan, native(), CancellationToken::default()).unwrap(),
            |plan, width, schedule| match schedule {
                None => host.execute_int8_judge_with_threads(&model, plan, native(),
                    process_kernel(), width, CancellationToken::default()).unwrap(),
                Some(prefill) => host.execute_int8_judge_layer_major_with_threads(&model, plan, native(),
                    process_kernel(), width, prefill, CancellationToken::default()).unwrap(),
            });
    }
    for mode in [ScoringMode::FullVocabulary, ScoringMode::TrieConditional] {
        compare(&host, baseline,
            || sentiment_plan(model.artifact_identity(), mode, true),
            |plan| {
                let limits = sentiment_limits(plan.planned_work());
                host.execute_int8_sentiment(&model, plan, limits, CancellationToken::default()).unwrap()
            },
            |plan, width, schedule| {
                let limits = sentiment_limits(plan.planned_work());
                match schedule {
                    None => host.execute_int8_sentiment_with_threads(&model, plan, limits,
                        process_kernel(), width, CancellationToken::default()).unwrap(),
                    Some(prefill) => host.execute_int8_sentiment_layer_major_with_threads(&model, plan, limits,
                        process_kernel(), width, prefill, CancellationToken::default()).unwrap(),
                }
            });
    }

    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let cancellation = CancellationToken::default(); cancellation.cancel(cause);
        let error = host.execute_int8_classify_layer_major_with_threads(&model,
            classification(model.artifact_identity(), ClassificationMode::MultiLabel, true),
            native(), process_kernel(), 4, prefill(4), cancellation)
            .err().expect("pre-cancelled scoring must not publish");
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
        drained(&host, baseline);
    }
    for width in [0, 1, 5, 65, usize::MAX] {
        let error = host.execute_int8_judge_with_threads(&model, judgment(model.artifact_identity(), 0, false),
            native(), process_kernel(), width, CancellationToken::default())
            .err().expect("unadmitted width must not enter work");
        assert!(matches!(error, HostedError::Limits(_)));
        drained(&host, baseline);
    }
    let mut foreign = model.artifact_identity().clone(); foreign.logical_model_sha256 = "ef".repeat(32);
    let plan = sentiment_plan(&foreign, ScoringMode::FullVocabulary, true);
    let limits = sentiment_limits(plan.planned_work());
    assert!(matches!(host.execute_int8_sentiment_with_threads(&model, plan, limits,
        process_kernel(), 2, CancellationToken::default()), Err(HostedError::ModelIdentity)));
    drained(&host, baseline);
    drop(model); drop(caller_preparation); drop(lease); drained(&host, 0);
}
