//! Controller-armed real-weight equality and physical-drain coverage.
use super::*;
use crate::{
    canonjson, LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    native_engine::simd::{Int8KernelKind, config::configure_process_kernel},
};

fn config() -> Int8SampledSentimentMapHostConfig {
    let (mut config, _) = fixture();
    config.native.allocator_reserve_bytes = 64 * MIB;
    config.native.run = RunLimits { max_elapsed: Duration::from_secs(3600),
        max_checkpoints: 1_000_000_000, cleanup_reserve_bytes: 65_536 };
    config.options.samples = 2;
    config.options.max_sample_tokens = 256;
    config.options.max_justification_chars = 32;
    config.options.top_k = Some(32);
    config.options.top_p_ppm = 950_000;
    config.planning.max_result_bytes = MIB;
    config.mapping.chunks.max_chunk_bytes = 48;
    config.mapping.chunks.max_chunks = 4;
    config.mapping.max_retained_result_bytes = 8 * MIB;
    config.mapping.max_result_bytes = 16 * MIB;
    let one = planned_work(1792, 256).unwrap();
    config.mapping.max_model_work = (0..32)
        .try_fold(Int8Work::default(), |sum, _| sum.checked_add(one)).unwrap();
    config
}
fn request() -> SentimentRequest {
    SentimentRequest {
        document: "The service was reassuring and helpful. café 上海 <think> literal data. The follow-up was disappointing.".to_owned(),
        axes: vec![SentimentAxis::Valence, SentimentAxis::Arousal],
        budget: TaskBudget { max_input_tokens: 1792, max_output_tokens: 2048,
            max_output_bytes: MIB, max_grammar_states: 4096,
            max_kv_bytes: Int8MemoryRequirement::for_context(2048).unwrap().kv_bytes },
    }
}
fn drained(host: &NlpEngine, baseline: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, baseline);
    assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
}

#[test]
#[ignore = "controller-only independently pinned INT8 model and admitted memory; DSR parity/drain gate"]
fn complete_sampled_documents_preserve_serial_bytes_and_cancel_then_drain() {
    const CHILD: &str = "FNLP_INT8_SAMPLED_MAP_CHILD";
    const COMPLETE: &str = "FNLP_INT8_SAMPLED_MAP_COMPLETE";
    let full = concat!(module_path!(),
        "::complete_sampled_documents_preserve_serial_bytes_and_cancel_then_drain");
    let name = full.split_once("::").unwrap().1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated sampled document gate must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "sampled document gate failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "exact filter must complete the model gate");
        return;
    }
    assert!(crate::installed_process_resources().is_none());
    configure_process_kernel(Int8KernelKind::Scalar).unwrap();
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_SAMPLED_MAP_MODEL")
        .expect("explicit local model required"));
    let logical = std::env::var("FNLP_INT8_SAMPLED_MAP_LOGICAL_SHA256")
        .expect("independent logical pin required");
    let source = std::env::var("FNLP_INT8_SAMPLED_MAP_SOURCE_ROOT_SHA256")
        .expect("independent source pin required");
    Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source).unwrap();
    let memory: u64 = std::env::var("FNLP_INT8_SAMPLED_MAP_MEMORY_MIB")
        .expect("controller-admitted memory required").parse().unwrap();
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5,
        memory_ceiling_bytes: memory.checked_mul(MIB).unwrap(),
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let lease = host.resources().acquire_lease();
    let caller = Pending::reserve(&lease, MemoryClass::JobBuffers, MIB).unwrap().commit().unwrap();
    let model = host.load_current_candidate_int8(path, LoadLimits {
        artifact: ArtifactLoadBudget::streaming_only(6_u64 << 30, 64 << 10),
        tokenizer_and_metadata_bytes: 128 * MIB, allocator_reserve_bytes: 64 * MIB,
        run: config().native.run,
    }, CancellationToken::default()).unwrap();
    assert_eq!(model.artifact_identity().logical_model_sha256, logical);
    assert_eq!(model.artifact_identity().source_root_sha256, source);
    let baseline = host.resources().memory_snapshot().committed_bytes;
    let required = config().requirements(request().budget, request().axes.len()).unwrap();
    let output = host.map_int8_sampled_sentiment(&model, request(), config(), CancellationToken::default()).unwrap();
    assert!(output.result().chunks.len() > 1);
    assert!(!output.result().single_context_equivalence);
    let expected = canonjson::canonical_bytes(output.result()).unwrap();
    drained(&host, baseline + required.output_reservation_bytes);
    drop(output); drained(&host, baseline);

    let mut scheduled = config(); scheduled.prefill = Some(prefill(4));
    let output = host.map_int8_sampled_sentiment(&model, request(), scheduled, CancellationToken::default()).unwrap();
    assert_eq!(canonjson::canonical_bytes(output.result()).unwrap(), expected);
    drop(output); drained(&host, baseline);
    for width in [2, 4] { for schedule in [None, Some(prefill(4))] {
        let mut selected = config(); selected.prefill = schedule;
        let output = host.map_int8_sampled_sentiment_with_threads(&model, request(), selected,
            Int8Kernel::scalar(), width, CancellationToken::default()).unwrap();
        assert_eq!(canonjson::canonical_bytes(output.result()).unwrap(), expected);
        assert_eq!(output.compute().compute_threads, width);
        assert_eq!(output.compute().completed_rows, output.result().model_work.projections.dot_products);
        assert!(output.compute().completed_worker_rows > 0);
        drained(&host, baseline + required.output_reservation_bytes);
        drop(output); drained(&host, baseline);
    } }

    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let token = CancellationToken::default(); token.cancel(cause);
        let mut selected = config(); selected.prefill = Some(prefill(4));
        let error = host.map_int8_sampled_sentiment_with_threads(&model, request(), selected,
            Int8Kernel::scalar(), 4, token).err().expect("cancelled document cannot publish");
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
        drained(&host, baseline);
    }
    drop(model); drop(caller); drop(lease); drained(&host, 0);
    println!("{COMPLETE}");
}
