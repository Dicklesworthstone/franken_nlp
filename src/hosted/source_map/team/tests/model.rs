//! Controller-only real-artifact differential and physical-drain gate.
//! Run this ignored test alone in a fresh resource-host process. Independent
//! artifact pins are required; source definitions are not execution evidence.
use super::*;
use crate::{
    canonjson, LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    hosted::{LoadLimits, RunStop},
    native_engine::{simd::config::process_kernel, strict_int8::{Int8Work, STRICT_INT8_EXECUTION}},
    tasks::{mapreduce::{ChunkLimits, ExecutionLimits}, ner::NerOptions,
        keyphrases::KeyphraseOptions, summarize::SummaryOptions,
        source_planning::quantized::long::{
            summary::{Int8CorpusSummaryRun, Int8SummaryLimits},
            keyphrase_reduction::{Int8CorpusKeyphraseRun, Int8KeyphraseLimits}}},
    tokenizer::{pinned_controls, untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding}},
};

const SOURCE: &str = "Alice visits Paris. Bob visits Rome. Alice likes maps. é 上海.";
const MAX_NEW: u32 = 512;

fn assets(facts: &ArtifactIdentity, kind: usize) -> (Arc<SourceTaskPlanner>, SourceMapConfig) {
    let registry = pinned_controls::pinned().unwrap();
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.surface == crate::template::IM_END).unwrap().id;
    let mut planner = SourceTaskPlanner::pinned(controls, eos).unwrap();
    if kind % 2 == 1 {
        planner = planner.with_map_document_encoding(UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: 4096, max_output_tokens: 1536, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
        }), SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
            max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 }).unwrap();
    }
    let task = match kind {
        0 => SourceMapTask::Ner(NerOptions::default()),
        1 | 4 => SourceMapTask::Keyphrases(KeyphraseOptions::default()),
        2 | 3 => SourceMapTask::Summarize(SummaryOptions::default()),
        _ => unreachable!(),
    };
    let name = match &task { SourceMapTask::Ner(_) => "ner-v1",
        SourceMapTask::Keyphrases(_) => "keyphrases-v1", SourceMapTask::Summarize(_) => "summarize-v1" };
    let absent = Sha256Digest::of_bytes(b"null");
    let packing = canonjson::canonical_bytes(&("current-candidate-portable-int8-v1", Vec::<String>::new())).unwrap();
    let identity = ExecutionIdentity { schema_version: 1, source_revision: facts.revision.clone(),
        logical_model_digest: Sha256Digest::from_hex(&facts.logical_model_sha256).unwrap(),
        artifact_format: "fnlpq-current-candidate-v1".to_owned(), quant_recipe: facts.recipe_id.clone(),
        packing_set_digest: Sha256Digest::of_bytes(&packing), tokenizer_digest: planner.tokenizer_digest(),
        template_digest: *planner.template_digest(), task_spec: name.to_owned(),
        taskir_digest: absent, prompt_digest: absent, grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    let budget = TaskBudget { max_input_tokens: 1536, max_output_tokens: MAX_NEW, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: requirements(native()).unwrap().kv_bytes };
    let mapping = Int8SourceMapLimits {
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 32, max_chunk_tokens: 256,
            context_tokens: 2048, reserved_tokens: 512, max_chunks: 8, max_tokenizer_calls: 4096 },
        reduction: ExecutionLimits { map_batch_chunks: 2, reduce_fan_in: 2,
            max_value_bytes: 4 << 20, max_result_bytes: 16 << 20,
            max_live_value_bytes: 32 << 20, max_total_value_bytes: 64 << 20, ..ExecutionLimits::default() },
        max_model_work: Int8Work::for_sequence(0, 65_536, 8192 * NANBEIGE_VOCAB_SIZE).unwrap(),
        mask_limits: MaskWorkLimits { max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
        mask_visits_per_chunk: 1_024_000_000, max_mask_visits: 8_192_000_000,
    };
    let config = SourceMapConfig { identity, task, budget,
        planning: SourcePlanningLimits { max_context_tokens: 2048, ..SourcePlanningLimits::default() },
        mapping, native: native(), preparation_reserve_bytes: 512 << 20, reduction_reserve_bytes: 64 << 20 };
    (Arc::new(planner), config)
}

fn drained(host: &NlpEngine, baseline: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, baseline);
    assert_eq!(memory.reserved_bytes, 0); assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
}

enum Output<T> { Serial(HostedOutput<T>), Team(HostedInt8TeamResult<T>) }
impl<T> Output<T> {
    fn result(&self) -> &T { match self { Self::Serial(o) => o.result(), Self::Team(o) => o.result() } }
    fn compute(&self) -> Option<&Int8ComputeReceipt> { match self { Self::Serial(_) => None, Self::Team(o) => Some(o.compute()) } }
}
trait WorkResult: serde::Serialize {
    fn work(&self) -> Int8Work;
    fn output_allowance(&self) -> (u64, u64);
}
impl WorkResult for Int8SourceMapRun {
    fn work(&self) -> Int8Work { self.model_work }
    fn output_allowance(&self) -> (u64, u64) {
        (16 << 20, self.mapped.root().value().len() as u64 * u64::from(MAX_NEW))
    }
}
impl WorkResult for Int8CorpusSummaryRun {
    fn work(&self) -> Int8Work { self.model_work }
    fn output_allowance(&self) -> (u64, u64) { (4 << 20, 0) }
}
impl WorkResult for Int8CorpusKeyphraseRun {
    fn work(&self) -> Int8Work { self.model_work }
    fn output_allowance(&self) -> (u64, u64) { (4 << 20, 0) }
}
fn capture<T: WorkResult>(host: &NlpEngine, baseline: u64, output: Output<T>, width: usize) -> Vec<u8> {
    let bytes = canonjson::canonical_bytes(output.result()).unwrap();
    if let Some(compute) = output.compute() {
        assert_eq!(compute.compute_threads, width);
        assert_eq!(compute.completed_rows, output.result().work().projections.dot_products);
        assert!(compute.completed_worker_rows > 0);
        assert!(compute.completed_projections > 0);
    } else { assert_eq!(width, 1); }
    let (cap, tokens) = output.result().output_allowance();
    // Complete input, native and worker storage has drained. Only the result
    // and its exact serialization/token allowance remain charged.
    drained(host, baseline + cap * 4 + tokens * 8 + 4096);
    drop(output); drained(host, baseline);
    bytes
}

fn execute(host: &NlpEngine, model: &ResidentInt8, vocabulary: &Arc<ExtractionVocabulary>,
    baseline: u64, kind: usize, width: usize, schedule: Option<Int8PrefillLimits>) -> Vec<u8> {
    let (planner, config) = assets(model.artifact_identity(), kind);
    let source = SOURCE.to_owned(); let vocabulary = Arc::clone(vocabulary);
    let cancellation = CancellationToken::default(); let kernel = process_kernel();
    if kind == 3 {
        let summary = Int8SummaryLimits::default();
        let output = if width == 1 { Output::Serial(match schedule {
            None => host.summarize_int8_source(model, source, planner, vocabulary, config, summary, cancellation),
            Some(prefill) => host.summarize_int8_source_layer_major(model, source, planner, vocabulary, config, summary, prefill, cancellation),
        }.unwrap()) } else { Output::Team(match schedule {
            None => host.summarize_int8_source_with_threads(model, source, planner, vocabulary, config, summary, kernel, width, cancellation),
            Some(prefill) => host.summarize_int8_source_layer_major_with_threads(model, source, planner, vocabulary, config, summary, kernel, width, prefill, cancellation),
        }.unwrap()) };
        return capture(host, baseline, output, width);
    }
    if kind == 4 {
        let keyphrases = Int8KeyphraseLimits::default();
        let output = if width == 1 { Output::Serial(match schedule {
            None => host.keyphrases_int8_source(model, source, planner, vocabulary, config, keyphrases, cancellation),
            Some(prefill) => host.keyphrases_int8_source_layer_major(model, source, planner, vocabulary, config, keyphrases, prefill, cancellation),
        }.unwrap()) } else { Output::Team(match schedule {
            None => host.keyphrases_int8_source_with_threads(model, source, planner, vocabulary, config, keyphrases, kernel, width, cancellation),
            Some(prefill) => host.keyphrases_int8_source_layer_major_with_threads(model, source, planner, vocabulary, config, keyphrases, kernel, width, prefill, cancellation),
        }.unwrap()) };
        return capture(host, baseline, output, width);
    }
    let output = if width == 1 { Output::Serial(match schedule {
        None => host.map_int8_source(model, source, planner, vocabulary, config, cancellation),
        Some(prefill) => host.map_int8_source_layer_major(model, source, planner, vocabulary, config, prefill, cancellation),
    }.unwrap()) } else { Output::Team(match schedule {
        None => host.map_int8_source_with_threads(model, source, planner, vocabulary, config, kernel, width, cancellation),
        Some(prefill) => host.map_int8_source_layer_major_with_threads(model, source, planner, vocabulary, config, kernel, width, prefill, cancellation),
    }.unwrap()) };
    capture(host, baseline, output, width)
}

#[test]
#[ignore = "controller-only real INT8 artifact, independent pins and fresh process; execution evidence pending"]
fn all_source_maps_and_exact_reducers_preserve_complete_results_with_packed_persistent_teams() {
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_SOURCE_MAP_TEAM_MODEL").expect("admitted INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_SOURCE_MAP_TEAM_LOGICAL_SHA256").expect("independent logical pin required");
    let source_root = std::env::var("FNLP_INT8_SOURCE_MAP_TEAM_SOURCE_ROOT_SHA256").expect("independent source-root pin required");
    Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source_root).unwrap();
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: 16_u64 << 30,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let lease = host.resources().acquire_lease();
    let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, 512 << 20).unwrap().commit().unwrap();
    let model = host.load_current_candidate_int8(path, LoadLimits {
        artifact: ArtifactLoadBudget::streaming_only(6_u64 << 30, 64 << 10),
        tokenizer_and_metadata_bytes: 128 << 20, allocator_reserve_bytes: 64 << 20, run: native().run,
    }, CancellationToken::default()).unwrap();
    assert_eq!(model.artifact_identity().logical_model_sha256, logical);
    assert_eq!(model.artifact_identity().source_root_sha256, source_root);
    let vocabulary = Arc::new(ExtractionVocabulary::pinned(pinned_controls::pinned().unwrap().template_controls()).unwrap());
    let baseline = host.resources().memory_snapshot().committed_bytes;
    for kind in 0..5 {
        let expected = execute(&host, &model, &vocabulary, baseline, kind, 1, None);
        for width in [1, 2, 4] { for schedule in [Some(prefill(1)), Some(prefill(7))] {
            assert_eq!(execute(&host, &model, &vocabulary, baseline, kind, width, schedule), expected);
            assert_eq!(Arc::strong_count(&vocabulary), 1);
        } }
        for width in [2, 4] {
            assert_eq!(execute(&host, &model, &vocabulary, baseline, kind, width, None), expected);
        }
    }

    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let (planner, config) = assets(model.artifact_identity(), 0);
        let cancellation = CancellationToken::default(); cancellation.cancel(cause);
        let error = host.map_int8_source_layer_major_with_threads(&model, SOURCE.to_owned(), planner,
            Arc::clone(&vocabulary), config, process_kernel(), 4, prefill(7), cancellation)
            .err().expect("cancelled document cannot publish");
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
        assert_eq!(Arc::strong_count(&vocabulary), 1); drained(&host, baseline);
    }
    for width in [0, 1, 5, 65, usize::MAX] {
        let (planner, config) = assets(model.artifact_identity(), 0);
        assert!(matches!(host.map_int8_source_layer_major_with_threads(&model, SOURCE.to_owned(), planner,
            Arc::clone(&vocabulary), config, process_kernel(), width, prefill(7), CancellationToken::default()),
            Err(HostedError::Limits(_))));
        drained(&host, baseline);
    }
    for schedule in [Int8PrefillLimits { max_batch_rows: 65, max_extra_scratch_bytes: u64::MAX },
        Int8PrefillLimits { max_extra_scratch_bytes: prefill(7).max_extra_scratch_bytes - 1, ..prefill(7) }] {
        let (planner, config) = assets(model.artifact_identity(), 0);
        assert!(matches!(host.map_int8_source_layer_major_with_threads(&model, SOURCE.to_owned(), planner,
            Arc::clone(&vocabulary), config, process_kernel(), 2, schedule, CancellationToken::default()),
            Err(HostedError::Native(_))));
        drained(&host, baseline);
    }
    let (planner, config) = assets(model.artifact_identity(), 0);
    assert!(host.map_int8_source_layer_major_with_threads(&model, String::new(), planner,
        Arc::clone(&vocabulary), config, process_kernel(), 2, prefill(7), CancellationToken::default()).is_err());
    drained(&host, baseline);
    for axis in 0..5 {
        let (planner, mut config) = assets(model.artifact_identity(), 0);
        match axis { 0 => config.mapping.max_model_work.forward_positions = 1,
            1 => config.mapping.max_model_work.projected_logits = NANBEIGE_VOCAB_SIZE as u64,
            2 => config.mapping.max_model_work.attention_pairs = 1,
            3 => config.mapping.max_model_work.projections.dot_products = 1,
            _ => config.mapping.max_model_work.projections.multiply_accumulates = 1 }
        assert!(host.map_int8_source_layer_major_with_threads(&model, SOURCE.to_owned(), planner,
            Arc::clone(&vocabulary), config, process_kernel(), 2, prefill(7), CancellationToken::default()).is_err());
        drained(&host, baseline);
    }
    let (planner, mut foreign) = assets(model.artifact_identity(), 0);
    foreign.identity.logical_model_digest = Sha256Digest::of_bytes(b"foreign model");
    assert!(matches!(host.map_int8_source_layer_major_with_threads(&model, SOURCE.to_owned(), planner,
        Arc::clone(&vocabulary), foreign, process_kernel(), 2, prefill(7), CancellationToken::default()),
        Err(HostedError::ModelIdentity)));
    drained(&host, baseline);
    drop(vocabulary); drop(model); drop(preparation); drop(lease); drained(&host, 0);
}
