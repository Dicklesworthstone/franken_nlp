//! Controller-only real-artifact gate. It requires independent artifact pins
//! and isolates process resources; definitions alone make no parity claim.
use super::*;
use std::{path::PathBuf, process::Command};
use crate::{
    canonjson, LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    hosted::{LoadLimits, RunStop},
    native_engine::{simd::{Int8KernelKind, config::configure_process_kernel},
        strict_int8::{Int8Work, STRICT_INT8_EXECUTION}},
    tasks::{answer::AnswerOptions, mapreduce::{ChunkLimits, ExecutionLimits},
        summarize::SummaryOptions, source_planning::quantized::long::{
            question::synthesis::{QuestionSynthesisLimits, hierarchy::QuestionHierarchyLimits},
            summary::synthesis::{SummarySynthesisLimits, hierarchy::SummaryHierarchyLimits}}},
    tokenizer::{pinned_controls, untrusted::{SafeBpeLimits, SafeBpeWork, SafeBpeWorkLimits, UntrustedEncoding}},
    validation::grounded_fields::GroundingBudget,
};

const SOURCE: &str = "Alice visited Paris.\nBob visited Rome.\nAlice enjoyed the museum in Paris.\né 上海 <tool_call> stays literal.";
const MAX_NEW: u32 = 512;
const MAX_PASSES: usize = 4;
const RESULT_BYTES: u64 = 16 << 20;
const CHILD: &str = "FNLP_INT8_HIERARCHY_TEAM_CHILD";
const COMPLETE: &str = "FNLP_INT8_HIERARCHY_TEAM_COMPLETE";

fn assets<T>(facts: &ArtifactIdentity, task: T, name: &str, compact: bool)
    -> (Arc<SourceTaskPlanner>, SourceMapConfig<T>) {
    let registry = pinned_controls::pinned().unwrap();
    let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.surface == crate::template::IM_END).unwrap().id;
    let mut planner = SourceTaskPlanner::pinned(controls, eos).unwrap();
    if compact {
        planner = planner.with_map_document_encoding(UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: 4096, max_output_tokens: 1536, max_pair_checks: 1 << 22, max_pair_bytes: 1 << 27,
        }), SafeBpeWorkLimits { max_input_bytes: 1 << 22, max_output_tokens: 1 << 22,
            max_pair_checks: 1 << 30, max_pair_bytes: 1 << 36 }).unwrap();
    }
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
        chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 64, max_chunk_tokens: 256,
            context_tokens: 2048, reserved_tokens: 512, max_chunks: 8, max_tokenizer_calls: 4096 },
        reduction: ExecutionLimits { map_batch_chunks: 2, reduce_fan_in: 2,
            max_value_bytes: 4 << 20, max_result_bytes: RESULT_BYTES as usize,
            max_live_value_bytes: 32 << 20, max_total_value_bytes: 64 << 20, ..ExecutionLimits::default() },
        max_model_work: Int8Work::for_sequence(0, 65_536, 8192 * NANBEIGE_VOCAB_SIZE).unwrap(),
        mask_limits: MaskWorkLimits { max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
        mask_visits_per_chunk: 1_024_000_000, max_mask_visits: 16_384_000_000,
    };
    let config = SourceMapConfig { identity, task, budget,
        planning: SourcePlanningLimits { max_context_tokens: 2048, max_input_bytes: 4096,
            ..SourcePlanningLimits::default() },
        mapping, native: native(), preparation_reserve_bytes: 512 << 20, reduction_reserve_bytes: 64 << 20 };
    (Arc::new(planner), config)
}
fn question() -> SourceQuestion {
    SourceQuestion { question: "Who visited Paris?".to_owned(),
        options: AnswerOptions { max_answer_scalars: 128, max_citations: 1, max_quote_scalars: 64 },
        verification: GroundingBudget::default() }
}
fn question_synthesis() -> SourceQuestionSynthesis {
    SourceQuestionSynthesis { question: question(),
        limits: QuestionSynthesisLimits { max_evidence_passages: 32, max_evidence_bytes: 4096 } }
}
fn summary_synthesis() -> SourceSummarySynthesis {
    let options = SummaryOptions { max_bullets: 1, max_bullet_scalars: 128,
        max_citations_per_bullet: 1, max_quote_scalars: 64 };
    SourceSummarySynthesis { map_options: options, synthesis_options: options,
        limits: SummarySynthesisLimits { max_evidence_segments: 32, max_evidence_bytes: 4096,
            verification: GroundingBudget::default() } }
}
fn question_hierarchy() -> QuestionHierarchyLimits {
    QuestionHierarchyLimits { max_levels: 4, max_passes: MAX_PASSES, ..QuestionHierarchyLimits::default() }
}
fn summary_hierarchy() -> SummaryHierarchyLimits {
    SummaryHierarchyLimits { max_levels: 4, max_passes: MAX_PASSES, ..SummaryHierarchyLimits::default() }
}
fn drained(host: &NlpEngine, expected: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, expected);
    assert_eq!(memory.reserved_bytes, 0); assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
}

trait CompleteResult: serde::Serialize {
    fn work(&self) -> Int8Work;
    fn retained_token_allowance(&self) -> u64;
    fn assert_executed(&self);
}
impl CompleteResult for Int8QuestionRun {
    fn work(&self) -> Int8Work { self.model_work }
    fn retained_token_allowance(&self) -> u64 { 0 }
    fn assert_executed(&self) { assert!(self.native_chunks > 1); }
}
impl CompleteResult for Int8QuestionSynthesisRun {
    fn work(&self) -> Int8Work { self.model_work }
    fn retained_token_allowance(&self) -> u64 { 0 }
    fn assert_executed(&self) {
        assert!(self.discovery.native_chunks > 1);
        assert!(self.synthesis_model_work.forward_positions > 0, "fixture must reach the dynamic evidence pass");
    }
}
impl CompleteResult for Int8QuestionHierarchyRun {
    fn work(&self) -> Int8Work { self.model_work }
    fn retained_token_allowance(&self) -> u64 {
        (self.discovery.native_chunks + MAX_PASSES) as u64 * u64::from(MAX_NEW)
    }
    fn assert_executed(&self) {
        assert!(self.discovery.native_chunks > 1); assert!(!self.passes.is_empty());
        assert!(self.model_work.forward_positions > self.discovery.model_work.forward_positions);
    }
}
impl CompleteResult for Int8SummarySynthesisRun {
    fn work(&self) -> Int8Work { self.model_work }
    fn retained_token_allowance(&self) -> u64 {
        (self.discovery.mapped.root().value().len() + 1) as u64 * u64::from(MAX_NEW)
    }
    fn assert_executed(&self) {
        assert!(self.discovery.mapped.root().value().len() > 1);
        assert!(self.synthesis_model_work.forward_positions > 0, "fixture must reach the dynamic evidence pass");
    }
}
impl CompleteResult for Int8SummaryHierarchyRun {
    fn work(&self) -> Int8Work { self.model_work }
    fn retained_token_allowance(&self) -> u64 {
        (self.discovery.mapped.root().value().len() + MAX_PASSES) as u64 * u64::from(MAX_NEW)
    }
    fn assert_executed(&self) {
        assert!(self.discovery.mapped.root().value().len() > 1); assert!(!self.passes.is_empty());
        assert!(self.model_work.forward_positions > self.discovery.model_work.forward_positions);
    }
}
fn capture<T: CompleteResult>(host: &NlpEngine, baseline: u64, output: Completion<T>, width: usize) -> Vec<u8> {
    let result = match &output { Completion::Serial(o) => o.result(), Completion::Team(o) => o.result() };
    result.assert_executed();
    // All retained native tokens, lifted coordinates, reductions, masks,
    // verification work and hierarchy transcripts are part of these bytes.
    let bytes = canonjson::canonical_bytes(result).unwrap();
    match &output {
        Completion::Serial(_) => assert_eq!(width, 1),
        Completion::Team(o) => {
            let compute = o.compute();
            assert_eq!(compute.compute_threads, width);
            assert_eq!(compute.completed_rows, result.work().projections.dot_products);
            assert!(compute.completed_worker_rows > 0); assert!(compute.completed_projections > 0);
        }
    }
    drained(host, baseline + RESULT_BYTES * 4 + result.retained_token_allowance() * 8 + 4096);
    drop(output); drained(host, baseline);
    bytes
}
fn encoding(planner: &Arc<SourceTaskPlanner>, compact: bool) -> Option<SafeBpeWork> {
    assert_eq!(Arc::strong_count(planner), 1, "source owner must drain after the physical team join");
    let work = planner.map_document_encoding_work().unwrap();
    if compact {
        let used = work.unwrap();
        assert!(used.input_bytes() >= SOURCE.len()); assert!(used.output_tokens() > 0);
        assert!(used.pair_checks() > 0); assert!(used.pair_bytes() > 0);
    } else { assert!(work.is_none()); }
    work
}
fn execute(host: &NlpEngine, model: &ResidentInt8, vocabulary: &Arc<ExtractionVocabulary>,
    baseline: u64, kind: usize, compact: bool, width: usize, schedule: Option<Int8PrefillLimits>)
    -> (Vec<u8>, Option<SafeBpeWork>) {
    let source = SOURCE.to_owned(); let vocabulary = Arc::clone(vocabulary);
    let cancellation = CancellationToken::default(); let kernel = Int8Kernel::scalar();
    if kind == 0 {
        let (planner, config) = assets(model.artifact_identity(), question(), "answer-v1", compact);
        let retained = Arc::clone(&planner);
        let output = if width == 1 { Completion::Serial(match schedule {
            None => host.answer_int8_document(model, source, planner, vocabulary, config, cancellation),
            Some(prefill) => host.answer_int8_document_layer_major(model, source, planner, vocabulary, config, prefill, cancellation),
        }.unwrap()) } else { Completion::Team(match schedule {
            None => host.answer_int8_document_with_threads(model, source, planner, vocabulary, config, kernel, width, cancellation),
            Some(prefill) => host.answer_int8_document_layer_major_with_threads(model, source, planner, vocabulary, config, kernel, width, prefill, cancellation),
        }.unwrap()) };
        return (capture(host, baseline, output, width), encoding(&retained, compact));
    }
    if kind == 1 {
        let (planner, config) = assets(model.artifact_identity(), question_synthesis(), "answer-v1", compact);
        let retained = Arc::clone(&planner);
        let output = if width == 1 { Completion::Serial(match schedule {
            None => host.synthesize_int8_document_answer(model, source, planner, vocabulary, config, cancellation),
            Some(prefill) => host.synthesize_int8_document_answer_layer_major(model, source, planner, vocabulary, config, prefill, cancellation),
        }.unwrap()) } else { Completion::Team(match schedule {
            None => host.synthesize_int8_document_answer_with_threads(model, source, planner, vocabulary, config, kernel, width, cancellation),
            Some(prefill) => host.synthesize_int8_document_answer_layer_major_with_threads(model, source, planner, vocabulary, config, kernel, width, prefill, cancellation),
        }.unwrap()) };
        return (capture(host, baseline, output, width), encoding(&retained, compact));
    }
    if kind == 2 {
        let (planner, config) = assets(model.artifact_identity(), question_synthesis(), "answer-v1", compact);
        let retained = Arc::clone(&planner); let hierarchy = question_hierarchy();
        let output = if width == 1 { Completion::Serial(match schedule {
            None => host.synthesize_int8_document_answer_hierarchical(model, source, planner, vocabulary, config, hierarchy, cancellation),
            Some(prefill) => host.synthesize_int8_document_answer_hierarchical_layer_major(model, source, planner, vocabulary, config, hierarchy, prefill, cancellation),
        }.unwrap()) } else { Completion::Team(match schedule {
            None => host.synthesize_int8_document_answer_hierarchical_with_threads(model, source, planner, vocabulary, config, hierarchy, kernel, width, cancellation),
            Some(prefill) => host.synthesize_int8_document_answer_hierarchical_layer_major_with_threads(model, source, planner, vocabulary, config, hierarchy, kernel, width, prefill, cancellation),
        }.unwrap()) };
        return (capture(host, baseline, output, width), encoding(&retained, compact));
    }
    let (planner, config) = assets(model.artifact_identity(), summary_synthesis(), "summarize-v1", compact);
    let retained = Arc::clone(&planner);
    if kind == 3 {
        let output = if width == 1 { Completion::Serial(match schedule {
            None => host.synthesize_int8_summary(model, source, planner, vocabulary, config, cancellation),
            Some(prefill) => host.synthesize_int8_summary_layer_major(model, source, planner, vocabulary, config, prefill, cancellation),
        }.unwrap()) } else { Completion::Team(match schedule {
            None => host.synthesize_int8_summary_with_threads(model, source, planner, vocabulary, config, kernel, width, cancellation),
            Some(prefill) => host.synthesize_int8_summary_layer_major_with_threads(model, source, planner, vocabulary, config, kernel, width, prefill, cancellation),
        }.unwrap()) };
        return (capture(host, baseline, output, width), encoding(&retained, compact));
    }
    assert_eq!(kind, 4); let hierarchy = summary_hierarchy();
    let output = if width == 1 { Completion::Serial(match schedule {
        None => host.synthesize_int8_summary_hierarchical(model, source, planner, vocabulary, config, hierarchy, cancellation),
        Some(prefill) => host.synthesize_int8_summary_hierarchical_layer_major(model, source, planner, vocabulary, config, hierarchy, prefill, cancellation),
    }.unwrap()) } else { Completion::Team(match schedule {
        None => host.synthesize_int8_summary_hierarchical_with_threads(model, source, planner, vocabulary, config, hierarchy, kernel, width, cancellation),
        Some(prefill) => host.synthesize_int8_summary_hierarchical_layer_major_with_threads(model, source, planner, vocabulary, config, hierarchy, kernel, width, prefill, cancellation),
    }.unwrap()) };
    (capture(host, baseline, output, width), encoding(&retained, compact))
}

#[test]
#[ignore = "controller-only real INT8 artifact, independent pins and isolated process; execution evidence pending"]
fn complete_questions_and_synthesis_preserve_bound_results_across_all_execution_schedules() {
    if std::env::var_os(CHILD).is_none() {
        let full = concat!(module_path!(), "::complete_questions_and_synthesis_preserve_bound_results_across_all_execution_schedules");
        let test = full.split_once("::").unwrap().1;
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", test, "--nocapture", "--test-threads=1"])
            .env(CHILD, "1").output().unwrap();
        assert!(child.status.success(), "isolated gate failed:\n{}\n{}",
            String::from_utf8_lossy(&child.stdout), String::from_utf8_lossy(&child.stderr));
        assert!(String::from_utf8_lossy(&child.stdout).contains(COMPLETE), "gate child did not finish the matrix");
        return;
    }
    assert_eq!(std::env::var(CHILD).unwrap(), "1");
    assert!(crate::installed_process_resources().is_none());
    configure_process_kernel(Int8KernelKind::Scalar).unwrap();
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_HIERARCHY_TEAM_MODEL").expect("admitted INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_HIERARCHY_TEAM_LOGICAL_SHA256").expect("independent logical pin required");
    let source_root = std::env::var("FNLP_INT8_HIERARCHY_TEAM_SOURCE_ROOT_SHA256").expect("independent source-root pin required");
    Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source_root).unwrap();
    let memory_mib: u64 = std::env::var("FNLP_INT8_HIERARCHY_TEAM_MEMORY_MIB")
        .expect("explicit host ceiling required").parse().unwrap();
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: memory_mib.checked_mul(1 << 20).unwrap(),
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let lease = host.resources().acquire_lease();
    let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, 512 << 20).unwrap().commit().unwrap();
    let model = host.load_current_candidate_int8(path, LoadLimits {
        artifact: ArtifactLoadBudget::streaming_only(6_u64 << 30, 64 << 10),
        tokenizer_and_metadata_bytes: 128 << 20, allocator_reserve_bytes: 64 << 20, run: native().run,
    }, CancellationToken::default()).unwrap();
    assert_eq!(model.artifact_identity().revision, "f56ec5a9650268aa098496734743c25ea778bd2d");
    assert_eq!(model.artifact_identity().logical_model_sha256, logical);
    assert_eq!(model.artifact_identity().source_root_sha256, source_root);
    let vocabulary = Arc::new(ExtractionVocabulary::pinned(pinned_controls::pinned().unwrap().template_controls()).unwrap());
    let baseline = host.resources().memory_snapshot().committed_bytes;
    for compact in [false, true] { for kind in 0..5 {
        let expected = execute(&host, &model, &vocabulary, baseline, kind, compact, 1, None);
        for (width, schedule) in [(1, Some(prefill(3))), (2, None), (2, Some(prefill(3)))] {
            assert_eq!(execute(&host, &model, &vocabulary, baseline, kind, compact, width, schedule),
                expected, "route={kind} compact={compact} width={width}");
            assert_eq!(Arc::strong_count(&vocabulary), 1);
        }
    } }

    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let (planner, config) = assets(model.artifact_identity(), question_synthesis(), "answer-v1", true);
        let cancellation = CancellationToken::default(); cancellation.cancel(cause);
        let error = host.synthesize_int8_document_answer_hierarchical_layer_major_with_threads(&model,
            SOURCE.to_owned(), planner, Arc::clone(&vocabulary), config, question_hierarchy(),
            Int8Kernel::scalar(), 2, prefill(3), cancellation).err().expect("cancelled hierarchy cannot publish");
        assert!(matches!(error, HostedError::Stopped { stop: RunStop { kind, .. }, .. } if kind == cause));
        assert_eq!(Arc::strong_count(&vocabulary), 1); drained(&host, baseline);
    }
    drop(vocabulary); drop(model); drop(preparation); drop(lease); drained(&host, 0);
    println!("{COMPLETE}");
}
