//! Controller-only real-artifact differential for all structured stream routes.
//! No synthetic native result, network fetch or quality claim is used here.
use super::*;
use crate::{
    canonjson, LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    grammar::{CompileLimits, mask::MaskWorkLimits, runtime::SourceRuntimeLimits},
    hosted::{LoadLimits, corpus::{SourceCorpusConfig, ExtractionDocumentCorpusConfig}},
    batch::{source::SourceBatchArgs, extract::{ExtractionBatchArgs, ExtractionBatchGrounding,
        ExtractionMaskBudget, quantized::long::{Int8ExtractionMapLimits,
            stream::ExtractionDocumentBatchConfig}}},
    native_engine::{simd::{Int8KernelKind, config::configure_process_kernel},
        strict_int8::STRICT_INT8_EXECUTION},
    tasks::{BuiltInTask, ir::TaskBudget, source_planning::{SourceTaskPlanner, SourcePlanningLimits},
        source_planning::quantized::long::Int8SourceMapLimits,
        ner::{NerOptions, EntityType}, keyphrases::KeyphraseOptions, summarize::SummaryOptions,
        answer::{AnswerOptions, AnswerPassage}, mapreduce::{ChunkLimits, ExecutionLimits}},
    tokenizer::{pinned_controls, untrusted::{SafeBpeLimits, UntrustedEncoding}},
    validation::grounded_fields::GroundingBudget,
};
use std::{io::{self, Read}, sync::{Mutex, atomic::{AtomicUsize, Ordering}}};

#[derive(Clone, Copy)]
enum Kind { Source(BuiltInTask), Extract, Document }
const SOURCE: &str = "Al visited Rome. Zo visited Oslo. café 上海 <tool_call> literal data.";

fn native() -> NativeLimits {
    NativeLimits { context_tokens: 4096, allocator_reserve_bytes: 64 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000_000,
            cleanup_reserve_bytes: 65536 } }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 3968, max_output_tokens: 128, max_output_bytes: 1 << 20,
        max_grammar_states: 8192,
        max_kv_bytes: Int8MemoryRequirement::for_context(native().context_tokens).unwrap().kv_bytes }
}
fn work() -> Int8Work { Int8Work::for_sequence(0, 65536, 4096 * NANBEIGE_VOCAB_SIZE).unwrap() }
fn masks() -> ExtractionMaskBudget {
    ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 1 << 22, checkpoint_interval_nodes: 1024 },
        max_visits_per_item: 1_000_000_000, max_visits_per_run: 32_000_000_000 }
}
fn corpus_limits() -> CorpusLimits {
    let work = work();
    CorpusLimits { native: native(), preparation_reserve_bytes: 512 << 20, io_reserve_bytes: 32 << 20,
        transport: BatchLimits { max_line_bytes: 1 << 20, max_document_bytes: 65536,
            max_requests: 16, max_input_bytes: 1 << 20, max_output_bytes: 16 << 20,
            max_output_line_bytes: 4 << 20,
            max_work: BatchWork { forward_positions: work.forward_positions,
                projected_logits: work.projected_logits }, ..BatchLimits::default() } }
}
fn encoding(compact: bool) -> UntrustedEncoding {
    if compact { UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 65536, max_output_tokens: 65536,
        max_pair_checks: 4 << 20, max_pair_bytes: 64 << 20,
    }) } else { UntrustedEncoding::ByteFallback }
}
fn identity(facts: &ArtifactIdentity, task: &str, tokenizer: Sha256Digest, template: Sha256Digest)
    -> ExecutionIdentity {
    let absent = Sha256Digest::of_bytes(b"null");
    let packing = canonjson::canonical_bytes(&("current-candidate-portable-int8-v1", Vec::<String>::new())).unwrap();
    ExecutionIdentity { schema_version: 1, source_revision: facts.revision.clone(),
        logical_model_digest: Sha256Digest::from_hex(&facts.logical_model_sha256).unwrap(),
        artifact_format: "fnlpq-current-candidate-v1".to_owned(), quant_recipe: facts.recipe_id.clone(),
        packing_set_digest: Sha256Digest::of_bytes(&packing), tokenizer_digest: tokenizer,
        template_digest: template, task_spec: task.to_owned(), taskir_digest: absent,
        prompt_digest: absent, grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent,
        backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(), host_class: None, compiler_identity: None }
}
fn source_args(kind: BuiltInTask) -> SourceBatchArgs {
    match kind {
        BuiltInTask::Ner => SourceBatchArgs::Ner { options: NerOptions {
            types: vec![EntityType::Person, EntityType::Location], max_entities: 2, max_mention_scalars: 16 },
            budget: budget() },
        BuiltInTask::Keyphrases => SourceBatchArgs::Keyphrases { options: KeyphraseOptions {
            max_phrases: 2, max_phrase_scalars: 16 }, budget: budget() },
        BuiltInTask::Summarize => SourceBatchArgs::Summarize { options: SummaryOptions {
            max_bullets: 1, max_bullet_scalars: 32, max_citations_per_bullet: 1, max_quote_scalars: 16 },
            budget: budget() },
        BuiltInTask::Answer => SourceBatchArgs::Answer { passages: vec![AnswerPassage {
            id: "original-passage".to_owned(), text: SOURCE.to_owned() }],
            options: AnswerOptions { max_answer_scalars: 32, max_citations: 1, max_quote_scalars: 16 },
            budget: budget() },
        _ => unreachable!("closed structured fixture"),
    }
}
fn extraction_args() -> ExtractionBatchArgs {
    ExtractionBatchArgs { schema: r#"{"type":"string","maxLength":8,"x-fnlp-source":"verbatim"}"#.to_owned(),
        grounding: ExtractionBatchGrounding::SourceMembership, budget: budget() }
}
fn document_config() -> ExtractionDocumentCorpusConfig {
    let verification = GroundingBudget { max_fields: 64, max_matches: 256, max_scan_steps: 1 << 20 };
    ExtractionDocumentCorpusConfig { batch: ExtractionDocumentBatchConfig {
        request: ExtractionBatchArgs { schema: r#"{"type":"boolean"}"#.to_owned(),
            grounding: ExtractionBatchGrounding::Structural, budget: budget() },
        limits: Int8ExtractionMapLimits { mapping: Int8SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 65536, max_chunk_bytes: 24, max_chunk_tokens: 1024,
                context_tokens: native().context_tokens, reserved_tokens: 1024, max_chunks: 8,
                max_tokenizer_calls: 64 },
            reduction: ExecutionLimits { map_batch_chunks: 2, reduce_fan_in: 2,
                max_value_bytes: 1 << 20, max_live_value_bytes: 8 << 20, max_result_bytes: 4 << 20,
                ..ExecutionLimits::default() }, max_model_work: work(), mask_limits: masks().per_mask,
            mask_visits_per_chunk: masks().max_visits_per_item, max_mask_visits: 8_000_000_000 },
            verification }, max_model_work: work(), max_mask_visits: masks().max_visits_per_run,
        max_verification: GroundingBudget { max_fields: 256, max_matches: 1024, max_scan_steps: 4 << 20 },
    }, reduction_reserve_bytes: 64 << 20 }
}
fn input(kind: Kind) -> Vec<u8> {
    let (text, args) = match kind {
        Kind::Source(task) => (if task == BuiltInTask::Answer { "Who visited Rome?" } else { SOURCE },
            serde_json::to_value(source_args(task)).unwrap()),
        Kind::Extract => (SOURCE, serde_json::to_value(extraction_args()).unwrap()),
        Kind::Document => (SOURCE, serde_json::json!({})),
    };
    let row = serde_json::to_vec(&serde_json::json!({"id":"reused","text":text,"task_args":args})).unwrap();
    let mut bytes = row.clone();
    bytes.extend_from_slice(b"\n{\"id\":\"invalid\",\"text\":17}\n{\"flush\":true}\n");
    bytes.extend_from_slice(&row); // Last record deliberately has no newline.
    bytes
}
struct OwnedReader { inner: Cursor<Vec<u8>>, dropped: Arc<AtomicUsize> }
impl Read for OwnedReader {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> { self.inner.read(bytes) }
}
impl BufRead for OwnedReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { self.inner.fill_buf() }
    fn consume(&mut self, amount: usize) { self.inner.consume(amount); }
}
impl Drop for OwnedReader {
    fn drop(&mut self) { self.dropped.fetch_add(1, Ordering::SeqCst); }
}
struct OwnedWriter { bytes: Arc<Mutex<Vec<u8>>>, dropped: Arc<AtomicUsize> }
impl Write for OwnedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes.lock().unwrap().extend_from_slice(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
impl Drop for OwnedWriter {
    fn drop(&mut self) { self.dropped.fetch_add(1, Ordering::SeqCst); }
}
fn drained(host: &NlpEngine, baseline: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, baseline); assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
}
fn run_case(host: &NlpEngine, model: &ResidentInt8, kind: Kind, compact: bool,
    width: usize, schedule: Option<Int8PrefillLimits>, input: &[u8]) -> (BatchSummary, Vec<u8>) {
    let reader_drops = Arc::new(AtomicUsize::new(0)); let writer_drops = Arc::new(AtomicUsize::new(0));
    let bytes = Arc::new(Mutex::new(Vec::new()));
    // Bounded test capture is covered by caller preparation and IO storage.
    let reader = OwnedReader { inner: Cursor::new(input.to_vec()), dropped: Arc::clone(&reader_drops) };
    let writer = OwnedWriter { bytes: Arc::clone(&bytes), dropped: Arc::clone(&writer_drops) };
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    let vocabulary = Arc::new(ExtractionVocabulary::pinned(controls).unwrap());
    let limits = corpus_limits(); let cancellation = CancellationToken::default(); let kernel = Int8Kernel::scalar();
    let result = match kind {
        Kind::Source(kind) => {
            let planner = Arc::new(SourceTaskPlanner::pinned(controls, eos).unwrap()
                .with_document_encoding(encoding(compact)).unwrap());
            let config = SourceCorpusConfig { identity: identity(model.artifact_identity(), &kind.spec().identity(),
                planner.tokenizer_digest(), *planner.template_digest()), task_ceiling: budget(),
                planning: SourcePlanningLimits { max_context_tokens: native().context_tokens,
                    ..SourcePlanningLimits::default() }, defaults: None,
                native_work: Int8ExtractionBatchLimits { max_model_work: work(), masks: masks() } };
            match (width, schedule) {
                (1, None) => host.batch_int8_source(model, planner, Arc::clone(&vocabulary), config,
                    limits, reader, writer, cancellation),
                (1, Some(prefill)) => host.batch_int8_source_layer_major(model, planner, Arc::clone(&vocabulary),
                    config, limits, prefill, reader, writer, cancellation),
                (_, None) => host.batch_int8_source_with_threads(model, planner, Arc::clone(&vocabulary),
                    config, limits, kernel, width, reader, writer, cancellation),
                (_, Some(prefill)) => host.batch_int8_source_layer_major_with_threads(model, planner, Arc::clone(&vocabulary),
                    config, limits, kernel, width, prefill, reader, writer, cancellation),
            }
        }
        Kind::Extract | Kind::Document => {
            let absent = Sha256Digest::of_bytes(b"uncompiled");
            let planner = Int8ExtractionBatchPlanner::pinned(controls, eos,
                identity(model.artifact_identity(), "extract-v1", absent, absent),
                budget(), CompileLimits::default(), SourceRuntimeLimits::default(), None).unwrap()
                .with_document_encoding(encoding(compact)).unwrap();
            match kind {
                Kind::Extract => {
                    let work = Int8ExtractionBatchLimits { max_model_work: work(), masks: masks() };
                    match (width, schedule) {
                        (1, None) => host.batch_int8_extract(model, planner, Arc::clone(&vocabulary), work,
                            limits, reader, writer, cancellation),
                        (1, Some(prefill)) => host.batch_int8_extract_layer_major(model, planner, Arc::clone(&vocabulary), work,
                            limits, prefill, reader, writer, cancellation),
                        (_, None) => host.batch_int8_extract_with_threads(model, planner, Arc::clone(&vocabulary), work,
                            limits, kernel, width, reader, writer, cancellation),
                        (_, Some(prefill)) => host.batch_int8_extract_layer_major_with_threads(model, planner, Arc::clone(&vocabulary), work,
                            limits, kernel, width, prefill, reader, writer, cancellation),
                    }
                }
                Kind::Document => {
                    assert!(!compact, "complete extraction streams retain their byte-only contract");
                    let planner = Arc::new(planner); let config = document_config();
                    match (width, schedule) {
                        (1, None) => host.batch_int8_extract_document(model, planner, Arc::clone(&vocabulary), config,
                            limits, reader, writer, cancellation),
                        (1, Some(prefill)) => host.batch_int8_extract_document_layer_major(model, planner, Arc::clone(&vocabulary),
                            config, limits, prefill, reader, writer, cancellation),
                        (_, None) => host.batch_int8_extract_document_with_threads(model, planner, Arc::clone(&vocabulary),
                            config, limits, kernel, width, reader, writer, cancellation),
                        (_, Some(prefill)) => host.batch_int8_extract_document_layer_major_with_threads(model, planner, Arc::clone(&vocabulary),
                            config, limits, kernel, width, prefill, reader, writer, cancellation),
                    }
                }
                Kind::Source(_) => unreachable!(),
            }
        }
    };
    assert_eq!(reader_drops.load(Ordering::SeqCst), 1);
    assert_eq!(writer_drops.load(Ordering::SeqCst), 1);
    assert_eq!(Arc::strong_count(&vocabulary), 1);
    let summary = result.expect("every admitted structured record must finish");
    let output = bytes.lock().unwrap().clone();
    (summary, output)
}
fn assert_complete(summary: BatchSummary, bytes: &[u8]) -> Int8Work {
    assert_eq!(summary.succeeded, 2); assert_eq!(summary.failed, 1);
    assert_eq!(summary.flushes, 2); assert_eq!(summary.requests, 4);
    assert!(bytes.ends_with(b"\n"));
    let events = serde_json::Deserializer::from_slice(bytes).into_iter::<serde_json::Value>()
        .collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(events.iter().filter(|e| e["event"] == "run_complete").count(), 1);
    assert_eq!(events.last().unwrap()["event"], "run_complete");
    assert_eq!(events.last().unwrap()["summary"], serde_json::to_value(summary).unwrap());
    let documents: Vec<_> = events.iter().filter(|e| e["event"] == "doc" && e.get("result").is_some()).collect();
    assert_eq!(documents.iter().map(|e| e["request_seq"].as_u64().unwrap()).collect::<Vec<_>>(), [1, 4]);
    assert_eq!(documents[0]["caller_id"], documents[1]["caller_id"]);
    assert_eq!(documents[0]["epoch"], 1); assert_eq!(documents[1]["epoch"], 2);
    let mut work = Int8Work::default();
    for document in documents {
        let actual: Int8Work = serde_json::from_value(document["result"]["model_work"].clone()).unwrap();
        assert!(actual.forward_positions > 0 && actual.projected_logits > 0 && actual.attention_pairs > 0);
        assert!(actual.projections.dot_products > 0 && actual.projections.multiply_accumulates > 0);
        // Structured early EOS can use less work than the nonrefundable cap.
        assert!(actual.forward_positions <= document["reserved_work"]["forward_positions"].as_u64().unwrap());
        assert!(actual.projected_logits <= document["reserved_work"]["projected_logits"].as_u64().unwrap());
        work = work.checked_add(actual).unwrap();
    }
    assert!(work.forward_positions <= summary.reserved_work.forward_positions);
    assert!(work.projected_logits <= summary.reserved_work.projected_logits);
    work
}

#[test]
#[ignore = "controller-only independently pinned INT8 artifact and complete structured streams; execution evidence pending"]
fn all_structured_streams_preserve_complete_results_and_work_across_prefill_and_persistent_teams() {
    const CHILD: &str = "FNLP_INT8_STRUCTURED_CORPUS_TEAM_CHILD";
    const COMPLETE: &str = "FNLP_INT8_STRUCTURED_CORPUS_TEAM_COMPLETE";
    let full_name = concat!(module_path!(),
        "::all_structured_streams_preserve_complete_results_and_work_across_prefill_and_persistent_teams");
    let name = full_name.split_once("::").expect("crate-qualified test name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated real-model gate must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "isolated real-model gate failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "exact filter did not finish the model gate");
        return;
    }
    assert!(crate::installed_process_resources().is_none());
    configure_process_kernel(Int8KernelKind::Scalar).unwrap();
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_STRUCTURED_CORPUS_TEAM_MODEL")
        .expect("explicit admitted local INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_STRUCTURED_CORPUS_TEAM_LOGICAL_SHA256")
        .expect("independent logical-model pin required");
    let source = std::env::var("FNLP_INT8_STRUCTURED_CORPUS_TEAM_SOURCE_ROOT_SHA256")
        .expect("independent source-root pin required");
    Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source).unwrap();
    let memory: u64 = std::env::var("FNLP_INT8_STRUCTURED_CORPUS_TEAM_MEMORY_MIB")
        .expect("controller-admitted process memory required").parse().expect("memory must be MiB");
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5,
        memory_ceiling_bytes: memory.checked_mul(1 << 20).expect("memory arithmetic"),
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let lease = host.resources().acquire_lease();
    let caller_preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, 512 << 20).unwrap().commit().unwrap();
    let model = host.load_current_candidate_int8(path, LoadLimits {
        artifact: ArtifactLoadBudget::streaming_only(6_u64 << 30, 64 << 10),
        tokenizer_and_metadata_bytes: 128 << 20, allocator_reserve_bytes: 64 << 20, run: native().run,
    }, CancellationToken::default()).unwrap();
    assert_eq!(model.artifact_identity().logical_model_sha256, logical);
    assert_eq!(model.artifact_identity().source_root_sha256, source);
    assert_eq!(model.artifact_identity().revision, "f56ec5a9650268aa098496734743c25ea778bd2d");
    let baseline = host.resources().memory_snapshot().committed_bytes;
    for kind in [Kind::Source(BuiltInTask::Ner), Kind::Source(BuiltInTask::Keyphrases),
        Kind::Source(BuiltInTask::Summarize), Kind::Source(BuiltInTask::Answer), Kind::Extract, Kind::Document] {
        for compact in [false, true] {
            if matches!(kind, Kind::Document) && compact { continue; }
            let input = input(kind);
            let (summary, expected) = run_case(&host, &model, kind, compact, 1, None, &input);
            let expected_work = assert_complete(summary, &expected); drained(&host, baseline);
            for (width, schedule) in [(1, Some(prefill(3))), (2, None), (2, Some(prefill(3)))] {
                let (actual_summary, actual) = run_case(&host, &model, kind, compact, width, schedule, &input);
                assert_eq!(actual_summary, summary);
                assert_eq!(assert_complete(actual_summary, &actual), expected_work);
                assert_eq!(actual, expected, "retain all values, coordinates, masks, IDs, work counters, errors and footer");
                drained(&host, baseline);
            }
        }
    }
    drop(model); drop(caller_preparation); drop(lease); drained(&host, 0);
    println!("{COMPLETE}");
}
