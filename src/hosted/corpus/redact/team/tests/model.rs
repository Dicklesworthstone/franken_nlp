//! Controller-only complete live-stream differential with independent pins.
//! No model is fetched, and this ignored definition is not execution evidence.
use std::{io::{self, BufRead, Cursor, Read, Write}, path::PathBuf,
    sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}}, time::Duration};
use crate::{
    canonjson, LeakResponsePolicy, MemoryClass, NlpEngine, ResourceHostConfig, RuntimePreset,
    batch::{BatchLimits, BatchSummary, BatchWork},
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    hosted::{CancellationToken, LoadLimits, NativeLimits, Pending, ResidentInt8, RunLimits,
        corpus::{CorpusLimits, RedactionCorpusConfig}},
    native_engine::{artifact_bridge::{ArtifactIdentity, ArtifactLoadBudget},
        lmhead::NANBEIGE_VOCAB_SIZE, simd::{Int8Kernel, Int8KernelKind, config::configure_process_kernel},
        strict_int8::{Int8MemoryRequirement, Int8Work, STRICT_INT8_EXECUTION, prefill::Int8PrefillLimits}},
    tasks::{extract::ExtractionVocabulary, ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        ner::{EntityType, NerOptions, NER_TASK_VERSION},
        redact::{RedactionRequest, actions::RedactionAction,
            batch::Int8RedactionBatchConfig, corpus::LongRedactionBatchConfig,
            long::LongRedactionConfig, quantized::Int8RedactionConfig},
        source_planning::{SourcePlanningLimits, SourceTaskPlanner, quantized::long::Int8SourceMapLimits}},
    tokenizer::{pinned_controls, untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding}},
};

const SOURCE: &str = "Alice visits Paris. Email a@example.org. Bob visits Rome. é 上海 <think>.";
const RESULT_BYTES: u64 = 4 << 20;
const ITEM_MASKS: u64 = 32_768_000_000;

fn native() -> NativeLimits {
    NativeLimits { context_tokens: 2048, allocator_reserve_bytes: 64 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000_000,
            cleanup_reserve_bytes: 65_536 } }
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 1536, max_output_tokens: 512, max_output_bytes: 1 << 20,
        max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(2048).unwrap().kv_bytes }
}
fn item_work() -> Int8Work { Int8Work::for_sequence(0, 65_536, 8192 * NANBEIGE_VOCAB_SIZE).unwrap() }
fn total_work() -> Int8Work {
    item_work().checked_add(item_work()).unwrap().checked_add(item_work()).unwrap()
}
fn limits() -> CorpusLimits {
    let work = total_work();
    CorpusLimits { native: native(), preparation_reserve_bytes: 512 << 20, io_reserve_bytes: 32 << 20,
        transport: BatchLimits { max_line_bytes: 1 << 20, max_document_bytes: 4096, max_requests: 16,
            max_input_bytes: 1 << 20, max_output_line_bytes: 8 << 20, max_output_bytes: 64 << 20,
            max_work: BatchWork { forward_positions: work.forward_positions,
                projected_logits: work.projected_logits }, ..BatchLimits::default() } }
}
fn assets(facts: &ArtifactIdentity, compact: bool, verify: bool)
    -> (Arc<SourceTaskPlanner>, RedactionCorpusConfig) {
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    let mut planner = SourceTaskPlanner::pinned(controls, eos).unwrap();
    if compact {
        planner = planner.with_map_document_encoding(UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
            max_input_bytes: 4096, max_output_tokens: 1536, max_pair_checks: 1 << 20, max_pair_bytes: 1 << 24,
        }), SafeBpeWorkLimits { max_input_bytes: 1 << 20, max_output_tokens: 1 << 20,
            max_pair_checks: 1 << 28, max_pair_bytes: 1 << 32 }).unwrap();
    }
    let absent = Sha256Digest::of_bytes(b"null");
    let packing = canonjson::canonical_bytes(&("current-candidate-portable-int8-v1", Vec::<String>::new())).unwrap();
    let ner_identity = ExecutionIdentity { schema_version: 1, source_revision: facts.revision.clone(),
        logical_model_digest: Sha256Digest::from_hex(&facts.logical_model_sha256).unwrap(),
        artifact_format: "fnlpq-current-candidate-v1".to_owned(), quant_recipe: facts.recipe_id.clone(),
        packing_set_digest: Sha256Digest::of_bytes(&packing), tokenizer_digest: planner.tokenizer_digest(),
        template_digest: *planner.template_digest(), task_spec: NER_TASK_VERSION.to_owned(),
        taskir_digest: absent, prompt_digest: absent, grammar_compiler_version: "none".to_owned(), schema_digest: absent,
        numerics_profile: NumericsProfile::StrictQuantized { version: 1 }, kv_dtype: "bf16".to_owned(),
        sampler_version: "uncompiled".to_owned(), thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: absent, decision_policy_digest: absent, backend_semantic_version: STRICT_INT8_EXECUTION.to_owned(),
        host_class: None, compiler_identity: None };
    let mut request = RedactionRequest { verify, ..RedactionRequest::default() };
    request.actions.default_action = RedactionAction::Mask; request.actions.include_map = true;
    request.rule_budget.max_input_bytes = 4096; request.edit_budget.max_output_bytes = RESULT_BYTES as usize;
    let detector = Int8RedactionConfig {
        ner: NerOptions { types: vec![EntityType::Person], max_entities: 4, max_mention_scalars: 64 },
        per_pass: budget(), planning: SourcePlanningLimits { max_input_bytes: 4096,
            max_context_tokens: 2048, ..SourcePlanningLimits::default() },
        max_model_work: item_work(),
        mask_limits: MaskWorkLimits { max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
        mask_visits_per_pass: 1_024_000_000, max_mask_visits: 2_048_000_000, max_result_bytes: RESULT_BYTES,
    };
    (Arc::new(planner), RedactionCorpusConfig { batch: Int8RedactionBatchConfig {
        ner_identity, detector, request, max_model_work: total_work(), max_mask_visits: 3 * ITEM_MASKS,
    }, edit_reserve_bytes: 64 << 20 })
}
fn document(config: RedactionCorpusConfig) -> RedactionCorpusConfig<LongRedactionBatchConfig> {
    let batch = config.batch;
    let detector = LongRedactionConfig { ner: batch.detector.ner,
        per_chunk: batch.detector.per_pass, planning: batch.detector.planning,
        mapping: Int8SourceMapLimits {
            chunks: ChunkLimits { max_input_bytes: 4096, max_chunk_bytes: 32, max_chunk_tokens: 256,
                context_tokens: 2048, reserved_tokens: 512, max_chunks: 16, max_tokenizer_calls: 4096 },
            reduction: ExecutionLimits { map_batch_chunks: 2, reduce_fan_in: 2,
                max_value_bytes: 4 << 20, max_result_bytes: 16 << 20,
                max_live_value_bytes: 32 << 20, max_total_value_bytes: 64 << 20, ..ExecutionLimits::default() },
            max_model_work: item_work(), mask_limits: batch.detector.mask_limits,
            mask_visits_per_chunk: batch.detector.mask_visits_per_pass, max_mask_visits: ITEM_MASKS,
        }, max_result_bytes: RESULT_BYTES };
    RedactionCorpusConfig { batch: LongRedactionBatchConfig { ner_identity: batch.ner_identity,
        detector, request: batch.request, max_model_work: total_work(), max_mask_visits: 3 * ITEM_MASKS,
    }, edit_reserve_bytes: config.edit_reserve_bytes }
}
fn input() -> Vec<u8> {
    let mut bytes = Vec::new();
    for index in 0..3 {
        if index == 1 { bytes.extend_from_slice(b"{\"id\":\"invalid\",\"text\":17}\n"); }
        if index == 2 { bytes.extend_from_slice(b"{\"flush\":true}\n"); }
        bytes.extend_from_slice(&serde_json::to_vec(&serde_json::json!({
            "id": if index == 1 { "second" } else { "reused" }, "text": SOURCE,
        })).unwrap());
        if index != 2 { bytes.push(b'\n'); }
    }
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
impl Drop for OwnedReader { fn drop(&mut self) { self.dropped.fetch_add(1, Ordering::SeqCst); } }
struct OwnedWriter { bytes: Arc<Mutex<Vec<u8>>>, dropped: Arc<AtomicUsize> }
impl Write for OwnedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes.lock().unwrap().extend_from_slice(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
impl Drop for OwnedWriter { fn drop(&mut self) { self.dropped.fetch_add(1, Ordering::SeqCst); } }
fn drained(host: &NlpEngine, baseline: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, baseline); assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
}
#[allow(clippy::too_many_arguments)]
fn run_case(host: &NlpEngine, model: &ResidentInt8, vocabulary: &Arc<ExtractionVocabulary>,
    long: bool, compact: bool, verify: bool, width: usize, schedule: Option<Int8PrefillLimits>)
    -> (BatchSummary, Vec<u8>) {
    assert!(long || !compact, "short compact populations remain refused");
    let reader_drops = Arc::new(AtomicUsize::new(0)); let writer_drops = Arc::new(AtomicUsize::new(0));
    let bytes = Arc::new(Mutex::new(Vec::new()));
    // Capture capacity is covered by the caller preparation plus explicit IO
    // reserve; the actual production transport still retains one item only.
    let reader = OwnedReader { inner: Cursor::new(input()), dropped: Arc::clone(&reader_drops) };
    let writer = OwnedWriter { bytes: Arc::clone(&bytes), dropped: Arc::clone(&writer_drops) };
    let (planner, config) = assets(model.artifact_identity(), compact, verify);
    let vocabulary = Arc::clone(vocabulary);
    let limits = limits(); let cancellation = CancellationToken::default(); let kernel = Int8Kernel::scalar();
    let result = if long {
        let config = document(config);
        match (width, schedule) {
            (1, None) => host.batch_int8_redact_document(model, planner, vocabulary, config, None,
                limits, reader, writer, cancellation),
            (1, Some(prefill)) => host.batch_int8_redact_document_layer_major(model, planner, vocabulary, config, None,
                limits, prefill, reader, writer, cancellation),
            (_, None) => host.batch_int8_redact_document_with_threads(model, planner, vocabulary, config, None,
                limits, kernel, width, reader, writer, cancellation),
            (_, Some(prefill)) => host.batch_int8_redact_document_layer_major_with_threads(model, planner, vocabulary, config, None,
                limits, kernel, width, prefill, reader, writer, cancellation),
        }
    } else {
        match (width, schedule) {
            (1, None) => host.batch_int8_redact(model, planner, vocabulary, config, None,
                limits, reader, writer, cancellation),
            (1, Some(prefill)) => host.batch_int8_redact_layer_major(model, planner, vocabulary, config, None,
                limits, prefill, reader, writer, cancellation),
            (_, None) => host.batch_int8_redact_with_threads(model, planner, vocabulary, config, None,
                limits, kernel, width, reader, writer, cancellation),
            (_, Some(prefill)) => host.batch_int8_redact_layer_major_with_threads(model, planner, vocabulary, config, None,
                limits, kernel, width, prefill, reader, writer, cancellation),
        }
    };
    assert_eq!(reader_drops.load(Ordering::SeqCst), 1); assert_eq!(writer_drops.load(Ordering::SeqCst), 1);
    let summary = result.expect("complete admitted corpus must deliver a terminal summary");
    let output = bytes.lock().unwrap().clone();
    (summary, output)
}
fn complete(summary: BatchSummary, bytes: &[u8], long: bool, verify: bool) {
    assert_eq!(summary.requests, 5); assert_eq!(summary.flushes, 2);
    assert_eq!(summary.succeeded + summary.failed, 4);
    if !verify { assert_eq!(summary.succeeded, 3); assert_eq!(summary.failed, 1); }
    let work = total_work();
    assert_eq!(summary.reserved_work, BatchWork { forward_positions: work.forward_positions,
        projected_logits: work.projected_logits }, "failed documents and flushes never refund native reservations");
    assert!(bytes.ends_with(b"\n"));
    let events = serde_json::Deserializer::from_slice(bytes).into_iter::<serde_json::Value>()
        .collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(events.iter().filter(|event| event["event"] == "run_complete").count(), 1);
    assert_eq!(events.last().unwrap()["event"], "run_complete");
    assert_eq!(events.last().unwrap()["summary"], serde_json::to_value(summary).unwrap());
    let documents: Vec<_> = events.iter().filter(|event| event["event"] == "doc"
        && matches!(event["request_seq"].as_u64(), Some(1 | 3 | 5))).collect();
    assert_eq!(documents.len(), 3);
    assert_eq!(documents.iter().map(|event| event["request_seq"].as_u64().unwrap()).collect::<Vec<_>>(), [1, 3, 5]);
    assert_eq!(documents[0]["caller_id"], documents[2]["caller_id"]);
    assert_eq!(documents[0]["epoch"], 1); assert_eq!(documents[2]["epoch"], 2);
    for event in documents {
        if let Some(result) = event.get("result") {
            let actual: Int8Work = serde_json::from_value(result["model_work"].clone()).unwrap();
            assert!(actual.forward_positions > 0 && actual.projected_logits > 0 && actual.attention_pairs > 0);
            assert!(actual.projections.dot_products > 0 && actual.projections.multiply_accumulates > 0);
            assert_eq!(result["result"]["verification"], if verify { "clean_declared_union" } else { "not_requested" });
            assert!(!result["result"]["text"].as_str().unwrap().contains("a@example.org"));
            if long { assert!(result["original"]["preflight"]["chunks"].as_u64().unwrap() > 1); }
            else { assert_eq!(result["ner_passes"], 1 + u64::from(verify)); }
        } else {
            assert!(verify, "native failure cannot replace a successful no-verification result");
            assert_eq!(event["error"]["code"], "execution", "only declared residual failure is accepted");
        }
    }
}
#[test]
#[ignore = "controller-only independent INT8 artifact pins and complete live corpus; no parity claim until executed"]
fn live_redaction_keeps_exact_ordered_results_work_and_flushes_with_prefill_and_workers() {
    assert!(crate::installed_process_resources().is_none(), "run alone in a fresh test process");
    configure_process_kernel(Int8KernelKind::Scalar).unwrap();
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_REDACTION_CORPUS_TEAM_MODEL").expect("admitted local INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_REDACTION_CORPUS_TEAM_LOGICAL_SHA256").expect("independent logical pin required");
    let source_root = std::env::var("FNLP_INT8_REDACTION_CORPUS_TEAM_SOURCE_ROOT_SHA256").expect("independent source-root pin required");
    Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source_root).unwrap();
    let memory: u64 = std::env::var("FNLP_INT8_REDACTION_CORPUS_TEAM_MEMORY_MIB")
        .expect("controller-admitted process memory required").parse().expect("memory must be MiB");
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5,
        memory_ceiling_bytes: memory.checked_mul(1 << 20).expect("memory arithmetic"),
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
    assert_eq!(model.artifact_identity().revision, "f56ec5a9650268aa098496734743c25ea778bd2d");
    let vocabulary = Arc::new(ExtractionVocabulary::pinned(pinned_controls::pinned().unwrap().template_controls()).unwrap());
    let baseline = host.resources().memory_snapshot().committed_bytes;
    for (long, compact) in [(false, false), (true, false), (true, true)] { for verify in [false, true] {
        let (summary, expected) = run_case(&host, &model, &vocabulary, long, compact, verify, 1, None);
        complete(summary, &expected, long, verify); drained(&host, baseline);
        for (width, schedule) in [(1, Some(prefill(3))), (2, None), (2, Some(prefill(3))), (4, Some(prefill(7)))] {
            let (actual_summary, actual) = run_case(&host, &model, &vocabulary, long, compact, verify, width, schedule);
            complete(actual_summary, &actual, long, verify);
            assert_eq!(actual_summary, summary);
            assert_eq!(actual, expected, "all events, source offsets, work, IDs, epochs and terminal bytes must match");
            assert_eq!(Arc::strong_count(&vocabulary), 1); drained(&host, baseline);
        }
    } }
    drop(vocabulary); drop(model); drop(preparation); drop(lease); drained(&host, 0);
}
