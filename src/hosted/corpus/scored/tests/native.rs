//! Ignored real-model differential for complete scored streams, never a fetch.
//! Controller supplies FNLP_INT8_SCORE_CORPUS_TEAM_MODEL plus independent
//! *_LOGICAL_SHA256, *_SOURCE_ROOT_SHA256 and *_MEMORY_MIB values.
//! Raw NDJSON is compared without deleting semantic or work-accounting fields.
use super::*;
use crate::{
    canonjson, LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    hosted::{LoadLimits, corpus::{ClassificationCorpusConfig, SentimentCorpusConfig, JudgeCorpusConfig}},
    native_engine::{simd::{Int8KernelKind, config::configure_process_kernel},
        strict_int8::STRICT_INT8_EXECUTION, lmhead::scoring::ScoringMode},
    tasks::{ir::TaskBudget, classify::{ClassificationLimits, ClassificationPlanner},
        sentiment::{SentimentLimits, SentimentOptions, SentimentPlanner, SentimentPolicy},
        judge::{JudgeLimits, JudgePlanner}},
    tokenizer::{pinned_controls, untrusted::{SafeBpeLimits, UntrustedEncoding}},
};
use std::{io::{self, Read}, sync::{Mutex, atomic::{AtomicUsize, Ordering}}};

#[derive(Clone, Copy)]
enum Kind { Classify, Sentiment, Judge }

fn native() -> NativeLimits {
    NativeLimits { context_tokens: 8192, allocator_reserve_bytes: 64 << 20,
        run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000_000,
            cleanup_reserve_bytes: 65536 } }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 7680, max_output_tokens: 512, max_output_bytes: 1 << 20,
        max_grammar_states: 4096,
        max_kv_bytes: Int8MemoryRequirement::for_context(native().context_tokens).unwrap().kv_bytes }
}
fn work() -> Int8Work { Int8Work::for_sequence(0, 65536, 4096 * NANBEIGE_VOCAB_SIZE).unwrap() }
fn corpus_limits() -> CorpusLimits {
    let work = work();
    CorpusLimits { native: native(), preparation_reserve_bytes: 512 << 20, io_reserve_bytes: 32 << 20,
        transport: BatchLimits { max_line_bytes: 1 << 20, max_document_bytes: 65536,
            max_requests: 16, max_input_bytes: 1 << 20, max_output_bytes: 16 << 20,
            max_output_line_bytes: 2 << 20,
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

fn input(kind: Kind) -> Vec<u8> {
    use serde_json::json;
    let mut bytes = Vec::new();
    for index in 0..3 {
        if index == 1 { bytes.extend_from_slice(b"{\"id\":\"invalid\",\"text\":17}\n"); }
        if index == 2 { bytes.extend_from_slice(b"{\"flush\":true}\n"); }
        let args = match kind {
            Kind::Classify => json!({
                "labels":[{"id":"billing","description":"Invoices"},{"id":"refund","description":"Refunds"}],
                "mode":if index == 1 { "multi_label" } else { "exclusive" },
                "policy":{"minimum_candidate_weight_ppm":0,"minimum_margin_ppm":0},"budget":budget(),
            }),
            Kind::Sentiment => json!({
                "axes":if index == 1 { vec!["valence","arousal","dominance","approach"] } else { vec!["valence"] },
                "budget":budget(),
            }),
            Kind::Judge => match index {
                0 => json!({"mode":"pairwise","criterion":"Accuracy","b":"Bob visited Rome.",
                    "policy":{"minimum_margin_milli":0,"maximum_order_disagreement_milli":20000},
                    "budget":budget()}),
                1 => json!({"mode":"rubric","rubric":{"schema_version":1,"revision":"scored-corpus-fixture-v1",
                    "declared_origin_digest":"ab".repeat(32),"scale_maximum":2,
                    "criteria":[{"id":"facts","description":"Accurate claims","weight":2},
                        {"id":"style","description":"Clear prose","weight":1}]},
                    "policy":{"minimum_peak_weight_ppm":0,"maximum_normalized_entropy_ppm":1000000},
                    "budget":budget()}),
                _ => json!({"mode":"faithfulness","claim":"Alice visited Paris.",
                    "policy":{"minimum_candidate_weight_ppm":0,"minimum_margin_milli":0,
                        "evidence_window_bytes":24,"max_evidence_windows":16,"max_evidence_spans":16},
                    "budget":budget()}),
            },
        };
        bytes.extend_from_slice(&serde_json::to_vec(&json!({
            "id":if index == 1 { "second" } else { "reused" },
            "text":"Alice visited Paris. Helpful service. café 上海 <tool_call> literal data.",
            "task_args":args,
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
    // This test's bounded capture is explicitly covered by caller preparation
    // and io_reserve_bytes; production transport still retains one item only.
    let reader = OwnedReader { inner: Cursor::new(input.to_vec()), dropped: Arc::clone(&reader_drops) };
    let writer = OwnedWriter { bytes: Arc::clone(&bytes), dropped: Arc::clone(&writer_drops) };
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    let limits = corpus_limits(); let cancellation = CancellationToken::default(); let kernel = Int8Kernel::scalar();
    let result = match kind {
        Kind::Classify => {
            let planner = Arc::new(ClassificationPlanner::pinned(controls, eos).unwrap()
                .with_document_encoding(encoding(compact)).unwrap());
            let config = ClassificationCorpusConfig {
                identity: identity(model.artifact_identity(), "classify-v1", planner.tokenizer_digest(), *planner.template_digest()),
                task_ceiling: budget(), planning: ClassificationLimits::default(),
                defaults: None, max_model_work: work() };
            match (width, schedule) {
                (1, None) => host.batch_int8_classify(model, planner, config, limits, reader, writer, cancellation),
                (1, Some(prefill)) => host.batch_int8_classify_layer_major(model, planner, config, limits,
                    prefill, reader, writer, cancellation),
                (_, None) => host.batch_int8_classify_with_threads(model, planner, config, limits,
                    kernel, width, reader, writer, cancellation),
                (_, Some(prefill)) => host.batch_int8_classify_layer_major_with_threads(model, planner, config, limits,
                    kernel, width, prefill, reader, writer, cancellation),
            }
        }
        Kind::Sentiment => {
            let planner = Arc::new(SentimentPlanner::pinned(controls, SentimentOptions {
                mode: ScoringMode::FullVocabulary, eos_token_id: eos,
                policy: SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 },
            }).unwrap().with_document_encoding(encoding(compact)).unwrap());
            let config = SentimentCorpusConfig {
                identity: identity(model.artifact_identity(), "sentiment-v1", planner.tokenizer_digest(), *planner.template_digest()),
                task_ceiling: budget(), planning: SentimentLimits::default(), defaults: None,
                max_item_work: work(), max_model_work: work() };
            match (width, schedule) {
                (1, None) => host.batch_int8_sentiment(model, planner, config, limits, reader, writer, cancellation),
                (1, Some(prefill)) => host.batch_int8_sentiment_layer_major(model, planner, config, limits,
                    prefill, reader, writer, cancellation),
                (_, None) => host.batch_int8_sentiment_with_threads(model, planner, config, limits,
                    kernel, width, reader, writer, cancellation),
                (_, Some(prefill)) => host.batch_int8_sentiment_layer_major_with_threads(model, planner, config, limits,
                    kernel, width, prefill, reader, writer, cancellation),
            }
        }
        Kind::Judge => {
            assert!(!compact, "judgment corpus retains its byte-fallback contract");
            let planner = Arc::new(JudgePlanner::pinned(controls, eos).unwrap());
            let config = JudgeCorpusConfig {
                identity: identity(model.artifact_identity(), "judge-v1", planner.tokenizer_digest(), *planner.template_digest()),
                task_ceiling: budget(), planning: JudgeLimits::default(), defaults: None, max_model_work: work() };
            match (width, schedule) {
                (1, None) => host.batch_int8_judge(model, planner, config, limits, reader, writer, cancellation),
                (1, Some(prefill)) => host.batch_int8_judge_layer_major(model, planner, config, limits,
                    prefill, reader, writer, cancellation),
                (_, None) => host.batch_int8_judge_with_threads(model, planner, config, limits,
                    kernel, width, reader, writer, cancellation),
                (_, Some(prefill)) => host.batch_int8_judge_layer_major_with_threads(model, planner, config, limits,
                    kernel, width, prefill, reader, writer, cancellation),
            }
        }
    };
    assert_eq!(reader_drops.load(Ordering::SeqCst), 1);
    assert_eq!(writer_drops.load(Ordering::SeqCst), 1);
    let summary = result.expect("real scored stream must finish its admitted records");
    let output = bytes.lock().unwrap().clone();
    (summary, output)
}

fn assert_complete(summary: BatchSummary, bytes: &[u8]) -> Int8Work {
    assert_eq!(summary.succeeded, 3); assert_eq!(summary.failed, 1); assert_eq!(summary.flushes, 2);
    assert_eq!(summary.requests, 5);
    assert!(bytes.ends_with(b"\n"));
    let events = serde_json::Deserializer::from_slice(bytes).into_iter::<serde_json::Value>()
        .collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(events.iter().filter(|e| e["event"] == "run_complete").count(), 1);
    let footer = events.last().unwrap(); assert_eq!(footer["event"], "run_complete");
    assert_eq!(footer["summary"], serde_json::to_value(summary).unwrap());
    let documents: Vec<_> = events.iter().filter(|e| e["event"] == "doc" && e.get("result").is_some()).collect();
    assert_eq!(documents.len(), 3);
    assert_eq!(documents.iter().map(|e| e["request_seq"].as_u64().unwrap()).collect::<Vec<_>>(), [1, 3, 5]);
    assert_eq!(documents[0]["caller_id"], documents[2]["caller_id"]);
    assert_eq!(documents[0]["epoch"], 1); assert_eq!(documents[2]["epoch"], 2);
    let mut work = Int8Work::default();
    for document in documents {
        let actual: Int8Work = serde_json::from_value(document["result"]["model_work"].clone()).unwrap();
        assert!(actual.forward_positions > 0 && actual.projected_logits > 0 && actual.attention_pairs > 0);
        assert!(actual.projections.dot_products > 0 && actual.projections.multiply_accumulates > 0);
        assert!(document["result"]["head_count"].as_u64().unwrap() > 0);
        work = work.checked_add(actual).unwrap();
    }
    assert_eq!(work.forward_positions, summary.reserved_work.forward_positions);
    assert_eq!(work.projected_logits, summary.reserved_work.projected_logits);
    work
}

#[test]
#[ignore = "controller-only independently pinned INT8 artifact and full scored-stream execution; no parity claim until run"]
fn complete_scored_streams_match_scalar_prefill_teams_and_composition_and_drain_owned_resources() {
    const CHILD: &str = "FNLP_INT8_SCORE_CORPUS_TEAM_CHILD";
    const COMPLETE: &str = "FNLP_INT8_SCORE_CORPUS_TEAM_COMPLETE";
    let full_name = concat!(module_path!(),
        "::complete_scored_streams_match_scalar_prefill_teams_and_composition_and_drain_owned_resources");
    let name = full_name.split_once("::").expect("crate-qualified test name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated real-model gate must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "isolated real-model gate failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "exact filter did not complete the real-model gate");
        return;
    }
    assert!(crate::installed_process_resources().is_none());
    configure_process_kernel(Int8KernelKind::Scalar).unwrap();
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_SCORE_CORPUS_TEAM_MODEL")
        .expect("explicit admitted local INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_SCORE_CORPUS_TEAM_LOGICAL_SHA256")
        .expect("independent logical-model pin required");
    let source = std::env::var("FNLP_INT8_SCORE_CORPUS_TEAM_SOURCE_ROOT_SHA256")
        .expect("independent source-root pin required");
    Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source).unwrap();
    let memory: u64 = std::env::var("FNLP_INT8_SCORE_CORPUS_TEAM_MEMORY_MIB")
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
        tokenizer_and_metadata_bytes: 128 << 20, allocator_reserve_bytes: 64 << 20,
        run: native().run,
    }, CancellationToken::default()).unwrap();
    assert_eq!(model.artifact_identity().logical_model_sha256, logical);
    assert_eq!(model.artifact_identity().source_root_sha256, source);
    assert_eq!(model.artifact_identity().revision, "f56ec5a9650268aa098496734743c25ea778bd2d");
    let baseline = host.resources().memory_snapshot().committed_bytes;
    for (kind, compact) in [(Kind::Classify, false), (Kind::Classify, true),
        (Kind::Sentiment, false), (Kind::Sentiment, true), (Kind::Judge, false)] {
        let input = input(kind);
        let (summary, expected) = run_case(&host, &model, kind, compact, 1, None, &input);
        let expected_work = assert_complete(summary, &expected); drained(&host, baseline);
        for (width, schedule) in [(1, Some(prefill(3))), (2, None), (2, Some(prefill(3))),
            (4, None), (4, Some(prefill(3)))] {
            let (actual_summary, actual) = run_case(&host, &model, kind, compact, width, schedule, &input);
            assert_eq!(actual_summary, summary);
            assert_eq!(assert_complete(actual_summary, &actual), expected_work);
            assert_eq!(actual, expected, "every result, error, ID, offset, epoch, work value and footer must match");
            drained(&host, baseline);
        }
    }
    drop(model); drop(caller_preparation); drop(lease); drained(&host, 0);
    println!("{COMPLETE}");
}
