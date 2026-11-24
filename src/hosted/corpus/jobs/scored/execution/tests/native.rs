//! Controller-only actual-artifact stored-result and explicit-resume gate.
//! Requires a supplied model, independent pins, admitted memory and protected
//! parent directory. No downloads, fake neural output or parity claim if UNRUN.
use super::*;
use crate::{
    canonjson, LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    execution_identity::{NumericsProfile, ThinkingMode, ToolMode},
    hosted::LoadLimits,
    jobs::{JobContract, OwnedJob, runner::OWNED_BATCH_PROTOCOL},
    native_engine::{simd::{Int8KernelKind, config::configure_process_kernel},
        strict_int8::STRICT_INT8_EXECUTION, lmhead::scoring::ScoringMode},
    tasks::{classify::ClassificationLimits,
        sentiment::{SentimentLimits, SentimentOptions, SentimentPolicy},
        judge::{JudgeLimits, JudgePlanner}},
    tokenizer::pinned_controls,
};
use std::{fs, io::{self, Cursor, Read}, os::unix::fs::DirBuilderExt,
    path::Path, sync::atomic::{AtomicUsize, Ordering}, time::SystemTime};

#[derive(Clone, Copy, Debug)]
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
fn lifetime() -> JobLimits {
    let mut model = Int8Work::default();
    for _ in 0..8 { model = model.checked_add(work()).unwrap(); }
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: 65536,
        max_snapshot_bytes: 1 << 20, max_result_bytes: 1 << 20,
        max_spool_bytes: 16 << 20, max_materialized_bytes: 16 << 20,
        max_journal_bytes: 1 << 20, max_attempts: 64,
        max_work: JobWork { model, mask_node_visits: 0 } }
}
fn host_limits() -> JobHostLimits {
    JobHostLimits { native: native(),
        transport: PopulationReadLimits { max_stream_bytes: 1 << 20, max_lines: 3 },
        preparation_reserve_bytes: 512 << 20, io_reserve_bytes: 32 << 20,
        journal_reserve_bytes: 64 << 20, serialization_reserve_bytes: 64 << 20 }
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
fn classification(model: &ResidentInt8) -> (Arc<ClassificationPlanner>, ClassificationCorpusConfig) {
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    let planner = Arc::new(ClassificationPlanner::pinned(controls, eos).unwrap());
    let config = ClassificationCorpusConfig {
        identity: identity(model.artifact_identity(), "classify-v1", planner.tokenizer_digest(), *planner.template_digest()),
        task_ceiling: budget(), planning: ClassificationLimits::default(),
        defaults: None, max_model_work: work() };
    (planner, config)
}
fn input(kind: Kind) -> Vec<u8> {
    use serde_json::json;
    let mut bytes = Vec::new();
    for index in 0..3 {
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
                1 => json!({"mode":"rubric","rubric":{"schema_version":1,"revision":"scored-job-fixture-v1",
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
            "id":format!("item-{index}"),
            "text":"Alice visited Paris. Helpful service. café 上海 <tool_call> literal data.",
            "task_args":args,
        })).unwrap());
        if index != 2 { bytes.push(b'\n'); }
    }
    bytes
}

// A retained-input replay must not read, consume, or copy the supplied source.
// The owned reader's drop is observed only after the hosted call has returned.
struct OwnedReader { inner: Option<Cursor<Vec<u8>>>, dropped: Arc<AtomicUsize> }
impl Read for OwnedReader {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.inner.as_mut().expect("retained replay read supplied input").read(bytes)
    }
}
impl BufRead for OwnedReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.inner.as_mut().expect("retained replay read supplied input").fill_buf()
    }
    fn consume(&mut self, amount: usize) {
        self.inner.as_mut().expect("retained replay consumed supplied input").consume(amount);
    }
}
impl Drop for OwnedReader {
    fn drop(&mut self) { self.dropped.fetch_add(1, Ordering::SeqCst); }
}
fn drained(host: &NlpEngine, baseline: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, baseline); assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.recorded_leaks, 0);
    assert_eq!(host.resources().outstanding_closure_snapshot().active_closures, 0);
}
fn request(root: &Path, mode: JobOpenMode) -> SourceJobRequest {
    SourceJobRequest { root: root.to_path_buf(), key: JobSecret::from_bytes([41; 32]),
        job_id: JobId([42; 16]), limits: lifetime(), mode, materialize: true }
}
fn run_case(host: &NlpEngine, model: &ResidentInt8, kind: Kind, root: &Path,
    mode: JobOpenMode, width: usize, schedule: Option<Int8PrefillLimits>, input: &[u8]) -> JobProgress {
    let dropped = Arc::new(AtomicUsize::new(0));
    let inner = if matches!(mode, JobOpenMode::ResumeRetained { .. }) { None }
        else { Some(Cursor::new(input.to_vec())) };
    let reader = OwnedReader { inner, dropped: Arc::clone(&dropped) };
    let request = request(root, mode); let limits = host_limits();
    let cancellation = CancellationToken::default(); let kernel = Int8Kernel::scalar();
    let registry = pinned_controls::pinned().unwrap(); let controls = registry.template_controls();
    let eos = controls.entries().iter().find(|e| e.special && e.surface == crate::template::IM_END).unwrap().id;
    let result = match kind {
        Kind::Classify => {
            let (planner, config) = classification(model);
            match (width, schedule) {
                (1, None) => host.job_int8_classify(model, planner, config, request, limits, reader, cancellation),
                (1, Some(prefill)) => host.job_int8_classify_layer_major(model, planner, config, request,
                    limits, prefill, reader, cancellation),
                (_, None) => host.job_int8_classify_with_threads(model, planner, config, request,
                    limits, kernel, width, reader, cancellation),
                (_, Some(prefill)) => host.job_int8_classify_layer_major_with_threads(model, planner, config, request,
                    limits, kernel, width, prefill, reader, cancellation),
            }
        }
        Kind::Sentiment => {
            let planner = Arc::new(SentimentPlanner::pinned(controls, SentimentOptions {
                mode: ScoringMode::FullVocabulary, eos_token_id: eos,
                policy: SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 },
            }).unwrap());
            let config = SentimentCorpusConfig {
                identity: identity(model.artifact_identity(), "sentiment-v1", planner.tokenizer_digest(), *planner.template_digest()),
                task_ceiling: budget(), planning: SentimentLimits::default(), defaults: None,
                max_item_work: work(), max_model_work: work() };
            match (width, schedule) {
                (1, None) => host.job_int8_sentiment(model, planner, config, request, limits, reader, cancellation),
                (1, Some(prefill)) => host.job_int8_sentiment_layer_major(model, planner, config, request,
                    limits, prefill, reader, cancellation),
                (_, None) => host.job_int8_sentiment_with_threads(model, planner, config, request,
                    limits, kernel, width, reader, cancellation),
                (_, Some(prefill)) => host.job_int8_sentiment_layer_major_with_threads(model, planner, config, request,
                    limits, kernel, width, prefill, reader, cancellation),
            }
        }
        Kind::Judge => {
            let planner = Arc::new(JudgePlanner::pinned(controls, eos).unwrap());
            let config = JudgeCorpusConfig {
                identity: identity(model.artifact_identity(), "judge-v1", planner.tokenizer_digest(), *planner.template_digest()),
                task_ceiling: budget(), planning: JudgeLimits::default(), defaults: None, max_model_work: work() };
            match (width, schedule) {
                (1, None) => host.job_int8_judge(model, planner, config, request, limits, reader, cancellation),
                (1, Some(prefill)) => host.job_int8_judge_layer_major(model, planner, config, request,
                    limits, prefill, reader, cancellation),
                (_, None) => host.job_int8_judge_with_threads(model, planner, config, request,
                    limits, kernel, width, reader, cancellation),
                (_, Some(prefill)) => host.job_int8_judge_layer_major_with_threads(model, planner, config, request,
                    limits, kernel, width, prefill, reader, cancellation),
            }
        }
    };
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    result.expect("complete actual-model retained job must finish")
}

// Deterministically abandon the real planner's exact first-item debit at the
// durable pre-forward boundary. This is an interrupted reservation fixture,
// not a simulated successful processor or a claim about mid-kernel crashes.
fn abandon_first_attempt(host: &NlpEngine, model: &ResidentInt8, root: &Path, input: &[u8]) -> JobWork {
    #[derive(serde::Serialize)]
    struct Envelope<'a, R> { protocol: &'static str, input_profile: &'static str, processor: &'a R }
    let (planner, config) = classification(model);
    let root = root.to_path_buf(); let input = input.to_vec();
    dispatch::run(host, native().run, CancellationToken::default(), move |control| {
        let factory = Int8ClassificationJobPlanner::new(&planner, config.identity, config.task_ceiling,
            config.planning, config.defaults, config.max_model_work).unwrap();
        let key = JobSecret::from_bytes([41; 32]); let job_id = JobId([42; 16]);
        let population = JobPopulation::read_ndjson(&mut Cursor::new(input), &key, job_id,
            lifetime(), host_limits().transport, control).unwrap();
        let envelope = Envelope { protocol: OWNED_BATCH_PROTOCOL, input_profile: "identity-json-envelope-v1",
            processor: factory.job_recipe() };
        let manifest = population.freeze(&key, JobContract { job_id, execution: factory.execution_identity(),
            recipe: &envelope, limits: lifetime() }, control).unwrap();
        let first = population.inputs().next().unwrap();
        let prepared = factory.prepare_with_control(serde_json::from_slice(first.original).unwrap(), control).unwrap();
        let reserved = JobWork { model: prepared.model_work(), mask_node_visits: 0 };
        let mut job = OwnedJob::create(&root, key, manifest, control).unwrap();
        drop(job.begin(&first, reserved, control).unwrap());
        assert!(job.is_poisoned());
        let progress = job.progress();
        assert_eq!(progress.attempts, 1); assert_eq!(progress.committed, 0);
        assert_eq!(progress.reserved_work, reserved); assert!(!progress.materialized);
        Ok(reserved)
    }).unwrap()
}

#[test]
#[ignore = "controller-only independently pinned INT8 artifact and durable storage; no parity or recovery claim until run"]
fn stored_scoring_matches_schedules_replays_without_redebit_and_preserves_abandoned_attempts() {
    const CHILD: &str = "FNLP_INT8_SCORE_JOB_TEAM_CHILD";
    const COMPLETE: &str = "FNLP_INT8_SCORE_JOB_TEAM_COMPLETE";
    let full_name = concat!(module_path!(),
        "::stored_scoring_matches_schedules_replays_without_redebit_and_preserves_abandoned_attempts");
    let name = full_name.split_once("::").expect("crate-qualified test name").1;
    if std::env::var(CHILD).ok().as_deref() != Some(name) {
        let result = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", name, "--ignored", "--nocapture"]).env(CHILD, name)
            .output().expect("isolated actual-model gate must start");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(), "isolated actual-model gate failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr));
        assert!(stdout.contains(COMPLETE), "exact filter did not complete the actual-model gate");
        return;
    }
    assert!(crate::installed_process_resources().is_none());
    configure_process_kernel(Int8KernelKind::Scalar).unwrap();
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_SCORE_JOB_TEAM_MODEL")
        .expect("explicit admitted local INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_SCORE_JOB_TEAM_LOGICAL_SHA256")
        .expect("independent logical-model pin required");
    let source = std::env::var("FNLP_INT8_SCORE_JOB_TEAM_SOURCE_ROOT_SHA256")
        .expect("independent source-root pin required");
    Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source).unwrap();
    let memory: u64 = std::env::var("FNLP_INT8_SCORE_JOB_TEAM_MEMORY_MIB")
        .expect("controller-admitted process memory required").parse().expect("memory must be MiB");
    let parent = PathBuf::from(std::env::var_os("FNLP_INT8_SCORE_JOB_TEAM_PARENT")
        .expect("explicit protected controller directory required"));
    let unique = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
    let parent = parent.join(format!("scored-int8-{}-{unique}", std::process::id()));
    fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5,
        memory_ceiling_bytes: memory.checked_mul(1 << 20).expect("memory arithmetic"),
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let lease = host.resources().acquire_lease();
    // Covers fixture input/JSON/planners, capture copies, and the explicit
    // bounded journal/preparation owner used by the pre-forward interruption.
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
    let retention = InputRetentionLimits { max_spool_bytes: 1 << 20 };
    for (task, kind) in [Kind::Classify, Kind::Sentiment, Kind::Judge].into_iter().enumerate() {
        let input = input(kind); let mut expected = Vec::new(); let mut progress = None;
        for (case, (width, schedule)) in [(1, None), (1, Some(prefill(3))), (2, None),
            (2, Some(prefill(3))), (4, None), (4, Some(prefill(3)))].into_iter().enumerate() {
            let root = parent.join(format!("{task}-{case}"));
            fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
            let actual = run_case(&host, &model, kind, &root,
                JobOpenMode::CreateRetainingInputs(retention), width, schedule, &input);
            assert_eq!(actual.items, 3); assert_eq!(actual.committed, 3);
            assert_eq!(actual.attempts, 3); assert!(actual.materialized);
            assert!(actual.reserved_work.model.forward_positions > 0);
            let output = fs::read(root.join("materialized.ndjson")).unwrap();
            assert_eq!(serde_json::Deserializer::from_slice(&output).into_iter::<serde_json::Value>()
                .collect::<Result<Vec<_>, _>>().unwrap().len(), 3);
            if let Some(expected_progress) = progress {
                assert_eq!(actual, expected_progress, "{kind:?}: durable work and stored-byte counts must match");
                assert_eq!(output, expected, "{kind:?}: compare complete native records without removing fields");
            } else { expected = output; progress = Some(actual); }
            drained(&host, baseline);
            // Worker width is always physical. Classification/sentiment may
            // also change prompt rows; judge keeps its historical frozen row recipe.
            let replay_schedule = match kind {
                Kind::Judge => schedule,
                _ if schedule.is_some() => None,
                _ => Some(prefill(4)),
            };
            let replay_width = if width == 4 { 2 } else { 4 };
            let replay = run_case(&host, &model, kind, &root,
                JobOpenMode::ResumeRetained { tail: TailPolicy::Refuse, retention },
                replay_width, replay_schedule, &input);
            assert_eq!(replay, actual, "completed retained replay must not renew attempts or work");
            assert_eq!(fs::read(root.join("materialized.ndjson")).unwrap(), expected);
            drained(&host, baseline);
            let original = run_case(&host, &model, kind, &root,
                JobOpenMode::Resume(TailPolicy::Refuse), 1, replay_schedule, &input);
            assert_eq!(original, actual, "original-input replay authenticates the same complete contract");
            drained(&host, baseline);
        }
        if matches!(kind, Kind::Classify) {
            let root = parent.join("abandoned-classification");
            fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
            let abandoned = abandon_first_attempt(&host, &model, &root, &input);
            drained(&host, baseline);
            let resumed = run_case(&host, &model, kind, &root,
                JobOpenMode::Resume(TailPolicy::Refuse), 4, Some(prefill(3)), &input);
            let expected_progress = progress.unwrap();
            assert_eq!(resumed.items, expected_progress.items);
            assert_eq!(resumed.committed, expected_progress.committed);
            assert_eq!(resumed.attempts, expected_progress.attempts + 1);
            assert_eq!(resumed.reserved_work, expected_progress.reserved_work.checked_add(abandoned).unwrap());
            assert_eq!(resumed.spool_bytes, expected_progress.spool_bytes); assert!(resumed.materialized);
            assert_eq!(fs::read(root.join("materialized.ndjson")).unwrap(), expected);
            drained(&host, baseline);
        }
    }
    drop(model); drop(caller_preparation); drop(lease); drained(&host, 0);
    // Leave isolated protected artifacts for controller inspection; never clean
    // or adopt an existing user's job directory in this gate.
    println!("{COMPLETE}");
}
