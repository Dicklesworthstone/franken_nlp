//! Controller-only independently pinned INT8 execution and authenticated replay.
//! These definitions are not model, parity or crash-recovery execution evidence.
use std::{fs, io::{self, BufRead, Cursor, Read},
    os::unix::fs::{DirBuilderExt, PermissionsExt}, path::{Path, PathBuf},
    sync::Arc, time::{Duration, SystemTime}};
use crate::{
    canonjson, LeakResponsePolicy, MemoryClass, NlpEngine, ResourceHostConfig, RuntimePreset,
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    grammar::mask::MaskWorkLimits,
    hosted::{CancellationToken, HostedError, LoadLimits, NativeLimits, Pending, ResidentInt8, RunLimits,
        corpus::{RedactionCorpusConfig,
            jobs::{HostedJobError, JobHostLimits, JobOpenMode, SourceJobRequest}}},
    jobs::{InputRetentionLimits, JobError, JobId, JobLimits, JobProgress, JobSecret, JobWork,
        MismatchField, TailPolicy, population::PopulationReadLimits, runner::JobRunError},
    native_engine::{artifact_bridge::{ArtifactIdentity, ArtifactLoadBudget},
        decode::DecodeCancellationKind, lmhead::NANBEIGE_VOCAB_SIZE,
        simd::{Int8Kernel, Int8KernelKind, config::configure_process_kernel},
        strict_int8::{Int8MemoryRequirement, Int8Work, STRICT_INT8_EXECUTION, prefill::Int8PrefillLimits}},
    tasks::{extract::ExtractionVocabulary, ir::TaskBudget, mapreduce::{ChunkLimits, ExecutionLimits},
        ner::{EntityType, NerOptions, NER_TASK_VERSION},
        redact::{RedactionRequest, actions::RedactionAction, batch::Int8RedactionBatchConfig,
            corpus::LongRedactionBatchConfig, long::{LongRedactionConfig, LONG_REDACTION_EXECUTION},
            quantized::{Int8RedactionConfig, INT8_REDACTION_EXECUTION}},
        source_planning::{SourcePlanningLimits, SourceTaskPlanner, quantized::long::Int8SourceMapLimits}},
    tokenizer::{pinned_controls, untrusted::{SafeBpeLimits, SafeBpeWorkLimits, UntrustedEncoding}},
};

const MIB: u64 = 1 << 20;
const SOURCE: &str = "Alice visits Paris. Email a@example.org. Bob visits Rome. é 上海 <think>.";
const RESULT_BYTES: u64 = 4 * MIB;
const SHORT_MASKS: u64 = 1_024_000_000;
const LONG_MASKS: u64 = 32_768_000_000;

fn native() -> NativeLimits {
    NativeLimits { context_tokens: 2048, allocator_reserve_bytes: 64 * MIB,
        run: RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000_000,
            cleanup_reserve_bytes: 65_536 } }
}
fn prefill(rows: usize) -> Int8PrefillLimits {
    Int8PrefillLimits { max_batch_rows: rows,
        max_extra_scratch_bytes: Int8PrefillLimits::required_extra_scratch_bytes(rows).unwrap() }
}
fn budget() -> TaskBudget {
    TaskBudget { max_input_tokens: 1536, max_output_tokens: 512, max_output_bytes: MIB,
        max_grammar_states: 4096, max_kv_bytes: Int8MemoryRequirement::for_context(2048).unwrap().kv_bytes }
}
fn item_work() -> Int8Work { Int8Work::for_sequence(0, 65_536, 8192 * NANBEIGE_VOCAB_SIZE).unwrap() }
fn total_work() -> Int8Work { item_work().checked_add(item_work()).unwrap() }
fn lifetime(long: bool) -> JobLimits {
    JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 4096,
        max_snapshot_bytes: MIB, max_result_bytes: RESULT_BYTES as usize,
        max_spool_bytes: 16 * MIB, max_materialized_bytes: 16 * MIB,
        max_journal_bytes: 16 * MIB, max_attempts: 2,
        max_work: JobWork { model: total_work(), mask_node_visits: 2 * if long { LONG_MASKS } else { SHORT_MASKS } } }
}
fn limits() -> JobHostLimits {
    JobHostLimits { native: native(), preparation_reserve_bytes: 512 * MIB, io_reserve_bytes: 32 * MIB,
        journal_reserve_bytes: 64 * MIB, serialization_reserve_bytes: 64 * MIB,
        transport: PopulationReadLimits { max_stream_bytes: MIB, max_lines: 16 } }
}
fn assets(facts: &ArtifactIdentity, compact: bool, changed_recipe: bool)
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
    // Successful durable cases explicitly disable verification. The separate
    // redaction model fixture covers fresh verification and residual refusal.
    let mut request = RedactionRequest { verify: false, ..RedactionRequest::default() };
    request.actions.default_action = RedactionAction::Mask;
    request.actions.include_map = !changed_recipe;
    request.rule_budget.max_input_bytes = 4096; request.edit_budget.max_output_bytes = RESULT_BYTES as usize;
    let detector = Int8RedactionConfig {
        ner: NerOptions { types: vec![EntityType::Person], max_entities: 4, max_mention_scalars: 64 },
        per_pass: budget(), planning: SourcePlanningLimits { max_input_bytes: 4096,
            max_context_tokens: 2048, ..SourcePlanningLimits::default() },
        max_model_work: item_work(),
        mask_limits: MaskWorkLimits { max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
        mask_visits_per_pass: SHORT_MASKS, max_mask_visits: 2 * SHORT_MASKS, max_result_bytes: RESULT_BYTES,
    };
    (Arc::new(planner), RedactionCorpusConfig { batch: Int8RedactionBatchConfig {
        ner_identity, detector, request, max_model_work: total_work(), max_mask_visits: 2 * SHORT_MASKS,
    }, edit_reserve_bytes: 64 * MIB })
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
            mask_visits_per_chunk: SHORT_MASKS, max_mask_visits: LONG_MASKS,
        }, max_result_bytes: RESULT_BYTES };
    RedactionCorpusConfig { batch: LongRedactionBatchConfig { ner_identity: batch.ner_identity,
        detector, request: batch.request, max_model_work: total_work(), max_mask_visits: 2 * LONG_MASKS,
    }, edit_reserve_bytes: config.edit_reserve_bytes }
}
fn input() -> Vec<u8> {
    let mut bytes = Vec::new();
    for id in ["first", "second"] {
        bytes.extend_from_slice(&serde_json::to_vec(&serde_json::json!({ "id": id, "text": SOURCE })).unwrap());
        bytes.push(b'\n');
    }
    bytes
}
struct NeverRead;
impl Read for NeverRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("retained redaction replay read external input") }
}
impl BufRead for NeverRead {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { panic!("retained redaction replay filled external input") }
    fn consume(&mut self, _: usize) { panic!("retained redaction replay consumed external input") }
}
fn request(root: &Path, long: bool, mode: JobOpenMode) -> SourceJobRequest {
    SourceJobRequest { root: root.to_owned(), key: JobSecret::from_bytes([83; 32]),
        job_id: JobId([84; 16]), limits: lifetime(long), mode, materialize: true }
}
fn drained(host: &NlpEngine, baseline: u64) {
    let memory = host.resources().memory_snapshot();
    assert_eq!(memory.committed_bytes, baseline); assert_eq!(memory.reserved_bytes, 0);
    assert_eq!(memory.recorded_leaks, 0);
    let closures = host.resources().outstanding_closure_snapshot();
    assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
}
#[allow(clippy::too_many_arguments)]
fn execute<R: BufRead + Send + 'static>(host: &NlpEngine, model: &ResidentInt8,
    vocabulary: &Arc<ExtractionVocabulary>, long: bool, compact: bool, changed_recipe: bool,
    width: usize, schedule: Option<Int8PrefillLimits>, request: SourceJobRequest,
    reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError> {
    assert!(long || !compact, "compact single-context populations remain refused");
    // Fresh factories prevent comparison calls from sharing compact work.
    let (planner, config) = assets(model.artifact_identity(), compact, changed_recipe);
    let vocabulary = Arc::clone(vocabulary); let limits = limits(); let kernel = Int8Kernel::scalar();
    if long {
        let config = document(config);
        match (width, schedule) {
            (1, None) => host.job_int8_redact_document(model, planner, vocabulary, config, None,
                request, limits, reader, cancellation),
            (1, Some(prefill)) => host.job_int8_redact_document_layer_major(model, planner, vocabulary, config, None,
                request, limits, prefill, reader, cancellation),
            (_, None) => host.job_int8_redact_document_with_threads(model, planner, vocabulary, config, None,
                request, limits, kernel, width, reader, cancellation),
            (_, Some(prefill)) => host.job_int8_redact_document_layer_major_with_threads(model, planner, vocabulary, config, None,
                request, limits, kernel, width, prefill, reader, cancellation),
        }
    } else {
        match (width, schedule) {
            (1, None) => host.job_int8_redact(model, planner, vocabulary, config, None,
                request, limits, reader, cancellation),
            (1, Some(prefill)) => host.job_int8_redact_layer_major(model, planner, vocabulary, config, None,
                request, limits, prefill, reader, cancellation),
            (_, None) => host.job_int8_redact_with_threads(model, planner, vocabulary, config, None,
                request, limits, kernel, width, reader, cancellation),
            (_, Some(prefill)) => host.job_int8_redact_layer_major_with_threads(model, planner, vocabulary, config, None,
                request, limits, kernel, width, prefill, reader, cancellation),
        }
    }
}
fn complete(progress: JobProgress, bytes: &[u8], long: bool) {
    assert_eq!((progress.committed, progress.attempts), (2, 2)); assert!(progress.materialized);
    // Both attempt and work ceilings are exhausted: an accidental reexecution
    // on resume cannot fit. Actual model work never refunds these reservations.
    assert_eq!(progress.reserved_work, lifetime(long).max_work);
    assert!(bytes.ends_with(b"\n"));
    let rows = bytes.split(|&byte| byte == b'\n').filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap()).collect::<Vec<_>>();
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert_eq!(row["execution"], if long { LONG_REDACTION_EXECUTION } else { INT8_REDACTION_EXECUTION });
        assert_eq!(row["result"]["verification"], "not_requested");
        assert!(!row["result"]["text"].as_str().unwrap().contains("a@example.org"));
        let work: Int8Work = serde_json::from_value(row["model_work"].clone()).unwrap();
        assert!(work.forward_positions > 0 && work.projected_logits > 0 && work.attention_pairs > 0);
        assert!(work.projections.dot_products > 0 && work.projections.multiply_accumulates > 0);
        if long { assert!(row["original"]["preflight"]["chunks"].as_u64().unwrap() > 1); }
        else { assert_eq!(row["ner_passes"], 1); }
    }
}
#[test]
#[ignore = "controller-only independent INT8 artifact pins: retained redaction schedule parity, exhausted-work replay and physical drain"]
fn retained_redaction_preserves_materialized_results_and_lifetime_work_across_schedule_changes() {
    assert!(crate::installed_process_resources().is_none(), "run alone in a fresh test process");
    configure_process_kernel(Int8KernelKind::Scalar).unwrap();
    let path = PathBuf::from(std::env::var_os("FNLP_INT8_REDACTION_JOBS_TEAM_MODEL").expect("admitted local INT8 artifact required"));
    let logical = std::env::var("FNLP_INT8_REDACTION_JOBS_TEAM_LOGICAL_SHA256").expect("independent logical pin required");
    let source_root = std::env::var("FNLP_INT8_REDACTION_JOBS_TEAM_SOURCE_ROOT_SHA256").expect("independent source-root pin required");
    Sha256Digest::from_hex(&logical).unwrap(); Sha256Digest::from_hex(&source_root).unwrap();
    let memory: u64 = std::env::var("FNLP_INT8_REDACTION_JOBS_TEAM_MEMORY_MIB")
        .expect("controller-admitted process memory required").parse().expect("memory must be MiB");
    let parent = PathBuf::from(std::env::var_os("FNLP_INT8_REDACTION_JOBS_TEAM_DIR").expect("protected test parent required"));
    let metadata = fs::symlink_metadata(&parent).unwrap();
    assert!(metadata.is_dir() && metadata.permissions().mode() & 0o077 == 0);
    let unique = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
    let parent = parent.join(format!("int8-redaction-jobs-{}-{unique}", std::process::id()));
    fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
    let host = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5,
        memory_ceiling_bytes: memory.checked_mul(MIB).expect("memory arithmetic"),
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let idle = host.resources().memory_snapshot().committed_bytes;
    let lease = host.resources().acquire_lease();
    let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, 512 * MIB).unwrap().commit().unwrap();
    let model = host.load_current_candidate_int8(path, LoadLimits {
        artifact: ArtifactLoadBudget::streaming_only(6_u64 << 30, 64 << 10),
        tokenizer_and_metadata_bytes: 128 * MIB, allocator_reserve_bytes: 64 * MIB, run: native().run,
    }, CancellationToken::default()).unwrap();
    assert_eq!(model.artifact_identity().logical_model_sha256, logical);
    assert_eq!(model.artifact_identity().source_root_sha256, source_root);
    assert_eq!(model.artifact_identity().revision, "f56ec5a9650268aa098496734743c25ea778bd2d");
    let vocabulary = Arc::new(ExtractionVocabulary::pinned(pinned_controls::pinned().unwrap().template_controls()).unwrap());
    let baseline = host.resources().memory_snapshot().committed_bytes;
    let retention = InputRetentionLimits { max_spool_bytes: MIB };
    for (long, compact) in [(false, false), (true, false), (true, true)] {
        let mut expected: Option<Vec<u8>> = None;
        for retained in [false, true] {
            for (index, (width, schedule)) in [(1, None), (1, Some(prefill(3))),
                (2, None), (4, Some(prefill(7)))].into_iter().enumerate() {
                let root = parent.join(format!("{long}-{compact}-{retained}-{index}"));
                fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
                let mode = if retained { JobOpenMode::CreateRetainingInputs(retention) } else { JobOpenMode::Create };
                let progress = execute(&host, &model, &vocabulary, long, compact, false, width, schedule,
                    request(&root, long, mode), Cursor::new(input()), CancellationToken::default()).unwrap();
                let bytes = fs::read(root.join("materialized.ndjson")).unwrap();
                complete(progress, &bytes, long);
                if let Some(expected) = &expected { assert_eq!(expected, &bytes); }
                else { expected = Some(bytes.clone()); }
                assert_eq!(Arc::strong_count(&vocabulary), 1); drained(&host, baseline);
                let (next_width, next_schedule) = match index {
                    0 => (4, Some(prefill(7))), 1 => (2, None),
                    2 => (1, Some(prefill(3))), _ => (1, None),
                };
                let mode = if retained { JobOpenMode::ResumeRetained { tail: TailPolicy::Refuse, retention } }
                    else { JobOpenMode::Resume(TailPolicy::Refuse) };
                let replay = if retained {
                    execute(&host, &model, &vocabulary, long, compact, false, next_width, next_schedule,
                        request(&root, long, mode), NeverRead, CancellationToken::default())
                } else {
                    execute(&host, &model, &vocabulary, long, compact, false, next_width, next_schedule,
                        request(&root, long, mode), Cursor::new(input()), CancellationToken::default())
                }.unwrap();
                assert_eq!(replay, progress, "a schedule change never reruns committed items or renews durable work");
                assert_eq!(fs::read(root.join("materialized.ndjson")).unwrap(), bytes);
                assert_eq!(Arc::strong_count(&vocabulary), 1); drained(&host, baseline);
                let altered = if retained {
                    execute(&host, &model, &vocabulary, long, compact, true, next_width, next_schedule,
                        request(&root, long, mode), NeverRead, CancellationToken::default())
                } else {
                    execute(&host, &model, &vocabulary, long, compact, true, next_width, next_schedule,
                        request(&root, long, mode), Cursor::new(input()), CancellationToken::default())
                };
                assert!(matches!(altered, Err(HostedJobError::Job(JobRunError::Storage(
                    JobError::Mismatch(MismatchField::Recipe))))));
                assert_eq!(fs::read(root.join("materialized.ndjson")).unwrap(), bytes);
                assert_eq!(Arc::strong_count(&vocabulary), 1); drained(&host, baseline);
                let stop = CancellationToken::default(); stop.cancel(DecodeCancellationKind::ParentCancelled);
                assert!(matches!(execute(&host, &model, &vocabulary, long, compact, false,
                    next_width, next_schedule, request(&root, long, mode), NeverRead, stop),
                    Err(HostedJobError::Host(HostedError::Stopped { stop, .. }))
                    if stop.kind == DecodeCancellationKind::ParentCancelled));
                assert_eq!(fs::read(root.join("materialized.ndjson")).unwrap(), bytes);
                assert_eq!(Arc::strong_count(&vocabulary), 1); drained(&host, baseline);
            }
        }
    }
    drop(vocabulary); drop(model); drop(preparation); drop(lease); drained(&host, idle);
    // Keep isolated protected job artifacts for controller inspection.
}
