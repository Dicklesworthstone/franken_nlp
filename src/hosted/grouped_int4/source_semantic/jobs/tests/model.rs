//! Controller-only independently pinned native execution and durable replay.
//! Every source result and judgment comes from the model, never a canned head.
use super::*;
use crate::{
    LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    hosted::corpus::jobs::JobOpenMode,
    jobs::{InputRetentionLimits, JobError, JobId, JobSecret, MismatchField, TailPolicy},
    native_engine::{decode::DecodeCancellationKind, lmhead::NANBEIGE_VOCAB_SIZE,
        strict_int4::weights::snapshot::{Int4SnapshotBudget, Int4SnapshotPin, Int4SnapshotRequirements}},
};
use std::{fs, io::{self, Read}, os::unix::fs::{DirBuilderExt, PermissionsExt},
    path::{Path, PathBuf}, time::SystemTime};

struct NeverRead;
impl Read for NeverRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("retained source replay read original input") }
}
impl BufRead for NeverRead {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { panic!("retained source replay filled original input") }
    fn consume(&mut self, _: usize) { panic!("retained source replay consumed original input") }
}
fn request(root: &Path, job: JobLimits, mode: JobOpenMode) -> SourceJobRequest {
    SourceJobRequest { root: root.to_owned(), key: JobSecret::from_bytes([79; 32]),
        job_id: JobId([80; 16]), limits: job, mode, materialize: true }
}
fn drained(engine: &NlpEngine, baseline: &crate::MemorySnapshot) {
    assert_eq!(&engine.resources().memory_snapshot(), baseline);
    let closures = engine.resources().outstanding_closure_snapshot();
    assert_eq!(closures.active_closures, 0); assert_eq!(closures.wrapper_cancelled_closures, 0);
}
#[allow(clippy::too_many_arguments)]
fn execute<R: BufRead + Send + 'static>(engine: &NlpEngine, model: &ResidentInt4,
    config: Int4SemanticSourceJobConfig, request: SourceJobRequest, host: JobHostLimits,
    threads: usize, reader: R, cancellation: CancellationToken) -> Result<JobProgress, HostedJobError> {
    if threads == 1 {
        engine.job_int4_semantic_source(model, config, request, host, PackedInt4Kernel::scalar(), reader, cancellation)
    } else {
        engine.job_int4_semantic_source_with_threads(model, config, request, host,
            PackedInt4Kernel::scalar(), threads, reader, cancellation)
    }
}
fn complete_reviews(bytes: &[u8], answer: bool) -> u64 {
    let rows = bytes.split(|&byte| byte == b'\n').filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap()).collect::<Vec<_>>();
    assert_eq!(rows.len(), 2);
    let mut attempts = 0;
    for row in rows {
        assert_eq!(row["execution"], INT4_SOURCE_SEMANTIC_EXECUTION);
        assert_eq!(row["source_task"]["execution"], INT4_SOURCE_EXECUTION);
        let original = &row["source_task"]["result"];
        assert_eq!(original["task"], if answer { "answer" } else { "summarize" });
        assert_eq!(original["result"]["semantic_support"], "not_assessed");
        assert_eq!(original["result"]["citation_guarantee"], "structural_source_membership");
        assert!(original["result"]["generated_token_ids"].as_array().is_some_and(|tokens| !tokens.is_empty()));
        assert_eq!(row["semantic"]["same_model_correlated"], true);
        assert_eq!(row["semantic"]["evidence_scope"], SOURCE_SEMANTIC_SCOPE);
        assert_eq!(row["semantic"]["calibration"], SOURCE_SEMANTIC_CALIBRATION);
        let attempted = row["semantic"]["attempted_statements"].as_u64().unwrap();
        let not_checked = row["semantic"]["not_checked_statements"].as_u64().unwrap();
        let statements = row["semantic"]["statements"].as_array().unwrap();
        assert_eq!(statements.len() as u64, attempted + not_checked);
        assert_eq!(statements.iter().filter(|s| s["judge"].is_object()).count() as u64, attempted);
        if attempted != 0 {
            assert!(row["judge_head_count"].as_u64().unwrap() > 0);
            assert!(row["semantic_model_work"]["forward_positions"].as_u64().unwrap() > 0);
        }
        // Legitimate empty summaries/answer abstention stay explicitly not
        // checked. They must never be converted into favorable judge evidence.
        for statement in statements.iter().filter(|s| s["judge"].is_null()) {
            assert_eq!(statement["status"], "not_checked");
            assert!(statement["claim"].is_null());
        }
        attempts += attempted;
    }
    assert!(bytes.ends_with(b"\n"));
    attempts
}

#[test]
#[ignore = "controller-only independent snapshot: summary/QA semantic jobs, serial/team byte/compact replay, recipe refusal and retained input"]
fn real_source_jobs_preserve_citations_reviews_and_spent_attempts_across_authenticated_resume() {
    assert!(crate::installed_process_resources().is_none(), "requires a fresh process");
    let snapshot = PathBuf::from(std::env::var("FNLP_INT4_PREFILL_SNAPSHOT").expect("explicit complete INT4 snapshot"));
    let digest = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256").expect("independent model-content pin");
    Sha256Digest::from_hex(&digest).expect("canonical lowercase SHA-256");
    let width: usize = std::env::var("FNLP_INT4_TEST_GROUP_WIDTH").expect("snapshot group width").parse().unwrap();
    let pin = Int4SnapshotPin { group_width: width, model_content_sha256: std::array::from_fn(|index|
        u8::from_str_radix(&digest[2 * index..2 * index + 2], 16).unwrap()) };
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_BYTES").expect("explicit memory ceiling").parse().unwrap();
    let parent = PathBuf::from(std::env::var("FNLP_INT4_TEST_JOB_DIR").expect("protected test parent"));
    let metadata = fs::symlink_metadata(&parent).unwrap();
    assert!(metadata.is_dir() && metadata.permissions().mode() & 0o077 == 0);
    let unique = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
    let parent = parent.join(format!("source-semantic-jobs-{}-{unique}", std::process::id()));
    fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 3,
        helper_threads: 0, thread_ceiling: 5, memory_ceiling_bytes: memory,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let idle = engine.resources().memory_snapshot();
    let run = RunLimits { max_elapsed: Duration::from_secs(86_400), max_checkpoints: 1_000_000_000,
        cleanup_reserve_bytes: 65_536 };
    let need = Int4SnapshotRequirements::for_group_width(width).unwrap();
    let model = engine.load_int4_snapshot(snapshot, pin, Int4SnapshotBudget {
        max_weight_bytes: need.weight_payload_bytes, max_scratch_bytes: need.scratch_bytes,
        max_io_bytes: need.read_io_bytes().unwrap(),
    }, 64 * MIB, run, CancellationToken::default()).unwrap();
    let native = Int4MemoryRequirement::for_context(4096, width).unwrap();
    let baseline = engine.resources().memory_snapshot();
    let retention = InputRetentionLimits { max_spool_bytes: MIB };
    let mut checked_statements = 0;
    for answer in [false, true] {
        let (mut config, mut job, mut host) = fixture(answer);
        let source = TaskBudget { max_input_tokens: 3072, max_output_tokens: 512,
            max_output_bytes: 65_536, max_grammar_states: 8192, max_kv_bytes: native.kv_bytes };
        config.task = if answer {
            Int4SemanticSourceBatchTask::Answer { options: AnswerOptions {
                max_answer_scalars: 128, max_citations: 2, max_quote_scalars: 64 }, budget: source }
        } else {
            Int4SemanticSourceBatchTask::Summarize { options: SummaryOptions {
                max_bullets: 1, max_bullet_scalars: 128, max_citations_per_bullet: 2,
                max_quote_scalars: 64 }, budget: source }
        };
        config.planning.max_context_tokens = 4096;
        config.planning.max_input_bytes = 8192;
        config.options.judge_budget = TaskBudget { max_input_tokens: 3072, max_output_tokens: 16,
            max_output_bytes: 65_536, max_grammar_states: 4096, max_kv_bytes: native.kv_bytes };
        config.options.limits.max_statements = 1;
        config.options.limits.max_claim_bytes = 4096;
        config.options.limits.max_total_claim_bytes = 4096;
        config.options.limits.max_total_prompt_tokens = 32_768;
        config.options.limits.max_total_projected_logits = 1_000_000_000;
        config.options.max_model_work = Int4Work::for_sequence(width, 0, 32_768, 1024 * NANBEIGE_VOCAB_SIZE).unwrap();
        config.judge_planning.max_total_prompt_tokens = 32_768;
        config.judge_planning.max_total_projected_logits = 1_000_000_000;
        config.judge_planning.max_output_bytes = 65_536;
        config.work.max_model_work = (0..4).try_fold(Int4Work::default(), |total, _|
            total.checked_add(config.options.max_model_work)).unwrap();
        config.work.masks = ExtractionMaskBudget { per_mask: MaskWorkLimits {
            max_trie_node_visits: 2_000_000, checkpoint_interval_nodes: 256 },
            max_visits_per_item: 2_000_000_000, max_visits_per_run: 8_000_000_000 };
        job.max_work = JobWork::from_int4(config.work.max_model_work, config.work.masks.max_visits_per_run);
        job.max_spool_bytes = 16 * MIB; job.max_materialized_bytes = 16 * MIB; job.max_journal_bytes = 16 * MIB;
        host.native.context_tokens = 4096; host.native.allocator_reserve_bytes = 64 * MIB; host.native.run = run;
        host.journal_reserve_bytes = 64 * MIB; host.serialization_reserve_bytes = 64 * MIB;
        let input = if answer {
            concat!("{\"id\":\"first\",\"text\":\"Where does Alice live?\",\"task_args\":{\"passages\":[",
                "{\"id\":\"p1\",\"text\":\"Alice lives in Paris.\"},{\"id\":\"p2\",\"text\":\"Bob lives in Lyon. café 上海\"}]}}\n",
                "{\"id\":\"second\",\"text\":\"Where does Alice work?\",\"task_args\":{\"passages\":[",
                "{\"id\":\"p1\",\"text\":\"Alice works in Lyon.\"},{\"id\":\"p2\",\"text\":\"Alice lives in Paris. <think>\"}]}}\n")
        } else {
            concat!("{\"id\":\"first\",\"text\":\"Alice lives in Paris. She works in Lyon. café 上海\"}\n",
                "{\"id\":\"second\",\"text\":\"Alice lives in Paris and works in Lyon. <think>\"}\n")
        }.as_bytes();
        for (encoding_name, encoding) in [("byte", UntrustedEncoding::ByteFallback),
            ("compact", UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 16_384,
                max_output_tokens: 16_384, max_pair_checks: 1 << 24, max_pair_bytes: 1 << 28 }))] {
            config.encoding = encoding;
            let mut expected = None;
            for retained in [false, true] { for threads in [1, 4] {
                let root = parent.join(format!("{answer}-{encoding_name}-{retained}-{threads}"));
                fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
                let mode = if retained { JobOpenMode::CreateRetainingInputs(retention) } else { JobOpenMode::Create };
                let progress = execute(&engine, &model, config.clone(), request(&root, job, mode), host,
                    threads, Cursor::new(input.to_vec()), CancellationToken::default()).unwrap();
                assert_eq!((progress.committed, progress.attempts), (2, 2)); assert!(progress.materialized);
                assert_eq!(progress.reserved_work, JobWork::from_int4(
                    config.options.max_model_work.checked_add(config.options.max_model_work).unwrap(),
                    2 * config.work.masks.max_visits_per_item));
                let bytes = fs::read(root.join("materialized.ndjson")).unwrap();
                checked_statements += complete_reviews(&bytes, answer);
                if let Some(expected) = &expected { assert_eq!(expected, &bytes); }
                else { expected = Some(bytes.clone()); }
                drained(&engine, &baseline);
                let next = if threads == 1 { 4 } else { 1 };
                let mode = if retained { JobOpenMode::ResumeRetained { tail: TailPolicy::Refuse, retention } }
                    else { JobOpenMode::Resume(TailPolicy::Refuse) };
                let replay = if retained {
                    execute(&engine, &model, config.clone(), request(&root, job, mode), host, next,
                        NeverRead, CancellationToken::default())
                } else {
                    execute(&engine, &model, config.clone(), request(&root, job, mode), host, next,
                        Cursor::new(input.to_vec()), CancellationToken::default())
                }.unwrap();
                assert_eq!(replay, progress); assert_eq!(fs::read(root.join("materialized.ndjson")).unwrap(), bytes);
                drained(&engine, &baseline);
                let mut changed = config.clone(); changed.options.judge_policy.minimum_margin_milli += 1;
                let altered = if retained {
                    execute(&engine, &model, changed, request(&root, job, mode), host, next,
                        NeverRead, CancellationToken::default())
                } else {
                    execute(&engine, &model, changed, request(&root, job, mode), host, next,
                        Cursor::new(input.to_vec()), CancellationToken::default())
                };
                assert!(matches!(altered, Err(HostedJobError::Job(JobRunError::Storage(
                    JobError::Mismatch(MismatchField::Recipe))))));
                assert_eq!(fs::read(root.join("materialized.ndjson")).unwrap(), bytes);
                drained(&engine, &baseline);
                let stop = CancellationToken::default(); stop.cancel(DecodeCancellationKind::ParentCancelled);
                assert!(matches!(execute(&engine, &model, config.clone(), request(&root, job, mode),
                    host, next, NeverRead, stop), Err(HostedJobError::Host(HostedError::Stopped { stop, .. }))
                    if stop.kind == DecodeCancellationKind::ParentCancelled));
                assert_eq!(fs::read(root.join("materialized.ndjson")).unwrap(), bytes);
                drained(&engine, &baseline);
            } }
        }
    }
    assert!(checked_statements > 0, "the model fixture must exercise actual statement reviews, not only abstention");
    drop(model); drained(&engine, &idle);
    // Keep isolated protected job artifacts for controller inspection.
}
