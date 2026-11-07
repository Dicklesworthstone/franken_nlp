//! Controller-only actual-model parity and durable-replay fixture. UNRUN.
//! Real pinned weights are mandatory; no synthetic scorer enters this path.
use super::*;
use crate::{
    LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    hosted::corpus::jobs::JobOpenMode,
    jobs::{JobId, JobSecret, TailPolicy},
    native_engine::{decode::DecodeCancellationKind, lmhead::scoring::ScoringMode},
    tasks::sentiment::{SentimentAxis, SentimentOptions, SentimentPolicy,
        batch::SentimentBatchArgs, grouped_int4::Int4SentimentPlanner},
};
use std::{fs, io::{self, BufRead, Cursor, Read}, os::unix::fs::DirBuilderExt,
    path::PathBuf, sync::{Arc, atomic::{AtomicBool, Ordering}}, time::SystemTime};

struct NeverRead(Arc<AtomicBool>);
impl Read for NeverRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("cancelled job read its source") }
}
impl BufRead for NeverRead {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { panic!("cancelled job read its source") }
    fn consume(&mut self, _: usize) { panic!("cancelled job consumed its source") }
}
impl Drop for NeverRead { fn drop(&mut self) { self.0.store(true, Ordering::SeqCst); } }

fn drained(engine: &NlpEngine, resident_bytes: u64) {
    let snapshot = engine.resources().memory_snapshot();
    assert_eq!(snapshot.reserved_bytes, 0);
    assert_eq!(snapshot.committed_bytes, resident_bytes);
    assert_eq!(snapshot.outstanding_obligations, 0);
    assert_eq!(snapshot.recorded_leaks, 0);
    assert_eq!(engine.resources().outstanding_closure_snapshot().active_closures, 0);
}

fn options() -> SentimentOptions {
    let registry = crate::tokenizer::pinned_controls::pinned().unwrap();
    let mut ends = registry.template_controls().entries().iter()
        .filter(|entry| entry.special && entry.surface == crate::template::IM_END);
    let eos_token_id = ends.next().unwrap().id;
    assert!(ends.next().is_none());
    SentimentOptions { mode: ScoringMode::FullVocabulary, eos_token_id,
        policy: SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: 1_000_000 } }
}

#[allow(clippy::too_many_arguments)]
fn planner(model: &ResidentInt4, settings: SentimentBatchArgs, options: SentimentOptions,
    planning: SentimentLimits, mapping: Int4SentimentMapLimits, job: JobLimits, cohorts: bool)
    -> Int4SentimentDocumentJobPlanner {
    let compiler = Int4SentimentPlanner::for_model(model.model_identity(), options,
        settings.budget, planning).unwrap();
    let planner = Int4SentimentDocumentJobPlanner::new(compiler, settings, mapping, job).unwrap();
    if cohorts { planner.with_chunk_cohorts(2).unwrap() } else { planner }
}

fn request(root: PathBuf, limits: JobLimits, mode: JobOpenMode) -> SourceJobRequest {
    SourceJobRequest { root, key: JobSecret::from_bytes([43; 32]), job_id: JobId([44; 16]),
        limits, mode, materialize: true }
}

#[test]
#[ignore = "requires controller-admitted pinned source, explicit memory and full native durable-job execution"]
fn complete_native_sentiment_jobs_match_serial_and_cohorts_and_resume_without_more_work() {
    let source = PathBuf::from(std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit source required"));
    let manifest = PathBuf::from(std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit manifest required"));
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_BYTES").expect("explicit process memory cap required")
        .parse::<u64>().unwrap();
    let parent = PathBuf::from(std::env::var("FNLP_INT4_TEST_JOB_DIR").expect("explicit private test parent required"));
    let unique = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
    let parent = parent.join(format!("sentiment-document-{}-{unique}", std::process::id()));
    fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
    let engine = NlpEngine::builder().resource_config(ResourceHostConfig {
        runtime_preset: RuntimePreset::CurrentThread, runtime_workers: 1,
        max_blocking_coordinators: 1, scoped_cpu_children_per_coordinator: 0,
        helper_threads: 0, thread_ceiling: 2, memory_ceiling_bytes: memory,
        leak_response_policy: LeakResponsePolicy::PanicInLabOrCi,
    }).build().unwrap();
    let run = RunLimits { max_elapsed: Duration::from_secs(3600), max_checkpoints: 1_000_000_000,
        cleanup_reserve_bytes: 65536 };
    let load = Int4ModelLoadRequirement::for_group_width(32).unwrap();
    let model = engine.load_source_candidate_int4(source, manifest, 32, Int4LoadLimits {
        model: Int4ModelLoadBudget { max_payload_bytes: load.peak_payload_bytes().unwrap(),
            max_source_values: load.source_values, max_groups: load.groups,
            max_shard_revalidation_bytes: load.shard_revalidation_bytes },
        source_preparation_bytes: 128 << 20, allocator_reserve_bytes: 64 << 20, run,
    }, CancellationToken::default()).unwrap();
    let baseline = engine.resources().memory_snapshot().committed_bytes;
    let source = concat!(
        "{\"id\":\"first\",\"text\":\"I am delighted. é\\nDelivery was late.\",\"task_args\":{}}\n",
        "{\"id\":\"second\",\"text\":\"上海 arrived. Service was calm.\",\"task_args\":{}}\n");
    let (mut task, planning, mut mapping, mut job, mut host) = fixture();
    task.max_input_tokens = 768;
    task.max_kv_bytes = Int4MemoryRequirement::for_context(1024, 32).unwrap().kv_bytes * 4;
    mapping.chunks.context_tokens = 1024;
    mapping.chunks.max_chunk_tokens = 768;
    mapping.chunks.max_chunk_bytes = 12;
    mapping.chunks.max_chunks = 8;
    mapping.max_model_work = Int4Work::for_sequence(32, 0, 8 * 4 * 1024,
        8 * 4 * 16 * NANBEIGE_VOCAB_SIZE).unwrap();
    job.max_work = JobWork::from_int4(mapping.max_model_work.checked_add(mapping.max_model_work).unwrap(), 0);
    host.native.context_tokens = 1024; host.native.run = run;
    let options = options();
    for axes in [vec![SentimentAxis::Valence], SentimentAxis::ALL.to_vec()] {
        let settings = SentimentBatchArgs { axes, budget: task };
        let mut serial: Vec<serde_json::Value> = Vec::new();
        let mut serial_work = None;
        for cohorts in [false, true] {
            let root = parent.join(format!("{}-{cohorts}", settings.axes.len()));
            fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
            let factory = || planner(&model, settings.clone(), options, planning, mapping, job, cohorts);
            let progress = engine.job_int4_sentiment_document(&model, factory(),
                request(root.clone(), job, JobOpenMode::Create), host, PackedInt4Kernel::scalar(),
                Cursor::new(source.as_bytes().to_vec()), CancellationToken::default()).unwrap();
            assert_eq!(progress.committed, 2); assert_eq!(progress.attempts, 2); assert!(progress.materialized);
            drained(&engine, baseline);
            let output = fs::read_to_string(root.join("materialized.ndjson")).unwrap();
            let values: Vec<serde_json::Value> = output.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
            assert_eq!(values.len(), 2);
            if cohorts {
                assert_eq!(Some(progress.reserved_work), serial_work);
                for (expected, actual) in serial.iter().zip(&values) {
                    assert_eq!(expected, &actual["output"]);
                    assert!(actual["chunk_cohorts"].as_u64().unwrap() > 0);
                }
            } else {
                for value in &values {
                    assert!(value["chunks"].as_array().unwrap().len() > 1);
                    assert_eq!(value["aggregate"]["dimensions"].as_array().unwrap().len(), settings.axes.len());
                }
                serial = values; serial_work = Some(progress.reserved_work);
            }
            let replay = engine.job_int4_sentiment_document(&model, factory(),
                request(root.clone(), job, JobOpenMode::Resume(TailPolicy::Refuse)), host, PackedInt4Kernel::scalar(),
                Cursor::new(source.as_bytes().to_vec()), CancellationToken::default()).unwrap();
            assert_eq!(replay, progress, "committed sources must consume no additional attempts or native work");
            assert_eq!(fs::read_to_string(root.join("materialized.ndjson")).unwrap(), output);
            drained(&engine, baseline);
            let before = fs::read(root.join("results.spool")).unwrap();
            let mut changed_options = options;
            changed_options.policy.minimum_peak_weight_ppm = 1;
            let changed = planner(&model, settings.clone(), changed_options, planning, mapping, job, cohorts);
            assert!(engine.job_int4_sentiment_document(&model, changed,
                request(root.clone(), job, JobOpenMode::Resume(TailPolicy::DiscardUncommitted)), host,
                PackedInt4Kernel::scalar(), Cursor::new(source.as_bytes().to_vec()), CancellationToken::default()).is_err());
            let changed_source = source.replace("late", "slow");
            assert!(engine.job_int4_sentiment_document(&model, factory(),
                request(root.clone(), job, JobOpenMode::Resume(TailPolicy::DiscardUncommitted)), host,
                PackedInt4Kernel::scalar(), Cursor::new(changed_source.into_bytes()), CancellationToken::default()).is_err());
            assert_eq!(fs::read(root.join("results.spool")).unwrap(), before);
            drained(&engine, baseline);
            let token = CancellationToken::default(); token.cancel(DecodeCancellationKind::User);
            let dropped = Arc::new(AtomicBool::new(false));
            assert!(engine.job_int4_sentiment_document(&model, factory(),
                request(root, job, JobOpenMode::Resume(TailPolicy::Refuse)), host,
                PackedInt4Kernel::scalar(), NeverRead(Arc::clone(&dropped)), token).is_err());
            assert!(dropped.load(Ordering::SeqCst)); drained(&engine, baseline);
        }
    }
    drop(model); drained(&engine, 0);
    // Keep the isolated owner-only job artifacts for controller inspection.
}
