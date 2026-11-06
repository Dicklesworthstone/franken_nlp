//! Controller-only actual-model parity and durable-replay fixture. UNRUN.
use super::*;
use crate::{
    LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    hosted::corpus::jobs::JobOpenMode,
    jobs::{JobId, JobSecret, TailPolicy},
    native_engine::decode::DecodeCancellationKind,
    tasks::classify::grouped_int4::Int4ClassificationPlanner,
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

fn planner(model: &ResidentInt4, settings: ClassificationBatchArgs, planning: ClassificationLimits,
    document: Int4ClassificationMapLimits, job: JobLimits, host: JobHostLimits, cohorts: bool)
    -> Int4ClassificationDocumentJobPlanner {
    let compiler = Int4ClassificationPlanner::for_model(model.model_identity(), settings.budget, planning).unwrap();
    let planner = Int4ClassificationDocumentJobPlanner::new(compiler, settings, document, job).unwrap();
    if cohorts { planner.with_chunk_cohorts(2, memory_limits(host)).unwrap() } else { planner }
}

fn request(root: PathBuf, limits: JobLimits, mode: JobOpenMode) -> SourceJobRequest {
    SourceJobRequest { root, key: JobSecret::from_bytes([41; 32]), job_id: JobId([42; 16]),
        limits, mode, materialize: true }
}

#[test]
#[ignore = "requires controller-admitted pinned source, explicit memory and full native durable-job execution"]
fn complete_native_classification_jobs_match_serial_and_cohorts_and_resume_without_more_work() {
    let source = PathBuf::from(std::env::var("FNLP_INT4_TEST_SOURCE_DIR").expect("explicit source required"));
    let manifest = PathBuf::from(std::env::var("FNLP_INT4_TEST_SOURCE_MANIFEST").expect("explicit manifest required"));
    let memory = std::env::var("FNLP_INT4_TEST_MEMORY_BYTES").expect("explicit process memory cap required")
        .parse::<u64>().unwrap();
    let parent = PathBuf::from(std::env::var("FNLP_INT4_TEST_JOB_DIR").expect("explicit private test parent required"));
    let unique = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
    let parent = parent.join(format!("classification-document-{}-{unique}", std::process::id()));
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
        "{\"id\":\"first\",\"text\":\"é first.\\nSecond part.\",\"task_args\":{}}\n",
        "{\"id\":\"second\",\"text\":\"Another complete source.\",\"task_args\":{}}\n");
    for (index, mode) in [ClassificationMode::Exclusive, ClassificationMode::MultiLabel].into_iter().enumerate() {
        let (mut settings, planning, mut document, job, mut host) = fixture(32);
        settings.mode = mode; document.chunks.max_chunk_bytes = 12; host.native.run = run;
        let mut serial = Vec::new();
        for cohorts in [false, true] {
            let root = parent.join(format!("{index}-{cohorts}"));
            fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
            let factory = || planner(&model, settings.clone(), planning, document, job, host, cohorts);
            let progress = engine.job_int4_classification_document(&model, factory(),
                request(root.clone(), job, JobOpenMode::Create), host, PackedInt4Kernel::scalar(),
                Cursor::new(source.as_bytes().to_vec()), CancellationToken::default()).unwrap();
            assert_eq!(progress.committed, 2); assert_eq!(progress.attempts, 2); assert!(progress.materialized);
            drained(&engine, baseline);
            let output = fs::read_to_string(root.join("materialized.ndjson")).unwrap();
            let values: Vec<serde_json::Value> = output.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
            assert_eq!(values.len(), 2);
            if cohorts {
                for (expected, actual) in serial.iter().zip(&values) {
                    assert_eq!(expected, &actual["output"]);
                    assert!(actual["chunk_cohorts"].as_u64().unwrap() > 0);
                }
            } else {
                for value in &values { assert!(value["chunks"].as_array().unwrap().len() > 1); }
                serial = values;
            }
            let replay = engine.job_int4_classification_document(&model, factory(),
                request(root.clone(), job, JobOpenMode::Resume(TailPolicy::Refuse)), host, PackedInt4Kernel::scalar(),
                Cursor::new(source.as_bytes().to_vec()), CancellationToken::default()).unwrap();
            assert_eq!(replay, progress, "committed documents must consume no additional attempts or native work");
            assert_eq!(fs::read_to_string(root.join("materialized.ndjson")).unwrap(), output);
            drained(&engine, baseline);
            let mut changed = settings.clone(); changed.labels[0].description = "changed private policy".to_owned();
            let changed = planner(&model, changed, planning, document, job, host, cohorts);
            let before = fs::read(root.join("results.spool")).unwrap();
            assert!(engine.job_int4_classification_document(&model, changed,
                request(root.clone(), job, JobOpenMode::Resume(TailPolicy::DiscardUncommitted)),
                host, PackedInt4Kernel::scalar(), Cursor::new(source.as_bytes().to_vec()),
                CancellationToken::default()).is_err());
            assert_eq!(fs::read(root.join("results.spool")).unwrap(), before);
            drained(&engine, baseline);
            let token = CancellationToken::default(); token.cancel(DecodeCancellationKind::User);
            let dropped = Arc::new(AtomicBool::new(false));
            assert!(engine.job_int4_classification_document(&model, factory(),
                request(root, job, JobOpenMode::Resume(TailPolicy::Refuse)), host,
                PackedInt4Kernel::scalar(), NeverRead(Arc::clone(&dropped)), token).is_err());
            assert!(dropped.load(Ordering::SeqCst)); drained(&engine, baseline);
        }
    }
    drop(model); drained(&engine, 0);
    // Retain the isolated owner-only job artifacts for controller inspection.
}
