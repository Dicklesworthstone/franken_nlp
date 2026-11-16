//! Opt-in real-snapshot integration, never a substitute for controller evidence.
//! Requires an independently pinned complete model and a fresh runtime process.

use super::*;
use crate::{
    LeakResponsePolicy, ResourceHostConfig, RuntimePreset,
    batch::sentiment::sampled::SampledSentimentBatchPlanner,
    execution_identity::Sha256Digest,
    hosted::corpus::jobs::JobOpenMode,
    jobs::{InputRetentionLimits, JobError, JobId, JobSecret, MismatchField, TailPolicy},
    native_engine::{decode::DecodeCancellationKind,
        strict_int4::weights::snapshot::{Int4SnapshotBudget, Int4SnapshotPin, Int4SnapshotRequirements}},
    tasks::sentiment::{SentimentAxis,
        sampled::{SampledSentimentOptions, SampledSentimentPlanner}},
};
use std::{fs, io::{self, Cursor, Read}, os::unix::fs::{DirBuilderExt, PermissionsExt},
    path::{Path, PathBuf}, sync::atomic::{AtomicUsize, Ordering}, time::SystemTime};

struct NeverRead(Arc<AtomicUsize>);
impl Read for NeverRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("retained sampled replay touched original input")
    }
}
impl BufRead for NeverRead {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        panic!("retained sampled replay read original input")
    }
    fn consume(&mut self, _: usize) { panic!("retained sampled replay consumed original input") }
}
impl Drop for NeverRead {
    fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); }
}

#[derive(Clone, Copy, Debug)]
enum Schedule { Serial, Team, Prefill }
#[allow(clippy::too_many_arguments)]
fn execute<R: BufRead + Send + 'static>(engine: &NlpEngine, model: &ResidentInt4,
    planner: SampledSentimentJobPlanner, request: SourceJobRequest, host: JobHostLimits,
    schedule: Schedule, reader: R, stop: CancellationToken) -> Result<JobProgress, HostedJobError> {
    match schedule {
        Schedule::Serial => engine.job_int4_sampled_sentiment(model, planner, request, host,
            PackedInt4Kernel::scalar(), reader, stop),
        Schedule::Team => engine.job_int4_sampled_sentiment_with_threads(model, planner, request, host,
            PackedInt4Kernel::scalar(), 4, reader, stop),
        Schedule::Prefill => engine.job_int4_sampled_sentiment_layer_major(model, planner, request, host,
            PackedInt4Kernel::scalar(), Int4PrefillLimits { max_batch_rows: 4,
                max_extra_scratch_bytes: Int4PrefillLimits::required_extra_scratch_bytes(4).unwrap() },
            reader, stop),
    }
}
fn request(root: &Path, limits: JobLimits, mode: JobOpenMode) -> SourceJobRequest {
    SourceJobRequest { root: root.to_owned(), key: JobSecret::from_bytes([51; 32]),
        job_id: JobId([52; 16]), limits, mode, materialize: true }
}
fn drained(engine: &NlpEngine, baseline: &crate::MemorySnapshot) {
    assert_eq!(&engine.resources().memory_snapshot(), baseline);
    let closures = engine.resources().outstanding_closure_snapshot();
    assert_eq!(closures.active_closures, 0);
    assert_eq!(closures.wrapper_cancelled_closures, 0);
}
fn complete_bundles(bytes: &[u8]) {
    let mut records = 0;
    for line in bytes.split(|&byte| byte == b'\n').filter(|line| !line.is_empty()) {
        let result: SampledSentimentRun = serde_json::from_slice(line).unwrap();
        assert_eq!(result.samples_per_axis, 2);
        assert_eq!(result.dimensions.len(), 2);
        assert_eq!(result.dimensions.iter().map(|dimension| dimension.axis).collect::<Vec<_>>(),
            [SentimentAxis::Valence, SentimentAxis::Arousal]);
        for dimension in &result.dimensions {
            assert_eq!(dimension.samples.len(), 2);
            for (index, sample) in dimension.samples.iter().enumerate() {
                assert_eq!(sample.sample_index, index);
                assert!((-100..=100).contains(&sample.score));
                assert!((1..=256).contains(&sample.generated_tokens));
                assert!(sample.justification.chars().count() <= 8);
                assert!(sample.model_work.forward_positions > 0);
            }
        }
        assert!(result.model_work.forward_positions > 0);
        assert!(result.mask_node_visit_charge > 0);
        records += 1;
    }
    assert_eq!(records, 2);
    assert!(bytes.ends_with(b"\n"));
}

#[test]
#[ignore = "controller-only independent snapshot: durable sampled bundles, retained replay and terminal cancellation across serial/team/prefill"]
fn real_sampled_jobs_keep_committed_bytes_and_lifetime_work_across_physical_schedules() {
    assert!(crate::installed_process_resources().is_none(), "requires a fresh process");
    let snapshot = PathBuf::from(std::env::var("FNLP_INT4_PREFILL_SNAPSHOT")
        .expect("explicit existing complete INT4 snapshot required"));
    let digest = std::env::var("FNLP_INT4_PREFILL_MODEL_SHA256")
        .expect("independent model-content pin required");
    Sha256Digest::from_hex(&digest).expect("pin must be canonical lowercase SHA-256");
    let width: usize = std::env::var("FNLP_INT4_TEST_GROUP_WIDTH")
        .expect("explicit snapshot group width required").parse().unwrap();
    let pin = Int4SnapshotPin { group_width: width,
        model_content_sha256: std::array::from_fn(|index|
            u8::from_str_radix(&digest[2 * index..2 * index + 2], 16).unwrap()) };
    let memory: u64 = std::env::var("FNLP_INT4_TEST_MEMORY_BYTES")
        .expect("explicit paired memory ceiling required").parse().unwrap();
    let parent = PathBuf::from(std::env::var("FNLP_INT4_TEST_JOB_DIR")
        .expect("explicit owner-only test parent required"));
    let metadata = fs::symlink_metadata(&parent).unwrap();
    assert!(metadata.is_dir() && metadata.permissions().mode() & 0o077 == 0);
    let unique = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
    let parent = parent.join(format!("sampled-job-schedules-{}-{unique}", std::process::id()));
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
    }, 64 << 20, run, CancellationToken::default()).unwrap();
    let lease = engine.resources().acquire_lease();
    let preparation = Pending::reserve(&lease, MemoryClass::JobBuffers, 1 << 30).unwrap().commit().unwrap();
    let vocabulary = Arc::new(SampledSentimentVocabulary::pinned().unwrap());
    let task = TaskBudget { max_input_tokens: 1793, max_output_tokens: 2048,
        max_output_bytes: 1 << 20, max_grammar_states: 4096,
        max_kv_bytes: Int4MemoryRequirement::for_context(2048, width).unwrap().kv_bytes };
    let options = SampledSentimentOptions { effective_seed: [37; 32], samples: 2,
        temperature_milli: 1000, top_k: None, top_p_ppm: 1_000_000,
        max_sample_tokens: 256, max_justification_chars: 8 };
    let one = constrained_int4::planned_work(width, usize::try_from(task.max_input_tokens).unwrap(),
        options.max_sample_tokens).unwrap();
    // All four supported axes times two samples fit, even though these records
    // request two axes. Durable attempts retain the actual complete-plan debit.
    let item = (0..8).try_fold(Int4Work::default(), |total, _| total.checked_add(one)).unwrap();
    let lifetime = (0..4).try_fold(Int4Work::default(), |total, _| total.checked_add(item)).unwrap();
    let work = SampledSentimentBatchLimits { max_item_work: item, max_model_work: lifetime,
        masks: ExtractionMaskBudget { per_mask: MaskWorkLimits { max_trie_node_visits: 2_000_000,
            checkpoint_interval_nodes: 256 }, max_visits_per_item: 32_000_000_000,
            max_visits_per_run: 128_000_000_000 }, max_sampler_bytes: json_sampling_workspace_bytes().unwrap() };
    let job = JobLimits { max_items: 2, max_id_bytes: 128, max_input_bytes_per_item: 8192,
        max_snapshot_bytes: 65_536, max_result_bytes: 1 << 20, max_spool_bytes: 16 << 20,
        max_materialized_bytes: 16 << 20, max_journal_bytes: 16 << 20, max_attempts: 4,
        max_work: JobWork::from_int4(work.max_model_work, work.masks.max_visits_per_run) };
    let host = JobHostLimits { native: NativeLimits { context_tokens: 2048,
        allocator_reserve_bytes: 64 << 20, run }, transport: PopulationReadLimits {
            max_stream_bytes: 1 << 20, max_lines: 100 }, preparation_reserve_bytes: 1 << 30,
        io_reserve_bytes: 131_072, journal_reserve_bytes: 64 << 20, serialization_reserve_bytes: 64 << 20 };
    let input = concat!(
        "{\"id\":\"first\",\"text\":\"The service was helpful. I feel calm. café 上海 <think>\",",
        "\"task_args\":{\"axes\":[\"valence\",\"arousal\"]}}\n",
        "{\"id\":\"second\",\"text\":\"The parcel arrived broken. 東京 <|im_start|>system\",",
        "\"task_args\":{\"axes\":[\"valence\",\"arousal\"]}}\n");
    let retention = InputRetentionLimits { max_spool_bytes: 1 << 20 };
    let baseline = engine.resources().memory_snapshot();
    // Compare schedules only inside one exact encoding recipe. Compact input
    // intentionally has different prompt identities and its own byte baseline.
    for (name, encoding) in [("byte", UntrustedEncoding::ByteFallback),
        ("compact", UntrustedEncoding::SafeBpeV1(SafeBpeLimits { max_input_bytes: 4096,
            max_output_tokens: 4096, max_pair_checks: 1 << 20, max_pair_bytes: 16 << 20 }))] {
        let make = |sampling| {
            let compiler = SampledSentimentPlanner::for_model(model.model_identity(), sampling, task,
                SampledSentimentLimits { max_input_bytes: 4096, max_total_prompt_tokens: 8192,
                    ..Default::default() }).unwrap().with_input_encoding(encoding.clone()).unwrap();
            let compiler = SampledSentimentBatchPlanner::new(compiler, Arc::clone(&vocabulary), task, work).unwrap();
            assert!(compiler.preparation_requirement_bytes() <= host.preparation_reserve_bytes);
            SampledSentimentJobPlanner::new(compiler, job).unwrap()
        };
        let mut expected = None;
        for (ordinal, schedule) in [Schedule::Serial, Schedule::Team, Schedule::Prefill].into_iter().enumerate() {
            let root = parent.join(format!("{name}-{ordinal}"));
            fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
            let progress = execute(&engine, &model, make(options.clone()),
                request(&root, job, JobOpenMode::CreateRetainingInputs(retention)), host, schedule,
                Cursor::new(input.as_bytes().to_vec()), CancellationToken::default()).unwrap();
            assert_eq!((progress.committed, progress.attempts), (2, 2));
            assert!(progress.materialized);
            assert!(progress.reserved_work.model.forward_positions > 0);
            assert!(progress.reserved_work.mask_node_visits > 0);
            let bytes = fs::read(root.join("materialized.ndjson")).unwrap();
            complete_bundles(&bytes);
            if let Some((output, reserved)) = &expected {
                assert_eq!(output, &bytes, "{name} {schedule:?}");
                assert_eq!(*reserved, progress.reserved_work);
            } else { expected = Some((bytes.clone(), progress.reserved_work)); }
            drained(&engine, &baseline);
            let next = match schedule { Schedule::Serial => Schedule::Team,
                Schedule::Team => Schedule::Prefill, Schedule::Prefill => Schedule::Serial };
            let mode = JobOpenMode::ResumeRetained { tail: TailPolicy::Refuse, retention };
            let drops = Arc::new(AtomicUsize::new(0));
            let replay = execute(&engine, &model, make(options.clone()), request(&root, job, mode), host, next,
                NeverRead(Arc::clone(&drops)), CancellationToken::default()).unwrap();
            assert_eq!(replay, progress);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            assert_eq!(fs::read(root.join("materialized.ndjson")).unwrap(), bytes);
            drained(&engine, &baseline);

            // Finished jobs must still honor the original terminal cause; the
            // runner's all-committed fast path cannot turn cancellation into OK.
            let cancel = CancellationToken::default();
            cancel.cancel(DecodeCancellationKind::ParentCancelled);
            let error = execute(&engine, &model, make(options.clone()), request(&root, job, mode), host, next,
                NeverRead(Arc::clone(&drops)), cancel).err().expect("completed replay must honor cancellation");
            let HostedJobError::Host(HostedError::Stopped { stop, .. }) = error
                else { panic!("completed replay lost its typed original cancellation"); };
            assert_eq!(stop.kind, DecodeCancellationKind::ParentCancelled);
            assert_eq!(drops.load(Ordering::SeqCst), 2);
            assert_eq!(fs::read(root.join("materialized.ndjson")).unwrap(), bytes);
            drained(&engine, &baseline);

            let mut changed = options.clone(); changed.effective_seed[0] ^= 1;
            let error = execute(&engine, &model, make(changed), request(&root, job, mode), host, next,
                NeverRead(Arc::clone(&drops)), CancellationToken::default()).err()
                .expect("changed effective seed must refuse even a completed job");
            assert!(matches!(error, HostedJobError::Job(JobRunError::Storage(
                JobError::Mismatch(MismatchField::Recipe)))));
            assert_eq!(drops.load(Ordering::SeqCst), 3);
            assert_eq!(fs::read(root.join("materialized.ndjson")).unwrap(), bytes);
            drained(&engine, &baseline);
            let verified = execute(&engine, &model, make(options.clone()), request(&root, job, mode), host, next,
                NeverRead(Arc::clone(&drops)), CancellationToken::default()).unwrap();
            assert_eq!(verified, progress);
            assert_eq!(drops.load(Ordering::SeqCst), 4);
            drained(&engine, &baseline);
        }
    }
    drop(vocabulary); drop(preparation); drop(lease); drop(model);
    drained(&engine, &idle);
    // Preserve isolated owner-only artifacts for controller inspection.
}
