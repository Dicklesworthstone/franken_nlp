//! Cross-layer durable configuration, admission and terminal-cause regressions.
use super::*;
use crate::{
    batch::{BatchCode, BatchFault},
    candidate_cli::sampled_sentiment::job::tests::{arguments, lifetime, NoIo},
    hosted::RunStop,
    jobs::{JobId, JobProgress},
};

fn facts() -> ArtifactIdentity {
    ArtifactIdentity { model_id: "Nanbeige4.2-3B".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "sampled-job-cli-planning-fixture".to_owned(),
        source_root_sha256: "ab".repeat(32), logical_model_sha256: "cd".repeat(32) }
}
fn configured(flags: &[&str], job: JobLimits, vocabulary: Arc<SampledSentimentVocabulary>)
    -> Int8SampledSentimentJobPlanner {
    let (_, args) = arguments("resume", flags);
    let (common, limits) = args.validate().unwrap();
    let envelope = args.lifetime_limits(limits, job).unwrap();
    prepare(&args, &common, limits, job, envelope, &facts(), vocabulary).unwrap()
}
fn quoted_host(args: &JobArgs, common: &CandidateArgs, limits: Limits) -> JobHostLimits {
    JobHostLimits { native: NativeLimits { context_tokens: common.context_tokens,
        allocator_reserve_bytes: 64 * MIB, run: RunLimits { max_elapsed: Duration::from_secs(30),
            max_checkpoints: 100_000, cleanup_reserve_bytes: 65_536 } },
        transport: PopulationReadLimits { max_stream_bytes: args.storage.max_stream_mib * MIB,
            max_lines: args.storage.max_input_lines },
        preparation_reserve_bytes: limits.preparation_bytes,
        io_reserve_bytes: (2 * crate::candidate_cli::batch::IO_BUFFER_BYTES) as u64,
        journal_reserve_bytes: args.storage.journal_memory_mib * MIB,
        serialization_reserve_bytes: args.storage.serialization_memory_mib * MIB }
}

#[test]
fn parsed_seed_caps_tokenizer_totals_and_prompt_policy_reach_the_authenticated_factory() {
    let (_, args) = arguments("start", &[]);
    let (_, limits) = args.validate().unwrap();
    let job = lifetime(&args, limits);
    let vocabulary = Arc::new(SampledSentimentVocabulary::pinned().unwrap());
    let base = configured(&[], job, Arc::clone(&vocabulary));
    let original = canonjson::canonical_bytes(base.job_recipe()).unwrap();
    let other_seed = "5b".repeat(32);
    for flags in [
        vec!["--seed", other_seed.as_str()], vec!["--samples", "3"], vec!["--top-k", "7"],
        vec!["--max-forward-positions", "65536"], vec!["--max-mask-node-visits", "16000000000"],
        vec!["--prefill-rows", "4"],
    ] {
        let changed = configured(&flags, job, Arc::clone(&vocabulary));
        assert_ne!(canonjson::canonical_bytes(changed.job_recipe()).unwrap(), original);
    }
    for width in ["2", "4"] {
        let changed = configured(&["--compute-threads", width], job, Arc::clone(&vocabulary));
        assert_eq!(canonjson::canonical_bytes(changed.job_recipe()).unwrap(), original,
            "physical row width must not change the private sampler or replay contract");
    }
    let compact = configured(&["--input-encoding", "safe-bpe-v1"], job, Arc::clone(&vocabulary));
    let narrowed = configured(&["--input-encoding", "safe-bpe-v1",
        "--max-total-tokenizer-pair-checks", "1000"], job, vocabulary);
    assert_ne!(canonjson::canonical_bytes(compact.job_recipe()).unwrap(),
        canonjson::canonical_bytes(narrowed.job_recipe()).unwrap());
    let mut drift = job; drift.max_work.mask_node_visits += 1;
    assert_eq!(base.check_job_limits(drift).unwrap_err().code, BatchCode::InvalidLimits);
}

#[test]
fn shared_quote_prices_original_and_retained_populations_at_the_exact_memory_boundary() {
    for scheduling in [vec![], vec!["--compute-threads", "4", "--prefill-rows", "4"]] {
        for retained in [false, true] {
            for compact in [false, true] {
                let mut flags = scheduling.clone();
                if retained { flags.extend(["--use-retained-input", "--max-retained-input-mib", "32"]); }
                if compact { flags.extend(["--input-encoding", "safe-bpe-v1"]); }
                let (_, args) = arguments("resume", &flags);
                let (common, limits) = args.validate().unwrap();
                let job = lifetime(&args, limits);
                let envelope = args.lifetime_limits(limits, job).unwrap();
                let host = quoted_host(&args, &common, limits);
                let required = quote(&args, envelope, job, limits, host).unwrap();
                assert_eq!(required.input_bytes, host.required_buffer_bytes_for_input(job,
                    args.storage.input_retention.input_memory().unwrap()).unwrap());
                assert_eq!(required.output_reservation_bytes,
                    4 * envelope.task.max_output_bytes + 8 * u64::from(envelope.task.max_output_tokens) + 4096);
                assert_eq!(required.team_workspace_bytes, limits.compute.workspace_bytes);
                assert!(required.workspace_bytes >= envelope.sampler_bytes);
                let peak = minimum_memory(limits, required).unwrap();
                let mut exact = limits; exact.memory_bytes = peak;
                check_memory(exact, required).unwrap();
                exact.memory_bytes -= 1;
                assert_eq!(check_memory(exact, required).unwrap_err().code,
                    "int8_sampled_sentiment_job_population_memory");
                let mut overflow = limits; overflow.weight_bytes = u64::MAX;
                assert!(minimum_memory(overflow, required).is_err());
                let mut overflow = host; overflow.serialization_reserve_bytes = u64::MAX;
                assert!(quote(&args, envelope, job, limits, overflow).is_err());
            }
        }
    }
}

#[test]
fn runtime_rechecks_storage_and_schedule_before_installing_a_host_or_opening_private_files() {
    for case in 0..4 {
        let (mode, mut args) = arguments("resume", &["--compute-threads", "4", "--prefill-rows", "4"]);
        let (mut common, mut limits) = args.validate().unwrap();
        match case {
            0 => args.storage.store_results = false,
            1 => limits.compute.workspace_bytes -= 1,
            2 => common.compute = compute::Args::default(),
            _ => args.storage.input_retention.spool_input = true,
        }
        assert_eq!(execute(mode, args, common, limits, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
    }
}

fn stopped(cause: DecodeCancellationKind) -> HostedError {
    HostedError::Stopped { stop: RunStop { kind: cause,
        runtime_reason: None, runtime_checkpoint_error: None }, task_error: None }
}
const CAUSES: [DecodeCancellationKind; 11] = [
    DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
    DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
    DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled,
    DecodeCancellationKind::ResourceUnavailable, DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit,
];

#[test]
fn inspection_loading_storage_and_processor_errors_keep_the_original_cause_after_late_deadlines() {
    for cause in CAUSES {
        let expected = cancelled(cause);
        for stage in [
            SessionStageError::Artifact(ArtifactBridgeError::Cancelled(cause)),
            SessionStageError::Host(stopped(cause)),
            SessionStageError::Host(HostedError::Native(StrictInt8Error::Cancelled(cause))),
        ] {
            let result: Result<(), SessionStageError> = Err(stage);
            let error = checked_stage(result, || Err(CandidateError::Timeout), stage_failure).unwrap_err();
            assert_eq!(error.cancellation, Some(cause));
            assert_eq!(error.exit, expected.exit);
        }
        for error in [
            HostedJobError::Job(JobRunError::Storage(JobError::Cancelled(cause))),
            HostedJobError::Job(JobRunError::Processor(BatchFault::cancelled(cause))),
            HostedJobError::Host(stopped(cause)),
        ] {
            let failure = failure(error);
            assert_eq!(failure.cancellation, Some(cause)); assert_eq!(failure.exit, expected.exit);
        }
    }
    let error = checked_stage::<(), _>(Err(SessionStageError::Setup(CandidateError::Input)),
        || Err(CandidateError::Timeout), stage_failure).unwrap_err();
    assert_eq!(error.cancellation, Some(DecodeCancellationKind::Deadline));
}

#[test]
fn metadata_publication_contains_no_private_bundle_and_preserves_stops_from_io_callbacks() {
    let (_, args) = arguments("start", &[]);
    let (_, limits) = args.validate().unwrap();
    let job = lifetime(&args, limits); let id = JobId([9; 16]);
    let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: job.max_work, spool_bytes: 4096, materialized: true };
    let report = owned_jobs::progress_report(progress, id, job, true, RunMode::Start, "sentiment-sampled").unwrap();
    let response = Response { response: &report, protocol: "fnlp-candidate-int8-sentiment-sampled-job-v1",
        durable_unit: "complete_independent_sample_bundle", input_encoding: None,
        invocation_encoding_work_limits: None, compute_threads: None, prefill_rows: None };
    let encoded = serde_json::to_string(&response).unwrap();
    for private in ["effective_seed", "sampling", "dimensions", "justification", "task_args", "source_text"] {
        assert!(!encoded.contains(private));
    }
    struct NoSerialize;
    impl Serialize for NoSerialize {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            panic!("cancelled metadata must not serialize");
        }
    }
    struct Writer {
        token: CancellationToken, cause: DecodeCancellationKind, on_flush: bool, fail: bool,
        writes: usize, flushes: usize,
    }
    impl Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            if !self.on_flush {
                self.token.cancel(self.cause);
                if self.fail { return Err(io::Error::other("private reporting failure")); }
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            if self.on_flush {
                self.token.cancel(self.cause);
                if self.fail { return Err(io::Error::other("private reporting failure")); }
            }
            Ok(())
        }
    }
    for cause in CAUSES {
        let token = CancellationToken::default(); token.cancel(cause);
        let error = publish_report(&NoSerialize, &mut NoIo, &|| Ok(Duration::from_secs(1)), &token).unwrap_err();
        assert_eq!(error.cancellation, Some(cause));
        for on_flush in [false, true] { for fail in [false, true] {
            let token = CancellationToken::default();
            let mut writer = Writer { token: token.clone(), cause, on_flush, fail, writes: 0, flushes: 0 };
            let error = publish_report(&response, &mut writer, &|| Ok(Duration::from_secs(1)), &token).unwrap_err();
            assert_eq!(error.cancellation, Some(cause));
            assert_eq!(writer.writes, 1);
            assert_eq!(writer.flushes, usize::from(on_flush || !fail));
        } }
    }
}
