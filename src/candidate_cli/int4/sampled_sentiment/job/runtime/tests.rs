//! Durable sampled-sentiment admission and metadata publication boundaries.
use super::*;
use crate::{
    hosted::{Int4HostError, RunStop},
    jobs::{JobId, JobProgress, JobWork, runner::JobRunError},
    native_engine::{constrained::JsonDecodeError, constrained_int4::Int4JsonError,
        decode::DecodeCancellationKind,
        strict_int4::{Int4ModelError, StrictInt4Error, weights::snapshot::Int4SnapshotError}},
};

struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unexpected private input access"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("unexpected metadata publication"); }
    fn flush(&mut self) -> io::Result<()> { panic!("unexpected metadata flush"); }
}
struct NoSerialize;
impl Serialize for NoSerialize {
    fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        panic!("stopped metadata must not be serialized");
    }
}
fn parts(operation: &str, extra: &[&str]) -> (JobCommand, Int4Args) {
    let mut words = ["source-int4", "sentiment-sampled-job", operation,
        "--source-dir", "private-source", "--source-manifest", "private-manifest.json",
        "--memory-mib", "65536", "--job-dir", "private-job", "--key-file", "private-key",
        "--limits", "private-limits.json", "--job-id", "0123456789abcdef0123456789abcdef",
        "--store-results", "--seed"].into_iter().map(str::to_owned).collect::<Vec<_>>();
    words.push("ab".repeat(32));
    words.extend(extra.iter().map(|word| (*word).to_owned()));
    let matches = crate::candidate_cli::int4::definition().try_get_matches_from(words).unwrap();
    let Int4Command { task: Int4Task::SampledSentimentJob, args, batch: None,
        job: Some(Int4JobCommand::SampledSentiment(command)) } = Int4Command::from_matches(&matches).unwrap()
        else { panic!("lost owned sampled sentiment job"); };
    (command, args)
}
fn lifetime(job: &JobArgs, args: &Int4Args, common: Int4Limits) -> JobLimits {
    JobLimits { max_items: 3, max_id_bytes: 128, max_input_bytes_per_item: args.max_input_bytes,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: common.result_bytes,
        max_spool_bytes: 16 * MIB, max_materialized_bytes: 16 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 5,
        max_work: JobWork::from_int4(job.sampled.work_ceiling(), job.sampled.max_mask_node_visits) }
}

#[test]
fn runtime_revalidates_storage_and_retention_before_runtime_or_private_io() {
    for case in 0..3 {
        let (mut command, args) = parts("resume", &[]);
        let admitted = command.args.validate(&args).unwrap();
        let expected = match case {
            0 => { command.args.storage.max_stream_mib = 0; "explicit_sampled_sentiment_job_storage" }
            1 => { command.args.storage.store_results = false; "explicit_sampled_sentiment_job_storage" }
            _ => {
                command.args.storage.input_retention.spool_input = true;
                command.args.storage.input_retention.max_retained_input_mib = Some(8);
                "job_input_retention_options"
            }
        };
        let failure = execute(command.mode, args, command.args, admitted, NoIo, &mut NoIo).unwrap_err();
        assert_eq!(failure.exit, ErrorCode::Usage);
        assert_eq!(failure.code, expected);
    }
}

#[test]
fn expired_config_read_refuses_before_opening_the_private_path() {
    let failure = read_config(std::path::Path::new("private-limits-that-must-not-be-opened.json"),
        job_cli::LIMIT_BYTES, &|| Err(CandidateError::Timeout)).unwrap_err();
    assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(failure.code, Failure::from(CandidateError::Timeout).code);
}

#[test]
fn original_and_retained_input_quotes_match_the_host_and_preserve_invocation_control() {
    let (original, args) = parts("resume", &[]);
    let common = original.args.validate(&args).unwrap();
    let fixed = lifetime(&original.args, &args, common);
    let original_limits = original.args.lifetime_limits(&args, common, fixed).unwrap();
    for (operation, extra, input_memory) in [
        ("resume", &[][..], JobInputMemory::CompletePopulation),
        ("start", &["--spool-input", "--max-retained-input-mib", "8"][..], JobInputMemory::RetainedRecord),
        ("resume", &["--use-retained-input", "--max-retained-input-mib", "8"][..], JobInputMemory::RetainedRecord),
    ] {
        let (command, args) = parts(operation, extra);
        let common = command.args.validate(&args).unwrap();
        command.args.storage.input_retention.validate(command.mode, &args.input).unwrap();
        assert_eq!(command.args.storage.input_retention.input_memory().unwrap(), input_memory);
        let limits = command.args.lifetime_limits(&args, common, fixed).unwrap();
        let remaining = Duration::from_millis(17_321);
        let host = host_limits(&args, limits, remaining);
        check_host_quote(host, fixed, limits.buffer_bytes, input_memory).unwrap();
        assert_eq!(host.required_buffer_bytes_for_input(fixed, input_memory).unwrap(), limits.buffer_bytes);
        for wrong in [limits.buffer_bytes - 1, limits.buffer_bytes + 1] {
            assert_eq!(check_host_quote(host, fixed, wrong, input_memory).unwrap_err().code,
                "job_host_memory_quote");
        }
        let mut invalid = host; invalid.journal_reserve_bytes = 0;
        assert_eq!(check_host_quote(invalid, fixed, limits.buffer_bytes, input_memory).unwrap_err().code,
            "job_host_admission");
        assert_eq!(host.native.context_tokens, args.context_tokens);
        assert_eq!(host.native.allocator_reserve_bytes, ALLOCATOR_BYTES);
        assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
        assert_eq!(host.native.run.max_elapsed, remaining);
        assert_eq!(host.native.run.cleanup_reserve_bytes, CLEANUP_BYTES);
        assert_eq!(host.transport.max_lines, command.args.storage.max_input_lines);
        assert_eq!(host.transport.max_stream_bytes, command.args.storage.max_stream_mib * MIB);
        assert_eq!(host.preparation_reserve_bytes, common.preparation_bytes);
        assert_eq!(JobWork::from_int4(limits.native.max_model_work,
            limits.native.masks.max_visits_per_run), fixed.max_work);
        let removed = if input_memory == JobInputMemory::RetainedRecord { 2 * fixed.max_snapshot_bytes } else { 0 };
        assert_eq!(original_limits.buffer_bytes - limits.buffer_bytes, removed);
        assert_eq!(original_limits.common.minimum_memory_bytes - limits.common.minimum_memory_bytes, removed);
    }
}

fn assert_public_metadata(value: &serde_json::Value) {
    match value {
        serde_json::Value::Object(fields) => for (key, value) in fields {
            assert!(!["text", "documents", "results", "score", "scores", "samples", "sampling",
                "seed", "effective_seed", "policy", "recipe", "options", "task_args", "axes",
                "justification", "justifications", "key", "job_dir"].contains(&key.as_str()),
                "private field in metadata: {key}");
            assert_public_metadata(value);
        },
        serde_json::Value::Array(values) => for value in values { assert_public_metadata(value); },
        _ => {},
    }
}

#[test]
fn completion_metadata_binds_the_candidate_and_all_six_spent_work_counters_without_private_samples() {
    let (command, args) = parts("resume", &[]);
    let common = command.args.validate(&args).unwrap();
    let fixed = lifetime(&command.args, &args, common);
    let id = JobId([0x12; 16]);
    let mut spent = fixed.max_work;
    spent.model.forward_positions = 11;
    spent.model.projected_logits = 13;
    spent.model.attention_pairs = 17;
    spent.model.projections.dot_products = 19;
    spent.model.projections.multiply_accumulates = 23;
    spent.mask_node_visits = 29;
    let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: spent, spool_bytes: 1024, materialized: true };
    let report = progress_report(progress, id, fixed, true,
        RunMode::Resume { discard_uncommitted: true }, "sentiment-sampled").unwrap();
    let content = "cd".repeat(32); let source = "ef".repeat(32);
    let response = Response {
        protocol: "fnlp-candidate-int4-sentiment-sampled-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "sentiment-sampled",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-public-revision",
        source_root_sha256: &source, model_content_sha256: &content,
        candidate_format: "fixture-candidate-format", quant_recipe: "fixture-grouped-int4",
        group_width: args.group_width, integer_kernel: common.kernel.kind().id(),
        snapshot: Some(SnapshotRecord::new("loaded", common.snapshot_need)),
        input_encoding: None, compute_schedule: None, prefill_schedule: None, output: &report,
    };
    let mut bytes = Vec::new();
    publish_report(&response, &mut bytes, &|| Ok(Duration::from_secs(1)),
        &CancellationToken::default()).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["protocol"], "fnlp-candidate-int4-sentiment-sampled-job-v1");
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["scope"], "real-source-current-candidate-int4");
    assert_eq!(value["evidence"], "non_authoritative");
    assert_eq!(value["task"], "sentiment-sampled");
    assert_eq!(value["model_content_sha256"], content);
    assert_eq!(value["source_root_sha256"], source);
    assert_eq!(value["group_width"], args.group_width);
    assert_eq!(value["snapshot"]["operation"], "loaded");
    assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
    assert_eq!(value["snapshot"]["file_bytes"], common.snapshot_need.file_bytes);
    assert_eq!(value["output"]["operation"], "resume");
    assert_eq!(value["output"]["task"], "sentiment-sampled");
    assert_eq!(value["output"]["status"], "complete");
    assert_eq!(value["output"]["job_id"], "12".repeat(16));
    assert_eq!(value["output"]["items"], 2);
    assert_eq!(value["output"]["committed"], 2);
    assert_eq!(value["output"]["attempts"], 3);
    assert_eq!(value["output"]["materialized"], true);
    assert_eq!(value["output"]["committed_spool_bytes"], 1024);
    assert_eq!(value["output"]["reserved_work"], serde_json::json!({
        "model": { "forward_positions": 11, "projected_logits": 13, "attention_pairs": 17,
            "projections": { "dot_products": 19, "multiply_accumulates": 23 } },
        "mask_node_visits": 29,
    }));
    assert_public_metadata(&value);
    assert!(!std::str::from_utf8(&bytes).unwrap().contains(&"ab".repeat(32)));
    assert!(bytes.ends_with(b"\n"));
    assert_eq!(publish(&response, 1, &mut NoIo), Err(CandidateError::Output));
    for key in ["input_encoding", "compute_schedule", "prefill_schedule"] {
        assert!(value.get(key).is_none());
    }
    let encoding = crate::tokenizer::untrusted::UntrustedEncoding::SafeBpeV1(
        crate::tokenizer::untrusted::SafeBpeLimits { max_input_bytes: 8192, max_output_tokens: 4096,
            max_pair_checks: 123456, max_pair_bytes: 2345678 });
    let mut response = response;
    response.input_encoding = Some(&encoding);
    response.compute_schedule = compute::Schedule::for_threads(4);
    let mut threaded = serde_json::to_value(&response).unwrap();
    assert_eq!(threaded.as_object_mut().unwrap().remove("input_encoding"),
        Some(serde_json::to_value(&encoding).unwrap()));
    assert_eq!(threaded.as_object_mut().unwrap().remove("compute_schedule"),
        compute::Schedule::for_threads(4).map(serde_json::to_value).transpose().unwrap());
    assert_eq!(threaded, value);
    let (prefill_command, prefill_args) = parts("resume", &["--prefill-rows", "3"]);
    let prefill = prefill_command.args.validate(&prefill_args).unwrap().prefill.unwrap();
    response.input_encoding = None;
    response.compute_schedule = None;
    response.prefill_schedule = Some(PrefillSchedule::new(prefill).unwrap());
    let mut layer_major = serde_json::to_value(&response).unwrap();
    assert_eq!(layer_major.as_object_mut().unwrap().remove("prefill_schedule"), Some(serde_json::json!({
        "execution": INT4_LAYER_MAJOR_PREFILL_VERSION, "max_batch_rows": 3,
        "admitted_workspace_bytes": prefill.validate().unwrap(),
    })));
    assert_eq!(layer_major, value);
}

fn cancellations() -> [(DecodeCancellationKind, ErrorCode, &'static str, &'static str); 11] {
    use DecodeCancellationKind::*;
    [
        (User, ErrorCode::Cancelled, "sampled_sentiment_cancelled", "job_cancelled"),
        (Timeout, ErrorCode::BudgetOrTimeout, "sampled_sentiment_budget", "job_execution_budget"),
        (Deadline, ErrorCode::BudgetOrTimeout, "sampled_sentiment_budget", "job_execution_budget"),
        (PollQuota, ErrorCode::BudgetOrTimeout, "sampled_sentiment_budget", "job_execution_budget"),
        (CostBudget, ErrorCode::BudgetOrTimeout, "sampled_sentiment_budget", "job_execution_budget"),
        (FailFast, ErrorCode::Generic, "sampled_sentiment_supervised_failure", "job_supervised_failure"),
        (RaceLost, ErrorCode::Generic, "sampled_sentiment_root_race", "job_internal_race_lost"),
        (ParentCancelled, ErrorCode::Cancelled, "sampled_sentiment_cancelled", "job_cancelled"),
        (ResourceUnavailable, ErrorCode::AdmissionOrResourceLimit, "sampled_sentiment_resources", "job_resource_limit"),
        (Shutdown, ErrorCode::Cancelled, "sampled_sentiment_cancelled", "job_cancelled"),
        (LinkedExit, ErrorCode::Generic, "sampled_sentiment_supervised_failure", "job_supervised_failure"),
    ]
}

#[test]
fn original_cancellation_is_preserved_across_host_native_task_and_durable_boundaries() {
    for (cause, exit, sampled_code, job_code) in cancellations() {
        let stop = HostedError::Stopped { stop: RunStop { kind: cause,
            runtime_reason: None, runtime_checkpoint_error: None }, task_error: None };
        for error in [stop,
            HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::SampledSentiment(SampledSentimentError::Cancelled(cause))),
        ] {
            let actual = execution_failure(HostedJobError::Host(error));
            assert_eq!(actual.exit, exit); assert_eq!(actual.code, sampled_code);
        }
        for error in [SampledSentimentError::Cancelled(cause),
            SampledSentimentError::Native(Int4JsonError::Native(StrictInt4Error::Cancelled(cause))),
            SampledSentimentError::Native(Int4JsonError::Decode(JsonDecodeError::Cancelled(cause))),
        ] {
            let actual = task_failure(error);
            assert_eq!(actual.exit, exit); assert_eq!(actual.code, sampled_code);
        }
        for error in [JobRunError::Storage(JobError::Cancelled(cause)),
            JobRunError::Processor(crate::batch::BatchFault::cancelled(cause))] {
            let actual = execution_failure(HostedJobError::Job(error));
            assert_eq!(actual.exit, exit); assert_eq!(actual.code, job_code);
        }
    }
}

#[test]
fn supervised_failures_keep_the_typed_error_and_ordinary_stops_keep_the_original_cause() {
    use DecodeCancellationKind::*;
    for cause in [FailFast, LinkedExit] {
        let actual = execution_failure(HostedJobError::Host(HostedError::Stopped {
            stop: RunStop { kind: cause, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::Int4(Int4HostError::SampledSentiment(
                SampledSentimentError::NoResult)))),
        }));
        assert_eq!(actual.exit, ErrorCode::StructuredTaskNoResult);
        assert_eq!(actual.code, "sampled_sentiment_no_result");
    }
    let actual = execution_failure(HostedJobError::Host(HostedError::Stopped {
        stop: RunStop { kind: User, runtime_reason: None, runtime_checkpoint_error: None },
        task_error: Some(Box::new(HostedError::Limits("fixture memory"))),
    }));
    assert_eq!(actual.exit, ErrorCode::Cancelled);
    assert_eq!(actual.code, "sampled_sentiment_cancelled");
    let panicked = execution_failure(HostedJobError::Host(HostedError::Panicked));
    assert_eq!(panicked.exit, ErrorCode::Generic);
}

#[test]
fn bounded_metadata_and_original_deadline_are_checked_before_staging_and_before_the_first_byte() {
    let cancellation = CancellationToken::default();
    let available = || Ok(Duration::from_secs(1));
    let expired = publish_report(&NoSerialize, &mut NoIo, &|| Err(CandidateError::Timeout),
        &cancellation).unwrap_err();
    assert_eq!(expired.exit, ErrorCode::BudgetOrTimeout);
    let oversized = "x".repeat(job_cli::REPORT_BYTES);
    assert_eq!(publish_report(&oversized, &mut NoIo, &available, &cancellation).unwrap_err().code,
        Failure::from(CandidateError::Output).code);
    let checks = std::cell::Cell::new(0);
    let expired_before_write = || {
        checks.set(checks.get() + 1);
        if checks.get() == 1 { available() } else { Err(CandidateError::Timeout) }
    };
    assert_eq!(publish_report(&true, &mut NoIo, &expired_before_write, &cancellation).unwrap_err().exit,
        ErrorCode::BudgetOrTimeout);
    assert_eq!(checks.get(), 2);
}

#[test]
fn expired_delivery_and_failed_flush_never_report_success_after_metadata_is_written() {
    let cancellation = CancellationToken::default();
    let checks = std::cell::Cell::new(0);
    let expired_after_flush = || {
        checks.set(checks.get() + 1);
        if checks.get() < 3 { Ok(Duration::from_secs(1)) } else { Err(CandidateError::Timeout) }
    };
    let mut delivered = Vec::new();
    assert_eq!(publish_report(&true, &mut delivered, &expired_after_flush, &cancellation).unwrap_err().exit,
        ErrorCode::BudgetOrTimeout);
    assert_eq!(delivered, b"true\n");
    assert_eq!(checks.get(), 3);
    struct BrokenFlush(Vec<u8>);
    impl Write for BrokenFlush {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("fixture flush")) }
    }
    let mut output = BrokenFlush(Vec::new());
    let failure = publish_report(&true, &mut output, &|| Ok(Duration::from_secs(1)),
        &cancellation).unwrap_err();
    assert_eq!(failure.code, Failure::from(CandidateError::Output).code);
    assert_eq!(output.0, b"true\n");
}

#[test]
fn every_original_cancellation_prevents_serialization_and_wins_over_an_expired_deadline() {
    for (cause, exit, _, code) in cancellations() {
        let cancellation = CancellationToken::default();
        cancellation.cancel(cause);
        cancellation.cancel(DecodeCancellationKind::User);
        let failure = publish_report(&NoSerialize, &mut NoIo, &|| Err(CandidateError::Timeout),
            &cancellation).unwrap_err();
        assert_eq!(cancellation.cause(), Some(cause));
        assert_eq!(failure.exit, exit); assert_eq!(failure.code, code);
    }
}

#[test]
fn cancellation_during_serialization_prevents_the_first_byte_and_retains_the_first_cause() {
    struct CancelWhileSerializing<'a>(&'a CancellationToken);
    impl Serialize for CancelWhileSerializing<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.0.cancel(DecodeCancellationKind::ParentCancelled);
            self.0.cancel(DecodeCancellationKind::Deadline);
            serializer.serialize_bool(true)
        }
    }
    let cancellation = CancellationToken::default();
    let failure = publish_report(&CancelWhileSerializing(&cancellation), &mut NoIo,
        &|| Ok(Duration::from_secs(1)), &cancellation).unwrap_err();
    assert_eq!(cancellation.cause(), Some(DecodeCancellationKind::ParentCancelled));
    assert_eq!(failure.exit, ErrorCode::Cancelled);
    assert_eq!(failure.code, "job_cancelled");
}

#[test]
fn cancellation_during_write_or_flush_fails_after_delivery_with_the_first_cause() {
    struct CancelOnDelivery { bytes: Vec<u8>, cancellation: CancellationToken, at_flush: bool }
    impl Write for CancelOnDelivery {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(bytes);
            if !self.at_flush { self.cancellation.cancel(DecodeCancellationKind::ResourceUnavailable); }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            if self.at_flush { self.cancellation.cancel(DecodeCancellationKind::ResourceUnavailable); }
            self.cancellation.cancel(DecodeCancellationKind::User);
            Ok(())
        }
    }
    for at_flush in [false, true] {
        let cancellation = CancellationToken::default();
        let mut output = CancelOnDelivery { bytes: Vec::new(), cancellation: cancellation.clone(), at_flush };
        let checks = std::cell::Cell::new(0);
        let remaining = || {
            checks.set(checks.get() + 1);
            if checks.get() < 3 { Ok(Duration::from_secs(1)) } else { Err(CandidateError::Timeout) }
        };
        let failure = publish_report(&true, &mut output, &remaining, &cancellation).unwrap_err();
        assert_eq!(output.bytes, b"true\n");
        assert_eq!(cancellation.cause(), Some(DecodeCancellationKind::ResourceUnavailable));
        assert_eq!(failure.exit, ErrorCode::AdmissionOrResourceLimit);
        assert_eq!(failure.code, "job_resource_limit");
        assert_eq!(checks.get(), 2, "the recorded stop retains authority over the later deadline");
    }
}
