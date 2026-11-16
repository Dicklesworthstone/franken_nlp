//! Cross-layer admission and failure-contract regressions, without model weights.
use super::*;
use super::super::tests::{command, lifetime, NoIo};
use crate::{
    hosted::{Int4HostError, RunStop},
    jobs::{JobId, JobProgress, JobWork, JobInputMemory},
    native_engine::{constrained::JsonDecodeError, constrained_int4::Int4JsonError,
        strict_int4::{Int4ModelError, StrictInt4Error, weights::snapshot::Int4SnapshotError}},
    tasks::sentiment::sampled::SampledSentimentError,
};

struct NoSerialize;
impl Serialize for NoSerialize {
    fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        panic!("stopped metadata must not be serialized");
    }
}

#[test]
fn cli_peak_and_host_quotes_agree_for_every_model_input_and_schedule() {
    for route in 0..3 {
        for flags in [vec![], vec!["--compute-threads", "4"], vec!["--prefill-rows", "4"]] {
            for retained in [false, true] {
                let mut options = flags.clone();
                if retained {
                    options.extend(["--use-retained-input", "--max-retained-input-mib", "8"]);
                }
                let (command, args) = command("resume", route, &options);
                let common = command.args.validate(&args).unwrap();
                let job = lifetime(&args, &command, common);
                let limits = command.args.lifetime_limits(&args, common, job).unwrap();
                let encoding = args.input_encoding.policy(args.max_input_bytes, common.max_prompt_tokens).unwrap();
                let preparation = sampled_sentiment_job_preparation_bytes(limits.task, limits.planning,
                    &encoding).unwrap();
                let host = host_limits(&args, limits, Duration::from_secs(1));
                let quote = int4_sampled_sentiment_job_requirements(args.group_width, limits.task,
                    limits.native, preparation, job, host,
                    command.args.storage.input_retention.input_memory().unwrap(), common.prefill,
                    (common.compute.threads > 1).then_some(common.compute.threads)).unwrap();
                assert_eq!(quote.native, common.native);
                assert_eq!(quote.input_bytes, limits.buffer_bytes);
                assert_eq!(quote.output_reservation_bytes, 4 * limits.task.max_output_bytes + 4096);
                assert_eq!(quote.team_workspace_bytes, common.compute.workspace_bytes);
                assert_eq!(quote.input_bytes + quote.native.kv_bytes + quote.workspace_bytes
                    + quote.output_reservation_bytes + quote.team_workspace_bytes, limits.execution_bytes);
            }
        }
    }
}

#[test]
fn every_batch_failure_has_a_stable_resource_parse_budget_or_no_result_exit() {
    let groups: &[(ErrorCode, &[BatchCode])] = &[
        (ErrorCode::Usage, &[BatchCode::InvalidLimits]),
        (ErrorCode::InputDecodeOrParse, &[BatchCode::InvalidUtf8, BatchCode::InvalidJson,
            BatchCode::InvalidEnvelope, BatchCode::InvalidId, BatchCode::DuplicateId,
            BatchCode::Planning, BatchCode::InputIo]),
        (ErrorCode::BudgetOrTimeout, &[BatchCode::LineLimit, BatchCode::DocumentLimit,
            BatchCode::EpochIdLimit, BatchCode::InputLimit, BatchCode::RequestLimit,
            BatchCode::OutputLineLimit, BatchCode::OutputLimit, BatchCode::WorkLimit]),
        (ErrorCode::AdmissionOrResourceLimit, &[BatchCode::Admission, BatchCode::Allocation]),
        (ErrorCode::StructuredTaskNoResult, &[BatchCode::Execution]),
        (ErrorCode::Generic, &[BatchCode::Cancelled, BatchCode::SequenceOverflow,
            BatchCode::InvalidExecution, BatchCode::Serialization, BatchCode::OutputIo]),
    ];
    for &(expected, codes) in groups {
        for &code in codes {
            let failure = batch_failure(code.into());
            assert_eq!(failure.exit, expected);
            assert_eq!(failure.cancellation, None);
        }
    }
}

#[test]
fn durable_storage_processor_and_host_keep_the_original_cancellation_cause() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, LinkedExit, Shutdown] {
        let expected = cancelled(cause);
        let storage = job_failure(HostedJobError::Job(JobRunError::Storage(JobError::Cancelled(cause))));
        let processor = job_failure(HostedJobError::Job(JobRunError::Processor(BatchFault::cancelled(cause))));
        let host = job_failure(HostedJobError::Host(HostedError::BatchSetup(BatchFault::cancelled(cause))));
        for failure in [storage, processor, host] {
            assert_eq!(failure.cancellation, Some(cause));
            assert_eq!(failure.exit, expected.exit);
        }
    }
    let missing = batch_failure(BatchCode::Cancelled.into());
    assert_eq!(missing.code, "sampled_sentiment_job_missing_cancellation");
    assert_eq!(missing.exit, ErrorCode::Generic);
}

#[test]
fn invalid_envelopes_and_completion_invariants_remain_distinct() {
    let invalid = job_failure(HostedJobError::Job(JobRunError::InvalidEnvelope));
    assert_eq!(invalid.exit, ErrorCode::InputDecodeOrParse);
    assert_eq!(invalid.code, "sampled_sentiment_job_envelope");
    for error in [JobRunError::WorkContract, JobRunError::Stopped] {
        let failure = job_failure(HostedJobError::Job(error));
        assert_eq!(failure.exit, ErrorCode::Generic);
        assert_eq!(failure.code, "sampled_sentiment_job_completion_invariant");
    }
}

#[test]
fn runtime_revalidates_storage_and_retention_before_runtime_or_private_io() {
    for case in 0..3 {
        let (mut selected, args) = command("resume", 0, &[]);
        let admitted = selected.args.validate(&args).unwrap();
        let expected = match case {
            0 => { selected.args.storage.max_stream_mib = 0; "explicit_sampled_sentiment_job_storage" }
            1 => { selected.args.storage.store_results = false; "explicit_sampled_sentiment_job_storage" }
            _ => {
                selected.args.storage.input_retention.spool_input = true;
                selected.args.storage.input_retention.max_retained_input_mib = Some(8);
                "job_input_retention_options"
            }
        };
        let failure = execute(selected.mode, args, selected.args, admitted, NoIo, &mut NoIo).unwrap_err();
        assert_eq!(failure.exit, ErrorCode::Usage);
        assert_eq!(failure.code, expected);
    }
}

#[test]
fn expired_config_read_refuses_before_opening_the_private_path() {
    let failure = read_config(std::path::Path::new("private-limits-that-must-not-be-opened.json"),
        &|| Err(cancelled(DecodeCancellationKind::Deadline))).unwrap_err();
    assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Deadline));
}

#[test]
fn original_and_retained_input_quotes_preserve_the_exact_invocation_controller() {
    let (original, args) = command("resume", 0, &[]);
    let common = original.args.validate(&args).unwrap();
    let fixed = lifetime(&args, &original, common);
    let original_limits = original.args.lifetime_limits(&args, common, fixed).unwrap();
    for (operation, flags, input_memory) in [
        ("resume", &[][..], JobInputMemory::CompletePopulation),
        ("start", &["--spool-input", "--max-retained-input-mib", "8"][..], JobInputMemory::RetainedRecord),
        ("resume", &["--use-retained-input", "--max-retained-input-mib", "8"][..], JobInputMemory::RetainedRecord),
    ] {
        let (selected, args) = command(operation, 0, flags);
        let common = selected.args.validate(&args).unwrap();
        selected.args.storage.input_retention.validate(selected.mode, &args.input).unwrap();
        assert_eq!(selected.args.storage.input_retention.input_memory().unwrap(), input_memory);
        let limits = selected.args.lifetime_limits(&args, common, fixed).unwrap();
        let remaining = Duration::from_millis(17_321);
        let host = host_limits(&args, limits, remaining);
        assert_eq!(host.required_buffer_bytes_for_input(fixed, input_memory).unwrap(), limits.buffer_bytes);
        assert_eq!(host.native.context_tokens, args.context_tokens);
        assert_eq!(host.native.allocator_reserve_bytes, ALLOCATOR_BYTES);
        assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
        assert_eq!(host.native.run.max_elapsed, remaining);
        assert_eq!(host.native.run.cleanup_reserve_bytes, CLEANUP_BYTES);
        assert_eq!(host.transport.max_lines, selected.args.storage.max_input_lines);
        assert_eq!(host.transport.max_stream_bytes, selected.args.storage.max_stream_mib * MIB);
        assert_eq!(host.preparation_reserve_bytes, common.preparation_bytes);
        assert_eq!(JobWork::from_int4(limits.native.max_model_work,
            limits.native.masks.max_visits_per_run), fixed.max_work);
        let removed = if input_memory == JobInputMemory::RetainedRecord { 2 * fixed.max_snapshot_bytes } else { 0 };
        assert_eq!(original_limits.buffer_bytes - limits.buffer_bytes, removed);
        assert_eq!(original_limits.common.minimum_memory_bytes - limits.common.minimum_memory_bytes, removed);
        let mut invalid = host; invalid.journal_reserve_bytes = 0;
        let encoding = args.input_encoding.policy(args.max_input_bytes, common.max_prompt_tokens).unwrap();
        let preparation = sampled_sentiment_job_preparation_bytes(limits.task, limits.planning, &encoding).unwrap();
        let failure = int4_sampled_sentiment_job_requirements(args.group_width, limits.task,
            limits.native, preparation, fixed, invalid, input_memory, common.prefill, None).unwrap_err();
        assert_eq!(job_failure(failure).exit, ErrorCode::AdmissionOrResourceLimit);
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
fn completion_metadata_binds_candidate_and_all_six_spent_axes_without_private_samples() {
    let (selected, args) = command("resume", 0, &[]);
    let common = selected.args.validate(&args).unwrap();
    let fixed = lifetime(&args, &selected, common);
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
        durable_unit: "complete_independent_sample_bundle",
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
    assert_eq!(value["durable_unit"], "complete_independent_sample_bundle");
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
    assert!(!std::str::from_utf8(&bytes).unwrap().contains("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"));
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
    let (prefill_command, prefill_args) = command("resume", 0, &["--prefill-rows", "3"]);
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

fn cancellations() -> [(DecodeCancellationKind, ErrorCode, &'static str); 11] {
    use DecodeCancellationKind::*;
    [
        (User, ErrorCode::Cancelled, "sampled_sentiment_cancelled"),
        (Timeout, ErrorCode::BudgetOrTimeout, "sampled_sentiment_budget"),
        (Deadline, ErrorCode::BudgetOrTimeout, "sampled_sentiment_budget"),
        (PollQuota, ErrorCode::BudgetOrTimeout, "sampled_sentiment_budget"),
        (CostBudget, ErrorCode::BudgetOrTimeout, "sampled_sentiment_budget"),
        (FailFast, ErrorCode::Generic, "sampled_sentiment_supervised_failure"),
        (RaceLost, ErrorCode::Generic, "sampled_sentiment_root_race"),
        (ParentCancelled, ErrorCode::Cancelled, "sampled_sentiment_cancelled"),
        (ResourceUnavailable, ErrorCode::AdmissionOrResourceLimit, "sampled_sentiment_resources"),
        (Shutdown, ErrorCode::Cancelled, "sampled_sentiment_cancelled"),
        (LinkedExit, ErrorCode::Generic, "sampled_sentiment_supervised_failure"),
    ]
}

#[test]
fn original_cancellation_survives_host_native_task_and_durable_boundaries() {
    for (cause, exit, code) in cancellations() {
        let stop = HostedError::Stopped { stop: RunStop { kind: cause,
            runtime_reason: None, runtime_checkpoint_error: None }, task_error: None };
        for error in [stop,
            HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Model(Int4ModelError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause))),
            HostedError::Int4(Int4HostError::SampledSentiment(SampledSentimentError::Cancelled(cause))),
        ] {
            let actual = job_failure(HostedJobError::Host(error));
            assert_eq!(actual.exit, exit); assert_eq!(actual.code, code);
            assert_eq!(actual.cancellation, Some(cause));
        }
        for error in [SampledSentimentError::Cancelled(cause),
            SampledSentimentError::Native(Int4JsonError::Native(StrictInt4Error::Cancelled(cause))),
            SampledSentimentError::Native(Int4JsonError::Decode(JsonDecodeError::Cancelled(cause))),
        ] {
            let actual = task_failure(error);
            assert_eq!(actual.exit, exit); assert_eq!(actual.code, code);
            assert_eq!(actual.cancellation, Some(cause));
        }
        for error in [JobRunError::Storage(JobError::Cancelled(cause)),
            JobRunError::Processor(BatchFault::cancelled(cause))] {
            let actual = job_failure(HostedJobError::Job(error));
            assert_eq!(actual.exit, exit); assert_eq!(actual.code, code);
            assert_eq!(actual.cancellation, Some(cause));
        }
    }
}

#[test]
fn supervised_failures_keep_typed_errors_and_ordinary_stops_keep_the_original_cause() {
    use DecodeCancellationKind::*;
    for cause in [FailFast, LinkedExit] {
        let actual = job_failure(HostedJobError::Host(HostedError::Stopped {
            stop: RunStop { kind: cause, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::Int4(Int4HostError::SampledSentiment(
                SampledSentimentError::NoResult)))),
        }));
        assert_eq!(actual.exit, ErrorCode::StructuredTaskNoResult);
        assert_eq!(actual.code, "sampled_sentiment_no_result");
        assert_eq!(actual.cancellation, Some(cause));
    }
    let actual = job_failure(HostedJobError::Host(HostedError::Stopped {
        stop: RunStop { kind: User, runtime_reason: None, runtime_checkpoint_error: None },
        task_error: Some(Box::new(HostedError::Limits("fixture memory"))),
    }));
    assert_eq!(actual.exit, ErrorCode::Cancelled);
    assert_eq!(actual.code, "sampled_sentiment_cancelled");
    assert_eq!(actual.cancellation, Some(User));
    let panicked = job_failure(HostedJobError::Host(HostedError::Panicked));
    assert_eq!(panicked.exit, ErrorCode::Generic);
}

#[test]
fn bounded_metadata_and_deadline_are_checked_before_staging_and_the_first_byte() {
    let cancellation = CancellationToken::default();
    let available = || Ok(Duration::from_secs(1));
    let expired = publish_report(&NoSerialize, &mut NoIo,
        &|| Err(cancelled(DecodeCancellationKind::Deadline)), &cancellation).unwrap_err();
    assert_eq!(expired.exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(expired.cancellation, Some(DecodeCancellationKind::Deadline));
    let oversized = "x".repeat(job_cli::REPORT_BYTES);
    assert_eq!(publish_report(&oversized, &mut NoIo, &available, &cancellation).unwrap_err().code,
        Failure::from(CandidateError::Output).code);
    let checks = std::cell::Cell::new(0);
    let expired_before_write = || {
        checks.set(checks.get() + 1);
        if checks.get() == 1 { available() }
        else { Err(cancelled(DecodeCancellationKind::Deadline)) }
    };
    let failure = publish_report(&true, &mut NoIo, &expired_before_write, &cancellation).unwrap_err();
    assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(checks.get(), 2);
}

#[test]
fn expired_delivery_and_failed_flush_never_report_success_after_metadata_is_written() {
    let cancellation = CancellationToken::default();
    let checks = std::cell::Cell::new(0);
    let expired_after_flush = || {
        checks.set(checks.get() + 1);
        if checks.get() < 3 { Ok(Duration::from_secs(1)) }
        else { Err(cancelled(DecodeCancellationKind::Deadline)) }
    };
    let mut delivered = Vec::new();
    let failure = publish_report(&true, &mut delivered, &expired_after_flush, &cancellation).unwrap_err();
    assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Deadline));
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
fn every_original_cancellation_prevents_serialization_and_short_circuits_the_deadline() {
    for (cause, exit, code) in cancellations() {
        let cancellation = CancellationToken::default();
        cancellation.cancel(cause);
        cancellation.cancel(DecodeCancellationKind::User);
        let failure = publish_report(&NoSerialize, &mut NoIo,
            &|| panic!("an existing cancellation must win without reading the later deadline"),
            &cancellation).unwrap_err();
        assert_eq!(cancellation.cause(), Some(cause));
        assert_eq!(failure.exit, exit); assert_eq!(failure.code, code);
        assert_eq!(failure.cancellation, Some(cause));
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
    assert_eq!(failure.code, "sampled_sentiment_cancelled");
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::ParentCancelled));
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
            if checks.get() < 3 { Ok(Duration::from_secs(1)) }
            else { Err(cancelled(DecodeCancellationKind::Deadline)) }
        };
        let failure = publish_report(&true, &mut output, &remaining, &cancellation).unwrap_err();
        assert_eq!(output.bytes, b"true\n");
        assert_eq!(cancellation.cause(), Some(DecodeCancellationKind::ResourceUnavailable));
        assert_eq!(failure.exit, ErrorCode::AdmissionOrResourceLimit);
        assert_eq!(failure.code, "sampled_sentiment_resources");
        assert_eq!(failure.cancellation, Some(DecodeCancellationKind::ResourceUnavailable));
        assert_eq!(checks.get(), 2, "the recorded stop retains authority over the later deadline");
    }
}

#[test]
fn first_observed_deadline_failure_is_not_replaced_by_a_token_set_inside_that_check() {
    let cancellation = CancellationToken::default();
    let original = Failure { exit: ErrorCode::BudgetOrTimeout, code: "fixture_original_deadline",
        cancellation: Some(DecodeCancellationKind::Deadline) };
    let checks = std::cell::Cell::new(0);
    let remaining = || {
        checks.set(checks.get() + 1);
        cancellation.cancel(DecodeCancellationKind::User);
        Err(original)
    };
    let failure = publish_report(&NoSerialize, &mut NoIo, &remaining, &cancellation).unwrap_err();
    assert_eq!(failure.exit, original.exit);
    assert_eq!(failure.code, original.code);
    assert_eq!(failure.cancellation, original.cancellation);
    assert_eq!(cancellation.cause(), Some(DecodeCancellationKind::User));
    assert_eq!(checks.get(), 1);
}

#[test]
fn failed_writer_keeps_the_delivery_error_without_an_unobserved_cancellation_override() {
    struct FailedDelivery { cancellation: CancellationToken, at_flush: bool }
    impl Write for FailedDelivery {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.at_flush { return Ok(bytes.len()); }
            self.cancellation.cancel(DecodeCancellationKind::User);
            Err(io::Error::other("fixture write"))
        }
        fn flush(&mut self) -> io::Result<()> {
            self.cancellation.cancel(DecodeCancellationKind::User);
            Err(io::Error::other("fixture flush"))
        }
    }
    for at_flush in [false, true] {
        let cancellation = CancellationToken::default();
        let mut output = FailedDelivery { cancellation: cancellation.clone(), at_flush };
        let failure = publish_report(&true, &mut output, &|| Ok(Duration::from_secs(1)),
            &cancellation).unwrap_err();
        assert_eq!(failure.code, Failure::from(CandidateError::Output).code);
        assert_eq!(failure.cancellation, None);
        assert_eq!(cancellation.cause(), Some(DecodeCancellationKind::User));
    }
}
