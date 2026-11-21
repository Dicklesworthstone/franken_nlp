//! Cross-layer whole-document quotes, original cancellation and public metadata.
use super::*;
use super::super::tests::{command, lifetime, NoIo};
use crate::{
    hosted::{Int4HostError, RunStop},
    jobs::{JobId, JobProgress},
    tasks::sentiment::sampled::long::SampledSentimentMapError,
};

#[test]
fn complete_host_quote_matches_cli_for_every_input_and_physical_schedule() {
    for snapshot in [false, true] { for flags in [
        &[][..], &["--reuse-prefill"][..], &["--compute-threads", "2"][..],
        &["--compute-threads", "64", "--reuse-prefill"][..],
        &["--prefill-rows", "1"][..], &["--prefill-rows", "64", "--reuse-prefill"][..],
        &["--input-encoding", "safe-bpe-v1", "--compute-threads", "4"][..],
        &["--input-encoding", "safe-bpe-v1", "--reuse-prefill", "--prefill-rows", "4"][..],
    ] { for retained in [false, true] {
        let mut flags = flags.to_vec();
        if retained { flags.extend(["--use-retained-input", "--max-retained-input-mib", "8"]); }
        let (selected, args) = command("resume", snapshot, &flags);
        let admitted = selected.args.validate(&args).unwrap();
        let fixed = lifetime(&args, &selected, admitted);
        let limits = selected.args.lifetime_limits(&args, admitted, fixed).unwrap();
        let remaining = Duration::from_millis(17_321);
        check_host_quote(&args, &selected.args, limits, fixed, remaining).unwrap();
        let host = host_limits(&args, limits, remaining);
        assert_eq!(host.native.run.max_elapsed, remaining);
        assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
        assert_eq!(host.native.run.cleanup_reserve_bytes, CLEANUP_BYTES);
        assert_eq!(host.preparation_reserve_bytes, limits.preparation_reserve_bytes);
        assert_eq!(host.required_buffer_bytes_for_input(fixed,
            selected.args.storage.input_retention.input_memory().unwrap()).unwrap(), limits.buffer_bytes);
        assert_eq!(limits.mapping.max_model_work, limits.native.max_item_work);
        assert_eq!(limits.memory.max_output_bytes, limits.mapping.max_result_bytes);
        for field in 0..8 {
            let mut wrong = limits;
            match field {
                0 => wrong.memory.input_bytes -= 1,
                1 => wrong.memory.workspace_bytes -= 1,
                2 => wrong.memory.output_reservation_bytes -= 1,
                3 => wrong.memory.max_output_bytes -= 1,
                4 => wrong.memory.max_output_tokens -= 1,
                5 => wrong.memory.native.kv_bytes -= 1,
                6 => wrong.execution_bytes -= 1,
                _ => wrong.buffer_bytes -= 1,
            }
            assert_eq!(check_host_quote(&args, &selected.args, wrong, fixed, remaining).unwrap_err().code,
                "sampled_document_job_host_quote");
        }
    } } }
}

#[test]
fn runtime_rechecks_storage_retention_and_map_before_any_private_io() {
    for fault in 0..4 {
        let (mut selected, mut args) = command("resume", false, &[]);
        let admitted = selected.args.validate(&args).unwrap();
        match fault {
            0 => selected.args.storage.store_results = false,
            1 => selected.args.document.mapping.max_chunks = 0,
            2 => {
                selected.args.storage.input_retention.spool_input = true;
                selected.args.storage.input_retention.max_retained_input_mib = Some(8);
            }
            _ => args.save_snapshot = Some(PathBuf::from("private-new-snapshot")),
        }
        let failure = execute(selected.mode, args, selected.args, admitted, NoIo, &mut NoIo).unwrap_err();
        assert_eq!(failure.exit, ErrorCode::Usage);
    }
    let failure = read_config(std::path::Path::new("private-limits-not-opened.json"),
        &|| Err(cancelled(DecodeCancellationKind::Deadline))).unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Deadline));
}

#[test]
fn cancellation_survives_map_host_storage_processor_and_report_boundaries() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, LinkedExit, Shutdown] {
        let expected = cancelled(cause);
        for error in [
            HostedJobError::Job(JobRunError::Storage(JobError::Cancelled(cause))),
            HostedJobError::Job(JobRunError::Processor(BatchFault::cancelled(cause))),
            HostedJobError::Host(HostedError::BatchSetup(BatchFault::cancelled(cause))),
            HostedJobError::Host(HostedError::Int4(Int4HostError::SampledSentimentMap(
                SampledSentimentMapError::Cancelled(cause)))),
            HostedJobError::Host(HostedError::Stopped { stop: RunStop { kind: cause,
                runtime_reason: None, runtime_checkpoint_error: None }, task_error: None }),
        ] {
            let actual = job_failure(error);
            assert_eq!(actual.exit, expected.exit);
            assert_eq!(actual.cancellation, Some(cause));
        }
        let cancellation = CancellationToken::default();
        cancellation.cancel(cause);
        let actual = publish_report(&NoSerialize, &mut NoIo,
            &|| panic!("original cancellation must precede the later deadline"), &cancellation).unwrap_err();
        assert_eq!(actual.exit, expected.exit);
        assert_eq!(actual.cancellation, Some(cause));
    }
    for cause in [FailFast, LinkedExit] {
        let failure = job_failure(HostedJobError::Host(HostedError::Stopped {
            stop: RunStop { kind: cause, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::BatchSetup(BatchCode::WorkLimit.into()))),
        }));
        assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
        assert_eq!(failure.code, "sampled_document_job_budget");
        assert_eq!(failure.cancellation, Some(cause));
    }
}

#[test]
fn parse_budget_native_and_delivery_errors_do_not_become_success_or_partial_documents() {
    assert_eq!(batch_failure(BatchCode::InvalidJson.into()).exit, ErrorCode::InputDecodeOrParse);
    assert_eq!(batch_failure(BatchCode::WorkLimit.into()).exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(batch_failure(BatchCode::Allocation.into()).exit, ErrorCode::AdmissionOrResourceLimit);
    assert_eq!(batch_failure(BatchCode::Execution.into()).exit, ErrorCode::StructuredTaskNoResult);
    assert_eq!(batch_failure(BatchCode::OutputIo.into()).exit, ErrorCode::Generic);
    assert_eq!(batch_failure(BatchCode::Cancelled.into()).code, "sampled_document_job_missing_cancellation");
    assert_eq!(job_failure(HostedJobError::Job(JobRunError::InvalidEnvelope)).exit,
        ErrorCode::InputDecodeOrParse);
    for error in [JobRunError::WorkContract, JobRunError::Stopped] {
        assert_eq!(job_failure(HostedJobError::Job(error)).exit, ErrorCode::Generic);
    }
}

struct NoSerialize;
impl Serialize for NoSerialize {
    fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        panic!("stopped metadata must not be serialized");
    }
}

#[test]
fn metadata_delivery_is_bounded_and_preserves_deadline_before_staging_write_and_after_flush() {
    let cancellation = CancellationToken::default();
    let expired = publish_report(&NoSerialize, &mut NoIo,
        &|| Err(cancelled(DecodeCancellationKind::Deadline)), &cancellation).unwrap_err();
    assert_eq!(expired.cancellation, Some(DecodeCancellationKind::Deadline));
    let oversized = "x".repeat(job_cli::REPORT_BYTES);
    assert_eq!(publish_report(&oversized, &mut NoIo, &|| Ok(Duration::from_secs(1)),
        &cancellation).unwrap_err().code, Failure::from(CandidateError::Output).code);
    let checks = Cell::new(0);
    let mut output = Vec::new();
    let failure = publish_report(&true, &mut output, &|| {
        checks.set(checks.get() + 1);
        if checks.get() < 3 { Ok(Duration::from_secs(1)) }
        else { Err(cancelled(DecodeCancellationKind::Deadline)) }
    }, &cancellation).unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(output, b"true\n");
    assert_eq!(checks.get(), 3);
}

fn assert_public_metadata(value: &serde_json::Value) {
    match value {
        serde_json::Value::Object(fields) => for (key, value) in fields {
            assert!(!["text", "document", "documents", "chunks", "results", "score", "scores",
                "samples", "sampling", "seed", "effective_seed", "recipe", "task_args", "axes",
                "justification", "justifications", "key", "job_dir"].contains(&key.as_str()),
                "private field in metadata: {key}");
            assert_public_metadata(value);
        },
        serde_json::Value::Array(values) => for value in values { assert_public_metadata(value); },
        _ => {},
    }
}

#[test]
fn success_reports_complete_document_progress_and_no_source_seed_or_retained_result() {
    let (selected, args) = command("resume", true, &["--reuse-prefill", "--prefill-rows", "3"]);
    let admitted = selected.args.validate(&args).unwrap();
    let fixed = lifetime(&args, &selected, admitted);
    let id = JobId([0x12; 16]);
    let progress = JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: fixed.max_work, spool_bytes: 1024, materialized: true };
    let report = progress_report(progress, id, fixed, true,
        RunMode::Resume { discard_uncommitted: true }, "sentiment-sampled-map").unwrap();
    let content = "cd".repeat(32); let source = "ef".repeat(32);
    let response = Response {
        protocol: "fnlp-candidate-int4-sentiment-sampled-document-job-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative",
        task: "sentiment-sampled-map", durable_unit: "complete_sampled_document",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-public-revision",
        source_root_sha256: &source, model_content_sha256: &content,
        candidate_format: "fixture-candidate-format", quant_recipe: "fixture-grouped-int4",
        group_width: args.group_width, integer_kernel: admitted.common.kernel.kind().id(),
        snapshot: Some(SnapshotRecord::new(admitted.common.snapshot_need)),
        input_encoding: None, compute_schedule: None,
        prefill_schedule: admitted.common.prefill.map(PrefillSchedule::new).transpose().unwrap(),
        prefill_reuse: Some(SAMPLED_SENTIMENT_MAP_REUSED_PREFILL_EXECUTION), output: &report,
    };
    let mut bytes = Vec::new();
    publish_report(&response, &mut bytes, &|| Ok(Duration::from_secs(1)),
        &CancellationToken::default()).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["task"], "sentiment-sampled-map");
    assert_eq!(value["durable_unit"], "complete_sampled_document");
    assert_eq!(value["prefill_reuse"], SAMPLED_SENTIMENT_MAP_REUSED_PREFILL_EXECUTION);
    assert_eq!(value["prefill_schedule"]["max_batch_rows"], 3);
    assert_eq!(value["output"]["status"], "complete");
    assert_eq!(value["output"]["committed"], 2);
    assert_eq!(value["output"]["attempts"], 3);
    assert_eq!(value["output"]["reserved_work"], serde_json::to_value(fixed.max_work).unwrap());
    assert_eq!(value["snapshot"]["operation"], "loaded");
    assert_eq!(value["evidence"], "non_authoritative");
    assert_public_metadata(&value);
    assert!(bytes.ends_with(b"\n"));
    assert!(!String::from_utf8(bytes).unwrap().contains("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"));
}
