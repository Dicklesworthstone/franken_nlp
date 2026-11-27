//! Cross-layer lifetime quotes, original cancellation and metadata delivery.
use super::*;
use std::io;
use crate::{
    candidate_cli::sampled_sentiment::map::job::{self as command, SampledSentimentDocumentJobCommand},
    hosted::RunStop,
    jobs::{JobId, JobProgress, JobWork},
};

const SEED: &str = "4545454545454545454545454545454545454545454545454545454545454545";
fn parsed(operation: &str, flags: &[&str]) -> SampledSentimentDocumentJobCommand {
    let mut words = vec!["sentiment-sampled-document-job", operation, "--model", "never-opened.fnlpq",
        "--memory-mib", "131072", "--seed", SEED, "--job-dir", "never-opened-job",
        "--key-file", "never-opened-key", "--limits", "never-opened-limits",
        "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"];
    words.extend_from_slice(flags);
    SampledSentimentDocumentJobCommand::from_matches(
        &command::definition().try_get_matches_from(words).unwrap()).unwrap()
}
fn lifetime(selected: &SampledSentimentDocumentJobCommand, admitted: &MapLimits) -> JobLimits {
    let work = selected.args.document.sampled.work_ceiling();
    JobLimits { max_items: 3, max_id_bytes: 128,
        max_input_bytes_per_item: admitted.common.max_input_bytes, max_snapshot_bytes: 4 * MIB,
        max_result_bytes: admitted.mapping.max_result_bytes as usize,
        max_spool_bytes: 256 * MIB, max_materialized_bytes: 256 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 6,
        max_work: JobWork { model: work.checked_add(work).unwrap(),
            mask_node_visits: 2 * selected.args.document.sampled.max_mask_node_visits } }
}
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused runtime read private input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused runtime published metadata"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused runtime flushed metadata"); }
}

#[test]
fn cli_and_host_agree_for_original_or_retained_input_and_all_supported_schedules() {
    for schedule in [
        &[][..], &["--compute-threads", "2"][..], &["--compute-threads", "64"][..],
        &["--prefill-rows", "1"][..], &["--prefill-rows", "64"][..],
        &["--compute-threads", "4", "--prefill-rows", "7"][..],
        &["--input-encoding", "safe-bpe-v1", "--compute-threads", "4", "--prefill-rows", "7"][..],
    ] { for retained in [false, true] {
        let mut flags = schedule.to_vec();
        if retained { flags.extend(["--use-retained-input", "--max-retained-input-mib", "8"]); }
        let selected = parsed("resume", &flags);
        let admitted = selected.args.validate().unwrap();
        let fixed = lifetime(&selected, &admitted);
        let limits = selected.args.lifetime_limits(admitted, fixed).unwrap();
        let remaining = Duration::from_millis(17_321);
        check_host_quote(&selected.args, &limits, fixed, remaining).unwrap();
        let host = host_limits(&limits, remaining);
        assert_eq!(host.native.run.max_elapsed, remaining);
        assert_eq!(host.native.run.max_checkpoints, limits.document.common.max_checkpoints);
        assert_eq!(host.native.run.cleanup_reserve_bytes, 65_536);
        assert_eq!(host.required_buffer_bytes_for_input(fixed,
            selected.args.storage.input_retention.input_memory().unwrap()).unwrap(), limits.buffer_bytes);
        assert_eq!(limits.native.max_item_work, limits.document.mapping.max_model_work);
        for field in 0..9 {
            let mut wrong = selected.args.lifetime_limits(selected.args.validate().unwrap(), fixed).unwrap();
            match field {
                0 => wrong.document.memory.input_bytes -= 1,
                1 => wrong.document.memory.workspace_bytes -= 1,
                2 => wrong.document.memory.output_reservation_bytes -= 1,
                3 => wrong.document.memory.max_output_bytes -= 1,
                4 => wrong.document.memory.max_output_tokens -= 1,
                5 => wrong.document.memory.native.kv_bytes -= 1,
                6 => wrong.execution_bytes -= 1,
                7 => wrong.buffer_bytes -= 1,
                _ => wrong.document.minimum_memory_bytes = wrong.document.shared.memory_bytes + 1,
            }
            assert_eq!(check_host_quote(&selected.args, &wrong, fixed, remaining).unwrap_err().code,
                "int8_sampled_document_job_host_quote", "field {field}");
        }
    } }
}

#[test]
fn runtime_revalidates_actual_storage_retention_and_map_before_any_private_io() {
    for fault in 0..3 {
        let mut selected = parsed("resume", &[]);
        let admitted = selected.args.validate().unwrap();
        match fault {
            0 => selected.args.storage.store_results = false,
            1 => selected.args.document.mapping.max_chunks = 0,
            _ => {
                selected.args.storage.input_retention.spool_input = true;
                selected.args.storage.input_retention.max_retained_input_mib = Some(8);
            }
        }
        let failure = execute(selected.mode, selected.args, admitted, NoIo, &mut NoIo).unwrap_err();
        assert_eq!(failure.exit, ErrorCode::Usage);
    }
    let failure = read_config(std::path::Path::new("never-opened.json"),
        &|| Err(CandidateError::Timeout)).unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Deadline));
}

#[test]
fn all_cancellation_causes_survive_storage_processor_map_host_and_report_boundaries() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, LinkedExit, Shutdown] {
        let expected = cancelled(cause);
        for error in [
            HostedJobError::Job(JobRunError::Storage(JobError::Cancelled(cause))),
            HostedJobError::Job(JobRunError::Processor(BatchFault::cancelled(cause))),
            HostedJobError::Host(HostedError::BatchSetup(BatchFault::cancelled(cause))),
            HostedJobError::Host(HostedError::SampledSentimentMap(Int8SampledSentimentMapError::Cancelled(cause))),
            HostedJobError::Host(HostedError::Stopped { stop: RunStop { kind: cause,
                runtime_reason: None, runtime_checkpoint_error: None }, task_error: None }),
        ] {
            let actual = job_failure(error);
            assert_eq!(actual.exit, expected.exit);
            assert_eq!(actual.cancellation, Some(cause));
        }
        let token = CancellationToken::default(); token.cancel(cause);
        let actual = publish_report(&NoSerialize, &mut NoIo,
            &|| panic!("original cancellation must precede a later deadline"), &token).unwrap_err();
        assert_eq!(actual.exit, expected.exit);
        assert_eq!(actual.cancellation, Some(cause));
    }
    for cause in [FailFast, LinkedExit] {
        let failure = job_failure(HostedJobError::Host(HostedError::Stopped {
            stop: RunStop { kind: cause, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::BatchSetup(BatchCode::WorkLimit.into()))),
        }));
        assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
        assert_eq!(failure.cancellation, Some(cause));
    }
}

#[test]
fn invalid_input_work_exhaustion_and_missing_completion_keep_distinct_failure_classes() {
    assert_eq!(batch_failure(BatchCode::InvalidJson.into()).exit, ErrorCode::InputDecodeOrParse);
    assert_eq!(batch_failure(BatchCode::WorkLimit.into()).exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(batch_failure(BatchCode::Allocation.into()).exit, ErrorCode::AdmissionOrResourceLimit);
    assert_eq!(batch_failure(BatchCode::Execution.into()).exit, ErrorCode::StructuredTaskNoResult);
    assert_eq!(batch_failure(BatchCode::OutputIo.into()).exit, ErrorCode::Generic);
    assert_eq!(batch_failure(BatchCode::Cancelled.into()).code, "int8_sampled_document_job_missing_cancellation");
    assert_eq!(job_failure(HostedJobError::Job(JobRunError::InvalidEnvelope)).exit, ErrorCode::InputDecodeOrParse);
    for error in [JobRunError::WorkContract, JobRunError::Stopped] {
        assert_eq!(job_failure(HostedJobError::Job(error)).exit, ErrorCode::Generic);
    }
}
struct NoSerialize;
impl Serialize for NoSerialize {
    fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        panic!("stopped metadata must not serialize");
    }
}

#[test]
fn bounded_metadata_delivery_polls_before_staging_and_after_successful_or_failed_flush() {
    let cancellation = CancellationToken::default();
    let expired = publish_report(&NoSerialize, &mut NoIo,
        &|| Err(cancelled(DecodeCancellationKind::Deadline)), &cancellation).unwrap_err();
    assert_eq!(expired.cancellation, Some(DecodeCancellationKind::Deadline));
    let oversized = "x".repeat(job_cli::REPORT_BYTES);
    assert_eq!(publish_report(&oversized, &mut NoIo, &|| Ok(Duration::from_secs(1)),
        &cancellation).unwrap_err().code, setup_failure(CandidateError::Output).code);
    let checks = Cell::new(0);
    let mut bytes = Vec::new();
    let failure = publish_report(&true, &mut bytes, &|| {
        checks.set(checks.get() + 1);
        if checks.get() < 3 { Ok(Duration::from_secs(1)) }
        else { Err(cancelled(DecodeCancellationKind::Deadline)) }
    }, &cancellation).unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(bytes, b"true\n");

    struct CancelOnFlush { cancellation: CancellationToken, bytes: Vec<u8> }
    impl Write for CancelOnFlush {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(bytes); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.cancellation.cancel(DecodeCancellationKind::ParentCancelled);
            Err(io::Error::other("private flush failure"))
        }
    }
    let mut writer = CancelOnFlush { cancellation: cancellation.clone(), bytes: Vec::new() };
    let failure = publish_report(&true, &mut writer, &|| Ok(Duration::from_secs(1)), &cancellation).unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::ParentCancelled));
    assert_eq!(writer.bytes, b"true\n");
}

fn assert_public_metadata(value: &serde_json::Value) {
    match value {
        serde_json::Value::Object(fields) => for (key, value) in fields {
            assert!(!["text", "document", "documents", "chunks", "results", "score", "scores", "samples",
                "sampling", "seed", "effective_seed", "recipe", "task_args", "axes", "justification",
                "justifications", "key", "job_dir"].contains(&key.as_str()), "private field: {key}");
            assert_public_metadata(value);
        },
        serde_json::Value::Array(values) => for value in values { assert_public_metadata(value); },
        _ => {},
    }
}

#[test]
fn only_verified_complete_progress_becomes_metadata_without_source_seed_or_samples() {
    let selected = parsed("resume", &["--compute-threads", "4", "--prefill-rows", "7"]);
    let admitted = selected.args.validate().unwrap();
    let fixed = lifetime(&selected, &admitted);
    let id = JobId([0x12; 16]);
    let progress = || JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: fixed.max_work, spool_bytes: 1024, materialized: true };
    for field in 0..5 {
        let mut wrong = progress();
        match field {
            0 => wrong.committed -= 1,
            1 => wrong.attempts = wrong.committed - 1,
            2 => wrong.reserved_work.model.forward_positions += 1,
            3 => wrong.job_id = JobId([0x13; 16]),
            _ => wrong.materialized = false,
        }
        assert!(progress_report(wrong, id, fixed, true, selected.mode, "sentiment-sampled-map").is_err());
    }
    let report = progress_report(progress(), id, fixed, true, selected.mode, "sentiment-sampled-map").unwrap();
    let logical = "cd".repeat(32); let source = "ef".repeat(32);
    let response = Response {
        protocol: "fnlp-candidate-int8-sentiment-sampled-document-job-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative",
        task: "sentiment-sampled-map", durable_unit: "complete_sampled_document",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-public-revision",
        source_root_sha256: &source, logical_model_sha256: &logical,
        candidate_format: "fixture-portable-int8", quant_recipe: "fixture-int8",
        input_encoding: None, compute_threads: Some(4),
        prefill_schedule: admitted.prefill.map(PrefillSchedule::new).transpose().unwrap(),
        output: &report,
    };
    let mut bytes = Vec::new();
    publish_report(&response, &mut bytes, &|| Ok(Duration::from_secs(1)), &CancellationToken::default()).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["durable_unit"], "complete_sampled_document");
    assert_eq!(value["output"]["status"], "complete");
    assert_eq!(value["output"]["committed"], 2);
    assert_eq!(value["output"]["attempts"], 3);
    assert_eq!(value["output"]["reserved_work"], serde_json::to_value(fixed.max_work).unwrap());
    assert_eq!(value["compute_threads"], 4);
    assert_eq!(value["prefill_schedule"]["max_batch_rows"], 7);
    assert_eq!(value["prefill_schedule"]["execution"], LAYER_MAJOR_PREFILL_VERSION);
    assert_eq!(value["evidence"], "non_authoritative");
    assert_public_metadata(&value);
    assert!(bytes.ends_with(b"\n"));
    let text = String::from_utf8(bytes).unwrap();
    for secret in [SEED, "never-opened-job", "never-opened-key", "never-opened-limits"] {
        assert!(!text.contains(secret));
    }
}
