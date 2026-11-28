//! Independent host quotes, typed terminal causes and metadata-only delivery.
use super::*;
use std::io;
use crate::{
    batch::{BatchCode, BatchFault},
    candidate_cli::jobs::question_document::{self as command, QuestionDocumentJobCommand},
    hosted::RunStop,
    jobs::{JobId, JobProgress, JobWork},
    native_engine::decode::DecodeCancellationKind,
    tasks::source_planning::quantized::{Int8SourceError, long::Int8SourceMapError},
};

fn parsed(operation: &str, flags: &[&str]) -> QuestionDocumentJobCommand {
    let mut words = vec!["answer-job", operation, "--model", "never-opened.fnlpq", "--memory-mib", "131072",
        "--job-dir", "never-opened-job", "--key-file", "never-opened-key", "--limits", "never-opened-limits",
        "--job-id", "0123456789abcdef0123456789abcdef", "--store-results"];
    words.extend_from_slice(flags);
    QuestionDocumentJobCommand::from_matches(&command::definition().try_get_matches_from(words).unwrap()).unwrap()
}
fn lifetime(document: &QuestionDocumentLimits) -> JobLimits {
    let work = document.mapping.max_model_work;
    JobLimits { max_items: 3, max_id_bytes: 128,
        max_input_bytes_per_item: document.common.max_input_bytes + document.max_question_bytes + 4096,
        max_snapshot_bytes: 4 * MIB, max_result_bytes: document.mapping.reduction.max_result_bytes,
        max_spool_bytes: 256 * MIB, max_materialized_bytes: 256 * MIB,
        max_journal_bytes: 64 * MIB, max_attempts: 6,
        max_work: JobWork { model: work.checked_add(work).unwrap(),
            mask_node_visits: document.mapping.max_mask_visits * 2 } }
}
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused question runtime read private input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused question runtime published metadata"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused question runtime flushed metadata"); }
}

#[test]
fn original_and_retained_jobs_match_full_host_quotes_for_composed_schedules() {
    for schedule in [&[][..], &["--compute-threads", "2"][..], &["--compute-threads", "64"][..],
        &["--prefill-rows", "1"][..], &["--prefill-rows", "64"][..],
        &["--compute-threads", "4", "--prefill-rows", "7"][..],
        &["--input-encoding", "safe-bpe-v1", "--compute-threads", "4", "--prefill-rows", "7"][..]] {
        for retained in [false, true] {
            let mut flags = schedule.to_vec();
            if retained { flags.extend(["--use-retained-input", "--max-retained-input-mib", "8"]); }
            let selected = parsed("resume", &flags);
            let document = selected.args.validate().unwrap(); let fixed = lifetime(&document);
            let limits = selected.args.lifetime_limits(document, fixed, question::parse_options(None).unwrap()).unwrap();
            let remaining = Duration::from_millis(17_321);
            check_host_quote(&selected.args, &limits, fixed, remaining).unwrap();
            let host = host_limits(&limits, remaining);
            assert_eq!(host.native.run.max_elapsed, remaining);
            assert_eq!(host.native.run.max_checkpoints, limits.document.common.max_checkpoints);
            assert_eq!(host.required_buffer_bytes_for_input(fixed,
                selected.args.storage.input_retention.input_memory().unwrap()).unwrap(), limits.memory.population_buffer_bytes);
            assert_eq!(limits.config.max_model_work, fixed.max_work.model);
            assert_eq!(limits.config.max_mask_visits, fixed.max_work.mask_node_visits);
            for field in 0..11 {
                let mut wrong = selected.args.lifetime_limits(selected.args.validate().unwrap(), fixed,
                    question::parse_options(None).unwrap()).unwrap();
                match field {
                    0 => wrong.memory.document.input_bytes -= 1,
                    1 => wrong.memory.document.workspace_bytes -= 1,
                    2 => wrong.memory.document.document_workspace_bytes -= 1,
                    3 => wrong.memory.document.output_reservation_bytes -= 1,
                    4 => wrong.memory.document.max_output_bytes -= 1,
                    5 => wrong.memory.document.max_output_tokens = 1,
                    6 => wrong.memory.document.native.kv_bytes -= 1,
                    7 => wrong.memory.population_buffer_bytes -= 1,
                    8 => wrong.execution_bytes -= 1,
                    9 => wrong.document.shared.compute.workspace_bytes += 1,
                    _ => wrong.minimum_memory_bytes = wrong.document.shared.memory_bytes + 1,
                }
                assert_eq!(check_host_quote(&selected.args, &wrong, fixed, remaining).unwrap_err().code,
                    "int8_question_document_job_host_quote", "field {field}");
            }
        }
    }
}

#[test]
fn runtime_revalidates_actual_options_and_retention_before_any_private_configuration_read() {
    for fault in 0..3 {
        let mut selected = parsed("resume", &[]); let admitted = selected.args.validate().unwrap();
        match fault {
            0 => selected.args.storage.store_results = false,
            1 => selected.args.document.max_question_bytes = 0,
            _ => { selected.args.storage.input_retention.spool_input = true;
                selected.args.storage.input_retention.max_retained_input_mib = Some(8); }
        }
        assert_eq!(execute(selected.mode, selected.args, admitted, NoIo, &mut NoIo).unwrap_err().exit, ErrorCode::Usage);
    }
    let failure = read_config(std::path::Path::new("never-opened.json"), job_cli::LIMIT_BYTES,
        &|| Err(CandidateError::Timeout)).unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Deadline));
}

#[test]
fn all_original_cancellation_causes_survive_storage_native_map_and_publication() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, LinkedExit, Shutdown] {
        let expected = cancelled(cause);
        for error in [
            HostedJobError::Job(JobRunError::Storage(JobError::Cancelled(cause))),
            HostedJobError::Job(JobRunError::Processor(BatchFault::cancelled(cause))),
            HostedJobError::Host(HostedError::BatchSetup(BatchFault::cancelled(cause))),
            HostedJobError::Host(HostedError::SourceMap(Int8SourceMapError::Source(Int8SourceError::Cancelled(cause)))),
            HostedJobError::Host(HostedError::Stopped { stop: RunStop { kind: cause,
                runtime_reason: None, runtime_checkpoint_error: None }, task_error: None }),
        ] {
            let actual = job_failure(error);
            assert_eq!(actual.exit, expected.exit); assert_eq!(actual.cancellation, Some(cause));
        }
        let token = CancellationToken::default(); token.cancel(cause);
        let actual = publish_report(&NoSerialize, &mut NoIo,
            &|| panic!("original stop must precede a later deadline"), &token).unwrap_err();
        assert_eq!(actual.exit, expected.exit); assert_eq!(actual.cancellation, Some(cause));
    }
    for cause in [FailFast, LinkedExit] {
        let failure = job_failure(HostedJobError::Host(HostedError::Stopped {
            stop: RunStop { kind: cause, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::BatchSetup(BatchCode::WorkLimit.into()))),
        }));
        assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout); assert_eq!(failure.cancellation, Some(cause));
    }
}

struct NoSerialize;
impl Serialize for NoSerialize {
    fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        panic!("stopped metadata must not serialize");
    }
}
#[test]
fn bounded_metadata_delivery_polls_after_success_or_failed_flush_without_replacing_a_stop() {
    let cancellation = CancellationToken::default();
    assert_eq!(publish_report(&NoSerialize, &mut NoIo,
        &|| Err(cancelled(DecodeCancellationKind::Deadline)), &cancellation).unwrap_err().cancellation,
        Some(DecodeCancellationKind::Deadline));
    let oversized = "x".repeat(job_cli::REPORT_BYTES);
    assert_eq!(publish_report(&oversized, &mut NoIo, &|| Ok(Duration::from_secs(1)),
        &cancellation).unwrap_err().code, setup_failure(CandidateError::Output).code);
    let checks = Cell::new(0); let mut bytes = Vec::new();
    let failure = publish_report(&true, &mut bytes, &|| {
        checks.set(checks.get() + 1);
        if checks.get() < 3 { Ok(Duration::from_secs(1)) }
        else { Err(cancelled(DecodeCancellationKind::Deadline)) }
    }, &cancellation).unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::Deadline));
    assert_eq!(bytes, b"true\n");
    struct CancelOnFlush { cancellation: CancellationToken, bytes: Vec<u8> }
    impl Write for CancelOnFlush {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.bytes.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> {
            self.cancellation.cancel(DecodeCancellationKind::ParentCancelled);
            Err(io::Error::other("private output failure"))
        }
    }
    let mut writer = CancelOnFlush { cancellation: cancellation.clone(), bytes: Vec::new() };
    let failure = publish_report(&true, &mut writer, &|| Ok(Duration::from_secs(1)), &cancellation).unwrap_err();
    assert_eq!(failure.cancellation, Some(DecodeCancellationKind::ParentCancelled));
    assert_eq!(writer.bytes, b"true\n");
}

fn public_metadata(value: &serde_json::Value) {
    match value {
        serde_json::Value::Object(fields) => for (key, value) in fields {
            assert!(!["text", "source", "document", "question", "questions", "answer", "answers",
                "passages", "citations", "quote", "quotes", "results", "chunks", "task_args", "recipe",
                "seed", "key", "job_dir"].contains(&key.as_str()), "private field: {key}");
            public_metadata(value);
        },
        serde_json::Value::Array(values) => for value in values { public_metadata(value); },
        _ => {},
    }
}
#[test]
fn only_complete_verified_progress_becomes_metadata_without_source_question_or_answers() {
    let selected = parsed("resume", &["--input-encoding", "safe-bpe-v1", "--prefill-rows", "7", "--compute-threads", "4"]);
    let document = selected.args.validate().unwrap(); let fixed = lifetime(&document);
    let id = JobId([0x12; 16]);
    let progress = || JobProgress { job_id: id, items: 2, committed: 2, attempts: 3,
        reserved_work: fixed.max_work, spool_bytes: 1024, materialized: true };
    for field in 0..5 {
        let mut wrong = progress();
        match field { 0 => wrong.committed -= 1, 1 => wrong.attempts = wrong.committed - 1,
            2 => wrong.reserved_work.model.forward_positions += 1,
            3 => wrong.job_id = JobId([0x13; 16]), _ => wrong.materialized = false }
        assert!(progress_report(wrong, id, fixed, true, selected.mode, "answer").is_err());
    }
    let report = progress_report(progress(), id, fixed, true, selected.mode, "answer").unwrap();
    let response = Response { protocol: "fnlp-candidate-int8-question-document-job-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative", task: "answer",
        execution: INT8_QUESTION_EXECUTION, durable_unit: "complete_original_source_question",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-public-revision", source_root_sha256: "fixture-source",
        logical_model_sha256: "fixture-logical", candidate_format: "fixture-int8", quant_recipe: "fixture-recipe",
        input_encoding: input_encoding::reported_policy(&document.encoding),
        document_encoding_work_limits: document.encoding_work,
        document_encoding_work_scope: Some("one-original-source-question-prepare-and-execute-reconstruction-v1"),
        compute_threads: Some(4), prefill_schedule: document.prefill.map(PrefillSchedule::new).transpose().unwrap(),
        output: &report };
    let mut bytes = Vec::new();
    publish_report(&response, &mut bytes, &|| Ok(Duration::from_secs(1)), &CancellationToken::default()).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["durable_unit"], "complete_original_source_question");
    assert_eq!(value["output"]["status"], "complete");
    assert_eq!(value["output"]["reserved_work"], serde_json::to_value(fixed.max_work).unwrap());
    assert_eq!(value["document_encoding_work_scope"], "one-original-source-question-prepare-and-execute-reconstruction-v1");
    assert!(value.get("invocation_encoding_work_limits").is_none());
    public_metadata(&value);
}
