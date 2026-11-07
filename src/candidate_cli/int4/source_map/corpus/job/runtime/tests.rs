//! Deadline, cancellation and metadata contracts for durable per-record QA.
use super::*;
use std::cell::Cell;
use crate::native_engine::decode::DecodeCancellationKind;

#[test]
fn host_and_cli_quote_the_same_complete_population_before_any_model_io() {
    let (command, args) = super::super::tests::parsed("start",
        &["--synthesize-answer", "--hierarchical-answer", "--answer-max-passes", "4",
            "--cohort-chunks", "3", "--cohort-hierarchy-groups", "2"]);
    let job = super::super::tests::lifetime(&command, &args);
    let common = command.args.validate(&args).unwrap();
    let limits = command.args.lifetime_limits(&args, common, AnswerOptions::default(), job).unwrap();
    let host = host_limits(&args, limits, Duration::from_secs(60));
    let quote = crate::hosted::int4_question_document_job_requirements(args.group_width,
        limits.config, job, host, limits.reduction_reserve_bytes,
        command.args.questions.cohort_chunks, command.args.questions.cohort_hierarchy_groups,
        command.args.storage.input_retention.input_memory().unwrap()).unwrap();
    assert_eq!(quote, limits.memory);
    assert_eq!(host.required_buffer_bytes_for_input(job,
        command.args.storage.input_retention.input_memory().unwrap()).unwrap(), quote.population_buffer_bytes);
    assert_eq!(limits.cli_preparation_bytes,
        limits.common.preparation_bytes + 2 * IO_BUFFER_BYTES as u64 + job_cli::REPORT_BYTES as u64);
}

#[test]
fn terminal_native_cancellation_retains_its_public_job_exit_category() {
    for (cause, expected) in [
        (DecodeCancellationKind::Deadline, ErrorCode::BudgetOrTimeout),
        (DecodeCancellationKind::Timeout, ErrorCode::BudgetOrTimeout),
    ] {
        let error = HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause)));
        assert_eq!(model_failure(error).exit, expected);
    }
}

#[test]
fn serialization_deadline_does_not_publish_a_success_report() {
    let checks = Cell::new(0);
    let remaining = || {
        checks.set(checks.get() + 1);
        if checks.get() == 2 { Err(CandidateError::Timeout) } else { Ok(Duration::from_secs(1)) }
    };
    let mut output = Vec::new();
    let failure = publish_report(&serde_json::json!({"status":"complete"}), &mut output, &remaining).unwrap_err();
    assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
    assert!(output.is_empty());
}

#[test]
fn deadline_after_delivery_is_nonzero_even_when_a_complete_frame_exists() {
    let checks = Cell::new(0);
    let remaining = || {
        checks.set(checks.get() + 1);
        if checks.get() == 3 { Err(CandidateError::Timeout) } else { Ok(Duration::from_secs(1)) }
    };
    let mut output = Vec::new();
    let failure = publish_report(&serde_json::json!({"status":"complete"}), &mut output, &remaining).unwrap_err();
    assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout);
    assert_eq!(output.iter().filter(|&&byte| byte == b'\n').count(), 1);
    let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(value["status"], "complete");
}

#[test]
fn public_question_job_modes_are_explicit_and_private_question_content_is_absent() {
    for (extra, expected) in [
        (&[][..], "independent_passage_answers"),
        (&["--synthesize-answer"][..], "verified_evidence_synthesis"),
        (&["--synthesize-answer", "--hierarchical-answer", "--answer-max-passes", "4"][..],
            "hierarchical_verified_evidence_synthesis"),
    ] {
        let (command, args) = super::super::tests::parsed("start", extra);
        let job = super::super::tests::lifetime(&command, &args);
        let common = command.args.validate(&args).unwrap();
        let limits = command.args.lifetime_limits(&args, common, AnswerOptions::default(), job).unwrap();
        assert_eq!(answer_mode(limits.config), expected);
        let response = Response { protocol: "fnlp-candidate-int4-answer-document-job-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "answer",
            durable_unit: "complete_original_source_question", question_source: "per_record",
            answer_mode: expected, model_id: "Nanbeige4.2-3B", source_revision: "public-revision",
            source_root_sha256: "public-source", model_content_sha256: "public-model",
            candidate_format: "candidate", quant_recipe: "fixed", group_width: 128,
            integer_kernel: "s_scalar_i32", snapshot: None, cohort_chunks: None,
            cohort_hierarchy_groups: None, output: &serde_json::json!({"completed":2}) };
        let encoded = serde_json::to_string(&response).unwrap();
        assert!(!encoded.contains("/private/"));
        assert!(!encoded.contains("task_args"));
        assert!(!encoded.contains("max_question_bytes"));
        let value: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        assert_eq!(value["question_source"], "per_record");
        assert_eq!(value["answer_mode"], expected);
    }
}
