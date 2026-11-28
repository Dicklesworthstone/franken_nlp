//! Independent host quotation and typed terminal causes; no artifact execution.
use super::*;
use std::io;

fn admitted(extra: &[&str]) -> Admitted {
    let mut argv = vec!["answer-corpus", "--model", "never-opened.fnlpq", "--memory-mib", "65536"];
    argv.extend_from_slice(extra);
    QuestionCorpusCommand::from_matches(&command::definition().try_get_matches_from(argv).unwrap())
        .unwrap().admit().unwrap()
}

#[test]
fn host_and_cli_price_the_same_complete_source_question_frontier_and_schedules() {
    for width in ["1", "2", "4", "64"] {
        for rows in [None, Some("4")] {
            for encoding in ["byte-fallback", "safe-bpe-v1"] {
                let mut args = vec!["--compute-threads", width, "--input-encoding", encoding];
                if let Some(rows) = rows { args.extend(["--prefill-rows", rows]); }
                let admitted = admitted(&args);
                assert_eq!(admit(&admitted).unwrap(), command::validate_peak(&admitted).unwrap());
                let config = admitted.config; let d = &admitted.document;
                let quoted = int8_question_document_corpus_requirements(config,
                    corpus_limits(&admitted, Duration::from_secs(1)), d.reduction_reserve_bytes,
                    d.prefill, (d.shared.compute.threads > 1).then_some(d.shared.compute.threads)).unwrap();
                assert_eq!(quoted.document.input_bytes, admitted.memory.input_bytes);
                assert_eq!(quoted.document.document_workspace_bytes, admitted.memory.document_workspace_bytes);
                assert_eq!(quoted.document.max_output_tokens, 0);
                assert_eq!(quoted.team_workspace_bytes, d.shared.compute.workspace_bytes);
                let extra = d.prefill.map(|p| p.validate().unwrap()).unwrap_or(0);
                assert_eq!(quoted.document.workspace_bytes, admitted.memory.workspace_bytes + extra);
            }
        }
    }
}

#[test]
fn independent_quote_refuses_dropped_complete_input_frontier_output_or_team_storage() {
    for axis in 0..9 {
        let mut admitted = admitted(&["--compute-threads", "4", "--prefill-rows", "4", "--input-encoding", "safe-bpe-v1"]);
        match axis { 0 => admitted.memory.input_bytes -= 1, 1 => admitted.memory.document_workspace_bytes -= 1,
            2 => admitted.memory.output_reservation_bytes -= 1, 3 => admitted.memory.workspace_bytes -= 1,
            4 => admitted.memory.max_output_bytes -= 1, 5 => admitted.memory.max_output_tokens = 1,
            6 => admitted.memory.native.kv_bytes -= 1, 7 => admitted.document.shared.compute.workspace_bytes -= 1,
            _ => admitted.document.shared.memory_bytes = command::validate_peak(&admitted).unwrap() - 1 }
        assert_eq!(admit(&admitted), Err(CandidateError::Arguments), "axis {axis}");
    }
}

#[test]
fn every_typed_cancellation_survives_host_mapping_and_the_original_terminal_poll() {
    use DecodeCancellationKind::*;
    for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
        ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
        let expected = cancelled(cause);
        for error in [HostedError::BatchSetup(BatchFault::cancelled(cause)),
            HostedError::Source(Int8SourceError::Cancelled(cause)),
            HostedError::SourceMap(Int8SourceMapError::Source(Int8SourceError::Cancelled(cause))),
            HostedError::Native(StrictInt8Error::Cancelled(cause)),
            HostedError::Model(CurrentCandidateInt8Error::Artifact(ArtifactBridgeError::Cancelled(cause)))] {
            let failure = checked_stage::<(), _>(Err(error), || Err(CandidateError::Timeout), host_failure).unwrap_err();
            assert_eq!(failure.cancellation, Some(cause)); assert_eq!(failure.exit, expected.exit);
        }
        let failure = session_failure(SessionStageError::Artifact(ArtifactBridgeError::Cancelled(cause)));
        assert_eq!(failure.cancellation, Some(cause)); assert_eq!(failure.exit, expected.exit);
    }
    let timeout = checked_stage::<(), _>(Err(CandidateError::Input), || Err(CandidateError::Timeout), setup_failure).unwrap_err();
    assert_eq!(timeout.cancellation, Some(Deadline));
    let timeout = checked_stage(Ok::<_, CandidateError>(()), || Err(CandidateError::Timeout), setup_failure).unwrap_err();
    assert_eq!(timeout.cancellation, Some(Deadline));
}

#[test]
fn supervision_keeps_underlying_budget_classification_and_broken_diagnostics_preserve_cause() {
    for cause in [DecodeCancellationKind::FailFast, DecodeCancellationKind::LinkedExit] {
        let failure = host_failure(HostedError::Stopped {
            stop: crate::hosted::RunStop { kind: cause, runtime_reason: None, runtime_checkpoint_error: None },
            task_error: Some(Box::new(HostedError::BatchSetup(BatchCode::WorkLimit.into()))),
        });
        assert_eq!(failure.exit, ErrorCode::BudgetOrTimeout); assert_eq!(failure.cancellation, Some(cause));
        assert_eq!(command::report_failure(failure, &mut Broken), failure.exit.as_process_exit());
    }
    assert_eq!(command::report_failure(cancelled(DecodeCancellationKind::User), &mut Broken),
        ErrorCode::Cancelled.as_process_exit());
}
struct Broken;
impl Write for Broken {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { Err(io::Error::other("broken diagnostics")) }
    fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("broken diagnostics")) }
}

#[test]
fn public_provenance_names_per_document_encoding_without_private_request_authority() {
    let admitted = admitted(&["--input-encoding", "safe-bpe-v1", "--compute-threads", "4", "--prefill-rows", "4"]);
    let document = &admitted.document;
    let provenance = Provenance { protocol: "fnlp-candidate-int8-answer-corpus-v1", schema_version: 1,
        scope: "real-artifact-current-candidate", evidence: "non_authoritative", task: "answer",
        source_mode: "complete_document", question_source: "per_record", answer_mode: "independent_passage_answers",
        model_id: "fixture", source_revision: "fixture", source_root_sha256: "public-artifact",
        logical_model_sha256: "public-model", quant_recipe: "fixture",
        input_encoding: input_encoding::reported_policy(&document.encoding),
        document_encoding_work_limits: document.encoding_work, prefill_rows: Some(4), compute_threads: Some(4) };
    let value = serde_json::to_value(provenance).unwrap();
    assert_eq!(value["question_source"], "per_record");
    assert_eq!(value["source_mode"], "complete_document");
    assert!(value.get("document_encoding_work_limits").is_some());
    for key in ["invocation_encoding_work_limits", "question", "source", "text", "prompt_digest", "taskir_digest", "recipe_fingerprint"] {
        assert!(value.get(key).is_none(), "{key}");
    }
}
