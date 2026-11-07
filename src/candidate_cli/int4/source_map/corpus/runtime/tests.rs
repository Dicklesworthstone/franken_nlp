//! Host quote, cancellation and publication fixtures without synthetic native answers.
use super::*;
use super::super::tests::{arguments, NoIo};
use crate::batch::{BatchFault, BatchRunError, BatchSummary};

#[test]
fn stale_admission_never_bypasses_option_validation_before_owned_io() {
    let (mut command, args) = arguments(&[]);
    let admitted = command.validate(&args).unwrap();
    command.stream.max_question_bytes = 0;
    assert_eq!(execute(args, command, admitted, NoIo, NoIo), Err(CandidateError::Arguments));
    let (command, mut args) = arguments(&["--synthesize-answer", "--cohort-chunks", "3"]);
    let admitted = command.validate(&args).unwrap(); args.seed = Some("ab".repeat(32));
    assert_eq!(execute(args, command, admitted, NoIo, NoIo), Err(CandidateError::Arguments));
}

#[test]
fn host_and_cli_share_complete_source_question_discovery_and_synthesis_admission() {
    for flags in [vec![], vec!["--synthesize-answer"], vec!["--cohort-chunks", "1"],
        vec!["--synthesize-answer", "--cohort-chunks", "3"]] {
        let (command, args) = arguments(&flags);
        let options = AnswerOptions { max_answer_scalars: 80, max_citations: 2, max_quote_scalars: 24 };
        let limits = command.validate_options(&args, options).unwrap();
        let host = host_limits(&args, limits, Duration::from_secs(19));
        let quoted = crate::hosted::int4_question_document_corpus_requirements(args.group_width, limits.config,
            host, limits.reduction_reserve_bytes, command.stream.cohort_chunks).unwrap();
        assert_eq!(quoted, limits.memory);
        assert_eq!(host.native.context_tokens, args.context_tokens);
        assert_eq!(host.native.run.max_checkpoints, args.max_checkpoints);
        assert_eq!(host.native.run.max_elapsed, Duration::from_secs(19));
        assert_eq!(host.preparation_reserve_bytes, limits.common.preparation_bytes);
        let mut changed = host; changed.native.context_tokens += 1;
        assert!(crate::hosted::int4_question_document_corpus_requirements(args.group_width, limits.config,
            changed, limits.reduction_reserve_bytes, command.stream.cohort_chunks).is_err());
    }
}

fn provenance(synthesis: bool, chunks: Option<usize>) -> Provenance<'static> {
    Provenance { protocol: "fnlp-candidate-int4-answer-corpus-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "answer",
        source_mode: "complete_document", question_source: "per_record", answer_mode: answer_mode(synthesis),
        chunk_schedule: chunk_schedule(chunks), cohort_chunks: chunks,
        model_id: "Nanbeige4.2-3B", source_revision: "fixture-public-revision",
        source_root_sha256: "fixture-public-source", model_content_sha256: "fixture-public-model",
        candidate_format: "fixture-public-format", quant_recipe: "fixture-grouped-int4",
        group_width: 128, integer_kernel: "s_scalar_i32", snapshot: None }
}

#[test]
fn public_mode_metadata_cannot_export_questions_options_or_private_paths() {
    for (synthesis, chunks) in [(false, None), (false, Some(1)), (true, Some(3))] {
        let metadata = provenance(synthesis, chunks);
        let event = serde_json::json!({"event":"publication_fixture","exact":12345678901234567890_u64});
        let mut frame = crate::canonjson::canonical_bytes(&event).unwrap(); frame.push(b'\n');
        let mut output = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut output, &metadata, 8192, 16384).unwrap();
            writer.write_all(&frame).unwrap(); writer.flush().unwrap();
        }
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["record"], event);
        assert_eq!(value["question_source"], "per_record");
        assert_eq!(value["answer_mode"], answer_mode(synthesis));
        assert_eq!(value["chunk_schedule"], chunk_schedule(chunks));
        assert_eq!(value.get("cohort_chunks").is_some(), chunks.is_some());
        for private in ["question", "text", "source", "options", "defaults", "task_args", "prompt",
            "input", "question_path", "options_path", "snapshot_path", "source_dir", "source_manifest"] {
            assert!(value.get(private).is_none(), "{private}");
        }
    }
}

#[test]
fn cancellation_and_partial_population_failures_remain_unsuccessful() {
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout,
        DecodeCancellationKind::User, DecodeCancellationKind::PollQuota] {
        let expected = stopped(cause);
        assert_eq!(host_error(HostedError::BatchSetup(BatchFault::cancelled(cause)), CandidateError::Planning), expected);
        assert_eq!(host_error(HostedError::Batch(BatchRunError { fault: BatchFault::cancelled(cause),
            summary: BatchSummary { succeeded: 1, ..BatchSummary::default() } }), CandidateError::Batch), expected);
        assert_eq!(host_error(HostedError::Int4(Int4HostError::Native(StrictInt4Error::Cancelled(cause))),
            CandidateError::Model), expected);
        assert_eq!(host_error(HostedError::Int4(Int4HostError::Snapshot(Int4SnapshotError::Cancelled(cause))),
            CandidateError::Model), expected);
    }
    assert_eq!(host_error(HostedError::ModelIdentity, CandidateError::Batch), CandidateError::Identity);
    assert_eq!(completed(BatchSummary { succeeded: 2, failed: 1, ..BatchSummary::default() }), Err(CandidateError::Batch));
    assert_eq!(completed(BatchSummary { succeeded: 2, ..BatchSummary::default() }), Ok(()));
    assert!(diagnostic(CandidateError::Timeout).contains("earlier completed frames"));
    assert!(diagnostic(CandidateError::Execution).contains("earlier completed frames"));
}
