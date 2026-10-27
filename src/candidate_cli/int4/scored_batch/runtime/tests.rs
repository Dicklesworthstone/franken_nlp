use super::*;
use crate::batch::{BatchCode, BatchFault, BatchRunError, BatchSummary};

struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("invalid scored stream read input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("invalid scored stream wrote output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("invalid scored stream flushed output"); }
}

#[test]
fn owned_boundary_revalidates_policy_before_defaults_model_or_stream_io() {
    for task in ["classify", "sentiment", "judge"] {
        let command = super::super::tests::command(task, &["--defaults", "never-open.json"]);
        let Int4Command { task: Int4Task::ScoredBatch(batch), mut args, batch: None, job: None } = command
            else { panic!("wrong scored stream task"); };
        let quote = batch.validate(&args).unwrap();
        args.seed = Some("ab".repeat(32));
        assert_eq!(execute(args, batch, quote, NoIo, NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn sentiment_stream_policy_keeps_code_owned_full_vocabulary_and_pinned_eos() {
    let policy = SentimentPolicy { minimum_peak_weight_ppm: 700000, maximum_normalized_entropy_ppm: 600000 };
    let selected = sentiment_options(policy).unwrap();
    assert_eq!(selected.mode, ScoringMode::FullVocabulary);
    assert_eq!(selected.policy, policy);
    let controls = crate::tokenizer::pinned_controls::pinned().unwrap();
    let expected = controls.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    assert_eq!(selected.eos_token_id, expected);
}

#[test]
fn typed_stream_cancellation_stays_distinct_from_failed_items_and_abstention() {
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout, DecodeCancellationKind::PollQuota] {
        let expected = if cause == DecodeCancellationKind::PollQuota { CandidateError::Execution } else { CandidateError::Timeout };
        assert_eq!(execution_error(HostedError::BatchSetup(BatchFault::cancelled(cause))), expected);
        assert_eq!(execution_error(HostedError::Batch(BatchRunError {
            fault: BatchFault::cancelled(cause), summary: BatchSummary::default(),
        })), expected);
    }
    assert_eq!(execution_error(HostedError::BatchSetup(BatchCode::InvalidLimits.into())), CandidateError::Batch);
    assert_eq!(execution_error(HostedError::ModelIdentity), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::ResourceDomain), CandidateError::Identity);
    assert_eq!(completed(BatchSummary { succeeded: 2, failed: 1, ..BatchSummary::default() }), Err(CandidateError::Batch));
    assert_eq!(completed(BatchSummary { succeeded: 2, ..BatchSummary::default() }), Ok(()));
}

#[test]
fn every_scored_task_keeps_exact_native_records_and_explicit_snapshot_provenance() {
    let need = Int4SnapshotRequirements::for_group_width(128).unwrap();
    for task in ["classify", "sentiment", "judge"] {
        let provenance = Provenance {
            protocol: "fnlp-candidate-int4-score-batch-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task,
            model_id: "Nanbeige4.2-3B", source_revision: "fixture-revision",
            source_root_sha256: "fixture-source", model_content_sha256: "fixture-content",
            candidate_format: "fnlp-int4-source-candidate-v1", quant_recipe: "fixture:group=128",
            group_width: 128, integer_kernel: "s_scalar_i32",
            snapshot: Some(SnapshotRecord::new("loaded", need)),
        };
        // Publication fixture only; these synthetic bytes assert no model result.
        let inner = "{\"event\":\"fixture\",\"label\":\"café 上海\",\"calibration\":\"uncalibrated\",\"exact\":12345678901234567890123456789012345678}\n";
        let mut output = Vec::new();
        {
            let mut writer = CandidateWriter::new(&mut output, &provenance, 4096, 8192).unwrap();
            writer.write_all(inner.as_bytes()).unwrap(); writer.flush().unwrap();
        }
        let bytes = std::str::from_utf8(&output).unwrap();
        assert!(bytes.ends_with(&format!("\"record\":{}}}\n", inner.trim_end_matches('\n'))));
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["task"], task);
        assert_eq!(value["record"]["label"], "café 上海");
        assert_eq!(value["record"]["calibration"], "uncalibrated");
        assert_eq!(value["protocol"], "fnlp-candidate-int4-score-batch-v1");
        assert_eq!(value["evidence"], "non_authoritative");
        assert_eq!(value["model_content_sha256"], "fixture-content");
        assert_eq!(value["group_width"], 128);
        assert_eq!(value["snapshot"]["format"], INT4_SNAPSHOT_VERSION);
        assert_eq!(value["snapshot"]["file_bytes"], need.file_bytes);
        for absent in ["document", "prompt", "source_dir", "source_manifest", "snapshot_path",
            "defaults_path", "prompt_digest", "taskir_digest"] {
            assert!(value.get(absent).is_none());
        }
        assert!(matches!(CandidateWriter::new(NoIo, &provenance, 4096, 1), Err(CandidateError::Output)));
    }
}
