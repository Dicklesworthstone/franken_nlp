//! Admission and wire-policy checks need neither a model nor private input IO.
use super::*;

pub(in crate::candidate_cli) const POLICY: &str = r#"{
    "minimum_candidate_weight_ppm":700000,"minimum_margin_milli":100,
    "evidence_window_bytes":256,"max_evidence_windows":8,"max_evidence_spans":4
}"#;

pub(in crate::candidate_cli) fn command(extra: &[&str]) -> FaithfulnessCorpusCommand {
    let mut argv = vec!["candidate", "faithfulness-corpus", "--model", "fixture.fnlpq",
        "--memory-mib", "65536", "--policy", "policy.json"];
    argv.extend_from_slice(extra);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::FaithfulnessCorpus(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("complete-source faithfulness corpus required"); };
    command
}

#[test]
fn record_count_does_not_multiply_document_or_invocation_authority() {
    let small = command(&["--max-requests", "2"]).validate().unwrap();
    let large = command(&["--max-requests", "1000"]).validate().unwrap();
    assert_eq!(small.document.mapping.chunks, large.document.mapping.chunks);
    assert_eq!(small.document.mapping.max_model_work, large.document.mapping.max_model_work);
    assert_eq!(small.document.mapping.max_result_bytes, large.document.mapping.max_result_bytes);
    assert_eq!(small.document.mapping.max_retained_result_bytes, large.document.mapping.max_retained_result_bytes);
    assert_eq!(small.document.mapping.aggregation.minimum_classified_coverage_ppm,
        large.document.mapping.aggregation.minimum_classified_coverage_ppm);
    assert_eq!(small.document.task, large.document.task);
    assert_eq!(small.transport.max_work, large.transport.max_work);
    assert_eq!(small.document.common.max_checkpoints, large.document.common.max_checkpoints);
    assert_eq!(small.document.common.max_input_bytes, large.document.common.max_input_bytes);
    assert_eq!((small.transport.max_requests, large.transport.max_requests), (2, 1000));
    assert!(large.transport.max_output_bytes < small.transport.max_output_bytes);
}

#[test]
fn policy_is_fixed_strict_and_never_accepts_source_or_budget_authority() {
    let parsed = parse_policy(POLICY).unwrap();
    assert_eq!(parsed.evidence_window_bytes, 256);
    assert_eq!(parsed.max_evidence_windows, 8);
    assert_eq!(parsed.max_evidence_spans, 4);
    for text in ["", "null", "{}",
        r#"{"minimum_candidate_weight_ppm":0,"\u006dinimum_candidate_weight_ppm":1}"#] {
        assert!(parse_policy(text).is_err());
    }
    for (field, value) in [
        ("claim", serde_json::json!("private")), ("source", serde_json::json!("private")),
        ("budget", serde_json::json!({})), ("max_evidence_windows", serde_json::json!(32)),
        ("evidence_window_bytes", serde_json::json!(0)), ("max_evidence_spans", serde_json::json!(9)),
        ("minimum_candidate_weight_ppm", serde_json::json!(1_000_001)),
    ] {
        let mut wire: serde_json::Value = serde_json::from_str(POLICY).unwrap();
        wire[field] = value;
        assert!(parse_policy(&wire.to_string()).is_err(), "{field}");
    }
    assert!(parse_policy(&" ".repeat(POLICY_BYTES + 1)).is_err());
}

#[test]
fn compact_document_and_invocation_allowances_keep_separate_scopes() {
    let plain = command(&[]).validate().unwrap();
    assert_eq!(plain.document.encoding, UntrustedEncoding::ByteFallback);
    assert!(plain.max_tokenizer_work.is_none());
    let compact = command(&["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-input-bytes", "65536", "--max-tokenizer-output-tokens", "131072",
        "--max-total-tokenizer-input-bytes", "262144",
        "--max-total-tokenizer-output-tokens", "524288"]).validate().unwrap();
    let UntrustedEncoding::SafeBpeV1(per_document) = compact.document.encoding
        else { panic!("compact document policy"); };
    assert_eq!((per_document.max_input_bytes, per_document.max_output_tokens), (65536, 131072));
    assert!(per_document.max_output_tokens > compact.document.common.context_tokens);
    let invocation = compact.max_tokenizer_work.unwrap();
    assert_eq!((invocation.max_input_bytes, invocation.max_output_tokens), (262144, 524288));
    assert_eq!(compact.document.mapping.max_model_work, plain.document.mapping.max_model_work);
    for flag in ["--max-total-tokenizer-input-bytes", "--max-total-tokenizer-output-tokens",
        "--max-total-tokenizer-pair-checks", "--max-total-tokenizer-pair-bytes"] {
        assert!(command(&[flag, "1"]).validate().is_err());
        assert!(command(&["--input-encoding", "safe-bpe-v1", flag, "0"]).validate().is_err());
    }
}

#[test]
fn chunk_document_and_stream_outputs_have_independent_byte_limits() {
    let admitted = command(&["--max-result-bytes", "65536", "--max-map-result-bytes", "4194304",
        "--max-retained-result-bytes", "8388608", "--max-total-output-bytes", "33554432"])
        .validate().unwrap();
    assert_eq!(admitted.document.task.max_output_bytes, 65536);
    assert_eq!(admitted.document.mapping.max_result_bytes, 4 * MIB);
    assert_eq!(admitted.document.mapping.max_retained_result_bytes, 8 * MIB);
    assert!(admitted.transport.max_output_line_bytes > 4 * MIB as usize);
    assert_eq!(admitted.output_bytes, 32 * MIB);
    assert!(admitted.transport.max_output_bytes < admitted.output_bytes);
}

struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input read"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output write"); }
    fn flush(&mut self) -> io::Result<()> { panic!("output flush"); }
}

#[test]
fn malformed_stream_admission_refuses_before_policy_source_or_model_io() {
    for flags in [vec!["--max-requests", "0"], vec!["--max-requests", "100001"],
        vec!["--max-total-input-bytes", "0"], vec!["--max-total-output-bytes", "1"],
        vec!["--max-line-bytes", "1"], vec!["--max-line-bytes", "4194305"],
        vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
        vec!["--prefill-rows", "0"], vec!["--max-chunks", "0"]] {
        assert!(matches!(command(&flags).execute_owned(NoIo, NoIo), Err(CandidateError::Arguments)));
    }
    let mut invalid = command(&[]);
    invalid.transport.policy = "-".into();
    assert!(matches!(invalid.execute_owned(NoIo, NoIo), Err(CandidateError::Arguments)));
    assert!(matches!(command(&["--input-encoding", "safe-bpe-v1",
        "--max-total-tokenizer-output-tokens", "1"]).execute_owned(NoIo, NoIo),
        Err(CandidateError::Arguments)));
}

#[test]
fn corpus_requires_owned_stdio_and_cannot_enable_unimplemented_modes() {
    assert!(matches!(CandidateCommand::FaithfulnessCorpus(command(&[]))
        .execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments)));
    assert!(crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "faithfulness-corpus", "--model", "m", "--memory-mib", "65536"]).is_err());
    for flag in ["--seed", "--thinking", "--tools", "--cohort-chunks", "--cohort-heads",
        "--job-dir", "--defaults", "--max-new-tokens"] {
        assert!(crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "faithfulness-corpus", "--model", "m", "--memory-mib", "65536",
            "--policy", "policy.json", flag, "1"]).is_err(), "{flag}");
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_runtime_refuses_a_valid_corpus_before_any_private_io() {
    assert!(matches!(command(&[]).execute_owned(NoIo, NoIo), Err(CandidateError::Unavailable)));
}
