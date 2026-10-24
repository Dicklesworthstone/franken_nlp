use super::*;

#[test]
fn code_owned_scoring_always_uses_pinned_terminal_eos_and_full_vocabulary() {
    let policy = SentimentPolicy { minimum_peak_weight_ppm: 700000, maximum_normalized_entropy_ppm: 600000 };
    let selected = options(policy).unwrap();
    assert_eq!(selected.mode, ScoringMode::FullVocabulary);
    assert_eq!(selected.policy, policy);
    let controls = crate::tokenizer::pinned_controls::pinned().unwrap();
    let expected = controls.template_controls().entries().iter()
        .find(|entry| entry.special && entry.surface == crate::template::IM_END).unwrap().id;
    assert_eq!(selected.eos_token_id, expected);
}

#[test]
fn typed_planning_and_host_cancellation_never_become_abstention() {
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout] {
        assert_eq!(planning_error(Int4SentimentError::Cancelled(cause)), CandidateError::Timeout);
        assert_eq!(execution_error(HostedError::Int4(Int4HostError::Sentiment(
            Int4SentimentError::Cancelled(cause)))), CandidateError::Timeout);
    }
    assert_eq!(planning_error(Int4SentimentError::Cancelled(DecodeCancellationKind::PollQuota)),
        CandidateError::Execution);
    assert_eq!(planning_error(Int4SentimentError::WorkBudget), CandidateError::Planning);
    assert_eq!(execution_error(HostedError::Int4(Int4HostError::Sentiment(Int4SentimentError::Accounting))),
        CandidateError::Execution);
    assert_eq!(execution_error(HostedError::ModelIdentity), CandidateError::Identity);
    let mut expired = DeadlineControl { started: Instant::now(), deadline: Duration::ZERO };
    assert_eq!(expired.checkpoint(0), Some(DecodeCancellationKind::Deadline));
}

#[test]
fn complete_publication_retains_native_metadata_and_never_serializes_private_inputs() {
    let output = serde_json::json!({"fixture_only":true,"head_count":2,"numerics_profile":"strict-quantized-v2",
        "result":{"calibration":"uncalibrated","dimensions":[
            {"axis":"valence","decision":"estimated","estimate":0.2},
            {"axis":"arousal","decision":"abstained","estimate":null}]}});
    let provenance = Provenance { protocol: "fnlp-candidate-int4-sentiment-v1", schema_version: 1,
        scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "sentiment",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
        model_content_sha256: "fixture-model", candidate_format: "fixture-format", quant_recipe: "fixture:group=32",
        group_width: 32, integer_kernel: "s_scalar_i32", snapshot: None };
    let response = Response { provenance: &provenance, output: &output };
    let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["output"], output);
    assert_eq!(value["protocol"], "fnlp-candidate-int4-sentiment-v1");
    assert_eq!(value["evidence"], "non_authoritative");
    for absent in ["document", "prompt", "prompt_digest", "source_dir", "source_manifest", "snapshot_path"] {
        assert!(value.get(absent).is_none());
    }
    let mut refused = Vec::new();
    assert_eq!(publish(&response, 1, &mut refused), Err(CandidateError::Output));
    assert!(refused.is_empty());
    struct FlushFailure(Vec<u8>);
    impl Write for FlushFailure {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("fixture flush failure")) }
    }
    let mut failed = FlushFailure(Vec::new());
    assert_eq!(publish(&response, 4096, &mut failed), Err(CandidateError::Output));
    assert!(!failed.0.is_empty(), "a transport failure must be reported after any written bytes");
}
