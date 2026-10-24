use super::*;

#[test]
fn typed_planning_and_host_cancellation_never_become_a_judgment() {
    for cause in [DecodeCancellationKind::Deadline, DecodeCancellationKind::Timeout] {
        assert_eq!(planning_error(Int4JudgeError::Cancelled(cause)), CandidateError::Timeout);
        assert_eq!(execution_error(HostedError::Int4(Int4HostError::Judge(
            Int4JudgeError::Cancelled(cause)))), CandidateError::Timeout);
    }
    assert_eq!(planning_error(Int4JudgeError::Cancelled(DecodeCancellationKind::PollQuota)), CandidateError::Execution);
    assert_eq!(planning_error(Int4JudgeError::WorkBudget), CandidateError::Planning);
    assert_eq!(planning_error(Int4JudgeError::Identity), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::Int4(Int4HostError::Judge(Int4JudgeError::Accounting))),
        CandidateError::Execution);
    assert_eq!(execution_error(HostedError::ModelIdentity), CandidateError::Identity);
    assert_eq!(execution_error(HostedError::ResourceDomain), CandidateError::Identity);
    let mut expired = DeadlineControl { started: Instant::now(), deadline: Duration::ZERO };
    assert_eq!(expired.checkpoint(0), Some(DecodeCancellationKind::Deadline));
}

#[test]
fn complete_judgment_publication_preserves_evidence_but_omits_private_provenance() {
    for mode in ["pairwise", "rubric", "faithfulness"] {
        let output = serde_json::json!({"fixture_only":true,"head_count":2,"numerics_profile":"strict-quantized-v2",
            "result":{"mode":mode,"result":{"calibration":"uncalibrated", "evidence":[
                {"quote":"é上海", "span":{"byte_start":4,"byte_end":12}}]}}});
        let provenance = Provenance { protocol: "fnlp-candidate-int4-judge-v1", schema_version: 1,
            scope: "real-source-current-candidate-int4", evidence: "non_authoritative", task: "judge", mode,
            model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture-source",
            model_content_sha256: "fixture-model", candidate_format: "fixture-format", quant_recipe: "fixture:group=32",
            group_width: 32, integer_kernel: "s_scalar_i32", snapshot: None };
        let response = Response { provenance: &provenance, output: &output };
        let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["output"], output);
        assert_eq!(value["protocol"], "fnlp-candidate-int4-judge-v1");
        assert_eq!(value["mode"], mode);
        assert_eq!(value["evidence"], "non_authoritative");
        for absent in ["document", "source", "claim", "criterion", "a", "b", "prompt", "prompt_digest",
            "source_dir", "source_manifest", "snapshot_path"] { assert!(value.get(absent).is_none()); }
        let mut refused = Vec::new();
        assert_eq!(publish(&response, 1, &mut refused), Err(CandidateError::Output));
        assert!(refused.is_empty());
    }
}

#[test]
fn writer_flush_failure_reports_output_failure_after_complete_staging() {
    struct FlushFailure(Vec<u8>);
    impl Write for FlushFailure {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("fixture flush failure")) }
    }
    let complete = serde_json::json!({"fixture_only":true,"complete_judgment":true});
    let mut writer = FlushFailure(Vec::new());
    assert_eq!(publish(&complete, 4096, &mut writer), Err(CandidateError::Output));
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&writer.0).unwrap(), complete);
}
