use super::*;

fn team_command(flags: &[&str]) -> Int4Command {
    let mut argv = vec!["candidate", "source-int4", "judge", "--memory-mib", "8192",
        "--source-dir", "explicit-source", "--source-manifest", "explicit-manifest"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("judge dispatcher required"); };
    command
}

#[test]
fn judge_teams_admit_scratch_and_refuse_unsupported_composition() {
    let command = team_command(&["--compute-threads", "2"]);
    let Int4Task::Judge(options) = command.task else { panic!("judge required"); };
    let limits = options.validate(&command.args).unwrap();
    assert_eq!(limits.compute.threads, 2);
    assert_eq!(limits.compute.workspace_bytes,
        crate::native_engine::strict_int4::projection::int4_team_workspace_bytes(2, command.args.group_width).unwrap());
    assert!(limits.compute.workspace_bytes > 0);
    assert!(command.args.input_encoding.is_default());
    struct NoRead;
    impl Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused judgment reached private input"); }
    }
    let mut under = command;
    under.args.memory_mib = limits.minimum_memory_bytes.saturating_sub(1) / MIB;
    assert_eq!(under.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Arguments));
    for flags in [vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
        vec!["--compute-threads", "2", "--prefill-rows", "2"],
        vec!["--compute-threads", "2", "--cohort-heads", "4"]] {
        let mut output = Vec::new();
        assert_eq!(team_command(&flags).execute(&mut NoRead, &mut output), Err(CandidateError::Arguments));
        assert!(output.is_empty());
    }
    let compact = team_command(&["--compute-threads", "2", "--input-encoding", "safe-bpe-v1"]);
    let Int4Task::Judge(options) = compact.task else { panic!("judge required"); };
    let admitted = options.validate(&compact.args).unwrap();
    assert_eq!(admitted.compute, limits.compute);
    assert_eq!(admitted.minimum_memory_bytes,
        limits.minimum_memory_bytes + options.encoding.workspace_bytes(&compact.args).unwrap());
}

#[test]
fn judge_team_receipt_is_optional_and_keeps_evidence_identical() {
    let output = serde_json::json!({"fixture_only":true,"output":{"head_count":2,"result":{
        "mode":"faithfulness","evidence":[{"quote":"é上海","span":{"byte_start":4,"byte_end":12}}]}}});
    let receipt = crate::hosted::Int4ComputeReceipt { schedule: "int4-output-row-team-v1", compute_threads: 2,
        completed_projections: 7, completed_rows: 100, completed_worker_rows: 50 };
    for reported in [None, Some(&receipt)] {
        let envelope = compute::Response {
            response: &cohort::Response { response: &output, cohort: None }, compute: reported,
        };
        let mut bytes = Vec::new(); publish_checked(&envelope, 4096, &mut bytes, &|| Ok(())).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["output"], output["output"]);
        assert!(value.get("cohort").is_none());
        if reported.is_some() {
            assert_eq!(value["compute"]["compute_threads"], 2);
            assert_eq!(value["compute"]["completed_worker_rows"], 50);
        } else { assert_eq!(value, output); }
    }
}

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
            group_width: 32, integer_kernel: "s_scalar_i32", snapshot: None,
            input_encoding: None, input_encoding_work: None };
        let response = Response { provenance: &provenance, output: &output };
        let mut bytes = Vec::new(); publish(&response, 4096, &mut bytes).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["output"], output);
        assert!(value.get("input_encoding").is_none());
        assert!(value.get("input_encoding_work").is_none());
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

#[test]
fn compact_judgment_provenance_preserves_the_complete_inner_result() {
    let output = serde_json::json!({"fixture_only":true,"result":{"mode":"faithfulness",
        "evidence":[{"quote":"é上海","span":{"byte_start":4,"byte_end":12}}]}});
    let encoding = UntrustedEncoding::SafeBpeV1(SafeBpeLimits {
        max_input_bytes: 4096, max_output_tokens: 2048,
        max_pair_checks: 20000, max_pair_bytes: 200000 });
    let work = crate::tokenizer::untrusted::SafeBpeWork::default();
    let provenance = Provenance { protocol: "fnlp-candidate-int4-judge-v1", schema_version: 1,
        scope: "fixture", evidence: "non_authoritative", task: "judge", mode: "faithfulness",
        model_id: "Nanbeige4.2-3B", source_revision: "fixture", source_root_sha256: "fixture",
        model_content_sha256: "fixture", candidate_format: "fixture", quant_recipe: "fixture",
        group_width: 32, integer_kernel: "s_scalar_i32", snapshot: None,
        input_encoding: Some(&encoding), input_encoding_work: Some(work) };
    let response = Response { provenance: &provenance, output: &output };
    let mut bytes = Vec::new();
    publish_checked(&response, 4096, &mut bytes, &|| Ok(())).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["output"], output);
    assert_eq!(value["input_encoding"], serde_json::to_value(encoding).unwrap());
    assert_eq!(value["input_encoding_work"], serde_json::to_value(work).unwrap());
}
