use super::*;

fn team_command(flags: &[&str]) -> Int4Command {
    let mut argv = vec!["candidate", "source-int4", "sentiment", "--memory-mib", "8192",
        "--source-dir", "explicit-source", "--source-manifest", "explicit-manifest"];
    argv.extend_from_slice(flags);
    let matches = crate::candidate_cli::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::SourceInt4(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("sentiment dispatcher required"); };
    command
}

#[test]
fn compact_sentiment_teams_are_admitted_and_unsupported_combinations_refuse_before_input() {
    let command = team_command(&["--compute-threads", "2", "--input-encoding", "safe-bpe-v1"]);
    let Int4Task::Sentiment(options) = command.task else { panic!("sentiment required"); };
    let limits = options.validate(&command.args).unwrap();
    assert_eq!(limits.compute.threads, 2);
    assert_eq!(limits.compute.workspace_bytes,
        crate::native_engine::strict_int4::projection::int4_team_workspace_bytes(2, command.args.group_width).unwrap());
    assert!(limits.compute.workspace_bytes > 0);
    assert!(limits.prefill.is_none() && limits.cohort.is_none());
    assert!(!command.args.input_encoding.is_default());
    struct NoRead;
    impl Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused team reached private input"); }
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
}

#[test]
fn sentiment_team_receipt_is_optional_and_composes_with_compact_policy_without_changing_output() {
    let output = serde_json::json!({"fixture_only":true,"output":{"head_count":4,"result":{"dimensions":[]}}});
    let receipt = crate::hosted::Int4ComputeReceipt { schedule: "int4-output-row-team-v1", compute_threads: 2,
        completed_projections: 7, completed_rows: 100, completed_worker_rows: 50 };
    let policy = crate::tokenizer::untrusted::UntrustedEncoding::SafeBpeV1(
        crate::tokenizer::untrusted::SafeBpeLimits { max_input_bytes: 1024, max_output_tokens: 512,
            max_pair_checks: 4096, max_pair_bytes: 65536 });
    for (reported, encoding) in [(None, None), (Some(&receipt), Some(&policy))] {
        let envelope = input_encoding::Response { response: &compute::Response {
            response: &cohort::Response { response: &output, cohort: None }, compute: reported,
        }, input_encoding: encoding };
        let mut bytes = Vec::new(); publish_checked(&envelope, 4096, &mut bytes, &|| Ok(())).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["output"], output["output"]);
        assert!(value.get("cohort").is_none());
        if reported.is_some() {
            assert_eq!(value["compute"]["compute_threads"], 2);
            assert_eq!(value["compute"]["completed_worker_rows"], 50);
            assert_eq!(value["input_encoding"]["kind"], "safe_bpe_v1");
        } else {
            assert_eq!(value, output);
        }
    }
}

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
