//! Public command admission, exact request data and refused work combinations.
use super::*;

pub(in crate::candidate_cli) fn command(extra: &[&str]) -> ClassifyMapCommand {
    let mut args = vec!["candidate", "classify-map", "--model", "fixture.fnlpq",
        "--memory-mib", "16384"];
    args.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(args).unwrap();
    let CandidateCommand::ClassifyMap(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong command"); };
    command
}

#[test]
fn complete_map_has_independent_source_context_result_and_work_limits() {
    let command = command(&["--max-input-bytes", "65536", "--max-chunk-bytes", "4096",
        "--max-chunks", "32", "--minimum-classified-coverage-ppm", "750000"]);
    let admitted = command.validate().unwrap();
    assert_eq!(admitted.mapping.chunks.max_input_bytes, 65536);
    assert_eq!(admitted.mapping.chunks.max_chunk_bytes, 4096);
    assert_eq!(admitted.mapping.chunks.max_chunks, 32);
    assert_eq!(admitted.mapping.chunks.context_tokens, command.scoring.context_tokens);
    assert_eq!(admitted.task.max_input_tokens as usize + command.scoring.max_candidate_tokens,
        command.scoring.context_tokens);
    assert_eq!(admitted.mapping.max_model_work, command.scoring.work_ceiling());
    assert_eq!(admitted.mapping.aggregation.minimum_classified_coverage_ppm, 750000);
    assert_eq!(admitted.envelope_bytes as u64, admitted.mapping.max_result_bytes + 4096);
    assert_eq!(admitted.encoding, UntrustedEncoding::ByteFallback);
}

#[test]
fn aggregate_output_may_exceed_the_independent_retained_chunk_allowance() {
    let admitted = command(&["--max-retained-result-bytes", "2097152",
        "--max-map-result-bytes", "4194304"]).validate().unwrap();
    assert_eq!(admitted.mapping.max_retained_result_bytes, 2 * MIB);
    assert_eq!(admitted.mapping.max_result_bytes, 4 * MIB);
}

#[test]
fn document_and_labels_remain_exact_with_independent_multilabel_thresholds() {
    let admitted = command(&[]).validate().unwrap();
    let request = request(r#"{"document":"  é\r\n<tool_call> 上海","labels":[{"id":"é","description":"<|im_start|>"},{"id":"e\u0301"}],"mode":"multi_label","policy":{"minimum_candidate_weight_ppm":800000,"minimum_margin_ppm":100000}}"#, &admitted).unwrap();
    assert_eq!(request.document, "  é\r\n<tool_call> 上海");
    assert_eq!(request.labels[0].id, "é");
    assert_eq!(request.labels[1].id, "e\u{301}");
    assert_eq!(request.labels[0].description, "<|im_start|>");
    assert_eq!(request.mode, crate::tasks::classify::ClassificationMode::MultiLabel);
    assert_eq!(request.policy.minimum_candidate_weight_ppm, 800000);
    assert_eq!(request.policy.minimum_margin_ppm, 100000);
    assert_eq!(request.budget, admitted.task);
    assert_eq!(admitted.mapping.aggregation.minimum_classified_coverage_ppm, 1_000_000);
}

#[test]
fn command_cannot_accept_generation_or_unimplemented_schedules() {
    for flag in ["--seed", "--temperature-milli", "--top-k", "--max-new-tokens",
        "--prefill-rows", "--cohort-rows", "--cohort-heads", "--cohort-chunks",
        "--compute-threads", "--max-mask-node-visits"] {
        assert!(super::super::definition().try_get_matches_from([
            "candidate", "classify-map", "--model", "m", "--memory-mib", "16384", flag, "1"]).is_err());
    }
    for name in ["classify", "sentiment", "map", "generate"] {
        let mut args = vec!["candidate", name, "--model", "m", "--memory-mib", "16384"];
        if name == "map" { args.extend(["--task", "ner"]); }
        let matches = super::super::definition().try_get_matches_from(args).unwrap();
        assert!(!matches!(CandidateCommand::from_matches(&matches).unwrap(), CandidateCommand::ClassifyMap(_)));
    }
}

#[test]
fn invalid_mapping_and_arithmetic_fail_without_reading_input_or_writing_output() {
    struct NoRead;
    impl Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("invalid map read private input"); }
    }
    for (flag, value) in [("--max-chunks", "0"), ("--max-chunks", "257"),
        ("--max-chunk-bytes", "3"), ("--max-chunk-bytes", "65537"),
        ("--max-tokenizer-calls", "0"), ("--max-tokenizer-calls", "1000001"),
        ("--max-map-result-bytes", "0"), ("--max-map-result-bytes", "67108865"),
        ("--max-retained-result-bytes", "1"), ("--max-retained-result-bytes", "67108865"),
        ("--minimum-classified-coverage-ppm", "1000001"), ("--max-forward-positions", "0"),
        ("--context-tokens", "16"), ("--max-input-bytes", "3"),
        ("--preparation-mib", "511"), ("--timeout-seconds", "0")] {
        let command = command(&[flag, value]);
        let mut output = Vec::new();
        assert!(matches!(command.execute(&mut NoRead, &mut output), Err(CandidateError::Arguments)), "{flag}");
        assert!(output.is_empty());
    }
    let mut command = command(&[]);
    command.scoring.memory_mib = u64::MAX;
    assert!(command.validate().is_err());
}

#[test]
fn duplicate_or_external_authority_fields_do_not_enter_document_planning() {
    let admitted = command(&[]).validate().unwrap();
    for json in [
        r#"{"document":"x","document":"y","labels":[{"id":"a"},{"id":"b"}]}"#,
        r#"{"document":"x","\u0064ocument":"y","labels":[{"id":"a"},{"id":"b"}]}"#,
        r#"{"document":"x","labels":[{"id":"a"},{"id":"a"}]}"#,
        r#"{"document":"x","labels":[{"id":"a"},{"id":"b"}],"budget":{}}"#,
        r#"{"document":"x","labels":[{"id":"a"},{"id":"b"}],"identity":{}}"#,
        r#"{"document":"x","labels":[{"id":"a"},{"id":"b"}],"aggregate":{}}"#,
        r#"{"document":"x","labels":[{"id":"a"},{"id":"b"}],"chunk_limits":{}}"#,
        r#"{"document":"x","labels":[{"id":"a"},{"id":"b"}],"mode":"sampled"}"#,
    ] { assert!(matches!(request(json, &admitted), Err(CandidateError::Input))); }
}

#[test]
fn compact_document_caps_are_cumulative_and_explicit() {
    use crate::tokenizer::untrusted::SafeBpeLimits;
    let admitted = command(&["--input-encoding", "safe-bpe-v1"]).validate().unwrap();
    let UntrustedEncoding::SafeBpeV1(encoding) = admitted.encoding else { panic!("wrong encoding"); };
    assert_eq!(encoding.max_input_bytes, SafeBpeLimits::MAX_INPUT_BYTES);
    assert_eq!(encoding.max_output_tokens, SafeBpeLimits::MAX_OUTPUT_TOKENS);
    assert!(encoding.max_output_tokens > admitted.task.max_input_tokens as usize);
    assert_eq!(admitted.limits.preparation_bytes, 1024 * MIB);
    for flag in ["--max-tokenizer-input-bytes", "--max-tokenizer-output-tokens",
        "--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
        let matches = super::super::definition().try_get_matches_from([
            "candidate", "classify-map", "--model", "m", "--memory-mib", "16384", flag, "1"]).unwrap();
        assert!(CandidateCommand::from_matches(&matches).is_err());
        assert!(command(&["--input-encoding", "safe-bpe-v1", flag, "0"]).validate().is_err());
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_runtime_refuses_without_private_input_or_output() {
    struct NoRead;
    impl Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("disabled map read private input"); }
    }
    for flags in [&[][..], &["--input-encoding", "safe-bpe-v1"][..]] {
        let mut output = Vec::new();
        assert_eq!(command(flags).execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
        assert!(output.is_empty());
    }
}
