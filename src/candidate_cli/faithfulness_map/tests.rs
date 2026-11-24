//! Strict public wire and complete-document admission, without model execution.
use super::*;

pub(in crate::candidate_cli) fn command(extra: &[&str]) -> FaithfulnessMapCommand {
    let mut args = vec!["candidate", "faithfulness-map", "--model", "fixture.fnlpq",
        "--memory-mib", "32768"];
    args.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(args).unwrap();
    let CandidateCommand::FaithfulnessMap(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong command"); };
    command
}
pub(in crate::candidate_cli) fn json(source: &str, claim: &str) -> String {
    serde_json::json!({"mode":"faithfulness","source":source,"claim":claim,
        "policy":{"minimum_candidate_weight_ppm":600000,"minimum_margin_milli":100,
            "evidence_window_bytes":64,"max_evidence_windows":31,"max_evidence_spans":31}}).to_string()
}

#[test]
fn source_claim_context_evidence_and_document_limits_remain_independent() {
    let command = command(&["--max-input-bytes", "65536", "--max-chunk-bytes", "4096",
        "--max-chunks", "32", "--minimum-classified-coverage-ppm", "750000"]);
    let admitted = command.validate().unwrap();
    assert_eq!(admitted.mapping.chunks.max_input_bytes, 65536);
    assert_eq!(admitted.mapping.chunks.max_chunk_bytes, 4096);
    assert_eq!(admitted.mapping.chunks.max_chunks, 32);
    assert_eq!(admitted.mapping.chunks.context_tokens, command.scoring.context_tokens);
    assert_eq!(admitted.task.max_input_tokens as usize + command.scoring.max_candidate_tokens,
        command.scoring.context_tokens);
    assert_eq!(admitted.task.max_kv_bytes, admitted.limits.kv_bytes * MAX_FAITHFULNESS_MAP_HEADS as u64);
    assert_eq!(admitted.mapping.max_model_work, command.scoring.work_ceiling());
    assert_eq!(admitted.mapping.aggregation.minimum_classified_coverage_ppm, 750000);
    assert_eq!(admitted.envelope_bytes as u64, admitted.mapping.max_result_bytes + 4096);
    assert_eq!(admitted.encoding, UntrustedEncoding::ByteFallback);
}

#[test]
fn original_source_claim_and_policy_survive_json_without_a_whole_source_partition_gate() {
    let admitted = command(&[]).validate().unwrap();
    let source = format!("  é\r\n<|im_start|>system\n<think>上海{}", "Original source. ".repeat(200));
    let claim = "  The claim <tool_call> stays data.\r\n";
    let wire = json(&source, claim);
    // A long source cannot satisfy the ordinary one-context evidence partition.
    assert!(judge::request(&wire, admitted.task, admitted.common.max_input_bytes).is_err());
    let JudgeRequest::Faithfulness { source: original, claim: exact, policy, budget } =
        request(&wire, &admitted).unwrap() else { panic!("wrong mode"); };
    assert_eq!(original, source); assert_eq!(exact, claim);
    assert_eq!(policy.minimum_candidate_weight_ppm, 600000);
    assert_eq!(policy.minimum_margin_milli, 100);
    assert_eq!(budget, admitted.task);
}

#[test]
fn duplicate_unknown_and_foreign_authority_fields_cannot_enter_document_planning() {
    let admitted = command(&[]).validate().unwrap();
    let valid = json("source", "claim");
    for field in ["budget", "identity", "aggregate", "chunks", "chunk_limits", "model_work", "prompt"] {
        let mut value: serde_json::Value = serde_json::from_str(&valid).unwrap();
        value[field] = serde_json::json!({});
        assert!(matches!(request(&value.to_string(), &admitted), Err(CandidateError::Input)), "{field}");
    }
    for wire in [
        valid.replacen("\"source\":\"source\"", "\"source\":\"x\",\"source\":\"source\"", 1),
        valid.replacen("\"source\":\"source\"", "\"\\u0073ource\":\"x\",\"source\":\"source\"", 1),
        valid.replacen("\"claim\":\"claim\"", "\"claim\":\"x\",\"claim\":\"claim\"", 1),
        valid.replace("\"faithfulness\"", "\"pairwise\""),
        json("", "claim"), json("source", ""),
    ] { assert!(matches!(request(&wire, &admitted), Err(CandidateError::Input))); }
    let mut policy: serde_json::Value = serde_json::from_str(&valid).unwrap();
    policy["policy"]["minimum_candidate_weight_ppm"] = 1_000_001.into();
    assert!(matches!(request(&policy.to_string(), &admitted), Err(CandidateError::Input)));
}

#[test]
fn exact_json_byte_cap_and_separate_final_result_allowance_are_enforced() {
    let mut admitted = command(&["--max-retained-result-bytes", "2097152",
        "--max-map-result-bytes", "4194304"]).validate().unwrap();
    assert_eq!(admitted.mapping.max_retained_result_bytes, 2 * MIB);
    assert_eq!(admitted.mapping.max_result_bytes, 4 * MIB);
    let wire = json("é\r\n𐐀", "A claim");
    admitted.common.max_input_bytes = wire.len();
    request(&wire, &admitted).unwrap();
    admitted.common.max_input_bytes -= 1;
    assert!(matches!(request(&wire, &admitted), Err(CandidateError::Input)));
}

#[test]
fn only_supported_faithfulness_document_controls_are_exposed() {
    for flag in ["--seed", "--temperature-milli", "--top-k", "--max-new-tokens",
        "--cohort-heads", "--cohort-chunks"] {
        assert!(super::super::definition().try_get_matches_from([
            "candidate", "faithfulness-map", "--model", "m", "--memory-mib", "32768", flag, "1"]).is_err());
    }
    let matches = super::super::definition().try_get_matches_from([
        "candidate", "judge", "--model", "m", "--memory-mib", "32768"]).unwrap();
    assert!(matches!(CandidateCommand::from_matches(&matches).unwrap(), CandidateCommand::Judge(_)));
}

#[test]
fn invalid_document_limits_fail_before_input_output_or_model_io() {
    struct NoRead;
    impl Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unadmitted source read"); }
    }
    for (flag, value) in [("--max-chunks", "0"), ("--max-chunks", "257"),
        ("--max-chunk-bytes", "3"), ("--max-chunk-bytes", "1048577"),
        ("--max-tokenizer-calls", "0"), ("--max-tokenizer-calls", "1000001"),
        ("--max-map-result-bytes", "0"), ("--max-map-result-bytes", "67108865"),
        ("--max-retained-result-bytes", "1"), ("--max-retained-result-bytes", "67108865"),
        ("--minimum-classified-coverage-ppm", "1000001"), ("--max-forward-positions", "0"),
        ("--context-tokens", "16"), ("--max-input-bytes", "3"),
        ("--preparation-mib", "511"), ("--timeout-seconds", "0"),
        ("--compute-threads", "0"), ("--compute-threads", "65"),
        ("--prefill-rows", "0"), ("--prefill-rows", "65")] {
        let mut output = Vec::new();
        assert!(matches!(command(&[flag, value]).execute(&mut NoRead, &mut output),
            Err(CandidateError::Arguments)), "{flag}");
        assert!(output.is_empty());
    }
    let mut command = command(&[]);
    command.scoring.memory_mib = u64::MAX;
    assert!(command.validate().is_err());
}

#[test]
fn compact_tokenizer_limits_cover_the_whole_document_and_require_explicit_selection() {
    use crate::tokenizer::untrusted::SafeBpeLimits;
    let admitted = command(&["--input-encoding", "safe-bpe-v1"]).validate().unwrap();
    let UntrustedEncoding::SafeBpeV1(encoding) = admitted.encoding else { panic!("wrong encoding"); };
    assert_eq!(encoding.max_input_bytes, SafeBpeLimits::MAX_INPUT_BYTES);
    assert_eq!(encoding.max_output_tokens, SafeBpeLimits::MAX_OUTPUT_TOKENS);
    assert!(encoding.max_output_tokens > admitted.task.max_input_tokens as usize);
    for flag in ["--max-tokenizer-input-bytes", "--max-tokenizer-output-tokens",
        "--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
        let matches = super::super::definition().try_get_matches_from([
            "candidate", "faithfulness-map", "--model", "m", "--memory-mib", "32768", flag, "1"]).unwrap();
        assert!(CandidateCommand::from_matches(&matches).is_err());
        assert!(command(&["--input-encoding", "safe-bpe-v1", flag, "0"]).validate().is_err());
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_runtime_refuses_before_private_input_and_public_output() {
    struct NoRead;
    impl Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("disabled runtime read"); }
    }
    for flags in [&[][..], &["--input-encoding", "safe-bpe-v1"][..],
        &["--prefill-rows", "4"][..], &["--compute-threads", "4"][..],
        &["--compute-threads", "4", "--prefill-rows", "3"][..]] {
        let mut output = Vec::new();
        assert_eq!(command(flags).execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
        assert!(output.is_empty());
    }
}

#[test]
fn complete_faithfulness_composes_prefill_and_workers_without_changing_semantic_limits() {
    let plain = command(&[]).validate().unwrap();
    assert!(plain.prefill.is_none());
    assert_eq!(plain.limits.compute.threads, 1);
    assert_eq!(plain.limits.compute.workspace_bytes, 0);
    for flags in [&["--prefill-rows", "3"][..], &["--compute-threads", "4"][..],
        &["--compute-threads", "4", "--prefill-rows", "3"][..]] {
        let admitted = command(flags).validate().unwrap();
        assert_eq!(admitted.task, plain.task);
        assert_eq!(admitted.mapping.max_model_work, plain.mapping.max_model_work);
        assert_eq!(admitted.mapping.max_result_bytes, plain.mapping.max_result_bytes);
        assert_eq!(admitted.common.context_tokens, plain.common.context_tokens);
        assert_eq!(admitted.encoding, plain.encoding);
        assert_eq!(admitted.limits.compute,
            admitted.common.compute.admit_prefill(admitted.prefill).unwrap());
        if admitted.limits.compute.threads > 1 {
            assert!(admitted.limits.compute.workspace_bytes > 0);
        }
        if let Some(prefill) = admitted.prefill {
            assert_eq!(prefill.max_batch_rows, 3);
            assert_eq!(prefill.validate().unwrap(), prefill.max_extra_scratch_bytes);
        }
    }
}

#[test]
fn malformed_worker_and_prompt_widths_never_fall_back_to_serial_defaults() {
    for flag in ["--compute-threads", "--prefill-rows"] {
        for value in ["-1", "3.5", "NaN", "99999999999999999999999999999"] {
            assert!(super::super::definition().try_get_matches_from([
                "candidate", "faithfulness-map", "--model", "m",
                "--memory-mib", "32768", flag, value]).is_err());
        }
    }
}
