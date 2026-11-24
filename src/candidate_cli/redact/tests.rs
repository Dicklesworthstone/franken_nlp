//! Model-free input, secret and policy contracts; no neural-success fixtures.
use super::*;
use std::io::Cursor;
pub(in crate::candidate_cli) fn command(extra: &[&str]) -> RedactCommand {
    let mut argv = vec!["candidate", "redact", "--model", "local.fnlpq", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(argv).unwrap();
    let CandidateCommand::Redact(command) = CandidateCommand::from_matches(&matches).unwrap()
        else { panic!("wrong dispatch") };
    command
}
#[test]
fn defaults_run_ner_and_rules_with_verification_and_no_coordinate_map() {
    let cmd = command(&[]); cmd.validate().unwrap(); let r = cmd.request();
    assert!(r.verify); assert!(!r.actions.include_map);
    assert_eq!(r.actions.default_action, RedactionAction::Mask);
    assert_eq!(r.rules.enabled.len(), 5);
    assert_eq!(ner_options(None).unwrap(), NerOptions::default());
    assert!(r.rule_budget.max_input_bytes >= cmd.host.max_result_bytes);
}
#[test]
fn rule_scope_and_verification_opt_out_are_explicit() {
    let cmd = command(&["--rules", "date,email", "--no-verify", "--include-map", "--action", "placeholder"]);
    cmd.validate().unwrap(); let r = cmd.request();
    assert!(!r.verify); assert!(r.actions.include_map);
    assert_eq!(r.actions.default_action, RedactionAction::Placeholder);
    assert_eq!(r.rules.enabled, [PiiKind::Date, PiiKind::Email].into_iter().collect());
}
#[test]
fn key_and_source_cannot_compete_for_stdin_or_leak_into_other_actions() {
    for extra in [vec!["--key-stdin"], vec!["--action", "pseudonymize"],
        vec!["--action", "pseudonymize", "--key-stdin", "--key-id", "k", "--namespace", "n"],
        vec!["--key-id", "k"], vec!["--namespace", "n"],
        vec!["document.txt", "--action", "pseudonymize", "--key-stdin", "--key-id", "bad id", "--namespace", "n"]] {
        assert!(command(&extra).validate().is_err());
    }
    command(&["document.txt", "--action", "pseudonymize", "--key-stdin", "--key-id", "rotation-1", "--namespace", "job"])
        .validate().unwrap();
    for flag in ["--key", "--key-file", "--seed", "--rules-only"] {
        assert!(super::super::definition().try_get_matches_from(["candidate", "redact", flag, "private"]).is_err());
    }
}
#[test]
fn binary_keys_are_bounded_and_commitments_are_checked() {
    let cmd = command(&["document.txt", "--action", "pseudonymize", "--key-stdin", "--key-id", "rotation", "--namespace", "job"]);
    let bytes = [0_u8; 32];
    let (key, namespace) = cmd.key(&mut Cursor::new(bytes)).unwrap().unwrap();
    assert_eq!(namespace, "job");
    assert_eq!(key.commitment(), PseudonymKey::from_bytes(&bytes, "rotation").unwrap().commitment());
    for len in [0, 31, 4097] { assert!(cmd.key(&mut Cursor::new(vec![0; len])).is_err()); }
    let expected = key.commitment();
    let valid = command(&["document.txt", "--action", "pseudonymize", "--key-stdin", "--key-id", "rotation", "--namespace", "job",
        "--expected-key-commitment", &expected]);
    assert!(valid.key(&mut Cursor::new(bytes)).is_ok());
    assert!(valid.key(&mut Cursor::new([1_u8; 32])).is_err());
}
#[test]
fn secret_options_and_memory_bounds_refuse_before_io() {
    for extra in [vec!["--ner-options", "-"], vec!["--max-detections", "0"], vec!["--max-detections", "16385"],
        vec!["--max-rule-work", "0"], vec!["--edit-reserve-mib", "1"], vec!["--edit-reserve-mib", "18446744073709551615"]] {
        assert!(command(&extra).validate().is_err());
    }
}
#[test]
fn ner_options_reject_duplicate_unknown_or_partial_configurations() {
    for text in ["{}", r#"{"types":["person"],"types":["date"]}"#,
        r#"{"types":["person"],"\u0074ypes":["date"]}"#,
        r#"{"types":["person","person"],"max_entities":2,"max_mention_scalars":32}"#,
        r#"{"types":["person"],"max_entities":2,"max_mention_scalars":32,"instruction":"override"}"#] {
        assert!(ner_options(Some(text)).is_err());
    }
    let text = r#"{"types":["person"],"max_entities":2,"max_mention_scalars":32}"#;
    assert_eq!(ner_options(Some(text)).unwrap().max_entities, 2);
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_feature_never_consumes_even_a_private_key() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    for cmd in [command(&[]), command(&["private.txt", "--action", "pseudonymize", "--key-stdin", "--key-id", "k", "--namespace", "n"])] {
        assert_eq!(cmd.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
    }
}


#[test]
fn compact_redaction_uses_one_operation_allowance_and_preserves_original_input_cap() {
    for verify in [false, true] {
        let mut extra = vec!["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "4",
            "--max-result-bytes", "4096", "--max-tokenizer-pair-checks", "123456",
            "--max-tokenizer-pair-bytes", "7654321"];
        if !verify { extra.push("--no-verify"); }
        let mut cmd = command(&extra);
        cmd.host.memory_mib = 16_384;
        let (common, limits) = cmd.validate().unwrap();
        assert_eq!(cmd.host.preparation_mib, 1024);
        assert_eq!(common.max_input_bytes, 4);
        assert!(read_input(&mut Cursor::new(b"12345"), common.max_input_bytes).is_err());
        let (policy, complete) = cmd.encoding_policy(limits).unwrap();
        let UntrustedEncoding::SafeBpeV1(call) = policy else { panic!("compact policy required"); };
        assert_eq!(call.max_input_bytes, 4096);
        assert_eq!(call.max_output_tokens, limits.max_prompt_tokens);
        let passes = 2 + u64::from(verify);
        assert_eq!(complete.unwrap(), SafeBpeWorkLimits {
            max_input_bytes: 4096 * passes,
            max_output_tokens: limits.max_prompt_tokens as u64 * passes,
            max_pair_checks: 123456, max_pair_bytes: 7654321,
        });
    }
}

#[test]
fn compact_redaction_prices_expanded_workspace_and_every_live_owner() {
    let mut cmd = command(&["--input-encoding", "safe-bpe-v1"]);
    cmd.host.memory_mib = 16_384;
    let (_, limits) = cmd.validate().unwrap();
    let (policy, _) = cmd.encoding_policy(limits).unwrap();
    let UntrustedEncoding::SafeBpeV1(call) = policy else { panic!("compact policy required"); };
    let workspace = call.requirements().unwrap().workspace_bytes + 1024;
    let floor = 256 * MIB + call.max_input_bytes as u64 * 32
        + cmd.host.max_result_bytes as u64 * 16 + workspace;
    let mut exact = limits;
    exact.preparation_bytes = floor;
    assert!(cmd.encoding_policy(exact).is_ok());
    exact.preparation_bytes -= 1;
    assert!(matches!(cmd.encoding_policy(exact), Err(CandidateError::Arguments)));

    let native = Int8MemoryRequirement::for_context(cmd.host.context_tokens).unwrap();
    let result = cmd.host.max_result_bytes as u64;
    let edit = cmd.edit_reserve_mib * MIB;
    // Independent owner inventory: model/metadata/load slack, two preparation
    // owners, native state, final output, cleanup/load buffer, captured source
    // and secret metadata, plus the overlapping intermediate NER/edit owner.
    let expected = limits.weight_bytes + 128 * MIB + 64 * MIB
        + 2 * limits.preparation_bytes + native.kv_bytes + native.rope_bytes
        + native.scratch_payload_bound + 64 * MIB + result * 4 + 4096
        + 65_536 + 64 * 1024 + cmd.host.max_input_bytes as u64 + 1 + 4096
        + result * 4 + cmd.host.max_new_tokens as u64 * 8 + edit;
    assert_eq!(cmd.compact_peak_bytes(limits, edit).unwrap(), expected);
    cmd.host.memory_mib = expected.div_ceil(MIB) - 1;
    assert!(cmd.host.common_with_encoding(cmd.input.clone(), &cmd.encoding).is_ok());
    assert!(matches!(cmd.validate(), Err(CandidateError::Arguments)));
    cmd.host.memory_mib += 1;
    assert!(cmd.validate().is_ok());
}

#[test]
fn compact_redaction_refuses_unadmitted_composition_and_limits_before_private_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("result write") }
        fn flush(&mut self) -> io::Result<()> { panic!("result flush") }
    }
    for mode in 0..6 {
        let mut cmd = command(&["private.txt", "--input-encoding", "safe-bpe-v1",
            "--action", "pseudonymize", "--key-stdin", "--key-id", "rotation", "--namespace", "scope"]);
        cmd.host.memory_mib = 16_384;
        match mode {
            0 => { cmd.long.chunked = true; cmd.corpus.ndjson = true; }
            1 => cmd.corpus.ndjson = true,
            2 => cmd.retention.store_results = true,
            3 => cmd.host.preparation_mib = 512,
            4 => cmd.host.max_result_bytes = (1 << 20) + 1,
            5 => cmd.host.memory_mib = 1024,
            _ => unreachable!(),
        }
        assert_eq!(cmd.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
    for flag in ["--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
        let matches = super::super::definition().try_get_matches_from(
            ["candidate", "redact", "--model", "local.fnlpq", "--memory-mib", "16384", flag, "1"]).unwrap();
        assert!(CandidateCommand::from_matches(&matches).is_err());
        let mut cmd = command(&["--input-encoding", "safe-bpe-v1", flag, "0"]);
        cmd.host.memory_mib = 16_384;
        assert!(matches!(cmd.validate(), Err(CandidateError::Arguments)));
    }
}

#[test]
fn explicit_byte_redaction_preserves_default_admission_and_has_no_encoding_allowance() {
    let default = command(&[]);
    let explicit = command(&["--input-encoding", "byte-fallback"]);
    let (common, limits) = default.validate().unwrap();
    let (explicit_common, explicit_limits) = explicit.validate().unwrap();
    assert_eq!(limits, explicit_limits);
    assert_eq!(common.input_encoding, explicit_common.input_encoding);
    assert_eq!(default.host.preparation_mib, 512);
    assert_eq!(explicit.host.preparation_mib, 512);
    assert_eq!(explicit.encoding_policy(explicit_limits).unwrap(), (UntrustedEncoding::ByteFallback, None));
}

fn scheduled(extra: &[&str]) -> RedactCommand {
    let mut command = command(extra);
    command.host.memory_mib = 16_384;
    command
}

#[test]
fn one_shot_redaction_schedules_compose_without_changing_semantic_or_encoding_authority() {
    for chunked in [false, true] {
        for compact in [false, true] {
            let mut base = Vec::new();
            if chunked { base.push("--chunked"); }
            if compact { base.extend(["--input-encoding", "safe-bpe-v1"]); }
            let ordinary = scheduled(&base);
            let (_, ordinary_limits) = ordinary.validate().unwrap();
            let policy = ordinary.encoding_policy(ordinary_limits).unwrap();
            for extra in [
                vec!["--compute-threads", "4"],
                vec!["--prefill-rows", "7"],
                vec!["--compute-threads", "4", "--prefill-rows", "7"],
            ] {
                let mut arguments = base.clone(); arguments.extend(extra);
                let command = scheduled(&arguments);
                let (common, limits) = command.validate().unwrap();
                let mut expected = ordinary_limits;
                expected.compute = command.compute.admit_prefill(command.prefill.limits().unwrap()).unwrap();
                assert_eq!(limits, expected);
                assert_eq!(common.compute, command.compute);
                assert!(common.policy.prefill().unwrap().is_none(), "source scheduling has no sampler policy");
                assert_eq!(command.encoding_policy(limits).unwrap(), policy);
                assert_eq!(serde_json::to_vec(&command.request()).unwrap(),
                    serde_json::to_vec(&ordinary.request()).unwrap());
                assert_eq!(command.host.task_budget(limits), ordinary.host.task_budget(ordinary_limits));
            }
        }
    }
}

#[test]
fn redaction_worker_and_prompt_storage_coexist_with_both_preparation_owners() {
    use crate::native_engine::strict_int8::{prefill::Int8PrefillLimits,
        projection::int8_team_workspace_bytes_with_prefill};
    for chunked in [false, true] {
        let mut arguments = vec!["--compute-threads", "4", "--prefill-rows", "7"];
        if chunked { arguments.push("--chunked"); }
        let mut command = scheduled(&arguments);
        let (_, limits) = command.validate().unwrap();
        let native = Int8MemoryRequirement::for_context(command.host.context_tokens).unwrap();
        let worker_bytes = int8_team_workspace_bytes_with_prefill(4, 7).unwrap();
        let prompt_bytes = Int8PrefillLimits::required_extra_scratch_bytes(7).unwrap();
        let result = command.host.max_result_bytes as u64;
        let edit = command.edit_reserve_mib * MIB;
        let temporary = if chunked {
            let config = command.long.config(&command.host, limits, NerOptions::default()).unwrap();
            let count = config.mapping.chunks.max_chunks as u64;
            4 * (config.per_chunk.max_output_bytes * count
                + config.mapping.reduction.max_live_value_bytes as u64)
                + 8 * u64::from(config.per_chunk.max_output_tokens) * count + edit
        } else { 4 * result + 8 * command.host.max_new_tokens as u64 + edit };
        let expected = limits.weight_bytes + 192 * MIB + 2 * limits.preparation_bytes
            + native.kv_bytes + native.rope_bytes + native.scratch_payload_bound + 64 * MIB
            + worker_bytes + prompt_bytes + 4 * result + 4096 + 65_536 + 64 * 1024
            + command.host.max_input_bytes as u64 + 1 + 4096 + temporary;
        assert_eq!(limits.compute.workspace_bytes, worker_bytes);
        assert_eq!(command.compact_peak_bytes(limits, edit).unwrap(), expected);
        command.host.memory_mib = expected.div_ceil(MIB) - 1;
        assert!(matches!(command.validate(), Err(CandidateError::Arguments)));
        command.host.memory_mib += 1;
        assert!(command.validate().is_ok());
    }
}

#[test]
fn invalid_redaction_schedules_refuse_before_private_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input"); } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("output"); }
        fn flush(&mut self) -> io::Result<()> { panic!("flush"); }
    }
    for extra in [
        vec!["--compute-threads", "0"], vec!["--compute-threads", "65"],
        vec!["--prefill-rows", "0"], vec!["--prefill-rows", "65"],
    ] {
        let command = scheduled(&extra);
        assert!(matches!(command.validate(), Err(CandidateError::Arguments)));
        assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
    let mut command = scheduled(&["private.txt", "--compute-threads", "64", "--prefill-rows", "64",
        "--action", "pseudonymize", "--key-stdin", "--key-id", "k", "--namespace", "scope"]);
    command.host.memory_mib = 1024;
    assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
}

#[test]
fn explicit_serial_redaction_preserves_existing_defaults() {
    let default = command(&[]);
    let explicit = command(&["--compute-threads", "1"]);
    assert!(default.prefill.limits().unwrap().is_none());
    assert_eq!(default.validate().unwrap().1, explicit.validate().unwrap().1);
    for value in ["-1", "4.0", "NaN", "9999999999999999999999999999999"] {
        for flag in ["--compute-threads", "--prefill-rows"] {
            assert!(super::super::definition().try_get_matches_from([
                "candidate", "redact", "--model", "local.fnlpq", "--memory-mib", "16384", flag, value,
            ]).is_err());
        }
    }
}
