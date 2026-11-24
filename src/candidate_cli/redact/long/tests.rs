//! Model-free CLI policy/admission regressions, never model-success fixtures.
use super::*;
use crate::candidate_cli::redact::tests::command;

#[test]
fn chunked_mode_is_explicit_and_preserves_existing_actions_and_verification() {
    let plain = command(&[]); plain.validate().unwrap(); assert!(!plain.long.chunked);
    let cmd = command(&["--chunked"]); let (_, limits) = cmd.validate().unwrap();
    assert!(cmd.long.chunked); assert!(cmd.request().verify);
    assert_eq!(cmd.request().actions, plain.request().actions);
    let config = cmd.long.config(&cmd.host, limits, NerOptions::default()).unwrap();
    assert_eq!(config.per_chunk.max_output_tokens, cmd.host.max_new_tokens as u32);
    assert_eq!(config.mapping.chunks.max_chunks, 64);
    assert_eq!(config.mapping.chunks.max_input_bytes, cmd.request().rule_budget.max_input_bytes);
    assert_eq!(config.max_result_bytes, cmd.host.max_result_bytes as u64);
    assert!(!command(&["--chunked", "--no-verify"]).request().verify);
}
#[test]
fn long_only_options_require_the_explicit_mode_in_complete_valid_commands() {
    for flag in ["--max-ner-chunks", "--max-ner-chunk-bytes", "--max-ner-tokenizer-calls", "--max-ner-map-bytes",
        "--max-total-mask-node-visits", "--max-forward-positions", "--max-projected-logits",
        "--max-attention-pairs", "--max-dot-products", "--max-multiply-accumulates",
        "--max-total-tokenizer-input-bytes", "--max-total-tokenizer-output-tokens",
        "--max-total-tokenizer-pair-checks", "--max-total-tokenizer-pair-bytes"] {
        assert!(crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "redact", "--model", "local.fnlpq", "--memory-mib", "8192", flag, "1",
        ]).is_err(), "{flag}");
    }
}
#[test]
fn every_native_axis_and_partition_envelope_has_finite_limits() {
    for (flag, value) in [("--max-ner-chunks", "0"), ("--max-ner-chunks", "257"),
        ("--max-ner-chunk-bytes", "3"), ("--max-ner-tokenizer-calls", "0"),
        ("--max-ner-map-bytes", "0"), ("--max-ner-map-bytes", "16777217"),
        ("--max-total-mask-node-visits", "1"), ("--max-forward-positions", "0"),
        ("--max-projected-logits", "0"), ("--max-attention-pairs", "0"),
        ("--max-dot-products", "0"), ("--max-multiply-accumulates", "0")] {
        assert!(command(&["--chunked", flag, value]).validate().is_err(), "{flag}");
    }
    assert!(command(&["--chunked", "--max-ner-chunks", "256"]).validate().is_err());
    command(&["--chunked", "--max-ner-chunks", "256", "--preparation-mib", "1024"]).validate().unwrap();
}
#[test]
fn increasing_chunks_never_multiplies_whole_operation_work_or_mask_authority() {
    let a = command(&["--chunked"]);
    let b = command(&["--chunked", "--max-ner-chunks", "256", "--preparation-mib", "1024"]);
    let (_, la) = a.validate().unwrap(); let (_, lb) = b.validate().unwrap();
    let ca = a.long.config(&a.host, la, NerOptions::default()).unwrap();
    let cb = b.long.config(&b.host, lb, NerOptions::default()).unwrap();
    assert_eq!(ca.mapping.max_model_work, cb.mapping.max_model_work);
    assert_eq!(ca.mapping.max_mask_visits, cb.mapping.max_mask_visits);
    assert_eq!(ca.mapping.mask_visits_per_chunk, cb.mapping.mask_visits_per_chunk);
    assert_eq!(ca.mapping.reduction.max_result_bytes, cb.mapping.reduction.max_result_bytes);
    assert_eq!(ca.mapping.reduction.max_live_value_bytes, 2 * ca.mapping.reduction.max_value_bytes);
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_feature_refuses_before_source_options_secret_or_output_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
        fn flush(&mut self) -> io::Result<()> { panic!("flush") }
    }
    let cmd = command(&["--chunked", "--ner-options", "never-open.json"]);
    assert_eq!(cmd.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
}


fn compact(extra: &[&str]) -> RedactCommand {
    let mut args = vec!["--chunked", "--input-encoding", "safe-bpe-v1"];
    args.extend_from_slice(extra);
    let mut command = command(&args);
    command.host.memory_mib = 16_384;
    command
}

#[test]
fn compact_document_admits_oversized_probes_and_one_whole_operation_budget() {
    let command = compact(&["--max-ner-chunk-bytes", "8192"]);
    let (common, limits) = command.validate().unwrap();
    let (policy, complete) = command.encoding_policy(limits).unwrap();
    let UntrustedEncoding::SafeBpeV1(call) = policy else { panic!("compact policy required"); };
    assert_eq!(call.max_input_bytes, 8192);
    assert_eq!(call.max_output_tokens, 8192);
    assert!(call.max_output_tokens > limits.max_prompt_tokens);
    assert_eq!(common.max_input_bytes, command.host.max_input_bytes);
    let bytes = 8 * (command.host.max_input_bytes as u64 + command.host.max_result_bytes as u64);
    assert_eq!(complete.unwrap(), SafeBpeWorkLimits {
        max_input_bytes: bytes, max_output_tokens: bytes,
        max_pair_checks: 268_435_456, max_pair_bytes: 4_294_967_296,
    });
    let config = command.long.config(&command.host, limits, NerOptions::default()).unwrap();
    assert_eq!(config.mapping.chunks.context_tokens, command.host.context_tokens);
    assert_eq!(config.per_chunk, command.host.task_budget(limits));
    assert_eq!(config.mapping.chunks.max_chunk_bytes, 8192);
}

#[test]
fn compact_document_total_limits_are_explicit_and_never_multiplied_by_chunk_count() {
    let flags = ["--max-tokenizer-pair-checks", "1234", "--max-tokenizer-pair-bytes", "76543",
        "--max-total-tokenizer-input-bytes", "1000000", "--max-total-tokenizer-output-tokens", "900000",
        "--max-total-tokenizer-pair-checks", "777777", "--max-total-tokenizer-pair-bytes", "8888888"];
    let a = compact(&flags); let (_, limits) = a.validate().unwrap();
    let mut b = compact(&flags);
    b.long.max_ner_chunks = Some(128);
    b.no_verify = true;
    let (_, other) = b.validate().unwrap();
    let (policy, complete) = a.encoding_policy(limits).unwrap();
    assert_eq!(b.encoding_policy(other).unwrap(), (policy.clone(), complete));
    let UntrustedEncoding::SafeBpeV1(call) = policy else { panic!("compact policy required"); };
    assert_eq!(call.max_pair_checks, 1234);
    assert_eq!(call.max_pair_bytes, 76543);
    assert_eq!(complete.unwrap(), SafeBpeWorkLimits {
        max_input_bytes: 1_000_000, max_output_tokens: 900_000,
        max_pair_checks: 777_777, max_pair_bytes: 8_888_888,
    });
}

#[test]
fn compact_document_workspace_and_map_frontier_have_exact_admission_boundaries() {
    let mut command = compact(&["--max-ner-chunk-bytes", "8192"]);
    let (_, limits) = command.validate().unwrap();
    let (policy, _) = command.encoding_policy(limits).unwrap();
    let UntrustedEncoding::SafeBpeV1(call) = policy else { panic!("compact policy required"); };
    let floor = command.long.preparation_floor(&command.host).unwrap()
        + call.requirements().unwrap().workspace_bytes + 1024;
    let mut exact = limits;
    exact.preparation_bytes = floor;
    assert!(command.encoding_policy(exact).is_ok());
    exact.preparation_bytes -= 1;
    assert!(matches!(command.encoding_policy(exact), Err(CandidateError::Arguments)));

    let config = command.long.config(&command.host, limits, NerOptions::default()).unwrap();
    let chunks = config.mapping.chunks.max_chunks as u64;
    let edit = command.edit_reserve_mib * MIB;
    let temporary = 4 * (config.per_chunk.max_output_bytes * chunks
        + config.mapping.reduction.max_live_value_bytes as u64)
        + 8 * u64::from(config.per_chunk.max_output_tokens) * chunks + edit;
    assert_eq!(command.long.temporary_bytes(&command.host, edit).unwrap(), temporary);
    let native = Int8MemoryRequirement::for_context(command.host.context_tokens).unwrap();
    let expected = limits.weight_bytes + 128 * MIB + 64 * MIB
        + 2 * limits.preparation_bytes + native.kv_bytes + native.rope_bytes
        + native.scratch_payload_bound + 64 * MIB
        + command.host.max_result_bytes as u64 * 4 + 4096 + 65_536 + 64 * 1024
        + command.host.max_input_bytes as u64 + 1 + 4096 + temporary;
    assert_eq!(command.compact_peak_bytes(limits, edit).unwrap(), expected);
    command.host.memory_mib = expected.div_ceil(MIB) - 1;
    assert!(command.host.common(command.input.clone()).is_ok());
    assert!(matches!(command.validate(), Err(CandidateError::Arguments)));
    command.host.memory_mib += 1;
    assert!(command.validate().is_ok());
}

#[test]
fn complete_tokenizer_flags_require_compact_mode_and_refuse_invalid_caps_before_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("result write") }
        fn flush(&mut self) -> io::Result<()> { panic!("result flush") }
    }
    for flag in ["--max-total-tokenizer-input-bytes", "--max-total-tokenizer-output-tokens",
        "--max-total-tokenizer-pair-checks", "--max-total-tokenizer-pair-bytes"] {
        let byte = command(&["--chunked", flag, "1"]);
        assert_eq!(byte.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        for bad in ["0", "18446744073709551615"] {
            let invalid = compact(&[flag, bad]);
            assert_eq!(invalid.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        }
    }
    let byte = command(&["--chunked"]);
    let (_, limits) = byte.validate().unwrap();
    assert_eq!(byte.encoding_policy(limits).unwrap(), (UntrustedEncoding::ByteFallback, None));
    let mut compact = compact(&[]);
    compact.long.chunked = false;
    compact.corpus.ndjson = true;
    assert!(matches!(compact.validate(), Err(CandidateError::Arguments)));
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn compact_document_disabled_runtime_keeps_private_io_unopened() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("result write") }
        fn flush(&mut self) -> io::Result<()> { panic!("result flush") }
    }
    let command = compact(&["private.txt", "--action", "pseudonymize", "--key-stdin",
        "--key-id", "rotation", "--namespace", "scope", "--ner-options", "private-options.json"]);
    assert_eq!(command.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
}
