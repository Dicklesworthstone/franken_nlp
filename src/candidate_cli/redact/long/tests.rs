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
        "--max-attention-pairs", "--max-dot-products", "--max-multiply-accumulates"] {
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
