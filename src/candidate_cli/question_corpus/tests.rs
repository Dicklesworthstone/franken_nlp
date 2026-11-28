//! Early admission tests never open a model, options, input or output stream.
use super::*;
use std::io;

fn parsed(extra: &[&str]) -> Result<QuestionCorpusCommand, clap::Error> {
    let mut argv = vec!["answer-corpus", "--model", "private-never-opened.fnlpq", "--memory-mib", "65536"];
    argv.extend_from_slice(extra);
    QuestionCorpusCommand::from_matches(&definition().try_get_matches_from(argv)?)
}
fn command(extra: &[&str]) -> QuestionCorpusCommand { parsed(extra).unwrap() }
struct NoIo;
impl Read for NoIo {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("refused QA population read input"); }
}
impl Write for NoIo {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("refused QA population wrote output"); }
    fn flush(&mut self) -> io::Result<()> { panic!("refused QA population flushed output"); }
}

#[test]
fn root_routes_owned_question_population_and_excludes_unimplemented_semantics() {
    let matches = crate::candidate_cli::definition().try_get_matches_from([
        "candidate", "answer-corpus", "--model", "never-opened", "--memory-mib", "65536",
    ]).unwrap();
    assert!(matches!(CandidateCommand::from_matches(&matches).unwrap(), CandidateCommand::QuestionCorpus(_)));
    for extra in [vec!["--question", "fixed"], vec!["--synthesize-answer"], vec!["--seed", "1"],
        vec!["--reuse-prefill"], vec!["--max-selected-rows", "128"], vec!["--store-results"]] {
        assert!(parsed(&extra).is_err());
    }
    for argv in [vec!["answer-corpus", "--model", "missing-memory"],
        vec!["answer-corpus", "--memory-mib", "65536"]] {
        assert!(definition().try_get_matches_from(argv).is_err());
    }
}

#[test]
fn complete_document_work_and_question_bounds_do_not_replace_population_authority() {
    let base = command(&[]).admit().unwrap();
    let changed = command(&["--max-document-forward-positions", "32768",
        "--max-document-projected-logits", "123456", "--max-document-attention-pairs", "234567",
        "--max-document-dot-products", "345678", "--max-document-multiply-accumulates", "456789",
        "--max-document-mask-node-visits", "2000000000", "--max-question-bytes", "8192"])
        .admit().unwrap();
    assert_eq!(changed.config.max_model_work, base.config.max_model_work);
    assert_eq!(changed.config.max_mask_visits, base.config.max_mask_visits);
    assert_eq!(changed.config.max_verification, base.config.max_verification);
    assert_eq!(changed.transport.max_work, base.transport.max_work);
    assert_eq!(changed.config.budget, base.config.budget);
    assert_eq!(changed.config.options, base.config.options);
    assert_eq!(changed.config.max_question_bytes, 8192);
    assert_eq!(changed.config.mapping.max_mask_visits, 2_000_000_000);
    let w = changed.config.mapping.max_model_work;
    assert_eq!((w.forward_positions, w.projected_logits, w.attention_pairs), (32768, 123456, 234567));
    assert_eq!((w.projections.dot_products, w.projections.multiply_accumulates), (345678, 456789));
}

#[test]
fn invalid_limits_and_document_run_widening_refuse_before_any_io() {
    for args in [vec!["--max-question-bytes", "0"], vec!["--max-chunks", "0"],
        vec!["--max-requests", "0"], vec!["--max-line-bytes", "1024"],
        vec!["--max-document-forward-positions", "0"], vec!["--max-document-projected-logits", "0"],
        vec!["--max-document-attention-pairs", "0"], vec!["--max-document-dot-products", "0"],
        vec!["--max-document-multiply-accumulates", "0"], vec!["--max-document-mask-node-visits", "1"],
        vec!["--max-forward-positions", "1"], vec!["--max-projected-logits", "1"],
        vec!["--max-attention-pairs", "1"], vec!["--max-dot-products", "1"],
        vec!["--max-multiply-accumulates", "1"], vec!["--max-run-mask-node-visits", "1"],
        vec!["--max-run-qa-citations", "1"], vec!["--max-run-qa-evidence-spans", "1"],
        vec!["--max-run-qa-scan-steps", "1"], vec!["--options", "-"],
        vec!["--max-total-output-bytes", "1"]] {
        let mut diagnostics = Vec::new();
        assert_eq!(command(&args).run_owned(NoIo, NoIo, &mut diagnostics), ErrorCode::Usage.as_process_exit(), "{args:?}");
        let text = String::from_utf8(diagnostics).unwrap();
        assert!(text.contains("int8_question_corpus_options")); assert!(!text.contains("private-never-opened"));
    }
}

#[test]
fn options_are_complete_strict_and_cannot_override_record_question_or_budgets() {
    assert_eq!(parse_options(None).unwrap(), AnswerOptions::default());
    let valid = r#"{"max_answer_scalars":128,"max_citations":2,"max_quote_scalars":64}"#;
    let selected = parse_options(Some(valid)).unwrap();
    let admitted = command(&[]).admit_options(selected).unwrap();
    assert_eq!(admitted.config.options, selected);
    for invalid in ["{}", "null", "[]", r#"{"max_answer_scalars":128}"#,
        r#"{"max_answer_scalars":128,"max_citations":2,"max_quote_scalars":64,"question":"private"}"#,
        r#"{"max_answer_scalars":128,"max_citations":2,"max_citations":3,"max_quote_scalars":64}"#,
        r#"{"max_answer_scalars":128,"max_citations":0,"max_quote_scalars":64}"#] {
        assert!(parse_options(Some(invalid)).is_err());
    }
}

#[test]
fn framing_original_question_and_complete_frontier_are_in_the_declared_peak() {
    let declared = command(&["--max-requests", "30"]).admit().unwrap();
    assert_eq!(declared.transport.max_output_bytes + 34 * FRAME_ALLOWANCE, declared.output_bytes);
    assert_eq!(declared.transport.max_output_line_bytes,
        declared.config.mapping.reduction.max_result_bytes + FRAME_ALLOWANCE as usize);
    assert_eq!(declared.transport.max_work.forward_positions, declared.config.max_model_work.forward_positions);
    assert_eq!(declared.transport.max_work.projected_logits, declared.config.max_model_work.projected_logits);
    assert_eq!(declared.memory.max_output_tokens, 0);
    let peak = validate_peak(&declared).unwrap();
    let mut refused = command(&[]);
    refused.document.host.memory_mib = (peak - 1) / MIB;
    assert!(refused.admit().is_err());
    let smaller = command(&["--max-question-bytes", "1024"]).admit().unwrap();
    assert!(smaller.memory.input_bytes < declared.memory.input_bytes);
    assert_eq!(smaller.memory.document_workspace_bytes, declared.memory.document_workspace_bytes);
}

#[test]
fn compact_encoding_shares_one_original_document_allowance_and_preserves_whole_input_quote() {
    let byte = command(&[]).admit().unwrap();
    let compact = command(&["--input-encoding", "safe-bpe-v1",
        "--max-total-tokenizer-input-bytes", "134217728",
        "--max-total-tokenizer-output-tokens", "8388608"])
        .admit().unwrap();
    assert!(matches!(compact.document.encoding, UntrustedEncoding::SafeBpeV1(_)));
    let total = compact.document.encoding_work.unwrap();
    assert_eq!(total.max_input_bytes, 134_217_728);
    assert_eq!(total.max_output_tokens, 8_388_608);
    assert_eq!(compact.config.budget, byte.config.budget);
    assert_eq!(compact.config.mapping.chunks, byte.config.mapping.chunks);
    assert_eq!(compact.memory, byte.memory);
    assert!(command(&["--max-total-tokenizer-input-bytes", "1"]).admit().is_err());
    assert!(command(&["--input-encoding", "safe-bpe-v1", "--max-total-tokenizer-input-bytes", "1"])
        .admit().is_err());
    let help = definition().render_long_help().to_string();
    assert!(help.contains("ONE complete original source/question"));
    assert!(!help.contains("One cumulative tokenizer allowance across a whole invocation"));
}

#[test]
fn prefill_and_persistent_team_compose_without_renewing_document_or_run_work() {
    let base = command(&[]).admit().unwrap();
    for width in ["1", "2", "4", "64"] {
        let scheduled = command(&["--prefill-rows", "4", "--compute-threads", width]).admit().unwrap();
        assert_eq!(scheduled.config.budget, base.config.budget);
        assert_eq!(scheduled.config.options, base.config.options);
        assert_eq!(scheduled.config.mapping.max_model_work, base.config.mapping.max_model_work);
        assert_eq!(scheduled.config.max_model_work, base.config.max_model_work);
        assert_eq!(scheduled.config.max_mask_visits, base.config.max_mask_visits);
        assert_eq!(scheduled.config.max_verification, base.config.max_verification);
        assert_eq!(scheduled.memory, base.memory);
        assert!(validate_peak(&scheduled).unwrap() > validate_peak(&base).unwrap());
    }
    for width in ["0", "65"] { assert!(command(&["--compute-threads", width]).admit().is_err()); }
}

#[test]
fn shared_policy_accepts_immutable_durable_lifetime_beyond_resident_forward_limit() {
    let command = command(&[]); let document = command.document.admit(PathBuf::from("-")).unwrap();
    let base = command.admit().unwrap(); let mut lifetime = base.config.max_model_work;
    lifetime.forward_positions = 16 * 1024 * 1024 + 1;
    let config = command.document.configuration(&document, AnswerOptions::default(), lifetime,
        base.config.max_mask_visits, base.config.max_verification).unwrap();
    assert_eq!(config.max_model_work, lifetime);
    assert!(super::tests::command(&["--max-forward-positions", "16777217"]).admit().is_err());
}
