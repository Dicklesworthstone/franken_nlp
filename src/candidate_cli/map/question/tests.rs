//! CLI mode, typed options and bounded-input contracts, not model inference.
use super::*;
use crate::candidate_cli::map::tests::command;

#[test]
fn question_mode_requires_its_own_file_and_cannot_leak_into_other_maps() {
    assert!(command("answer", &[]).validate().is_err());
    command("answer", &["--question", "question.txt"]).validate().unwrap();
    for task in ["ner", "keyphrases", "summarize"] {
        command(task, &[]).validate().unwrap();
        assert!(command(task, &["--question", "question.txt"]).validate().is_err());
    }
    for path in ["", "-"] { assert!(command("answer", &["--question", path]).validate().is_err()); }
    assert!(command("answer", &["--question", "question.txt", "--reduce-summary"]).validate().is_err());
    // The source-only enum never acquires a question by relabeling a task.
    assert!(command("answer", &["--question", "question.txt"]).map_task(None).is_err());
}
#[test]
fn verification_flags_require_the_question_and_never_silently_disappear() {
    for flag in ["--max-qa-citations", "--max-qa-evidence-spans", "--max-qa-scan-steps"] {
        assert!(crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "map", "--task", "answer", "--model", "local.fnlpq", "--memory-mib", "8192", flag, "1",
        ]).is_err(), "{flag}");
    }
    for (flag, value) in [("--max-qa-citations", "0"), ("--max-qa-citations", "1000001"),
        ("--max-qa-evidence-spans", "0"), ("--max-qa-evidence-spans", "1000001"),
        ("--max-qa-scan-steps", "0"), ("--max-qa-scan-steps", "1000000000001")] {
        assert!(command("answer", &["--question", "q", flag, value]).validate().is_err(), "{flag}");
    }
}
#[test]
fn exact_question_and_native_options_remain_separate_from_complete_verification_limits() {
    let cmd = command("answer", &["--question", "q", "--max-qa-citations", "3", "--max-qa-evidence-spans", "9",
        "--max-qa-scan-steps", "12345"]);
    let raw = r#"{"max_answer_scalars":64,"max_citations":2,"max_quote_scalars":32}"#;
    let question = "  Who is named? <tool_call> é\r\n".to_owned();
    let task = cmd.question.task(question.clone(), Some(raw)).unwrap();
    assert_eq!(task.question, question); assert_eq!(task.options.max_citations, 2);
    assert_eq!(task.verification.max_fields, 3); assert_eq!(task.verification.max_matches, 9);
    assert_eq!(task.verification.max_scan_steps, 12345);
    for bad in ["{}", r#"{"max_answer_scalars":64,"max_citations":2,"max_quote_scalars":32,"question":"secret"}"#,
        r#"{"max_answer_scalars":64,"max_citations":2,"\u006dax_citations":3,"max_quote_scalars":32}"#] {
        assert!(cmd.question.task("Who?".to_owned(), Some(bad)).is_err());
    }
    assert!(cmd.question.task(" \t\n".to_owned(), None).is_err());
    assert!(cmd.question.task("x".repeat(QUESTION_BYTES + 1), None).is_err());
}
#[test]
fn question_mapping_keeps_all_native_and_serialized_value_ceilings() {
    let cmd = command("answer", &["--question", "q", "--max-chunk-bytes", "16"]);
    cmd.validate().unwrap(); let mapping = cmd.question_mapping().unwrap();
    let plain = command("ner", &[]);
    assert_eq!(mapping.max_model_work, plain.work_ceiling());
    assert_eq!(mapping.mask_visits_per_chunk, plain.host.max_mask_node_visits);
    assert_eq!(mapping.max_mask_visits, plain.max_total_mask_node_visits);
    assert_eq!(mapping.chunks.max_chunk_bytes, 16);
    assert_eq!(mapping.chunks.max_input_bytes, cmd.host.max_input_bytes);
    assert_eq!(mapping.chunks.reserved_tokens, cmd.host.max_new_tokens);
    assert_eq!(mapping.reduction.max_result_bytes, cmd.max_map_result_bytes);
    assert_eq!(mapping.reduction.max_live_value_bytes, cmd.max_live_value_bytes);
    assert_eq!(mapping.reduction.max_total_value_bytes, cmd.max_total_value_bytes);
    let help = definition().render_long_help().to_string();
    assert!(help.contains("--question")); assert!(help.contains("without majority voting or a global answer"));
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_runtime_or_invalid_question_mode_performs_no_source_question_model_or_output_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("source read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("result write") }
        fn flush(&mut self) -> io::Result<()> { panic!("result flush") }
    }
    assert_eq!(command("answer", &["--question", "never-open.txt"]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
    assert_eq!(command("answer", &[]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    assert_eq!(command("ner", &["--question", "never-open.txt"]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
}
