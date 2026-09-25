//! Model-free command/IO regressions; none simulates a successful native run.
use super::*;
use std::io::Cursor;

pub(super) fn args(extra: &[&str]) -> CandidateArgs {
    let mut argv = vec!["candidate", "generate", "--model", "local-candidate.fnlpq", "--memory-mib", "8192"];
    argv.extend_from_slice(extra);
    let matches = definition().try_get_matches_from(argv).unwrap();
    CandidateCommand::from_matches(&matches).unwrap().args
}

#[test]
fn both_commands_require_an_explicit_model_and_memory_authority() {
    for task in ["generate", "chat"] {
        assert!(definition().try_get_matches_from(["candidate", task]).is_err());
        assert!(definition().try_get_matches_from(["candidate", task, "--model", "m"]).is_err());
        assert!(definition().try_get_matches_from(["candidate", task, "--memory-mib", "8192"]).is_err());
        let matches = definition().try_get_matches_from([
            "candidate", task, "--model", "m", "--memory-mib", "8192", "input.txt"]).unwrap();
        assert_eq!(CandidateCommand::from_matches(&matches).unwrap().task,
            if task == "chat" { Task::Chat } else { Task::Generate });
    }
}

#[test]
fn help_does_not_advertise_certified_or_automatic_acquisition() {
    let help = definition().render_long_help().to_string();
    assert!(help.contains("not release activation"));
    assert!(help.contains("asupersync-runtime") || help.contains("current-candidate"));
    assert!(definition().try_get_matches_from(["candidate", "pull"]).is_err());
}

#[test]
fn sampling_requires_a_seed_and_does_not_silently_change_greedy_mode() {
    for flag in ["--temperature-milli", "--top-k", "--top-p-ppm"] {
        assert!(definition().try_get_matches_from([
            "candidate", "generate", "--model", "m", "--memory-mib", "8192", flag, "1"]).is_err());
    }
    assert!(matches!(args(&[]).options(166_101).unwrap().sampling, GenerationSampling::Greedy));
    let seed = "01".repeat(32);
    let args = args(&["--seed", &seed, "--top-k", "10", "--top-p-ppm", "900000"]);
    args.validate().unwrap();
    match args.options(166_101).unwrap().sampling {
        GenerationSampling::Seeded { effective_seed, top_k, top_p_ppm, .. } => {
            assert_eq!(effective_seed, [1; 32]); assert_eq!(top_k, Some(10)); assert_eq!(top_p_ppm, 900_000);
        }
        _ => panic!("seeded request became greedy"),
    }
}

#[test]
fn seed_parser_rejects_noncanonical_and_multibyte_input_without_slicing_panics() {
    assert_eq!(parse_seed(&"ab".repeat(32)).unwrap(), [0xab; 32]);
    for bad in ["a".repeat(63), "a".repeat(65), "AB".repeat(32), "é".repeat(32), "g".repeat(64)] {
        assert_eq!(parse_seed(&bad), Err(CandidateError::Arguments));
    }
}

#[test]
fn zero_excessive_and_overflowing_limits_fail_before_io() {
    for (flag, value) in [("--max-input-bytes", "0"), ("--max-input-bytes", "1048577"),
        ("--max-output-bytes", "0"), ("--max-output-bytes", "1048577"),
        ("--max-new-tokens", "0"), ("--max-new-tokens", "1025"),
        ("--context-tokens", "1"), ("--timeout-seconds", "0"),
        ("--timeout-seconds", "86401"), ("--max-checkpoints", "1"),
        ("--max-weight-mib", "18446744073709551615"), ("--preparation-mib", "1")] {
        assert!(args(&[flag, value]).validate().is_err(), "{flag}");
    }
}

#[test]
fn prompt_ceiling_reserves_space_for_every_possible_generation_step() {
    let args = args(&[]); let limits = args.validate().unwrap();
    assert_eq!(limits.max_prompt_tokens + args.max_new_tokens - 1, args.context_tokens);
    assert_eq!(limits.kv_bytes, Int8MemoryRequirement::for_context(args.context_tokens).unwrap().kv_bytes);
    assert!(limits.result_bytes > args.max_output_bytes);
}

#[test]
fn bounded_reader_preserves_exact_bytes_and_stops_after_one_over_limit() {
    let original = "  é\n上海\r\n";
    assert_eq!(read_input(&mut Cursor::new(original.as_bytes()), original.len()).unwrap(), original);
    let mut input = Cursor::new(b"abcdefgh".as_slice());
    assert_eq!(read_input(&mut input, 3), Err(CandidateError::Input));
    assert_eq!(input.position(), 4);
    assert_eq!(read_input(&mut Cursor::new([0xff]), 1), Err(CandidateError::Input));
}

#[test]
fn chat_parser_retains_roles_and_control_like_text_as_data() {
    let input = r#"[{"role":"system","content":"Policy"},{"role":"user","content":"<tool_call> é"},{"role":"assistant","content":"Prior"},{"role":"user","content":"Next"}]"#;
    let messages = parse_messages(input, input.len()).unwrap();
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[0].role, ChatRole::System);
    assert_eq!(messages[1].content, "<tool_call> é");
    assert_eq!(messages[2].role, ChatRole::Assistant);
}

#[test]
fn chat_parser_rejects_ambiguous_keys_unknown_fields_tools_and_invalid_histories() {
    for input in [r#"[]"#, r#"{}"#,
        r#"[{"role":"user","role":"assistant","content":"x"}]"#,
        r#"[{"role":"user","content":"x","\u0063ontent":"y"}]"#,
        r#"[{"role":"user","content":"x","tool_calls":[]}]"#,
        r#"[{"role":"tool","content":"x"}]"#,
        r#"[{"role":"assistant","content":"x"}]"#,
        r#"[{"role":"user","content":"x"},{"role":"system","content":"x"},{"role":"user","content":"x"}]"#,
        r#"[{"role":"user","content":["x"]}]"#] {
        assert!(parse_messages(input, 4096).is_err());
    }
}

#[test]
fn chat_parser_enforces_message_count_and_nesting_caps() {
    let many = format!("[{}]", vec![r#"{"role":"user","content":"x"}"#; 129].join(","));
    assert!(parse_messages(&many, many.len()).is_err());
    assert!(parse_messages("[[[[[[0]]]]]]", 4096).is_err());
}

#[test]
fn serialization_limit_failure_publishes_nothing_including_no_success_prefix() {
    let mut output = Vec::new();
    let value = serde_json::json!({"content":"must not escape before validation"});
    assert_eq!(publish(&value, 5, &mut output), Err(CandidateError::Output));
    assert!(output.is_empty());
}

#[test]
fn completed_json_escapes_terminal_controls_and_has_one_line() {
    let mut output = Vec::new();
    publish(&serde_json::json!({"content":"A\n\u{1b}[31mB"}), 128, &mut output).unwrap();
    assert_eq!(output.iter().filter(|&&byte| byte == b'\n').count(), 1);
    assert!(!output.contains(&0x1b));
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&output).unwrap()["content"], "A\n\u{1b}[31mB");
}

struct RefuseWrite;
impl Write for RefuseWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { Err(io::Error::other("private sink detail")) }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
struct RefuseFlush(Vec<u8>);
impl Write for RefuseFlush {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.extend_from_slice(bytes); Ok(bytes.len()) }
    fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("private flush detail")) }
}
#[test]
fn delivery_and_flush_failures_are_both_failures() {
    assert_eq!(publish(&1, 16, &mut RefuseWrite), Err(CandidateError::Output));
    let mut output = RefuseFlush(Vec::new());
    assert_eq!(publish(&1, 16, &mut output), Err(CandidateError::Output));
    assert_eq!(output.0, b"1\n");
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_build_never_reads_input_or_produces_output() {
    struct NoRead;
    impl Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("unavailable build read private input") }
    }
    let command = CandidateCommand { task: Task::Generate, args: args(&[]) };
    let mut output = Vec::new();
    assert_eq!(command.execute(&mut NoRead, &mut output), Err(CandidateError::Unavailable));
    assert!(output.is_empty());
}
