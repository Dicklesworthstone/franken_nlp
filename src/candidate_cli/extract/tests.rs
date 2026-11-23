//! Command and real grammar checks without model weights or fake inference.
use super::*;

pub(in crate::candidate_cli) fn command(extra: &[&str]) -> ExtractCommand {
    let mut argv = vec!["candidate", "extract", "--model", "local.fnlpq", "--memory-mib", "8192", "--schema", "schema.json"];
    argv.extend_from_slice(extra);
    let matches = super::super::definition().try_get_matches_from(argv).unwrap();
    match CandidateCommand::from_matches(&matches).unwrap() {
        CandidateCommand::Extract(command) => command,
        _ => panic!("wrong command route"),
    }
}

#[test]
fn extraction_requires_explicit_schema_model_and_memory() {
    for args in [vec!["candidate", "extract"],
        vec!["candidate", "extract", "--model", "m", "--memory-mib", "8192"],
        vec!["candidate", "extract", "--schema", "s", "--memory-mib", "8192"],
        vec!["candidate", "extract", "--schema", "s", "--model", "m"]] {
        assert!(super::super::definition().try_get_matches_from(args).is_err());
    }
    let command = command(&["document.txt", "--source-membership"]);
    command.validate().unwrap();
    assert_eq!(command.input, std::path::Path::new("document.txt"));
    assert!(command.source_membership);
}

#[test]
fn schema_is_a_local_separate_input_and_sampling_cannot_change_extraction() {
    assert!(check_schema_path(std::path::Path::new("-")).is_err());
    assert!(check_schema_path(std::path::Path::new("")).is_err());
    for option in ["--seed", "--temperature-milli", "--top-k", "--options"] {
        assert!(super::super::definition().try_get_matches_from([
            "candidate", "extract", "--model", "m", "--memory-mib", "8192", "--schema", "s", option, "1"]).is_err());
    }
    let mut command = command(&[]);
    command.schema = PathBuf::from("-");
    assert!(command.validate().is_err());
}

#[test]
fn schema_validation_rejects_duplicates_unsupported_keywords_and_wrong_grounding() {
    let command = command(&[]);
    let (_, limits) = command.validate().unwrap();
    let budget = command.host.task_budget(limits);
    for schema in [r#"{"type":"string","type":"number"}"#,
        r#"{"type":"string","\u0074ype":"number"}"#,
        r#"{"type":"string","pattern":".*"}"#,
        r#"{"$ref":"https://example.invalid/remote"}"#,
        r#"{"type":"string","x-fnlp-source":"verbatim"}"#] {
        let request = arguments(schema.to_owned(), false, budget).unwrap();
        assert_eq!(check_schema(&request, "Alice", &command.host), Err(CandidateError::Planning));
    }
    let request = arguments(r#"{"type":"string"}"#.to_owned(), true, budget).unwrap();
    assert_eq!(check_schema(&request, "Alice", &command.host), Err(CandidateError::Planning));
}

#[test]
fn exact_schema_bytes_and_thirty_eight_digit_decimals_are_not_rounded() {
    let command = command(&[]);
    let (_, limits) = command.validate().unwrap();
    let number = "0.12345678901234567890123456789012345678";
    let schema = format!(" {{\"type\":\"number\",\"const\":{number}}}\n");
    let request = arguments(schema.clone(), false, command.host.task_budget(limits)).unwrap();
    check_schema(&request, "source", &command.host).unwrap();
    assert_eq!(request.schema.as_bytes(), schema.as_bytes());
    let program = JsonProgram::compile(&request.schema, compiler(&command.host)).unwrap();
    program.validate_json(number).unwrap();
    assert!(program.validate_json("0.12345678901234567890123456789012345679").is_err());
}

#[test]
fn source_membership_uses_actual_unicode_source_not_schema_property_names() {
    let command = command(&["--source-membership"]);
    let (_, limits) = command.validate().unwrap();
    let schema = r#"{"type":"string","maxLength":8,"x-fnlp-source":"verbatim"}"#;
    let request = arguments(schema.to_owned(), true, command.host.task_budget(limits)).unwrap();
    let source = "é 上海 é";
    check_schema(&request, source, &command.host).unwrap();
    let program = JsonProgram::compile_with_source(schema, source, compiler(&command.host), command.host.planning().source).unwrap();
    program.validate_json(r#""上海""#).unwrap();
    assert!(program.validate_json(r#""invented""#).is_err());
    assert!(program.validate_json(r#""verbatim""#).is_err());
    assert!(!program.source_fields(r#""é""#).unwrap().is_empty());
}

#[test]
fn empty_oversized_schemas_and_documents_refuse_before_model_access() {
    let command = command(&[]);
    let (_, limits) = command.validate().unwrap();
    let budget = command.host.task_budget(limits);
    assert!(arguments(String::new(), false, budget).is_err());
    assert!(arguments(" ".repeat(SCHEMA_BYTES + 1), false, budget).is_err());
    let request = arguments(r#"{"type":"string"}"#.to_owned(), false, budget).unwrap();
    assert_eq!(check_schema(&request, &"x".repeat(command.host.max_input_bytes + 1), &command.host), Err(CandidateError::Input));
}

#[test]
fn host_budget_reserves_complete_context_and_fixed_schema_cap() {
    let command = command(&[]);
    let (_, limits) = command.validate().unwrap();
    let budget = command.host.task_budget(limits);
    assert_eq!(budget.max_input_tokens as usize + budget.max_output_tokens as usize, command.host.context_tokens);
    assert_eq!(budget.max_kv_bytes, limits.kv_bytes);
    assert_eq!(compiler(&command.host).max_schema_bytes, SCHEMA_BYTES);
    assert_eq!(compiler(&command.host).max_output_bytes, command.host.max_result_bytes);
}

#[test]
fn extraction_workers_and_prompt_morsels_compose_without_changing_shared_task_limits() {
    let serial = command(&[]);
    let (serial_args, serial_limits) = serial.validate().unwrap();
    assert_eq!(serial_args.compute.threads(), 1);
    assert_eq!(serial_limits.compute.workspace_bytes, 0);
    assert!(serial.prefill().unwrap().is_none());
    for width in ["1", "4", "64"] {
        for rows in [None, Some("1"), Some("4"), Some("64")] {
            let mut flags = vec!["--compute-threads", width, "--selected-rows", "31",
                "--source-membership", "--input-encoding", "safe-bpe-v1"];
            if let Some(rows) = rows { flags.extend(["--prefill-rows", rows]); }
            let mut cmd = command(&flags);
            cmd.host.memory_mib = 16_384;
            let (args, limits) = cmd.validate().unwrap();
            let prefill = cmd.prefill().unwrap();
            assert_eq!(args.compute.threads(), width.parse::<usize>().unwrap());
            assert_eq!(limits.compute, args.compute.admit_prefill(prefill).unwrap());
            assert_eq!(prefill.map(|p| p.max_batch_rows), rows.map(|n| n.parse::<usize>().unwrap()));
            assert!(args.policy.prefill().unwrap().is_none());
            assert_eq!(canonjson::canonical_bytes(&cmd.host.task_budget(limits)).unwrap(),
                canonjson::canonical_bytes(&serial.host.task_budget(serial_limits)).unwrap());
            assert_eq!(cmd.head.limits().unwrap().unwrap().max_rows_per_step, 31);
            assert!(cmd.source_membership);
            // Corpus/job consumers of the shared host retain their own serial contract.
            let (shared, shared_limits) = cmd.host.common(cmd.input.clone()).unwrap();
            assert_eq!(shared.compute.threads(), 1);
            assert_eq!(shared_limits.compute.workspace_bytes, 0);
        }
    }
}

#[test]
fn invalid_extraction_schedules_refuse_before_schema_source_model_or_output_io() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private source read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("extraction output written") }
        fn flush(&mut self) -> io::Result<()> { panic!("extraction output flushed") }
    }
    for flag in ["--compute-threads", "--prefill-rows"] {
        for value in ["0", "65", "1000000"] {
            assert_eq!(command(&[flag, value]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
        }
        for value in ["-1", "NaN", "4.0", "999999999999999999999999999999999999"] {
            assert!(super::super::definition().try_get_matches_from(["candidate", "extract",
                "--model", "must-not-open.fnlpq", "--memory-mib", "8192",
                "--schema", "must-not-open.json", flag, value]).is_err());
        }
    }
    for flags in [vec!["--compute-threads", "4"], vec!["--prefill-rows", "4"],
        vec!["--compute-threads", "4", "--prefill-rows", "4"]] {
        let mut cmd = command(&flags);
        cmd.host.memory_mib = 16_384;
        let (args, mut limits) = cmd.validate().unwrap();
        let prefill = cmd.prefill().unwrap();
        let peak = compute::source_peak_bytes(&args, limits, cmd.host.max_result_bytes, prefill).unwrap();
        limits.memory_bytes = peak;
        compute::check_source_memory(&args, limits, cmd.host.max_result_bytes, prefill).unwrap();
        limits.memory_bytes -= 1;
        assert_eq!(compute::check_source_memory(&args, limits, cmd.host.max_result_bytes, prefill),
            Err(CandidateError::Arguments));
        // The exact peak must still fail if either preparation owner, the
        // coordinator's prompt scratch or the fixed child payload is omitted.
        let extra = prefill.map(|p| p.validate().unwrap()).unwrap_or(0);
        for missing in [limits.preparation_bytes, extra, limits.compute.workspace_bytes] {
            if missing == 0 { continue; }
            limits.memory_bytes = peak - missing;
            assert_eq!(compute::check_source_memory(&args, limits, cmd.host.max_result_bytes, prefill),
                Err(CandidateError::Arguments));
        }
        cmd.host.memory_mib = (peak - 1) / MIB;
        assert_eq!(cmd.execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
}

#[test]
fn compact_extraction_admits_the_complete_schema_and_source_workspace_before_io() {
    use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};
    let cmd = command(&["--input-encoding", "safe-bpe-v1",
        "--max-tokenizer-pair-checks", "1234", "--max-tokenizer-pair-bytes", "10000"]);
    let (_, limits) = cmd.validate().unwrap();
    let UntrustedEncoding::SafeBpeV1(policy) = cmd.encoding_policy(limits).unwrap()
        else { panic!("compact extraction policy was discarded"); };
    assert_eq!(policy.max_input_bytes, cmd.host.max_input_bytes + SCHEMA_BYTES);
    assert_eq!(policy.max_output_tokens, limits.max_prompt_tokens);
    assert_eq!(policy.max_pair_checks, 1234); assert_eq!(policy.max_pair_bytes, 10000);
    assert!(limits.preparation_bytes >= 256 * MIB + SCHEMA_BYTES as u64 * 32
        + policy.requirements().unwrap().workspace_bytes);

    // The document-only reserve fits 512 MiB. The additional exact schema and
    // its tokenizer workspace must still refuse before any private IO.
    let cmd = command(&["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "850000"]);
    cmd.host.common_with_encoding(cmd.input.clone(), &cmd.encoding).unwrap();
    assert!(cmd.validate().is_err());
    let cmd = command(&["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "850000",
        "--preparation-mib", "1024"]);
    cmd.validate().unwrap();

    let cmd = command(&["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "1048576",
        "--preparation-mib", "1024"]);
    let (_, limits) = cmd.validate().unwrap();
    let UntrustedEncoding::SafeBpeV1(policy) = cmd.encoding_policy(limits).unwrap() else { unreachable!() };
    assert_eq!(policy.max_input_bytes, SafeBpeLimits::MAX_INPUT_BYTES);
    assert_eq!(cmd.host.max_input_bytes, 1_048_576);
}

#[test]
fn invalid_compact_extraction_arguments_never_open_schema_source_or_model() {
    struct NoIo;
    impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private source read") } }
    impl Write for NoIo {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("extraction output written") }
        fn flush(&mut self) -> io::Result<()> { panic!("extraction output flushed") }
    }
    for flags in [
        vec!["--max-tokenizer-pair-checks", "0"],
        vec!["--max-tokenizer-pair-bytes", "0"],
        vec!["--max-tokenizer-pair-checks", "67108865"],
        vec!["--max-tokenizer-pair-bytes", "536870913"],
        vec!["--max-input-bytes", "850000"],
        vec!["--max-input-bytes", "1048577", "--preparation-mib", "1024"],
    ] {
        let mut args = vec!["--input-encoding", "safe-bpe-v1"];
        args.extend(flags);
        assert_eq!(command(&args).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Arguments));
    }
    for flag in ["--max-tokenizer-pair-checks", "--max-tokenizer-pair-bytes"] {
        let matches = super::super::definition().try_get_matches_from(["candidate", "extract",
            "--schema", "must-not-open.json", "--model", "must-not-open.fnlpq",
            "--memory-mib", "8192", flag, "1"]).unwrap();
        assert!(CandidateCommand::from_matches(&matches).is_err());
    }
}

#[test]
fn compact_encoding_keeps_original_schema_and_document_caps_and_exact_numbers() {
    let cmd = command(&["--input-encoding", "safe-bpe-v1", "--max-input-bytes", "64"]);
    let (_, limits) = cmd.validate().unwrap();
    let number = "0.12345678901234567890123456789012345678";
    let schema = format!(" {{\"type\":\"number\",\"const\":{number}}}\n");
    let request = arguments(schema.clone(), false, cmd.host.task_budget(limits)).unwrap();
    check_schema(&request, &"x".repeat(64), &cmd.host).unwrap();
    assert_eq!(request.schema, schema);
    let program = JsonProgram::compile(&request.schema, compiler(&cmd.host)).unwrap();
    program.validate_json(number).unwrap();
    assert_eq!(check_schema(&request, &"x".repeat(65), &cmd.host), Err(CandidateError::Input));
    assert!(arguments(" ".repeat(SCHEMA_BYTES + 1), false, request.budget).is_err());
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn disabled_feature_never_reads_schema_source_or_model() {
    struct NoRead;
    impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("private input read"); } }
    let mut output = Vec::new();
    let mut diagnostics = Vec::new();
    let status = CandidateCommand::Extract(command(&[])).run(&mut NoRead, &mut output, &mut diagnostics);
    assert_ne!(status, ExitCode::SUCCESS);
    assert!(output.is_empty());
    let diagnostic = String::from_utf8(diagnostics).unwrap();
    assert!(diagnostic.contains("asupersync-runtime"));
    assert!(!diagnostic.contains("local.fnlpq") && !diagnostic.contains("schema.json"));
    let cmd = command(&["--input-encoding", "safe-bpe-v1"]);
    assert_eq!(cmd.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Unavailable));
    for width in ["1", "4"] {
        let mut cmd = command(&["--compute-threads", width, "--prefill-rows", "4",
            "--input-encoding", "safe-bpe-v1", "--selected-rows", "31"]);
        cmd.host.memory_mib = 16_384;
        assert_eq!(cmd.execute(&mut NoRead, &mut Vec::new()), Err(CandidateError::Unavailable));
    }
}
