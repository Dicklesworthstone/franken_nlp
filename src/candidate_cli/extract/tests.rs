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
}
