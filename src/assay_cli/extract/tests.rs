use super::*;

pub(super) const PIN: &str = "1111111111111111111111111111111111111111111111111111111111111111";
pub(super) const BASELINE_PIN: &str = "2222222222222222222222222222222222222222222222222222222222222222";
pub(super) fn parse(arguments: &[&str]) -> Command {
    let definition = super::super::definitions().into_iter().find(|value| value.get_name() == arguments[0]).unwrap();
    let matches = definition.try_get_matches_from(arguments).unwrap();
    match AssayCommand::from_matches(arguments[0], &matches).unwrap() {
        AssayCommand::NativeExtract(command) => command,
        AssayCommand::Native(_) | AssayCommand::NativeNer(_) | AssayCommand::Scored(_) =>
            panic!("extraction fixture selected another task"),
    }
}
fn arguments<'a>(paired: bool, flags: &[&'a str]) -> Vec<&'a str> {
    let mut arguments = vec![if paired { "qualify" } else { "eval" }, "native-extract-int4",
        "--suite", "absent-private-suite", "--cases", "-", "--key-file", "absent-private-key",
        "--key-id", "fixture-key", "--namespace", "fixture-suite", "--snapshot", "absent-private-candidate",
        "--expected-model-sha256", PIN, "--memory-mib", "131072"];
    if paired { arguments.extend_from_slice(&["--baseline-snapshot", "absent-private-baseline",
        "--baseline-model-sha256", BASELINE_PIN, "--gates", "absent-private-gates", "--host-scope", "declared-fixture"]); }
    arguments.extend_from_slice(flags); arguments
}
pub(super) fn command(paired: bool, flags: &[&str]) -> Command { parse(&arguments(paired, flags)) }
pub(super) struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> { panic!("preflight reached source input"); }
}
pub(super) struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> { panic!("preflight published a result"); }
    fn flush(&mut self) -> std::io::Result<()> { panic!("preflight flushed a result"); }
}

#[test]
fn extraction_schema_discovery_preserves_exact_inner_json_and_no_prediction_authority() {
    for operation in ["eval", "qualify"] {
        let mut output = Vec::new(); let mut errors = Vec::new();
        assert_eq!(parse(&[operation, "native-extract-int4", "--schema"]).run(&mut NoRead, &mut output, &mut errors),
            ExitCode::SUCCESS);
        assert!(errors.is_empty()); assert_eq!(output.last(), Some(&b'\n'));
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["protocol"], EXTRACT_PROTOCOL); assert_eq!(value["operation"], operation);
        assert_eq!(value["score_space"], "not_computed"); assert_eq!(value["activation_authority"], false);
        assert_eq!(value["supported_tasks"], serde_json::json!(["structured_extraction"]));
        assert_eq!(value["private_bindings"]["execution_collection"]["format"], native_extract::EXECUTION_COLLECTION_FORMAT);
        assert_eq!(value["private_bindings"]["observation_collection"]["format"], native_extract::OBSERVATION_COLLECTION_FORMAT);
        let definitions = &value["json_schema"]["$defs"];
        assert_eq!(definitions["suite"]["properties"]["task"]["const"], "structured_extraction");
        assert_eq!(definitions["schema_definition"]["additionalProperties"], false);
        assert_eq!(definitions["schema_definition"]["required"], serde_json::json!(["id", "schema_json", "grounding"]));
        assert_eq!(definitions["schema_definition"]["properties"]["schema_json"]["type"], "string");
        assert_eq!(definitions["case"]["properties"]["expected_json"]["type"], "string");
        assert_eq!(definitions["case"]["required"].as_array().unwrap().len(), 7);
        assert_eq!(definitions["case"]["additionalProperties"], false);
        for field in ["execution_identity", "probability", "prediction", "unit_sha256", "output"] {
            assert!(definitions["case"]["properties"].get(field).is_none());
        }
        assert_eq!(definitions["qualification_gates"]["required"].as_array().unwrap().len(), 4);
        assert!(value.get("calibration").is_none());
    }
}

#[test]
fn extraction_routes_refuse_unsupported_controls_even_with_otherwise_complete_arguments() {
    let definition = || super::super::definitions().into_iter().find(|value| value.get_name() == "eval").unwrap();
    assert!(definition().try_get_matches_from(arguments(false, &[])).is_ok());
    for flags in [vec!["--acceptance-threshold", "0.8"], vec!["--calibration", "foreign.json"],
        vec!["--observations", "predictions.ndjson"], vec!["--max-labels", "3"], vec!["--max-spans-per-row", "4"],
        vec!["--seed", "1"], vec!["--activate"], vec!["--source-membership"], vec!["--verify-semantic"]] {
        let error = definition().try_get_matches_from(arguments(false, &flags)).unwrap_err();
        assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
    }
    let calibration = super::super::definitions().into_iter().find(|value| value.get_name() == "calibrate").unwrap();
    assert!(calibration.try_get_matches_from(["calibrate", "native-extract-int4", "--schema"]).is_err());
    for missing in 0..3 {
        let mut options = command(false, &[]);
        match missing { 0 => options.args.snapshot = None, 1 => options.args.expected_model_sha256 = None,
            _ => options.args.memory_mib = None }
        assert_eq!(options.validate().unwrap_err().code, ErrorCode::Usage);
    }
}

#[test]
fn invalid_exact_json_geometry_and_native_work_refuse_before_private_input() {
    for field in 0..28 {
        let mut options = command(false, &[]);
        match field {
            0 => options.args.expected_model_sha256 = Some("PRIVATE-invalid-pin".to_owned()),
            1 => options.args.group_width = 0,
            2 => options.args.context_tokens = options.args.max_new_tokens,
            3 => options.args.max_new_tokens = 0,
            4 => options.args.max_result_bytes = 0,
            5 => options.args.preparation_mib = 511,
            6 => options.args.timeout_seconds = 0,
            7 => options.args.snapshot_checkpoints = 1,
            8 => options.args.max_checkpoints = 1,
            9 => options.args.max_forward_positions = 0,
            10 => options.args.max_projected_logits = 0,
            11 => options.args.max_attention_pairs = 0,
            12 => options.args.max_dot_products = 0,
            13 => options.args.max_multiply_accumulates = 0,
            14 => options.args.max_schemas = 0,
            15 => options.args.max_schemas = native_extract::MAX_SCHEMAS + 1,
            16 => options.args.max_schema_bytes = 0,
            17 => options.args.max_fields_per_row = native_extract::MAX_FIELDS_PER_ROW + 1,
            18 => options.args.max_json_nodes_per_row = native_extract::MAX_JSON_NODES_PER_ROW + 1,
            19 => options.args.max_json_depth = native_extract::MAX_JSON_DEPTH + 1,
            20 => options.args.max_languages = native_extract::MAX_LANGUAGES + 1,
            21 => options.args.max_occurrences_per_row = native_extract::MAX_OCCURRENCES_PER_ROW + 1,
            22 => options.args.max_verification_steps = 0,
            23 => options.args.max_grammar_states = 0,
            24 => options.args.mask_step_node_visits = 0,
            25 => options.args.max_mask_node_visits = options.args.mask_step_node_visits as u64 - 1,
            26 => options.args.statistics.minimum_samples = 4,
            _ => options.args.max_json_nodes_per_row = 0,
        }
        let mut errors = Vec::new();
        assert_eq!(options.run(&mut NoRead, &mut NoWrite, &mut errors), ErrorCode::Usage.as_process_exit(), "field {field}");
        let text = String::from_utf8(errors).unwrap();
        assert!(!text.contains("PRIVATE")); assert!(!text.contains("absent-private"));
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["protocol"], EXTRACT_PROTOCOL); assert_eq!(value["event"], "assay_error");
    }
}

#[test]
fn extraction_compiler_policy_and_comparison_pins_remain_explicit() {
    let options = command(true, &[]); let (limits, extract) = options.validate().unwrap();
    assert_eq!(options.args.compiler().max_array_items, CompileLimits::default().max_array_items);
    assert_eq!(options.args.source(limits).verification.max_fields, extract.max_fields_per_row);
    assert_eq!(options.args.source(limits).verification.max_scan_steps, extract.max_verification_steps);
    let comparison = options.comparison.as_ref().unwrap();
    let candidate = model_input(options.args.snapshot.as_deref(), options.args.expected_model_sha256.as_deref(), 128).unwrap().0;
    let baseline = model_input(comparison.baseline_snapshot.as_deref(), comparison.baseline_model_sha256.as_deref(), 127).unwrap().0;
    assert_eq!(candidate.model_content_sha256, [0x11; 32]); assert_eq!(baseline.model_content_sha256, [0x22; 32]);
    assert_eq!(baseline.group_width, 127);
    for field in 0..5 {
        let mut options = command(true, &[]); let comparison = options.comparison.as_mut().unwrap();
        match field { 0 => comparison.baseline_model_sha256 = None,
            1 => comparison.baseline_snapshot = Some(PathBuf::from("-")), 2 => comparison.baseline_group_width = 0,
            3 => comparison.gates = Some(PathBuf::new()), _ => comparison.host_scope = Some("PRIVATE\ninvalid".to_owned()) }
        let mut errors = Vec::new();
        assert_eq!(options.run(&mut NoRead, &mut NoWrite, &mut errors), ErrorCode::Usage.as_process_exit());
        assert!(!String::from_utf8(errors).unwrap().contains("PRIVATE"));
    }
}

#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn extraction_runtime_unavailability_is_explicit_before_source_io() {
    let mut errors = Vec::new();
    assert_eq!(command(false, &[]).run(&mut NoRead, &mut NoWrite, &mut errors), ErrorCode::Generic.as_process_exit());
    let value: serde_json::Value = serde_json::from_slice(&errors).unwrap();
    assert_eq!(value["reason"], "native_runtime_unavailable");
}
