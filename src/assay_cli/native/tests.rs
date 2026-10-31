use super::*;

pub(super) const PIN: &str = "1111111111111111111111111111111111111111111111111111111111111111";
pub(super) const BASELINE_PIN: &str = "2222222222222222222222222222222222222222222222222222222222222222";
pub(super) fn parse(arguments: &[&str]) -> Command {
    let definition = super::super::definitions().into_iter().find(|command| command.get_name() == arguments[0]).unwrap();
    let matches = definition.try_get_matches_from(arguments).unwrap();
    match AssayCommand::from_matches(arguments[0], &matches).unwrap() {
        AssayCommand::Native(command) => command,
        AssayCommand::Scored(_) => panic!("native fixture selected supplied observations"),
    }
}
pub(super) fn command(paired: bool, flags: &[&str]) -> Command {
    let mut arguments = vec![if paired { "qualify" } else { "eval" }, "native-int4",
        "--suite", "absent-private-suite", "--cases", "-", "--key-file", "absent-private-key",
        "--key-id", "fixture-key", "--namespace", "fixture-suite", "--snapshot", "absent-private-candidate",
        "--expected-model-sha256", PIN, "--memory-mib", "32768"];
    if paired { arguments.extend_from_slice(&["--baseline-snapshot", "absent-private-baseline",
        "--baseline-model-sha256", BASELINE_PIN, "--gates", "absent-private-gates", "--host-scope", "declared-fixture"]); }
    arguments.extend_from_slice(flags); parse(&arguments)
}
struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> { panic!("preflight reached input"); }
}
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> { panic!("preflight published a result"); }
    fn flush(&mut self) -> std::io::Result<()> { panic!("preflight flushed a result"); }
}

#[test]
fn native_schema_discovery_needs_no_files_models_keys_or_runtime() {
    for operation in ["eval", "qualify"] {
        let mut output = Vec::new(); let mut errors = Vec::new();
        assert_eq!(parse(&[operation, "native-int4", "--schema"]).run(&mut NoRead, &mut output, &mut errors), ExitCode::SUCCESS);
        assert!(errors.is_empty()); assert_eq!(output.last(), Some(&b'\n'));
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["protocol"], NATIVE_PROTOCOL); assert_eq!(value["operation"], operation);
        assert_eq!(value["native_model_replay"], true); assert_eq!(value["activation_authority"], false);
        assert_eq!(value["private_bindings"]["execution_collection"]["format"],
            crate::assay::native_classify::EXECUTION_COLLECTION_FORMAT);
        assert_eq!(value["private_bindings"]["observation_collection"]["format"],
            crate::assay::native_classify::OBSERVATION_COLLECTION_FORMAT);
        let definitions = &value["json_schema"]["$defs"];
        assert_eq!(definitions["suite"]["properties"]["task"]["const"], "binary_classification");
        assert!(definitions["suite"]["properties"].get("execution_identity").is_none());
        assert!(definitions["case"]["properties"].get("probability").is_none());
        assert_eq!(definitions["case"]["additionalProperties"], false);
        assert_eq!(definitions["case"]["required"].as_array().unwrap().len(), 5);
        let gates = &definitions["qualification_gates"];
        assert_eq!(gates["additionalProperties"], false);
        assert!(gates["required"].as_array().unwrap().iter().any(|field| field == "max_macro_f1_decrease"));
    }
}

#[test]
fn native_routes_require_explicit_pins_and_do_not_accept_scores_sampling_or_activation_controls() {
    for arguments in [vec!["eval", "native-int4"], vec!["qualify", "native-int4"],
        vec!["calibrate", "native-int4", "--schema"],
        vec!["eval", "native-int4", "--schema", "--memory-mib", "100"],
        vec!["eval", "native-int4", "--observations", "scores.ndjson"],
        vec!["eval", "native-int4", "--schema", "--seed", "1"],
        vec!["eval", "native-int4", "--schema", "--activate"],
        vec!["eval", "native-int4", "--schema", "--baseline-snapshot", "another"],
        vec!["qualify", "native-int4", "--schema", "--baseline-suite", "different-suite"]] {
        let definition = super::super::definitions().into_iter().find(|command| command.get_name() == arguments[0]).unwrap();
        assert!(definition.try_get_matches_from(arguments).is_err());
    }
    for missing in 0..3 {
        let mut options = command(false, &[]);
        match missing { 0 => options.args.snapshot = None, 1 => options.args.expected_model_sha256 = None,
            _ => options.args.memory_mib = None }
        assert_eq!(options.validate().unwrap_err().code, ErrorCode::Usage);
    }
}

#[test]
fn invalid_native_bounds_are_rejected_before_input_with_private_values_redacted() {
    for field in 0..14 {
        let mut options = command(false, &[]);
        match field {
            0 => options.args.expected_model_sha256 = Some("PRIVATE-invalid-pin".to_owned()),
            1 => options.args.group_width = 0,
            2 => options.args.context_tokens = 1,
            3 => options.args.max_candidate_tokens = 65,
            4 => options.args.max_result_bytes = 0,
            5 => options.args.preparation_mib = 0,
            6 => options.args.timeout_seconds = 0,
            7 => options.args.snapshot_checkpoints = 1,
            8 => options.args.max_checkpoints = 1,
            9 => options.args.max_forward_positions = 0,
            10 => options.args.max_projected_logits = 0,
            11 => options.args.max_attention_pairs = 0,
            12 => options.args.max_dot_products = 0,
            _ => options.args.max_multiply_accumulates = 0,
        }
        let mut errors = Vec::new();
        assert_eq!(options.run(&mut NoRead, &mut NoWrite, &mut errors), ErrorCode::Usage.as_process_exit(), "field {field}");
        let text = String::from_utf8(errors).unwrap();
        assert!(!text.contains("PRIVATE")); assert!(!text.contains("absent-private"));
        let error: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(error["protocol"], NATIVE_PROTOCOL); assert_eq!(error["event"], "assay_error");
    }
}

#[test]
fn qualification_pins_both_models_and_requires_explicit_safe_scope_and_gate_input() {
    let options = command(true, &[]); options.validate().unwrap();
    let baseline = options.comparison.as_ref().unwrap();
    let candidate = model_input(options.args.snapshot.as_deref(), options.args.expected_model_sha256.as_deref(), 128).unwrap().0;
    let baseline = model_input(baseline.baseline_snapshot.as_deref(), baseline.baseline_model_sha256.as_deref(), 127).unwrap().0;
    assert_eq!(candidate.model_content_sha256, [0x11; 32]);
    assert_eq!(baseline.model_content_sha256, [0x22; 32]); assert_eq!(baseline.group_width, 127);
    for field in 0..5 {
        let mut options = command(true, &[]); let comparison = options.comparison.as_mut().unwrap();
        match field {
            0 => comparison.baseline_model_sha256 = None,
            1 => comparison.baseline_snapshot = Some(PathBuf::from("-")),
            2 => comparison.baseline_group_width = 0,
            3 => comparison.gates = Some(PathBuf::new()),
            _ => comparison.host_scope = Some("PRIVATE\ninvalid".to_owned()),
        }
        let mut errors = Vec::new();
        assert_eq!(options.run(&mut NoRead, &mut NoWrite, &mut errors), ErrorCode::Usage.as_process_exit());
        assert!(!String::from_utf8(errors).unwrap().contains("PRIVATE"));
    }
}

#[test]
fn common_statistical_and_input_limits_are_also_native_preflight_requirements() {
    for flags in [vec!["--max-rows", "2"], vec!["--max-input-bytes", "0"],
        vec!["--acceptance-threshold", "NaN"], vec!["--minimum-samples", "4"],
        vec!["--bootstrap-resamples", "1"], vec!["--max-bootstrap-draws", "0"]] {
        let mut errors = Vec::new();
        assert_eq!(command(false, &flags).run(&mut NoRead, &mut NoWrite, &mut errors), ErrorCode::Usage.as_process_exit());
    }
}
