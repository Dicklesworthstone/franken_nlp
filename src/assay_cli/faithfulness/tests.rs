use super::*;
const PIN: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const BASELINE_PIN: &str = "2222222222222222222222222222222222222222222222222222222222222222";
pub(super) fn parse(arguments: &[&str]) -> Command {
    let definition = super::super::definitions().into_iter().find(|value| value.get_name() == arguments[0]).unwrap();
    let matches = definition.try_get_matches_from(arguments).unwrap();
    match AssayCommand::from_matches(arguments[0], &matches).unwrap() {
        AssayCommand::NativeFaithfulness(command) => command,
        _ => panic!("faithfulness fixture selected another task"),
    }
}
pub(super) fn command(paired: bool, flags: &[&str]) -> Command {
    let mut arguments = vec![if paired { "qualify" } else { "eval" }, "native-faithfulness-int4",
        "--suite", "absent-private-suite", "--cases", "-", "--key-file", "absent-private-key",
        "--key-id", "fixture-key", "--namespace", "fixture-suite", "--snapshot", "absent-private-candidate",
        "--expected-model-sha256", PIN, "--memory-mib", "32768"];
    if paired { arguments.extend_from_slice(&["--baseline-snapshot", "absent-private-baseline",
        "--baseline-model-sha256", BASELINE_PIN, "--gates", "absent-private-gates", "--host-scope", "declared-fixture"]); }
    arguments.extend_from_slice(flags); parse(&arguments)
}
pub(super) struct NoRead;
impl Read for NoRead { fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> { panic!("preflight read private input"); } }
pub(super) struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> { panic!("preflight published output"); }
    fn flush(&mut self) -> std::io::Result<()> { panic!("preflight flushed output"); }
}
#[test]
fn schema_is_model_free_and_explains_three_relations_source_clusters_and_no_truth_claim() {
    for operation in ["eval", "qualify"] {
        let mut output = Vec::new(); let mut errors = Vec::new();
        assert_eq!(parse(&[operation, "native-faithfulness-int4", "--schema"])
            .run(&mut NoRead, &mut output, &mut errors), ExitCode::SUCCESS);
        assert!(errors.is_empty()); assert_eq!(output.last(), Some(&b'\n'));
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["protocol"], FAITHFULNESS_PROTOCOL); assert_eq!(value["operation"], operation);
        assert_eq!(value["activation_authority"], false); assert_eq!(value["supported_tasks"], serde_json::json!(["faithfulness"]));
        assert_eq!(value["metrics"]["minimum_samples"]["unit"], "distinct locked-test sources");
        assert_eq!(value["private_bindings"]["execution_collection"]["format"], native_faithfulness::EXECUTION_COLLECTION_FORMAT);
        assert_eq!(value["private_bindings"]["observation_collection"]["format"], native_faithfulness::OBSERVATION_COLLECTION_FORMAT);
        let defs = &value["json_schema"]["$defs"];
        assert_eq!(defs["case"]["additionalProperties"], false);
        assert_eq!(defs["case"]["required"], serde_json::json!(["schema_version","id","split","source","claim","relation"]));
        assert_eq!(defs["case"]["properties"]["relation"]["enum"], serde_json::json!(["contradicted","entailed","unsupported"]));
        for field in ["probability", "confidence", "execution_identity", "evidence", "unit_sha256"] {
            assert!(defs["case"]["properties"].get(field).is_none());
        }
    }
}
#[test]
fn route_requires_pinned_models_and_rejects_score_policy_sampling_and_activation_overrides() {
    for arguments in [vec!["eval", "native-faithfulness-int4"], vec!["qualify", "native-faithfulness-int4"],
        vec!["calibrate", "native-faithfulness-int4", "--schema"],
        vec!["eval", "native-faithfulness-int4", "--schema", "--memory-mib", "100"],
        vec!["eval", "native-faithfulness-int4", "--schema", "--acceptance-threshold", "0.8"],
        vec!["eval", "native-faithfulness-int4", "--schema", "--calibration", "foreign.json"],
        vec!["eval", "native-faithfulness-int4", "--observations", "scores.ndjson"],
        vec!["eval", "native-faithfulness-int4", "--schema", "--seed", "1"],
        vec!["eval", "native-faithfulness-int4", "--schema", "--activate"],
        vec!["eval", "native-faithfulness-int4", "--schema", "--minimum-margin-milli", "0"],
        vec!["qualify", "native-faithfulness-int4", "--schema", "--baseline-cases", "different.ndjson"]] {
        let definition = super::super::definitions().into_iter().find(|value| value.get_name() == arguments[0]).unwrap();
        assert!(definition.try_get_matches_from(arguments).is_err());
    }
}
#[test]
fn invalid_complete_resource_plans_refuse_before_input_and_redact_private_options() {
    for field in 0..19 {
        let mut options = command(false, &[]);
        match field { 0 => options.args.snapshot = None,
            1 => options.args.expected_model_sha256 = Some("PRIVATE-invalid-pin".to_owned()),
            2 => options.args.memory_mib = None, 3 => options.args.group_width = 0,
            4 => options.args.context_tokens = 1, 5 => options.args.max_heads = 0,
            6 => options.args.max_heads = 33, 7 => options.args.max_result_bytes = 0,
            8 => options.args.preparation_mib = 511, 9 => options.args.timeout_seconds = 0,
            10 => options.args.snapshot_checkpoints = 1, 11 => options.args.max_checkpoints = 1,
            12 => options.args.max_forward_positions = 0, 13 => options.args.max_projected_logits = 0,
            14 => options.args.max_attention_pairs = 0, 15 => options.args.max_dot_products = 0,
            16 => options.args.max_multiply_accumulates = 0, 17 => options.args.statistics.minimum_samples = 4,
            _ => options.args.statistics.confidence_level = f64::NAN }
        let mut errors = Vec::new();
        assert_eq!(options.run(&mut NoRead, &mut NoWrite, &mut errors), ErrorCode::Usage.as_process_exit(), "field {field}");
        let text = String::from_utf8(errors).unwrap(); assert!(!text.contains("PRIVATE")); assert!(!text.contains("absent-private"));
        let value: serde_json::Value = serde_json::from_str(&text).unwrap(); assert_eq!(value["protocol"], FAITHFULNESS_PROTOCOL);
    }
}
#[test]
fn comparison_preserves_independent_pins_and_requires_explicit_safe_policy_scope() {
    let options = command(true, &[]); options.validate().unwrap(); let comparison = options.comparison.as_ref().unwrap();
    let candidate = model_input(options.args.snapshot.as_deref(), options.args.expected_model_sha256.as_deref(), 128).unwrap().0;
    let baseline = model_input(comparison.baseline_snapshot.as_deref(), comparison.baseline_model_sha256.as_deref(), 127).unwrap().0;
    assert_eq!(candidate.model_content_sha256, [0x11; 32]); assert_eq!(baseline.model_content_sha256, [0x22; 32]);
    assert_eq!(baseline.group_width, 127);
    for field in 0..4 {
        let mut options = command(true, &[]); let comparison = options.comparison.as_mut().unwrap();
        match field { 0 => comparison.baseline_model_sha256 = None,
            1 => comparison.baseline_snapshot = Some(PathBuf::from("-")),
            2 => comparison.gates = Some(PathBuf::new()), _ => comparison.host_scope = Some("PRIVATE\ninvalid".to_owned()) }
        let mut errors = Vec::new(); assert_eq!(options.run(&mut NoRead, &mut NoWrite, &mut errors), ErrorCode::Usage.as_process_exit());
        assert!(!String::from_utf8(errors).unwrap().contains("PRIVATE"));
    }
}
#[cfg(not(feature = "asupersync-runtime"))]
#[test]
fn unavailable_native_runtime_is_typed_before_source_io() {
    let mut errors = Vec::new();
    assert_eq!(command(false, &[]).run(&mut NoRead, &mut NoWrite, &mut errors), ErrorCode::Generic.as_process_exit());
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&errors).unwrap()["reason"], "native_runtime_unavailable");
}
