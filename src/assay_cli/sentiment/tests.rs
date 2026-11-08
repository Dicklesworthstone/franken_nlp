use super::*;
const PIN: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const BASELINE_PIN: &str = "2222222222222222222222222222222222222222222222222222222222222222";
pub(super) fn parse(arguments: &[&str]) -> Command {
    let definition = super::super::definitions().into_iter().find(|value| value.get_name() == arguments[0]).unwrap();
    let matches = definition.try_get_matches_from(arguments).unwrap();
    match AssayCommand::from_matches(arguments[0], &matches).unwrap() {
        AssayCommand::NativeSentiment(command) => command,
        _ => panic!("sentiment fixture selected another task"),
    }
}
pub(super) fn command(paired: bool, flags: &[&str]) -> Command {
    let mut arguments = vec![if paired { "qualify" } else { "eval" }, "native-sentiment-int4",
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
fn model_free_schema_discloses_human_milli_units_document_uncertainty_and_no_calibration() {
    for operation in ["eval", "qualify"] {
        let mut output = Vec::new(); let mut errors = Vec::new();
        assert_eq!(parse(&[operation, "native-sentiment-int4", "--schema"])
            .run(&mut NoRead, &mut output, &mut errors), ExitCode::SUCCESS);
        assert!(errors.is_empty()); assert_eq!(output.last(), Some(&b'\n'));
        let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(value["protocol"], SENTIMENT_PROTOCOL); assert_eq!(value["operation"], operation);
        assert_eq!(value["activation_authority"], false); assert_eq!(value["supported_tasks"], serde_json::json!(["sentiment"]));
        assert_eq!(value["metrics"]["minimum_samples"]["unit"], "distinct locked-test documents");
        assert_eq!(value["private_bindings"]["execution_collection"]["format"], native_sentiment::EXECUTION_COLLECTION_FORMAT);
        assert_eq!(value["private_bindings"]["observation_collection"]["format"], native_sentiment::OBSERVATION_COLLECTION_FORMAT);
        let defs = &value["json_schema"]["$defs"];
        assert_eq!(defs["case"]["additionalProperties"], false);
        assert_eq!(defs["case"]["required"], serde_json::json!(["schema_version","id","split","text","gold"]));
        assert_eq!(defs["gold"]["properties"]["value_milli"]["minimum"], -1000);
        assert_eq!(defs["gold"]["properties"]["value_milli"]["maximum"], 1000);
        for field in ["probability", "confidence", "prediction", "execution_identity", "unit_sha256"] {
            assert!(defs["case"]["properties"].get(field).is_none());
        }
        assert!(value["gold_units"]["coordinate"].as_str().unwrap().contains("NOT -100..100"));
    }
}
#[test]
fn routes_require_independent_pins_and_refuse_gold_prediction_or_task_policy_overrides() {
    for arguments in [vec!["eval", "native-sentiment-int4"], vec!["qualify", "native-sentiment-int4"],
        vec!["calibrate", "native-sentiment-int4", "--schema"],
        vec!["eval", "native-sentiment-int4", "--schema", "--memory-mib", "100"],
        vec!["eval", "native-sentiment-int4", "--schema", "--axes", "valence"],
        vec!["eval", "native-sentiment-int4", "--schema", "--calibration", "foreign.json"],
        vec!["eval", "native-sentiment-int4", "--observations", "predictions.ndjson"],
        vec!["eval", "native-sentiment-int4", "--schema", "--seed", "1"],
        vec!["eval", "native-sentiment-int4", "--schema", "--activate"],
        vec!["eval", "native-sentiment-int4", "--schema", "--minimum-peak-weight-ppm", "0"],
        vec!["qualify", "native-sentiment-int4", "--schema", "--baseline-cases", "different.ndjson"]] {
        let definition = super::super::definitions().into_iter().find(|value| value.get_name() == arguments[0]).unwrap();
        assert!(definition.try_get_matches_from(arguments).is_err());
    }
}
#[test]
fn invalid_resources_are_typed_and_redacted_before_any_private_io() {
    for field in 0..19 {
        let mut options = command(false, &[]);
        match field { 0 => options.args.snapshot = None,
            1 => options.args.expected_model_sha256 = Some("PRIVATE-invalid-pin".to_owned()),
            2 => options.args.memory_mib = None, 3 => options.args.group_width = 0,
            4 => options.args.context_tokens = 4, 5 => options.args.max_axes = 0,
            6 => options.args.max_axes = 5, 7 => options.args.max_result_bytes = 0,
            8 => options.args.preparation_mib = 511, 9 => options.args.timeout_seconds = 0,
            10 => options.args.snapshot_checkpoints = 1, 11 => options.args.max_checkpoints = 1,
            12 => options.args.max_forward_positions = 0, 13 => options.args.max_projected_logits = 0,
            14 => options.args.max_attention_pairs = 0, 15 => options.args.max_dot_products = 0,
            16 => options.args.max_multiply_accumulates = 0, 17 => options.args.statistics.minimum_samples = 4,
            _ => options.args.statistics.confidence_level = f64::NAN }
        let mut errors = Vec::new();
        assert_eq!(options.run(&mut NoRead, &mut NoWrite, &mut errors), ErrorCode::Usage.as_process_exit(), "field {field}");
        let text = String::from_utf8(errors).unwrap(); assert!(!text.contains("PRIVATE")); assert!(!text.contains("absent-private"));
        assert_eq!(serde_json::from_str::<serde_json::Value>(&text).unwrap()["protocol"], SENTIMENT_PROTOCOL);
    }
}
#[test]
fn independent_comparison_pins_and_explicit_safe_scope_preserve_fixed_task_bounds() {
    let options = command(true, &[]); options.validate().unwrap(); let comparison = options.comparison.as_ref().unwrap();
    let candidate = model_input(options.args.snapshot.as_deref(), options.args.expected_model_sha256.as_deref(), 128).unwrap().0;
    let baseline = model_input(comparison.baseline_snapshot.as_deref(), comparison.baseline_model_sha256.as_deref(), 127).unwrap().0;
    assert_eq!(candidate.model_content_sha256, [0x11; 32]); assert_eq!(baseline.model_content_sha256, [0x22; 32]);
    assert_eq!(baseline.group_width, 127);
    let task = options.args.task().unwrap(); assert_eq!(task.max_output_tokens, 5); assert_eq!(task.max_grammar_states, 17);
    assert_eq!(task.max_input_tokens as usize + 4, options.args.context_tokens);
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
fn missing_native_runtime_refuses_before_source_io() {
    let mut errors = Vec::new();
    assert_eq!(command(false, &[]).run(&mut NoRead, &mut NoWrite, &mut errors), ErrorCode::Generic.as_process_exit());
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&errors).unwrap()["reason"], "native_runtime_unavailable");
}
