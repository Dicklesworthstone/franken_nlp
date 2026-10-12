//! Emitter fixtures are synthetic protocol data, never kernel evidence.
use super::*;
use serde_json::{Value, json};
use crate::native_engine::selftest::{CaseFailure, Mismatch, Operation, Pattern, TierReport};

fn sample() -> SelftestReport {
    let failed = CaseReport { operation: Operation::Gemv, pattern: Pattern::Mixed,
        m: 1, k: 1, n: 1, requested_kernel: "s_scalar_i32_int8_gemv",
        observed_kernel: Some("s_scalar_i32_int8_gemv"), expected_sha256: "a".repeat(64),
        actual_sha256: Some("b".repeat(64)), failure: Some(CaseFailure::IntegerMismatch),
        first_mismatch: Some(Mismatch { row: 0, column: 0, expected: 1, actual: 2 }) };
    let mut passed = failed.clone();
    passed.failure = None; passed.first_mismatch = None; passed.actual_sha256 = Some(passed.expected_sha256.clone());
    SelftestReport { schema_version: SELFTEST_SCHEMA_VERSION, suite_version: SELFTEST_SUITE_VERSION,
        scope: "finite_synthetic_integer_cases_only_not_model_or_performance_qualification",
        detected_features: dispatch::host_features(), requested_tier: Some(KernelTier::Scalar),
        cases_per_tier: CASES_PER_TIER,
        tiers: vec![TierReport { tier: KernelTier::Scalar, status: TierStatus::Fail,
            cases_run: 2, cases_passed: 1, cases: vec![passed, failed] }],
        cases_run: 2, cases_passed: 1, tiers_skipped: 0, verdict: Verdict::Fail }
}
fn records(bytes: &[u8]) -> Vec<Value> {
    std::str::from_utf8(bytes).unwrap().lines().map(|line| canonjson::parse_str(line).unwrap()).collect()
}

// Independent test-side checker for the subset used by THIS event schema.
// This is not a new product JSON-Schema compiler or external-input boundary.
fn conforms(schema: &Value, value: &Value) -> bool {
    if let Some(choices) = schema.get("oneOf").and_then(Value::as_array) {
        return choices.iter().filter(|choice| conforms(choice, value)).count() == 1;
    }
    if let Some(choices) = schema.get("anyOf").and_then(Value::as_array) {
        return choices.iter().any(|choice| conforms(choice, value));
    }
    if schema.get("const").is_some_and(|constant| constant != value) { return false; }
    if schema.get("enum").and_then(Value::as_array).is_some_and(|choices| !choices.contains(value)) { return false; }
    match schema.get("type").and_then(Value::as_str) {
        Some("object") => {
            let Some(object) = value.as_object() else { return false; };
            let properties = schema["properties"].as_object().unwrap();
            let required = schema["required"].as_array().unwrap();
            if required.iter().any(|key| !object.contains_key(key.as_str().unwrap())) { return false; }
            if object.keys().any(|key| !properties.contains_key(key)) { return false; }
            object.iter().all(|(key, value)| conforms(&properties[key], value))
        }
        Some("integer") => {
            let Some(number) = value.as_i64() else { return false; };
            !schema.get("minimum").and_then(Value::as_i64).is_some_and(|minimum| number < minimum)
                && !schema.get("maximum").and_then(Value::as_i64).is_some_and(|maximum| number > maximum)
        }
        Some("string") => {
            let Some(text) = value.as_str() else { return false; };
            if schema.get("minLength").and_then(Value::as_u64).is_some_and(|minimum| (text.len() as u64) < minimum) { return false; }
            if schema.get("pattern").is_some() {
                return text.len() == 64 && text.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
            }
            true
        }
        Some("boolean") => value.is_boolean(),
        Some("null") => value.is_null(),
        None => true,
        _ => panic!("unreviewed selftest schema feature"),
    }
}

#[test]
fn default_output_always_includes_failures_and_details_adds_successful_cases() {
    for details in [false, true] {
        let mut output = Vec::new();
        emit_report(&sample(), details, &mut output).unwrap();
        let rows = records(&output);
        assert_eq!(rows.iter().filter(|row| row["event"] == "selftest_case").count(), if details { 2 } else { 1 });
        assert_eq!(rows.last().unwrap()["event"], "selftest_complete");
        assert_eq!(rows.last().unwrap()["data"]["verdict"], "fail");
        assert_eq!(rows.last().unwrap()["data"]["cases_run"], 2);
        assert!(rows.iter().all(|row| conforms(&schema::document(), row)));
    }
}

#[test]
fn every_start_summary_case_and_error_record_obeys_its_namespaced_schema() {
    let report = sample(); let mut output = Vec::new();
    event(&mut output, "selftest_start", &json!({"suite_version": SELFTEST_SUITE_VERSION,
        "requested_tier": report.requested_tier, "cases_per_tier": CASES_PER_TIER,
        "detected_features": report.detected_features})).unwrap();
    emit_report(&report, true, &mut output).unwrap();
    for code in ["allocation_refused", "oracle_overflow"] {
        event(&mut output, "selftest_error", &json!({"code": code})).unwrap();
    }
    let schema = schema::document();
    for row in records(&output) {
        assert!(conforms(&schema, &row), "invalid selftest record: {row}");
        assert_eq!(row["protocol"], PROTOCOL);
        assert_eq!(row["schema_version"], SELFTEST_SCHEMA_VERSION);
    }
}

#[test]
fn schema_rejects_unknown_fields_bad_checksums_and_wrong_protocol() {
    let mut output = Vec::new(); emit_report(&sample(), true, &mut output).unwrap();
    let case = records(&output).remove(0); let schema = schema::document();
    let mut extra = case.clone(); extra["data"]["unreviewed"] = json!(true);
    assert!(!conforms(&schema, &extra));
    let mut bad_hash = case.clone(); bad_hash["data"]["expected_sha256"] = json!("not-a-digest");
    assert!(!conforms(&schema, &bad_hash));
    let mut wrong = case.clone(); wrong["protocol"] = json!("task-output");
    assert!(!conforms(&schema, &wrong));
    let mut version = case; version["schema_version"] = json!(3);
    assert!(!conforms(&schema, &version));
}

#[test]
fn schema_mode_emits_only_the_schema_and_needs_no_runtime_or_model() {
    let matches = definition().try_get_matches_from(["selftest", "--schema"]).unwrap();
    let mut output = Vec::new(); let mut errors = Vec::new();
    assert_eq!(run(&matches, &mut output, &mut errors), ExitCode::SUCCESS);
    let docs = records(&output); assert_eq!(docs.len(), 1);
    assert_eq!(docs[0]["$id"], "urn:franken-nlp:cpu-selftest:1");
    assert_eq!(docs[0]["oneOf"].as_array().unwrap().len(), 5);
    assert!(errors.is_empty());
}

struct Disconnected;
impl Write for Disconnected {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { Err(io::ErrorKind::BrokenPipe.into()) }
    fn flush(&mut self) -> io::Result<()> { Err(io::ErrorKind::BrokenPipe.into()) }
}
#[test]
fn disconnected_startup_and_truncated_output_are_non_successful() {
    let matches = definition().try_get_matches_from(["selftest"]).unwrap();
    let mut errors = Vec::new();
    assert_eq!(run(&matches, &mut Disconnected, &mut errors), ErrorCode::Generic.as_process_exit());
    assert_eq!(String::from_utf8(errors).unwrap(), "fnlp robot selftest: output unavailable\n");
    assert!(emit_report(&sample(), true, &mut Disconnected).is_err());
}

#[test]
fn verdicts_preserve_nonzero_failure_and_unsupported_exit_codes() {
    assert_eq!(verdict_exit(Verdict::Pass), ExitCode::SUCCESS);
    assert_eq!(verdict_exit(Verdict::Fail), ErrorCode::Generic.as_process_exit());
    assert_eq!(verdict_exit(Verdict::Unsupported), ErrorCode::Usage.as_process_exit());
    assert_ne!(verdict_exit(Verdict::Unsupported), ExitCode::SUCCESS);
}
