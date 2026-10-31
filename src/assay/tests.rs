use super::*;
use crate::execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode};
use crate::receipt::CommitmentKey;

pub(crate) fn identity() -> ExecutionIdentity {
    let digest = |value: &str| Sha256Digest::of_bytes(value.as_bytes());
    ExecutionIdentity::new(ExecutionIdentity {
        schema_version: 1, source_revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        logical_model_digest: digest("public-model"), artifact_format: "fixture-int4-v1".to_owned(),
        quant_recipe: "fixture-g128".to_owned(), packing_set_digest: digest("public-pack"),
        tokenizer_digest: digest("public-tokenizer"), template_digest: digest("public-template"),
        task_spec: "classify-v1".to_owned(), taskir_digest: digest("private-ir"),
        prompt_digest: digest("private-prompt"), grammar_compiler_version: "fixture-grammar-v1".to_owned(),
        schema_digest: digest("private-schema"), numerics_profile: NumericsProfile::StrictQuantized { version: 2 },
        kv_dtype: "bf16".to_owned(), sampler_version: "greedy-v1".to_owned(),
        thinking_mode: ThinkingMode::Disabled, tool_mode: ToolMode::None,
        calibration_digest: digest("none"), decision_policy_digest: digest("none"),
        backend_semantic_version: "fixture-backend-v1".to_owned(), host_class: None, compiler_identity: None,
    }).unwrap()
}
pub(crate) fn key() -> CommitmentKey { CommitmentKey::new("assay-tests", [7; 32]).unwrap() }
pub(crate) fn options() -> EvalOptions {
    EvalOptions { minimum_samples: 5, acceptance_threshold: 0.8,
        bootstrap: BootstrapOptions { resamples: 32, confidence_level: 0.95, seed: 17 } }
}
/// Distinct owned fixture units; no native model or externally acquired data.
pub(crate) fn fixture(calibration_rows: usize, locked_rows: usize) -> (Vec<u8>, Vec<u8>) {
    let mut rows = Vec::new();
    let mut splits = SplitIds { development: Vec::new(), calibration: Vec::new(), locked_test: Vec::new() };
    for (split, count) in [(Split::Development, 5), (Split::Calibration, calibration_rows), (Split::LockedTest, locked_rows)] {
        for index in 0..count {
            let id = format!("private-{}-{index:05}", split.name());
            match split {
                Split::Development => splits.development.push(id.clone()),
                Split::Calibration => splits.calibration.push(id.clone()),
                Split::LockedTest => splits.locked_test.push(id.clone()),
            }
            rows.push(Observation { schema_version: 1, unit_sha256: Sha256Digest::of_bytes(id.as_bytes()),
                id, split, probability: if index % 2 == 0 { 0.8 } else { 0.2 }, positive: index % 2 == 0 });
        }
    }
    let input = encode_rows(&rows);
    let manifest = SuiteManifest { schema_version: 1, task: "binary_event".to_owned(),
        dataset_id: "owned-suite".to_owned(), dataset_version: "v1".to_owned(),
        named_population: "owned fixture binary events".to_owned(), positive_label: "positive-event".to_owned(),
        negative_label: "negative-event".to_owned(), execution_identity: identity(),
        expected_input_sha256: Sha256Digest::of_bytes(&input), splits };
    (crate::canonjson::canonical_bytes(&manifest).unwrap(), input)
}
pub(crate) fn dataset() -> Dataset {
    let (manifest, input) = fixture(8, 8);
    Dataset::parse(&manifest, &input, Limits::default()).unwrap()
}
pub(crate) fn encode_rows(rows: &[Observation]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for row in rows {
        bytes.extend(crate::canonjson::canonical_bytes(row).unwrap()); bytes.push(b'\n');
    }
    bytes
}
fn manifest(bytes: &[u8]) -> SuiteManifest {
    serde_json::from_value(crate::canonjson::parse_str(std::str::from_utf8(bytes).unwrap()).unwrap()).unwrap()
}
fn repin(mut suite: SuiteManifest, input: &[u8]) -> Vec<u8> {
    suite.expected_input_sha256 = Sha256Digest::of_bytes(input);
    crate::canonjson::canonical_bytes(&suite).unwrap()
}
fn refusal(suite: &[u8], input: &[u8]) -> AssayError {
    Dataset::parse(suite, input, Limits::default()).err().expect("dataset must refuse")
}

#[test]
fn exact_input_pin_precedes_parsing_and_is_never_replaced_by_canonicalized_bytes() {
    let (suite, input) = fixture(5, 5);
    let mut changed = input.clone(); changed.push(b'\n');
    assert_eq!(refusal(&suite, &changed), AssayError::InputDigest);
    let malformed = b"not-json\n";
    assert_eq!(refusal(&suite, malformed), AssayError::InputDigest);
    let pinned = repin(manifest(&suite), &changed);
    assert_eq!(refusal(&pinned, &changed), AssayError::Observation);
    let crlf = String::from_utf8(input.clone()).unwrap().replace('\n', "\r\n").into_bytes();
    assert_eq!(refusal(&suite, &crlf), AssayError::InputDigest);
    let other = Dataset::parse(&repin(manifest(&suite), &crlf), &crlf, Limits::default()).unwrap();
    let original = Dataset::parse(&suite, &input, Limits::default()).unwrap();
    assert_eq!(original.rows(), other.rows());
    assert_ne!(original.bindings(&key(), "scope").unwrap().input, other.bindings(&key(), "scope").unwrap().input);
}

#[test]
fn complete_membership_and_unit_identity_are_checked_across_all_three_splits() {
    let (suite, input) = fixture(5, 5);
    let original = Dataset::parse(&suite, &input, Limits::default()).unwrap();
    let rows = original.rows().to_vec();
    for axis in 0..7 {
        let mut changed = rows.clone(); let mut definition = manifest(&suite);
        let expected = match axis {
            0 => { changed[1].id = changed[0].id.clone(); changed[1].split = changed[0].split; AssayError::DuplicateId }
            1 => { changed[1].unit_sha256 = changed[0].unit_sha256; AssayError::DuplicateUnit }
            2 => {
                let other = changed.iter().position(|r| r.split != changed[0].split).unwrap();
                changed[other].unit_sha256 = changed[0].unit_sha256;
                AssayError::DuplicateUnit
            }
            3 => { changed[0].split = Split::LockedTest; AssayError::SplitMembership }
            4 => { changed[0].id = "undeclared-id".to_owned(); AssayError::SplitMembership }
            5 => { changed.pop(); AssayError::SplitMembership }
            _ => { definition.splits.locked_test[0] = definition.splits.development[0].clone(); AssayError::SplitMembership }
        };
        let bytes = encode_rows(&changed);
        assert_eq!(refusal(&repin(definition, &bytes), &bytes), expected, "axis {axis}");
    }
}

#[test]
fn duplicate_unknown_and_escaped_duplicate_keys_are_rejected_at_every_input_boundary() {
    let (suite, input) = fixture(5, 5);
    let suite_text = std::str::from_utf8(&suite).unwrap();
    for prefix in [r#"{"schema_version":1,"#, r#"{"unknown":1,"#, r#"{"schema\u005fversion":1,"#] {
        let changed = format!("{prefix}{}", &suite_text[1..]);
        assert_eq!(refusal(changed.as_bytes(), &input), AssayError::Json);
    }
    let first_newline = input.iter().position(|&byte| byte == b'\n').unwrap();
    let line = std::str::from_utf8(&input[..first_newline]).unwrap();
    for prefix in [r#"{"schema_version":1,"#, r#"{"unknown":1,"#, r#"{"schema\u005fversion":1,"#] {
        let mut changed = format!("{prefix}{}\n", &line[1..]).into_bytes();
        changed.extend_from_slice(&input[first_newline + 1..]);
        assert_eq!(refusal(&repin(manifest(&suite), &changed), &changed), AssayError::Json);
    }
    let mut value = crate::canonjson::parse_str(suite_text).unwrap();
    value["execution_identity"]["unknown"] = serde_json::json!(true);
    let changed = serde_json::to_vec(&value).unwrap();
    assert_eq!(refusal(&changed, &input), AssayError::Json);
}

#[test]
fn invalid_id_forms_probabilities_schemas_and_private_control_text_refuse() {
    let (suite, input) = fixture(5, 5);
    let original = Dataset::parse(&suite, &input, Limits::default()).unwrap();
    for id in [" leading", "trailing ", "embedded space", "two\tparts", "line\nbreak", "\u{00a0}id", ""] {
        let mut definition = manifest(&suite);
        definition.splits.development[0] = id.to_owned();
        assert_eq!(refusal(&crate::canonjson::canonical_bytes(&definition).unwrap(), &input), AssayError::SplitMembership);
    }
    for probability in [-0.1, 1.1] {
        let mut rows = original.rows().to_vec(); rows[0].probability = probability;
        let bytes = encode_rows(&rows);
        assert_eq!(refusal(&repin(manifest(&suite), &bytes), &bytes), AssayError::Observation);
    }
    let mut rows = original.rows().to_vec(); rows[0].schema_version = 2;
    let bytes = encode_rows(&rows);
    assert_eq!(refusal(&repin(manifest(&suite), &bytes), &bytes), AssayError::Observation);
    let mut definition = manifest(&suite); definition.named_population = "private\nlog-injection".to_owned();
    assert_eq!(refusal(&crate::canonjson::canonical_bytes(&definition).unwrap(), &input), AssayError::Manifest);
    definition = manifest(&suite); definition.execution_identity.task_spec = "private\nlog-injection".to_owned();
    assert_eq!(refusal(&crate::canonjson::canonical_bytes(&definition).unwrap(), &input), AssayError::Identity);
}

#[test]
fn canonical_row_order_makes_bootstrap_replay_independent_of_file_order() {
    let (suite, input) = fixture(5, 12);
    let original = Dataset::parse(&suite, &input, Limits::default()).unwrap();
    let mut rows = original.rows().to_vec(); rows.reverse();
    let reversed = encode_rows(&rows);
    let reordered = Dataset::parse(&repin(manifest(&suite), &reversed), &reversed, Limits::default()).unwrap();
    assert_eq!(original.rows(), reordered.rows());
    assert!(original.rows().windows(2).all(|pair| pair[0].id < pair[1].id));
    assert_eq!(measure(&original, options(), Limits::default(), Ok).unwrap(),
        measure(&reordered, options(), Limits::default(), Ok).unwrap());
}

#[test]
fn locked_test_metrics_match_hand_arithmetic_and_positive_event_acceptance() {
    let (suite, input) = fixture(5, 5);
    let original = Dataset::parse(&suite, &input, Limits::default()).unwrap();
    let mut rows = original.rows().to_vec();
    let supplied = [(0.9, true), (0.8, false), (0.7, true), (0.2, false), (0.1, true)];
    for (row, (probability, positive)) in rows.iter_mut().filter(|r| r.split == Split::LockedTest).zip(supplied) {
        row.probability = probability; row.positive = positive;
    }
    let bytes = encode_rows(&rows);
    let changed = Dataset::parse(&repin(manifest(&suite), &bytes), &bytes, Limits::default()).unwrap();
    let Measurement::Measured { metrics } = measure(&changed, options(), Limits::default(), Ok).unwrap()
        else { panic!("sufficient locked test required"); };
    assert!((metrics.ece - 0.46).abs() < 1e-12);
    assert!((metrics.brier - 0.318).abs() < 1e-12);
    assert_eq!((metrics.rows, metrics.accepted, metrics.abstained), (5, 2, 3));
    assert_eq!(metrics.selective_risk, Some(0.5));
    assert!(metrics.ece_interval.lower.is_finite() && metrics.ece_interval.upper.is_finite());
    assert!(metrics.brier_interval.lower <= metrics.brier_interval.upper);
}

#[test]
fn tiny_samples_are_successful_typed_refusals_without_metrics_or_transform_calls() {
    let (suite, input) = fixture(5, 4);
    let tiny = Dataset::parse(&suite, &input, Limits::default()).unwrap();
    let result = measure(&tiny, options(), Limits::default(), |_| panic!("tiny set must not be evaluated")).unwrap();
    assert_eq!(result, Measurement::InsufficientData { split: Split::LockedTest, rows: 4, minimum_samples: 5 });
    let report = evaluate(&tiny, &key(), "scope", options(), Limits::default()).unwrap();
    let bytes = render(&report, Limits::default()).unwrap();
    let value = crate::canonjson::parse_str(std::str::from_utf8(&bytes).unwrap()).unwrap();
    assert_eq!(value["measurement"]["status"], "insufficient_data");
    assert!(value["measurement"].get("metrics").is_none());
    assert_eq!(value["activation_authority"], false);
    assert_eq!(value["evidence"], "provided_observations");
}

#[test]
fn no_accepted_rows_produce_no_fabricated_selective_risk_or_interval() {
    let (suite, input) = fixture(5, 5);
    let original = Dataset::parse(&suite, &input, Limits::default()).unwrap();
    let mut rows = original.rows().to_vec();
    for row in &mut rows { if row.split == Split::LockedTest { row.probability = 0.1; } }
    let bytes = encode_rows(&rows);
    let changed = Dataset::parse(&repin(manifest(&suite), &bytes), &bytes, Limits::default()).unwrap();
    let Measurement::Measured { metrics } = measure(&changed, options(), Limits::default(), Ok).unwrap()
        else { panic!("measured"); };
    assert_eq!(metrics.accepted, 0); assert_eq!(metrics.abstained, 5);
    assert_eq!(metrics.selective_risk, None); assert_eq!(metrics.selective_risk_interval, None);
}

#[test]
fn reports_export_only_keyed_private_bindings_and_scope_changes_invalidate_them() {
    let data = dataset();
    let report = evaluate(&data, &key(), "scope-a", options(), Limits::default()).unwrap();
    let metadata = &report.bindings.metadata;
    assert_eq!(metadata.dataset_id, "owned-suite");
    assert_eq!(metadata.dataset_version, "v1");
    assert_eq!(metadata.named_population, data.manifest().named_population);
    assert_eq!(metadata.positive_label, data.manifest().positive_label);
    assert_eq!(metadata.negative_label, data.manifest().negative_label);
    assert_eq!(metadata.task_spec, "classify-v1");
    assert_eq!(metadata.model_recipe_id, "fixture-g128");
    assert_eq!(metadata.prompt_template_sha256, data.manifest().execution_identity.template_digest);
    assert_eq!(metadata.numerics_profile, "strict-quantized-v2");
    assert_eq!(metadata.thinking_mode, "disabled");
    let exported = String::from_utf8(render(&report, Limits::default()).unwrap()).unwrap();
    let wire = crate::canonjson::parse_str(&exported).unwrap();
    assert_eq!(wire["bindings"]["metadata"]["model_recipe_id"], "fixture-g128");
    assert_eq!(wire["bindings"]["metadata"]["prompt_template_sha256"],
        data.manifest().execution_identity.template_digest.to_hex());
    for row in data.rows() {
        assert!(!exported.contains(&row.id));
        assert!(!exported.contains(&row.unit_sha256.to_hex()));
    }
    for digest in [data.manifest().expected_input_sha256, data.manifest().execution_identity.prompt_digest,
        data.manifest().execution_identity.schema_digest, data.manifest().execution_identity.taskir_digest,
        data.partition().split_digests().locked_test] {
        assert!(!exported.contains(&digest.to_hex()));
    }
    assert!(!exported.contains("execution_identity_schema"));
    assert!(!exported.contains("private-prompt"));
    let alternate_key = CommitmentKey::new("assay-tests", [8; 32]).unwrap();
    assert_ne!(report.bindings, data.bindings(&alternate_key, "scope-a").unwrap());
    assert_ne!(report.bindings, data.bindings(&key(), "scope-b").unwrap());
    assert_eq!(report.bindings, data.bindings(&key(), "scope-a").unwrap());
}

#[test]
fn input_output_and_bootstrap_limits_refuse_before_large_work_or_publication() {
    let (suite, input) = fixture(5, 5);
    let data = Dataset::parse(&suite, &input, Limits::default()).unwrap();
    let mut limits = Limits::default(); limits.max_rows = 14;
    assert_eq!(Dataset::parse(&suite, &input, limits).err(), Some(AssayError::InputLimit));
    assert_eq!(data.validate_limits(limits), Err(AssayError::InputLimit));
    limits = Limits::default(); limits.max_manifest_bytes = suite.len() - 1;
    assert_eq!(Dataset::parse(&suite, &input, limits).err(), Some(AssayError::InputLimit));
    limits = Limits::default(); limits.max_line_bytes = 1;
    assert_eq!(Dataset::parse(&suite, &input, limits).err(), Some(AssayError::InputLimit));
    limits = Limits::default(); limits.max_bootstrap_draws = 159;
    assert_eq!(measure(&data, options(), limits, |_| panic!("work refused before metric allocation")), Err(AssayError::InputLimit));
    limits.max_bootstrap_draws = 160;
    assert!(measure(&data, options(), limits, Ok).is_ok());
    let report = evaluate(&data, &key(), "scope", options(), Limits::default()).unwrap();
    let encoded = render(&report, Limits::default()).unwrap();
    limits = Limits::default(); limits.max_output_bytes = encoded.len();
    assert_eq!(render(&report, limits).unwrap(), encoded);
    limits.max_output_bytes -= 1; assert_eq!(render(&report, limits), Err(AssayError::Output));
    assert_eq!(encoded.last(), Some(&b'\n'));
    assert_eq!(encoded.iter().filter(|&&byte| byte == b'\n').count(), 1);
}

#[test]
fn invalid_options_and_nonfinite_output_are_sanitized_and_never_succeed() {
    for axis in 0..8 {
        let mut options = options();
        match axis {
            0 => options.minimum_samples = 4,
            1 => options.acceptance_threshold = f64::NAN,
            2 => options.acceptance_threshold = 1.1,
            3 => options.bootstrap.resamples = 1,
            4 => options.bootstrap.resamples = 10_001,
            5 => options.bootstrap.confidence_level = f64::INFINITY,
            6 => options.bootstrap.confidence_level = 1.0,
            _ => options.bootstrap.confidence_level = 0.0,
        }
        assert_eq!(options.validate(Limits::default()), Err(AssayError::Options));
    }
    #[derive(Serialize)]
    struct NonFinite { score: f64 }
    assert_eq!(render(&NonFinite { score: f64::NAN }, Limits::default()), Err(AssayError::Output));
    assert_eq!(AssayError::InputDigest.exit_code(), ErrorCode::ArtifactIntegrityOrFormatOrVersion);
    assert_eq!(AssayError::InputLimit.exit_code(), ErrorCode::AdmissionOrResourceLimit);
    assert_eq!(AssayError::Json.to_string(), "invalid_json");
}
