//! Owned observation fixtures; no native-model or calibration-quality claim.
use super::*;
use crate::assay::{Observation, SuiteManifest, tests as fixtures};

const NAMESPACE: &str = "calibration-fixture";

fn fit_options(method: FitMethod, shift_policy: ShiftPolicy) -> FitOptions {
    FitOptions {
        method,
        validity: ValidityWindow::new(parse_date("2026-07-01").unwrap(), parse_date("2026-07-31").unwrap()).unwrap(),
        shift_policy,
    }
}

fn current() -> ApplyOptions {
    ApplyOptions { observed_date: parse_date("2026-07-15").unwrap(), shift: ShiftAssessment::InDistribution }
}

fn fitted(dataset: &Dataset, method: FitMethod, policy: ShiftPolicy) -> FitOutcome {
    fit(dataset, &fixtures::key(), NAMESPACE, fit_options(method, policy),
        fixtures::options(), Limits::default()).unwrap()
}

fn encoded(report: &FitOutcome) -> Vec<u8> {
    crate::assay::render(report, Limits::default()).unwrap()
}

fn apply_current(bytes: &[u8], dataset: &Dataset) -> Result<ApplyOutcome, AssayError> {
    apply(bytes, dataset, &fixtures::key(), NAMESPACE, current(), fixtures::options(), Limits::default())
}

fn parts(report: &FitOutcome) -> (&FittedArtifact, &Measurement) {
    let FitOutcome::Fitted { artifact, measurement, .. } = report else { panic!("fit required"); };
    (artifact, measurement)
}

fn rebuild(dataset: &Dataset, rows: &[Observation], change: impl FnOnce(&mut SuiteManifest)) -> Dataset {
    let input = fixtures::encode_rows(rows);
    let mut manifest = dataset.manifest().clone();
    manifest.expected_input_sha256 = Sha256Digest::of_bytes(&input);
    change(&mut manifest);
    Dataset::parse(&canonjson::canonical_bytes(&manifest).unwrap(), &input, Limits::default()).unwrap()
}

/// Recompute authentic commitments to exercise structural checks independently
/// of MAC corruption. The fixture owns this key; no caller gets a trusted model
/// merely by deserializing a report.
fn reseal(report: &mut FitOutcome, dataset: &Dataset) {
    let key = fixtures::key();
    let FitOutcome::Fitted { artifact, options, measurement, report_commitment: seal } = report
        else { panic!("fit required"); };
    let body = &mut artifact.body;
    body.parameter_commitment = commit(&key, NAMESPACE, "assay_calibration_parameters_v1", &body.parameters).unwrap();
    let validity = ValidityWindow::new(parse_date(&body.valid_from).unwrap(), parse_date(&body.valid_through).unwrap()).unwrap();
    body.calibration_subject = subject(dataset, &body.parameters, validity, body.shift_policy.into(), &key, NAMESPACE).unwrap();
    artifact.seal = commit(&key, NAMESPACE, "assay_calibration_artifact_v1", &artifact.body).unwrap();
    *seal = report_commitment(artifact, *options, measurement, &key, NAMESPACE, Limits::default()).unwrap();
}

#[test]
fn complete_fit_persist_reload_recomputes_the_same_independent_measurements() {
    let dataset = fixtures::dataset();
    for method in [FitMethod::Temperature, FitMethod::Isotonic] {
        let report = fitted(&dataset, method, ShiftPolicy::ConservativeAbstain);
        let (artifact, measurement) = parts(&report);
        let bytes = encoded(&report);
        let parsed = parse_report(&bytes, Limits::default()).unwrap();
        assert_eq!(encoded(&parsed), bytes);
        for date in ["2026-07-01", "2026-07-15", "2026-07-31"] {
            let result = apply(&bytes, &dataset, &fixtures::key(), NAMESPACE,
                ApplyOptions { observed_date: parse_date(date).unwrap(), shift: ShiftAssessment::InDistribution },
                fixtures::options(), Limits::default()).unwrap();
            let ApplyOutcome::Calibrated { bindings, calibration, options, measurement: reloaded } = result
                else { panic!("validity boundaries are inclusive"); };
            assert_eq!(bindings, dataset.bindings(&fixtures::key(), NAMESPACE).unwrap());
            assert_eq!(calibration, artifact.seal);
            assert_eq!(options, fixtures::options());
            assert_eq!(&reloaded, measurement);
        }
        let model = artifact.body.parameters.restore(dataset.partition().calibration().len(), Limits::default()).unwrap();
        assert!(dataset.rows().iter().all(|row| model.transform(row.probability).unwrap().is_finite()));
    }
}

#[test]
fn fitted_float_parameters_survive_the_canonical_authority_boundary_bit_exactly() {
    // Any parser rounding changes the authenticated material even without an
    // attacker. These cover subnormal, decimal, bounded-fit and wide exponents.
    for temperature in [
        f64::from_bits(1), f64::MIN_POSITIVE, 0.05, 0.1, 1.0 / 3.0,
        1.234_567_890_123_456_7, 19.999_999_999_999_996, f64::MAX,
    ] {
        let original = Parameters::Temperature { fitted_rows: 8, temperature };
        let bytes = canonjson::canonical_bytes(&original).unwrap();
        let value = canonjson::parse_str(std::str::from_utf8(&bytes).unwrap()).unwrap();
        let recovered: Parameters = serde_json::from_value(value).unwrap();
        let Parameters::Temperature { temperature: recovered, .. } = recovered else { panic!("method changed"); };
        assert_eq!(recovered.to_bits(), temperature.to_bits());
        TemperatureModel::from_parameters(recovered, 8).unwrap();
    }
}

#[test]
fn validated_restoration_preserves_fitted_transforms_and_refuses_invalid_knots() {
    let dataset = fixtures::dataset();
    let calibration = dataset.partition().calibration();
    let temperature = TemperatureModel::fit(calibration).unwrap();
    let restored_temperature = TemperatureModel::from_parameters(
        temperature.temperature(), temperature.fitted_rows()).unwrap();
    let isotonic = IsotonicModel::fit(calibration);
    let restored_isotonic = IsotonicModel::from_blocks(isotonic.blocks().to_vec()).unwrap();
    for point in 0..=100 {
        let probability = f64::from(point) / 100.0;
        assert_eq!(temperature.calibrate(probability).unwrap().to_bits(),
            restored_temperature.calibrate(probability).unwrap().to_bits());
        assert_eq!(isotonic.calibrate(probability).unwrap().to_bits(),
            restored_isotonic.calibrate(probability).unwrap().to_bits());
    }
    assert!(TemperatureModel::from_parameters(1.0, 0).is_err());
    let knot = |upper_score, calibrated_probability| IsotonicBlock { upper_score, calibrated_probability };
    for blocks in [
        vec![],
        vec![knot(-0.1, 0.5)],
        vec![knot(0.5, 1.1)],
        vec![knot(0.5, 0.2), knot(0.5, 0.8)],
        vec![knot(0.8, 0.2), knot(0.2, 0.8)],
        vec![knot(0.2, 0.8), knot(0.8, 0.2)],
    ] {
        assert!(IsotonicModel::from_blocks(blocks).is_err());
    }
    let plateau = IsotonicModel::from_blocks(vec![knot(0.2, 0.5), knot(0.8, 0.5)]).unwrap();
    assert_eq!(plateau.calibrate(0.0).unwrap(), 0.5);
    assert_eq!(plateau.calibrate(1.0).unwrap(), 0.5);
}

#[test]
fn only_calibration_rows_affect_fit_and_row_order_does_not_affect_arithmetic() {
    let dataset = fixtures::dataset();
    let mut changed = dataset.rows().to_vec();
    for row in &mut changed {
        if row.split != Split::Calibration {
            row.probability = 1.0 - row.probability;
            row.positive = true;
        }
    }
    let changed = rebuild(&dataset, &changed, |_| {});
    let mut reversed = dataset.rows().to_vec();
    reversed.reverse();
    let reversed = rebuild(&dataset, &reversed, |_| {});
    for method in [FitMethod::Temperature, FitMethod::Isotonic] {
        let original = fitted(&dataset, method, ShiftPolicy::ConservativeAbstain);
        let new_labels = fitted(&changed, method, ShiftPolicy::ConservativeAbstain);
        let reordered = fitted(&reversed, method, ShiftPolicy::ConservativeAbstain);
        let (a, am) = parts(&original);
        let (b, bm) = parts(&new_labels);
        let (c, cm) = parts(&reordered);
        assert_eq!(canonjson::canonical_bytes(&a.body.parameters).unwrap(), canonjson::canonical_bytes(&b.body.parameters).unwrap());
        assert_eq!(canonjson::canonical_bytes(&a.body.parameters).unwrap(), canonjson::canonical_bytes(&c.body.parameters).unwrap());
        assert_ne!(am, bm);
        assert_eq!(am, cm);
        // Exact input-byte commitments still distinguish different supplied
        // files, although their deterministic metric arithmetic is identical.
        assert_ne!(a.body.dataset.input, c.body.dataset.input);
    }
}

#[test]
fn reports_export_keyed_bindings_without_private_ids_or_unkeyed_hashes() {
    let dataset = fixtures::dataset();
    let report = fitted(&dataset, FitMethod::Isotonic, ShiftPolicy::ConservativeAbstain);
    let text = String::from_utf8(encoded(&report)).unwrap();
    assert!(!text.contains("private-calibration-"));
    assert!(!text.contains("private-locked_test-"));
    assert!(!text.contains("source_revision"));
    for digest in [
        dataset.manifest().expected_input_sha256,
        dataset.manifest().execution_identity.prompt_digest,
        dataset.manifest().execution_identity.taskir_digest,
        dataset.partition().split_digests().calibration,
        dataset.partition().split_digests().locked_test,
        dataset.rows()[0].unit_sha256,
    ] {
        assert!(!text.contains(&digest.to_hex()));
    }
    assert!(text.contains("\"scope\":\"supplied_observations_only\""));
    assert!(text.contains("\"hmac_sha256\""));
    assert!(text.contains("\"options\""));
}

#[test]
fn wrong_key_namespace_identity_population_labels_data_or_split_cannot_reuse_a_fit() {
    let dataset = fixtures::dataset();
    let bytes = encoded(&fitted(&dataset, FitMethod::Temperature, ShiftPolicy::ConservativeAbstain));
    for key in [
        CommitmentKey::new("assay-tests", [8; 32]).unwrap(),
        CommitmentKey::new("another-key", [7; 32]).unwrap(),
    ] {
        assert_eq!(apply(&bytes, &dataset, &key, NAMESPACE, current(), fixtures::options(), Limits::default()).unwrap_err(),
            AssayError::CalibrationBindingMismatch);
    }
    assert_eq!(apply(&bytes, &dataset, &fixtures::key(), "another-namespace", current(),
        fixtures::options(), Limits::default()).unwrap_err(), AssayError::CalibrationBindingMismatch);
    for axis in 0..6 {
        let mut rows = dataset.rows().to_vec();
        let other = match axis {
            0 => rebuild(&dataset, &rows, |suite| suite.execution_identity.logical_model_digest = Sha256Digest::of_bytes(b"other model")),
            1 => rebuild(&dataset, &rows, |suite| suite.named_population = "another named population".to_owned()),
            2 => rebuild(&dataset, &rows, |suite| suite.positive_label = "another positive event".to_owned()),
            3 => {
                rows.iter_mut().find(|row| row.split == Split::LockedTest).unwrap().positive = false;
                rebuild(&dataset, &rows, |_| {})
            }
            4 => {
                let calibration = rows.iter().position(|row| row.split == Split::Calibration).unwrap();
                let locked = rows.iter().position(|row| row.split == Split::LockedTest).unwrap();
                rows[calibration].split = Split::LockedTest;
                rows[locked].split = Split::Calibration;
                rebuild(&dataset, &rows, |suite| {
                    suite.splits.calibration[0] = rows[locked].id.clone();
                    suite.splits.locked_test[0] = rows[calibration].id.clone();
                })
            }
            _ => rebuild(&dataset, &rows, |suite| suite.execution_identity.taskir_digest = Sha256Digest::of_bytes(b"other task")),
        };
        assert_eq!(apply_current(&bytes, &other).unwrap_err(), AssayError::CalibrationBindingMismatch, "axis {axis}");
    }
}

#[test]
fn authentication_covers_parameters_original_metrics_and_bootstrap_options() {
    let dataset = fixtures::dataset();
    let report = fitted(&dataset, FitMethod::Temperature, ShiftPolicy::ConservativeAbstain);
    for axis in 0..7 {
        let mut changed = report.clone();
        let FitOutcome::Fitted { artifact, options, measurement, report_commitment } = &mut changed
            else { panic!("fit required"); };
        match axis {
            0 => {
                let Parameters::Temperature { temperature, .. } = &mut artifact.body.parameters else { panic!("temperature"); };
                *temperature *= 2.0;
            }
            1 => artifact.body.valid_through = "2027-07-31".to_owned(),
            2 => artifact.body.shift_policy = PersistedShiftPolicy::RawScoresUncalibrated,
            3 => options.bootstrap.seed += 1,
            4 => {
                let Measurement::Measured { metrics } = measurement else { panic!("metrics"); };
                metrics.brier += 0.01;
            }
            5 => artifact.body.parameter_commitment.hmac_sha256 = Sha256Digest::of_bytes(b"replaced"),
            _ => report_commitment.hmac_sha256 = Sha256Digest::of_bytes(b"replaced report"),
        }
        assert_eq!(apply_current(&encoded(&changed), &dataset).unwrap_err(), AssayError::CalibrationBindingMismatch);
        // Expiry cannot turn a corrupt report into a successful abstention.
        assert_eq!(apply(&encoded(&changed), &dataset, &fixtures::key(), NAMESPACE,
            ApplyOptions { observed_date: parse_date("2028-01-01").unwrap(), shift: ShiftAssessment::InDistribution },
            fixtures::options(), Limits::default()).unwrap_err(), AssayError::CalibrationBindingMismatch);
    }
}

#[test]
fn malformed_and_authentically_sealed_invalid_models_are_rejected_before_application() {
    let dataset = fixtures::dataset();
    for method in [FitMethod::Temperature, FitMethod::Isotonic] {
        let report = fitted(&dataset, method, ShiftPolicy::ConservativeAbstain);
        for axis in 0..6 {
            let mut changed = report.clone();
            let FitOutcome::Fitted { artifact, .. } = &mut changed else { panic!("fit required"); };
            match &mut artifact.body.parameters {
                Parameters::Temperature { fitted_rows, temperature } => match axis {
                    0 => *temperature = 0.0,
                    1 => *temperature = -1.0,
                    2 => *fitted_rows = 0,
                    3 => *fitted_rows += 1,
                    4 => *fitted_rows = Limits::default().max_rows + 1,
                    _ => *fitted_rows -= 1,
                },
                Parameters::Isotonic { fitted_rows, knots } => match axis {
                    0 => knots.clear(),
                    1 => knots[0].calibrated_probability = -0.1,
                    2 => knots[0].upper_score = 1.1,
                    3 => knots[1].upper_score = knots[0].upper_score,
                    4 => {
                        knots[0].calibrated_probability = 1.0;
                        knots[1].calibrated_probability = 0.0;
                    }
                    _ => *fitted_rows = 0,
                },
            }
            reseal(&mut changed, &dataset);
            assert_eq!(apply_current(&encoded(&changed), &dataset).unwrap_err(), AssayError::InvalidCalibration);
        }
    }
    for temperature in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(TemperatureModel::from_parameters(temperature, 8).is_err());
    }
    for probability in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(IsotonicModel::from_blocks(vec![IsotonicBlock {
            upper_score: probability, calibrated_probability: 0.5,
        }]).is_err());
        assert!(IsotonicModel::from_blocks(vec![IsotonicBlock {
            upper_score: 0.5, calibrated_probability: probability,
        }]).is_err());
    }
}

#[test]
fn strict_reload_rejects_duplicate_unknown_nested_fields_versions_and_truncated_json() {
    let dataset = fixtures::dataset();
    let report = fitted(&dataset, FitMethod::Isotonic, ShiftPolicy::ConservativeAbstain);
    let bytes = encoded(&report);
    let text = std::str::from_utf8(&bytes).unwrap();
    let duplicate = format!("{{\"status\":\"fitted\",{}", &text[1..]);
    assert_eq!(apply_current(duplicate.as_bytes(), &dataset).unwrap_err(), AssayError::Json);
    assert!(apply_current(&bytes[..bytes.len() - 2], &dataset).is_err());
    for path in ["", "/artifact", "/artifact/body", "/artifact/body/parameters"] {
        let mut value = serde_json::to_value(&report).unwrap();
        value.pointer_mut(path).unwrap().as_object_mut().unwrap().insert("unknown".to_owned(), serde_json::json!(true));
        assert_eq!(apply_current(&serde_json::to_vec(&value).unwrap(), &dataset).unwrap_err(), AssayError::InvalidCalibration);
    }
    let mut unknown_version = report.clone();
    let FitOutcome::Fitted { artifact, .. } = &mut unknown_version else { panic!("fit required"); };
    artifact.body.schema_version = 2;
    reseal(&mut unknown_version, &dataset);
    assert_eq!(apply_current(&encoded(&unknown_version), &dataset).unwrap_err(), AssayError::CalibrationBindingMismatch);
}

#[test]
fn expiry_future_validity_and_shift_never_silently_keep_calibrated_status() {
    let dataset = fixtures::dataset();
    let raw = measure(&dataset, fixtures::options(), Limits::default(), Ok).unwrap();
    for policy in [ShiftPolicy::ConservativeAbstain, ShiftPolicy::RawScoresUncalibrated] {
        let report = fitted(&dataset, FitMethod::Isotonic, policy);
        let bytes = encoded(&report);
        for (date, shift, expected) in [
            ("2026-06-30", ShiftAssessment::InDistribution, InvalidationReason::NotYetValid),
            ("2026-08-01", ShiftAssessment::InDistribution, InvalidationReason::Expired),
            ("2026-07-15", ShiftAssessment::Detected { indicator: "declared-population-shift".to_owned() }, InvalidationReason::DistributionShift),
        ] {
            let result = apply(&bytes, &dataset, &fixtures::key(), NAMESPACE,
                ApplyOptions { observed_date: parse_date(date).unwrap(), shift },
                fixtures::options(), Limits::default()).unwrap();
            match (policy, result) {
                (ShiftPolicy::ConservativeAbstain, ApplyOutcome::Abstained { reason, .. }) => assert_eq!(reason, expected),
                (ShiftPolicy::RawScoresUncalibrated, ApplyOutcome::Uncalibrated { reason, measurement, .. }) => {
                    assert_eq!(reason, expected);
                    assert_eq!(measurement, raw);
                }
                _ => panic!("invalidated calibration must follow its declared policy"),
            }
        }
        for indicator in ["", "bad\nindicator", "bad\u{2028}indicator", "bad\u{2029}indicator"] {
            assert_eq!(apply(&bytes, &dataset, &fixtures::key(), NAMESPACE,
                ApplyOptions { observed_date: parse_date("2026-07-15").unwrap(),
                    shift: ShiftAssessment::Detected { indicator: indicator.to_owned() } },
                fixtures::options(), Limits::default()).unwrap_err(), AssayError::Options);
        }
    }
}

#[test]
fn tiny_splits_and_tightened_bounds_return_no_artifact_or_metrics() {
    fn scoped<T: Serialize>(report: &T, dataset: &Dataset, options: EvalOptions) {
        let value = serde_json::to_value(report).unwrap();
        assert_eq!(value["bindings"], serde_json::to_value(dataset.bindings(&fixtures::key(), NAMESPACE).unwrap()).unwrap());
        assert_eq!(value["options"], serde_json::to_value(options).unwrap());
        assert_eq!(value["bindings"]["metadata"]["dataset_id"], dataset.manifest().dataset_id);
        assert!(value.get("artifact").is_none());
        assert!(value.get("measurement").is_none());
        let text = serde_json::to_string(&value).unwrap();
        assert!(!text.contains("private-calibration-"));
        assert!(!text.contains("private-locked_test-"));
        assert!(text.contains("\"hmac_sha256\""));
        for digest in [
            dataset.manifest().expected_input_sha256,
            dataset.manifest().execution_identity.prompt_digest,
            dataset.manifest().execution_identity.taskir_digest,
            dataset.partition().split_digests().calibration,
            dataset.partition().split_digests().locked_test,
            dataset.rows()[0].unit_sha256,
        ] {
            assert!(!text.contains(&digest.to_hex()));
        }
    }
    for (calibration_rows, locked_rows, expected_split) in [
        (4, 8, Split::Calibration), (8, 4, Split::LockedTest), (4, 4, Split::Calibration),
    ] {
        let (manifest, input) = fixtures::fixture(calibration_rows, locked_rows);
        let dataset = Dataset::parse(&manifest, &input, Limits::default()).unwrap();
        let report = fitted(&dataset, FitMethod::Isotonic, ShiftPolicy::ConservativeAbstain);
        assert!(matches!(report, FitOutcome::InsufficientData { split, rows: 4, minimum_samples: 5, .. } if split == expected_split));
        scoped(&report, &dataset, fixtures::options());
        assert_eq!(apply_current(&encoded(&report), &dataset).unwrap_err(), AssayError::InvalidCalibration);
        let too_small = Limits { max_output_bytes: 2, ..Limits::default() };
        assert_eq!(fit(&dataset, &fixtures::key(), NAMESPACE,
            fit_options(FitMethod::Isotonic, ShiftPolicy::ConservativeAbstain),
            fixtures::options(), too_small).unwrap_err(), AssayError::Output);
    }
    let dataset = fixtures::dataset();
    for limits in [
        Limits { max_rows: 10, ..Limits::default() },
        Limits { max_input_bytes: 128, max_line_bytes: 128, ..Limits::default() },
        Limits { max_bootstrap_draws: 1, ..Limits::default() },
    ] {
        assert_eq!(fit(&dataset, &fixtures::key(), NAMESPACE,
            fit_options(FitMethod::Temperature, ShiftPolicy::ConservativeAbstain),
            fixtures::options(), limits).unwrap_err(), AssayError::InputLimit);
    }
    let bytes = encoded(&fitted(&dataset, FitMethod::Temperature, ShiftPolicy::ConservativeAbstain));
    let higher_minimum = EvalOptions { minimum_samples: 9, ..fixtures::options() };
    let report = apply(&bytes, &dataset, &fixtures::key(), NAMESPACE,
        current(), higher_minimum, Limits::default()).unwrap();
    assert!(matches!(report,
        ApplyOutcome::InsufficientData { split: Split::Calibration, rows: 8, minimum_samples: 9, .. }));
    scoped(&report, &dataset, higher_minimum);
    let too_small = Limits { max_output_bytes: bytes.len() - 1, ..Limits::default() };
    assert_eq!(apply(&bytes, &dataset, &fixtures::key(), NAMESPACE,
        current(), fixtures::options(), too_small).unwrap_err(), AssayError::InputLimit);
}

#[test]
fn dates_are_canonical_and_temperature_endpoints_are_an_explicit_fit_refusal() {
    for bad in ["2026-7-01", "2026-07-1", "2026/07/01", "0000-01-01", "2026-02-29", "2100-02-29", "2026-07-01\n", "é026-07-1"] {
        assert!(parse_date(bad).is_err());
    }
    assert_eq!(parse_date("2000-02-29").unwrap().to_string(), "2000-02-29");
    let dataset = fixtures::dataset();
    let mut wide_year = fit_options(FitMethod::Isotonic, ShiftPolicy::ConservativeAbstain);
    wide_year.validity = ValidityWindow::new(
        ValidityDate::new(10_000, 1, 1).unwrap(), ValidityDate::new(10_000, 12, 31).unwrap()).unwrap();
    assert_eq!(fit(&dataset, &fixtures::key(), NAMESPACE, wide_year,
        fixtures::options(), Limits::default()).unwrap_err(), AssayError::Options);
    let mut rows = dataset.rows().to_vec();
    rows.iter_mut().find(|row| row.split == Split::Calibration).unwrap().probability = 0.0;
    let endpoint = rebuild(&dataset, &rows, |_| {});
    assert_eq!(fit(&endpoint, &fixtures::key(), NAMESPACE,
        fit_options(FitMethod::Temperature, ShiftPolicy::ConservativeAbstain),
        fixtures::options(), Limits::default()).unwrap_err(), AssayError::Calibration);
    assert!(matches!(fitted(&endpoint, FitMethod::Isotonic, ShiftPolicy::ConservativeAbstain), FitOutcome::Fitted { .. }));
}
