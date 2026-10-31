//! Model-free arithmetic and authority fixtures through the private test-only
//! replay builder. Actual model execution remains the separate ignored fixture.
use super::*;
use super::super::{replay::ReplayBuilder, tests as native, Row};
use crate::{calibration::{ShiftAssessment, ShiftPolicy}, assay::qualify::Decision};

const CALIBRATION: [f64; 6] = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9];
const LOCKED: [f64; 6] = [0.8, 0.2, 0.4, 0.6, 0.9, 0.1];
fn dataset() -> Dataset {
    let (mut manifest, mut rows) = native::wire(&[true, false, true, false, true, false], Default::default());
    let gold = [false, false, true, false, true, true];
    for (index, &positive) in gold.iter().enumerate().skip(1) {
        let id = format!("private-calibration-{index:03}"); manifest.splits.calibration.push(id.clone());
        rows.push(Row { schema_version: 1, id, split: Split::Calibration,
            text: format!("PRIVATE_EXTRA_CALIBRATION_{index} é"), positive });
    }
    let (manifest, input) = native::encode(manifest, &rows);
    Dataset::parse(&manifest, &input, Limits::default()).unwrap()
}
fn replay(data: &Dataset, calibration: &[f64], locked: &[f64], subject: &str) -> NativeReplay {
    let mut builder = ReplayBuilder::new(data, Limits::default()).unwrap();
    let mut calibration_index = 0; let mut locked_index = 0;
    for (index, row) in data.rows().iter().enumerate() {
        let probability = match row.split {
            Split::Development => 0.5,
            Split::Calibration => { let value = calibration[calibration_index]; calibration_index += 1; value },
            Split::LockedTest => { let value = locked[locked_index]; locked_index += 1; value },
        };
        builder.accept_fixture(index, &native::identity(row, subject), native::budget(), native::work(128), 128,
            native::run(data.manifest(), probability, 128)).unwrap();
    }
    builder.finish().unwrap()
}
fn fit_options(method: FitMethod, shift_policy: ShiftPolicy) -> FitOptions {
    FitOptions { method, shift_policy, validity: ValidityWindow::new(
        crate::assay::calibrate::parse_date("2026-10-01").unwrap(),
        crate::assay::calibrate::parse_date("2026-10-31").unwrap()).unwrap() }
}
fn apply_options() -> ApplyOptions {
    ApplyOptions { observed_date: crate::assay::calibrate::parse_date("2026-10-15").unwrap(),
        shift: ShiftAssessment::InDistribution }
}
fn fit(replay: &NativeReplay, method: FitMethod, policy: ShiftPolicy) -> FitOutcome {
    fit_with_checkpoint::<AssayError>(replay, &native::key(), "scope", fit_options(method, policy),
        native::options(), Limits::default(), &mut || Ok(())).unwrap()
}
fn load(fitted: &FitOutcome, options: ApplyOptions) -> PreparedApplication {
    let bytes = crate::assay::render(fitted, Limits::default()).unwrap();
    prepare_application(&bytes, &native::key(), "scope", options, Limits::default()).unwrap()
}
fn apply(replay: &NativeReplay, application: &PreparedApplication) -> ApplicationOutcome {
    apply_with_checkpoint::<AssayError>(replay, application, &native::key(), "scope", native::options(),
        Limits::default(), &mut || Ok(())).unwrap()
}
fn measurement(fitted: &FitOutcome) -> &Measurement {
    let FitDecision::Fitted { measurement, .. } = &fitted.body.outcome else { panic!("expected fit") }; measurement
}
fn evaluated(applied: &ApplicationOutcome) -> &Evaluation {
    let ApplicationDecision::Evaluated { evaluation } = &applied.body.outcome else { panic!("expected evaluation") }; evaluation
}
fn parameters(fitted: &FitOutcome) -> serde_json::Value {
    serde_json::to_value(fitted).unwrap()["body"]["outcome"]["artifact"]["body"]["parameters"].clone()
}

#[test]
fn actual_replay_fit_reloads_exactly_and_retains_original_native_decisions() {
    let data = dataset(); let replay = replay(&data, &CALIBRATION, &LOCKED, "model-a");
    for method in [FitMethod::Isotonic, FitMethod::Temperature] {
        let fitted = fit(&replay, method, ShiftPolicy::RawScoresUncalibrated);
        let prepared = load(&fitted, apply_options()); prepared.validate_dataset(&data, &native::key(), "scope", Limits::default()).unwrap();
        let applied = apply(&replay, &prepared);
        assert_eq!(applied.body.application.status, ApplicationStatus::Calibrated);
        assert_eq!(&evaluated(&applied).body.measurement, measurement(&fitted));
        let raw = report::evaluate_with_checkpoint::<AssayError>(&replay, &native::key(), "scope", native::options(),
            Limits::default(), &mut || Ok(())).unwrap();
        assert_eq!(evaluated(&applied).body.classification, raw.body.classification);
        assert_eq!(evaluated(&applied).body.bindings, raw.body.bindings);
        assert!(!applied.body.activation_authority);
        if method == FitMethod::Isotonic {
            let Measurement::Measured { metrics } = measurement(&fitted) else { panic!("held-out metrics") };
            assert!((metrics.brier - 1.0 / 12.0).abs() < 1e-12);
            assert_eq!(metrics.accepted, 4); assert_eq!(metrics.selective_risk, Some(0.25));
        }
    }
}

#[test]
fn development_and_locked_test_changes_cannot_change_fitted_parameters() {
    let data = dataset(); let first = replay(&data, &CALIBRATION, &LOCKED, "model-a");
    let mut rows = data.rows().to_vec();
    for row in &mut rows { if row.split != Split::Calibration { row.positive = !row.positive; } }
    let (manifest, input) = native::encode(data.manifest().clone(), &rows);
    let changed = Dataset::parse(&manifest, &input, Limits::default()).unwrap();
    let other = replay(&changed, &CALIBRATION, &[0.3; 6], "model-a");
    for method in [FitMethod::Temperature, FitMethod::Isotonic] {
        let a = fit(&first, method, ShiftPolicy::RawScoresUncalibrated);
        let b = fit(&other, method, ShiftPolicy::RawScoresUncalibrated);
        assert_eq!(parameters(&a), parameters(&b));
        assert_ne!(serde_json::to_value(measurement(&a)).unwrap(), serde_json::to_value(measurement(&b)).unwrap());
    }
}

#[test]
fn exact_model_observation_and_source_scope_mismatches_are_hard_errors_even_for_raw_fallback() {
    let data = dataset(); let original = replay(&data, &CALIBRATION, &LOCKED, "model-a");
    let fitted = fit(&original, FitMethod::Isotonic, ShiftPolicy::RawScoresUncalibrated);
    let mut expired = apply_options(); expired.observed_date = crate::assay::calibrate::parse_date("2026-11-01").unwrap();
    let application = load(&fitted, expired);
    let mut changed_scores = LOCKED; changed_scores[5] = 0.15;
    for changed in [replay(&data, &CALIBRATION, &LOCKED, "model-b"), replay(&data, &CALIBRATION, &changed_scores, "model-a")] {
        assert_eq!(apply_with_checkpoint::<AssayError>(&changed, &application, &native::key(), "scope", native::options(),
            Limits::default(), &mut || Ok(())).err(), Some(AssayError::CalibrationBindingMismatch));
    }
    let mut rows = data.rows().to_vec();
    let last = rows.last_mut().unwrap(); last.positive = !last.positive;
    let (manifest, input) = native::encode(data.manifest().clone(), &rows);
    let changed = Dataset::parse(&manifest, &input, Limits::default()).unwrap();
    assert_eq!(application.validate_dataset(&changed, &native::key(), "scope", Limits::default()),
        Err(AssayError::CalibrationBindingMismatch));
}

fn reseal(value: &mut serde_json::Value) {
    let key = native::key(); let limits = Limits::default();
    let parameters = value["body"]["outcome"]["artifact"]["body"]["parameters"].clone();
    value["body"]["outcome"]["artifact"]["body"]["parameter_commitment"] = serde_json::to_value(
        commit(&key, "scope", "native_calibration_parameters_v1", &parameters, limits).unwrap()).unwrap();
    let artifact_body = value["body"]["outcome"]["artifact"]["body"].clone();
    value["body"]["outcome"]["artifact"]["seal"] = serde_json::to_value(
        commit(&key, "scope", "native_calibration_artifact_v1", &artifact_body, limits).unwrap()).unwrap();
    let body = value["body"].clone();
    value["seal"] = serde_json::to_value(commit(&key, "scope", "native_calibration_fit_report_v1", &body, limits).unwrap()).unwrap();
}
#[test]
fn report_artifact_and_parameter_tampering_unknown_fields_and_wrong_keys_are_rejected() {
    let data = dataset(); let replay = replay(&data, &CALIBRATION, &LOCKED, "model-a");
    let fitted = fit(&replay, FitMethod::Temperature, ShiftPolicy::RawScoresUncalibrated);
    let bytes = crate::assay::render(&fitted, Limits::default()).unwrap();
    let wrong_key = CommitmentKey::new("native-test-key", [74; 32]).unwrap();
    let correct_key = native::key();
    for (key, namespace) in [(&wrong_key, "scope"), (&correct_key, "wrong-scope")] {
        assert!(prepare_application(&bytes, key, namespace, apply_options(), Limits::default()).is_err());
    }
    let mut value = serde_json::to_value(&fitted).unwrap();
    value["body"]["outcome"]["classification"]["accuracy"] = serde_json::json!(0.123);
    assert_eq!(prepare_application(&serde_json::to_vec(&value).unwrap(), &native::key(), "scope", apply_options(),
        Limits::default()).err(), Some(AssayError::CalibrationBindingMismatch));
    let mut value = serde_json::to_value(&fitted).unwrap(); value["body"]["extra"] = serde_json::json!(true);
    assert!(prepare_application(&serde_json::to_vec(&value).unwrap(), &native::key(), "scope", apply_options(), Limits::default()).is_err());
    let mut value = serde_json::to_value(&fitted).unwrap();
    value["body"]["outcome"]["artifact"]["body"]["parameters"]["temperature"] = serde_json::json!(-1.0);
    reseal(&mut value); // A valid owner HMAC cannot make invalid numerical parameters valid.
    assert_eq!(prepare_application(&serde_json::to_vec(&value).unwrap(), &native::key(), "scope", apply_options(),
        Limits::default()).err(), Some(AssayError::InvalidCalibration));
    let duplicate = String::from_utf8(bytes).unwrap().replacen("\"schema_version\":1", "\"schema_version\":1,\"schema_version\":1", 1);
    assert_eq!(prepare_application(duplicate.as_bytes(), &native::key(), "scope", apply_options(), Limits::default()).err(), Some(AssayError::Json));
}

#[test]
fn validity_and_declared_shift_choose_raw_fallback_or_distinct_nongating_abstention() {
    let data = dataset(); let replay = replay(&data, &CALIBRATION, &LOCKED, "model-a");
    let raw = report::evaluate_with_checkpoint::<AssayError>(&replay, &native::key(), "scope", native::options(),
        Limits::default(), &mut || Ok(())).unwrap();
    for policy in [ShiftPolicy::RawScoresUncalibrated, ShiftPolicy::ConservativeAbstain] {
        let fitted = fit(&replay, FitMethod::Isotonic, policy);
        for (date, shift, reason) in [
            ("2026-09-30", ShiftAssessment::InDistribution, InvalidationReason::NotYetValid),
            ("2026-11-01", ShiftAssessment::InDistribution, InvalidationReason::Expired),
            ("2026-10-15", ShiftAssessment::Detected { indicator: "named population shifted".to_owned() }, InvalidationReason::DistributionShift),
        ] {
            let prepared = load(&fitted, ApplyOptions { observed_date: crate::assay::calibrate::parse_date(date).unwrap(), shift });
            let applied = apply(&replay, &prepared); assert_eq!(applied.body.application.reason, Some(reason));
            let paired = compare_with_checkpoint::<AssayError>(&replay, &replay, Some(&prepared), None,
                &native::key(), "scope", "host", native::gates(), native::options(), Limits::default(), &mut || Ok(())).unwrap();
            match policy {
                ShiftPolicy::RawScoresUncalibrated => {
                    assert_eq!(applied.body.application.status, ApplicationStatus::Uncalibrated);
                    assert_eq!(evaluated(&applied).body.measurement, raw.body.measurement);
                    assert!(matches!(paired.body.outcome, QualificationDecision::Compared { .. }));
                }
                ShiftPolicy::ConservativeAbstain => {
                    assert_eq!(applied.body.application.status, ApplicationStatus::Abstained);
                    assert!(matches!(applied.body.outcome, ApplicationDecision::Abstained { .. }));
                    assert!(matches!(paired.body.outcome, QualificationDecision::Abstained { .. }));
                }
            }
        }
    }
}

#[test]
fn paired_brier_deltas_use_each_effective_transform_and_artifacts_cannot_swap_arms() {
    let data = dataset(); let candidate = replay(&data, &CALIBRATION, &LOCKED, "model-a");
    let baseline = replay(&data, &CALIBRATION, &LOCKED, "model-b");
    let current = load(&fit(&candidate, FitMethod::Isotonic, ShiftPolicy::RawScoresUncalibrated), apply_options());
    let prior = load(&fit(&baseline, FitMethod::Isotonic, ShiftPolicy::RawScoresUncalibrated), apply_options());
    let mixed = compare_with_checkpoint::<AssayError>(&candidate, &baseline, Some(&current), None,
        &native::key(), "scope", "host", native::gates(), native::options(), Limits::default(), &mut || Ok(())).unwrap();
    let QualificationDecision::Compared { qualification } = mixed.body.outcome else { panic!("comparison") };
    let Decision::Passed { paired, .. } = qualification.body.decision else { panic!("permissive gates") };
    assert!((paired.mean_brier_change + 0.32 / 6.0).abs() < 1e-12); assert_eq!(paired.candidate_better, 6);
    let both = compare_with_checkpoint::<AssayError>(&candidate, &baseline, Some(&current), Some(&prior),
        &native::key(), "scope", "host", native::gates(), native::options(), Limits::default(), &mut || Ok(())).unwrap();
    let QualificationDecision::Compared { qualification } = both.body.outcome else { panic!("comparison") };
    let Decision::Passed { paired, .. } = qualification.body.decision else { panic!("permissive gates") };
    assert_eq!(paired.mean_brier_change, 0.0); assert_eq!(paired.equal, 6);
    assert_eq!(compare_with_checkpoint::<AssayError>(&candidate, &baseline, Some(&prior), Some(&current),
        &native::key(), "scope", "host", native::gates(), native::options(), Limits::default(), &mut || Ok(())).err(),
        Some(AssayError::CalibrationBindingMismatch));
}

#[test]
fn temperature_endpoints_refuse_without_clipping_and_isotonic_accepts_endpoints() {
    let data = dataset(); let mut calibration = CALIBRATION; calibration[0] = 0.0; calibration[5] = 1.0;
    let replay = replay(&data, &calibration, &LOCKED, "model-a");
    assert_eq!(fit_with_checkpoint::<AssayError>(&replay, &native::key(), "scope",
        fit_options(FitMethod::Temperature, ShiftPolicy::RawScoresUncalibrated), native::options(), Limits::default(),
        &mut || Ok(())).err(), Some(AssayError::Calibration));
    let fitted = fit(&replay, FitMethod::Isotonic, ShiftPolicy::RawScoresUncalibrated);
    assert!(matches!(fitted.body.outcome, FitDecision::Fitted { .. }));
}

#[test]
fn insufficient_calibration_never_emits_an_artifact_and_higher_eval_minimum_stays_insufficient() {
    let (manifest, rows) = native::wire(&[true, false, true, false, true, false], Default::default());
    let (manifest, input) = native::encode(manifest, &rows);
    let tiny = Dataset::parse(&manifest, &input, Limits::default()).unwrap();
    let replay = replay(&tiny, &CALIBRATION, &LOCKED, "model-a");
    let fitted = fit(&replay, FitMethod::Isotonic, ShiftPolicy::RawScoresUncalibrated);
    assert!(matches!(fitted.body.outcome, FitDecision::InsufficientData { split: Split::Calibration, rows: 1, .. }));
    assert!(prepare_application(&crate::assay::render(&fitted, Limits::default()).unwrap(), &native::key(), "scope",
        apply_options(), Limits::default()).is_err());
    let data = dataset(); let full = self::replay(&data, &CALIBRATION, &LOCKED, "model-a");
    let prepared = load(&fit(&full, FitMethod::Isotonic, ShiftPolicy::RawScoresUncalibrated), apply_options());
    let mut options = native::options(); options.minimum_samples = 20;
    let applied = apply_with_checkpoint::<AssayError>(&full, &prepared, &native::key(), "scope", options,
        Limits::default(), &mut || Ok(())).unwrap();
    assert!(matches!(evaluated(&applied).body.measurement, Measurement::InsufficientData { rows: 6, .. }));
}

#[derive(Debug, Eq, PartialEq)]
enum Stop { Requested, Assay, Calibration }
impl From<AssayError> for Stop { fn from(_: AssayError) -> Self { Self::Assay } }
impl From<CalibrationError> for Stop { fn from(_: CalibrationError) -> Self { Self::Calibration } }
#[test]
fn fitting_and_held_out_finalization_share_cancellation_through_the_last_checkpoint() {
    let data = dataset(); let replay = replay(&data, &CALIBRATION, &LOCKED, "model-a"); let mut calls = 0;
    let options = fit_options(FitMethod::Temperature, ShiftPolicy::RawScoresUncalibrated);
    fit_with_checkpoint::<Stop>(&replay, &native::key(), "scope", options, native::options(), Limits::default(),
        &mut || { calls += 1; Ok(()) }).unwrap();
    assert!(calls > 128);
    for at in [0, calls / 2, calls - 1] {
        let mut current = 0;
        let result = fit_with_checkpoint::<Stop>(&replay, &native::key(), "scope", options, native::options(), Limits::default(),
            &mut || { let turn = current; current += 1; if turn == at { Err(Stop::Requested) } else { Ok(()) } });
        assert_eq!(result.err(), Some(Stop::Requested)); assert_eq!(current, at + 1);
    }
}

#[test]
fn artifact_load_lower_limits_output_bounds_and_private_source_omission_are_enforced() {
    let data = dataset(); let replay = replay(&data, &CALIBRATION, &LOCKED, "model-a");
    let fitted = fit(&replay, FitMethod::Isotonic, ShiftPolicy::RawScoresUncalibrated);
    let bytes = crate::assay::render(&fitted, Limits::default()).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(!text.contains("PRIVATE_")); assert!(!text.contains("private-test"));
    assert!(!text.contains(&data.manifest().expected_input_sha256.to_hex()));
    let prepared = load(&fitted, apply_options());
    let mut limits = Limits::default(); limits.max_output_bytes = bytes.len() - 1;
    assert_eq!(prepared.validate_limits(limits), Err(AssayError::InputLimit));
    assert!(prepare_application(&bytes, &native::key(), "scope", apply_options(), limits).is_err());
    let mut limits = Limits::default(); limits.max_output_bytes = 64;
    assert_eq!(fit_with_checkpoint::<AssayError>(&replay, &native::key(), "scope",
        fit_options(FitMethod::Isotonic, ShiftPolicy::RawScoresUncalibrated), native::options(), limits,
        &mut || Ok(())).err(), Some(AssayError::Output));
    assert_eq!(required_calibration_bytes(Limits::default(), false, 0), Ok(0));
    assert!(required_calibration_bytes(Limits::default(), true, 0).unwrap() > 0);
    assert!(required_calibration_bytes(Limits::default(), false, 2).unwrap()
        > required_calibration_bytes(Limits::default(), false, 1).unwrap());
    assert_eq!(required_calibration_bytes(Limits::default(), true, 1), Err(AssayError::Limits));
    assert_eq!(required_calibration_bytes(Limits::default(), false, 3), Err(AssayError::Limits));
    assert_eq!(AssayError::from(CalibrationError::AllocationRefused), AssayError::Allocation);
}
