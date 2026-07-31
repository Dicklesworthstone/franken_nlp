use franken_nlp::{
    calibration::{
        BinaryLabel, CalibrationArtifact, CalibrationArtifactSpec, CalibrationError,
        CalibrationState, ConformalModel, ExchangeabilityMemo, IsotonicModel, LabeledScore,
        ShiftAssessment, ShiftPolicy, SplitMembership, SplitName, TemperatureModel, ValidityDate,
        ValidityWindow, report_locked_test,
    },
    error::StructuredTaskStatus,
    execution_identity::{
        EXECUTION_IDENTITY_SCHEMA_VERSION, ExecutionIdentity, NumericsProfile, Sha256Digest,
        ThinkingMode, ToolMode,
    },
};

fn score(id: &str, probability: f64, positive: bool) -> LabeledScore {
    LabeledScore::new(id, probability, positive).unwrap()
}

fn partition() -> franken_nlp::calibration::CalibrationPartition {
    let membership = SplitMembership::new(
        ["dev-1".to_owned()],
        [
            "cal-1".to_owned(),
            "cal-2".to_owned(),
            "cal-3".to_owned(),
            "cal-4".to_owned(),
        ],
        ["test-1".to_owned(), "test-2".to_owned()],
    )
    .unwrap();
    membership
        .partition(
            vec![score("dev-1", 0.5, true)],
            vec![
                score("cal-1", 0.9, true),
                score("cal-2", 0.8, true),
                score("cal-3", 0.2, false),
                score("cal-4", 0.1, false),
            ],
            vec![score("test-1", 0.8, true), score("test-2", 0.2, false)],
        )
        .unwrap()
}

fn digest(label: &str) -> Sha256Digest {
    Sha256Digest::of_bytes(label.as_bytes())
}

fn identity(calibration_digest: Sha256Digest) -> ExecutionIdentity {
    ExecutionIdentity::new(ExecutionIdentity {
        schema_version: EXECUTION_IDENTITY_SCHEMA_VERSION,
        source_revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        logical_model_digest: digest("model"),
        artifact_format: "fnlpq-v1".to_owned(),
        quant_recipe: "int8-symmetric-v1".to_owned(),
        packing_set_digest: digest("packing"),
        tokenizer_digest: digest("tokenizer"),
        template_digest: digest("template"),
        task_spec: "toy-binary-v1".to_owned(),
        taskir_digest: digest("taskir"),
        prompt_digest: digest("prompt"),
        grammar_compiler_version: "grammar-v1".to_owned(),
        schema_digest: digest("schema"),
        numerics_profile: NumericsProfile::HfBf16Eager,
        kv_dtype: "bf16".to_owned(),
        sampler_version: "greedy-v1".to_owned(),
        thinking_mode: ThinkingMode::Disabled,
        tool_mode: ToolMode::None,
        calibration_digest,
        decision_policy_digest: digest("decision-policy"),
        backend_semantic_version: "cpu-v1".to_owned(),
        host_class: None,
        compiler_identity: None,
    })
    .unwrap()
}

#[test]
fn partition_seals_parameter_fitting_away_from_locked_test_rows() {
    let membership = SplitMembership::new(
        ["dev-1".to_owned()],
        ["cal-1".to_owned()],
        ["test-1".to_owned()],
    )
    .unwrap();
    let leaked = membership.partition(
        vec![score("dev-1", 0.5, true)],
        vec![score("test-1", 0.9, true)],
        vec![score("test-1", 0.8, true)],
    );
    assert!(matches!(
        leaked,
        Err(CalibrationError::UnexpectedId {
            expected: SplitName::Calibration,
            ..
        })
    ));

    let partition = partition();
    let temperature = TemperatureModel::fit(partition.calibration()).unwrap();
    assert_eq!(temperature.fitted_rows(), 4);
    assert_eq!(partition.development().len(), 1);

    let (metrics, _) = report_locked_test(
        partition.locked_test(),
        |probability| temperature.calibrate(probability),
        0.5,
    )
    .unwrap();
    assert_eq!(metrics.rows, 2);
}

#[test]
fn temperature_is_explicit_and_isotonic_pav_is_monotonic() {
    let membership = SplitMembership::new(
        ["dev-1".to_owned()],
        [
            "cal-1".to_owned(),
            "cal-2".to_owned(),
            "cal-3".to_owned(),
            "cal-4".to_owned(),
        ],
        ["test-1".to_owned()],
    )
    .unwrap();
    let partition = membership
        .partition(
            vec![score("dev-1", 0.5, true)],
            vec![
                score("cal-1", 0.1, false),
                score("cal-2", 0.4, true),
                score("cal-3", 0.6, false),
                score("cal-4", 0.9, true),
            ],
            vec![score("test-1", 0.7, true)],
        )
        .unwrap();
    let isotonic = IsotonicModel::fit(partition.calibration());
    let calibrated = isotonic.blocks();
    assert_eq!(calibrated.len(), 3);
    assert!((calibrated[0].calibrated_probability - 0.0).abs() < 1e-12);
    assert!((calibrated[1].calibrated_probability - 0.5).abs() < 1e-12);
    assert!((calibrated[2].calibrated_probability - 1.0).abs() < 1e-12);
    assert!(isotonic.calibrate(0.2).unwrap() <= isotonic.calibrate(0.8).unwrap());

    let temperature = TemperatureModel::fit(partition.calibration()).unwrap();
    assert!(temperature.temperature().is_finite());
    assert!((0.0..=1.0).contains(&temperature.calibrate(0.7).unwrap()));
}

#[test]
fn ece_brier_and_selective_risk_are_locked_test_only() {
    let partition = partition();
    let (metrics, selective) =
        report_locked_test(partition.locked_test(), |probability| Ok(probability), 0.5).unwrap();
    assert!((metrics.brier - 0.04).abs() < 1e-12);
    assert!((metrics.ece - 0.2).abs() < 1e-12);
    assert_eq!(selective.accepted, 1);
    assert_eq!(selective.abstained, 1);
    assert_eq!(selective.risk, Some(0.0));
}

#[test]
fn conformal_requires_a_named_exchangeability_memo_and_scopes_coverage() {
    let partition = partition();
    assert!(matches!(
        ConformalModel::fit(partition.calibration(), None, 0.25),
        Err(CalibrationError::MissingExchangeabilityMemo)
    ));
    let memo = ExchangeabilityMemo::new(
        "repo-authored synthetic binary population",
        "one independent synthetic row",
        "Calibration and locked-test rows are treated as exchangeable only for this fixture.",
    )
    .unwrap();
    let conformal = ConformalModel::fit(partition.calibration(), Some(memo), 0.25).unwrap();
    assert_eq!(conformal.fitted_rows(), 4);
    assert!(
        conformal
            .prediction_set(0.8)
            .unwrap()
            .contains(&BinaryLabel::Positive)
    );
    let coverage = conformal
        .coverage_on_locked_test(partition.locked_test())
        .unwrap();
    assert_eq!(
        coverage.named_population,
        "repo-authored synthetic binary population"
    );
    assert_eq!(coverage.total, 2);
    assert_eq!(coverage.covered, 2);
    assert_eq!(coverage.empirical_coverage(), 1.0);
}

#[test]
fn artifact_is_identity_bound_and_shift_or_expiry_never_stays_calibrated() {
    let partition = partition();
    let validity = ValidityWindow::new(
        ValidityDate::new(2026, 7, 1).unwrap(),
        ValidityDate::new(2026, 7, 31).unwrap(),
    )
    .unwrap();
    let spec = CalibrationArtifactSpec::new(
        ["yes".to_owned(), "no".to_owned()],
        validity,
        "repo-authored synthetic binary population",
        digest("temperature-fit"),
        partition.split_digests(),
        ShiftPolicy::RawScoresUncalibrated,
    )
    .unwrap();
    let artifact = CalibrationArtifact::new(&identity(spec.digest()), spec).unwrap();
    let shifted = artifact
        .decide(
            0.9,
            0.8,
            ValidityDate::new(2026, 7, 20).unwrap(),
            ShiftAssessment::Detected {
                indicator: "label-prior-drift".to_owned(),
            },
        )
        .unwrap();
    assert_eq!(shifted.status, StructuredTaskStatus::Completed);
    assert_eq!(shifted.calibration_state, CalibrationState::Uncalibrated);
    assert!(shifted.is_success());
    assert!(artifact.diagnostic_line().contains("locked_test_split="));
    let expired_raw = artifact
        .decide(
            0.9,
            0.8,
            ValidityDate::new(2026, 8, 1).unwrap(),
            ShiftAssessment::InDistribution,
        )
        .unwrap();
    assert_eq!(
        expired_raw.calibration_state,
        CalibrationState::Uncalibrated
    );

    let abstain_spec = CalibrationArtifactSpec::new(
        ["yes".to_owned(), "no".to_owned()],
        validity,
        "repo-authored synthetic binary population",
        digest("temperature-fit"),
        partition.split_digests(),
        ShiftPolicy::ConservativeAbstain,
    )
    .unwrap();
    let abstaining =
        CalibrationArtifact::new(&identity(abstain_spec.digest()), abstain_spec).unwrap();
    assert_ne!(artifact.key(), abstaining.key());
    let shifted_to_abstention = abstaining
        .decide(
            0.9,
            0.8,
            ValidityDate::new(2026, 7, 20).unwrap(),
            ShiftAssessment::Detected {
                indicator: "label-prior-drift".to_owned(),
            },
        )
        .unwrap();
    assert_eq!(
        shifted_to_abstention.status,
        StructuredTaskStatus::Abstained
    );
    assert_eq!(
        shifted_to_abstention.calibration_state,
        CalibrationState::Invalidated
    );
    assert!(shifted_to_abstention.is_success());

    let temperature = TemperatureModel::fit(partition.calibration()).unwrap();
    let (metrics, selective) = report_locked_test(
        partition.locked_test(),
        |probability| temperature.calibrate(probability),
        0.5,
    )
    .unwrap();
    let memo = ExchangeabilityMemo::new(
        "repo-authored synthetic binary population",
        "one independent synthetic row",
        "The fixture's calibration and locked-test rows are exchangeable only for this test.",
    )
    .unwrap();
    let conformal = ConformalModel::fit(partition.calibration(), Some(memo), 0.25).unwrap();
    let coverage = conformal
        .coverage_on_locked_test(partition.locked_test())
        .unwrap();
    assert!(temperature.diagnostic_line().contains("fitted_rows=4"));
    assert!(metrics.diagnostic_line().contains("split=locked_test"));
    assert!(selective.diagnostic_line().contains("accepted="));
    assert!(coverage.diagnostic_line().contains("population="));

    eprintln!(
        "{}\n{}\n{}\n{}\n{}\nCALIBRATION RESULT=PASS coverage_scope=locked_test",
        artifact.diagnostic_line(),
        temperature.diagnostic_line(),
        metrics.diagnostic_line(),
        selective.diagnostic_line(),
        coverage.diagnostic_line(),
    );
}
