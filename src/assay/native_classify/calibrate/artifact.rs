//! Strict, authenticated native calibration transport and application scope.
use super::*;
use crate::calibration::{IsotonicBlock, ShiftAssessment, ShiftPolicy};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FittedArtifact { body: ArtifactBody, seal: ContentCommitment }
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ArtifactBody {
    schema_version: u32,
    evidence: String,
    scope: String,
    activation_authority: bool,
    receipt_grade: ReceiptGrade,
    bindings: Bindings,
    development_rows: usize,
    calibration_rows: usize,
    locked_test_rows: usize,
    valid_from: String,
    valid_through: String,
    shift_policy: PersistedShiftPolicy,
    parameters: Parameters,
    parameter_commitment: ContentCommitment,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum PersistedShiftPolicy { RawScoresUncalibrated, ConservativeAbstain }
impl From<ShiftPolicy> for PersistedShiftPolicy {
    fn from(policy: ShiftPolicy) -> Self {
        match policy { ShiftPolicy::RawScoresUncalibrated => Self::RawScoresUncalibrated,
            ShiftPolicy::ConservativeAbstain => Self::ConservativeAbstain }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Parameters {
    Temperature { fitted_rows: usize, temperature: f64 },
    Isotonic { fitted_rows: usize, knots: Vec<Knot> },
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Knot { upper_score: f64, calibrated_probability: f64 }
pub(super) enum Model { Temperature(TemperatureModel), Isotonic(IsotonicModel) }
impl Model {
    pub(super) fn transform(&self, probability: f64) -> Result<f64, CalibrationError> {
        match self { Self::Temperature(model) => model.calibrate(probability), Self::Isotonic(model) => model.calibrate(probability) }
    }
}
impl Parameters {
    pub(super) fn from_model<E>(model: &Model, rows: usize, checkpoint: &mut impl FnMut() -> Result<(), E>)
        -> Result<Self, E> where E: From<AssayError> {
        checkpoint()?;
        Ok(match model {
            Model::Temperature(model) => Self::Temperature { fitted_rows: model.fitted_rows(), temperature: model.temperature() },
            Model::Isotonic(model) => {
                let mut knots = super::super::reserved(model.blocks().len())?;
                for (index, block) in model.blocks().iter().enumerate() {
                    if index % 64 == 0 { checkpoint()?; }
                    knots.push(Knot { upper_score: block.upper_score, calibrated_probability: block.calibrated_probability });
                }
                Self::Isotonic { fitted_rows: rows, knots }
            }
        })
    }
    fn rows(&self) -> usize {
        match self { Self::Temperature { fitted_rows, .. } | Self::Isotonic { fitted_rows, .. } => *fitted_rows }
    }
    fn validate(&self, expected_rows: usize, limits: Limits) -> Result<(), AssayError> {
        if expected_rows == 0 || expected_rows > limits.max_rows || self.rows() != expected_rows {
            return Err(AssayError::InvalidCalibration);
        }
        match self {
            Self::Temperature { temperature, .. } => {
                TemperatureModel::from_parameters(*temperature, expected_rows).map_err(|_| AssayError::InvalidCalibration)?;
            }
            Self::Isotonic { knots, .. } => {
                if knots.is_empty() || knots.len() > expected_rows
                    || knots.iter().any(|knot| !knot.upper_score.is_finite() || !(0.0..=1.0).contains(&knot.upper_score)
                        || !knot.calibrated_probability.is_finite() || !(0.0..=1.0).contains(&knot.calibrated_probability))
                    || knots.windows(2).any(|pair| pair[0].upper_score >= pair[1].upper_score
                        || pair[0].calibrated_probability > pair[1].calibrated_probability) {
                    return Err(AssayError::InvalidCalibration);
                }
            }
        }
        Ok(())
    }
    fn restore(&self, expected_rows: usize, limits: Limits) -> Result<Model, AssayError> {
        self.validate(expected_rows, limits)?;
        Ok(match self {
            Self::Temperature { temperature, .. } => Model::Temperature(
                TemperatureModel::from_parameters(*temperature, expected_rows).map_err(|_| AssayError::InvalidCalibration)?),
            Self::Isotonic { knots, .. } => {
                let mut blocks = super::super::reserved(knots.len())?;
                blocks.extend(knots.iter().map(|knot| IsotonicBlock { upper_score: knot.upper_score,
                    calibrated_probability: knot.calibrated_probability }));
                Model::Isotonic(IsotonicModel::from_blocks(blocks).map_err(|_| AssayError::InvalidCalibration)?)
            }
        })
    }
}

pub(super) fn build(replay: &NativeReplay, bindings: Bindings, parameters: Parameters, fit: FitOptions,
    key: &CommitmentKey, namespace: &str, limits: Limits) -> Result<FittedArtifact, AssayError> {
    parameters.validate(replay.manifest.splits.calibration.len(), limits)?;
    let parameter_commitment = commit(key, namespace, "native_calibration_parameters_v1", &parameters, limits)?;
    let body = ArtifactBody { schema_version: 1, evidence: NATIVE_EVIDENCE.to_owned(), scope: SCOPE.to_owned(),
        activation_authority: false, receipt_grade: ReceiptGrade::AuditOnly, bindings,
        development_rows: replay.manifest.splits.development.len(), calibration_rows: replay.manifest.splits.calibration.len(),
        locked_test_rows: replay.manifest.splits.locked_test.len(), valid_from: fit.validity.valid_from.to_string(),
        valid_through: fit.validity.valid_through.to_string(), shift_policy: fit.shift_policy.into(), parameters, parameter_commitment };
    let seal = commit(key, namespace, "native_calibration_artifact_v1", &body, limits)?;
    Ok(FittedArtifact { body, seal })
}

/// Authenticated fitted material, still awaiting an exact current replay match.
/// There is no deserializer or public constructor for an authorized transform.
pub struct PreparedApplication {
    artifact: FittedArtifact,
    model: Model,
    options: ApplyOptions,
    source_bytes: usize,
}
impl PreparedApplication {
    pub fn validate_limits(&self, limits: Limits) -> Result<(), AssayError> {
        limits.validate()?;
        if self.source_bytes > limits.max_output_bytes { return Err(AssayError::InputLimit); }
        validate_body(&self.artifact.body, limits)?;
        validate_apply_options(&self.options, limits)
    }
    /// Source preflight can refuse substituted suites before expensive models.
    /// Native execution and observation collections are checked after replay.
    pub fn validate_dataset(&self, dataset: &Dataset, key: &CommitmentKey, namespace: &str, limits: Limits)
        -> Result<(), AssayError> {
        report::validate_scope(namespace, None)?; self.validate_limits(limits)?; dataset.validate_limits(limits)?;
        authenticate(&self.artifact, key, namespace, limits)?;
        let body = &self.artifact.body;
        let manifest = dataset.manifest();
        let (suite, input, splits) = report::source_bindings::<AssayError>(manifest, dataset.input_digest, key, namespace, &mut || Ok(()))?;
        if body.bindings.suite != suite || body.bindings.input != input || body.bindings.splits != splits
            || body.development_rows != manifest.splits.development.len()
            || body.calibration_rows != manifest.splits.calibration.len()
            || body.locked_test_rows != manifest.splits.locked_test.len()
            || body.bindings.execution_collection_rows != dataset.len() {
            return Err(AssayError::CalibrationBindingMismatch);
        }
        Ok(())
    }
}

/// Load the inner native FitOutcome, after any outer transport/file pin check.
/// Both its report and artifact are authenticated before parameters become
/// usable. Unknown/duplicate JSON fields, malformed parameters and insufficient
/// reports refuse; observation date and shift assessment are always explicit.
pub fn prepare_application(bytes: &[u8], key: &CommitmentKey, namespace: &str, options: ApplyOptions,
    limits: Limits) -> Result<PreparedApplication, AssayError> {
    limits.validate()?; report::validate_scope(namespace, None)?; validate_apply_options(&options, limits)?;
    if bytes.is_empty() || bytes.len() > limits.max_output_bytes { return Err(AssayError::InputLimit); }
    let fitted: FitOutcome = super::super::dataset::strict_json(bytes, limits.max_line_bytes)?;
    if fitted.body.schema_version != 1 || fitted.body.evidence != NATIVE_EVIDENCE || fitted.body.scope != SCOPE
        || fitted.body.activation_authority || fitted.body.receipt_grade != ReceiptGrade::AuditOnly {
        return Err(AssayError::InvalidCalibration);
    }
    if fitted.seal != commit(key, namespace, "native_calibration_fit_report_v1", &fitted.body, limits)? {
        return Err(AssayError::CalibrationBindingMismatch);
    }
    let FitDecision::Fitted { artifact, measurement, classification } = fitted.body.outcome else {
        return Err(AssayError::InvalidCalibration);
    };
    validate_body(&artifact.body, limits)?; authenticate(&artifact, key, namespace, limits)?;
    fitted.body.options.validate_rows(artifact.body.locked_test_rows, limits)?;
    if artifact.body.calibration_rows < fitted.body.options.minimum_samples
        || artifact.body.locked_test_rows < fitted.body.options.minimum_samples
        || fitted.body.native.executed_rows != artifact.body.bindings.execution_collection_rows
        || fitted.body.native.group_width == 0 || classification.rows != artifact.body.locked_test_rows
        || !matches!(measurement, Measurement::Measured { ref metrics } if metrics.rows == artifact.body.locked_test_rows) {
        return Err(AssayError::InvalidCalibration);
    }
    let model = artifact.body.parameters.restore(artifact.body.calibration_rows, limits)?;
    Ok(PreparedApplication { artifact, model, options, source_bytes: bytes.len() })
}

fn validate_body(body: &ArtifactBody, limits: Limits) -> Result<(), AssayError> {
    let total = body.development_rows.checked_add(body.calibration_rows).and_then(|n| n.checked_add(body.locked_test_rows))
        .ok_or(AssayError::InputLimit)?;
    if body.schema_version != 1 || body.evidence != NATIVE_EVIDENCE || body.scope != SCOPE
        || body.activation_authority || body.receipt_grade != ReceiptGrade::AuditOnly
        || body.development_rows == 0 || body.calibration_rows == 0 || body.locked_test_rows == 0 || total > limits.max_rows
        || body.bindings.execution_collection_rows != total || body.bindings.observation_collection_rows != total
        || body.bindings.execution_collection_format != super::super::EXECUTION_COLLECTION_FORMAT
        || body.bindings.observation_collection_format != super::super::OBSERVATION_COLLECTION_FORMAT {
        return Err(AssayError::InvalidCalibration);
    }
    validity(body)?; body.parameters.validate(body.calibration_rows, limits)
}
fn validity(body: &ArtifactBody) -> Result<ValidityWindow, AssayError> {
    ValidityWindow::new(crate::assay::calibrate::parse_date(&body.valid_from).map_err(|_| AssayError::InvalidCalibration)?,
        crate::assay::calibrate::parse_date(&body.valid_through).map_err(|_| AssayError::InvalidCalibration)?)
        .map_err(|_| AssayError::InvalidCalibration)
}
fn authenticate(artifact: &FittedArtifact, key: &CommitmentKey, namespace: &str, limits: Limits) -> Result<(), AssayError> {
    if artifact.seal != commit(key, namespace, "native_calibration_artifact_v1", &artifact.body, limits)?
        || artifact.body.parameter_commitment != commit(key, namespace, "native_calibration_parameters_v1",
            &artifact.body.parameters, limits)? { return Err(AssayError::CalibrationBindingMismatch); }
    Ok(())
}
fn validate_apply_options(options: &ApplyOptions, limits: Limits) -> Result<(), AssayError> {
    crate::assay::calibrate::parse_date(&options.observed_date.to_string())?;
    if let ShiftAssessment::Detected { indicator } = &options.shift {
        if indicator.trim().is_empty() || indicator.len() > limits.max_line_bytes
            || indicator.chars().any(|ch| ch.is_control() || matches!(ch, '\u{2028}' | '\u{2029}')) {
            return Err(AssayError::Options);
        }
    }
    Ok(())
}

pub(super) struct Treatment<'a> { model: Option<&'a Model>, pub(super) disclosure: ApplicationDisclosure }
impl<'a> Treatment<'a> {
    pub(super) fn raw() -> Self {
        Self { model: None, disclosure: ApplicationDisclosure { status: ApplicationStatus::Raw, calibration: None,
            reason: None, observed_date: None, shift_assessment: None, shift_indicator: None,
            assessment_evidence: "not_applicable".to_owned() } }
    }
    pub(super) fn abstained(&self) -> bool { self.disclosure.status == ApplicationStatus::Abstained }
    pub(super) fn semantics(&self) -> &'static str {
        if self.disclosure.status == ApplicationStatus::Calibrated { CALIBRATED } else { RAW }
    }
    pub(super) fn transform(&self, probability: f64) -> Result<f64, CalibrationError> {
        if self.abstained() { return Err(CalibrationError::InvalidFittedParameters { method: "abstained" }); }
        match self.model { Some(model) => model.transform(probability), None => Ok(probability) }
    }
}

pub(super) fn verify_replay<'a, E>(application: &'a PreparedApplication, replay: &NativeReplay,
    key: &CommitmentKey, namespace: &str, limits: Limits, checkpoint: &mut impl FnMut() -> Result<(), E>)
    -> Result<Treatment<'a>, E> where E: From<AssayError> {
    checkpoint()?; application.validate_limits(limits)?;
    authenticate(&application.artifact, key, namespace, limits)?;
    let body = &application.artifact.body;
    if body.bindings != report::bindings(replay, key, namespace, checkpoint)? {
        return Err(AssayError::CalibrationBindingMismatch.into());
    }
    let window = validity(body)?;
    let reason = if application.options.observed_date < window.valid_from { Some(InvalidationReason::NotYetValid) }
        else if application.options.observed_date > window.valid_through { Some(InvalidationReason::Expired) }
        else if matches!(&application.options.shift, ShiftAssessment::Detected { .. }) { Some(InvalidationReason::DistributionShift) }
        else { None };
    let status = match (reason, body.shift_policy) {
        (None, _) => ApplicationStatus::Calibrated,
        (Some(_), PersistedShiftPolicy::RawScoresUncalibrated) => ApplicationStatus::Uncalibrated,
        (Some(_), PersistedShiftPolicy::ConservativeAbstain) => ApplicationStatus::Abstained,
    };
    let (shift_assessment, shift_indicator) = match &application.options.shift {
        ShiftAssessment::InDistribution => ("in_distribution", None),
        ShiftAssessment::Detected { indicator } => ("detected", Some(indicator.clone())),
    };
    checkpoint()?;
    Ok(Treatment { model: (status == ApplicationStatus::Calibrated).then_some(&application.model),
        disclosure: ApplicationDisclosure { status, calibration: Some(application.artifact.seal.clone()), reason,
            observed_date: Some(application.options.observed_date.to_string()), shift_assessment: Some(shift_assessment.to_owned()),
            shift_indicator, assessment_evidence: "caller_declared".to_owned() } })
}
