//! Persistable calibration of explicitly supplied binary-event observations.
//!
//! Fits consume only the sealed calibration split. Locked-test measurements
//! remain separate, and exported identities are keyed commitments. These
//! artifacts carry no native-inference, model-activation or release authority.

use serde::{Deserialize, Serialize};

use super::{AssayError, Dataset, DatasetBindings, EvalOptions, Limits, Measurement, Split, measure};
use crate::{
    calibration::{
        CalibrationArtifact, CalibrationArtifactSpec, CalibrationError, IsotonicBlock,
        IsotonicModel, ShiftAssessment, ShiftPolicy, TemperatureModel, ValidityDate, ValidityWindow,
    },
    canonjson::{self, ParseLimits},
    execution_identity::Sha256Digest,
    receipt::{CommitmentDomain, CommitmentKey, ContentCommitment},
};

const SCHEMA_VERSION: u32 = 1;
const SCOPE: &str = "supplied_observations_only";

/// The parameter family fitted only on the calibration split.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FitMethod {
    Temperature,
    Isotonic,
}

/// Caller-declared validity and fallback policy for an explicit fit.
#[derive(Clone, Copy, Debug)]
pub struct FitOptions {
    pub method: FitMethod,
    pub validity: ValidityWindow,
    pub shift_policy: ShiftPolicy,
}

/// Every application supplies its observation date and shift assessment.
#[derive(Clone, Debug)]
pub struct ApplyOptions {
    pub observed_date: ValidityDate,
    pub shift: ShiftAssessment,
}

/// A checked date spelling shared by CLI arguments and persisted artifacts.
pub fn parse_date(value: &str) -> Result<ValidityDate, AssayError> {
    let bytes = value.as_bytes();
    if bytes.len() != 10
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes.iter().enumerate().any(|(index, byte)| {
            index != 4 && index != 7 && !byte.is_ascii_digit()
        })
    {
        return Err(AssayError::Options);
    }
    let year = value[..4].parse().map_err(|_| AssayError::Options)?;
    let month = value[5..7].parse().map_err(|_| AssayError::Options)?;
    let day = value[8..].parse().map_err(|_| AssayError::Options)?;
    ValidityDate::new(year, month, day).map_err(|_| AssayError::Options)
}

/// The complete durable report. Insufficient data never carries fitted material.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum FitOutcome {
    InsufficientData {
        bindings: DatasetBindings,
        options: EvalOptions,
        split: Split,
        rows: usize,
        minimum_samples: usize,
    },
    Fitted {
        artifact: FittedArtifact,
        options: EvalOptions,
        measurement: Measurement,
        report_commitment: ContentCommitment,
    },
}

/// Sealed fitted material; deserialization alone does not validate its authority.
///
/// Call apply with the current dataset, key, namespace and explicit validity
/// observations. The private fields prevent constructing a usable model by
/// bypassing that check.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FittedArtifact {
    body: ArtifactBody,
    seal: ContentCommitment,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ArtifactBody {
    schema_version: u32,
    scope: String,
    dataset: DatasetBindings,
    positive_label: String,
    negative_label: String,
    named_population: String,
    valid_from: String,
    valid_through: String,
    shift_policy: PersistedShiftPolicy,
    parameters: Parameters,
    parameter_commitment: ContentCommitment,
    calibration_subject: ContentCommitment,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum PersistedShiftPolicy {
    RawScoresUncalibrated,
    ConservativeAbstain,
}

impl From<ShiftPolicy> for PersistedShiftPolicy {
    fn from(value: ShiftPolicy) -> Self {
        match value {
            ShiftPolicy::RawScoresUncalibrated => Self::RawScoresUncalibrated,
            ShiftPolicy::ConservativeAbstain => Self::ConservativeAbstain,
        }
    }
}

impl From<PersistedShiftPolicy> for ShiftPolicy {
    fn from(value: PersistedShiftPolicy) -> Self {
        match value {
            PersistedShiftPolicy::RawScoresUncalibrated => Self::RawScoresUncalibrated,
            PersistedShiftPolicy::ConservativeAbstain => Self::ConservativeAbstain,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
enum Parameters {
    Temperature {
        fitted_rows: usize,
        temperature: f64,
    },
    Isotonic {
        fitted_rows: usize,
        knots: Vec<Knot>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Knot {
    upper_score: f64,
    calibrated_probability: f64,
}

enum Model {
    Temperature(TemperatureModel),
    Isotonic(IsotonicModel),
}

impl Model {
    fn transform(&self, probability: f64) -> Result<f64, CalibrationError> {
        match self {
            Self::Temperature(model) => model.calibrate(probability),
            Self::Isotonic(model) => model.calibrate(probability),
        }
    }
}

impl Parameters {
    fn restore(&self, expected_rows: usize, limits: Limits) -> Result<Model, AssayError> {
        let fitted_rows = match self {
            Self::Temperature { fitted_rows, .. } | Self::Isotonic { fitted_rows, .. } => *fitted_rows,
        };
        if fitted_rows == 0 || fitted_rows > limits.max_rows || fitted_rows != expected_rows {
            return Err(AssayError::InvalidCalibration);
        }
        match self {
            Self::Temperature { temperature, .. } => {
                TemperatureModel::from_parameters(*temperature, fitted_rows)
                    .map(Model::Temperature)
                    .map_err(|_| AssayError::InvalidCalibration)
            }
            Self::Isotonic { knots, .. } => {
                if knots.is_empty() || knots.len() > fitted_rows {
                    return Err(AssayError::InvalidCalibration);
                }
                let mut blocks = Vec::new();
                blocks.try_reserve_exact(knots.len()).map_err(|_| AssayError::Allocation)?;
                blocks.extend(knots.iter().map(|knot| IsotonicBlock {
                    upper_score: knot.upper_score,
                    calibrated_probability: knot.calibrated_probability,
                }));
                IsotonicModel::from_blocks(blocks)
                    .map(Model::Isotonic)
                    .map_err(|_| AssayError::InvalidCalibration)
            }
        }
    }
}

/// Why a valid artifact cannot make a current calibrated claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InvalidationReason {
    NotYetValid,
    Expired,
    DistributionShift,
}

/// Results distinguish calibrated measurements, raw fallback and typed abstention.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ApplyOutcome {
    InsufficientData {
        bindings: DatasetBindings,
        options: EvalOptions,
        split: Split,
        rows: usize,
        minimum_samples: usize,
    },
    Calibrated {
        bindings: DatasetBindings,
        calibration: ContentCommitment,
        options: EvalOptions,
        measurement: Measurement,
    },
    Uncalibrated {
        bindings: DatasetBindings,
        calibration: ContentCommitment,
        reason: InvalidationReason,
        options: EvalOptions,
        measurement: Measurement,
    },
    Abstained {
        bindings: DatasetBindings,
        calibration: ContentCommitment,
        reason: InvalidationReason,
    },
}

fn insufficient(dataset: &Dataset, options: EvalOptions) -> Option<(Split, usize)> {
    let partition = dataset.partition();
    [
        (Split::Calibration, partition.calibration().len()),
        (Split::LockedTest, partition.locked_test().len()),
    ]
    .into_iter()
    .find(|(_, rows)| *rows < options.minimum_samples)
}

fn commit<T: Serialize>(
    key: &CommitmentKey,
    namespace: &str,
    kind: &str,
    value: &T,
) -> Result<ContentCommitment, AssayError> {
    let bytes = canonjson::canonical_bytes(value).map_err(|_| AssayError::InvalidCalibration)?;
    key.commit(CommitmentDomain::Configuration, namespace, kind, &bytes)
        .map_err(|_| AssayError::Commitment)
}

#[derive(Serialize)]
struct ReportMaterial<'a> {
    artifact: &'a FittedArtifact,
    options: EvalOptions,
    measurement: &'a Measurement,
}

fn report_commitment(
    artifact: &FittedArtifact,
    options: EvalOptions,
    measurement: &Measurement,
    key: &CommitmentKey,
    namespace: &str,
    limits: Limits,
) -> Result<ContentCommitment, AssayError> {
    let material = ReportMaterial { artifact, options, measurement };
    super::render(&material, limits)?;
    commit(key, namespace, "assay_calibration_report_v1", &material)
}

fn bounded<T: Serialize>(result: T, limits: Limits) -> Result<T, AssayError> {
    super::render(&result, limits)?;
    Ok(result)
}

fn subject(
    dataset: &Dataset,
    parameters: &Parameters,
    validity: ValidityWindow,
    shift_policy: ShiftPolicy,
    key: &CommitmentKey,
    namespace: &str,
) -> Result<ContentCommitment, AssayError> {
    let manifest = dataset.manifest();
    let parameter_bytes =
        canonjson::canonical_bytes(parameters).map_err(|_| AssayError::InvalidCalibration)?;
    let spec = CalibrationArtifactSpec::new(
        [manifest.positive_label.clone(), manifest.negative_label.clone()],
        validity,
        manifest.named_population.clone(),
        Sha256Digest::of_bytes(&parameter_bytes),
        dataset.partition().split_digests(),
        shift_policy,
    )
    .map_err(|_| AssayError::InvalidCalibration)?;
    // Derive the fitted identity from the caller's complete real execution
    // identity; only its calibration component becomes this newly fitted spec.
    // The original identity remains bound separately by DatasetBindings.
    let mut identity = manifest.execution_identity.clone();
    identity.calibration_digest = spec.digest();
    let artifact =
        CalibrationArtifact::new(&identity, spec).map_err(|_| AssayError::InvalidCalibration)?;
    key.commit(
        CommitmentDomain::ExecutionIdentity,
        namespace,
        "assay_fitted_calibration_identity_v1",
        artifact.key().to_hex().as_bytes(),
    )
    .map_err(|_| AssayError::Commitment)
}

/// Fit once on calibration rows and measure only the independently sealed test.
pub fn fit(
    dataset: &Dataset,
    key: &CommitmentKey,
    namespace: &str,
    fit_options: FitOptions,
    eval_options: EvalOptions,
    limits: Limits,
) -> Result<FitOutcome, AssayError> {
    limits.validate()?;
    dataset.validate_limits(limits)?;
    eval_options.validate_rows(dataset.partition().locked_test().len(), limits)?;
    // The reusable artifact uses exactly YYYY-MM-DD. The underlying calendar
    // type also supports wider years; reject them here rather than creating a
    // fitted report that the strict persisted format cannot reload.
    let valid_from = parse_date(&fit_options.validity.valid_from.to_string())?;
    let valid_through = parse_date(&fit_options.validity.valid_through.to_string())?;
    let validity = ValidityWindow::new(valid_from, valid_through)
        .map_err(|_| AssayError::Options)?;
    let bindings = dataset.bindings(key, namespace)?;
    if let Some((split, rows)) = insufficient(dataset, eval_options) {
        return bounded(FitOutcome::InsufficientData {
            bindings,
            options: eval_options,
            split,
            rows,
            minimum_samples: eval_options.minimum_samples,
        }, limits);
    }
    let calibration = dataset.partition().calibration();
    let parameters = match fit_options.method {
        FitMethod::Temperature => {
            let model = TemperatureModel::fit(calibration).map_err(|_| AssayError::Calibration)?;
            Parameters::Temperature {
                fitted_rows: model.fitted_rows(),
                temperature: model.temperature(),
            }
        }
        FitMethod::Isotonic => {
            let model = IsotonicModel::fit(calibration).map_err(AssayError::from)?;
            Parameters::Isotonic {
                fitted_rows: calibration.len(),
                knots: model.blocks().iter().map(|block| Knot {
                    upper_score: block.upper_score,
                    calibrated_probability: block.calibrated_probability,
                }).collect(),
            }
        }
    };
    super::render(&parameters, limits)?;
    let model = parameters.restore(calibration.len(), limits)?;
    let manifest = dataset.manifest();
    let body = ArtifactBody {
        schema_version: SCHEMA_VERSION,
        scope: SCOPE.to_owned(),
        dataset: bindings,
        positive_label: manifest.positive_label.clone(),
        negative_label: manifest.negative_label.clone(),
        named_population: manifest.named_population.clone(),
        valid_from: validity.valid_from.to_string(),
        valid_through: validity.valid_through.to_string(),
        shift_policy: fit_options.shift_policy.into(),
        parameter_commitment: commit(key, namespace, "assay_calibration_parameters_v1", &parameters)?,
        calibration_subject: subject(dataset, &parameters, validity, fit_options.shift_policy, key, namespace)?,
        parameters,
    };
    super::render(&body, limits)?;
    let seal = commit(key, namespace, "assay_calibration_artifact_v1", &body)?;
    let artifact = FittedArtifact { body, seal };
    let measurement = measure(dataset, eval_options, limits, |probability| model.transform(probability))?;
    let report_commitment = report_commitment(&artifact, eval_options, &measurement, key, namespace, limits)?;
    let result = FitOutcome::Fitted { artifact, options: eval_options, measurement, report_commitment };
    // The same bounded renderer used by the CLI checks serialized output before
    // publication. Keep this library result inside its declared transport cap.
    bounded(result, limits)
}

fn parse_report(input: &[u8], limits: Limits) -> Result<FitOutcome, AssayError> {
    limits.validate()?;
    if input.len() > limits.max_output_bytes {
        return Err(AssayError::InputLimit);
    }
    let text = std::str::from_utf8(input).map_err(|_| AssayError::Json)?;
    let value = canonjson::parse_str_with_limits(text, ParseLimits {
        max_depth: 16,
        max_string_bytes: limits.max_line_bytes,
    })
    .map_err(|_| AssayError::Json)?;
    serde_json::from_value(value).map_err(|_| AssayError::InvalidCalibration)
}

fn validate_shift(shift: &ShiftAssessment, limits: Limits) -> Result<(), AssayError> {
    if let ShiftAssessment::Detected { indicator } = shift {
        if indicator.trim().is_empty()
            || indicator.len() > limits.max_line_bytes
            || indicator.chars().any(|c| c.is_control() || matches!(c, '\u{2028}' | '\u{2029}'))
        {
            return Err(AssayError::Options);
        }
    }
    Ok(())
}

fn verified(
    artifact: &FittedArtifact,
    dataset: &Dataset,
    key: &CommitmentKey,
    namespace: &str,
    limits: Limits,
) -> Result<(Model, ValidityWindow), AssayError> {
    let body = &artifact.body;
    let manifest = dataset.manifest();
    if body.schema_version != SCHEMA_VERSION
        || body.scope != SCOPE
        || body.positive_label != manifest.positive_label
        || body.negative_label != manifest.negative_label
        || body.named_population != manifest.named_population
    {
        return Err(AssayError::CalibrationBindingMismatch);
    }
    let validity = ValidityWindow::new(
        parse_date(&body.valid_from).map_err(|_| AssayError::InvalidCalibration)?,
        parse_date(&body.valid_through).map_err(|_| AssayError::InvalidCalibration)?,
    )
    .map_err(|_| AssayError::InvalidCalibration)?;
    if body.dataset != dataset.bindings(key, namespace)?
        || artifact.seal != commit(key, namespace, "assay_calibration_artifact_v1", body)?
        || body.parameter_commitment
            != commit(key, namespace, "assay_calibration_parameters_v1", &body.parameters)?
        || body.calibration_subject
            != subject(dataset, &body.parameters, validity, body.shift_policy.into(), key, namespace)?
    {
        return Err(AssayError::CalibrationBindingMismatch);
    }
    let model = body.parameters.restore(dataset.partition().calibration().len(), limits)?;
    Ok((model, validity))
}

/// Strictly reload an exact FitOutcome report and apply its checked fit.
///
/// The caller checks any expected file-byte digest before calling this method;
/// the keyed report seal independently authenticates all fitted material and
/// the original reported measurements. Fresh metrics are computed again only
/// from the currently supplied, identity-matching locked-test rows.
pub fn apply(
    artifact_bytes: &[u8],
    dataset: &Dataset,
    key: &CommitmentKey,
    namespace: &str,
    options: ApplyOptions,
    eval_options: EvalOptions,
    limits: Limits,
) -> Result<ApplyOutcome, AssayError> {
    limits.validate()?;
    dataset.validate_limits(limits)?;
    eval_options.validate_rows(dataset.partition().locked_test().len(), limits)?;
    validate_shift(&options.shift, limits)?;
    let FitOutcome::Fitted { artifact, options: fitted_options, measurement, report_commitment: supplied_commitment } =
        parse_report(artifact_bytes, limits)?
    else {
        return Err(AssayError::InvalidCalibration);
    };
    fitted_options.validate_rows(dataset.partition().locked_test().len(), limits)?;
    if supplied_commitment != report_commitment(&artifact, fitted_options, &measurement, key, namespace, limits)? {
        return Err(AssayError::CalibrationBindingMismatch);
    }
    let (model, validity) = verified(&artifact, dataset, key, namespace, limits)?;
    if let Some((split, rows)) = insufficient(dataset, eval_options) {
        return bounded(ApplyOutcome::InsufficientData {
            bindings: artifact.body.dataset.clone(),
            options: eval_options,
            split,
            rows,
            minimum_samples: eval_options.minimum_samples,
        }, limits);
    }
    let reason = if options.observed_date < validity.valid_from {
        Some(InvalidationReason::NotYetValid)
    } else if options.observed_date > validity.valid_through {
        Some(InvalidationReason::Expired)
    } else if matches!(options.shift, ShiftAssessment::Detected { .. }) {
        Some(InvalidationReason::DistributionShift)
    } else {
        None
    };
    let bindings = artifact.body.dataset.clone();
    let calibration = artifact.seal.clone();
    let result = match (reason, artifact.body.shift_policy) {
        (Some(reason), PersistedShiftPolicy::ConservativeAbstain) => ApplyOutcome::Abstained {
            bindings, calibration, reason,
        },
        (Some(reason), PersistedShiftPolicy::RawScoresUncalibrated) => ApplyOutcome::Uncalibrated {
            bindings, calibration, reason, options: eval_options,
            measurement: measure(dataset, eval_options, limits, Ok)?,
        },
        (None, _) => ApplyOutcome::Calibrated {
            bindings,
            calibration,
            options: eval_options,
            measurement: measure(dataset, eval_options, limits, |probability| model.transform(probability))?,
        },
    };
    bounded(result, limits)
}

#[cfg(test)]
mod tests;
