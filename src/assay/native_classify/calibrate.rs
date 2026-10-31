//! Fit and explicitly reuse calibration over an exact frozen native replay.
//!
//! Raw native identities and observations remain unchanged. A separate sealed
//! artifact authorizes a reporting transform; it is never a fabricated model
//! execution identity, a new native classification decision or activation.
use serde::{Deserialize, Serialize};
use crate::{calibration::{CalibrationError, IsotonicModel, TemperatureModel, ValidityWindow},
    native_engine::strict_int4::Int4Work,
    receipt::{CommitmentDomain, CommitmentKey, ContentCommitment, ReceiptGrade}};
use super::{report, AssayError, Bindings, ClassificationMetrics, Dataset, EvalOptions,
    Evaluation, Gates, Limits, NativeReplay, QualificationReport, Split, NATIVE_EVIDENCE};
use crate::assay::Measurement;
pub use crate::assay::calibrate::{ApplyOptions, FitMethod, FitOptions, InvalidationReason};

mod artifact;
pub use artifact::{FittedArtifact, PreparedApplication, prepare_application};
use artifact::{Model, Parameters, Treatment};

pub const SCOPE: &str = "exact_frozen_native_classification_replay";
const RAW: &str = "uncalibrated_positive_candidate_weight_over_two_completed_labels; full_vocabulary_scored_eos";
const CALIBRATED: &str = "artifact_calibrated_positive_event_score; raw_native_decisions_unchanged; exact_frozen_replay_only";

/// Additional modeled storage for owned artifacts, fit scratch and authenticated
/// report staging. Source/native/replay storage and final delivery are separate.
pub fn required_calibration_bytes(limits: Limits, fit: bool, artifacts: usize) -> Result<u64, AssayError> {
    limits.validate()?;
    if artifacts > 2 || (fit && artifacts != 0) { return Err(AssayError::Limits); }
    if !fit && artifacts == 0 { return Ok(0); }
    let rows = (limits.max_rows as u64).checked_mul(if fit { 512 } else { 128 }).ok_or(AssayError::Limits)?;
    let staging = (limits.max_output_bytes as u64).checked_mul(16).ok_or(AssayError::Limits)?;
    rows.checked_add(staging).and_then(|one| one.checked_mul(if fit { 1 } else { artifacts as u64 }))
        .and_then(|all| all.checked_add(staging / 2)).ok_or(AssayError::Limits)
}

pub fn validate_fit_options(options: FitOptions, limits: Limits) -> Result<(), AssayError> {
    limits.validate()?;
    let from = crate::assay::calibrate::parse_date(&options.validity.valid_from.to_string())?;
    let through = crate::assay::calibrate::parse_date(&options.validity.valid_through.to_string())?;
    ValidityWindow::new(from, through).map_err(|_| AssayError::Options)?;
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeSummary {
    pub executed_rows: usize,
    pub group_width: usize,
    pub model_work: Int4Work,
    pub rewound_positions: u64,
}
impl NativeSummary {
    fn from_replay(replay: &NativeReplay) -> Self {
        Self { executed_rows: replay.rows.len(), group_width: replay.group_width,
            model_work: replay.model_work, rewound_positions: replay.rewound_positions }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum FitDecision {
    InsufficientData { bindings: Bindings, split: Split, rows: usize, minimum_samples: usize },
    Fitted { artifact: FittedArtifact, measurement: Measurement, classification: ClassificationMetrics },
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FitBody {
    pub schema_version: u32,
    pub evidence: String,
    pub scope: String,
    pub activation_authority: bool,
    pub receipt_grade: ReceiptGrade,
    pub options: EvalOptions,
    pub native: NativeSummary,
    pub outcome: FitDecision,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FitOutcome { pub body: FitBody, pub seal: ContentCommitment }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplicationStatus { Raw, Calibrated, Uncalibrated, Abstained }
/// The date and shift assessment are caller declarations, separately bound from
/// observed native execution. Raw mode has no claimed calibration authority.
#[derive(Clone, Debug, Serialize)]
pub struct ApplicationDisclosure {
    pub status: ApplicationStatus,
    pub calibration: Option<ContentCommitment>,
    pub reason: Option<InvalidationReason>,
    pub observed_date: Option<String>,
    pub shift_assessment: Option<String>,
    pub shift_indicator: Option<String>,
    pub assessment_evidence: String,
}
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ApplicationDecision {
    Evaluated { evaluation: Evaluation },
    Abstained { bindings: Bindings, native: NativeSummary },
}
#[derive(Clone, Debug, Serialize)]
pub struct ApplicationBody {
    pub schema_version: u32,
    pub evidence: String,
    pub scope: String,
    pub activation_authority: bool,
    pub receipt_grade: ReceiptGrade,
    pub options: EvalOptions,
    pub application: ApplicationDisclosure,
    pub outcome: ApplicationDecision,
}
#[derive(Clone, Debug, Serialize)]
pub struct ApplicationOutcome { pub body: ApplicationBody, pub seal: ContentCommitment }

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum QualificationDecision {
    Compared { qualification: QualificationReport },
    /// This is not a failed native decision, insufficient-data substitution or
    /// a passed gate. No numeric qualification gates run after this fallback.
    Abstained { candidate: Bindings, baseline: Bindings,
        candidate_native: NativeSummary, baseline_native: NativeSummary },
}
#[derive(Clone, Debug, Serialize)]
pub struct QualificationBody {
    pub schema_version: u32,
    pub evidence: String,
    pub scope: String,
    pub activation_authority: bool,
    pub receipt_grade: ReceiptGrade,
    pub declared_host_scope: String,
    pub host_scope_evidence: String,
    pub gate_basis: String,
    pub gates: Gates,
    pub options: EvalOptions,
    pub candidate_application: ApplicationDisclosure,
    pub baseline_application: ApplicationDisclosure,
    pub total_model_work: Int4Work,
    pub outcome: QualificationDecision,
}
#[derive(Clone, Debug, Serialize)]
pub struct QualificationOutcome { pub body: QualificationBody, pub seal: ContentCommitment }

#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_with_checkpoint<E>(replay: &NativeReplay, key: &CommitmentKey, namespace: &str,
    fit: FitOptions, options: EvalOptions, limits: Limits, checkpoint: &mut impl FnMut() -> Result<(), E>)
    -> Result<FitOutcome, E>
where E: From<AssayError> + From<CalibrationError> {
    if !replay.manifest.is_binary() { return Err(AssayError::CalibrationBindingMismatch.into()); }
    report::validate_scope(namespace, None)?; validate_fit_options(fit, limits)?;
    replay.sizes.validate(limits)?; options.validate_rows(replay.manifest.splits.locked_test.len(), limits)?;
    let bindings = report::bindings(replay, key, namespace, checkpoint)?;
    let insufficient = [(Split::Calibration, replay.manifest.splits.calibration.len()),
        (Split::LockedTest, replay.manifest.splits.locked_test.len())].into_iter()
        .find(|(_, rows)| *rows < options.minimum_samples);
    let outcome = if let Some((split, rows)) = insufficient {
        FitDecision::InsufficientData { bindings, split, rows, minimum_samples: options.minimum_samples }
    } else {
        let partition = replay.partition(checkpoint)?;
        let calibration = partition.calibration();
        let model = match fit.method {
            FitMethod::Temperature => Model::Temperature(TemperatureModel::fit_with_checkpoint(calibration, checkpoint)?),
            FitMethod::Isotonic => Model::Isotonic(IsotonicModel::fit_with_checkpoint(calibration, checkpoint)?),
        };
        let parameters = Parameters::from_model(&model, calibration.len(), checkpoint)?;
        drop(partition);
        let artifact = artifact::build(replay, bindings, parameters, fit, key, namespace, limits)?;
        checkpoint()?;
        let evaluation = report::evaluate_transformed_with_checkpoint(replay, key, namespace, options, limits,
            &|probability| model.transform(probability), CALIBRATED, checkpoint)?;
        FitDecision::Fitted { artifact, measurement: evaluation.body.measurement,
            classification: evaluation.body.classification.ok_or(AssayError::Calibration)? }
    };
    let body = FitBody { schema_version: 1, evidence: NATIVE_EVIDENCE.to_owned(), scope: SCOPE.to_owned(),
        activation_authority: false, receipt_grade: ReceiptGrade::AuditOnly, options,
        native: NativeSummary::from_replay(replay), outcome };
    checkpoint()?;
    let seal = commit(key, namespace, "native_calibration_fit_report_v1", &body, limits)?;
    let result = FitOutcome { body, seal }; crate::assay::render(&result, limits)?;
    checkpoint()?; Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_with_checkpoint<E>(replay: &NativeReplay, application: &PreparedApplication,
    key: &CommitmentKey, namespace: &str, options: EvalOptions, limits: Limits,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<ApplicationOutcome, E>
where E: From<AssayError> + From<CalibrationError> {
    if !replay.manifest.is_binary() { return Err(AssayError::CalibrationBindingMismatch.into()); }
    report::validate_scope(namespace, None)?; replay.sizes.validate(limits)?;
    options.validate_rows(replay.manifest.splits.locked_test.len(), limits)?;
    let treatment = artifact::verify_replay(application, replay, key, namespace, limits, checkpoint)?;
    let outcome = if treatment.abstained() {
        ApplicationDecision::Abstained { bindings: report::bindings(replay, key, namespace, checkpoint)?,
            native: NativeSummary::from_replay(replay) }
    } else {
        ApplicationDecision::Evaluated { evaluation: report::evaluate_transformed_with_checkpoint(replay, key,
            namespace, options, limits, &|probability| treatment.transform(probability), treatment.semantics(), checkpoint)? }
    };
    let body = ApplicationBody { schema_version: 1, evidence: NATIVE_EVIDENCE.to_owned(), scope: SCOPE.to_owned(),
        activation_authority: false, receipt_grade: ReceiptGrade::AuditOnly, options,
        application: treatment.disclosure, outcome };
    checkpoint()?;
    let seal = commit(key, namespace, "native_calibration_application_v1", &body, limits)?;
    let result = ApplicationOutcome { body, seal }; crate::assay::render(&result, limits)?;
    checkpoint()?; Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compare_with_checkpoint<E>(candidate: &NativeReplay, baseline: &NativeReplay,
    candidate_application: Option<&PreparedApplication>, baseline_application: Option<&PreparedApplication>,
    key: &CommitmentKey, namespace: &str, host_scope: &str, gates: Gates, options: EvalOptions,
    limits: Limits, checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<QualificationOutcome, E>
where E: From<AssayError> + From<CalibrationError> {
    if !candidate.manifest.is_binary() || !baseline.manifest.is_binary() {
        return Err(AssayError::CalibrationBindingMismatch.into());
    }
    report::validate_scope(namespace, Some(host_scope))?; gates.validate(limits)?;
    candidate.sizes.validate(limits)?; baseline.sizes.validate(limits)?;
    report::validate_pair(candidate, baseline, checkpoint)?;
    let rows = candidate.manifest.splits.locked_test.len(); options.validate_rows(rows, limits)?;
    if (rows as u64).checked_mul(options.bootstrap.resamples as u64).and_then(|n| n.checked_mul(2))
        .is_none_or(|draws| draws > limits.max_bootstrap_draws) { return Err(AssayError::InputLimit.into()); }
    let current = match candidate_application {
        Some(application) => artifact::verify_replay(application, candidate, key, namespace, limits, checkpoint)?,
        None => Treatment::raw(),
    };
    let prior = match baseline_application {
        Some(application) => artifact::verify_replay(application, baseline, key, namespace, limits, checkpoint)?,
        None => Treatment::raw(),
    };
    let outcome = if current.abstained() || prior.abstained() {
        QualificationDecision::Abstained { candidate: report::bindings(candidate, key, namespace, checkpoint)?,
            baseline: report::bindings(baseline, key, namespace, checkpoint)?,
            candidate_native: NativeSummary::from_replay(candidate), baseline_native: NativeSummary::from_replay(baseline) }
    } else {
        QualificationDecision::Compared { qualification: report::compare_transformed_with_checkpoint(candidate,
            baseline, key, namespace, host_scope, gates, options, limits,
            &|probability| current.transform(probability), &|probability| prior.transform(probability),
            current.semantics(), prior.semantics(), checkpoint)? }
    };
    let body = QualificationBody { schema_version: 1, evidence: NATIVE_EVIDENCE.to_owned(), scope: SCOPE.to_owned(),
        activation_authority: false, receipt_grade: ReceiptGrade::AuditOnly, declared_host_scope: host_scope.to_owned(),
        host_scope_evidence: "caller_declared".to_owned(), gate_basis: "point_estimates".to_owned(), gates, options,
        candidate_application: current.disclosure, baseline_application: prior.disclosure,
        total_model_work: candidate.model_work.checked_add(baseline.model_work).map_err(|_| AssayError::InputLimit)?, outcome };
    checkpoint()?;
    let seal = commit(key, namespace, "native_calibrated_qualification_v1", &body, limits)?;
    let result = QualificationOutcome { body, seal }; crate::assay::render(&result, limits)?;
    checkpoint()?; Ok(result)
}

fn commit<T: Serialize>(key: &CommitmentKey, namespace: &str, kind: &str, body: &T, limits: Limits)
    -> Result<ContentCommitment, AssayError> {
    let bytes = crate::assay::render(body, limits)?;
    key.commit(CommitmentDomain::Configuration, namespace, kind, &bytes).map_err(|_| AssayError::Commitment)
}

#[cfg(test)] mod tests;
