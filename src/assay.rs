//! Bounded, user-owned evaluation of explicitly supplied binary-event scores.
//!
//! These observations are caller data, not evidence of a native model replay.
//! Reports bind private suite, input, split and execution identities using
//! caller-owned HMAC keys. They never authorize model activation.

use std::{error::Error, fmt, io::Write};
use serde::{Deserialize, Serialize};
use crate::{
    calibration::{self, BootstrapConfig, CalibrationError},
    error::ErrorCode,
    receipt::CommitmentKey,
};

mod dataset;
pub mod calibrate;
pub mod qualify;
pub use dataset::{Dataset, DatasetBindings, DatasetMetadata, Observation, Split, SplitBindings, SplitIds, SuiteManifest};

pub const ASSAY_SCHEMA_VERSION: u32 = 1;
pub const ASSAY_EVIDENCE: &str = "provided_observations";

/// Independently selected input, output and computation limits. These are
/// payload limits, not a claim about process RSS or native model admission.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub max_manifest_bytes: usize,
    pub max_input_bytes: usize,
    pub max_line_bytes: usize,
    pub max_rows: usize,
    pub max_output_bytes: usize,
    pub max_bootstrap_resamples: usize,
    pub max_bootstrap_draws: u64,
}
impl Default for Limits {
    fn default() -> Self {
        Self { max_manifest_bytes: 1024 * 1024, max_input_bytes: 64 * 1024 * 1024,
            max_line_bytes: 64 * 1024, max_rows: 100_000, max_output_bytes: 4 * 1024 * 1024,
            max_bootstrap_resamples: 1000, max_bootstrap_draws: 10_000_000 }
    }
}
impl Limits {
    pub fn validate(self) -> Result<(), AssayError> {
        if !(1..=1024 * 1024).contains(&self.max_manifest_bytes)
            || !(1..=64 * 1024 * 1024).contains(&self.max_input_bytes)
            || !(1..=64 * 1024).contains(&self.max_line_bytes)
            || self.max_line_bytes > self.max_input_bytes
            || !(3..=100_000).contains(&self.max_rows)
            || !(2..=4 * 1024 * 1024).contains(&self.max_output_bytes)
            || !(2..=10_000).contains(&self.max_bootstrap_resamples)
            || !(1..=100_000_000).contains(&self.max_bootstrap_draws) {
            return Err(AssayError::Limits);
        }
        Ok(())
    }
}

/// Errors deliberately contain no private IDs, JSON fragments or raw digests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssayError {
    Limits, InputLimit, Json, Manifest, Observation, DuplicateId, DuplicateUnit,
    SplitMembership, InputDigest, Identity, Options, Calibration, Commitment,
    Allocation, Output, InvalidCalibration, CalibrationBindingMismatch,
    CalibrationInvalidated, Qualification, Scope,
}
impl AssayError {
    pub const fn code(self) -> &'static str {
        match self {
            Self::Limits => "invalid_limits",
            Self::InputLimit => "input_limit",
            Self::Json => "invalid_json",
            Self::Manifest => "invalid_suite_manifest",
            Self::Observation => "invalid_observation",
            Self::DuplicateId => "duplicate_observation_id",
            Self::DuplicateUnit => "duplicate_observation_unit",
            Self::SplitMembership => "invalid_split_membership",
            Self::InputDigest => "observation_digest_mismatch",
            Self::Identity => "invalid_execution_identity",
            Self::Options => "invalid_eval_options",
            Self::Calibration => "calibration_failed",
            Self::Commitment => "invalid_commitment_scope",
            Self::Allocation => "allocation_refused",
            Self::Output => "output_limit_or_encoding",
            Self::InvalidCalibration => "invalid_calibration",
            Self::CalibrationBindingMismatch => "calibration_binding_mismatch",
            Self::CalibrationInvalidated => "calibration_invalidated",
            Self::Qualification => "invalid_qualification",
            Self::Scope => "qualification_scope_mismatch",
        }
    }
    pub const fn exit_code(self) -> ErrorCode {
        match self {
            Self::Limits | Self::Options | Self::Commitment => ErrorCode::Usage,
            Self::InputLimit | Self::Allocation | Self::Output => ErrorCode::AdmissionOrResourceLimit,
            Self::InputDigest | Self::Identity | Self::InvalidCalibration
                | Self::CalibrationBindingMismatch | Self::CalibrationInvalidated
                | Self::Qualification | Self::Scope => ErrorCode::ArtifactIntegrityOrFormatOrVersion,
            Self::Json | Self::Manifest | Self::Observation | Self::DuplicateId
                | Self::DuplicateUnit | Self::SplitMembership | Self::Calibration => ErrorCode::InputDecodeOrParse,
        }
    }
}
impl fmt::Display for AssayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.code()) }
}
impl Error for AssayError {}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BootstrapOptions {
    pub resamples: usize,
    pub confidence_level: f64,
    pub seed: u64,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalOptions {
    pub minimum_samples: usize,
    /// A row is accepted as a positive event when p(positive) >= this value.
    /// This is not max(p,1-p) classification confidence.
    pub acceptance_threshold: f64,
    pub bootstrap: BootstrapOptions,
}
impl Default for EvalOptions {
    fn default() -> Self {
        Self { minimum_samples: 20, acceptance_threshold: 0.8,
            bootstrap: BootstrapOptions { resamples: 200, confidence_level: 0.95, seed: 0 } }
    }
}
impl EvalOptions {
    /// Validate all row-independent bounds before opening caller inputs.
    pub fn validate(self, limits: Limits) -> Result<(), AssayError> {
        limits.validate()?;
        if !(5..=100_000).contains(&self.minimum_samples)
            || !self.acceptance_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.acceptance_threshold)
            || self.bootstrap.resamples > limits.max_bootstrap_resamples
            || !self.bootstrap.confidence_level.is_finite()
            || self.bootstrap.confidence_level <= 0.0 || self.bootstrap.confidence_level >= 1.0 {
            return Err(AssayError::Options);
        }
        self.bootstrap_config()?;
        Ok(())
    }
    /// The product bounds actual sample draws before bootstrap allocations.
    pub fn validate_rows(self, locked_test_rows: usize, limits: Limits) -> Result<(), AssayError> {
        self.validate(limits)?;
        if locked_test_rows > limits.max_rows
            || (locked_test_rows as u64).checked_mul(self.bootstrap.resamples as u64)
                .is_none_or(|draws| draws > limits.max_bootstrap_draws) {
            return Err(AssayError::InputLimit);
        }
        Ok(())
    }
    fn bootstrap_config(self) -> Result<BootstrapConfig, AssayError> {
        BootstrapConfig::new(self.bootstrap.resamples, self.bootstrap.confidence_level, self.bootstrap.seed)
            .map_err(|_| AssayError::Options)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub confidence_level: f64,
    pub resamples: usize,
}
impl From<calibration::BootstrapConfidenceInterval> for ConfidenceInterval {
    fn from(value: calibration::BootstrapConfidenceInterval) -> Self {
        Self { lower: value.lower, upper: value.upper,
            confidence_level: value.confidence_level, resamples: value.resamples }
    }
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LockedTestMetrics {
    pub rows: usize,
    pub ece: f64,
    pub brier: f64,
    pub acceptance_threshold: f64,
    pub accepted: usize,
    pub abstained: usize,
    pub selective_risk: Option<f64>,
    pub ece_interval: ConfidenceInterval,
    pub brier_interval: ConfidenceInterval,
    /// Absent if any replica accepts zero rows; never silently conditioned.
    pub selective_risk_interval: Option<ConfidenceInterval>,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum Measurement {
    InsufficientData { split: Split, rows: usize, minimum_samples: usize },
    Measured { metrics: LockedTestMetrics },
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Evaluation {
    pub schema_version: u32,
    pub task: String,
    pub evidence: String,
    pub activation_authority: bool,
    pub bindings: DatasetBindings,
    pub options: EvalOptions,
    pub measurement: Measurement,
}

/// Report only held-out observations. No fitting or model execution occurs.
pub fn evaluate(dataset: &Dataset, key: &CommitmentKey, namespace: &str,
    options: EvalOptions, limits: Limits) -> Result<Evaluation, AssayError> {
    options.validate(limits)?;
    dataset.validate_limits(limits)?;
    let bindings = dataset.bindings(key, namespace)?;
    let measurement = measure(dataset, options, limits, Ok)?;
    let result = Evaluation { schema_version: ASSAY_SCHEMA_VERSION, task: "binary_event".to_owned(),
        evidence: ASSAY_EVIDENCE.to_owned(), activation_authority: false, bindings, options, measurement };
    // A successful result is publishable within the caller's declared ceiling.
    render(&result, limits)?;
    Ok(result)
}

/// Shared locked-test reporter for raw and explicitly fitted score transforms.
/// The transform is never passed development or calibration observations.
pub fn measure(dataset: &Dataset, options: EvalOptions, limits: Limits,
    transform: impl Fn(f64) -> Result<f64, CalibrationError>) -> Result<Measurement, AssayError> {
    dataset.validate_limits(limits)?;
    let locked = dataset.partition().locked_test();
    options.validate_rows(locked.len(), limits)?;
    if locked.len() < options.minimum_samples {
        return Ok(Measurement::InsufficientData { split: Split::LockedTest,
            rows: locked.len(), minimum_samples: options.minimum_samples });
    }
    let (metrics, selective) = calibration::report_locked_test(locked, &transform, options.acceptance_threshold)
        .map_err(|_| AssayError::Calibration)?;
    let intervals = calibration::bootstrap_locked_test_confidence_intervals(locked, transform,
        options.acceptance_threshold, options.bootstrap_config()?).map_err(|_| AssayError::Calibration)?;
    Ok(Measurement::Measured { metrics: LockedTestMetrics { rows: metrics.rows,
        ece: metrics.ece, brier: metrics.brier, acceptance_threshold: selective.threshold,
        accepted: selective.accepted, abstained: selective.abstained, selective_risk: selective.risk,
        ece_interval: intervals.ece.into(), brier_interval: intervals.brier.into(),
        selective_risk_interval: intervals.selective_risk.map(Into::into) } })
}

/// Canonical, single-line, newline-terminated JSON. Size admission precedes
/// canonical staging, and no partial bytes are published by this function.
pub fn render<T: Serialize + ?Sized>(value: &T, limits: Limits) -> Result<Vec<u8>, AssayError> {
    limits.validate()?;
    let mut count = OutputCount { length: 0, cap: limits.max_output_bytes - 1 };
    serde_json::to_writer(&mut count, value).map_err(|_| AssayError::Output)?;
    let mut bytes = crate::canonjson::canonical_bytes(value).map_err(|_| AssayError::Output)?;
    if bytes.len() >= limits.max_output_bytes { return Err(AssayError::Output); }
    bytes.try_reserve_exact(1).map_err(|_| AssayError::Allocation)?;
    bytes.push(b'\n');
    Ok(bytes)
}
struct OutputCount { length: usize, cap: usize }
impl Write for OutputCount {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.length = self.length.checked_add(bytes.len()).filter(|&n| n <= self.cap)
            .ok_or_else(|| std::io::Error::other("assay output bound"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

#[cfg(test)]
pub(crate) mod tests;
