//! Paired quality gates over caller-supplied observations of one frozen suite.
//!
//! These are declared point-estimate gates, not native candidate/active replay
//! or a statistical noninferiority certificate. A successful comparison has
//! no model activation authority. Both arms and the policy remain explicitly
//! scoped and keyed in the exported audit receipt.

use serde::{Deserialize, Serialize};
use crate::{canonjson, receipt::{CommitmentDomain, CommitmentKey, ContentCommitment, ReceiptGrade}};
use super::{AssayError, Dataset, EvalOptions, Evaluation, Limits, Measurement, Split,
    ASSAY_EVIDENCE, ASSAY_SCHEMA_VERSION};

/// Every bound is explicit. Negative increase bounds require an improvement;
/// they do not turn a finite-data point estimate into a confidence claim.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Gates {
    pub schema_version: u32,
    pub max_brier_increase: f64,
    pub max_ece_increase: f64,
    pub max_selective_risk: f64,
    pub min_acceptance_rate: f64,
    pub minimum_accepted: usize,
}
impl Gates {
    pub fn validate(self, limits: Limits) -> Result<(), AssayError> {
        limits.validate()?;
        if self.schema_version != ASSAY_SCHEMA_VERSION
            || !bounded(self.max_brier_increase, -1.0, 1.0)
            || !bounded(self.max_ece_increase, -1.0, 1.0)
            || !bounded(self.max_selective_risk, 0.0, 1.0)
            || !bounded(self.min_acceptance_rate, 0.0, 1.0)
            || self.minimum_accepted == 0 || self.minimum_accepted > limits.max_rows {
            return Err(AssayError::Qualification);
        }
        Ok(())
    }
}

pub fn parse_gates(bytes: &[u8], limits: Limits) -> Result<Gates, AssayError> {
    limits.validate()?;
    if bytes.is_empty() || bytes.len() > limits.max_manifest_bytes.min(65_536) {
        return Err(AssayError::InputLimit);
    }
    let text = std::str::from_utf8(bytes).map_err(|_| AssayError::Qualification)?;
    let value = canonjson::parse_str_with_limits(text,
        canonjson::ParseLimits { max_depth: 4, max_string_bytes: 128 })
        .map_err(|_| AssayError::Qualification)?;
    let gates: Gates = serde_json::from_value(value).map_err(|_| AssayError::Qualification)?;
    gates.validate(limits)?;
    Ok(gates)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GateResult {
    pub name: String,
    /// None means the required metric was undefined, and the gate fails.
    pub observed: Option<f64>,
    pub limit: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedSummary {
    pub rows: usize,
    /// Mean(candidate squared error - baseline squared error), same units and labels.
    pub mean_brier_change: f64,
    pub candidate_better: usize,
    pub candidate_worse: usize,
    pub equal: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum Decision {
    Passed { comparisons: Vec<GateResult>, paired: PairedSummary },
    Rejected { comparisons: Vec<GateResult>, paired: PairedSummary },
    InsufficientData { rows: usize, minimum_samples: usize },
}

/// The unsigned body is a single typed schema used both for HMAC creation and
/// output. No private execution identity or raw input/split digest is exposed.
#[derive(Clone, Debug, Serialize)]
pub struct QualificationBody {
    pub schema_version: u32,
    pub task: String,
    pub evidence: String,
    pub activation_authority: bool,
    pub receipt_grade: ReceiptGrade,
    pub declared_host_scope: String,
    pub host_scope_evidence: String,
    pub gate_basis: String,
    pub gates: Gates,
    pub policy_commitment: ContentCommitment,
    pub candidate: Evaluation,
    pub baseline: Evaluation,
    pub decision: Decision,
}

#[derive(Clone, Debug, Serialize)]
pub struct QualificationReport {
    #[serde(flatten)]
    pub body: QualificationBody,
    pub seal: ContentCommitment,
}

#[derive(Serialize)]
struct Policy<'a> {
    schema_version: u32,
    gate_basis: &'static str,
    declared_host_scope: &'a str,
    gates: Gates,
    options: EvalOptions,
}

/// Compare the complete matched population under caller-declared gates. A
/// changed label, unit, split, task configuration or population refuses before
/// either scorecard is computed. Probabilities and the named model/implementation
/// fields may differ; both complete identities remain independently committed.
#[allow(clippy::too_many_arguments)]
pub fn compare(candidate: &Dataset, baseline: &Dataset, key: &CommitmentKey,
    namespace: &str, host_scope: &str, gates: Gates, options: EvalOptions,
    limits: Limits) -> Result<QualificationReport, AssayError> {
    gates.validate(limits)?;
    options.validate(limits)?;
    candidate.validate_limits(limits)?;
    baseline.validate_limits(limits)?;
    if host_scope.is_empty() || host_scope.len() > 256
        || host_scope.chars().any(|ch| ch.is_control() || ch.is_whitespace()) {
        return Err(AssayError::Scope);
    }
    validate_pair(candidate, baseline)?;
    let rows = candidate.partition().locked_test().len();
    options.validate_rows(rows, limits)?;
    // Each arm performs its own marginal bootstrap. Price the combined work,
    // not two independent renewals of the caller's maximum draw allowance.
    if (rows as u64).checked_mul(options.bootstrap.resamples as u64)
        .and_then(|draws| draws.checked_mul(2))
        .is_none_or(|draws| draws > limits.max_bootstrap_draws) {
        return Err(AssayError::InputLimit);
    }
    let policy = Policy { schema_version: ASSAY_SCHEMA_VERSION, gate_basis: "point_estimates",
        declared_host_scope: host_scope, gates, options };
    let policy_commitment = commit(key, namespace, "qualification_policy_v1", &policy, limits)?;
    let candidate_report = super::evaluate(candidate, key, namespace, options, limits)?;
    let baseline_report = super::evaluate(baseline, key, namespace, options, limits)?;
    let decision = match (&candidate_report.measurement, &baseline_report.measurement) {
        (Measurement::Measured { metrics: current }, Measurement::Measured { metrics: prior }) => {
            let paired = paired_summary(candidate, baseline, rows);
            let comparisons = vec![
                upper("brier_increase", Some(paired.mean_brier_change), gates.max_brier_increase),
                upper("ece_increase", Some(current.ece - prior.ece), gates.max_ece_increase),
                upper("selective_risk", current.selective_risk, gates.max_selective_risk),
                lower("acceptance_rate", current.accepted as f64 / rows as f64, gates.min_acceptance_rate),
                lower("accepted_rows", current.accepted as f64, gates.minimum_accepted as f64),
            ];
            if comparisons.iter().all(|check| check.passed) { Decision::Passed { comparisons, paired } }
            else { Decision::Rejected { comparisons, paired } }
        }
        (Measurement::InsufficientData { .. }, Measurement::InsufficientData { .. }) =>
            Decision::InsufficientData { rows, minimum_samples: options.minimum_samples },
        _ => return Err(AssayError::Scope),
    };
    let body = QualificationBody { schema_version: ASSAY_SCHEMA_VERSION, task: "binary_event".to_owned(),
        evidence: ASSAY_EVIDENCE.to_owned(), activation_authority: false, receipt_grade: ReceiptGrade::AuditOnly,
        declared_host_scope: host_scope.to_owned(), host_scope_evidence: "caller_declared".to_owned(),
        gate_basis: "point_estimates".to_owned(), gates, policy_commitment,
        candidate: candidate_report, baseline: baseline_report, decision };
    let seal = commit(key, namespace, "qualification_report_v1", &body, limits)?;
    let report = QualificationReport { body, seal };
    super::render(&report, limits)?;
    Ok(report)
}

fn validate_pair(candidate: &Dataset, baseline: &Dataset) -> Result<(), AssayError> {
    let current = candidate.manifest();
    let prior = baseline.manifest();
    if current.schema_version != prior.schema_version || current.task != prior.task
        || current.dataset_id != prior.dataset_id || current.dataset_version != prior.dataset_version
        || current.named_population != prior.named_population || current.positive_label != prior.positive_label
        || current.negative_label != prior.negative_label || current.splits != prior.splits
        || candidate.rows().len() != baseline.rows().len() {
        return Err(AssayError::Scope);
    }
    // Compare the canonical identity as a whole after substituting only the
    // explicitly allowed candidate-model and execution-implementation fields.
    // This is a compatibility check, never a second partial identity key.
    let mut compatible = current.execution_identity.clone();
    let expected = &prior.execution_identity;
    compatible.logical_model_digest = expected.logical_model_digest;
    compatible.artifact_format.clone_from(&expected.artifact_format);
    compatible.quant_recipe.clone_from(&expected.quant_recipe);
    compatible.packing_set_digest = expected.packing_set_digest;
    compatible.numerics_profile.clone_from(&expected.numerics_profile);
    compatible.backend_semantic_version.clone_from(&expected.backend_semantic_version);
    compatible.host_class.clone_from(&expected.host_class);
    compatible.compiler_identity.clone_from(&expected.compiler_identity);
    if compatible != *expected { return Err(AssayError::Scope); }
    for (left, right) in candidate.rows().iter().zip(baseline.rows()) {
        if left.id != right.id || left.split != right.split || left.unit_sha256 != right.unit_sha256
            || left.positive != right.positive { return Err(AssayError::Scope); }
    }
    Ok(())
}

fn paired_summary(candidate: &Dataset, baseline: &Dataset, rows: usize) -> PairedSummary {
    let mut summary = PairedSummary { rows, mean_brier_change: 0.0,
        candidate_better: 0, candidate_worse: 0, equal: 0 };
    for (left, right) in candidate.rows_for_split(Split::LockedTest).zip(baseline.rows_for_split(Split::LockedTest)) {
        let label = if left.positive { 1.0 } else { 0.0 };
        let current_error = left.probability - label;
        let prior_error = right.probability - label;
        let change = current_error * current_error - prior_error * prior_error;
        summary.mean_brier_change += change;
        if change < 0.0 { summary.candidate_better += 1; }
        else if change > 0.0 { summary.candidate_worse += 1; }
        else { summary.equal += 1; }
    }
    summary.mean_brier_change /= rows as f64;
    summary
}

fn upper(name: &str, observed: Option<f64>, limit: f64) -> GateResult {
    GateResult { name: name.to_owned(), observed, limit, passed: observed.is_some_and(|value| value <= limit) }
}
fn lower(name: &str, observed: f64, limit: f64) -> GateResult {
    GateResult { name: name.to_owned(), observed: Some(observed), limit, passed: observed >= limit }
}
fn bounded(value: f64, minimum: f64, maximum: f64) -> bool {
    value.is_finite() && value >= minimum && value <= maximum
}
fn commit<T: Serialize>(key: &CommitmentKey, namespace: &str, kind: &str,
    value: &T, limits: Limits) -> Result<ContentCommitment, AssayError> {
    let bytes = super::render(value, limits)?;
    key.commit(CommitmentDomain::Configuration, namespace, kind, &bytes).map_err(|_| AssayError::Commitment)
}

#[cfg(test)] mod tests;
