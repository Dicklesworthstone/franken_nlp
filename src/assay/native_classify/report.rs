//! Private replay owners produce content-free evaluation and paired reports.
use serde::{Deserialize, Serialize};
use crate::{calibration::CalibrationError,
    native_engine::strict_int4::Int4Work,
    receipt::{CommitmentDomain, CommitmentKey, ContentCommitment, ReceiptGrade}};
use super::*;
use super::super::{DatasetMetadata, Measurement, qualify::{GateResult, Decision, PairedSummary}};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Gates {
    pub schema_version: u32,
    pub reliability: super::super::qualify::Gates,
    /// Baseline accuracy minus candidate accuracy. A negative limit requires improvement.
    pub max_accuracy_decrease: f64,
    /// Baseline macro F1 minus candidate macro F1. Undefined F1 never passes.
    pub max_macro_f1_decrease: f64,
}
impl Gates {
    pub fn validate(&self, limits: Limits) -> Result<(), AssayError> {
        self.reliability.validate(limits)?;
        if self.schema_version != ASSAY_SCHEMA_VERSION
            || !bounded(self.max_accuracy_decrease) || !bounded(self.max_macro_f1_decrease) {
            return Err(AssayError::Qualification);
        }
        Ok(())
    }
}
pub fn parse_gates(bytes: &[u8], limits: Limits) -> Result<Gates, AssayError> {
    limits.validate()?;
    if bytes.is_empty() || bytes.len() > limits.max_manifest_bytes.min(65_536) { return Err(AssayError::InputLimit); }
    let gates: Gates = dataset::strict_json(bytes, 128)?;
    gates.validate(limits)?; Ok(gates)
}
fn bounded(value: f64) -> bool { value.is_finite() && (-1.0..=1.0).contains(&value) }

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ClassMetrics {
    pub support: usize,
    pub predicted: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    /// None when neither truth nor predictions contain this class.
    pub f1: Option<f64>,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ClassificationMetrics {
    pub rows: usize,
    pub decided: usize,
    pub abstained: usize,
    pub true_positive: usize,
    pub false_positive: usize,
    pub true_negative: usize,
    pub false_negative: usize,
    pub abstained_positive: usize,
    pub abstained_negative: usize,
    /// Correct labels / ALL held-out rows; abstentions earn no accuracy credit.
    pub accuracy: f64,
    pub decided_accuracy: Option<f64>,
    pub positive: ClassMetrics,
    pub negative: ClassMetrics,
    /// Both class F1 values must be defined. No absent-class value is invented.
    pub macro_f1: Option<f64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Bindings {
    pub metadata: DatasetMetadata,
    pub suite: ContentCommitment,
    pub input: ContentCommitment,
    pub splits: SplitBindings,
    pub execution_collection: ContentCommitment,
    pub execution_collection_format: String,
    pub execution_collection_rows: usize,
    pub observation_collection: ContentCommitment,
    pub observation_collection_format: String,
    pub observation_collection_rows: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct EvaluationBody {
    pub schema_version: u32,
    pub task: String,
    pub evidence: String,
    pub activation_authority: bool,
    pub receipt_grade: ReceiptGrade,
    pub score_semantics: String,
    pub classification_semantics: String,
    pub bindings: Bindings,
    pub options: EvalOptions,
    pub measurement: Measurement,
    pub classification: Option<ClassificationMetrics>,
    pub executed_rows: usize,
    pub group_width: usize,
    pub model_work: Int4Work,
    pub rewound_positions: u64,
}
#[derive(Clone, Debug, Serialize)]
pub struct Evaluation {
    #[serde(flatten)]
    pub body: EvaluationBody,
    pub seal: ContentCommitment,
}
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
    pub total_model_work: Int4Work,
}
#[derive(Clone, Debug, Serialize)]
pub struct QualificationReport {
    #[serde(flatten)]
    pub body: QualificationBody,
    pub seal: ContentCommitment,
}

pub(crate) fn validate_scope(namespace: &str, host_scope: Option<&str>) -> Result<(), AssayError> {
    if !identifier(namespace) { return Err(AssayError::Commitment); }
    if host_scope.is_some_and(|scope| !identifier(scope)) { return Err(AssayError::Scope); }
    Ok(())
}

#[cfg(test)]
pub(crate) fn evaluate(replay: &NativeReplay, key: &CommitmentKey, namespace: &str,
    options: EvalOptions, limits: Limits) -> Result<Evaluation, AssayError> {
    evaluate_with_checkpoint(replay, key, namespace, options, limits, &mut || Ok(()))
}
pub(crate) fn evaluate_with_checkpoint<E>(replay: &NativeReplay, key: &CommitmentKey, namespace: &str,
    options: EvalOptions, limits: Limits, checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Evaluation, E>
where E: From<AssayError> + From<CalibrationError> {
    validate_scope(namespace, None)?; options.validate(limits)?; replay.sizes.validate(limits)?;
    checkpoint()?;
    let bindings = bindings(replay, key, namespace, checkpoint)?;
    let partition = replay.partition(checkpoint)?;
    let measurement = super::super::measure_partition_with_checkpoint(&partition, options, limits, Ok, checkpoint)?;
    let classification = match &measurement {
        Measurement::Measured { .. } => Some(classification(replay, checkpoint)?),
        Measurement::InsufficientData { .. } => None,
    };
    drop(partition);
    checkpoint()?;
    let body = EvaluationBody { schema_version: ASSAY_SCHEMA_VERSION, task: NATIVE_TASK.to_owned(),
        evidence: NATIVE_EVIDENCE.to_owned(), activation_authority: false, receipt_grade: ReceiptGrade::AuditOnly,
        score_semantics: "uncalibrated_positive_candidate_weight_over_two_completed_labels; full_vocabulary_scored_eos".to_owned(),
        classification_semantics: "native_sequence_score_argmax_then_lexical_label_id; task_policy_abstention; accuracy_correct_over_all_rows".to_owned(),
        bindings, options, measurement, classification, executed_rows: replay.rows.len(), group_width: replay.group_width,
        model_work: replay.model_work, rewound_positions: replay.rewound_positions };
    let seal = commit(key, namespace, "native_binary_evaluation_v1", &body, limits)?;
    let result = Evaluation { body, seal };
    super::super::render(&result, limits)?; checkpoint()?; Ok(result)
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn compare(candidate: &NativeReplay, baseline: &NativeReplay, key: &CommitmentKey,
    namespace: &str, host_scope: &str, gates: Gates, options: EvalOptions, limits: Limits)
    -> Result<QualificationReport, AssayError> {
    compare_with_checkpoint(candidate, baseline, key, namespace, host_scope, gates, options, limits, &mut || Ok(()))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compare_with_checkpoint<E>(candidate: &NativeReplay, baseline: &NativeReplay,
    key: &CommitmentKey, namespace: &str, host_scope: &str, gates: Gates, options: EvalOptions,
    limits: Limits, checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<QualificationReport, E>
where E: From<AssayError> + From<CalibrationError> {
    validate_scope(namespace, Some(host_scope))?; gates.validate(limits)?; options.validate(limits)?;
    candidate.sizes.validate(limits)?; baseline.sizes.validate(limits)?;
    validate_pair(candidate, baseline, checkpoint)?;
    let count = candidate.manifest.splits.locked_test.len();
    options.validate_rows(count, limits)?;
    if (count as u64).checked_mul(options.bootstrap.resamples as u64).and_then(|n| n.checked_mul(2))
        .is_none_or(|draws| draws > limits.max_bootstrap_draws) { return Err(AssayError::InputLimit.into()); }
    #[derive(Serialize)]
    struct Policy<'a> { schema_version: u32, gate_basis: &'static str, declared_host_scope: &'a str,
        gates: &'a Gates, options: EvalOptions }
    let policy = Policy { schema_version: ASSAY_SCHEMA_VERSION, gate_basis: "point_estimates",
        declared_host_scope: host_scope, gates: &gates, options };
    let policy_commitment = commit(key, namespace, "native_binary_qualification_policy_v1", &policy, limits)?;
    checkpoint()?;
    let current = evaluate_with_checkpoint(candidate, key, namespace, options, limits, checkpoint)?;
    let prior = evaluate_with_checkpoint(baseline, key, namespace, options, limits, checkpoint)?;
    let decision = match (&current.body.measurement, &prior.body.measurement) {
        (Measurement::Measured { metrics: a }, Measurement::Measured { metrics: b }) => {
            let a_class = current.body.classification.as_ref().ok_or(AssayError::Qualification)?;
            let b_class = prior.body.classification.as_ref().ok_or(AssayError::Qualification)?;
            let paired = paired_summary(candidate, baseline, count, checkpoint)?;
            let reliability = &gates.reliability;
            let comparisons = vec![
                upper("brier_increase", Some(paired.mean_brier_change), reliability.max_brier_increase),
                upper("ece_increase", Some(a.ece - b.ece), reliability.max_ece_increase),
                upper("selective_risk", a.selective_risk, reliability.max_selective_risk),
                lower("acceptance_rate", a.accepted as f64 / count as f64, reliability.min_acceptance_rate),
                lower("accepted_rows", a.accepted as f64, reliability.minimum_accepted as f64),
                upper("accuracy_decrease", Some(b_class.accuracy - a_class.accuracy), gates.max_accuracy_decrease),
                upper("macro_f1_decrease", b_class.macro_f1.zip(a_class.macro_f1).map(|(b, a)| b - a),
                    gates.max_macro_f1_decrease),
            ];
            if comparisons.iter().all(|gate| gate.passed) { Decision::Passed { comparisons, paired } }
            else { Decision::Rejected { comparisons, paired } }
        }
        (Measurement::InsufficientData { .. }, Measurement::InsufficientData { .. }) =>
            Decision::InsufficientData { rows: count, minimum_samples: options.minimum_samples },
        _ => return Err(AssayError::Qualification.into()),
    };
    let total_model_work = candidate.model_work.checked_add(baseline.model_work).map_err(|_| AssayError::InputLimit)?;
    let body = QualificationBody { schema_version: ASSAY_SCHEMA_VERSION, task: NATIVE_TASK.to_owned(),
        evidence: NATIVE_EVIDENCE.to_owned(), activation_authority: false, receipt_grade: ReceiptGrade::AuditOnly,
        declared_host_scope: host_scope.to_owned(), host_scope_evidence: "caller_declared".to_owned(),
        gate_basis: "point_estimates".to_owned(), gates, policy_commitment, candidate: current, baseline: prior,
        decision, total_model_work };
    checkpoint()?;
    let seal = commit(key, namespace, "native_binary_qualification_v1", &body, limits)?;
    let result = QualificationReport { body, seal };
    super::super::render(&result, limits)?; checkpoint()?; Ok(result)
}

fn bindings<E>(replay: &NativeReplay, key: &CommitmentKey, namespace: &str,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Bindings, E>
where E: From<AssayError> {
    checkpoint()?;
    let suite = crate::canonjson::canonical_bytes(&replay.manifest).map_err(|_| AssayError::Commitment)?;
    let suite = key.commit(CommitmentDomain::Configuration, namespace, "native_binary_suite_v1", &suite)
        .map_err(|_| AssayError::Commitment)?;
    let mut split = |name: &str, ids: &[String]| -> Result<ContentCommitment, E> {
        checkpoint()?;
        let bytes = crate::canonjson::canonical_bytes(ids).map_err(|_| AssayError::Commitment)?;
        Ok(key.commit(CommitmentDomain::Configuration, namespace, name, &bytes).map_err(|_| AssayError::Commitment)?)
    };
    let splits = SplitBindings {
        development: split("native_development_ids_v1", &replay.manifest.splits.development)?,
        calibration: split("native_calibration_ids_v1", &replay.manifest.splits.calibration)?,
        locked_test: split("native_locked_test_ids_v1", &replay.manifest.splits.locked_test)?,
    };
    let id = &replay.first_identity;
    let metadata = DatasetMetadata { dataset_id: replay.manifest.dataset_id.clone(),
        dataset_version: replay.manifest.dataset_version.clone(), named_population: replay.manifest.named_population.clone(),
        positive_label: replay.manifest.positive_label.id.clone(), negative_label: replay.manifest.negative_label.id.clone(),
        task_spec: id.task_spec.clone(), model_recipe_id: id.quant_recipe.clone(),
        prompt_template_sha256: id.template_digest, numerics_profile: id.numerics_profile.label(),
        thinking_mode: "disabled".to_owned() };
    let input = key.commit(CommitmentDomain::Input, namespace, "native_source_ndjson_sha256_hex_v1",
        replay.input_digest.to_hex().as_bytes()).map_err(|_| AssayError::Commitment)?;
    // Commit the binary digest of the specified framed full-identity sequence.
    // Never publish that raw digest or call it one synthetic ExecutionIdentity.
    let execution_collection = key.commit(CommitmentDomain::ExecutionIdentity, namespace,
        "native_execution_collection_sha256_binary_v1", &replay.executions).map_err(|_| AssayError::Commitment)?;
    let observation_collection = key.commit(CommitmentDomain::Output, namespace,
        "native_observation_collection_sha256_binary_v1", &replay.observations).map_err(|_| AssayError::Commitment)?;
    checkpoint()?;
    Ok(Bindings { metadata, suite, input, splits, execution_collection,
        execution_collection_format: EXECUTION_COLLECTION_FORMAT.to_owned(), execution_collection_rows: replay.rows.len(),
        observation_collection, observation_collection_format: OBSERVATION_COLLECTION_FORMAT.to_owned(),
        observation_collection_rows: replay.rows.len() })
}

fn classification<E>(replay: &NativeReplay, checkpoint: &mut impl FnMut() -> Result<(), E>)
    -> Result<ClassificationMetrics, E> {
    let (mut tp, mut fp, mut tn, mut fn_, mut ap, mut an) = (0, 0, 0, 0, 0, 0);
    for (index, row) in replay.rows.iter().enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        if row.split != Split::LockedTest { continue; }
        match (row.positive, row.predicted) {
            (true, Some(true)) => tp += 1, (false, Some(true)) => fp += 1,
            (false, Some(false)) => tn += 1, (true, Some(false)) => fn_ += 1,
            (true, None) => ap += 1, (false, None) => an += 1,
        }
    }
    let decided = tp + fp + tn + fn_; let abstained = ap + an; let rows = decided + abstained;
    let positive = class_metrics(tp, fp, fn_ + ap); let negative = class_metrics(tn, fn_, fp + an);
    let macro_f1 = positive.f1.zip(negative.f1).map(|(a, b)| (a + b) / 2.0);
    checkpoint()?;
    Ok(ClassificationMetrics { rows, decided, abstained, true_positive: tp, false_positive: fp,
        true_negative: tn, false_negative: fn_, abstained_positive: ap, abstained_negative: an,
        accuracy: (tp + tn) as f64 / rows as f64, decided_accuracy: ratio(tp + tn, decided),
        positive, negative, macro_f1 })
}
fn class_metrics(tp: usize, fp: usize, missed: usize) -> ClassMetrics {
    ClassMetrics { support: tp + missed, predicted: tp + fp, precision: ratio(tp, tp + fp),
        recall: ratio(tp, tp + missed), f1: ratio(2 * tp, 2 * tp + fp + missed) }
}
fn ratio(a: usize, b: usize) -> Option<f64> { (b != 0).then(|| a as f64 / b as f64) }

fn validate_pair<E>(a: &NativeReplay, b: &NativeReplay, checkpoint: &mut impl FnMut() -> Result<(), E>)
    -> Result<(), E> where E: From<AssayError> {
    if a.manifest_digest != b.manifest_digest || a.input_digest != b.input_digest
        || a.compatible_executions != b.compatible_executions || a.rows.len() != b.rows.len() {
        return Err(AssayError::Scope.into());
    }
    for (index, (a, b)) in a.rows.iter().zip(&b.rows).enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        if a.id != b.id || a.split != b.split || a.positive != b.positive || a.unit != b.unit {
            return Err(AssayError::Scope.into());
        }
    }
    checkpoint()?; Ok(())
}
fn paired_summary<E>(a: &NativeReplay, b: &NativeReplay, count: usize,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<PairedSummary, E> {
    let mut summary = PairedSummary { rows: count, mean_brier_change: 0.0,
        candidate_better: 0, candidate_worse: 0, equal: 0 };
    for (index, (a, b)) in a.rows.iter().zip(&b.rows).enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        if a.split != Split::LockedTest { continue; }
        let truth = if a.positive { 1.0 } else { 0.0 };
        let candidate_error = a.probability - truth; let baseline_error = b.probability - truth;
        let change = candidate_error * candidate_error - baseline_error * baseline_error;
        summary.mean_brier_change += change;
        if change < 0.0 { summary.candidate_better += 1; }
        else if change > 0.0 { summary.candidate_worse += 1; } else { summary.equal += 1; }
    }
    summary.mean_brier_change /= count as f64;
    checkpoint()?; Ok(summary)
}
fn upper(name: &str, observed: Option<f64>, limit: f64) -> GateResult {
    GateResult { name: name.to_owned(), observed, limit, passed: observed.is_some_and(|value| value <= limit) }
}
fn lower(name: &str, observed: f64, limit: f64) -> GateResult {
    GateResult { name: name.to_owned(), observed: Some(observed), limit, passed: observed >= limit }
}
fn commit<T: Serialize>(key: &CommitmentKey, namespace: &str, kind: &str,
    value: &T, limits: Limits) -> Result<ContentCommitment, AssayError> {
    let bytes = super::super::render(value, limits)?;
    key.commit(CommitmentDomain::Configuration, namespace, kind, &bytes).map_err(|_| AssayError::Commitment)
}
