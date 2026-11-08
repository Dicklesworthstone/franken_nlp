//! Content-free, caller-scoped native reports with no activation authority.
use serde::{Deserialize, Serialize};
use crate::{native_engine::strict_int4::Int4Work,
    receipt::{CommitmentDomain, CommitmentKey, ContentCommitment, ReceiptGrade},
    tasks::{ir::ScoreSpace, sentiment::SentimentPolicy}};
use crate::assay::qualify::GateResult;
use super::*;

/// Each requested gate must pass on EVERY requested axis. No average over
/// axes can hide degradation. Error increases are candidate minus baseline;
/// correlation/coverage decreases are baseline minus candidate.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Gates {
    pub schema_version: u32,
    pub max_mae_increase: f64,
    pub max_rmse_increase: f64,
    pub max_coverage_decrease: f64,
    /// Omit or null to leave rank out of the gate. If requested, an undefined
    /// candidate or baseline correlation refuses qualification for that axis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_spearman_decrease: Option<f64>,
    /// Conditional error alone cannot show improved whole-population quality;
    /// the independent coverage gate always remains required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_estimated_mae_increase: Option<f64>,
}
impl Gates {
    pub fn validate(self, limits: Limits) -> Result<(), AssayError> {
        limits.validate()?;
        let error_bound = |value: f64| value.is_finite() && (-2.0..=2.0).contains(&value);
        if self.schema_version != ASSAY_SCHEMA_VERSION || !error_bound(self.max_mae_increase)
            || !error_bound(self.max_rmse_increase) || !self.max_coverage_decrease.is_finite()
            || !(-1.0..=1.0).contains(&self.max_coverage_decrease)
            || self.max_spearman_decrease.is_some_and(|value| !error_bound(value))
            || self.max_estimated_mae_increase.is_some_and(|value| !error_bound(value)) {
            return Err(AssayError::Qualification);
        }
        Ok(())
    }
}
pub fn parse_gates(bytes: &[u8], limits: Limits) -> Result<Gates, AssayError> {
    limits.validate()?;
    if bytes.is_empty() || bytes.len() > limits.max_manifest_bytes.min(65_536) { return Err(AssayError::InputLimit); }
    let gates: Gates = dataset::strict_json(bytes, 128)?; gates.validate(limits)?; Ok(gates)
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeMetadata {
    pub dataset_id: String,
    pub dataset_version: String,
    pub named_population: String,
    pub task_spec: String,
    pub model_recipe_id: String,
    pub prompt_template_sha256: crate::execution_identity::Sha256Digest,
    pub numerics_profile: String,
    pub thinking_mode: String,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Bindings {
    pub metadata: NativeMetadata,
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
    pub score_space: ScoreSpace,
    pub score_semantics: String,
    pub gold_semantics: String,
    pub bootstrap_semantics: String,
    pub bindings: Bindings,
    pub axes: Vec<SentimentAxis>,
    pub policy: SentimentPolicy,
    pub options: EvalOptions,
    pub measurement: Measurement,
    pub validated_rows: usize,
    pub executed_rows: usize,
    pub executed_heads: usize,
    pub group_width: usize,
    pub model_work: Int4Work,
}
#[derive(Clone, Debug, Serialize)]
pub struct Evaluation { #[serde(flatten)] pub body: EvaluationBody, pub seal: ContentCommitment }
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum Decision {
    Passed { comparisons: Vec<GateResult>, paired: PairedSummary },
    Rejected { comparisons: Vec<GateResult>, paired: PairedSummary },
    InsufficientData { documents: usize, minimum_documents: usize },
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
pub struct QualificationReport { #[serde(flatten)] pub body: QualificationBody, pub seal: ContentCommitment }

pub(crate) fn validate_scope(namespace: &str, host_scope: Option<&str>) -> Result<(), AssayError> {
    if !identifier(namespace) { return Err(AssayError::Commitment); }
    if host_scope.is_some_and(|value| !identifier(value)) { return Err(AssayError::Scope); } Ok(())
}
pub(crate) fn evaluate_with_checkpoint<E: From<AssayError>>(replay: &NativeReplay, key: &CommitmentKey,
    namespace: &str, options: EvalOptions, limits: Limits, checkpoint: &mut impl FnMut() -> Result<(), E>)
    -> Result<Evaluation, E> {
    validate_scope(namespace, None)?; replay.sizes.validate(limits)?;
    let (measurement, _, _) = metrics::measure(replay, None, options, limits, checkpoint)?;
    evaluation(replay, key, namespace, options, limits, measurement, checkpoint)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn compare_with_checkpoint<E: From<AssayError>>(candidate: &NativeReplay, baseline: &NativeReplay,
    key: &CommitmentKey, namespace: &str, host_scope: &str, gates: Gates, options: EvalOptions,
    limits: Limits, checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<QualificationReport, E> {
    validate_scope(namespace, Some(host_scope))?; gates.validate(limits)?;
    candidate.sizes.validate(limits)?; baseline.sizes.validate(limits)?;
    validate_pair(candidate, baseline, checkpoint)?;
    let (current, prior, paired) = metrics::measure(candidate, Some(baseline), options, limits, checkpoint)?;
    let current = evaluation(candidate, key, namespace, options, limits, current, checkpoint)?;
    let prior = evaluation(baseline, key, namespace, options, limits, prior.ok_or(AssayError::Scope)?, checkpoint)?;
    let decision = match (&current.body.measurement, &prior.body.measurement, paired) {
        (Measurement::Measured { .. }, Measurement::Measured { .. }, Some(paired)) => {
            let mut comparisons = reserved(paired.axes.len() * 5)?;
            for change in &paired.axes {
                checkpoint()?;
                comparisons.push(upper(change.axis, "mae_increase", Some(change.mae_change), gates.max_mae_increase));
                comparisons.push(upper(change.axis, "rmse_increase", Some(change.rmse_change), gates.max_rmse_increase));
                comparisons.push(upper(change.axis, "coverage_decrease", Some(-change.estimated_coverage_change), gates.max_coverage_decrease));
                if let Some(limit) = gates.max_spearman_decrease {
                    comparisons.push(upper(change.axis, "spearman_decrease", change.spearman_change.map(|value| -value), limit));
                }
                if let Some(limit) = gates.max_estimated_mae_increase {
                    comparisons.push(upper(change.axis, "estimated_mae_increase", change.estimated_mae_change, limit));
                }
            }
            if comparisons.iter().all(|value| value.passed) { Decision::Passed { comparisons, paired } }
            else { Decision::Rejected { comparisons, paired } }
        }
        (Measurement::InsufficientData { documents, minimum_documents }, Measurement::InsufficientData { .. }, None) =>
            Decision::InsufficientData { documents: *documents, minimum_documents: *minimum_documents },
        _ => return Err(AssayError::Scope.into()),
    };
    #[derive(Serialize)]
    struct Policy<'a> { schema_version: u32, task: &'static str, gate_basis: &'static str, host_scope: &'a str,
        gates: Gates, options: EvalOptions, axes: &'a [SentimentAxis], sentiment: SentimentPolicy }
    let policy = Policy { schema_version: ASSAY_SCHEMA_VERSION, task: NATIVE_TASK, gate_basis: "per_axis_point_estimates",
        host_scope, gates, options, axes: &candidate.manifest.axes, sentiment: candidate.manifest.policy };
    let policy_commitment = commit(key, namespace, "native_sentiment_qualification_policy_v1", &policy, limits)?;
    let body = QualificationBody { schema_version: ASSAY_SCHEMA_VERSION, task: NATIVE_TASK.to_owned(),
        evidence: NATIVE_EVIDENCE.to_owned(), activation_authority: false, receipt_grade: ReceiptGrade::AuditOnly,
        declared_host_scope: host_scope.to_owned(), host_scope_evidence: "caller_declared".to_owned(),
        gate_basis: "per_axis_point_estimates".to_owned(), gates, policy_commitment, candidate: current, baseline: prior, decision,
        total_model_work: candidate.model_work.checked_add(baseline.model_work).map_err(|_| AssayError::InputLimit)? };
    checkpoint()?;
    let seal = commit(key, namespace, "native_sentiment_qualification_v1", &body, limits)?;
    let result = QualificationReport { body, seal };
    super::super::render(&result, limits)?; checkpoint()?; Ok(result)
}
#[allow(clippy::too_many_arguments)]
fn evaluation<E: From<AssayError>>(replay: &NativeReplay, key: &CommitmentKey, namespace: &str,
    options: EvalOptions, limits: Limits, measurement: Measurement, checkpoint: &mut impl FnMut() -> Result<(), E>)
    -> Result<Evaluation, E> {
    checkpoint()?;
    let body = EvaluationBody { schema_version: ASSAY_SCHEMA_VERSION, task: NATIVE_TASK.to_owned(),
        evidence: NATIVE_EVIDENCE.to_owned(), activation_authority: false, receipt_grade: ReceiptGrade::AuditOnly,
        score_space: ScoreSpace::FullVocabSequenceLogprob,
        score_semantics: "five_bin_candidate_weights_uncalibrated; descriptive_distribution_means_include_abstentions; estimated_mae_is_conditional_on_the_frozen_task_policy; no_captured_mass_or_psychological_measurement".to_owned(),
        gold_semantics: "caller_supplied_human_ratings_in_sentiment_v1_normalized_coordinate; value_milli_divided_by_1000; low_anchor_minus_one_high_anchor_plus_one; no_rounding_to_model_bins; mae_and_rmse_range_zero_to_two".to_owned(),
        bootstrap_semantics: "resample_whole_unique_documents; axes_and_paired_arms_reuse_identical_seeded_document_draws; minimum_samples_counts_documents; all_axis_arm_draws_share_one_budget; undefined_replicate_makes_interval_absent; spearman_is_tie_aware_point_estimate_only".to_owned(),
        bindings: bindings(replay, key, namespace, checkpoint)?, axes: replay.manifest.axes.clone(),
        policy: replay.manifest.policy, options, measurement, validated_rows: replay.sizes.rows,
        executed_rows: replay.rows.len(), executed_heads: replay.heads, group_width: replay.group_width,
        model_work: replay.model_work };
    let seal = commit(key, namespace, "native_sentiment_evaluation_v1", &body, limits)?;
    let result = Evaluation { body, seal }; super::super::render(&result, limits)?; checkpoint()?; Ok(result)
}
fn validate_pair<E: From<AssayError>>(a: &NativeReplay, b: &NativeReplay,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), E> {
    if a.manifest_digest != b.manifest_digest || a.input_digest != b.input_digest
        || a.compatible_executions != b.compatible_executions || a.rows.len() != b.rows.len() {
        return Err(AssayError::Scope.into());
    }
    for (index, (a, b)) in a.rows.iter().zip(&b.rows).enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        if a.id != b.id || a.source != b.source || a.gold != b.gold { return Err(AssayError::Scope.into()); }
    }
    checkpoint()?; Ok(())
}
fn bindings<E: From<AssayError>>(replay: &NativeReplay, key: &CommitmentKey, namespace: &str,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Bindings, E> {
    checkpoint()?;
    let suite = crate::canonjson::canonical_bytes(&replay.manifest).map_err(|_| AssayError::Commitment)?;
    let suite = key.commit(CommitmentDomain::Configuration, namespace, "native_sentiment_suite_v1", &suite)
        .map_err(|_| AssayError::Commitment)?;
    let input = key.commit(CommitmentDomain::Input, namespace, "native_sentiment_cases_sha256_hex_v1",
        replay.input_digest.to_hex().as_bytes()).map_err(|_| AssayError::Commitment)?;
    let mut split = |name: &str, ids: &[String]| -> Result<ContentCommitment, E> {
        checkpoint()?;
        let bytes = crate::canonjson::canonical_bytes(ids).map_err(|_| AssayError::Commitment)?;
        Ok(key.commit(CommitmentDomain::Configuration, namespace, name, &bytes).map_err(|_| AssayError::Commitment)?)
    };
    let splits = SplitBindings { development: split("native_sentiment_development_ids_v1", &replay.manifest.splits.development)?,
        calibration: split("native_sentiment_calibration_ids_v1", &replay.manifest.splits.calibration)?,
        locked_test: split("native_sentiment_locked_test_ids_v1", &replay.manifest.splits.locked_test)? };
    let identity = &replay.first_identity;
    let metadata = NativeMetadata { dataset_id: replay.manifest.dataset_id.clone(), dataset_version: replay.manifest.dataset_version.clone(),
        named_population: replay.manifest.named_population.clone(), task_spec: identity.task_spec.clone(),
        model_recipe_id: identity.quant_recipe.clone(), prompt_template_sha256: identity.template_digest,
        numerics_profile: identity.numerics_profile.label(), thinking_mode: "disabled".to_owned() };
    let execution_collection = key.commit(CommitmentDomain::ExecutionIdentity, namespace,
        "native_sentiment_execution_collection_sha256_binary_v1", &replay.executions).map_err(|_| AssayError::Commitment)?;
    let observation_collection = key.commit(CommitmentDomain::Output, namespace,
        "native_sentiment_observation_collection_sha256_binary_v1", &replay.observations).map_err(|_| AssayError::Commitment)?;
    checkpoint()?;
    Ok(Bindings { metadata, suite, input, splits, execution_collection,
        execution_collection_format: EXECUTION_COLLECTION_FORMAT.to_owned(), execution_collection_rows: replay.rows.len(),
        observation_collection, observation_collection_format: OBSERVATION_COLLECTION_FORMAT.to_owned(),
        observation_collection_rows: replay.rows.len() })
}
fn upper(axis: SentimentAxis, name: &str, observed: Option<f64>, limit: f64) -> GateResult {
    GateResult { name: format!("{}.{}", axis_name(axis), name), observed, limit,
        passed: observed.is_some_and(|value| value <= limit) }
}
fn commit<T: Serialize>(key: &CommitmentKey, namespace: &str, kind: &str, value: &T, limits: Limits)
    -> Result<ContentCommitment, AssayError> {
    let bytes = super::super::render(value, limits)?;
    key.commit(CommitmentDomain::Configuration, namespace, kind, &bytes).map_err(|_| AssayError::Commitment)
}
