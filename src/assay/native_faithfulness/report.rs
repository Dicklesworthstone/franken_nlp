//! Content-free reports; native agreement never grants model activation.
use serde::{Deserialize, Serialize};
use crate::{native_engine::strict_int4::Int4Work,
    receipt::{CommitmentDomain, CommitmentKey, ContentCommitment, ReceiptGrade},
    tasks::{ir::ScoreSpace, judge::FaithfulnessPolicy}};
use crate::assay::qualify::GateResult;
use super::*;
pub use super::metrics::{EvalOptions, Measurement, PairedSummary, Quality, QualityIntervals};

/// Decreases are baseline minus candidate; Brier increase is candidate minus
/// baseline. Gates use point estimates. Undefined class scores cannot pass.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Gates {
    pub schema_version: u32,
    pub max_accuracy_decrease: f64,
    pub max_macro_f1_decrease: f64,
    pub max_brier_increase: f64,
    pub max_coverage_decrease: f64,
}
impl Gates {
    pub fn validate(self, limits: Limits) -> Result<(), AssayError> {
        limits.validate()?;
        if self.schema_version != ASSAY_SCHEMA_VERSION || [self.max_accuracy_decrease,
            self.max_macro_f1_decrease, self.max_brier_increase, self.max_coverage_decrease]
            .iter().any(|value| !value.is_finite() || !(-1.0..=1.0).contains(value)) {
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
    pub evidence_semantics: String,
    pub bootstrap_semantics: String,
    pub bindings: Bindings,
    pub policy: FaithfulnessPolicy,
    pub options: EvalOptions,
    pub measurement: Measurement,
    pub validated_rows: usize,
    pub executed_rows: usize,
    pub executed_heads: usize,
    pub group_width: usize,
    pub model_work: Int4Work,
}
#[derive(Clone, Debug, Serialize)]
pub struct Evaluation {
    #[serde(flatten)] pub body: EvaluationBody,
    pub seal: ContentCommitment,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum Decision {
    Passed { comparisons: Vec<GateResult>, paired: PairedSummary },
    Rejected { comparisons: Vec<GateResult>, paired: PairedSummary },
    InsufficientData { claims: usize, source_clusters: usize, minimum_sources: usize },
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
    #[serde(flatten)] pub body: QualificationBody,
    pub seal: ContentCommitment,
}
pub(crate) fn validate_scope(namespace: &str, host_scope: Option<&str>) -> Result<(), AssayError> {
    if !identifier(namespace) { return Err(AssayError::Commitment); }
    if host_scope.is_some_and(|value| !identifier(value)) { return Err(AssayError::Scope); }
    Ok(())
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
            let comparisons = vec![
                upper("accuracy_decrease", Some(-paired.accuracy_change), gates.max_accuracy_decrease),
                upper("macro_f1_decrease", paired.macro_f1_change.map(|value| -value), gates.max_macro_f1_decrease),
                upper("brier_increase", Some(paired.brier_change), gates.max_brier_increase),
                upper("coverage_decrease", Some(-paired.coverage_change), gates.max_coverage_decrease),
            ];
            if comparisons.iter().all(|value| value.passed) { Decision::Passed { comparisons, paired } }
            else { Decision::Rejected { comparisons, paired } }
        }
        (Measurement::InsufficientData { claims, source_clusters, minimum_sources },
            Measurement::InsufficientData { .. }, None) => Decision::InsufficientData {
                claims: *claims, source_clusters: *source_clusters, minimum_sources: *minimum_sources },
        _ => return Err(AssayError::Scope.into()),
    };
    #[derive(Serialize)]
    struct Policy<'a> { schema_version: u32, task: &'static str, gate_basis: &'static str,
        host_scope: &'a str, gates: Gates, options: EvalOptions, faithfulness: FaithfulnessPolicy }
    let policy = Policy { schema_version: ASSAY_SCHEMA_VERSION, task: NATIVE_TASK, gate_basis: "point_estimates",
        host_scope, gates, options, faithfulness: candidate.manifest.policy };
    let policy_commitment = commit(key, namespace, "native_faithfulness_qualification_policy_v1", &policy, limits)?;
    let body = QualificationBody { schema_version: ASSAY_SCHEMA_VERSION, task: NATIVE_TASK.to_owned(),
        evidence: NATIVE_EVIDENCE.to_owned(), activation_authority: false, receipt_grade: ReceiptGrade::AuditOnly,
        declared_host_scope: host_scope.to_owned(), host_scope_evidence: "caller_declared".to_owned(),
        gate_basis: "point_estimates".to_owned(), gates, policy_commitment, candidate: current, baseline: prior, decision,
        total_model_work: candidate.model_work.checked_add(baseline.model_work).map_err(|_| AssayError::InputLimit)? };
    checkpoint()?;
    let seal = commit(key, namespace, "native_faithfulness_qualification_v1", &body, limits)?;
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
        score_semantics: "full_source_candidate_weights_uncalibrated; brier_half_sum_squared_one_hot_errors; decisions_preserve_all_native_evidence_policy_abstentions; human_gold_is_caller_supplied".to_owned(),
        evidence_semantics: "complete_source_and_all_evidence_windows; byte_verified_quotes_prove_source_membership_only; correlated_model_judgments_are_not_truth_or_entailment_certificates".to_owned(),
        bootstrap_semantics: "claim_weighted_metrics; resample_whole_exact_source_clusters_with_all_claims; same_source_disjoint_across_splits; paired_arms_share_draws; minimum_samples_counts_sources; undefined_replicate_makes_interval_absent; class_breakdowns_are_point_estimates".to_owned(),
        bindings: bindings(replay, key, namespace, checkpoint)?, policy: replay.manifest.policy, options, measurement,
        validated_rows: replay.sizes.rows, executed_rows: replay.rows.len(), executed_heads: replay.heads,
        group_width: replay.group_width, model_work: replay.model_work };
    let seal = commit(key, namespace, "native_faithfulness_evaluation_v1", &body, limits)?;
    let result = Evaluation { body, seal };
    super::super::render(&result, limits)?; checkpoint()?; Ok(result)
}
fn validate_pair<E: From<AssayError>>(a: &NativeReplay, b: &NativeReplay,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), E> {
    if a.manifest_digest != b.manifest_digest || a.input_digest != b.input_digest
        || a.compatible_executions != b.compatible_executions || a.rows.len() != b.rows.len() {
        return Err(AssayError::Scope.into());
    }
    for (index, (a, b)) in a.rows.iter().zip(&b.rows).enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        if a.id != b.id || a.source != b.source || a.claim != b.claim || a.gold != b.gold {
            return Err(AssayError::Scope.into());
        }
    }
    checkpoint()?; Ok(())
}
fn bindings<E: From<AssayError>>(replay: &NativeReplay, key: &CommitmentKey, namespace: &str,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Bindings, E> {
    checkpoint()?;
    let suite = crate::canonjson::canonical_bytes(&replay.manifest).map_err(|_| AssayError::Commitment)?;
    let suite = key.commit(CommitmentDomain::Configuration, namespace, "native_faithfulness_suite_v1", &suite)
        .map_err(|_| AssayError::Commitment)?;
    let input = key.commit(CommitmentDomain::Input, namespace, "native_faithfulness_cases_sha256_hex_v1",
        replay.input_digest.to_hex().as_bytes()).map_err(|_| AssayError::Commitment)?;
    let mut split = |name: &str, ids: &[String]| -> Result<ContentCommitment, E> {
        checkpoint()?;
        let bytes = crate::canonjson::canonical_bytes(ids).map_err(|_| AssayError::Commitment)?;
        Ok(key.commit(CommitmentDomain::Configuration, namespace, name, &bytes).map_err(|_| AssayError::Commitment)?)
    };
    let splits = SplitBindings { development: split("native_faithfulness_development_ids_v1", &replay.manifest.splits.development)?,
        calibration: split("native_faithfulness_calibration_ids_v1", &replay.manifest.splits.calibration)?,
        locked_test: split("native_faithfulness_locked_test_ids_v1", &replay.manifest.splits.locked_test)? };
    let identity = &replay.first_identity;
    let metadata = NativeMetadata { dataset_id: replay.manifest.dataset_id.clone(), dataset_version: replay.manifest.dataset_version.clone(),
        named_population: replay.manifest.named_population.clone(), task_spec: identity.task_spec.clone(),
        model_recipe_id: identity.quant_recipe.clone(), prompt_template_sha256: identity.template_digest,
        numerics_profile: identity.numerics_profile.label(), thinking_mode: "disabled".to_owned() };
    let execution_collection = key.commit(CommitmentDomain::ExecutionIdentity, namespace,
        "native_faithfulness_execution_collection_sha256_binary_v1", &replay.executions).map_err(|_| AssayError::Commitment)?;
    let observation_collection = key.commit(CommitmentDomain::Output, namespace,
        "native_faithfulness_observation_collection_sha256_binary_v1", &replay.observations).map_err(|_| AssayError::Commitment)?;
    checkpoint()?;
    Ok(Bindings { metadata, suite, input, splits, execution_collection,
        execution_collection_format: EXECUTION_COLLECTION_FORMAT.to_owned(), execution_collection_rows: replay.rows.len(),
        observation_collection, observation_collection_format: OBSERVATION_COLLECTION_FORMAT.to_owned(),
        observation_collection_rows: replay.rows.len() })
}
fn upper(name: &str, observed: Option<f64>, limit: f64) -> GateResult {
    GateResult { name: name.to_owned(), observed, limit, passed: observed.is_some_and(|value| value <= limit) }
}
fn commit<T: Serialize>(key: &CommitmentKey, namespace: &str, kind: &str, value: &T, limits: Limits)
    -> Result<ContentCommitment, AssayError> {
    let bytes = super::super::render(value, limits)?;
    key.commit(CommitmentDomain::Configuration, namespace, kind, &bytes).map_err(|_| AssayError::Commitment)
}
