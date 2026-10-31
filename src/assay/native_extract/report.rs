//! Content-free audit reports over sealed native exact-JSON extraction.
use serde::{Deserialize, Serialize};
use crate::{native_engine::strict_int4::Int4Work,
    receipt::{CommitmentDomain, CommitmentKey, ContentCommitment, ReceiptGrade},
    tasks::ir::ScoreSpace};
use super::*;
use super::super::{field_metrics::{self, Observation}, qualify::GateResult};
pub use super::super::field_metrics::{EvalOptions, FieldIntervals, FieldMetrics, FieldScores,
    LanguageMetrics, Measurement, PairedSummary, SchemaMetrics};

/// Baseline minus candidate: a negative bound requires improvement. These
/// explicit gates use point estimates; paired intervals accompany the result.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Gates {
    pub schema_version: u32,
    pub max_field_f1_decrease: f64,
    pub max_field_recall_decrease: f64,
    pub max_document_accuracy_decrease: f64,
}
impl Gates {
    pub fn validate(self, limits: Limits) -> Result<(), AssayError> {
        limits.validate()?;
        if self.schema_version != ASSAY_SCHEMA_VERSION
            || [self.max_field_f1_decrease, self.max_field_recall_decrease, self.max_document_accuracy_decrease]
                .iter().any(|value| !value.is_finite() || !(-1.0..=1.0).contains(value)) {
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeMetadata {
    pub dataset_id: String,
    pub dataset_version: String,
    pub named_population: String,
    /// Declared catalogue identifiers, never schema bodies or extracted keys.
    pub schema_ids: Vec<String>,
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
    pub field_semantics: String,
    pub grounding_semantics: String,
    pub bootstrap_semantics: String,
    pub language_scope: String,
    pub bindings: Bindings,
    pub options: EvalOptions,
    pub extract_limits: ExtractLimits,
    pub measurement: Measurement,
    pub validated_rows: usize,
    pub executed_rows: usize,
    pub group_width: usize,
    pub model_work: Int4Work,
    pub mask_node_visits: u64,
    /// All-split gold validation, shared by both qualification arms.
    pub gold_verification_steps: u64,
    pub gold_verification_fields: usize,
    pub gold_verification_matches: usize,
    /// Independent complete-output finalization after the two native passes.
    pub independent_verification_steps: u64,
    /// This standalone scorecard includes gold preparation once.
    pub total_source_verification_steps: u64,
}
#[derive(Clone, Debug, Serialize)]
pub struct Evaluation {
    #[serde(flatten)]
    pub body: EvaluationBody,
    pub seal: ContentCommitment,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum Decision {
    Passed { comparisons: Vec<GateResult>, paired: PairedSummary },
    Rejected { comparisons: Vec<GateResult>, paired: PairedSummary },
    InsufficientData { rows: usize, minimum_samples: usize },
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
    pub total_mask_node_visits: u64,
    /// Shared gold validation is counted once, then both native replay arms.
    pub total_source_verification_steps: u64,
}
#[derive(Clone, Debug, Serialize)]
pub struct QualificationReport {
    #[serde(flatten)]
    pub body: QualificationBody,
    pub seal: ContentCommitment,
}

pub(crate) fn validate_scope(namespace: &str, host_scope: Option<&str>) -> Result<(), AssayError> {
    if !identifier(namespace) { return Err(AssayError::Commitment); }
    if host_scope.is_some_and(|value| !identifier(value)) { return Err(AssayError::Scope); }
    Ok(())
}
#[cfg(test)]
pub(crate) fn evaluate(replay: &NativeReplay, key: &CommitmentKey, namespace: &str,
    options: EvalOptions, limits: Limits) -> Result<Evaluation, AssayError> {
    evaluate_with_checkpoint(replay, key, namespace, options, limits, &mut || Ok(()))
}
pub(crate) fn evaluate_with_checkpoint<E: From<AssayError>>(replay: &NativeReplay,
    key: &CommitmentKey, namespace: &str, options: EvalOptions, limits: Limits,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Evaluation, E> {
    validate_scope(namespace, None)?; replay.sizes.validate(limits)?; replay.extract_limits.validate()?;
    options.validate_rows(replay.rows.len(), limits)?;
    let schemas = schemas(replay)?; let rows = observations(replay, checkpoint)?;
    let measurement = field_metrics::measure_with_checkpoint(&schemas, &rows, options, limits, checkpoint)?;
    evaluation(replay, key, namespace, options, limits, measurement, checkpoint)
}
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn compare(candidate: &NativeReplay, baseline: &NativeReplay, key: &CommitmentKey,
    namespace: &str, host_scope: &str, gates: Gates, options: EvalOptions, limits: Limits)
    -> Result<QualificationReport, AssayError> {
    compare_with_checkpoint(candidate, baseline, key, namespace, host_scope, gates, options, limits, &mut || Ok(()))
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn compare_with_checkpoint<E: From<AssayError>>(candidate: &NativeReplay, baseline: &NativeReplay,
    key: &CommitmentKey, namespace: &str, host_scope: &str, gates: Gates, options: EvalOptions,
    limits: Limits, checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<QualificationReport, E> {
    validate_scope(namespace, Some(host_scope))?; gates.validate(limits)?;
    candidate.sizes.validate(limits)?; baseline.sizes.validate(limits)?;
    candidate.extract_limits.validate()?; baseline.extract_limits.validate()?;
    validate_pair(candidate, baseline, checkpoint)?;
    options.validate_paired_rows(candidate.rows.len(), limits)?;
    let total_source_verification_steps = candidate.gold_verification_steps
        .checked_add(candidate.total_source_verification_steps)
        .and_then(|steps| steps.checked_add(baseline.total_source_verification_steps)).ok_or(AssayError::InputLimit)?;
    if total_source_verification_steps > candidate.extract_limits.max_verification_steps { return Err(AssayError::InputLimit.into()); }
    let schemas = schemas(candidate)?;
    let current_rows = observations(candidate, checkpoint)?; let prior_rows = observations(baseline, checkpoint)?;
    let (current, prior, paired) = field_metrics::paired_with_checkpoint(&schemas, &current_rows, &prior_rows,
        options, limits, checkpoint)?;
    // The single paired pass already produced both marginal scorecards.
    let current = evaluation(candidate, key, namespace, options, limits, current, checkpoint)?;
    let prior = evaluation(baseline, key, namespace, options, limits, prior, checkpoint)?;
    let decision = match (&current.body.measurement, &prior.body.measurement, paired) {
        (Measurement::Measured { .. }, Measurement::Measured { .. }, Some(paired)) => {
            let comparisons = vec![
                upper("field_f1_decrease", paired.field_f1_change.map(|value| -value), gates.max_field_f1_decrease),
                upper("field_recall_decrease", paired.field_recall_change.map(|value| -value), gates.max_field_recall_decrease),
                upper("document_accuracy_decrease", Some(-paired.document_accuracy_change), gates.max_document_accuracy_decrease),
            ];
            if comparisons.iter().all(|value| value.passed) { Decision::Passed { comparisons, paired } }
            else { Decision::Rejected { comparisons, paired } }
        }
        (Measurement::InsufficientData { .. }, Measurement::InsufficientData { .. }, None) =>
            Decision::InsufficientData { rows: candidate.rows.len(), minimum_samples: options.minimum_samples },
        _ => return Err(AssayError::Scope.into()),
    };
    #[derive(Serialize)]
    struct Policy<'a> { schema_version: u32, task: &'static str, gate_basis: &'static str,
        declared_host_scope: &'a str, gates: Gates, options: EvalOptions, extract_limits: ExtractLimits }
    let policy = Policy { schema_version: ASSAY_SCHEMA_VERSION, task: NATIVE_TASK, gate_basis: "point_estimates",
        declared_host_scope: host_scope, gates, options, extract_limits: candidate.extract_limits };
    let policy_commitment = commit(key, namespace, "native_extract_qualification_policy_v1", &policy, limits)?;
    let body = QualificationBody { schema_version: ASSAY_SCHEMA_VERSION, task: NATIVE_TASK.to_owned(),
        evidence: NATIVE_EVIDENCE.to_owned(), activation_authority: false, receipt_grade: ReceiptGrade::AuditOnly,
        declared_host_scope: host_scope.to_owned(), host_scope_evidence: "caller_declared".to_owned(),
        gate_basis: "point_estimates".to_owned(), gates, policy_commitment, candidate: current, baseline: prior, decision,
        total_model_work: candidate.model_work.checked_add(baseline.model_work).map_err(|_| AssayError::InputLimit)?,
        total_mask_node_visits: candidate.mask_node_visits.checked_add(baseline.mask_node_visits).ok_or(AssayError::InputLimit)?,
        total_source_verification_steps };
    let seal = commit(key, namespace, "native_extract_qualification_report_v1", &body, limits)?;
    let result = QualificationReport { body, seal };
    super::super::render(&result, limits)?; checkpoint()?; Ok(result)
}
#[allow(clippy::too_many_arguments)]
fn evaluation<E: From<AssayError>>(replay: &NativeReplay, key: &CommitmentKey, namespace: &str,
    options: EvalOptions, limits: Limits, measurement: Measurement,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Evaluation, E> {
    checkpoint()?;
    let total_source_verification_steps = replay.gold_verification_steps.checked_add(replay.total_source_verification_steps)
        .ok_or(AssayError::InputLimit)?;
    if total_source_verification_steps > replay.extract_limits.max_verification_steps { return Err(AssayError::InputLimit.into()); }
    let bindings = bindings(replay, key, namespace, checkpoint)?;
    let body = EvaluationBody { schema_version: ASSAY_SCHEMA_VERSION, task: NATIVE_TASK.to_owned(),
        evidence: NATIVE_EVIDENCE.to_owned(), activation_authority: false, receipt_grade: ReceiptGrade::AuditOnly,
        score_space: ScoreSpace::NotComputed,
        field_semantics: "terminal_scalars_and_empty_containers; object_keys_and_array_positions; exact_normalized_decimals; missing_distinct_from_null; mismatched_value_is_fp_and_fn; distinct_container_kinds_do_not_align".to_owned(),
        grounding_semantics: "schema_validity_and_declared_source_membership_are_structural_checks; exact_gold_agreement_is_not_independent_semantic_truth".to_owned(),
        bootstrap_semantics: "micro_field_scores; bootstrap_whole_documents; paired_draws_for_comparison; undefined_replicate_makes_interval_absent; breakdowns_are_point_estimates".to_owned(),
        language_scope: "caller_declared_row_metadata".to_owned(), bindings, options, extract_limits: replay.extract_limits,
        measurement, validated_rows: replay.sizes.rows, executed_rows: replay.rows.len(), group_width: replay.group_width,
        model_work: replay.model_work, mask_node_visits: replay.mask_node_visits,
        gold_verification_steps: replay.gold_verification_steps, gold_verification_fields: replay.gold_verification_fields,
        gold_verification_matches: replay.gold_verification_matches, independent_verification_steps: replay.verification_steps,
        total_source_verification_steps };
    let seal = commit(key, namespace, "native_extract_evaluation_v1", &body, limits)?;
    let result = Evaluation { body, seal }; super::super::render(&result, limits)?; checkpoint()?; Ok(result)
}
fn schemas(replay: &NativeReplay) -> Result<Vec<&str>, AssayError> {
    let mut schemas = reserved(replay.manifest.schemas.len())?;
    schemas.extend(replay.manifest.schemas.iter().map(|schema| schema.id.as_str())); Ok(schemas)
}
fn observations<'a, E: From<AssayError>>(replay: &'a NativeReplay,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Vec<Observation<'a>>, E> {
    let mut rows = reserved(replay.rows.len())?;
    for (index, row) in replay.rows.iter().enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        rows.push(Observation { schema_id: &row.schema_id, language: &row.language, counts: &row.counts });
    }
    checkpoint()?; Ok(rows)
}
fn validate_pair<E: From<AssayError>>(a: &NativeReplay, b: &NativeReplay,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), E> {
    if a.manifest_digest != b.manifest_digest || a.input_digest != b.input_digest
        || a.compatible_executions != b.compatible_executions || a.extract_limits != b.extract_limits
        || a.rows.len() != b.rows.len() || a.gold_verification_steps != b.gold_verification_steps
        || a.gold_verification_fields != b.gold_verification_fields || a.gold_verification_matches != b.gold_verification_matches {
        return Err(AssayError::Scope.into());
    }
    for (index, (a, b)) in a.rows.iter().zip(&b.rows).enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        if a.id != b.id || a.schema_id != b.schema_id || a.language != b.language || a.unit != b.unit {
            return Err(AssayError::Scope.into());
        }
    }
    checkpoint()?; Ok(())
}
fn bindings<E: From<AssayError>>(replay: &NativeReplay, key: &CommitmentKey, namespace: &str,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Bindings, E> {
    checkpoint()?;
    let suite = crate::canonjson::canonical_bytes(&replay.manifest).map_err(|_| AssayError::Commitment)?;
    let suite = key.commit(CommitmentDomain::Configuration, namespace, "native_extract_suite_v1", &suite)
        .map_err(|_| AssayError::Commitment)?;
    let input = key.commit(CommitmentDomain::Input, namespace, "native_extract_source_ndjson_sha256_hex_v1",
        replay.input_digest.to_hex().as_bytes()).map_err(|_| AssayError::Commitment)?;
    let mut split = |name: &str, ids: &[String]| -> Result<ContentCommitment, E> {
        checkpoint()?;
        let bytes = crate::canonjson::canonical_bytes(ids).map_err(|_| AssayError::Commitment)?;
        Ok(key.commit(CommitmentDomain::Configuration, namespace, name, &bytes).map_err(|_| AssayError::Commitment)?)
    };
    let splits = SplitBindings { development: split("native_extract_development_ids_v1", &replay.manifest.splits.development)?,
        calibration: split("native_extract_calibration_ids_v1", &replay.manifest.splits.calibration)?,
        locked_test: split("native_extract_locked_test_ids_v1", &replay.manifest.splits.locked_test)? };
    let mut schema_ids = reserved(replay.manifest.schemas.len())?;
    schema_ids.extend(replay.manifest.schemas.iter().map(|schema| schema.id.clone()));
    let identity = &replay.first_identity;
    let metadata = NativeMetadata { dataset_id: replay.manifest.dataset_id.clone(), dataset_version: replay.manifest.dataset_version.clone(),
        named_population: replay.manifest.named_population.clone(), schema_ids, task_spec: identity.task_spec.clone(),
        model_recipe_id: identity.quant_recipe.clone(), prompt_template_sha256: identity.template_digest,
        numerics_profile: identity.numerics_profile.label(), thinking_mode: "disabled".to_owned() };
    let execution_collection = key.commit(CommitmentDomain::ExecutionIdentity, namespace,
        "native_extract_execution_collection_sha256_binary_v1", &replay.executions).map_err(|_| AssayError::Commitment)?;
    let observation_collection = key.commit(CommitmentDomain::Output, namespace,
        "native_extract_observation_collection_sha256_binary_v1", &replay.observations).map_err(|_| AssayError::Commitment)?;
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

#[cfg(test)] mod tests;
