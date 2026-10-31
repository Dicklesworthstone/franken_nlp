//! Content-free audit reports over sealed native NER observations.
use serde::{Deserialize, Serialize};
use crate::{native_engine::strict_int4::Int4Work,
    receipt::{CommitmentDomain, CommitmentKey, ContentCommitment, ReceiptGrade},
    tasks::ir::ScoreSpace};
use super::*;
use super::super::{span_metrics::{self, Observation}, qualify::GateResult};
pub use super::super::span_metrics::{AnchoringMetrics, EvalOptions, LanguageMetrics, Measurement,
    PairedSummary, SpanIntervals, SpanMetrics, SpanScores, TypeMetrics};

/// Every decrease is baseline minus candidate. A negative bound requires
/// improvement. Undefined scores cannot pass; these are point-estimate gates.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Gates {
    pub schema_version: u32,
    pub max_exact_f1_decrease: f64,
    pub max_type_relaxed_f1_decrease: f64,
    pub max_exact_recall_decrease: f64,
    pub max_document_accuracy_decrease: f64,
}
impl Gates {
    pub fn validate(self, limits: Limits) -> Result<(), AssayError> {
        limits.validate()?;
        if self.schema_version != ASSAY_SCHEMA_VERSION
            || [self.max_exact_f1_decrease, self.max_type_relaxed_f1_decrease,
                self.max_exact_recall_decrease, self.max_document_accuracy_decrease].iter()
                .any(|value| !value.is_finite() || !(-1.0..=1.0).contains(value)) {
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
    pub entity_types: Vec<String>,
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
    pub span_semantics: String,
    pub bootstrap_semantics: String,
    pub language_scope: String,
    pub bindings: Bindings,
    pub options: EvalOptions,
    pub ner_limits: NerLimits,
    pub measurement: Measurement,
    pub validated_rows: usize,
    pub executed_rows: usize,
    pub group_width: usize,
    pub model_work: Int4Work,
    pub mask_node_visits: u64,
    pub independent_verification_steps: u64,
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
    validate_scope(namespace, None)?; replay.sizes.validate(limits)?; replay.ner_limits.validate()?;
    options.validate_rows(replay.rows.len(), limits)?;
    let types = types(replay)?;
    let rows = observations(replay, checkpoint)?;
    let measurement = span_metrics::measure_with_checkpoint(&types, &rows, options, limits, checkpoint)?;
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
    candidate.ner_limits.validate()?; baseline.ner_limits.validate()?;
    validate_pair(candidate, baseline, checkpoint)?;
    options.validate_paired_rows(candidate.rows.len(), limits)?;
    let types = types(candidate)?;
    let current_rows = observations(candidate, checkpoint)?;
    let prior_rows = observations(baseline, checkpoint)?;
    let (current, prior, paired) = span_metrics::paired_with_checkpoint(&types, &current_rows, &prior_rows,
        options, limits, checkpoint)?;
    // The paired pass has already produced both marginal scorecards. Calling
    // evaluate again would renew bootstrap work and discard the pairing.
    let current = evaluation(candidate, key, namespace, options, limits, current, checkpoint)?;
    let prior = evaluation(baseline, key, namespace, options, limits, prior, checkpoint)?;
    let decision = match (&current.body.measurement, &prior.body.measurement, paired) {
        (Measurement::Measured { .. }, Measurement::Measured { .. }, Some(paired)) => {
            let comparisons = vec![
                upper("exact_f1_decrease", paired.exact_f1_change.map(|value| -value), gates.max_exact_f1_decrease),
                upper("type_relaxed_f1_decrease", paired.type_relaxed_f1_change.map(|value| -value), gates.max_type_relaxed_f1_decrease),
                upper("exact_recall_decrease", paired.exact_recall_change.map(|value| -value), gates.max_exact_recall_decrease),
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
        declared_host_scope: &'a str, gates: Gates, options: EvalOptions, ner_limits: NerLimits }
    let policy = Policy { schema_version: ASSAY_SCHEMA_VERSION, task: NATIVE_TASK, gate_basis: "point_estimates",
        declared_host_scope: host_scope, gates, options, ner_limits: candidate.ner_limits };
    let policy_commitment = commit(key, namespace, "native_ner_qualification_policy_v1", &policy, limits)?;
    let body = QualificationBody { schema_version: ASSAY_SCHEMA_VERSION, task: NATIVE_TASK.to_owned(),
        evidence: NATIVE_EVIDENCE.to_owned(), activation_authority: false, receipt_grade: ReceiptGrade::AuditOnly,
        declared_host_scope: host_scope.to_owned(), host_scope_evidence: "caller_declared".to_owned(),
        gate_basis: "point_estimates".to_owned(), gates, policy_commitment, candidate: current, baseline: prior, decision,
        total_model_work: candidate.model_work.checked_add(baseline.model_work).map_err(|_| AssayError::InputLimit)?,
        total_mask_node_visits: candidate.mask_node_visits.checked_add(baseline.mask_node_visits).ok_or(AssayError::InputLimit)?,
        total_source_verification_steps: candidate.total_source_verification_steps
            .checked_add(baseline.total_source_verification_steps).ok_or(AssayError::InputLimit)? };
    checkpoint()?;
    let seal = commit(key, namespace, "native_ner_qualification_v1", &body, limits)?;
    let result = QualificationReport { body, seal };
    super::super::render(&result, limits)?; checkpoint()?; Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn evaluation<E: From<AssayError>>(replay: &NativeReplay, key: &CommitmentKey, namespace: &str,
    options: EvalOptions, limits: Limits, measurement: Measurement,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Evaluation, E> {
    checkpoint()?;
    let bindings = bindings(replay, key, namespace, checkpoint)?;
    let body = EvaluationBody { schema_version: ASSAY_SCHEMA_VERSION, task: NATIVE_TASK.to_owned(),
        evidence: NATIVE_EVIDENCE.to_owned(), activation_authority: false, receipt_grade: ReceiptGrade::AuditOnly,
        score_space: ScoreSpace::NotComputed,
        span_semantics: "half_open_utf8_bytes; exact_interval_and_type; type_relaxed_exact_interval_only; one_to_one_matches; ambiguous_proposal_is_one_unmatched_prediction; duplicates_do_not_multiply_recall".to_owned(),
        bootstrap_semantics: "micro_scores; bootstrap_whole_documents; paired_draws_for_comparison; undefined_replicate_makes_interval_absent; breakdowns_are_point_estimates".to_owned(),
        language_scope: "caller_declared_row_metadata".to_owned(),
        bindings, options, ner_limits: replay.ner_limits, measurement, validated_rows: replay.sizes.rows,
        executed_rows: replay.rows.len(), group_width: replay.group_width, model_work: replay.model_work,
        mask_node_visits: replay.mask_node_visits, independent_verification_steps: replay.verification_steps,
        total_source_verification_steps: replay.total_source_verification_steps };
    let seal = commit(key, namespace, "native_ner_evaluation_v1", &body, limits)?;
    let result = Evaluation { body, seal };
    super::super::render(&result, limits)?; checkpoint()?; Ok(result)
}

fn types(replay: &NativeReplay) -> Result<Vec<&'static str>, AssayError> {
    let mut types = reserved(replay.manifest.options.types.len())?;
    types.extend(replay.manifest.options.types.iter().map(|kind| kind.label())); Ok(types)
}
fn observations<'a, E: From<AssayError>>(replay: &'a NativeReplay,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Vec<Observation<'a>>, E> {
    let mut rows = reserved(replay.rows.len())?;
    for (index, row) in replay.rows.iter().enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        rows.push(Observation { language: &row.language, counts: &row.counts });
    }
    checkpoint()?; Ok(rows)
}
fn validate_pair<E: From<AssayError>>(a: &NativeReplay, b: &NativeReplay,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), E> {
    if a.manifest_digest != b.manifest_digest || a.input_digest != b.input_digest
        || a.compatible_executions != b.compatible_executions || a.ner_limits != b.ner_limits
        || a.rows.len() != b.rows.len() { return Err(AssayError::Scope.into()); }
    for (index, (a, b)) in a.rows.iter().zip(&b.rows).enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        if a.id != b.id || a.language != b.language || a.unit != b.unit { return Err(AssayError::Scope.into()); }
    }
    checkpoint()?; Ok(())
}
fn bindings<E: From<AssayError>>(replay: &NativeReplay, key: &CommitmentKey, namespace: &str,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Bindings, E> {
    checkpoint()?;
    let suite = crate::canonjson::canonical_bytes(&replay.manifest).map_err(|_| AssayError::Commitment)?;
    let suite = key.commit(CommitmentDomain::Configuration, namespace, "native_ner_suite_v1", &suite)
        .map_err(|_| AssayError::Commitment)?;
    let input = key.commit(CommitmentDomain::Input, namespace, "native_ner_source_ndjson_sha256_hex_v1",
        replay.input_digest.to_hex().as_bytes()).map_err(|_| AssayError::Commitment)?;
    let mut split = |name: &str, ids: &[String]| -> Result<ContentCommitment, E> {
        checkpoint()?;
        let bytes = crate::canonjson::canonical_bytes(ids).map_err(|_| AssayError::Commitment)?;
        Ok(key.commit(CommitmentDomain::Configuration, namespace, name, &bytes).map_err(|_| AssayError::Commitment)?)
    };
    let splits = SplitBindings { development: split("native_ner_development_ids_v1", &replay.manifest.splits.development)?,
        calibration: split("native_ner_calibration_ids_v1", &replay.manifest.splits.calibration)?,
        locked_test: split("native_ner_locked_test_ids_v1", &replay.manifest.splits.locked_test)? };
    let mut entity_types = reserved(replay.manifest.options.types.len())?;
    entity_types.extend(replay.manifest.options.types.iter().map(|kind| kind.label().to_owned()));
    let identity = &replay.first_identity;
    let metadata = NativeMetadata { dataset_id: replay.manifest.dataset_id.clone(), dataset_version: replay.manifest.dataset_version.clone(),
        named_population: replay.manifest.named_population.clone(), entity_types,
        task_spec: identity.task_spec.clone(), model_recipe_id: identity.quant_recipe.clone(),
        prompt_template_sha256: identity.template_digest, numerics_profile: identity.numerics_profile.label(),
        thinking_mode: "disabled".to_owned() };
    let execution_collection = key.commit(CommitmentDomain::ExecutionIdentity, namespace,
        "native_ner_execution_collection_sha256_binary_v1", &replay.executions).map_err(|_| AssayError::Commitment)?;
    let observation_collection = key.commit(CommitmentDomain::Output, namespace,
        "native_ner_observation_collection_sha256_binary_v1", &replay.observations).map_err(|_| AssayError::Commitment)?;
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
