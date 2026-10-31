//! A closed collector called immediately after genuine native task execution.
use serde::Serialize;
use sha2::{Digest, Sha256};
use crate::{calibration::{CalibrationError, CalibrationPartition, LabeledScore, SplitMembership},
    execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{constrained_int4, lmhead::NANBEIGE_VOCAB_SIZE,
        strict_int4::{Int4ModelIdentity, Int4Work, STRICT_INT4_PROFILE}},
    tasks::{classify::{ClassificationCalibration, ClassificationDecision, ClassificationPolicy, ClassificationResult,
        ClassificationTaskResult, MultiLabelDecision,
        grouped_int4::{Int4ClassificationRun, PreparedInt4Classification, INT4_CLASSIFICATION_EXECUTION}},
        ir::{ScoreSpace, TaskBudget}}};
use super::*;

/// No public constructor, deserialize, callback scorer or mutable rows can
/// convert supplied probabilities into native evidence.
pub(crate) struct NativeReplay {
    pub(super) manifest: SuiteManifest,
    pub(super) manifest_digest: Sha256Digest,
    pub(super) input_digest: Sha256Digest,
    pub(super) sizes: dataset::InputSizes,
    pub(super) rows: Vec<ScoredRow>,
    pub(super) first_identity: ExecutionIdentity,
    pub(super) executions: [u8; 32],
    pub(super) compatible_executions: [u8; 32],
    pub(super) observations: [u8; 32],
    pub(super) model_work: Int4Work,
    pub(super) rewound_positions: u64,
    pub(super) group_width: usize,
}

pub(super) struct ScoredRow {
    pub id: String,
    pub split: Split,
    pub positive: Vec<bool>,
    pub unit: Sha256Digest,
    pub probability: Vec<f64>,
    pub predicted: Vec<Option<bool>>,
    /// Exclusive raw-sequence argmax, including when the task abstains. Raw
    /// ordering must survive candidate-weight underflow and lexical ties.
    pub top_label: Option<usize>,
}

pub(crate) struct Collector<'a> { model: &'a Int4ModelIdentity, builder: ReplayBuilder<'a> }
impl<'a> Collector<'a> {
    pub(crate) fn new(dataset: &'a Dataset, model: &'a Int4ModelIdentity, limits: Limits)
        -> Result<Self, AssayError> {
        Ok(Self { model, builder: ReplayBuilder::new(dataset, limits)? })
    }
    pub(crate) fn accept(&mut self, index: usize, prepared: &PreparedInt4Classification,
        run: Int4ClassificationRun) -> Result<(), AssayError> {
        self.builder.enter(index)?;
        let identity = prepared.execution_identity();
        prepared.verify_identity(identity).map_err(|_| AssayError::Identity)?;
        constrained_int4::check_model(identity, self.model).map_err(|_| AssayError::Identity)?;
        if prepared.head_count() != self.builder.dataset.head_count()
            || prepared.group_width() != self.model.group_width() {
            return Err(AssayError::Identity);
        }
        self.builder.append(index, identity, prepared.task_budget(), prepared.planned_work(),
            prepared.group_width(), run)?;
        self.builder.failed = false;
        Ok(())
    }
    pub(crate) fn finish(self) -> Result<NativeReplay, AssayError> { self.builder.finish() }
}

/// The separate private builder permits model-free regression fixtures. Only
/// Collector is visible to the native host, and its accept method checks the
/// actual sealed plan/model before this content collector receives a row.
pub(super) struct ReplayBuilder<'a> {
    dataset: &'a Dataset,
    rows: Vec<ScoredRow>,
    first_identity: Option<ExecutionIdentity>,
    executions: Sha256,
    compatible_executions: Sha256,
    observations: Sha256,
    model_work: Int4Work,
    rewound_positions: u64,
    group_width: Option<usize>,
    eos: u32,
    failed: bool,
}
impl<'a> ReplayBuilder<'a> {
    pub(super) fn new(dataset: &'a Dataset, limits: Limits) -> Result<Self, AssayError> {
        dataset.validate_limits(limits)?;
        let controls = crate::tokenizer::pinned_controls::pinned().map_err(|_| AssayError::Identity)?;
        let mut ends = controls.template_controls().entries().iter()
            .filter(|entry| entry.special && entry.surface == crate::template::IM_END);
        let eos = ends.next().ok_or(AssayError::Identity)?.id;
        if ends.next().is_some() { return Err(AssayError::Identity); }
        let mut executions = Sha256::new();
        executions.update(b"fnlp.native_assay.execution_collection.v1\0");
        executions.update((dataset.len() as u64).to_be_bytes());
        let mut compatible_executions = Sha256::new();
        compatible_executions.update(b"fnlp.native_assay.pair_compatibility.v1\0");
        compatible_executions.update((dataset.len() as u64).to_be_bytes());
        let mut observations = Sha256::new();
        observations.update(b"fnlp.native_assay.observation_collection.v1\0");
        observations.update((dataset.len() as u64).to_be_bytes());
        Ok(Self { dataset, rows: reserved(dataset.len())?, first_identity: None,
            executions, compatible_executions, observations, model_work: Int4Work::default(), rewound_positions: 0,
            group_width: None, eos, failed: false })
    }
    fn enter(&mut self, index: usize) -> Result<(), AssayError> {
        if self.failed { return Err(AssayError::Observation); }
        self.failed = true;
        if index != self.rows.len() || index >= self.dataset.len() { return Err(AssayError::Scope); }
        Ok(())
    }
    fn append(&mut self, index: usize, identity: &ExecutionIdentity, budget: TaskBudget,
        planned: Int4Work, width: usize, run: Int4ClassificationRun) -> Result<(), AssayError> {
        let row = self.dataset.rows().get(index).ok_or(AssayError::Scope)?;
        identity.validate().map_err(|_| AssayError::Identity)?;
        budget.validate().map_err(|_| AssayError::Limits)?;
        if identity.task_spec != "classify-v1" || identity.numerics_profile != (NumericsProfile::StrictQuantized { version: 2 })
            || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None
            || width == 0 || self.group_width.is_some_and(|prior| prior != width)
            || run.schema_version != ASSAY_SCHEMA_VERSION || run.execution != INT4_CLASSIFICATION_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != width
            || run.head_count != self.dataset.head_count()
            || run.model_work != planned || planned.forward_positions == 0 || planned.projected_logits == 0
            || planned.attention_pairs == 0 || planned.projections.dot_products == 0 || planned.projections.multiply_accumulates == 0 {
            return Err(AssayError::Identity);
        }
        let (probability, predicted, top_label) = score(self.dataset, &run.result, self.eos, planned)?;
        let positive = self.dataset.gold(index)?;
        let unit = Sha256Digest::of_bytes(row.text.as_bytes());
        #[derive(Serialize)]
        struct RowBinding<'a> { index: usize, id: &'a str, split: Split, positive: bool,
            source_unit: Sha256Digest, task_budget: TaskBudget }
        #[derive(Serialize)]
        struct VectorRowBinding<'a> { index: usize, id: &'a str, split: Split, positive: &'a [bool],
            source_unit: Sha256Digest, task_budget: TaskBudget }
        let record = if self.dataset.is_binary() {
            crate::canonjson::canonical_bytes(&RowBinding { index, id: &row.id, split: row.split,
                positive: positive[0], source_unit: unit, task_budget: budget })
        } else {
            crate::canonjson::canonical_bytes(&VectorRowBinding { index, id: &row.id, split: row.split,
                positive: &positive, source_unit: unit, task_budget: budget })
        }.map_err(|_| AssayError::Commitment)?;
        #[derive(Serialize)]
        struct ObservationBinding<'a> { positive_candidate_weight: f64,
            decision: ClassificationDecision, selected_id: Option<&'a str> }
        #[derive(Serialize)]
        struct VectorObservationBinding<'a> { candidate_weights: &'a [f64],
            decisions: &'a [Option<bool>], top_label: Option<usize> }
        let observed = if self.dataset.is_binary() {
            let ClassificationTaskResult::Exclusive(result) = &run.result else { return Err(AssayError::Observation); };
            crate::canonjson::canonical_bytes(&ObservationBinding { positive_candidate_weight: probability[0],
                decision: result.decision, selected_id: result.selected_id.as_deref() })
        } else {
            crate::canonjson::canonical_bytes(&VectorObservationBinding { candidate_weights: &probability,
                decisions: &predicted, top_label })
        }.map_err(|_| AssayError::Commitment)?;
        let full_identity = identity.canonical_json_bytes().map_err(|_| AssayError::Identity)?;
        let compatible = compatible_identity(identity).canonical_json_bytes().map_err(|_| AssayError::Identity)?;
        if full_identity.len() > 16_384 || compatible.len() > 16_384 { return Err(AssayError::Identity); }
        frame(&mut self.executions, &record); frame(&mut self.executions, &full_identity);
        frame(&mut self.compatible_executions, &record); frame(&mut self.compatible_executions, &compatible);
        // Actual outputs have a distinct stream from execution semantics. Both
        // streams bind the same ordered association record; neither raw stream
        // nor its unkeyed digest is returned to the caller.
        frame(&mut self.observations, &record); frame(&mut self.observations, &observed);
        self.model_work = self.model_work.checked_add(run.model_work).map_err(|_| AssayError::InputLimit)?;
        self.rewound_positions = self.rewound_positions.checked_add(run.rewound_positions).ok_or(AssayError::InputLimit)?;
        self.rows.push(ScoredRow { id: row.id.clone(), split: row.split, positive, unit,
            probability, predicted, top_label });
        if self.first_identity.is_none() { self.first_identity = Some(identity.clone()); }
        self.group_width = Some(width);
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn accept_fixture(&mut self, index: usize, identity: &ExecutionIdentity, budget: TaskBudget,
        planned: Int4Work, width: usize, run: Int4ClassificationRun) -> Result<(), AssayError> {
        self.enter(index)?; self.append(index, identity, budget, planned, width, run)?;
        self.failed = false; Ok(())
    }
    pub(super) fn finish(self) -> Result<NativeReplay, AssayError> {
        if self.failed || self.rows.len() != self.dataset.len() { return Err(AssayError::Observation); }
        Ok(NativeReplay { manifest: self.dataset.manifest.clone(), manifest_digest: self.dataset.manifest_digest,
            input_digest: self.dataset.input_digest, sizes: self.dataset.sizes, rows: self.rows,
            first_identity: self.first_identity.ok_or(AssayError::Identity)?, executions: self.executions.finalize().into(),
            compatible_executions: self.compatible_executions.finalize().into(), observations: self.observations.finalize().into(),
            model_work: self.model_work,
            rewound_positions: self.rewound_positions, group_width: self.group_width.ok_or(AssayError::Identity)? })
    }
}

fn frame(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_be_bytes()); hash.update(bytes);
}

/// Comparison only: preserve the complete task/label/policy/prompt/TaskIR and
/// budget contract while substituting exactly the allowed model/implementation
/// fields. This digest grants no cache, receipt or native-execution authority;
/// the unmodified complete identities are committed independently above.
fn compatible_identity(identity: &ExecutionIdentity) -> ExecutionIdentity {
    let mut value = identity.clone();
    let absent = Sha256Digest::of_bytes(b"native-assay-comparison-subject-v1");
    value.logical_model_digest = absent; value.packing_set_digest = absent;
    value.artifact_format = "comparison-subject".to_owned(); value.quant_recipe = "comparison-subject".to_owned();
    value.numerics_profile = NumericsProfile::StrictQuantized { version: 2 };
    value.backend_semantic_version = "comparison-subject".to_owned();
    value.host_class = None; value.compiler_identity = None;
    value
}

fn score(dataset: &Dataset, result: &ClassificationTaskResult, eos: u32, work: Int4Work)
    -> Result<(Vec<f64>, Vec<Option<bool>>, Option<usize>), AssayError> {
    let manifest = dataset.manifest();
    let mut probability = reserved(dataset.label_count())?;
    let mut predicted = reserved(dataset.label_count())?;
    match (dataset.task_kind(), result) {
        (TaskKind::Binary | TaskKind::Multiclass, ClassificationTaskResult::Exclusive(result)) => {
            let mut expected = reserved(dataset.label_count())?;
            expected.extend(manifest.labels().map(|label| label.id.as_str()));
            expected.sort_unstable();
            let projected = validate_classification(&expected, manifest.policy, result, eos)?;
            if projected != work.projected_logits { return Err(AssayError::Observation); }
            let mut top_label = None;
            for (index, label) in manifest.labels().enumerate() {
                let candidate = result.scores.candidates.binary_search_by(|candidate| candidate.id.cmp(&label.id))
                    .map_err(|_| AssayError::Observation)?;
                probability.push(result.scores.candidates[candidate].candidate_weight);
                predicted.push(result.selected_id.as_ref().map(|id| id == &label.id));
                if result.ranking.first() == Some(&label.id) { top_label = Some(index); }
            }
            Ok((probability, predicted, Some(top_label.ok_or(AssayError::Observation)?)))
        }
        (TaskKind::Multilabel, ClassificationTaskResult::MultiLabel(result)) => {
            if result.schema_version != ASSAY_SCHEMA_VERSION
                || result.decision_scope != "independent_binary_heads_not_a_distribution_over_labels"
                || result.calibration != ClassificationCalibration::Uncalibrated
                || result.labels.len() != dataset.label_count() { return Err(AssayError::Observation); }
            let mut projected = 0_u64;
            for (label, head) in manifest.labels().zip(&result.labels) {
                if label.id != head.label_id { return Err(AssayError::Observation); }
                projected = projected.checked_add(validate_classification(&["no", "yes"], manifest.policy,
                    &head.classification, eos)?).ok_or(AssayError::InputLimit)?;
                let positive = head.classification.scores.candidates[1].candidate_weight;
                let decision = match head.classification.selected_id.as_deref() {
                    Some("yes") => Some(true), Some("no") => Some(false), None => None,
                    _ => return Err(AssayError::Observation),
                };
                let expected = match decision { Some(true) => MultiLabelDecision::Included,
                    Some(false) => MultiLabelDecision::Excluded, None => MultiLabelDecision::Abstained };
                if head.positive_candidate_weight != positive || head.decision != expected {
                    return Err(AssayError::Observation);
                }
                probability.push(positive); predicted.push(decision);
            }
            if projected != work.projected_logits { return Err(AssayError::Observation); }
            for (decision, ids) in [(MultiLabelDecision::Included, &result.selected_ids),
                (MultiLabelDecision::Excluded, &result.excluded_ids), (MultiLabelDecision::Abstained, &result.abstained_ids)] {
                if ids.iter().map(String::as_str).ne(result.labels.iter().filter(|label| label.decision == decision)
                    .map(|label| label.label_id.as_str())) { return Err(AssayError::Observation); }
            }
            Ok((probability, predicted, None))
        }
        _ => Err(AssayError::Observation),
    }
}

/// One validator for an exclusive label language and each independent yes/no
/// language. Its caller separately checks the sum across all actual heads.
fn validate_classification(expected: &[&str], policy: ClassificationPolicy,
    result: &ClassificationResult, eos: u32) -> Result<u64, AssayError> {
    let scores = &result.scores;
    if result.schema_version != ASSAY_SCHEMA_VERSION || result.policy != policy
        || result.calibration != ClassificationCalibration::Uncalibrated
        || scores.score_space != ScoreSpace::FullVocabSequenceLogprob || !scores.full_vocab_denominators_computed
        || scores.normalization_scope != "full_vocabulary_per_position; candidate_weights_over_completed_candidates"
        || scores.eos_rule != "append_and_score_exactly_one_eos" || scores.length_rule != "sum_logprobs_including_eos"
        || scores.eos_token_id != eos || expected.len() < 2 || scores.candidates.len() != expected.len()
        || result.ranking.len() != expected.len() || scores.work.projected_logits == 0
        || (scores.work.prefix_evaluations as u64).checked_mul(NANBEIGE_VOCAB_SIZE as u64)
            != Some(scores.work.projected_logits) {
        return Err(AssayError::Observation);
    }
    let mut total = 0.0;
    for (candidate, id) in scores.candidates.iter().zip(expected) {
        if candidate.id != *id || candidate.scored_tokens == 0 || !candidate.sequence_score.is_finite()
            || !candidate.candidate_weight.is_finite() || !(0.0..=1.0).contains(&candidate.candidate_weight) {
            return Err(AssayError::Observation);
        }
        total += candidate.candidate_weight;
    }
    let tolerance = 32.0 * expected.len() as f64 * f64::EPSILON;
    if (total - 1.0).abs() > tolerance { return Err(AssayError::Observation); }
    let maximum = scores.candidates.iter().map(|candidate| candidate.sequence_score).fold(f64::NEG_INFINITY, f64::max);
    let denominator: f64 = scores.candidates.iter().map(|candidate| (candidate.sequence_score - maximum).exp()).sum();
    if scores.candidates.iter().any(|candidate|
        ((candidate.sequence_score - maximum).exp() / denominator - candidate.candidate_weight).abs()
            > tolerance) { return Err(AssayError::Observation); }
    let mut ranking = reserved(expected.len())?;
    ranking.extend(scores.candidates.iter());
    ranking.sort_by(|a, b| if a.sequence_score == b.sequence_score { a.id.cmp(&b.id) }
        else { b.sequence_score.total_cmp(&a.sequence_score) });
    if result.ranking.iter().map(String::as_str).ne(ranking.iter().map(|candidate| candidate.id.as_str())) {
        return Err(AssayError::Observation);
    }
    let best = ranking[0]; let margin = (best.candidate_weight - ranking[1].candidate_weight).max(0.0);
    let accepted = best.candidate_weight * 1_000_000.0 >= f64::from(policy.minimum_candidate_weight_ppm)
        && margin * 1_000_000.0 >= f64::from(policy.minimum_margin_ppm);
    if result.best_candidate_weight != best.candidate_weight || result.candidate_weight_margin != margin
        || result.decision != (if accepted { ClassificationDecision::Classified } else { ClassificationDecision::Abstained })
        || result.selected_id.as_deref() != accepted.then_some(best.id.as_str()) {
        return Err(AssayError::Observation);
    }
    Ok(scores.work.projected_logits)
}

impl NativeReplay {
    pub(super) fn partition<E>(&self, checkpoint: &mut impl FnMut() -> Result<(), E>)
        -> Result<CalibrationPartition, E>
    where E: From<AssayError> + From<CalibrationError> {
        checkpoint()?;
        if !self.manifest.is_binary() { return Err(AssayError::CalibrationBindingMismatch.into()); }
        let membership = SplitMembership::new(self.manifest.splits.development.clone(),
            self.manifest.splits.calibration.clone(), self.manifest.splits.locked_test.clone())?;
        let mut development = reserved(self.manifest.splits.development.len())?;
        let mut calibration = reserved(self.manifest.splits.calibration.len())?;
        let mut locked = reserved(self.manifest.splits.locked_test.len())?;
        for (index, row) in self.rows.iter().enumerate() {
            if index % 64 == 0 { checkpoint()?; }
            let score = LabeledScore::new(row.id.clone(), row.probability[0], row.positive[0])?;
            match row.split { Split::Development => development.push(score),
                Split::Calibration => calibration.push(score), Split::LockedTest => locked.push(score) }
        }
        checkpoint()?;
        let result = membership.partition(development, calibration, locked)?;
        checkpoint()?; Ok(result)
    }
}
