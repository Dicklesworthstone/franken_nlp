//! A descriptive, source-scalar-weighted mixture of complete classifications.
//! Per-chunk sequence scores stay in the original records. Mixture weights
//! are a separate probability space and never become calibrated confidence.
use super::{ClassificationCalibration, ClassificationDecision, ClassificationPolicy,
    ClassificationResult, ClassificationMode, ClassificationTaskResult, MultiLabelDecision};
use crate::{native_engine::decode::{DecodeCancellationKind, DecodeStepControl},
    tasks::ir::ScoreSpace, validation::grounded_fields::VerifiedSourceSpan};
use serde::{Deserialize, Serialize};

pub const CLASSIFICATION_AGGREGATION_PROFILE: &str = "classification-source-scalar-mixture-v1";
const SCALE: u32 = 1_000_000;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationAggregationPolicy {
    /// Original scalar coverage by non-abstaining chunk heads. Each label
    /// has its own coverage in multilabel mode; no head can authorize another.
    pub minimum_classified_coverage_ppm: u32,
}
impl Default for ClassificationAggregationPolicy {
    fn default() -> Self { Self { minimum_classified_coverage_ppm: SCALE } }
}
impl ClassificationAggregationPolicy {
    pub fn validate(self) -> Result<(), ClassificationAggregationError> {
        if self.minimum_classified_coverage_ppm > SCALE { return Err(ClassificationAggregationError::Policy); }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ClassificationMixtureWeight { pub candidate_id: String, pub weight: f64 }
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ClassificationMixture {
    pub decision: ClassificationDecision,
    pub selected_id: Option<String>,
    /// Sorted by aggregate weight, then exact candidate identifier.
    pub ranking: Vec<String>,
    /// Complete candidate set in canonical identifier order.
    pub candidate_weights: Vec<ClassificationMixtureWeight>,
    pub best_candidate_weight: f64,
    pub candidate_weight_margin: f64,
    pub classified_scalars: usize,
    pub classified_coverage: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MultiLabelAggregateLabel {
    pub label_id: String,
    pub decision: MultiLabelDecision,
    /// This label's independent yes/no mixture, never normalized over labels.
    pub positive_candidate_weight: f64,
    pub mixture: ClassificationMixture,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MultiLabelAggregate {
    pub selected_ids: Vec<String>, pub excluded_ids: Vec<String>, pub abstained_ids: Vec<String>,
    pub labels: Vec<MultiLabelAggregateLabel>,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "mode", content = "result", rename_all = "snake_case")]
pub enum ClassificationAggregateResult { Exclusive(ClassificationMixture), MultiLabel(MultiLabelAggregate) }
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ClassificationAggregate {
    pub schema_version: u32,
    pub profile: String,
    pub weight_basis: String,
    pub score_scope: String,
    pub calibration: ClassificationCalibration,
    pub policy: ClassificationAggregationPolicy,
    pub classification_policy: ClassificationPolicy,
    pub chunk_count: usize,
    pub source_bytes: usize,
    pub source_scalars: usize,
    pub result: ClassificationAggregateResult,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClassificationAggregationError { Policy, Coverage, Result, Allocation, Cancelled(DecodeCancellationKind) }
impl std::fmt::Display for ClassificationAggregationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Policy => "invalid classification aggregation policy",
            Self::Coverage => "classification chunks do not cover the complete source in order",
            Self::Result => "classification chunk result contract differs",
            Self::Allocation => "classification aggregation allocation refused",
            Self::Cancelled(_) => "classification aggregation cancelled",
        })
    }
}
impl std::error::Error for ClassificationAggregationError {}

pub(crate) struct ClassificationChunkView<'a> {
    pub chunk_id: usize, pub span: VerifiedSourceSpan, pub result: &'a ClassificationTaskResult,
}
struct Head {
    ids: Vec<String>, depths: Vec<usize>, weights: Vec<f64>, corrections: Vec<f64>, classified_scalars: usize, eos: u32,
}
impl Head {
    fn new(result: &ClassificationResult) -> Result<Self, ClassificationAggregationError> {
        if result.scores.candidates.is_empty() { return Err(ClassificationAggregationError::Result); }
        let mut ids = reserved(result.scores.candidates.len())?;
        let mut depths = reserved(result.scores.candidates.len())?;
        let mut weights = reserved(result.scores.candidates.len())?;
        let mut corrections = reserved(result.scores.candidates.len())?;
        for candidate in &result.scores.candidates {
            ids.push(candidate.id.clone()); depths.push(candidate.scored_tokens); weights.push(0.0); corrections.push(0.0);
        }
        if ids.windows(2).any(|pair| pair[0] >= pair[1]) { return Err(ClassificationAggregationError::Result); }
        Ok(Self { ids, depths, weights, corrections, classified_scalars: 0, eos: result.scores.eos_token_id })
    }
    fn add(&mut self, result: &ClassificationResult, scalars: usize,
        policy: ClassificationPolicy) -> Result<(), ClassificationAggregationError> {
        let scores = &result.scores;
        if result.schema_version != 1 || result.calibration != ClassificationCalibration::Uncalibrated
            || result.policy != policy || scores.score_space != ScoreSpace::FullVocabSequenceLogprob
            || !scores.full_vocab_denominators_computed || scores.eos_token_id != self.eos
            || scores.eos_rule != "append_and_score_exactly_one_eos"
            || scores.length_rule != "sum_logprobs_including_eos"
            || scores.normalization_scope != "full_vocabulary_per_position; candidate_weights_over_completed_candidates"
            || scores.candidates.len() != self.ids.len() || result.ranking.len() != self.ids.len() {
            return Err(ClassificationAggregationError::Result);
        }
        let mut sum = 0.0;
        for (index, candidate) in scores.candidates.iter().enumerate() {
            if candidate.id != self.ids[index] || candidate.scored_tokens != self.depths[index]
                || candidate.scored_tokens < 2 || !candidate.sequence_score.is_finite()
                || !candidate.candidate_weight.is_finite() || !(0.0..=1.0).contains(&candidate.candidate_weight) {
                return Err(ClassificationAggregationError::Result);
            }
            sum += candidate.candidate_weight;
        }
        if !near(sum, 1.0, self.ids.len()) { return Err(ClassificationAggregationError::Result); }
        let mut order = reserved(self.ids.len())?; order.extend(0..self.ids.len());
        order.sort_unstable_by(|&a, &b| scores.candidates[b].sequence_score
            .partial_cmp(&scores.candidates[a].sequence_score).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.ids[a].cmp(&self.ids[b])));
        if result.ranking.iter().zip(&order).any(|(id, &index)| id != &self.ids[index]) {
            return Err(ClassificationAggregationError::Result);
        }
        let best = scores.candidates[order[0]].candidate_weight;
        let runner_up = order.get(1).map_or(0.0, |&index| scores.candidates[index].candidate_weight);
        let margin = (best - runner_up).max(0.0);
        let accepted = passes(best, margin, policy);
        if !near(best, result.best_candidate_weight, self.ids.len())
            || !near(margin, result.candidate_weight_margin, self.ids.len())
            || (result.decision == ClassificationDecision::Classified) != accepted
            || result.selected_id.as_deref() != accepted.then_some(self.ids[order[0]].as_str()) {
            return Err(ClassificationAggregationError::Result);
        }
        // Divide once after folding scalar-weighted masses. In particular,
        // unanimous 1.0 retains exact endpoint thresholds for ragged chunks.
        let weight = scalars as f64;
        for ((sum, correction), candidate) in self.weights.iter_mut().zip(&mut self.corrections).zip(&scores.candidates) {
            let value = weight * candidate.candidate_weight;
            let next = *sum + value;
            *correction += if *sum >= value { (*sum - next) + value } else { (value - next) + *sum };
            *sum = next;
        }
        if accepted { self.classified_scalars = self.classified_scalars.checked_add(scalars)
            .ok_or(ClassificationAggregationError::Coverage)?; }
        Ok(())
    }
    fn finish(mut self, total: usize, chunks: usize, policy: ClassificationPolicy,
        aggregation: ClassificationAggregationPolicy) -> Result<ClassificationMixture, ClassificationAggregationError> {
        for (weight, correction) in self.weights.iter_mut().zip(self.corrections) {
            *weight = (*weight + correction) / total as f64;
        }
        if self.classified_scalars > total || !near(self.weights.iter().sum(), 1.0, self.ids.len() * chunks) {
            return Err(ClassificationAggregationError::Result);
        }
        let mut order = reserved(self.ids.len())?; order.extend(0..self.ids.len());
        order.sort_unstable_by(|&a, &b| self.weights[b].partial_cmp(&self.weights[a])
            .unwrap_or(std::cmp::Ordering::Equal).then_with(|| self.ids[a].cmp(&self.ids[b])));
        let best = self.weights[order[0]];
        let margin = (best - order.get(1).map_or(0.0, |&index| self.weights[index])).max(0.0);
        let covered = (self.classified_scalars as u128) * u128::from(SCALE)
            >= (total as u128) * u128::from(aggregation.minimum_classified_coverage_ppm);
        let accepted = covered && passes(best, margin, policy);
        let mut ranking = reserved(order.len())?;
        ranking.extend(order.iter().map(|&index| self.ids[index].clone()));
        let selected_id = accepted.then(|| self.ids[order[0]].clone());
        let mut candidate_weights = reserved(self.ids.len())?;
        candidate_weights.extend(self.ids.into_iter().zip(self.weights)
            .map(|(candidate_id, weight)| ClassificationMixtureWeight { candidate_id, weight }));
        Ok(ClassificationMixture { decision: if accepted { ClassificationDecision::Classified } else { ClassificationDecision::Abstained },
            selected_id, ranking, candidate_weights, best_candidate_weight: best, candidate_weight_margin: margin,
            classified_scalars: self.classified_scalars, classified_coverage: self.classified_scalars as f64 / total as f64 })
    }
}

pub(crate) fn aggregate_with_control<'a>(chunks: impl IntoIterator<Item = ClassificationChunkView<'a>>,
    count: usize, bytes: usize, scalars: usize, mode: ClassificationMode,
    classification_policy: ClassificationPolicy, policy: ClassificationAggregationPolicy,
    control: &mut impl DecodeStepControl) -> Result<ClassificationAggregate, ClassificationAggregationError> {
    policy.validate()?;
    classification_policy.validate().map_err(|_| ClassificationAggregationError::Policy)?;
    if count == 0 || bytes == 0 || scalars == 0 { return Err(ClassificationAggregationError::Coverage); }
    let mut labels = Vec::new(); let mut heads = Vec::new();
    let mut byte = 0; let mut scalar = 0; let mut seen = 0;
    for chunk in chunks {
        checkpoint(control)?;
        let span = chunk.span;
        if chunk.chunk_id != seen || seen >= count || span.byte_start != byte || span.scalar_start != scalar
            || span.byte_end <= byte || span.byte_end > bytes || span.scalar_end <= scalar || span.scalar_end > scalars {
            return Err(ClassificationAggregationError::Coverage);
        }
        let n = span.scalar_end - scalar;
        match (mode, chunk.result) {
            (ClassificationMode::Exclusive, ClassificationTaskResult::Exclusive(result)) => {
                if seen == 0 { heads = reserved(1)?; heads.push(Head::new(result)?); }
                heads[0].add(result, n, classification_policy)?;
            }
            (ClassificationMode::MultiLabel, ClassificationTaskResult::MultiLabel(result)) => {
                if result.schema_version != 1 || result.calibration != ClassificationCalibration::Uncalibrated
                    || result.decision_scope != "independent_binary_heads_not_a_distribution_over_labels"
                    || result.labels.is_empty() { return Err(ClassificationAggregationError::Result); }
                if seen == 0 {
                    labels = reserved(result.labels.len())?; heads = reserved(result.labels.len())?;
                    for label in &result.labels {
                        let head = Head::new(&label.classification)?;
                        if head.ids != ["no", "yes"] { return Err(ClassificationAggregationError::Result); }
                        labels.push(label.label_id.clone()); heads.push(head);
                    }
                    if labels.windows(2).any(|pair| pair[0] >= pair[1]) { return Err(ClassificationAggregationError::Result); }
                }
                if result.labels.len() != heads.len() { return Err(ClassificationAggregationError::Result); }
                let mut selected = result.selected_ids.iter(); let mut excluded = result.excluded_ids.iter();
                let mut abstained = result.abstained_ids.iter();
                for (index, (head, label)) in heads.iter_mut().zip(&result.labels).enumerate() {
                    checkpoint(control)?;
                    if label.label_id != labels[index] { return Err(ClassificationAggregationError::Result); }
                    head.add(&label.classification, n, classification_policy)?;
                    let expected = match label.classification.selected_id.as_deref() {
                        Some("yes") => MultiLabelDecision::Included, Some("no") => MultiLabelDecision::Excluded,
                        None => MultiLabelDecision::Abstained, _ => return Err(ClassificationAggregationError::Result),
                    };
                    let next = match expected { MultiLabelDecision::Included => selected.next(),
                        MultiLabelDecision::Excluded => excluded.next(), MultiLabelDecision::Abstained => abstained.next() };
                    if label.decision != expected || next != Some(&label.label_id)
                        || !near(label.positive_candidate_weight, label.classification.scores.candidates[1].candidate_weight, 2) {
                        return Err(ClassificationAggregationError::Result);
                    }
                }
                if selected.next().is_some() || excluded.next().is_some() || abstained.next().is_some() {
                    return Err(ClassificationAggregationError::Result);
                }
            }
            _ => return Err(ClassificationAggregationError::Result),
        }
        byte = span.byte_end; scalar = span.scalar_end; seen += 1;
    }
    if seen != count || byte != bytes || scalar != scalars { return Err(ClassificationAggregationError::Coverage); }
    checkpoint(control)?;
    let result = match mode {
        ClassificationMode::Exclusive => ClassificationAggregateResult::Exclusive(heads.pop()
            .ok_or(ClassificationAggregationError::Result)?.finish(scalars, count, classification_policy, policy)?),
        ClassificationMode::MultiLabel => {
            let mut output = MultiLabelAggregate { selected_ids: reserved(labels.len())?, excluded_ids: reserved(labels.len())?,
                abstained_ids: reserved(labels.len())?, labels: reserved(labels.len())? };
            for (label_id, head) in labels.into_iter().zip(heads) {
                checkpoint(control)?;
                let mixture = head.finish(scalars, count, classification_policy, policy)?;
                let decision = match mixture.selected_id.as_deref() {
                    Some("yes") => { output.selected_ids.push(label_id.clone()); MultiLabelDecision::Included },
                    Some("no") => { output.excluded_ids.push(label_id.clone()); MultiLabelDecision::Excluded },
                    None => { output.abstained_ids.push(label_id.clone()); MultiLabelDecision::Abstained },
                    _ => return Err(ClassificationAggregationError::Result),
                };
                output.labels.push(MultiLabelAggregateLabel { label_id, decision,
                    positive_candidate_weight: mixture.candidate_weights[1].weight, mixture });
            }
            ClassificationAggregateResult::MultiLabel(output)
        }
    };
    checkpoint(control)?;
    Ok(ClassificationAggregate { schema_version: 1, profile: CLASSIFICATION_AGGREGATION_PROFILE.to_owned(),
        weight_basis: "original_unicode_scalar_count_including_abstained_chunks".to_owned(),
        score_scope: match mode {
            ClassificationMode::Exclusive => "source_scalar_mixture_of_chunk_candidate_weights",
            ClassificationMode::MultiLabel => "independent_source_scalar_yes_no_mixture_per_label",
        }.to_owned(), calibration: ClassificationCalibration::Uncalibrated, policy, classification_policy,
        chunk_count: count, source_bytes: bytes, source_scalars: scalars, result })
}
fn passes(best: f64, margin: f64, policy: ClassificationPolicy) -> bool {
    best * f64::from(SCALE) >= f64::from(policy.minimum_candidate_weight_ppm)
        && margin * f64::from(SCALE) >= f64::from(policy.minimum_margin_ppm)
}
fn near(a: f64, b: f64, terms: usize) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 64.0 * f64::EPSILON * terms.max(1) as f64
}
fn reserved<T>(count: usize) -> Result<Vec<T>, ClassificationAggregationError> {
    let mut out = Vec::new(); out.try_reserve_exact(count).map_err(|_| ClassificationAggregationError::Allocation)?; Ok(out)
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), ClassificationAggregationError> {
    match control.prefill_checkpoint(0) { None => Ok(()), Some(cause) => Err(ClassificationAggregationError::Cancelled(cause)) }
}

#[cfg(test)] mod tests;
