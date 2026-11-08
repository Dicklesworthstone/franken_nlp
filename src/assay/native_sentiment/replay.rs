//! Sealed native observations; external score files cannot construct a replay.
use serde::Serialize;
use sha2::{Digest, Sha256};
use crate::{execution_identity::{ExecutionIdentity, NumericsProfile, Sha256Digest, ThinkingMode, ToolMode},
    native_engine::{decode::{DecodeCancellationKind, DecodeStepControl}, lmhead::NANBEIGE_VOCAB_SIZE,
        strict_int4::{Int4ModelIdentity, Int4Work, STRICT_INT4_PROFILE}},
    tasks::{ir::{ScoreSpace, TaskBudget}, sentiment::{SentimentDecision, SentimentDimension,
        SentimentPolicy, SentimentResult, grouped_int4::{self, Int4SentimentRun,
            PreparedInt4Sentiment, INT4_SENTIMENT_EXECUTION}}}};
use super::*;

const BIN_VALUES: [i32; 5] = [-1000, -500, 0, 500, 1000];
const BIN_IDS: [&str; 5] = ["bin-0", "bin-1", "bin-2", "bin-3", "bin-4"];
// The fixed native prompt uses exact byte-fallback numeric strings plus EOS.
const SCORED_TOKENS: [usize; 5] = [3, 5, 2, 4, 2];
const PREFIXES: usize = 10;
const EDGES: usize = 14;

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum ReplayError { Assay(AssayError), Cancelled(DecodeCancellationKind) }
impl From<AssayError> for ReplayError { fn from(value: AssayError) -> Self { Self::Assay(value) } }

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
    pub(super) group_width: usize,
    pub(super) heads: usize,
}
pub(super) struct ScoredRow {
    pub id: String,
    pub source: Sha256Digest,
    pub gold: Vec<Gold>,
    pub means: Vec<f64>,
    pub estimated: Vec<bool>,
}

pub(crate) struct Collector<'a> { model: &'a Int4ModelIdentity, builder: ReplayBuilder<'a> }
impl<'a> Collector<'a> {
    pub(crate) fn new(dataset: &'a Dataset, model: &'a Int4ModelIdentity, limits: Limits) -> Result<Self, AssayError> {
        Ok(Self { model, builder: ReplayBuilder::new(dataset, limits)? })
    }
    pub(crate) fn accept(&mut self, index: usize, prepared: &PreparedInt4Sentiment,
        run: Int4SentimentRun, control: &mut impl DecodeStepControl) -> Result<(), ReplayError> {
        self.builder.enter(index)?; checkpoint(control)?;
        let identity = prepared.execution_identity();
        prepared.verify_identity(identity).map_err(|_| AssayError::Identity)?;
        grouped_int4::check_model(identity, self.model).map_err(|_| AssayError::Identity)?;
        if prepared.group_width() != self.model.group_width()
            || prepared.head_count() != self.builder.dataset.manifest.axes.len() {
            return Err(AssayError::Identity.into());
        }
        self.builder.append(index, identity, prepared.task_budget(), prepared.planned_work(),
            prepared.group_width(), prepared.max_result_bytes(), run, control)?;
        self.builder.failed = false; Ok(())
    }
    pub(crate) fn finish(self) -> Result<NativeReplay, AssayError> { self.builder.finish() }
}

/// Only module-local arithmetic/tamper fixtures can append external values.
/// The resident host uses Collector immediately after executing its sealed row.
pub(super) struct ReplayBuilder<'a> {
    dataset: &'a Dataset,
    rows: Vec<ScoredRow>,
    first_identity: Option<ExecutionIdentity>,
    executions: Sha256,
    compatible_executions: Sha256,
    observations: Sha256,
    model_work: Int4Work,
    width: Option<usize>,
    heads: usize,
    eos: u32,
    failed: bool,
}
impl<'a> ReplayBuilder<'a> {
    pub(super) fn new(dataset: &'a Dataset, limits: Limits) -> Result<Self, AssayError> {
        dataset.validate_limits(limits)?;
        let registry = crate::tokenizer::pinned_controls::pinned().map_err(|_| AssayError::Identity)?;
        let mut ends = registry.template_controls().entries().iter()
            .filter(|entry| entry.special && entry.surface == crate::template::IM_END);
        let eos = ends.next().ok_or(AssayError::Identity)?.id;
        if ends.next().is_some() { return Err(AssayError::Identity); }
        let hash = |domain: &[u8]| { let mut value = Sha256::new(); value.update(domain);
            value.update((dataset.replay_len() as u64).to_be_bytes()); value };
        Ok(Self { dataset, rows: reserved(dataset.replay_len())?, first_identity: None,
            executions: hash(b"fnlp.native_sentiment.executions.v1\0"),
            compatible_executions: hash(b"fnlp.native_sentiment.pair.v1\0"),
            observations: hash(b"fnlp.native_sentiment.observations.v1\0"),
            model_work: Int4Work::default(), width: None, heads: 0, eos, failed: false })
    }
    fn enter(&mut self, index: usize) -> Result<(), AssayError> {
        if self.failed { return Err(AssayError::Observation); }
        self.failed = true;
        if index != self.rows.len() || index >= self.dataset.replay_len() { return Err(AssayError::Scope); }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn append(&mut self, index: usize, identity: &ExecutionIdentity, budget: TaskBudget,
        planned: Int4Work, width: usize, output_bytes: u64, run: Int4SentimentRun,
        control: &mut impl DecodeStepControl) -> Result<(), ReplayError> {
        checkpoint(control)?;
        let row = self.dataset.replay_rows().get(index).ok_or(AssayError::Scope)?;
        identity.validate().map_err(|_| AssayError::Identity)?;
        budget.validate().map_err(|_| AssayError::Limits)?;
        let axes = &self.dataset.manifest.axes;
        if identity.task_spec != "sentiment-v1"
            || identity.numerics_profile != (NumericsProfile::StrictQuantized { version: 2 })
            || identity.thinking_mode != ThinkingMode::Disabled || identity.tool_mode != ToolMode::None
            || identity.calibration_digest != Sha256Digest::of_bytes(b"null")
            || identity.decision_policy_digest != decision_digest(identity, width)?
            || width == 0 || self.width.is_some_and(|prior| prior != width)
            || run.schema_version != ASSAY_SCHEMA_VERSION || run.execution != INT4_SENTIMENT_EXECUTION
            || run.numerics_profile != STRICT_INT4_PROFILE || run.group_width != width
            || run.head_count != axes.len() || run.model_work != planned
            || planned.forward_positions == 0 || planned.projected_logits == 0
            || planned.attention_pairs == 0 || planned.projections.dot_products == 0
            || planned.projections.multiply_accumulates == 0 || output_bytes > budget.max_output_bytes
            || run.rewound_positions > run.model_work.forward_positions {
            return Err(AssayError::Identity.into());
        }
        let (means, estimated) = validate_result(&run.result, axes, self.dataset.manifest.policy, self.eos, planned)?;
        let observed = crate::canonjson::canonical_bytes(&run).map_err(|_| AssayError::Commitment)?;
        if observed.len() as u64 > output_bytes { return Err(AssayError::Output.into()); }
        checkpoint(control)?;
        let source = Sha256Digest::of_bytes(row.text.as_bytes());
        #[derive(Serialize)]
        struct Binding<'a> { index: usize, id: &'a str, split: Split, source: Sha256Digest,
            gold: &'a [Gold], budget: TaskBudget }
        let record = crate::canonjson::canonical_bytes(&Binding { index, id: &row.id, split: row.split,
            source, gold: &row.gold, budget }).map_err(|_| AssayError::Commitment)?;
        let full = identity.canonical_json_bytes().map_err(|_| AssayError::Identity)?;
        let compatible = compatible_identity(identity)?.canonical_json_bytes().map_err(|_| AssayError::Identity)?;
        if full.len() > 16_384 || compatible.len() > 16_384 { return Err(AssayError::Identity.into()); }
        frame(&mut self.executions, &record); frame(&mut self.executions, &full);
        frame(&mut self.compatible_executions, &record); frame(&mut self.compatible_executions, &compatible);
        frame(&mut self.observations, &record); frame(&mut self.observations, &observed);
        self.model_work = self.model_work.checked_add(planned).map_err(|_| AssayError::InputLimit)?;
        self.heads = self.heads.checked_add(axes.len()).ok_or(AssayError::InputLimit)?;
        self.rows.push(ScoredRow { id: row.id.clone(), source, gold: row.gold.clone(), means, estimated });
        if self.first_identity.is_none() { self.first_identity = Some(identity.clone()); }
        self.width = Some(width); checkpoint(control)?; Ok(())
    }
    #[cfg(test)]
    pub(super) fn accept_fixture(&mut self, index: usize, identity: &ExecutionIdentity,
        budget: TaskBudget, planned: Int4Work, width: usize, run: Int4SentimentRun,
        control: &mut impl DecodeStepControl) -> Result<(), ReplayError> {
        self.enter(index)?;
        self.append(index, identity, budget, planned, width, budget.max_output_bytes, run, control)?;
        self.failed = false; Ok(())
    }
    pub(super) fn finish(self) -> Result<NativeReplay, AssayError> {
        if self.failed || self.rows.len() != self.dataset.replay_len() { return Err(AssayError::Observation); }
        Ok(NativeReplay { manifest: self.dataset.manifest.clone(), manifest_digest: self.dataset.manifest_digest,
            input_digest: self.dataset.input_digest, sizes: self.dataset.sizes, rows: self.rows,
            first_identity: self.first_identity.ok_or(AssayError::Identity)?,
            executions: self.executions.finalize().into(), compatible_executions: self.compatible_executions.finalize().into(),
            observations: self.observations.finalize().into(), model_work: self.model_work,
            group_width: self.width.ok_or(AssayError::Identity)?, heads: self.heads })
    }
}

fn validate_result(result: &SentimentResult, axes: &[SentimentAxis], policy: SentimentPolicy,
    eos: u32, planned: Int4Work) -> Result<(Vec<f64>, Vec<bool>), AssayError> {
    let heads = axes.len();
    if result.schema_version != 1 || result.task_spec != "sentiment-v1"
        || result.interpretation != "independent_conditional_candidate_weight_projections"
        || result.calibration != "uncalibrated_not_a_psychological_measurement"
        || result.policy != policy || result.dimensions.len() != heads
        || result.dimensions.iter().map(|axis| axis.axis).ne(axes.iter().copied())
        || result.work.dimensions != heads || result.work.candidates != heads * BIN_VALUES.len()
        || result.work.prefix_evaluations != heads * PREFIXES || result.work.scored_edges != heads * EDGES
        || result.work.projected_logits != planned.projected_logits
        || planned.projected_logits != (heads * PREFIXES) as u64 * NANBEIGE_VOCAB_SIZE as u64 {
        return Err(AssayError::Observation);
    }
    let mut means = reserved(heads)?; let mut estimated = reserved(heads)?;
    for dimension in &result.dimensions {
        validate_dimension(dimension, policy, eos)?;
        means.push(dimension.distribution_mean);
        estimated.push(dimension.decision == SentimentDecision::Estimated);
    }
    Ok((means, estimated))
}
fn validate_dimension(dimension: &SentimentDimension, policy: SentimentPolicy, eos: u32) -> Result<(), AssayError> {
    let scores = &dimension.scores; let anchors = dimension.axis.anchors();
    if dimension.low_anchor != anchors.0 || dimension.high_anchor != anchors.1
        || dimension.anchors.len() != 5 || scores.candidates.len() != 5
        || scores.score_space != ScoreSpace::FullVocabSequenceLogprob || !scores.full_vocab_denominators_computed
        || scores.normalization_scope != "full_vocabulary_per_position; candidate_weights_over_completed_candidates"
        || scores.eos_rule != "append_and_score_exactly_one_eos" || scores.length_rule != "sum_logprobs_including_eos"
        || scores.eos_token_id != eos || scores.work.prefix_evaluations != PREFIXES || scores.work.scored_edges != EDGES
        || scores.work.projected_logits != PREFIXES as u64 * NANBEIGE_VOCAB_SIZE as u64 {
        return Err(AssayError::Observation);
    }
    let (mut sum, mut mean, mut entropy, mut peak) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut modal, mut best) = (-1000, f64::NEG_INFINITY);
    for (index, (anchor, candidate)) in dimension.anchors.iter().zip(&scores.candidates).enumerate() {
        if anchor.value_milli != BIN_VALUES[index] || anchor.candidate_id != BIN_IDS[index]
            || candidate.id != BIN_IDS[index] || candidate.scored_tokens != SCORED_TOKENS[index]
            || !candidate.sequence_score.is_finite() || candidate.sequence_score > 0.0
            || !candidate.candidate_weight.is_finite()
            || !(0.0..=1.0).contains(&candidate.candidate_weight) { return Err(AssayError::Observation); }
        let weight = candidate.candidate_weight;
        sum += weight; mean += weight * f64::from(anchor.value_milli) / f64::from(GOLD_SCALE);
        if weight > 0.0 { entropy -= weight * weight.ln(); }
        peak = peak.max(weight);
        if candidate.sequence_score > best { best = candidate.sequence_score; modal = anchor.value_milli; }
    }
    let tolerance = 160.0 * f64::EPSILON;
    let denominator: f64 = scores.candidates.iter().map(|candidate| (candidate.sequence_score - best).exp()).sum();
    if (sum - 1.0).abs() > tolerance || scores.candidates.iter().any(|candidate|
        ((candidate.sequence_score - best).exp() / denominator - candidate.candidate_weight).abs() > tolerance) {
        return Err(AssayError::Observation);
    }
    let mean = mean.clamp(-1.0, 1.0); let entropy = (entropy / 5.0_f64.ln()).clamp(0.0, 1.0);
    let variance: f64 = scores.candidates.iter().zip(BIN_VALUES).map(|(candidate, value)|
        candidate.candidate_weight * (f64::from(value) / f64::from(GOLD_SCALE) - mean).powi(2)).sum();
    let estimated = peak >= f64::from(policy.minimum_peak_weight_ppm) / 1_000_000.0
        && entropy <= f64::from(policy.maximum_normalized_entropy_ppm) / 1_000_000.0;
    if !near(dimension.distribution_mean, mean) || !near(dimension.distribution_stddev, variance.max(0.0).sqrt())
        || !near(dimension.normalized_entropy, entropy) || dimension.modal_value_milli != modal
        || dimension.decision != if estimated { SentimentDecision::Estimated } else { SentimentDecision::Abstained }
        || match dimension.estimate { Some(value) => !estimated || !near(value, mean), None => estimated } {
        return Err(AssayError::Observation);
    }
    Ok(())
}
fn near(a: f64, b: f64) -> bool { a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 }
fn frame(hash: &mut Sha256, bytes: &[u8]) { hash.update((bytes.len() as u64).to_be_bytes()); hash.update(bytes); }
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), ReplayError> {
    control.checkpoint(0).map_or(Ok(()), |kind| Err(ReplayError::Cancelled(kind)))
}
fn decision_digest(identity: &ExecutionIdentity, width: usize) -> Result<Sha256Digest, AssayError> {
    crate::canonjson::canonical_bytes(&(INT4_SENTIMENT_EXECUTION, identity.taskir_digest, width))
        .map(|bytes| Sha256Digest::of_bytes(&bytes)).map_err(|_| AssayError::Identity)
}
fn compatible_identity(identity: &ExecutionIdentity) -> Result<ExecutionIdentity, AssayError> {
    let mut value = identity.clone(); let absent = Sha256Digest::of_bytes(b"native-assay-comparison-subject-v1");
    value.logical_model_digest = absent; value.packing_set_digest = absent;
    value.artifact_format = "comparison-subject".to_owned(); value.quant_recipe = "comparison-subject".to_owned();
    value.backend_semantic_version = "comparison-subject".to_owned(); value.host_class = None; value.compiler_identity = None;
    // Verify the real digest above, then exclude only the compared group
    // partition. The complete unmodified identity remains separately bound.
    value.decision_policy_digest = decision_digest(identity, 0)?; Ok(value)
}
