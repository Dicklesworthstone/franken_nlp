//! A descriptive mixture of complete, disjoint source-chunk distributions.
//!
//! This reducer does not score text, calibrate a model, or grant native
//! execution authority. Every chunk, including abstained chunks, contributes
//! its original Unicode scalar count. A mixture is not the model's posterior
//! for the full document and never averages incomparable sequence log scores.

use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};

use crate::{
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl},
    tasks::ir::ScoreSpace,
    validation::grounded_fields::VerifiedSourceSpan,
};
use super::{SentimentAxis, SentimentDecision, SentimentDimension, SentimentPolicy, SentimentResult};

pub const SENTIMENT_AGGREGATION_PROFILE: &str = "source-scalar-candidate-mixture-v1";
const PPM: u32 = 1_000_000;
const MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;
const MAX_CHUNKS: usize = 1_000_000;
const MAX_BINS: usize = 33;
const TOLERANCE: f64 = 1e-10;

/// A caller-owned, uncalibrated coverage rule. Even when this rule passes,
/// the mixture must pass the original peak/entropy policy before an estimate
/// is present. Lowering this threshold never removes abstained chunk weights.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SentimentAggregationPolicy {
    pub minimum_estimated_coverage_ppm: u32,
}

impl Default for SentimentAggregationPolicy {
    fn default() -> Self { Self { minimum_estimated_coverage_ppm: PPM } }
}
impl SentimentAggregationPolicy {
    pub fn validate(self) -> Result<(), SentimentAggregationError> {
        if self.minimum_estimated_coverage_ppm > PPM {
            return Err(SentimentAggregationError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SentimentMixtureWeight {
    pub candidate_id: String,
    pub value_milli: i32,
    /// Scalar-weighted mixture of each chunk's conditional candidate weight.
    /// This is not a full-document sequence probability or captured mass.
    pub weight: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SentimentAggregateDimension {
    pub axis: SentimentAxis,
    pub low_anchor: String,
    pub high_anchor: String,
    pub chunk_score_space: ScoreSpace,
    pub decision: SentimentDecision,
    pub estimate: Option<f64>,
    pub distribution_mean: f64,
    /// Average conditional variance within the contributing chunks.
    pub within_chunk_variance: f64,
    /// Weighted variance of chunk means. Opposing decisive chunks remain
    /// distinguishable from a consistently central distribution.
    pub between_chunk_variance: f64,
    pub distribution_stddev: f64,
    pub normalized_entropy: f64,
    pub modal_value_milli: i32,
    pub estimated_scalars: usize,
    pub abstained_scalars: usize,
    pub estimated_coverage: f64,
    pub coverage_passed: bool,
    pub mixture_policy_passed: bool,
    pub mixture_weights: Vec<SentimentMixtureWeight>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SentimentAggregate {
    pub schema_version: u32,
    pub reduction_profile: String,
    pub weight_unit: String,
    pub interpretation: String,
    pub calibration: String,
    pub policy: SentimentAggregationPolicy,
    pub chunk_policy: SentimentPolicy,
    pub chunk_count: usize,
    pub source_bytes: usize,
    pub source_scalars: usize,
    pub dimensions: Vec<SentimentAggregateDimension>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SentimentAggregationError {
    InvalidPolicy,
    InvalidCoverage,
    InvalidDistribution,
    IncompatibleChunks,
    Allocation,
    Cancelled(DecodeCancellationKind),
}
impl fmt::Display for SentimentAggregationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidPolicy => "sentiment aggregation coverage must be in 0..=1000000",
            Self::InvalidCoverage => "sentiment aggregation requires the complete ordered source partition",
            Self::InvalidDistribution => "sentiment aggregation received an invalid chunk distribution",
            Self::IncompatibleChunks => "sentiment aggregation chunk semantics differ",
            Self::Allocation => "sentiment aggregation bounded allocation refused",
            Self::Cancelled(_) => "sentiment aggregation cancelled",
        })
    }
}
impl Error for SentimentAggregationError {}

/// Only a native map's sealed source geometry should create these views.
/// Keeping the reducer crate-private prevents a wire result from becoming a
/// substitute native execution path. Tests exercise the arithmetic separately.
pub(crate) struct SentimentChunkView<'a> {
    pub chunk_id: usize,
    pub span: VerifiedSourceSpan,
    pub result: &'a SentimentResult,
}

/// Linear in the bounded number of chunks, with at most four fixed-size axis
/// accumulators. No source bytes, chunk copies, hierarchical merge buffers or
/// input-sized temporary vectors are allocated here. Canonical source order
/// makes the reduction independent of native scheduling/cohort width.
pub(crate) fn aggregate_with_control<'a, I, C>(
    chunks: I, expected_chunks: usize, source_bytes: usize, source_scalars: usize,
    policy: SentimentAggregationPolicy, control: &mut C,
) -> Result<SentimentAggregate, SentimentAggregationError>
where I: IntoIterator<Item = SentimentChunkView<'a>>, C: DecodeStepControl {
    policy.validate()?;
    if !(1..=MAX_CHUNKS).contains(&expected_chunks)
        || !(1..=MAX_SOURCE_BYTES).contains(&source_bytes)
        || source_scalars == 0 || source_scalars > source_bytes {
        return Err(SentimentAggregationError::InvalidCoverage);
    }
    checkpoint(control)?;
    let mut chunks = chunks.into_iter();
    let first = chunks.next().ok_or(SentimentAggregationError::InvalidCoverage)?;
    let reference = first.result;
    validate_result(reference)?;
    let mut axes = reserved(reference.dimensions.len())?;
    for dimension in &reference.dimensions { axes.push(AxisAccumulator::new(dimension)?); }
    let (mut count, mut byte_end, mut scalar_end) = (0_usize, 0_usize, 0_usize);
    for chunk in std::iter::once(first).chain(chunks) {
        checkpoint(control)?;
        let span = chunk.span;
        if count == expected_chunks || chunk.chunk_id != count
            || span.byte_start != byte_end || span.scalar_start != scalar_end
            || span.byte_end <= span.byte_start || span.scalar_end <= span.scalar_start
            || span.byte_end > source_bytes || span.scalar_end > source_scalars {
            return Err(SentimentAggregationError::InvalidCoverage);
        }
        let weight = span.scalar_end - span.scalar_start;
        let bytes = span.byte_end - span.byte_start;
        if bytes < weight || bytes > weight.checked_mul(4).ok_or(SentimentAggregationError::InvalidCoverage)? {
            return Err(SentimentAggregationError::InvalidCoverage);
        }
        validate_result(chunk.result)?;
        if chunk.result.policy != reference.policy
            || chunk.result.dimensions.len() != axes.len() {
            return Err(SentimentAggregationError::IncompatibleChunks);
        }
        for (axis, dimension) in axes.iter_mut().zip(&chunk.result.dimensions) {
            checkpoint(control)?;
            axis.add(dimension, reference.policy, weight)?;
        }
        count += 1;
        byte_end = span.byte_end;
        scalar_end = span.scalar_end;
    }
    if count != expected_chunks || byte_end != source_bytes || scalar_end != source_scalars {
        return Err(SentimentAggregationError::InvalidCoverage);
    }
    let mut dimensions = reserved(axes.len())?;
    for axis in axes {
        checkpoint(control)?;
        dimensions.push(axis.finish(source_scalars, policy, reference.policy)?);
    }
    checkpoint(control)?;
    Ok(SentimentAggregate {
        schema_version: 1, reduction_profile: SENTIMENT_AGGREGATION_PROFILE.to_owned(),
        weight_unit: "original_unicode_scalar_count".to_owned(),
        interpretation: "descriptive_chunk_candidate_mixture_not_full_document_inference".to_owned(),
        calibration: "uncalibrated_not_a_psychological_measurement".to_owned(),
        policy, chunk_policy: reference.policy, chunk_count: count,
        source_bytes, source_scalars, dimensions,
    })
}

fn validate_result(result: &SentimentResult) -> Result<(), SentimentAggregationError> {
    if result.schema_version != 1 || result.task_spec != "sentiment-v1"
        || result.interpretation != "independent_conditional_candidate_weight_projections"
        || result.calibration != "uncalibrated_not_a_psychological_measurement"
        || result.policy.minimum_peak_weight_ppm > PPM
        || result.policy.maximum_normalized_entropy_ppm > PPM
        || result.dimensions.is_empty() || result.dimensions.len() > SentimentAxis::ALL.len()
        || result.dimensions.windows(2).any(|pair| pair[0].axis >= pair[1].axis) {
        return Err(SentimentAggregationError::IncompatibleChunks);
    }
    Ok(())
}

struct AxisAccumulator<'a> {
    reference: &'a SentimentDimension,
    bins: Vec<CompensatedSum>,
    scalar_count: usize,
    estimated_scalars: usize,
    mean: f64,
    between: CompensatedSum,
    within: CompensatedSum,
}
impl<'a> AxisAccumulator<'a> {
    fn new(reference: &'a SentimentDimension) -> Result<Self, SentimentAggregationError> {
        let count = reference.anchors.len();
        let (low, high) = reference.axis.anchors();
        if !(3..=MAX_BINS).contains(&count) || count % 2 == 0 || 2000 % (count - 1) != 0
            || reference.low_anchor != low || reference.high_anchor != high {
            return Err(SentimentAggregationError::InvalidDistribution);
        }
        for (index, anchor) in reference.anchors.iter().enumerate() {
            if anchor.value_milli != -1000 + (index * (2000 / (count - 1))) as i32
                || anchor.candidate_id.is_empty() || anchor.candidate_id.len() > 64
                || reference.anchors[..index].iter().any(|old| old.candidate_id == anchor.candidate_id) {
                return Err(SentimentAggregationError::InvalidDistribution);
            }
        }
        let mut bins = reserved(count)?;
        bins.resize(count, CompensatedSum::default());
        Ok(Self { reference, bins, scalar_count: 0, estimated_scalars: 0,
            mean: 0.0, between: CompensatedSum::default(), within: CompensatedSum::default() })
    }

    fn add(&mut self, dimension: &SentimentDimension, policy: SentimentPolicy, weight: usize)
        -> Result<(), SentimentAggregationError> {
        let reference = self.reference;
        let scores = &dimension.scores;
        let expected = &reference.scores;
        if dimension.axis != reference.axis || dimension.anchors != reference.anchors
            || dimension.low_anchor != reference.low_anchor || dimension.high_anchor != reference.high_anchor
            || scores.score_space != expected.score_space || scores.normalization_scope != expected.normalization_scope
            || scores.length_rule != expected.length_rule || scores.eos_rule != expected.eos_rule
            || scores.eos_token_id != expected.eos_token_id
            || scores.full_vocab_denominators_computed != expected.full_vocab_denominators_computed {
            return Err(SentimentAggregationError::IncompatibleChunks);
        }
        if scores.score_space == ScoreSpace::NotComputed
            || scores.candidates.len() != self.bins.len()
            || scores.candidates.windows(2).any(|pair| pair[0].id >= pair[1].id) {
            return Err(SentimentAggregationError::InvalidDistribution);
        }
        let (mut sum, mut mean, mut entropy, mut peak) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let (mut modal, mut modal_score) = (-1000, f64::NEG_INFINITY);
        for (index, anchor) in dimension.anchors.iter().enumerate() {
            let score = scores.candidates.iter().find(|score| score.id == anchor.candidate_id)
                .ok_or(SentimentAggregationError::InvalidDistribution)?;
            let prior = expected.candidates.iter().find(|score| score.id == anchor.candidate_id)
                .ok_or(SentimentAggregationError::InvalidDistribution)?;
            let probability = score.candidate_weight;
            if !score.sequence_score.is_finite() || !probability.is_finite()
                || !(0.0..=1.0).contains(&probability) || score.scored_tokens < 2
                || score.scored_tokens != prior.scored_tokens {
                return Err(SentimentAggregationError::InvalidDistribution);
            }
            sum += probability;
            mean += probability * f64::from(anchor.value_milli) / 1000.0;
            if probability > 0.0 { entropy -= probability * probability.ln(); }
            peak = peak.max(probability);
            if score.sequence_score > modal_score { modal_score = score.sequence_score; modal = anchor.value_milli; }
            self.bins[index].add(weight as f64 * probability);
        }
        if !near(sum, 1.0) { return Err(SentimentAggregationError::InvalidDistribution); }
        let mean = mean.clamp(-1.0, 1.0);
        let entropy = (entropy / (self.bins.len() as f64).ln()).clamp(0.0, 1.0);
        let mut variance = 0.0;
        for anchor in &dimension.anchors {
            let probability = scores.candidates.iter().find(|score| score.id == anchor.candidate_id)
                .ok_or(SentimentAggregationError::InvalidDistribution)?.candidate_weight;
            let delta = f64::from(anchor.value_milli) / 1000.0 - mean;
            variance += probability * delta * delta;
        }
        let accepted = passes(policy, peak, entropy);
        if !near(dimension.distribution_mean, mean)
            || !near(dimension.distribution_stddev, variance.sqrt())
            || !near(dimension.normalized_entropy, entropy)
            || dimension.modal_value_milli != modal
            || dimension.decision != if accepted { SentimentDecision::Estimated } else { SentimentDecision::Abstained }
            || match dimension.estimate { Some(value) => !accepted || !near(value, mean), None => accepted } {
            return Err(SentimentAggregationError::InvalidDistribution);
        }
        self.scalar_count = self.scalar_count.checked_add(weight).ok_or(SentimentAggregationError::InvalidCoverage)?;
        if accepted {
            self.estimated_scalars = self.estimated_scalars.checked_add(weight)
                .ok_or(SentimentAggregationError::InvalidCoverage)?;
        }
        // Weighted Welford accumulation avoids cancellation in E[mean²]-E[mean]².
        // The one canonical leaf order is preserved for every native schedule.
        let delta = mean - self.mean;
        self.mean += delta * (weight as f64 / self.scalar_count as f64);
        self.between.add(weight as f64 * delta * (mean - self.mean));
        self.within.add(weight as f64 * variance);
        Ok(())
    }

    fn finish(self, total: usize, policy: SentimentAggregationPolicy, chunk_policy: SentimentPolicy)
        -> Result<SentimentAggregateDimension, SentimentAggregationError> {
        if self.scalar_count != total || total == 0 {
            return Err(SentimentAggregationError::InvalidCoverage);
        }
        let mut mixture_weights = reserved(self.bins.len())?;
        let (mut sum, mut mean, mut entropy, mut peak) = (0.0_f64, 0.0_f64, 0.0_f64, -1.0_f64);
        let mut modal = -1000;
        for (bin, anchor) in self.bins.iter().zip(&self.reference.anchors) {
            let weight = bin.value() / total as f64;
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err(SentimentAggregationError::InvalidDistribution);
            }
            sum += weight;
            mean += weight * f64::from(anchor.value_milli) / 1000.0;
            if weight > 0.0 { entropy -= weight * weight.ln(); }
            // Lowest-coordinate tie-break is defined for the mixture weights;
            // there is deliberately no aggregate sequence score to break ties.
            if weight > peak { peak = weight; modal = anchor.value_milli; }
            mixture_weights.push(SentimentMixtureWeight {
                candidate_id: anchor.candidate_id.clone(), value_milli: anchor.value_milli, weight,
            });
        }
        if !near(sum, 1.0) || !near(mean, self.mean) {
            return Err(SentimentAggregationError::InvalidDistribution);
        }
        let mean = mean.clamp(-1.0, 1.0);
        let entropy = (entropy / (self.bins.len() as f64).ln()).clamp(0.0, 1.0);
        let within = (self.within.value() / total as f64).max(0.0);
        let between = (self.between.value() / total as f64).max(0.0);
        let mixture_variance = mixture_weights.iter().map(|bin| {
            let delta = f64::from(bin.value_milli) / 1000.0 - mean;
            bin.weight * delta * delta
        }).sum::<f64>();
        if !near(within + between, mixture_variance) {
            return Err(SentimentAggregationError::InvalidDistribution);
        }
        // Integer coverage comparison avoids threshold rounding near 100%.
        let coverage_passed = (self.estimated_scalars as u128) * u128::from(PPM)
            >= (total as u128) * u128::from(policy.minimum_estimated_coverage_ppm);
        let mixture_policy_passed = passes(chunk_policy, peak, entropy);
        let accepted = coverage_passed && mixture_policy_passed;
        Ok(SentimentAggregateDimension {
            axis: self.reference.axis, low_anchor: self.reference.low_anchor.clone(),
            high_anchor: self.reference.high_anchor.clone(), chunk_score_space: self.reference.scores.score_space,
            decision: if accepted { SentimentDecision::Estimated } else { SentimentDecision::Abstained },
            estimate: accepted.then_some(mean), distribution_mean: mean,
            within_chunk_variance: within, between_chunk_variance: between,
            distribution_stddev: (within + between).sqrt(), normalized_entropy: entropy,
            modal_value_milli: modal, estimated_scalars: self.estimated_scalars,
            abstained_scalars: total - self.estimated_scalars,
            estimated_coverage: self.estimated_scalars as f64 / total as f64,
            coverage_passed, mixture_policy_passed, mixture_weights,
        })
    }
}

#[derive(Clone, Copy, Default)]
struct CompensatedSum { sum: f64, correction: f64 }
impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let adjusted = value - self.correction;
        let next = self.sum + adjusted;
        self.correction = (next - self.sum) - adjusted;
        self.sum = next;
    }
    fn value(&self) -> f64 { self.sum }
}
fn near(actual: f64, expected: f64) -> bool {
    actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= TOLERANCE
}
fn passes(policy: SentimentPolicy, peak: f64, entropy: f64) -> bool {
    peak >= f64::from(policy.minimum_peak_weight_ppm) / f64::from(PPM)
        && entropy <= f64::from(policy.maximum_normalized_entropy_ppm) / f64::from(PPM)
}
fn reserved<T>(count: usize) -> Result<Vec<T>, SentimentAggregationError> {
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| SentimentAggregationError::Allocation)?;
    Ok(values)
}
fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), SentimentAggregationError> {
    match control.checkpoint(0) {
        Some(kind) => Err(SentimentAggregationError::Cancelled(kind)), None => Ok(()),
    }
}

#[cfg(test)]
mod tests;
