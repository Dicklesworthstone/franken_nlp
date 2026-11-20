//! Descriptive document replicates from complete, disjoint sampled chunks.
//!
//! Replicate j is the exact scalar-weighted sum of chunk j-th scores divided
//! by the complete source scalar count. This is an ordinal pairing of separate
//! chunk draws, not another full-context model call or a confidence interval.
//! The parent seals chunk addresses and native completions; this reducer never
//! turns caller-supplied wire data into native execution authority.

use super::{SampledSentimentMapChunk, SampledSentimentMapError};
use crate::{
    native_engine::{constrained_int4::JSON_SAMPLER_VERSION,
        decode::DecodeStepControl, lmhead::NANBEIGE_VOCAB_SIZE,
        quant_algebra::MAX_MODEL_K, strict_int4::{Int4Work, STRICT_INT4_PROFILE}},
    tasks::sentiment::{SentimentAxis, sampled::{SampledSentimentDimension,
        SampledSentimentRun, MAX_SENTIMENT_SAMPLES, SAMPLED_SENTIMENT_AGGREGATION_VERSION,
        SAMPLED_SENTIMENT_EXECUTION, SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION}},
    validation::grounded_fields::VerifiedSourceSpan,
};
use serde::Serialize;

pub const SAMPLED_SENTIMENT_MAP_AGGREGATION: &str = "source-scalar-ordinal-paired-sampled-replicates-v1";
const MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;
const MAX_CHUNKS: usize = 256;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SampledSentimentMapReplicate {
    pub sample_index: usize,
    /// Exact sum of original chunk scalar counts times integer sample scores.
    /// The denominator is the enclosing aggregate's source_scalars.
    pub weighted_score_sum: i64,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SampledSentimentMapAggregateDimension {
    pub axis: SentimentAxis,
    pub low_anchor: String,
    pub high_anchor: String,
    pub mean: f64,
    pub median: f64,
    pub minimum: f64,
    pub maximum: f64,
    /// Population spread of the retained document replicates, not a standard
    /// error, posterior uncertainty, human reliability or confidence interval.
    pub population_stddev: f64,
    pub replicates: Vec<SampledSentimentMapReplicate>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SampledSentimentMapAggregate {
    pub schema_version: u32,
    pub reduction_profile: String,
    pub weight_unit: String,
    pub interpretation: String,
    pub calibration: String,
    pub chunk_count: usize,
    pub source_bytes: usize,
    pub source_scalars: usize,
    pub samples_per_axis: usize,
    pub dimensions: Vec<SampledSentimentMapAggregateDimension>,
}

/// All original scalars count exactly once, including whitespace and CRLF.
/// Complete cold task finalizers remain responsible for each native receipt;
/// reduction checks full partition and compatible axis/sample provenance.
pub(super) fn aggregate_with_control(chunks: &[SampledSentimentMapChunk],
    source_span: VerifiedSourceSpan, samples: usize, control: &mut impl DecodeStepControl)
    -> Result<SampledSentimentMapAggregate, SampledSentimentMapError> {
    checkpoint(control)?;
    if !(1..=MAX_CHUNKS).contains(&chunks.len()) || !(1..=MAX_SENTIMENT_SAMPLES).contains(&samples)
        || source_span.byte_start != 0 || source_span.scalar_start != 0
        || !(1..=MAX_SOURCE_BYTES).contains(&source_span.byte_end)
        || source_span.scalar_end == 0 || source_span.scalar_end > source_span.byte_end
        || source_span.byte_end > source_span.scalar_end.checked_mul(4)
            .ok_or(SampledSentimentMapError::Aggregation)? {
        return Err(SampledSentimentMapError::Aggregation);
    }
    let reference = &chunks[0].run;
    validate_run(reference, samples, control)?;
    let mut sums = [[0_i64; MAX_SENTIMENT_SAMPLES]; 4];
    let (mut byte_end, mut scalar_end) = (0_usize, 0_usize);
    for (index, chunk) in chunks.iter().enumerate() {
        checkpoint(control)?;
        let span = chunk.source_span;
        if chunk.chunk_id != index || span.byte_start != byte_end || span.scalar_start != scalar_end
            || span.byte_end <= span.byte_start || span.scalar_end <= span.scalar_start
            || span.byte_end > source_span.byte_end || span.scalar_end > source_span.scalar_end
            || chunk.source_tokens == 0 {
            return Err(SampledSentimentMapError::Aggregation);
        }
        let scalars = span.scalar_end - span.scalar_start;
        let bytes = span.byte_end - span.byte_start;
        if bytes < scalars || bytes > scalars.checked_mul(4).ok_or(SampledSentimentMapError::Aggregation)? {
            return Err(SampledSentimentMapError::Aggregation);
        }
        validate_run(&chunk.run, samples, control)?;
        if chunk.run.execution != reference.execution
            || chunk.run.prompt_template_sha256 != reference.prompt_template_sha256
            || chunk.run.group_width != reference.group_width
            || chunk.run.dimensions.len() != reference.dimensions.len() {
            return Err(SampledSentimentMapError::Aggregation);
        }
        let weight = i64::try_from(scalars).map_err(|_| SampledSentimentMapError::Aggregation)?;
        for (axis_index, dimension) in chunk.run.dimensions.iter().enumerate() {
            checkpoint(control)?;
            if dimension.axis != reference.dimensions[axis_index].axis {
                return Err(SampledSentimentMapError::Aggregation);
            }
            for sample in &dimension.samples {
                checkpoint(control)?;
                let weighted = weight.checked_mul(i64::from(sample.score))
                    .ok_or(SampledSentimentMapError::Aggregation)?;
                let total = &mut sums[axis_index][sample.sample_index];
                *total = total.checked_add(weighted).ok_or(SampledSentimentMapError::Aggregation)?;
            }
        }
        byte_end = span.byte_end;
        scalar_end = span.scalar_end;
    }
    if byte_end != source_span.byte_end || scalar_end != source_span.scalar_end {
        return Err(SampledSentimentMapError::Aggregation);
    }
    let mut dimensions = reserved(reference.dimensions.len())?;
    for (index, dimension) in reference.dimensions.iter().enumerate() {
        checkpoint(control)?;
        dimensions.push(finish_axis(dimension.axis, &sums[index][..samples], source_span.scalar_end, control)?);
    }
    checkpoint(control)?;
    Ok(SampledSentimentMapAggregate {
        schema_version: 1, reduction_profile: SAMPLED_SENTIMENT_MAP_AGGREGATION.to_owned(),
        weight_unit: "original_unicode_scalar_count".to_owned(),
        interpretation: "descriptive_ordinal_paired_chunk_samples_not_full_context_inference".to_owned(),
        calibration: "uncalibrated_not_a_psychological_measurement".to_owned(),
        chunk_count: chunks.len(), source_bytes: source_span.byte_end, source_scalars: source_span.scalar_end,
        samples_per_axis: samples, dimensions,
    })
}

fn validate_run(run: &SampledSentimentRun, samples: usize, control: &mut impl DecodeStepControl)
    -> Result<(), SampledSentimentMapError> {
    if run.schema_version != 1 || run.task_spec != "sentiment-v1"
        || !matches!(run.execution.as_str(), SAMPLED_SENTIMENT_EXECUTION | SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION)
        || run.numerics_profile != STRICT_INT4_PROFILE || !(1..=MAX_MODEL_K).contains(&run.group_width)
        || run.aggregation != SAMPLED_SENTIMENT_AGGREGATION_VERSION
        || run.interpretation != "generated_score_variation_not_calibrated_confidence"
        || run.justification_status != "unverified_model_explanations_from_untrusted_document"
        || run.sampler_version != JSON_SAMPLER_VERSION || run.samples_per_axis != samples
        || run.dimensions.is_empty() || run.dimensions.len() > SentimentAxis::ALL.len()
        || run.dimensions.windows(2).any(|pair| pair[0].axis >= pair[1].axis) {
        return Err(SampledSentimentMapError::Aggregation);
    }
    let mut work = Int4Work::default();
    let (mut tokens, mut masks) = (0_u64, 0_u64);
    for dimension in &run.dimensions {
        checkpoint(control)?;
        validate_dimension(dimension, samples)?;
        for (index, sample) in dimension.samples.iter().enumerate() {
            checkpoint(control)?;
            if sample.sample_index != index || !(-100..=100).contains(&sample.score)
                || sample.justification.len() > 4 * 4096 || sample.justification.trim().is_empty()
                || sample.justification.chars().count() > 4096
                || !(1..=1024).contains(&sample.generated_tokens) || sample.mask_node_visit_charge == 0
                || sample.model_work.forward_positions < sample.generated_tokens as u64
                || sample.model_work.projected_logits != sample.generated_tokens as u64 * NANBEIGE_VOCAB_SIZE as u64 {
                return Err(SampledSentimentMapError::Aggregation);
            }
            work = work.checked_add(sample.model_work).map_err(|_| SampledSentimentMapError::Aggregation)?;
            tokens = tokens.checked_add(sample.generated_tokens as u64).ok_or(SampledSentimentMapError::Aggregation)?;
            masks = masks.checked_add(sample.mask_node_visit_charge).ok_or(SampledSentimentMapError::Aggregation)?;
        }
    }
    if work != run.model_work || tokens != run.generated_tokens || masks != run.mask_node_visit_charge {
        return Err(SampledSentimentMapError::Aggregation);
    }
    Ok(())
}

// Recompute the existing per-chunk descriptive moments from integer scores,
// without cloning any model explanation or trusting nonfinite wire metrics.
fn validate_dimension(dimension: &SampledSentimentDimension, samples: usize)
    -> Result<(), SampledSentimentMapError> {
    let (low, high) = dimension.axis.anchors();
    if dimension.samples.len() != samples || dimension.low_anchor != low || dimension.high_anchor != high {
        return Err(SampledSentimentMapError::Aggregation);
    }
    let mut sorted = [0_i16; MAX_SENTIMENT_SAMPLES];
    for (slot, sample) in sorted.iter_mut().zip(&dimension.samples) { *slot = sample.score; }
    let sorted = &mut sorted[..samples];
    sorted.sort_unstable();
    let mean = sorted.iter().map(|&score| i64::from(score)).sum::<i64>() as f64 / samples as f64;
    let middle = samples / 2;
    let median = if samples % 2 == 0 {
        (f64::from(sorted[middle - 1]) + f64::from(sorted[middle])) / 2.0
    } else { f64::from(sorted[middle]) };
    let variance = sorted.iter().map(|&score| (f64::from(score) - mean).powi(2)).sum::<f64>() / samples as f64;
    if dimension.mean != mean || dimension.median != median
        || dimension.minimum != sorted[0] || dimension.maximum != sorted[samples - 1]
        || dimension.population_stddev != variance.sqrt() {
        return Err(SampledSentimentMapError::Aggregation);
    }
    Ok(())
}

fn finish_axis(axis: SentimentAxis, sums: &[i64], scalar_weight: usize,
    control: &mut impl DecodeStepControl) -> Result<SampledSentimentMapAggregateDimension, SampledSentimentMapError> {
    let count = sums.len();
    let denominator = i64::try_from(scalar_weight).map_err(|_| SampledSentimentMapError::Aggregation)?;
    let mut total = 0_i64;
    let mut replicates = reserved(count)?;
    for (sample_index, &weighted_score_sum) in sums.iter().enumerate() {
        checkpoint(control)?;
        total = total.checked_add(weighted_score_sum).ok_or(SampledSentimentMapError::Aggregation)?;
        replicates.push(SampledSentimentMapReplicate { sample_index, weighted_score_sum,
            score: weighted_score_sum as f64 / denominator as f64 });
    }
    let mut sorted = [0_i64; MAX_SENTIMENT_SAMPLES];
    sorted[..count].copy_from_slice(sums);
    let sorted = &mut sorted[..count]; sorted.sort_unstable();
    let middle = count / 2;
    let mean_denominator = denominator.checked_mul(count as i64).ok_or(SampledSentimentMapError::Aggregation)?;
    let mean = total as f64 / mean_denominator as f64;
    let median = if count % 2 == 0 {
        sorted[middle - 1].checked_add(sorted[middle]).ok_or(SampledSentimentMapError::Aggregation)? as f64
            / denominator.checked_mul(2).ok_or(SampledSentimentMapError::Aggregation)? as f64
    } else { sorted[middle] as f64 / denominator as f64 };
    // Canonical value order makes descriptive moments insensitive to an
    // ordinal permutation while the retained replicates keep their addresses.
    let variance = sorted.iter().map(|&sum| (sum as f64 / denominator as f64 - mean).powi(2))
        .sum::<f64>() / count as f64;
    let (low, high) = axis.anchors();
    Ok(SampledSentimentMapAggregateDimension { axis, low_anchor: low.to_owned(), high_anchor: high.to_owned(),
        mean, median, minimum: sorted[0] as f64 / denominator as f64,
        maximum: sorted[count - 1] as f64 / denominator as f64,
        population_stddev: variance.sqrt(), replicates })
}

fn reserved<T>(count: usize) -> Result<Vec<T>, SampledSentimentMapError> {
    let mut result = Vec::new();
    result.try_reserve_exact(count).map_err(|_| SampledSentimentMapError::Allocation)?;
    Ok(result)
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), SampledSentimentMapError> {
    match control.prefill_checkpoint(0) {
        Some(cause) => Err(SampledSentimentMapError::Cancelled(cause)), None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    // Pure arithmetic and boundary fixtures; these fabricate no native/model
    // proof, addressed-draw independence or full-document quality evidence.
    use super::*;
    use crate::{execution_identity::Sha256Digest, native_engine::decode::DecodeCancellationKind,
        tasks::sentiment::sampled::{SampledSentimentSample, aggregate}};

    struct Control { calls: usize, stop: Option<(usize, DecodeCancellationKind)> }
    impl Control { fn uninterrupted() -> Self { Self { calls: 0, stop: None } } }
    impl DecodeStepControl for Control {
        fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
            self.calls += 1;
            self.stop.and_then(|(at, cause)| (self.calls >= at).then_some(cause))
        }
    }
    fn run(axes: &[(SentimentAxis, &[i16])]) -> SampledSentimentRun {
        let mut dimensions = Vec::new();
        let mut work = Int4Work::default();
        let (mut tokens, mut masks) = (0, 0);
        for &(axis, scores) in axes {
            let samples = scores.iter().enumerate().map(|(sample_index, &score)| {
                let sample = SampledSentimentSample { sample_index, score, justification: "fixture only".to_owned(),
                    generated_tokens: 2, model_work: Int4Work::for_sequence(128, 0, 20, 2 * NANBEIGE_VOCAB_SIZE).unwrap(),
                    mask_node_visit_charge: 6 };
                work = work.checked_add(sample.model_work).unwrap(); tokens += 2; masks += 6; sample
            }).collect();
            dimensions.push(aggregate(axis, samples).unwrap());
        }
        SampledSentimentRun { schema_version: 1, execution: SAMPLED_SENTIMENT_EXECUTION.to_owned(),
            task_spec: "sentiment-v1".to_owned(), prompt_template_sha256: Sha256Digest::of_bytes(b"fixed fixture template"),
            numerics_profile: STRICT_INT4_PROFILE.to_owned(), group_width: 128,
            aggregation: SAMPLED_SENTIMENT_AGGREGATION_VERSION.to_owned(),
            interpretation: "generated_score_variation_not_calibrated_confidence".to_owned(),
            justification_status: "unverified_model_explanations_from_untrusted_document".to_owned(),
            sampler_version: JSON_SAMPLER_VERSION.to_owned(), samples_per_axis: axes[0].1.len(),
            dimensions, model_work: work, generated_tokens: tokens, mask_node_visit_charge: masks }
    }
    fn chunk(id: usize, bytes: (usize, usize), scalars: (usize, usize), scores: &[i16]) -> SampledSentimentMapChunk {
        SampledSentimentMapChunk { chunk_id: id, source_span: VerifiedSourceSpan {
            byte_start: bytes.0, byte_end: bytes.1, scalar_start: scalars.0, scalar_end: scalars.1 },
            source_tokens: bytes.1 - bytes.0, run: run(&[(SentimentAxis::Valence, scores)]) }
    }
    fn span(bytes: usize, scalars: usize) -> VerifiedSourceSpan {
        VerifiedSourceSpan { byte_start: 0, byte_end: bytes, scalar_start: 0, scalar_end: scalars }
    }
    fn reduce(chunks: &[SampledSentimentMapChunk], source: VerifiedSourceSpan, samples: usize)
        -> Result<SampledSentimentMapAggregate, SampledSentimentMapError> {
        aggregate_with_control(chunks, source, samples, &mut Control::uninterrupted())
    }

    #[test]
    fn exact_scalar_weights_keep_ordinal_replicates_and_do_not_pool_chunk_variation() {
        // A two-byte scalar followed by three ASCII scalars has weights 1:3,
        // not UTF-8 byte weights 2:3 or equal chunk weights 1:1.
        let chunks = [chunk(0, (0, 2), (0, 1), &[-100, 100]), chunk(1, (2, 5), (1, 4), &[100, -100])];
        let result = reduce(&chunks, span(5, 4), 2).unwrap();
        let axis = &result.dimensions[0];
        assert_eq!(axis.replicates, vec![SampledSentimentMapReplicate { sample_index: 0, weighted_score_sum: 200, score: 50.0 },
            SampledSentimentMapReplicate { sample_index: 1, weighted_score_sum: -200, score: -50.0 }]);
        assert_eq!((axis.mean, axis.median, axis.minimum, axis.maximum, axis.population_stddev), (0.0, 0.0, -50.0, 50.0, 50.0));
        assert_eq!((result.source_bytes, result.source_scalars, result.chunk_count), (5, 4, 2));
        let equal = [chunk(0, (0, 1), (0, 1), &[-100, 100]), chunk(1, (1, 2), (1, 2), &[100, -100])];
        let axis = &reduce(&equal, span(2, 2), 2).unwrap().dimensions[0];
        assert_eq!(axis.population_stddev, 0.0);
        assert!(axis.replicates.iter().all(|replicate| replicate.score == 0.0));
    }

    #[test]
    fn odd_even_and_single_replicate_moments_follow_the_fixed_rational_recipe() {
        for scores in [&[-100, 0, 100][..], &[0, 1, 2, 3][..], &[37][..]] {
            let chunks = [chunk(0, (0, 3), (0, 3), scores)];
            let result = reduce(&chunks, span(3, 3), scores.len()).unwrap();
            let expected = &chunks[0].run.dimensions[0]; let actual = &result.dimensions[0];
            assert_eq!((actual.mean, actual.median, actual.minimum, actual.maximum, actual.population_stddev),
                (expected.mean, expected.median, f64::from(expected.minimum), f64::from(expected.maximum), expected.population_stddev));
        }
        let chunks = [chunk(0, (0, 1), (0, 1), &[1, 2]), chunk(1, (1, 3), (1, 3), &[0, 0])];
        let result = reduce(&chunks, span(3, 3), 2).unwrap();
        assert_eq!(result.dimensions[0].mean, 0.5);
        assert_eq!(result.dimensions[0].median, 0.5);
    }

    #[test]
    fn complete_cover_rejects_missing_tail_gaps_overlap_reordering_and_impossible_unicode() {
        for mutation in 0..9 {
            let mut chunks = vec![chunk(0, (0, 2), (0, 2), &[0, 1]), chunk(1, (2, 4), (2, 4), &[2, 3])];
            match mutation {
                0 => { chunks.pop(); }, 1 => chunks[1].source_span.byte_start += 1,
                2 => chunks[1].source_span.byte_start -= 1, 3 => chunks[1].source_span.scalar_start -= 1,
                4 => chunks[1].chunk_id = 0, 5 => chunks.swap(0, 1),
                6 => chunks[1].source_tokens = 0, 7 => chunks[0].source_span.scalar_end = 3,
                _ => chunks[1].source_span.byte_end = 5,
            }
            assert!(reduce(&chunks, span(4, 4), 2).is_err());
        }
        assert!(reduce(&[], span(4, 4), 2).is_err());
        assert!(reduce(&[chunk(0, (0, 5), (0, 1), &[0])], span(5, 1), 1).is_err());
    }

    #[test]
    fn every_sample_and_axis_must_share_the_exact_completed_semantics() {
        for mutation in 0..19 {
            let mut chunks = [chunk(0, (0, 1), (0, 1), &[0, 1]), chunk(1, (1, 2), (1, 2), &[2, 3])];
            let run = &mut chunks[1].run;
            match mutation {
                0 => run.schema_version += 1, 1 => run.execution = SAMPLED_SENTIMENT_REUSED_PREFILL_EXECUTION.to_owned(),
                2 => run.prompt_template_sha256 = Sha256Digest::of_bytes(b"changed"), 3 => run.group_width = 32,
                4 => run.aggregation.push('x'), 5 => run.sampler_version.push('x'),
                6 => run.samples_per_axis = 1, 7 => { run.dimensions[0].samples.pop(); },
                8 => run.dimensions[0].samples[1].sample_index = 0, 9 => run.dimensions[0].samples[0].score = 101,
                10 => run.dimensions[0].mean = f64::NAN, 11 => run.dimensions[0].population_stddev = f64::INFINITY,
                12 => run.dimensions[0].low_anchor.push('x'), 13 => run.dimensions[0].samples[0].justification.clear(),
                14 => run.generated_tokens += 1, 15 => run.mask_node_visit_charge += 1,
                16 => run.model_work.forward_positions += 1, 17 => run.numerics_profile.push('x'),
                _ => run.justification_status.clear(),
            }
            assert!(reduce(&chunks, span(2, 2), 2).is_err());
        }
        let mut chunk = chunk(0, (0, 2), (0, 2), &[0, 1]);
        chunk.run = run(&[(SentimentAxis::Arousal, &[1, 2]), (SentimentAxis::Valence, &[2, 3])]);
        assert!(reduce(&[chunk], span(2, 2), 2).is_err());
    }

    #[test]
    fn cancellation_is_typed_at_every_reduction_checkpoint_and_no_interval_is_reported() {
        let chunks = [chunk(0, (0, 1), (0, 1), &[0, 1]), chunk(1, (1, 2), (1, 2), &[2, 3])];
        let mut complete = Control::uninterrupted();
        let result = aggregate_with_control(&chunks, span(2, 2), 2, &mut complete).unwrap();
        use DecodeCancellationKind::*;
        for cause in [User, Timeout, Deadline, PollQuota, CostBudget, FailFast, RaceLost,
            ParentCancelled, ResourceUnavailable, Shutdown, LinkedExit] {
            for at in 1..=complete.calls {
                let error = aggregate_with_control(&chunks, span(2, 2), 2,
                    &mut Control { calls: 0, stop: Some((at, cause)) }).err().unwrap();
                assert!(matches!(error, SampledSentimentMapError::Cancelled(found) if found == cause));
            }
        }
        let value = serde_json::to_value(result).unwrap();
        assert_eq!(value["reduction_profile"], SAMPLED_SENTIMENT_MAP_AGGREGATION);
        assert!(value["dimensions"][0].get("confidence_interval").is_none());
        assert!(value.get("seed").is_none());
        assert!(value.get("document_sha256").is_none());
    }
}
