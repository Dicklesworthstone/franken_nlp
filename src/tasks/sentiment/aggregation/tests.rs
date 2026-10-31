//! Synthetic arithmetic fixtures, not model execution or quality evidence.
use super::*;
use crate::{
    native_engine::lmhead::scoring::{CandidateScore, CandidateScores, ScoringWork},
    tasks::sentiment::{SentimentAnchor, SentimentWork},
};

struct Continue;
impl DecodeStepControl for Continue {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None }
}
struct StopAfter(usize);
impl DecodeStepControl for StopAfter {
    fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> {
        if self.0 == 0 { Some(DecodeCancellationKind::Deadline) } else { self.0 -= 1; None }
    }
}

fn permissive() -> SentimentPolicy {
    SentimentPolicy { minimum_peak_weight_ppm: 0, maximum_normalized_entropy_ppm: PPM }
}
fn fixture(weights: [f64; 3], policy: SentimentPolicy) -> SentimentResult {
    let anchors: Vec<_> = [-1000, 0, 1000].into_iter().enumerate().map(|(index, value_milli)|
        SentimentAnchor { candidate_id: format!("bin-{index}"), value_milli }).collect();
    let mean: f64 = anchors.iter().zip(weights).map(|(anchor, weight)| weight * f64::from(anchor.value_milli) / 1000.0).sum();
    let variance: f64 = anchors.iter().zip(weights).map(|(anchor, weight)| {
        let delta = f64::from(anchor.value_milli) / 1000.0 - mean;
        weight * delta * delta
    }).sum();
    let entropy = -weights.iter().filter(|&&weight| weight > 0.0)
        .map(|&weight| weight * weight.ln()).sum::<f64>() / 3.0_f64.ln();
    let peak = weights.into_iter().fold(0.0_f64, f64::max);
    let modal = weights.iter().position(|&weight| weight == peak).unwrap();
    let accepted = passes(policy, peak, entropy);
    SentimentResult {
        schema_version: 1, task_spec: "sentiment-v1".to_owned(),
        interpretation: "independent_conditional_candidate_weight_projections".to_owned(),
        calibration: "uncalibrated_not_a_psychological_measurement".to_owned(), policy,
        dimensions: vec![SentimentDimension {
            axis: SentimentAxis::Valence, low_anchor: "negative".to_owned(), high_anchor: "positive".to_owned(),
            decision: if accepted { SentimentDecision::Estimated } else { SentimentDecision::Abstained },
            estimate: accepted.then_some(mean), distribution_mean: mean, distribution_stddev: variance.sqrt(),
            normalized_entropy: entropy, modal_value_milli: anchors[modal].value_milli,
            scores: CandidateScores {
                score_space: ScoreSpace::FullVocabSequenceLogprob,
                normalization_scope: "complete_candidate_set".to_owned(), length_rule: "none".to_owned(),
                eos_rule: "append_and_score_exactly_one_eos".to_owned(), eos_token_id: 166101,
                full_vocab_denominators_computed: true,
                candidates: anchors.iter().zip(weights).map(|(anchor, weight)| CandidateScore {
                    id: anchor.candidate_id.clone(), scored_tokens: 2,
                    sequence_score: if weight > 0.0 { weight.ln() } else { -1000.0 }, candidate_weight: weight,
                }).collect(),
                work: ScoringWork { prefix_evaluations: 4, scored_edges: 6, projected_logits: 664576 },
            }, anchors,
        }],
        work: SentimentWork { dimensions: 1, candidates: 3, prefix_evaluations: 4,
            scored_edges: 6, projected_logits: 664576 },
    }
}
fn view<'a>(id: usize, bytes: std::ops::Range<usize>, scalars: std::ops::Range<usize>, result: &'a SentimentResult)
    -> SentimentChunkView<'a> {
    SentimentChunkView { chunk_id: id, span: VerifiedSourceSpan {
        byte_start: bytes.start, byte_end: bytes.end, scalar_start: scalars.start, scalar_end: scalars.end,
    }, result }
}
fn run(results: &[SentimentResult], lengths: &[usize], policy: SentimentAggregationPolicy) -> SentimentAggregate {
    let mut start = 0;
    let views: Vec<_> = results.iter().zip(lengths).enumerate().map(|(id, (result, &length))| {
        let range = start..start + length;
        start += length;
        view(id, range.clone(), range, result)
    }).collect();
    aggregate_with_control(views, results.len(), start, start, policy, &mut Continue).unwrap()
}
fn assert_near(actual: f64, expected: f64) { assert!(near(actual, expected), "{actual} differs from {expected}"); }

#[test]
fn opposing_decisive_chunks_preserve_disagreement_instead_of_confident_neutrality() {
    let result = run(&[fixture([1.0, 0.0, 0.0], permissive()), fixture([0.0, 0.0, 1.0], permissive())],
        &[1, 1], SentimentAggregationPolicy::default());
    let axis = &result.dimensions[0];
    assert_near(axis.distribution_mean, 0.0);
    assert_near(axis.within_chunk_variance, 0.0);
    assert_near(axis.between_chunk_variance, 1.0);
    assert_near(axis.distribution_stddev, 1.0);
    assert_eq!(axis.modal_value_milli, -1000);
    assert_eq!(result.weight_unit, "original_unicode_scalar_count");
    assert!(result.interpretation.contains("not_full_document_inference"));
}

#[test]
fn diffuse_chunks_and_consistently_neutral_chunks_have_distinct_dispersion() {
    let diffuse = run(&[fixture([0.5, 0.0, 0.5], permissive())], &[2], SentimentAggregationPolicy::default());
    let neutral = run(&[fixture([0.0, 1.0, 0.0], permissive())], &[2], SentimentAggregationPolicy::default());
    assert_near(diffuse.dimensions[0].within_chunk_variance, 1.0);
    assert_near(diffuse.dimensions[0].between_chunk_variance, 0.0);
    assert_near(neutral.dimensions[0].distribution_stddev, 0.0);
}

#[test]
fn unicode_scalars_do_not_overweight_multibyte_source_regions() {
    let negative = fixture([1.0, 0.0, 0.0], permissive());
    let positive = fixture([0.0, 0.0, 1.0], permissive());
    // Source could be "éa": two bytes and one byte, but one scalar each.
    let result = aggregate_with_control([view(0, 0..2, 0..1, &negative), view(1, 2..3, 1..2, &positive)],
        2, 3, 2, SentimentAggregationPolicy::default(), &mut Continue).unwrap();
    assert_near(result.dimensions[0].distribution_mean, 0.0);
    assert_eq!(result.source_bytes, 3);
    assert_eq!(result.source_scalars, 2);
}

#[test]
fn unequal_chunks_use_exact_owned_scalar_weights() {
    let result = run(&[fixture([1.0, 0.0, 0.0], permissive()), fixture([0.0, 0.0, 1.0], permissive())],
        &[1, 3], SentimentAggregationPolicy::default());
    let axis = &result.dimensions[0];
    assert_near(axis.distribution_mean, 0.5);
    assert_near(axis.between_chunk_variance, 0.75);
    assert_near(axis.mixture_weights[0].weight, 0.25);
    assert_near(axis.mixture_weights[2].weight, 0.75);
}

#[test]
fn total_variance_decomposes_for_asymmetric_nontrivial_distributions() {
    let result = run(&[fixture([0.6, 0.3, 0.1], permissive()), fixture([0.2, 0.3, 0.5], permissive()),
        fixture([0.0, 0.9, 0.1], permissive())], &[17, 2, 11], SentimentAggregationPolicy::default());
    let axis = &result.dimensions[0];
    let variance = axis.mixture_weights.iter().map(|bin| {
        let delta = f64::from(bin.value_milli) / 1000.0 - axis.distribution_mean;
        bin.weight * delta * delta
    }).sum::<f64>();
    assert!(axis.within_chunk_variance > 0.0 && axis.between_chunk_variance > 0.0);
    assert_near(axis.within_chunk_variance + axis.between_chunk_variance, variance);
}

#[test]
fn default_withholds_partial_coverage_but_keeps_abstained_distribution() {
    let policy = SentimentPolicy { minimum_peak_weight_ppm: 600_000, ..permissive() };
    let results = [fixture([0.0, 0.0, 1.0], policy), fixture([1.0 / 3.0; 3], policy)];
    let strict = run(&results, &[1, 1], SentimentAggregationPolicy::default());
    let permitted = run(&results, &[1, 1], SentimentAggregationPolicy { minimum_estimated_coverage_ppm: 500_000 });
    let axis = &strict.dimensions[0];
    assert_eq!(axis.estimate, None);
    assert_eq!(axis.estimated_scalars, 1);
    assert_eq!(axis.abstained_scalars, 1);
    assert!(!axis.coverage_passed && axis.mixture_policy_passed);
    assert_near(axis.distribution_mean, 0.5);
    assert_near(permitted.dimensions[0].estimate.unwrap(), 0.5);
    assert_eq!(axis.mixture_weights, permitted.dimensions[0].mixture_weights);
}

#[test]
fn coverage_gate_uses_integer_comparison_at_fractional_ppm_boundary() {
    let policy = SentimentPolicy { minimum_peak_weight_ppm: 600_000, ..permissive() };
    let results = [fixture([0.0, 0.0, 1.0], policy), fixture([1.0 / 3.0; 3], policy)];
    let accepted = run(&results, &[2, 1], SentimentAggregationPolicy { minimum_estimated_coverage_ppm: 666_666 });
    let refused = run(&results, &[2, 1], SentimentAggregationPolicy { minimum_estimated_coverage_ppm: 666_667 });
    assert!(accepted.dimensions[0].coverage_passed);
    assert!(!refused.dimensions[0].coverage_passed);
}

#[test]
fn complete_chunk_acceptance_does_not_bypass_mixture_policy() {
    let policy = SentimentPolicy { minimum_peak_weight_ppm: 750_000, ..permissive() };
    let result = run(&[fixture([1.0, 0.0, 0.0], policy), fixture([0.0, 0.0, 1.0], policy)],
        &[1, 1], SentimentAggregationPolicy::default());
    let axis = &result.dimensions[0];
    assert!(axis.coverage_passed);
    assert!(!axis.mixture_policy_passed);
    assert_eq!(axis.decision, SentimentDecision::Abstained);
    assert_eq!(axis.estimate, None);
}

#[test]
fn rejects_missing_duplicate_reordered_gapped_and_overlapping_chunks() {
    let result = fixture([0.0, 1.0, 0.0], permissive());
    let cases = [
        vec![view(0, 0..1, 0..1, &result)],
        vec![view(0, 0..1, 0..1, &result), view(0, 1..2, 1..2, &result)],
        vec![view(1, 0..1, 0..1, &result), view(0, 1..2, 1..2, &result)],
        vec![view(0, 0..1, 0..1, &result), view(1, 2..3, 1..2, &result)],
        vec![view(0, 0..1, 0..1, &result), view(1, 0..2, 1..2, &result)],
        vec![view(0, 0..1, 0..1, &result), view(1, 1..2, 0..2, &result)],
    ];
    for views in cases {
        assert_eq!(aggregate_with_control(views, 2, 2, 2, SentimentAggregationPolicy::default(), &mut Continue),
            Err(SentimentAggregationError::InvalidCoverage));
    }
}

#[test]
fn rejects_empty_zero_width_extra_and_impossible_scalar_coverage() {
    let result = fixture([0.0, 1.0, 0.0], permissive());
    for (bytes, scalars) in [(0..0, 0..0), (0..1, 0..2), (0..5, 0..1)] {
        assert!(aggregate_with_control([view(0, bytes, scalars, &result)], 1, 5, 2,
            SentimentAggregationPolicy::default(), &mut Continue).is_err());
    }
    assert!(aggregate_with_control(std::iter::empty(), 0, 0, 0,
        SentimentAggregationPolicy::default(), &mut Continue).is_err());
    assert!(aggregate_with_control([view(0, 0..1, 0..1, &result), view(1, 1..2, 1..2, &result)],
        1, 2, 2, SentimentAggregationPolicy::default(), &mut Continue).is_err());
}

#[test]
fn refuses_changed_policy_axis_grid_scoring_eos_and_calibration() {
    let original = fixture([0.2, 0.5, 0.3], permissive());
    for mutation in 0..10 {
        let mut altered = original.clone();
        match mutation {
            0 => altered.policy.minimum_peak_weight_ppm = 1,
            1 => altered.dimensions[0].axis = SentimentAxis::Arousal,
            2 => altered.dimensions[0].anchors[0].value_milli = -999,
            3 => altered.dimensions[0].scores.score_space = ScoreSpace::TrieLocalConditionalProbability,
            4 => altered.dimensions[0].scores.eos_token_id += 1,
            5 => altered.dimensions[0].scores.full_vocab_denominators_computed = false,
            6 => altered.dimensions[0].scores.length_rule = "different".to_owned(),
            7 => altered.dimensions[0].scores.normalization_scope = "different".to_owned(),
            8 => altered.calibration = "unsupported_confidence".to_owned(),
            _ => altered.dimensions[0].scores.eos_rule = "none".to_owned(),
        }
        assert!(aggregate_with_control([view(0, 0..1, 0..1, &original), view(1, 1..2, 1..2, &altered)],
            2, 2, 2, SentimentAggregationPolicy::default(), &mut Continue).is_err());
    }
}

#[test]
fn rechecks_complete_distribution_and_derived_values() {
    for mutation in 0..12 {
        let mut altered = fixture([0.2, 0.5, 0.3], permissive());
        let axis = &mut altered.dimensions[0];
        match mutation {
            0 => axis.scores.candidates[0].candidate_weight = f64::NAN,
            1 => axis.scores.candidates[0].candidate_weight = -0.1,
            2 => axis.scores.candidates[0].candidate_weight = 0.3,
            3 => axis.scores.candidates[0].sequence_score = f64::INFINITY,
            4 => axis.distribution_mean = 0.9,
            5 => axis.distribution_stddev = -1.0,
            6 => axis.normalized_entropy = f64::NAN,
            7 => axis.estimate = None,
            8 => axis.decision = SentimentDecision::Abstained,
            9 => axis.scores.candidates[0].scored_tokens = 1,
            10 => axis.scores.candidates.swap(0, 1),
            _ => axis.scores.candidates[0].id = axis.scores.candidates[1].id.clone(),
        }
        assert!(aggregate_with_control([view(0, 0..1, 0..1, &altered)],
            1, 1, 1, SentimentAggregationPolicy::default(), &mut Continue).is_err());
    }
}

#[test]
fn axes_are_independent_and_canonical() {
    let mut one = fixture([0.0, 0.0, 1.0], permissive());
    let mut arousal = fixture([1.0, 0.0, 0.0], permissive()).dimensions.remove(0);
    arousal.axis = SentimentAxis::Arousal;
    arousal.low_anchor = "calm".to_owned();
    arousal.high_anchor = "activated".to_owned();
    one.dimensions.push(arousal);
    let result = run(&[one.clone()], &[1], SentimentAggregationPolicy::default());
    assert_eq!(result.dimensions[0].estimate, Some(1.0));
    assert_eq!(result.dimensions[1].estimate, Some(-1.0));
    one.dimensions.swap(0, 1);
    assert!(aggregate_with_control([view(0, 0..1, 0..1, &one)],
        1, 1, 1, SentimentAggregationPolicy::default(), &mut Continue).is_err());
}

#[test]
fn identical_canonical_inputs_produce_identical_serialized_aggregate() {
    let results = [fixture([0.1, 0.8, 0.1], permissive()), fixture([0.7, 0.2, 0.1], permissive())];
    let first = run(&results, &[1, 97], SentimentAggregationPolicy::default());
    let second = run(&results, &[1, 97], SentimentAggregationPolicy::default());
    assert_eq!(crate::canonjson::canonical_bytes(&first).unwrap(), crate::canonjson::canonical_bytes(&second).unwrap());
}

#[test]
fn cancellation_before_between_and_after_chunks_retains_attribution() {
    let result = fixture([0.0, 1.0, 0.0], permissive());
    for checkpoints in [0, 1, 2, 3, 4, 5, 6] {
        assert_eq!(aggregate_with_control([view(0, 0..1, 0..1, &result), view(1, 1..2, 1..2, &result)],
            2, 2, 2, SentimentAggregationPolicy::default(), &mut StopAfter(checkpoints)),
            Err(SentimentAggregationError::Cancelled(DecodeCancellationKind::Deadline)));
    }
}

#[test]
fn invalid_coverage_policy_rejects_before_consuming_input() {
    let never = std::iter::from_fn(|| -> Option<SentimentChunkView<'static>> { panic!("must not consume input") });
    assert_eq!(aggregate_with_control(never, 1, 1, 1,
        SentimentAggregationPolicy { minimum_estimated_coverage_ppm: PPM + 1 }, &mut Continue),
        Err(SentimentAggregationError::InvalidPolicy));
}
