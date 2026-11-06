//! Arithmetic and authority goldens; these fixtures are not native-model evidence.
use super::*;
use crate::{native_engine::lmhead::scoring::{CandidateScore, CandidateScores, ScoringWork},
    tasks::classify::{MultiLabelLabelResult, MultiLabelResult}};

struct Continue;
impl DecodeStepControl for Continue { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { None } }
fn classification(ids: &[&str], weights: &[f64], policy: ClassificationPolicy) -> ClassificationResult {
    let candidates: Vec<_> = ids.iter().zip(weights).map(|(&id, &weight)| CandidateScore {
        id: id.to_owned(), scored_tokens: 2, sequence_score: if weight == 0.0 { -1000.0 } else { weight.ln() },
        candidate_weight: weight }).collect();
    let mut order: Vec<_> = (0..ids.len()).collect();
    order.sort_unstable_by(|&a, &b| candidates[b].sequence_score.total_cmp(&candidates[a].sequence_score)
        .then_with(|| ids[a].cmp(ids[b])));
    let best = weights[order[0]]; let margin = best - weights[order[1]];
    let accepted = passes(best, margin, policy);
    ClassificationResult { schema_version: 1, decision: if accepted { ClassificationDecision::Classified } else { ClassificationDecision::Abstained },
        selected_id: accepted.then(|| ids[order[0]].to_owned()), ranking: order.into_iter().map(|i| ids[i].to_owned()).collect(),
        best_candidate_weight: best, candidate_weight_margin: margin, calibration: ClassificationCalibration::Uncalibrated,
        policy, scores: CandidateScores { score_space: ScoreSpace::FullVocabSequenceLogprob,
            normalization_scope: "full_vocabulary_per_position; candidate_weights_over_completed_candidates".to_owned(),
            length_rule: "sum_logprobs_including_eos".to_owned(), eos_rule: "append_and_score_exactly_one_eos".to_owned(),
            eos_token_id: 166101, full_vocab_denominators_computed: true, candidates,
            work: ScoringWork { prefix_evaluations: ids.len() + 1, scored_edges: ids.len() * 2, projected_logits: 1 } } }
}
fn exclusive(weights: &[f64], policy: ClassificationPolicy) -> ClassificationTaskResult {
    let names: Vec<_> = (0..weights.len()).map(|i| format!("label-{i:02}")).collect();
    ClassificationTaskResult::Exclusive(classification(&names.iter().map(String::as_str).collect::<Vec<_>>(), weights, policy))
}
fn multi(positive: &[f64], policy: ClassificationPolicy) -> ClassificationTaskResult {
    let mut result = MultiLabelResult { schema_version: 1,
        decision_scope: "independent_binary_heads_not_a_distribution_over_labels".to_owned(),
        calibration: ClassificationCalibration::Uncalibrated, selected_ids: Vec::new(), excluded_ids: Vec::new(),
        abstained_ids: Vec::new(), labels: Vec::new() };
    for (index, &yes) in positive.iter().enumerate() {
        let label_id = format!("topic-{index}");
        let classification = classification(&["no", "yes"], &[1.0 - yes, yes], policy);
        let decision = match classification.selected_id.as_deref() {
            Some("yes") => { result.selected_ids.push(label_id.clone()); MultiLabelDecision::Included },
            Some("no") => { result.excluded_ids.push(label_id.clone()); MultiLabelDecision::Excluded },
            _ => { result.abstained_ids.push(label_id.clone()); MultiLabelDecision::Abstained },
        };
        result.labels.push(MultiLabelLabelResult { label_id, decision, positive_candidate_weight: yes, classification });
    }
    ClassificationTaskResult::MultiLabel(result)
}
fn aggregate(results: &[ClassificationTaskResult], lengths: &[(usize, usize)], mode: ClassificationMode,
    classification: ClassificationPolicy, coverage: u32) -> Result<ClassificationAggregate, ClassificationAggregationError> {
    let (mut bytes, mut scalars) = (0, 0);
    let views: Vec<_> = results.iter().zip(lengths).enumerate().map(|(chunk_id, (result, &(b, s)))| {
        let span = VerifiedSourceSpan { byte_start: bytes, byte_end: bytes + b, scalar_start: scalars, scalar_end: scalars + s };
        bytes += b; scalars += s; ClassificationChunkView { chunk_id, span, result }
    }).collect();
    aggregate_with_control(views, results.len(), bytes, scalars, mode, classification,
        ClassificationAggregationPolicy { minimum_classified_coverage_ppm: coverage }, &mut Continue)
}

#[test]
fn unicode_scalar_weighting_differs_from_byte_weighting_and_keeps_every_candidate() {
    let policy = ClassificationPolicy::default();
    let result = aggregate(&[exclusive(&[0.9, 0.1], policy), exclusive(&[0.1, 0.9], policy)],
        &[(2, 1), (3, 3)], ClassificationMode::Exclusive, policy, SCALE).unwrap();
    let ClassificationAggregateResult::Exclusive(mixture) = result.result else { panic!("exclusive required") };
    assert!(near(mixture.candidate_weights[0].weight, 0.3, 1));
    assert!(near(mixture.candidate_weights[1].weight, 0.7, 1));
    assert_eq!(mixture.selected_id.as_deref(), Some("label-01"));
    assert_eq!(mixture.classified_scalars, 4); assert_eq!(result.source_bytes, 5);
    assert_eq!(result.calibration, ClassificationCalibration::Uncalibrated);
}
#[test]
fn multiclass_uses_complete_candidate_mixtures_and_deterministic_weight_ranking() {
    let policy = ClassificationPolicy::default();
    let result = aggregate(&[exclusive(&[0.6, 0.3, 0.1], policy), exclusive(&[0.1, 0.1, 0.8], policy)],
        &[(1, 1), (3, 3)], ClassificationMode::Exclusive, policy, SCALE).unwrap();
    let ClassificationAggregateResult::Exclusive(mixture) = result.result else { panic!("exclusive required") };
    assert_eq!(mixture.ranking, ["label-02", "label-00", "label-01"]);
    for (got, expected) in mixture.candidate_weights.iter().zip([0.225, 0.15, 0.625]) { assert!(near(got.weight, expected, 4)); }
}
#[test]
fn independent_labels_can_all_be_positive_without_cross_label_normalization() {
    let policy = ClassificationPolicy::default();
    let result = aggregate(&[multi(&[0.9, 0.95], policy), multi(&[0.9, 0.95], policy)],
        &[(1, 1), (1, 1)], ClassificationMode::MultiLabel, policy, SCALE).unwrap();
    let ClassificationAggregateResult::MultiLabel(mixture) = result.result else { panic!("multi required") };
    assert_eq!(mixture.selected_ids, ["topic-0", "topic-1"]);
    assert!(mixture.labels.iter().map(|label| label.positive_candidate_weight).sum::<f64>() > 1.8);
    for label in mixture.labels { assert_eq!(label.mixture.candidate_weights.len(), 2); }
}
#[test]
fn abstaining_chunks_remain_in_weights_and_coverage_is_per_label() {
    let policy = ClassificationPolicy { minimum_candidate_weight_ppm: 600_000, minimum_margin_ppm: 0 };
    let results = [multi(&[0.9, 0.5], policy), multi(&[0.9, 0.9], policy)];
    for coverage in [SCALE, 500_000] {
        let result = aggregate(&results, &[(1, 1), (1, 1)], ClassificationMode::MultiLabel, policy, coverage).unwrap();
        let ClassificationAggregateResult::MultiLabel(mixture) = result.result else { panic!("multi required") };
        assert_eq!(mixture.labels[0].decision, MultiLabelDecision::Included);
        assert_eq!(mixture.labels[1].mixture.classified_coverage, 0.5);
        assert!(near(mixture.labels[1].positive_candidate_weight, 0.7, 1));
        assert_eq!(mixture.labels[1].decision, if coverage == SCALE { MultiLabelDecision::Abstained } else { MultiLabelDecision::Included });
    }
}
#[test]
fn unanimous_and_uniform_exact_policy_endpoints_do_not_newly_abstain() {
    for (chunks, weights, policy) in [(7, vec![1.0, 0.0], ClassificationPolicy {
            minimum_candidate_weight_ppm: SCALE, minimum_margin_ppm: SCALE }),
        (10, vec![0.1; 10], ClassificationPolicy { minimum_candidate_weight_ppm: 100_000, minimum_margin_ppm: 0 })] {
        let rows = vec![exclusive(&weights, policy); chunks];
        let result = aggregate(&rows, &vec![(1, 1); chunks], ClassificationMode::Exclusive, policy, SCALE).unwrap();
        let ClassificationAggregateResult::Exclusive(mixture) = result.result else { panic!("exclusive required") };
        assert_eq!(mixture.decision, ClassificationDecision::Classified);
        assert_eq!(mixture.best_candidate_weight, weights[0]);
    }
}
#[test]
fn gaps_reordering_missing_chunks_and_out_of_range_spans_are_rejected() {
    let policy = ClassificationPolicy::default(); let result = exclusive(&[0.8, 0.2], policy);
    for fault in 0..5 {
        let mut span = VerifiedSourceSpan { byte_start: 0, byte_end: 1, scalar_start: 0, scalar_end: 1 };
        match fault { 0 => span.byte_start = 1, 1 => span.scalar_start = 1, 2 => span.byte_end = 2, _ => {} }
        assert!(aggregate_with_control([ClassificationChunkView { chunk_id: usize::from(fault == 3), span, result: &result }],
            if fault == 4 { 2 } else { 1 }, 1, 1, ClassificationMode::Exclusive, policy,
            ClassificationAggregationPolicy::default(), &mut Continue).is_err());
    }
}
#[test]
fn changed_candidate_sets_policies_and_forged_score_receipts_refuse() {
    let policy = ClassificationPolicy::default(); let first = exclusive(&[0.8, 0.2], policy);
    for field in 0..8 {
        let mut changed = first.clone(); let ClassificationTaskResult::Exclusive(result) = &mut changed else { unreachable!() };
        match field {
            0 => result.scores.candidates[1].id.push('x'),
            1 => result.policy.minimum_candidate_weight_ppm += 1,
            2 => result.scores.full_vocab_denominators_computed = false,
            3 => result.scores.candidates[0].candidate_weight = f64::NAN,
            4 => { result.scores.candidates.pop(); },
            5 => result.ranking.reverse(), 6 => result.selected_id = None,
            _ => result.scores.eos_token_id += 1,
        }
        assert!(aggregate(&[first.clone(), changed], &[(1, 1), (1, 1)], ClassificationMode::Exclusive, policy, SCALE).is_err());
    }
}
#[test]
fn cancellation_never_returns_a_partial_aggregate() {
    struct Stop(DecodeCancellationKind);
    impl DecodeStepControl for Stop { fn checkpoint(&mut self, _: usize) -> Option<DecodeCancellationKind> { Some(self.0) } }
    let policy = ClassificationPolicy::default(); let result = exclusive(&[0.8, 0.2], policy);
    for cause in [DecodeCancellationKind::User, DecodeCancellationKind::Timeout, DecodeCancellationKind::Deadline,
        DecodeCancellationKind::PollQuota, DecodeCancellationKind::CostBudget, DecodeCancellationKind::FailFast,
        DecodeCancellationKind::RaceLost, DecodeCancellationKind::ParentCancelled, DecodeCancellationKind::ResourceUnavailable,
        DecodeCancellationKind::Shutdown, DecodeCancellationKind::LinkedExit] {
        let view = ClassificationChunkView { chunk_id: 0, span: VerifiedSourceSpan {
            byte_start: 0, byte_end: 1, scalar_start: 0, scalar_end: 1 }, result: &result };
        assert_eq!(aggregate_with_control([view], 1, 1, 1, ClassificationMode::Exclusive, policy,
            ClassificationAggregationPolicy::default(), &mut Stop(cause)), Err(ClassificationAggregationError::Cancelled(cause)));
    }
}
