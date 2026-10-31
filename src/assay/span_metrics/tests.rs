use super::*;

fn span(kind: usize, start: usize, end: usize) -> Span { Span { kind, byte_start: start, byte_end: end } }
fn anchored(kind: usize, start: usize, end: usize) -> Prediction {
    Prediction { kind, span: Some((start, end)), candidate_spans: 1 }
}
fn ambiguous(kind: usize, candidates: usize) -> Prediction {
    Prediction { kind, span: None, candidate_spans: candidates }
}
fn counts(gold: &[Span], predictions: &[Prediction]) -> SpanCounts {
    SpanCounts::observe_with_checkpoint::<AssayError>(gold, predictions, 2, &mut || Ok(())).unwrap()
}
fn options() -> EvalOptions {
    EvalOptions { minimum_samples: 5,
        bootstrap: BootstrapOptions { resamples: 32, confidence_level: 0.9, seed: 47 } }
}
fn observations(counts: &[SpanCounts]) -> Vec<Observation<'_>> {
    counts.iter().enumerate().map(|(index, counts)| Observation { language: if index % 2 == 0 { "en" } else { "zh" }, counts }).collect()
}
fn measure(rows: &[SpanCounts]) -> SpanMetrics {
    let Measurement::Measured { metrics } = measure_with_checkpoint::<AssayError>(&["location", "person"],
        &observations(rows), options(), Limits::default(), &mut || Ok(())).unwrap() else { panic!("expected measured"); };
    metrics
}

#[test]
fn exact_and_type_relaxed_scoring_share_boundaries_but_not_type_credit() {
    let result = counts(&[span(0, 0, 3), span(1, 4, 9)], &[anchored(1, 0, 3), anchored(1, 4, 9)]);
    assert_eq!(result.totals.exact, Counts { tp: 1, fp: 1, missed: 1 });
    assert_eq!(result.totals.relaxed, Counts { tp: 2, fp: 0, missed: 0 });
    assert_eq!(result.by_type[0], Counts { tp: 0, fp: 0, missed: 1 });
    assert_eq!(result.by_type[1], Counts { tp: 1, fp: 1, missed: 0 });
    assert_eq!(result.totals.exact_documents, 0);
}

#[test]
fn overlaps_and_contained_mentions_are_not_exact_boundary_matches() {
    let result = counts(&[span(0, 0, 6), span(1, 1, 4)], &[anchored(0, 0, 5), anchored(1, 1, 4)]);
    assert_eq!(result.totals.exact, Counts { tp: 1, fp: 1, missed: 1 });
    assert_eq!(result.totals.relaxed, result.totals.exact);
}

#[test]
fn ambiguity_is_one_unmatched_proposal_even_when_every_candidate_is_gold() {
    let result = counts(&[span(1, 0, 3), span(1, 4, 7)], &[ambiguous(1, 2)]);
    assert_eq!(result.totals.exact, Counts { tp: 0, fp: 1, missed: 2 });
    assert_eq!(result.totals.relaxed, result.totals.exact);
    assert_eq!(result.anchoring().ambiguous_candidate_spans, 2);
    assert_eq!(result.anchoring().anchored_fraction, Some(0.0));
}

#[test]
fn duplicate_and_conflicting_predictions_cannot_multiply_recall() {
    let result = counts(&[span(1, 0, 3)],
        &[anchored(0, 0, 3), anchored(1, 0, 3), anchored(1, 0, 3), anchored(1, 0, 3)]);
    assert_eq!(result.totals.exact, Counts { tp: 1, fp: 3, missed: 0 });
    assert_eq!(result.totals.relaxed, result.totals.exact);
    assert_eq!(result.duplicate_anchored, 2);
    assert_eq!(result.totals.exact.scores().recall, Some(1.0));
}

#[test]
fn gold_cannot_assign_two_types_to_the_same_exact_interval() {
    for gold in [vec![span(1, 0, 3), span(1, 0, 3)], vec![span(0, 0, 3), span(1, 0, 3)]] {
        assert_eq!(SpanCounts::observe_with_checkpoint::<AssayError>(&gold, &[], 2, &mut || Ok(())).err(),
            Some(AssayError::Observation));
    }
}

#[test]
fn invalid_geometry_and_false_anchoring_are_refused() {
    for prediction in [Prediction { kind: 0, span: Some((1, 1)), candidate_spans: 1 },
        Prediction { kind: 0, span: Some((0, 1)), candidate_spans: 2 },
        ambiguous(0, 1), anchored(2, 0, 1), ambiguous(0, MAX_OCCURRENCES + 1)] {
        assert_eq!(SpanCounts::observe_with_checkpoint::<AssayError>(&[], &[prediction], 2, &mut || Ok(())).err(),
            Some(AssayError::Observation));
    }
    assert!(SpanCounts::observe_with_checkpoint::<AssayError>(&[span(0, 3, 2)], &[], 2, &mut || Ok(())).is_err());
}

#[test]
fn prediction_order_does_not_change_any_counts() {
    let gold = [span(0, 2, 5), span(1, 9, 12), span(1, 20, 23)];
    let mut predictions = vec![anchored(1, 9, 12), anchored(0, 2, 5), ambiguous(1, 3),
        anchored(1, 9, 12), anchored(0, 20, 23)];
    let first = counts(&gold, &predictions);
    predictions.reverse();
    assert_eq!(counts(&gold, &predictions), first);
}

#[test]
fn micro_scores_do_not_average_document_f1_and_breakdowns_are_consistent() {
    let many_gold: Vec<_> = (0..9).map(|index| span(0, index * 3, index * 3 + 2)).collect();
    let many_predictions: Vec<_> = (0..9).map(|index| anchored(0, index * 3, index * 3 + 2)).collect();
    let mut rows = vec![counts(&many_gold, &many_predictions), counts(&[span(1, 0, 1)], &[])];
    rows.extend([counts(&[], &[]); 3]);
    let metrics = measure(&rows);
    assert_eq!(metrics.exact.support, 10); assert_eq!(metrics.exact.predicted, 9);
    assert_eq!(metrics.exact.precision, Some(1.0)); assert_eq!(metrics.exact.recall, Some(0.9));
    assert_eq!(metrics.exact.f1, Some(18.0 / 19.0));
    assert_eq!(metrics.document_accuracy, 0.8);
    assert_eq!(metrics.by_type[0].exact.true_positive, 9);
    assert_eq!(metrics.by_type[1].exact.false_negative, 1);
    assert_eq!(metrics.by_language.iter().map(|row| row.exact.support).sum::<usize>(), 10);
    assert_eq!(metrics.by_language[0].language, "en"); assert_eq!(metrics.by_language[1].language, "zh");
}

#[test]
fn empty_span_corpus_has_perfect_document_accuracy_but_undefined_span_scores() {
    let metrics = measure(&[counts(&[], &[]); 5]);
    assert_eq!(metrics.document_accuracy, 1.0);
    assert_eq!(metrics.document_accuracy_interval.lower, 1.0);
    assert_eq!(metrics.document_accuracy_interval.upper, 1.0);
    assert_eq!(metrics.exact.precision, None); assert_eq!(metrics.exact.recall, None); assert_eq!(metrics.exact.f1, None);
    assert_eq!(metrics.exact_interval.f1, None);
    assert_eq!(metrics.anchoring.anchored_fraction, None);
}

#[test]
fn undefined_bootstrap_replicates_are_not_silently_discarded() {
    let mut series = Series::new(3).unwrap();
    series.push(Some(0.2)); series.push(None); series.push(Some(0.8));
    assert_eq!(series.finish::<AssayError>(options(), &mut || Ok(())).unwrap(), None);
    let rows = [counts(&[span(0, 0, 1)], &[]); 5];
    let metrics = measure(&rows);
    assert_eq!(metrics.exact.precision, None); assert_eq!(metrics.exact_interval.precision, None);
    assert_eq!(metrics.exact.f1, Some(0.0));
    assert_eq!(metrics.exact_interval.f1.unwrap().upper, 0.0);
}

#[test]
fn identical_models_have_zero_paired_change_even_when_marginal_scores_vary() {
    let good = counts(&[span(0, 0, 1)], &[anchored(0, 0, 1)]);
    let bad = counts(&[span(0, 0, 1)], &[anchored(1, 0, 1)]);
    let rows = [good, bad, good, bad, good]; let rows = observations(&rows);
    let (a, b, paired) = paired_with_checkpoint::<AssayError>(&["location", "person"], &rows, &rows,
        options(), Limits::default(), &mut || Ok(())).unwrap();
    assert_eq!(a, b);
    let paired = paired.unwrap();
    assert_eq!(paired.equal, 5); assert_eq!(paired.mean_exact_error_change, 0.0);
    for interval in [paired.exact_f1_change_interval, paired.type_relaxed_f1_change_interval,
        paired.exact_recall_change_interval, Some(paired.document_accuracy_change_interval)] {
        let interval = interval.unwrap(); assert_eq!(interval.lower, 0.0); assert_eq!(interval.upper, 0.0);
    }
}

#[test]
fn paired_marginals_match_standalone_evaluation_with_the_same_seed() {
    let good = counts(&[span(0, 0, 1)], &[anchored(0, 0, 1)]);
    let bad = counts(&[span(0, 0, 1)], &[]);
    let current = [good, good, bad, bad, good]; let prior = [bad, good, bad, bad, good];
    let (a, b, paired) = paired_with_checkpoint::<AssayError>(&["location", "person"],
        &observations(&current), &observations(&prior), options(), Limits::default(), &mut || Ok(())).unwrap();
    assert_eq!(a, Measurement::Measured { metrics: measure(&current) });
    assert_eq!(b, Measurement::Measured { metrics: measure(&prior) });
    let paired = paired.unwrap(); assert_eq!(paired.candidate_better, 1); assert_eq!(paired.candidate_worse, 0);
    assert_eq!(paired.mean_exact_error_change, -0.2);
}

#[test]
fn insufficient_data_and_two_arm_bootstrap_cost_are_explicit() {
    let counts = [counts(&[span(0, 0, 1)], &[anchored(0, 0, 1)]); 5];
    let rows = observations(&counts);
    let result = measure_with_checkpoint::<AssayError>(&["location", "person"], &rows[..4], options(),
        Limits::default(), &mut || Ok(())).unwrap();
    assert!(matches!(result, Measurement::InsufficientData { rows: 4, minimum_samples: 5, .. }));
    let limits = Limits { max_bootstrap_draws: 5 * options().bootstrap.resamples as u64, ..Limits::default() };
    assert!(measure_with_checkpoint::<AssayError>(&["location", "person"], &rows, options(), limits, &mut || Ok(())).is_ok());
    assert_eq!(paired_with_checkpoint::<AssayError>(&["location", "person"], &rows, &rows, options(), limits,
        &mut || Ok(())).err(), Some(AssayError::InputLimit));
}

#[derive(Debug, PartialEq)]
enum Stop { Cancelled, Assay(AssayError) }
impl From<AssayError> for Stop { fn from(error: AssayError) -> Self { Self::Assay(error) } }
#[test]
fn scoring_and_bootstrap_preserve_typed_cancellation() {
    let gold: Vec<_> = (0..256).map(|index| span(0, index * 3, index * 3 + 2)).collect();
    let mut calls = 0;
    let error = SpanCounts::observe_with_checkpoint::<Stop>(&gold, &[], 2, &mut || {
        calls += 1; if calls == 3 { Err(Stop::Cancelled) } else { Ok(()) }
    }).unwrap_err();
    assert_eq!(error, Stop::Cancelled);
    let rows = [counts(&[span(0, 0, 1)], &[anchored(0, 0, 1)]); 5];
    let mut calls = 0;
    let error = measure_with_checkpoint::<Stop>(&["location", "person"], &observations(&rows), options(),
        Limits::default(), &mut || { calls += 1; if calls == 8 { Err(Stop::Cancelled) } else { Ok(()) } }).unwrap_err();
    assert_eq!(error, Stop::Cancelled);
}

#[test]
fn mismatched_paired_language_and_support_are_refused() {
    let first = [counts(&[span(0, 0, 1)], &[]); 5];
    let second = [counts(&[], &[]); 5];
    assert_eq!(paired_with_checkpoint::<AssayError>(&["location", "person"], &observations(&first),
        &observations(&second), options(), Limits::default(), &mut || Ok(())).err(), Some(AssayError::Scope));
    let a = observations(&first); let mut b = observations(&first); b[1].language = "fr";
    assert_eq!(paired_with_checkpoint::<AssayError>(&["location", "person"], &a, &b,
        options(), Limits::default(), &mut || Ok(())).err(), Some(AssayError::Scope));
}
