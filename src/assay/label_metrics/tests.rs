use super::*;

fn options() -> EvalOptions {
    EvalOptions { minimum_samples: 5, acceptance_threshold: 0.7,
        bootstrap: crate::assay::BootstrapOptions { resamples: 32, confidence_level: 0.95, seed: 17 } }
}
fn close(a: f64, b: f64) { assert!((a - b).abs() < 1e-12, "{a} != {b}"); }
fn measured(mode: LabelMode, labels: &[&str], rows: &[Observation<'_>]) -> (LockedTestMetrics, LabelSetMetrics) {
    let (measurement, quality) = measure_with_checkpoint::<AssayError>(mode, labels, rows,
        options(), Limits::default(), &mut || Ok(())).unwrap();
    let Measurement::Measured { metrics } = measurement else { panic!("expected measured data"); };
    (metrics, quality.unwrap())
}

#[test]
fn multiclass_has_one_hot_brier_top_label_ece_and_abstention_aware_quality() {
    let rows = [
        Observation { positive: &[true, false, false], probability: &[0.8, 0.1, 0.1],
            predicted: &[Some(true), Some(false), Some(false)], top_label: Some(0) },
        Observation { positive: &[false, true, false], probability: &[0.1, 0.7, 0.2],
            predicted: &[Some(false), Some(true), Some(false)], top_label: Some(1) },
        Observation { positive: &[false, false, true], probability: &[0.6, 0.1, 0.3],
            predicted: &[Some(true), Some(false), Some(false)], top_label: Some(0) },
        Observation { positive: &[true, false, false], probability: &[0.4, 0.35, 0.25],
            predicted: &[None, None, None], top_label: Some(0) },
        Observation { positive: &[false, false, true], probability: &[0.2, 0.2, 0.6],
            predicted: &[Some(false), Some(false), Some(true)], top_label: Some(2) },
    ];
    let (metrics, quality) = measured(LabelMode::Multiclass, &["alpha", "beta", "gamma"], &rows);
    close(metrics.brier, 0.1845); close(metrics.ece, 0.26);
    assert_eq!(metrics.accepted, 2); assert_eq!(metrics.selective_risk, Some(0.0));
    close(quality.exact_match_accuracy, 3.0 / 5.0);
    close(quality.decided_exact_match_accuracy.unwrap(), 3.0 / 4.0);
    close(quality.micro_f1.unwrap(), 2.0 / 3.0); close(quality.macro_f1.unwrap(), 13.0 / 18.0);
    close(quality.hamming_loss, 1.0 / 3.0); close(quality.label_decision_coverage, 0.8);
    assert_eq!(quality.labels[0].abstained_positive, 1);
}

#[test]
fn multilabel_keeps_independent_scores_empty_gold_and_partial_abstentions() {
    let rows = [
        Observation { positive: &[true, true, false], probability: &[0.8, 0.9, 0.1],
            predicted: &[Some(true), Some(true), Some(false)], top_label: None },
        Observation { positive: &[false, true, true], probability: &[0.8, 0.7, 0.4],
            predicted: &[Some(true), Some(true), Some(false)], top_label: None },
        Observation { positive: &[false, false, false], probability: &[0.1, 0.2, 0.3],
            predicted: &[Some(false), Some(false), Some(false)], top_label: None },
        Observation { positive: &[true, false, true], probability: &[0.4, 0.2, 0.6],
            predicted: &[None, Some(false), Some(true)], top_label: None },
        Observation { positive: &[true, false, false], probability: &[0.9, 0.6, 0.2],
            predicted: &[Some(true), Some(true), Some(false)], top_label: None },
    ];
    let (metrics, quality) = measured(LabelMode::Multilabel, &["alpha", "beta", "gamma"], &rows);
    close(metrics.brier, 2.26 / 15.0); assert_eq!(metrics.accepted, 2);
    assert_eq!(quality.decided_rows, 4); assert_eq!(quality.partially_decided_rows, 1);
    close(quality.exact_match_accuracy, 0.4); close(quality.hamming_loss, 4.0 / 15.0);
    close(quality.label_decision_coverage, 14.0 / 15.0);
    close(quality.micro_f1.unwrap(), 5.0 / 7.0); close(quality.macro_f1.unwrap(), 32.0 / 45.0);
    assert_eq!(quality.labels[0].abstained_positive, 1);
}

#[test]
fn duplicating_correlated_labels_cannot_artificially_narrow_document_bootstrap() {
    let probabilities = [0.1, 0.2, 0.5, 0.7, 0.9];
    let gold = [false, false, true, true, false];
    let predictions = [Some(false), Some(false), Some(true), Some(true), Some(true)];
    let doubled_probabilities = probabilities.map(|p| [p, p]);
    let doubled_gold = gold.map(|g| [g, g]);
    let doubled_predictions = predictions.map(|p| [p, p]);
    let one: Vec<_> = (0..5).map(|i| Observation { positive: &gold[i..i + 1],
        probability: &probabilities[i..i + 1], predicted: &predictions[i..i + 1], top_label: None }).collect();
    let two: Vec<_> = (0..5).map(|i| Observation { positive: &doubled_gold[i],
        probability: &doubled_probabilities[i], predicted: &doubled_predictions[i], top_label: None }).collect();
    let (a, _) = measured(LabelMode::Multilabel, &["one"], &one);
    let (b, _) = measured(LabelMode::Multilabel, &["one", "correlated-copy"], &two);
    close(a.brier, b.brier); close(a.ece, b.ece);
    close(a.brier_interval.lower, b.brier_interval.lower); close(a.brier_interval.upper, b.brier_interval.upper);
    close(a.ece_interval.lower, b.ece_interval.lower); close(a.ece_interval.upper, b.ece_interval.upper);
    assert_eq!(a.selective_risk_interval, b.selective_risk_interval);
}

#[test]
fn absent_classes_have_no_invented_macro_f1_and_zero_acceptance_has_no_risk_interval() {
    let row = Observation { positive: &[true, false], probability: &[0.6, 0.4],
        predicted: &[Some(true), Some(false)], top_label: Some(0) };
    let (metrics, quality) = measured(LabelMode::Multiclass, &["present", "absent"], &[row; 5]);
    assert_eq!(quality.macro_f1, None); assert_eq!(quality.labels[1].f1, None);
    assert_eq!(metrics.accepted, 0); assert_eq!(metrics.selective_risk, None);
    assert_eq!(metrics.selective_risk_interval, None);
    close(metrics.brier_interval.lower, 0.16); close(metrics.brier_interval.upper, 0.16);
}

#[test]
fn native_abstention_never_passes_selective_acceptance_even_with_large_weights() {
    let row = Observation { positive: &[true, false], probability: &[0.99, 0.01],
        predicted: &[None, None], top_label: Some(0) };
    let (metrics, quality) = measured(LabelMode::Multiclass, &["yes", "no"], &[row; 5]);
    assert_eq!(metrics.accepted, 0); assert_eq!(quality.exact_match_accuracy, 0.0);
    assert_eq!(quality.decided_exact_match_accuracy, None);
    close(quality.hamming_loss, 1.0);
}

#[test]
fn invalid_vector_shapes_and_nonfinite_or_nonexclusive_scores_are_refused() {
    let good = Observation { positive: &[true, false], probability: &[0.7, 0.3],
        predicted: &[Some(true), Some(false)], top_label: Some(0) };
    for row in [Observation { probability: &[0.7], ..good },
        Observation { probability: &[f64::NAN, 0.3], ..good },
        Observation { probability: &[0.7, 0.4], ..good },
        Observation { positive: &[true, true], ..good },
        Observation { predicted: &[Some(false), Some(true)], ..good },
        Observation { predicted: &[None, Some(false)], ..good },
        Observation { top_label: None, ..good }] {
        assert_eq!(brier_loss(LabelMode::Multiclass, row), Err(AssayError::Observation));
    }
    assert_eq!(brier_loss(LabelMode::Multilabel, good), Err(AssayError::Observation));
}

#[test]
fn multiclass_two_label_loss_equals_binary_event_brier() {
    for p in [0.0, 0.1, 0.5, 0.9, 1.0] {
        for positive in [false, true] {
            let probabilities = [p, 1.0 - p]; let gold = [positive, !positive];
            let row = Observation { positive: &gold, probability: &probabilities,
                predicted: &[None, None], top_label: Some(usize::from(p < 0.5)) };
            let error = p - if positive { 1.0 } else { 0.0 };
            close(brier_loss(LabelMode::Multiclass, row).unwrap(), error * error);
        }
    }
}

#[test]
fn tiny_samples_are_explicit_and_draw_allowance_is_checked_before_bootstrap() {
    let row = Observation { positive: &[true], probability: &[0.8], predicted: &[Some(true)], top_label: None };
    let (measurement, quality) = measure_with_checkpoint::<AssayError>(LabelMode::Multilabel, &["a"], &[row; 4],
        options(), Limits::default(), &mut || Ok(())).unwrap();
    assert!(matches!(measurement, Measurement::InsufficientData { rows: 4, .. }));
    assert!(quality.is_none());
    let limits = Limits { max_bootstrap_draws: 159, ..Limits::default() };
    assert!(matches!(measure_with_checkpoint::<AssayError>(LabelMode::Multilabel, &["a"], &[row; 5],
        options(), limits, &mut || Ok(())), Err(AssayError::InputLimit)));
}

#[test]
fn cancellation_propagates_inside_one_large_bootstrap_replica() {
    #[derive(Debug, PartialEq)] enum Error { Cancelled, Assay(AssayError) }
    impl From<AssayError> for Error { fn from(e: AssayError) -> Self { Self::Assay(e) } }
    let row = Observation { positive: &[true], probability: &[0.8], predicted: &[Some(true)], top_label: None };
    let rows = [row; 1024];
    let mut polls = 0;
    let result = measure_with_checkpoint(LabelMode::Multilabel, &["a"], &rows,
        options(), Limits::default(), &mut || { polls += 1;
            if polls == 25 { Err(Error::Cancelled) } else { Ok(()) } });
    assert!(matches!(result, Err(Error::Cancelled)));
    assert_eq!(polls, 25);
}

#[test]
fn bounded_reservations_reject_capacity_overflow_without_panicking() {
    assert!(matches!(reserved::<Document>(usize::MAX), Err(AssayError::Allocation)));
}
