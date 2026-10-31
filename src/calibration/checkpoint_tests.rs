//! Independent metric goldens and cancellation inside admitted statistical work.
use super::*;
use std::cell::Cell;

#[derive(Debug)]
enum Fault { Metric(CalibrationError), Cancelled { origin: &'static str, ticket: u64 } }
impl From<CalibrationError> for Fault {
    fn from(value: CalibrationError) -> Self { Self::Metric(value) }
}
fn stopped() -> Fault { Fault::Cancelled { origin: "caller_deadline", ticket: 31 } }
fn assert_stopped(error: Fault) {
    assert!(matches!(error, Fault::Cancelled { origin: "caller_deadline", ticket: 31 }));
}
fn locked(rows: usize) -> LockedTestSet {
    LockedTestSet { rows: (0..rows).map(|index| LabeledScore::new(format!("test-{index:06}"),
        if index.is_multiple_of(2) { 0.75 } else { 0.25 }, index.is_multiple_of(2)).unwrap()).collect() }
}

#[test]
fn known_binary_errors_keep_exact_point_metrics_and_constant_bootstrap_intervals() {
    let locked = locked(2);
    let (metrics, selective) = report_locked_test(&locked, Ok, 0.0).unwrap();
    assert_eq!(metrics.rows, 2);
    assert_eq!(metrics.ece, 0.25);
    assert_eq!(metrics.brier, 0.0625);
    assert_eq!(selective.accepted, 2);
    assert_eq!(selective.abstained, 0);
    assert_eq!(selective.risk, Some(0.5));
    // Every sampled row has absolute error 1/4 and squared error 1/16,
    // independent of which rows the deterministic random stream selects.
    let config = BootstrapConfig::new(64, 0.95, 7).unwrap();
    let intervals = bootstrap_locked_test_confidence_intervals(&locked, Ok, 0.0, config).unwrap();
    assert_eq!(intervals.ece.lower, 0.25); assert_eq!(intervals.ece.upper, 0.25);
    assert_eq!(intervals.brier.lower, 0.0625); assert_eq!(intervals.brier.upper, 0.0625);
    assert_eq!(intervals.bootstrap, config);
    let risk = intervals.selective_risk.unwrap();
    assert!(risk.lower >= 0.0 && risk.lower <= risk.upper && risk.upper <= 1.0);
}

#[test]
fn cancellation_before_transformation_does_not_read_any_private_score() {
    let transformed = Cell::new(0_usize);
    let error = report_locked_test_with_checkpoint::<Fault>(&locked(8), |probability| {
        transformed.set(transformed.get() + 1); Ok(probability)
    }, 0.5, &mut || Err(stopped())).unwrap_err();
    assert_stopped(error);
    assert_eq!(transformed.get(), 0);
}

#[test]
fn one_large_metric_reduction_polls_before_finishing_the_whole_population() {
    let rows = vec![CalibratedLockedRow { probability: 0.75, positive: true };
        3 * METRIC_ROWS_PER_CHECKPOINT + 7];
    let mut calls = 0;
    let error = evaluate_locked_rows::<Fault>(&rows, 0.5, &mut || {
        calls += 1;
        if calls == 2 { Err(stopped()) } else { Ok(()) }
    }).unwrap_err();
    assert_stopped(error);
    assert_eq!(calls, 2);
}

#[test]
fn bootstrap_stops_during_replica_work_and_preserves_the_callers_typed_cause() {
    let locked = locked(2 * METRIC_ROWS_PER_CHECKPOINT + 3);
    for stop_at in [12, 40, 100] {
        let transformed = Cell::new(0_usize); let mut calls = 0;
        let error = bootstrap_locked_test_confidence_intervals_with_checkpoint::<Fault>(
            &locked, |probability| { transformed.set(transformed.get() + 1); Ok(probability) },
            0.5, BootstrapConfig::new(128, 0.95, 9).unwrap(), &mut || {
                calls += 1;
                if calls == stop_at { Err(stopped()) } else { Ok(()) }
            }).unwrap_err();
        assert_stopped(error);
        assert_eq!(calls, stop_at);
        // Original locked rows are transformed once. Resampling never gains
        // access to another split or reruns an arbitrary calibration transform.
        assert_eq!(transformed.get(), locked.len());
    }
}

#[test]
fn invalid_transformed_probabilities_remain_metric_errors() {
    let error = report_locked_test_with_checkpoint::<Fault>(&locked(2), |_| Ok(f64::NAN),
        0.5, &mut || Ok(())).unwrap_err();
    assert!(matches!(error, Fault::Metric(CalibrationError::InvalidProbability { .. })));
    let error = bootstrap_locked_test_confidence_intervals_with_checkpoint::<Fault>(
        &locked(2), Ok, f64::INFINITY, BootstrapConfig::new(8, 0.95, 1).unwrap(),
        &mut || Ok(())).unwrap_err();
    assert!(matches!(error, Fault::Metric(CalibrationError::InvalidThreshold)));
}

#[test]
fn zero_acceptance_keeps_an_explicit_undefined_selective_risk() {
    let locked = locked(2);
    let (metrics, risk) = report_locked_test_with_checkpoint::<Fault>(
        &locked, Ok, 1.0, &mut || Ok(())).unwrap();
    assert_eq!(metrics.rows, 2); assert_eq!(risk.accepted, 0);
    assert_eq!(risk.abstained, 2); assert_eq!(risk.risk, None);
    let result = bootstrap_locked_test_confidence_intervals_with_checkpoint::<Fault>(
        &locked, Ok, 1.0, BootstrapConfig::new(8, 0.95, 1).unwrap(),
        &mut || Ok(())).unwrap();
    assert_eq!(result.selective_risk, None);
}
