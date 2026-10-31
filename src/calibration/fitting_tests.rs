//! Closed-form fits, exact knot boundaries and interruption inside fit work.
use super::*;

#[derive(Debug)]
enum Fault { Metric(CalibrationError), Stopped { ticket: u64 } }
impl From<CalibrationError> for Fault {
    fn from(error: CalibrationError) -> Self { Self::Metric(error) }
}
fn stopped() -> Fault { Fault::Stopped { ticket: 73 } }
fn assert_stopped(error: Fault) { assert!(matches!(error, Fault::Stopped { ticket: 73 })); }
fn rows(points: &[(f64, bool)]) -> CalibrationSet {
    CalibrationSet { rows: points.iter().enumerate().map(|(index, &(probability, positive))|
        LabeledScore::new(format!("cal-{index:08}"), probability, positive).unwrap()).collect() }
}

#[test]
fn temperature_fit_matches_the_closed_form_constant_score_optimum() {
    // With a constant raw logit and three positives in four rows, the optimum
    // calibrated probability is 3/4 and T = logit(4/5) / logit(3/4).
    let calibration = rows(&[(0.8, true), (0.8, true), (0.8, true), (0.8, false)]);
    let mut checkpoints = 0;
    let fit = TemperatureModel::fit_with_checkpoint::<Fault>(&calibration, &mut || {
        checkpoints += 1; Ok(())
    }).unwrap();
    assert!((fit.temperature() - 4.0_f64.ln() / 3.0_f64.ln()).abs() < 1e-6);
    assert!((fit.calibrate(0.8).unwrap() - 0.75).abs() < 1e-6);
    assert_eq!(fit.fitted_rows(), 4);
    assert!(checkpoints > 128);
}

#[test]
fn temperature_endpoint_refusal_is_not_silently_clipped_and_isotonic_accepts_endpoints() {
    for endpoint in [0.0, 1.0] {
        let calibration = rows(&[(endpoint, false), (0.5, true)]);
        let error = TemperatureModel::fit_with_checkpoint::<Fault>(
            &calibration, &mut || Ok(())).unwrap_err();
        assert!(matches!(error, Fault::Metric(CalibrationError::TemperatureNeedsOpenProbability { .. })));
    }
    let fit = IsotonicModel::fit(&rows(&[(0.0, false), (1.0, true)])).unwrap();
    assert_eq!(fit.calibrate(0.0).unwrap(), 0.0);
    assert_eq!(fit.calibrate(1.0).unwrap(), 1.0);
}

#[test]
fn a_cancelled_fit_never_reads_an_invalid_score_or_changes_the_sealed_split() {
    let calibration = rows(&[(0.0, false), (1.0, true)]);
    let original = calibration.clone();
    assert_stopped(TemperatureModel::fit_with_checkpoint::<Fault>(
        &calibration, &mut || Err(stopped())).unwrap_err());
    assert_stopped(IsotonicModel::fit_with_checkpoint::<Fault>(
        &calibration, &mut || Err(stopped())).unwrap_err());
    assert_eq!(calibration, original);
}

#[test]
fn one_temperature_loss_pass_is_interruptible_before_all_rows_are_reduced() {
    let calibration = CalibrationSet { rows: (0..3 * FIT_ROWS_PER_CHECKPOINT + 7)
        .map(|index| LabeledScore::new(format!("cal-{index}"), 0.75, index % 2 == 0).unwrap()).collect() };
    let mut calls = 0;
    let error = temperature_loss::<Fault>(&calibration, 1.0, &mut || {
        calls += 1;
        if calls == 2 { Err(stopped()) } else { Ok(()) }
    }).unwrap_err();
    assert_stopped(error);
    assert_eq!(calls, 2);
}

#[test]
fn temperature_search_preserves_the_callers_stop_after_initial_loss_passes() {
    let calibration = rows(&[(0.1, false), (0.2, true), (0.8, false), (0.9, true)]);
    let original = calibration.clone();
    let mut calls = 0;
    let error = TemperatureModel::fit_with_checkpoint::<Fault>(&calibration, &mut || {
        calls += 1;
        if calls == 20 { Err(stopped()) } else { Ok(()) }
    }).unwrap_err();
    assert_stopped(error);
    assert_eq!(calls, 20);
    assert_eq!(calibration, original);
}

#[test]
fn isotonic_merges_across_sort_chunks_preserve_weighted_ties_and_input_order_independence() {
    let count = 8 * (FIT_ROWS_PER_CHECKPOINT + 1);
    let mut calibration = CalibrationSet { rows: (0..count).map(|index| {
        let lower = index % 8 < 4;
        LabeledScore::new(format!("cal-{index:08}"), if lower { 0.25 } else { 0.75 },
            if lower { index % 4 == 0 } else { index % 4 != 0 }).unwrap()
    }).collect() };
    calibration.rows.reverse();
    let mut checkpoints = 0;
    let model = IsotonicModel::fit_with_checkpoint::<Fault>(&calibration, &mut || {
        checkpoints += 1; Ok(())
    }).unwrap();
    assert_eq!(model.blocks(), &[
        IsotonicBlock { upper_score: 0.25, calibrated_probability: 0.25 },
        IsotonicBlock { upper_score: 0.75, calibrated_probability: 0.75 },
    ]);
    assert!(checkpoints > count / FIT_ROWS_PER_CHECKPOINT);
    calibration.rows.rotate_left(1537);
    assert_eq!(IsotonicModel::fit(&calibration).unwrap(), model);
}

#[test]
fn row_sort_can_stop_during_merges_without_cloning_or_mutating_source_rows() {
    let calibration = CalibrationSet { rows: (0..4 * FIT_ROWS_PER_CHECKPOINT + 7)
        .rev().map(|index| LabeledScore::new(format!("cal-{index:08}"),
            (index % 997) as f64 / 997.0, index % 2 == 0).unwrap()).collect() };
    let original = calibration.clone();
    // These cover collection, bounded leaf sorting and subsequent merge work.
    for stop_at in [1, 7, 14, 20] {
        let mut calls = 0;
        let error = sorted_calibration_rows::<Fault>(&calibration, &mut || {
            calls += 1;
            if calls == stop_at { Err(stopped()) } else { Ok(()) }
        }).unwrap_err();
        assert_stopped(error);
        assert_eq!(calls, stop_at);
        assert_eq!(calibration, original);
    }
}

#[test]
fn a_long_pav_cascade_remains_cancellable_through_final_knot_materialization() {
    let count = 3 * FIT_ROWS_PER_CHECKPOINT + 1;
    let calibration = CalibrationSet { rows: (0..count).map(|index|
        LabeledScore::new(format!("cal-{index:08}"), index as f64 / (count - 1) as f64,
            index + 1 != count).unwrap()).collect() };
    let mut completed_calls = 0;
    let model = IsotonicModel::fit_with_checkpoint::<Fault>(&calibration, &mut || {
        completed_calls += 1; Ok(())
    }).unwrap();
    assert_eq!(model.blocks(), &[IsotonicBlock { upper_score: 1.0,
        calibrated_probability: (count - 1) as f64 / count as f64 }]);
    for stop_at in [completed_calls / 2, completed_calls - 3, completed_calls] {
        let mut calls = 0;
        assert_stopped(IsotonicModel::fit_with_checkpoint::<Fault>(&calibration, &mut || {
            calls += 1;
            if calls == stop_at { Err(stopped()) } else { Ok(()) }
        }).unwrap_err());
        assert_eq!(calls, stop_at);
    }
}

#[test]
fn binary_knot_lookup_matches_inclusive_upper_boundary_semantics() {
    let knots: Vec<_> = (0..4097).map(|index| IsotonicBlock {
        upper_score: index as f64 / 4096.0,
        calibrated_probability: index as f64 / 8192.0,
    }).collect();
    let model = IsotonicModel::from_blocks(knots.clone()).unwrap();
    for index in 0..knots.len() {
        let bound = knots[index].upper_score;
        let preceding = knots[index.saturating_sub(1)].upper_score;
        for probability in [bound, (preceding + bound) / 2.0] {
            let expected = knots.iter().find(|knot| probability <= knot.upper_score)
                .unwrap_or_else(|| knots.last().unwrap()).calibrated_probability;
            assert_eq!(model.calibrate(probability).unwrap().to_bits(), expected.to_bits());
        }
    }
    for invalid in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert!(matches!(model.calibrate(invalid), Err(CalibrationError::InvalidProbability { .. })));
    }
}

#[test]
fn empty_fits_and_unrepresentable_storage_are_typed_refusals() {
    let empty = CalibrationSet { rows: Vec::new() };
    assert!(matches!(TemperatureModel::fit(&empty), Err(CalibrationError::EmptySplit(SplitName::Calibration))));
    assert!(matches!(IsotonicModel::fit(&empty), Err(CalibrationError::EmptySplit(SplitName::Calibration))));
    assert!(matches!(fit_storage::<IsotonicBlock>(usize::MAX), Err(CalibrationError::AllocationRefused)));
}
