use super::*;
use super::super::tests::{arithmetic_replay, dataset, options};

#[test]
fn tie_aware_ranks_and_constant_vectors_have_explicit_semantics() {
    let values = [-1.0, -1.0, -0.0, 0.0, 1.0];
    assert_eq!(ranks::<AssayError>(5, |index| values[index], &mut || Ok(())).unwrap(),
        vec![0.5, 0.5, 2.5, 2.5, 4.0]);
    let data = dataset(5, 1); let mut replay = arithmetic_replay(&data);
    for (row, mean) in replay.rows.iter_mut().zip([-1.0, -1.0, 0.0, 1.0, 1.0]) { row.means[0] = mean; }
    let (correlation, status) = spearman::<AssayError>(&replay, 0, &mut || Ok(())).unwrap();
    assert_eq!(status, SpearmanStatus::Measured); assert!((correlation.unwrap() - 9.0 / 90.0_f64.sqrt()).abs() < 1e-12);
    for row in &mut replay.rows { row.means[0] = 0.0; }
    assert_eq!(spearman::<AssayError>(&replay, 0, &mut || Ok(())).unwrap(), (None, SpearmanStatus::ConstantPrediction));
    for row in &mut replay.rows { row.gold[0].value_milli = 133; }
    assert_eq!(spearman::<AssayError>(&replay, 0, &mut || Ok(())).unwrap(), (None, SpearmanStatus::BothConstant));
    for (index, row) in replay.rows.iter_mut().enumerate() { row.means[0] = index as f64 / 4.0; }
    assert_eq!(spearman::<AssayError>(&replay, 0, &mut || Ok(())).unwrap(), (None, SpearmanStatus::ConstantGold));
}

#[test]
fn abstention_does_not_remove_errors_from_the_whole_document_population() {
    let data = dataset(5, 1); let mut replay = arithmetic_replay(&data);
    for (index, row) in replay.rows.iter_mut().enumerate() { row.means[0] = 0.0; row.estimated[0] = index == 0 || index == 2; }
    let (measurement, _, _) = measure::<AssayError>(&replay, None, options(), Limits::default(), &mut || Ok(())).unwrap();
    let Measurement::Measured { documents, axes } = measurement else { panic!("five documents suffice"); };
    assert_eq!(documents, 5); let axis = &axes[0];
    assert_eq!(axis.mae, 0.6); assert_eq!(axis.rmse, 0.5_f64.sqrt());
    assert_eq!(axis.estimated_coverage, 0.4); assert_eq!(axis.estimated_mae, Some(0.5));
    assert_eq!(axis.estimated_documents, 2); assert_eq!(axis.abstained_documents, 3);
    for row in &mut replay.rows { row.estimated[0] = false; }
    let (measurement, _, _) = measure::<AssayError>(&replay, None, options(), Limits::default(), &mut || Ok(())).unwrap();
    let Measurement::Measured { axes, .. } = measurement else { unreachable!() };
    assert_eq!(axes[0].mae, axis.mae); assert_eq!(axes[0].rmse, axis.rmse);
    assert_eq!(axes[0].estimated_coverage, 0.0); assert_eq!(axes[0].estimated_mae, None);
    assert_eq!(axes[0].intervals.estimated_mae, None);
}

#[test]
fn milli_gold_keeps_exact_between_bin_error_and_extreme_error_is_two() {
    let data = dataset(5, 1); let mut replay = arithmetic_replay(&data);
    for row in &mut replay.rows { row.gold[0].value_milli = 133; row.means[0] = 0.5; }
    let (_, counts) = axis_counts::<AssayError>(&replay, 0, &mut || Ok(())).unwrap();
    let values = counts.values(); assert!((values[0].unwrap() - 0.367).abs() < 1e-12);
    assert!((values[1].unwrap() - 0.367).abs() < 1e-12);
    for row in &mut replay.rows { row.gold[0].value_milli = -1000; row.means[0] = 1.0; }
    let (_, counts) = axis_counts::<AssayError>(&replay, 0, &mut || Ok(())).unwrap();
    assert_eq!(counts.values(), [Some(2.0), Some(2.0), Some(1.0), Some(2.0)]);
}

#[test]
fn bootstrap_resamples_whole_documents_with_one_axis_and_arm_draw_allowance() {
    let data = dataset(5, 4); let mut replay = arithmetic_replay(&data);
    for (index, row) in replay.rows.iter_mut().enumerate() {
        for mean in &mut row.means { *mean = if index % 2 == 0 { 0.0 } else { *mean }; }
    }
    let options = options(); let mut limits = Limits::default();
    limits.max_bootstrap_draws = 5 * options.bootstrap.resamples as u64 * 4 * 2 - 1;
    assert!(validate_replay(options, 5, 4, false, limits).is_ok());
    assert_eq!(validate_replay(options, 5, 4, true, limits), Err(AssayError::InputLimit));
    limits.max_bootstrap_draws += 1;
    let (measurement, _, paired) = measure::<AssayError>(&replay, Some(&replay), options, limits, &mut || Ok(())).unwrap();
    let Measurement::Measured { documents, axes } = measurement else { unreachable!() };
    assert_eq!(documents, 5);
    assert!(axes.iter().all(|axis| axis.documents == 5 && axis.intervals == axes[0].intervals));
    for axis in paired.unwrap().axes {
        for interval in [axis.intervals.mae, axis.intervals.rmse, axis.intervals.estimated_coverage, axis.intervals.estimated_mae] {
            let interval = interval.unwrap(); assert_eq!(interval.lower, 0.0); assert_eq!(interval.upper, 0.0);
        }
    }
}
