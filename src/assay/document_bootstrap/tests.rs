use super::*;

fn options() -> EvalOptions {
    EvalOptions { minimum_samples: 5, bootstrap: BootstrapOptions { resamples: 32, confidence_level: 0.9, seed: 47 } }
}
fn run(a: &[usize], b: Option<&[usize]>, options: EvalOptions, limits: Limits) -> Result<Intervals<1>, AssayError> {
    measure_with_checkpoint(a, b, options, limits, |a, b| { *a = a.checked_add(b).ok_or(AssayError::InputLimit)?; Ok(()) },
        |sum| [Some(sum as f64 / 5.0)], &mut || Ok(()))
}

#[test]
fn fixed_seed_marginals_match_both_arms_and_same_draws_cancel_exactly() {
    let rows = [0, 1, 0, 1, 1];
    let single = run(&rows, None, options(), Limits::default()).unwrap();
    let paired = run(&rows, Some(&rows), options(), Limits::default()).unwrap();
    assert_eq!(single.candidate, paired.candidate); assert_eq!(Some(single.candidate), paired.baseline);
    let interval = paired.changes.unwrap()[0].unwrap(); assert_eq!((interval.lower, interval.upper), (0.0, 0.0));
    assert_eq!(run(&rows, None, options(), Limits::default()).unwrap().candidate, single.candidate);
}

#[test]
fn missing_replicates_are_not_conditioned_away() {
    let mut series = Series::new(3).unwrap();
    series.push(Some(0.3)).unwrap(); series.push(None).unwrap(); series.push(Some(0.7)).unwrap();
    assert_eq!(series.finish(options(), &mut || Ok::<_, AssayError>(())).unwrap(), None);
    let result = measure_with_checkpoint::<1, usize, AssayError>(&[0; 5], Some(&[1; 5]), options(), Limits::default(),
        |a, b| { *a += b; Ok(()) }, |sum| [if sum == 0 { None } else { Some(1.0) }], &mut || Ok(())).unwrap();
    assert_eq!(result.candidate, [None]); assert!(result.baseline.unwrap()[0].is_some());
    assert_eq!(result.changes.unwrap(), [None]);
}

#[test]
fn non_finite_metrics_and_unbounded_metric_vectors_fail_closed() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(measure_with_checkpoint::<1, usize, AssayError>(&[1; 5], None, options(), Limits::default(),
            |a, b| { *a += b; Ok(()) }, |_| [Some(value)], &mut || Ok(())).is_err());
    }
    assert!(measure_with_checkpoint::<9, usize, AssayError>(&[1; 5], None, options(), Limits::default(),
        |_, _| Ok(()), |_| [Some(0.0); 9], &mut || Ok(())).is_err());
}

#[test]
fn paired_accumulations_are_priced_before_any_checkpoint_or_allocation() {
    let limits = Limits { max_bootstrap_draws: 5 * options().bootstrap.resamples as u64, ..Limits::default() };
    assert!(run(&[1; 5], None, options(), limits).is_ok());
    assert_eq!(run(&[1; 5], Some(&[1; 5]), options(), limits).err(), Some(AssayError::InputLimit));
    assert_eq!(run(&[1; 5], Some(&[1; 4]), options(), Limits::default()).err(), Some(AssayError::Scope));
    assert_eq!(run(&[], None, options(), Limits::default()).err(), Some(AssayError::InputLimit));
}

#[derive(Debug, PartialEq)]
enum Stop { Cancelled, Assay(AssayError) }
impl From<AssayError> for Stop { fn from(error: AssayError) -> Self { Self::Assay(error) } }
#[test]
fn bootstrap_preserves_typed_cancellation_inside_document_draws() {
    let mut calls = 0;
    let error = measure_with_checkpoint::<1, usize, Stop>(&[1; 129], None, options(), Limits::default(),
        |a, b| { *a += b; Ok(()) }, |sum| [Some(sum as f64)], &mut || {
            calls += 1; if calls == 4 { Err(Stop::Cancelled) } else { Ok(()) }
        }).err();
    assert_eq!(error, Some(Stop::Cancelled));
}
