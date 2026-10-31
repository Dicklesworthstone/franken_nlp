//! Bounded whole-document bootstrap shared by native structured evaluators.
//!
//! A caller supplies only compact additive document summaries and a fixed
//! metric vector. Paired arms consume identical draws in one traversal; no
//! private strings, field paths or mention lists enter the resampling loop.

use serde::{Deserialize, Serialize};
use super::{AssayError, BootstrapOptions, ConfidenceInterval, Limits};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalOptions {
    pub minimum_samples: usize,
    pub bootstrap: BootstrapOptions,
}
impl Default for EvalOptions {
    fn default() -> Self {
        Self { minimum_samples: 20,
            bootstrap: BootstrapOptions { resamples: 200, confidence_level: 0.95, seed: 0 } }
    }
}
impl EvalOptions {
    pub fn validate(self, limits: Limits) -> Result<(), AssayError> {
        limits.validate()?;
        if !(5..=100_000).contains(&self.minimum_samples)
            || !(2..=limits.max_bootstrap_resamples).contains(&self.bootstrap.resamples)
            || !self.bootstrap.confidence_level.is_finite()
            || self.bootstrap.confidence_level <= 0.0 || self.bootstrap.confidence_level >= 1.0 {
            return Err(AssayError::Options);
        }
        Ok(())
    }
    pub fn validate_rows(self, rows: usize, limits: Limits) -> Result<(), AssayError> { self.validate_arms(rows, 1, limits) }
    pub fn validate_paired_rows(self, rows: usize, limits: Limits) -> Result<(), AssayError> { self.validate_arms(rows, 2, limits) }
    fn validate_arms(self, rows: usize, arms: u64, limits: Limits) -> Result<(), AssayError> {
        self.validate(limits)?;
        if rows == 0 || rows > limits.max_rows
            || (rows as u64).checked_mul(self.bootstrap.resamples as u64)
                .and_then(|draws| draws.checked_mul(arms)).is_none_or(|draws| draws > limits.max_bootstrap_draws) {
            return Err(AssayError::InputLimit);
        }
        Ok(())
    }
}

pub(crate) struct Intervals<const N: usize> {
    pub candidate: [Option<ConfidenceInterval>; N],
    pub baseline: Option<[Option<ConfidenceInterval>; N]>,
    /// Candidate minus baseline on the SAME complete document draws.
    pub changes: Option<[Option<ConfidenceInterval>; N]>,
}

/// Price both per-arm accumulations before allocating up to three vectors per
/// metric. N <= 8 bounds bootstrap scalar storage at 192 bytes per resample.
#[allow(clippy::too_many_arguments)]
pub(crate) fn measure_with_checkpoint<const N: usize, A: Copy + Default, E: From<AssayError>>(
    candidate: &[A], baseline: Option<&[A]>, options: EvalOptions, limits: Limits,
    add: impl Fn(&mut A, A) -> Result<(), AssayError>, values: impl Fn(A) -> [Option<f64>; N],
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Intervals<N>, E> {
    if !(1..=8).contains(&N) { return Err(AssayError::Limits.into()); }
    if let Some(baseline) = baseline {
        options.validate_paired_rows(candidate.len(), limits)?;
        if baseline.len() != candidate.len() { return Err(AssayError::Scope.into()); }
    } else { options.validate_rows(candidate.len(), limits)?; }
    checkpoint()?;
    let mut current = SeriesSet::<N>::new(options.bootstrap.resamples)?;
    let mut prior = baseline.map(|_| SeriesSet::<N>::new(options.bootstrap.resamples)).transpose()?;
    let mut changes = baseline.map(|_| SeriesSet::<N>::new(options.bootstrap.resamples)).transpose()?;
    let mut state = options.bootstrap.seed;
    for _ in 0..options.bootstrap.resamples {
        checkpoint()?;
        let (mut a, mut b) = (A::default(), A::default());
        for draw in 0..candidate.len() {
            if draw % 64 == 0 { checkpoint()?; }
            let index = sampled_index(&mut state, candidate.len(), checkpoint)?;
            add(&mut a, candidate[index])?;
            if let Some(baseline) = baseline { add(&mut b, baseline[index])?; }
        }
        let a = values(a); current.push(a)?;
        if let Some(prior) = prior.as_mut() {
            let b = values(b); prior.push(b)?;
            let differences = std::array::from_fn(|index| a[index].zip(b[index]).map(|(a, b)| a - b));
            changes.as_mut().ok_or(AssayError::Scope)?.push(differences)?;
        }
    }
    let candidate = current.finish(options, checkpoint)?;
    let baseline = prior.as_mut().map(|values| values.finish(options, checkpoint)).transpose()?;
    let changes = changes.as_mut().map(|values| values.finish(options, checkpoint)).transpose()?;
    checkpoint()?; Ok(Intervals { candidate, baseline, changes })
}

struct Series { values: Vec<f64>, defined: bool }
impl Series {
    fn new(count: usize) -> Result<Self, AssayError> {
        let mut values = Vec::new(); values.try_reserve_exact(count).map_err(|_| AssayError::Allocation)?;
        Ok(Self { values, defined: true })
    }
    fn push(&mut self, value: Option<f64>) -> Result<(), AssayError> {
        if let Some(value) = value {
            if !value.is_finite() { return Err(AssayError::Observation); }
            self.values.push(value);
        } else { self.defined = false; }
        Ok(())
    }
    fn finish<E>(&mut self, options: EvalOptions, checkpoint: &mut impl FnMut() -> Result<(), E>)
        -> Result<Option<ConfidenceInterval>, E> {
        checkpoint()?;
        if !self.defined || self.values.is_empty() { return Ok(None); }
        self.values.sort_unstable_by(f64::total_cmp);
        let tail = (1.0 - options.bootstrap.confidence_level) / 2.0;
        let last = self.values.len() - 1;
        let interval = ConfidenceInterval { lower: self.values[(tail * last as f64).floor() as usize],
            upper: self.values[((1.0 - tail) * last as f64).ceil() as usize],
            confidence_level: options.bootstrap.confidence_level, resamples: options.bootstrap.resamples };
        checkpoint()?; Ok(Some(interval))
    }
}
struct SeriesSet<const N: usize> { series: [Series; N] }
impl<const N: usize> SeriesSet<N> {
    fn new(count: usize) -> Result<Self, AssayError> {
        let mut series = Vec::new(); series.try_reserve_exact(N).map_err(|_| AssayError::Allocation)?;
        for _ in 0..N { series.push(Series::new(count)?); }
        Ok(Self { series: series.try_into().map_err(|_| AssayError::Limits)? })
    }
    fn push(&mut self, values: [Option<f64>; N]) -> Result<(), AssayError> {
        for (series, value) in self.series.iter_mut().zip(values) { series.push(value)?; }
        Ok(())
    }
    fn finish<E>(&mut self, options: EvalOptions, checkpoint: &mut impl FnMut() -> Result<(), E>)
        -> Result<[Option<ConfidenceInterval>; N], E> {
        let mut result = [None; N];
        for (index, series) in self.series.iter_mut().enumerate() { result[index] = series.finish(options, checkpoint)?; }
        Ok(result)
    }
}
fn sampled_index<E>(state: &mut u64, upper: usize, checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<usize, E> {
    let upper = upper as u64;
    let limit = (u64::MAX / upper) * upper;
    loop {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        if value < limit { return Ok((value % upper) as usize); }
        checkpoint()?;
    }
}

#[cfg(test)] mod tests;
