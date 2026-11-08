//! Normalized-coordinate errors and paired whole-document uncertainty.
use serde::{Deserialize, Serialize};
use crate::assay::{ConfidenceInterval, document_bootstrap};
use super::*;
pub use crate::assay::document_bootstrap::EvalOptions;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpearmanStatus { Measured, ConstantPrediction, ConstantGold, BothConstant }

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualityIntervals {
    pub mae: Option<ConfidenceInterval>,
    pub rmse: Option<ConfidenceInterval>,
    pub estimated_coverage: Option<ConfidenceInterval>,
    /// Undefined if any bootstrap replica has no estimated documents.
    pub estimated_mae: Option<ConfidenceInterval>,
}
impl From<[Option<ConfidenceInterval>; 4]> for QualityIntervals {
    fn from(values: [Option<ConfidenceInterval>; 4]) -> Self {
        let [mae, rmse, estimated_coverage, estimated_mae] = values;
        Self { mae, rmse, estimated_coverage, estimated_mae }
    }
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AxisQuality {
    pub axis: SentimentAxis,
    pub documents: usize,
    /// All native distribution means participate, including abstained rows.
    /// Errors use the task's -1..=1 coordinate; maximum MAE/RMSE is two.
    pub mae: f64,
    pub rmse: f64,
    pub spearman: Option<f64>,
    pub spearman_status: SpearmanStatus,
    pub estimated_documents: usize,
    pub abstained_documents: usize,
    pub estimated_coverage: f64,
    pub estimated_mae: Option<f64>,
    pub intervals: QualityIntervals,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum Measurement {
    Measured { documents: usize, axes: Vec<AxisQuality> },
    InsufficientData { documents: usize, minimum_documents: usize },
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AxisChange {
    pub axis: SentimentAxis,
    /// Every difference and interval is candidate minus baseline.
    pub mae_change: f64,
    pub rmse_change: f64,
    pub spearman_change: Option<f64>,
    pub estimated_coverage_change: f64,
    pub estimated_mae_change: Option<f64>,
    pub intervals: QualityIntervals,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedSummary { pub documents: usize, pub axes: Vec<AxisChange> }

/// Axes are correlated measurements of a document, never extra sample units.
/// Each axis reuses identical seeded document draws, while this one preflight
/// charges every per-axis/arm accumulation to the global bootstrap budget.
pub(crate) fn validate_replay(options: EvalOptions, documents: usize, axes: usize,
    paired: bool, limits: Limits) -> Result<(), AssayError> {
    if !(1..=MAX_AXES).contains(&axes) { return Err(AssayError::Limits); }
    if paired { options.validate_paired_rows(documents, limits)?; }
    else { options.validate_rows(documents, limits)?; }
    if (documents as u64).checked_mul(options.bootstrap.resamples as u64)
        .and_then(|draws| draws.checked_mul(axes as u64))
        .and_then(|draws| draws.checked_mul(if paired { 2 } else { 1 }))
        .is_none_or(|draws| draws > limits.max_bootstrap_draws) { return Err(AssayError::InputLimit); }
    Ok(())
}

#[derive(Clone, Copy, Default)]
struct Counts { documents: usize, absolute: f64, squared: f64, estimated: usize, estimated_absolute: f64 }
impl Counts {
    fn add(&mut self, other: Self) -> Result<(), AssayError> {
        self.documents = self.documents.checked_add(other.documents).ok_or(AssayError::InputLimit)?;
        self.estimated = self.estimated.checked_add(other.estimated).ok_or(AssayError::InputLimit)?;
        self.absolute += other.absolute; self.squared += other.squared; self.estimated_absolute += other.estimated_absolute;
        if self.estimated > self.documents || !self.absolute.is_finite() || !self.squared.is_finite()
            || !self.estimated_absolute.is_finite() { return Err(AssayError::Observation); }
        Ok(())
    }
    fn values(self) -> [Option<f64>; 4] {
        if self.documents == 0 { return [None; 4]; }
        [Some(self.absolute / self.documents as f64), Some((self.squared / self.documents as f64).sqrt()),
            Some(self.estimated as f64 / self.documents as f64),
            (self.estimated != 0).then(|| self.estimated_absolute / self.estimated as f64)]
    }
    fn quality(self, axis: SentimentAxis, rank: (Option<f64>, SpearmanStatus), intervals: QualityIntervals)
        -> Result<AxisQuality, AssayError> {
        let [mae, rmse, coverage, estimated_mae] = self.values();
        Ok(AxisQuality { axis, documents: self.documents, mae: mae.ok_or(AssayError::Scope)?,
            rmse: rmse.ok_or(AssayError::Scope)?, spearman: rank.0, spearman_status: rank.1,
            estimated_documents: self.estimated, abstained_documents: self.documents - self.estimated,
            estimated_coverage: coverage.ok_or(AssayError::Scope)?, estimated_mae, intervals })
    }
}
fn axis_counts<E: From<AssayError>>(replay: &NativeReplay, axis_index: usize,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(Vec<Counts>, Counts), E> {
    let axis = *replay.manifest.axes.get(axis_index).ok_or(AssayError::Scope)?;
    let mut values = reserved(replay.rows.len())?; let mut total = Counts::default();
    for (index, row) in replay.rows.iter().enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        if row.gold.len() != replay.manifest.axes.len() || row.means.len() != row.gold.len()
            || row.estimated.len() != row.gold.len() { return Err(AssayError::Observation.into()); }
        let gold = row.gold[axis_index]; let mean = row.means[axis_index];
        if gold.axis != axis || !(-GOLD_SCALE..=GOLD_SCALE).contains(&gold.value_milli)
            || !mean.is_finite() || !(-1.0..=1.0).contains(&mean) { return Err(AssayError::Observation.into()); }
        let error = (mean - f64::from(gold.value_milli) / f64::from(GOLD_SCALE)).abs();
        let estimated = row.estimated[axis_index];
        let counts = Counts { documents: 1, absolute: error, squared: error * error,
            estimated: usize::from(estimated), estimated_absolute: if estimated { error } else { 0.0 } };
        total.add(counts)?; values.push(counts);
    }
    checkpoint()?; Ok((values, total))
}

/// Average ranks for exact ties. Sorting only bounded finite scalar/index
/// pairs retains no document text. +/-0 compare as one tied value.
fn ranks<E: From<AssayError>>(count: usize, value: impl Fn(usize) -> f64,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Vec<f64>, E> {
    let mut ordered = reserved(count)?;
    for index in 0..count {
        if index % 64 == 0 { checkpoint()?; }
        let value = value(index); if !value.is_finite() { return Err(AssayError::Observation.into()); }
        ordered.push((value, index));
    }
    checkpoint()?;
    ordered.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    checkpoint()?;
    let mut ranks = reserved(count)?; ranks.resize(count, 0.0);
    let mut start = 0;
    while start < count {
        checkpoint()?;
        let mut end = start + 1;
        while end < count && ordered[end].0 == ordered[start].0 {
            if end % 64 == 0 { checkpoint()?; } end += 1;
        }
        let rank = (start as f64 + end as f64 - 1.0) / 2.0;
        for (offset, &(_, index)) in ordered[start..end].iter().enumerate() {
            if offset % 64 == 0 { checkpoint()?; } ranks[index] = rank;
        }
        start = end;
    }
    Ok(ranks)
}
fn spearman<E: From<AssayError>>(replay: &NativeReplay, axis: usize,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(Option<f64>, SpearmanStatus), E> {
    let count = replay.rows.len();
    if count < 2 { return Err(AssayError::Scope.into()); }
    let predicted = ranks(count, |index| replay.rows[index].means[axis], checkpoint)?;
    let gold = ranks(count, |index| f64::from(replay.rows[index].gold[axis].value_milli), checkpoint)?;
    let center = (count as f64 - 1.0) / 2.0;
    let (mut covariance, mut prediction_variance, mut gold_variance) = (0.0, 0.0, 0.0);
    for (index, (a, b)) in predicted.iter().zip(&gold).enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        let a = a - center; let b = b - center;
        covariance += a * b; prediction_variance += a * a; gold_variance += b * b;
    }
    let status = match (prediction_variance == 0.0, gold_variance == 0.0) {
        (true, true) => SpearmanStatus::BothConstant, (true, false) => SpearmanStatus::ConstantPrediction,
        (false, true) => SpearmanStatus::ConstantGold, (false, false) => SpearmanStatus::Measured,
    };
    let correlation = if status == SpearmanStatus::Measured {
        let value = covariance / (prediction_variance * gold_variance).sqrt();
        if !value.is_finite() { return Err(AssayError::Observation.into()); } Some(value.clamp(-1.0, 1.0))
    } else { None };
    checkpoint()?; Ok((correlation, status))
}

pub(super) fn measure<E: From<AssayError>>(candidate: &NativeReplay, baseline: Option<&NativeReplay>,
    options: EvalOptions, limits: Limits, checkpoint: &mut impl FnMut() -> Result<(), E>)
    -> Result<(Measurement, Option<Measurement>, Option<PairedSummary>), E> {
    let documents = candidate.rows.len(); let axes = &candidate.manifest.axes;
    validate_replay(options, documents, axes.len(), baseline.is_some(), limits)?;
    if baseline.is_some_and(|prior| prior.rows.len() != documents || &prior.manifest.axes != axes) {
        return Err(AssayError::Scope.into());
    }
    checkpoint()?;
    if documents < options.minimum_samples {
        let insufficient = || Measurement::InsufficientData { documents, minimum_documents: options.minimum_samples };
        return Ok((insufficient(), baseline.map(|_| insufficient()), None));
    }
    let mut current_axes = reserved(axes.len())?; let mut prior_axes = reserved(axes.len())?;
    let mut changes = reserved(axes.len())?;
    for (axis_index, &axis) in axes.iter().enumerate() {
        checkpoint()?;
        let (current, current_total) = axis_counts(candidate, axis_index, checkpoint)?;
        let prior = baseline.map(|replay| axis_counts(replay, axis_index, checkpoint)).transpose()?;
        let current_rank = spearman(candidate, axis_index, checkpoint)?;
        let prior_rank = baseline.map(|replay| spearman(replay, axis_index, checkpoint)).transpose()?;
        // The same seed and source count make draws identical across axes as
        // well as across paired arms; axes never inflate the sample minimum.
        let intervals = document_bootstrap::measure_with_checkpoint(&current,
            prior.as_ref().map(|(rows, _)| rows.as_slice()), options, limits, Counts::add, Counts::values, checkpoint)?;
        let a = current_total.quality(axis, current_rank, intervals.candidate.into())?;
        if let Some((_, prior_total)) = prior {
            let b = prior_total.quality(axis, prior_rank.ok_or(AssayError::Scope)?,
                intervals.baseline.ok_or(AssayError::Scope)?.into())?;
            changes.push(AxisChange { axis, mae_change: a.mae - b.mae, rmse_change: a.rmse - b.rmse,
                spearman_change: a.spearman.zip(b.spearman).map(|(a,b)| a-b),
                estimated_coverage_change: a.estimated_coverage - b.estimated_coverage,
                estimated_mae_change: a.estimated_mae.zip(b.estimated_mae).map(|(a,b)| a-b),
                intervals: intervals.changes.ok_or(AssayError::Scope)?.into() });
            prior_axes.push(b);
        }
        current_axes.push(a);
    }
    let current = Measurement::Measured { documents, axes: current_axes };
    let prior = baseline.map(|_| Measurement::Measured { documents, axes: prior_axes });
    let paired = baseline.map(|_| PairedSummary { documents, axes: changes });
    checkpoint()?; Ok((current, prior, paired))
}

#[cfg(test)] mod tests;
