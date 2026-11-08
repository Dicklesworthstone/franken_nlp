//! Claim-weighted metrics, with whole-source paired resampling.
use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};
use crate::{execution_identity::Sha256Digest, tasks::judge::FaithfulnessAbstention};
use super::{AssayError, Limits, RELATIONS, NativeReplay, reserved};
use crate::assay::{ConfidenceInterval, document_bootstrap,
    label_metrics::{self, LabelMode, LabelMetrics, LabelSetMetrics, Observation}};
pub use crate::assay::document_bootstrap::EvalOptions;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Abstentions {
    pub ambiguous_distribution: usize,
    pub insufficient_evidence: usize,
    pub conflicting_evidence: usize,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Quality {
    pub source_clusters: usize,
    pub labels: LabelSetMetrics,
    /// Half the sum of squared one-hot errors for the full-source three-way
    /// candidate weights. This is descriptive, uncalibrated model agreement.
    pub brier: f64,
    pub selective_risk: Option<f64>,
    pub abstentions: Abstentions,
}
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualityIntervals {
    pub accuracy: Option<ConfidenceInterval>,
    pub macro_f1: Option<ConfidenceInterval>,
    pub brier: Option<ConfidenceInterval>,
    pub coverage: Option<ConfidenceInterval>,
    pub selective_risk: Option<ConfidenceInterval>,
}
impl From<[Option<ConfidenceInterval>; 5]> for QualityIntervals {
    fn from(values: [Option<ConfidenceInterval>; 5]) -> Self {
        let [accuracy, macro_f1, brier, coverage, selective_risk] = values;
        Self { accuracy, macro_f1, brier, coverage, selective_risk }
    }
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum Measurement {
    Measured { quality: Quality, intervals: QualityIntervals },
    InsufficientData { claims: usize, source_clusters: usize, minimum_sources: usize },
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedSummary {
    pub source_clusters: usize,
    /// Every change and interval is candidate minus baseline.
    pub accuracy_change: f64,
    pub macro_f1_change: Option<f64>,
    pub brier_change: f64,
    pub coverage_change: f64,
    pub selective_risk_change: Option<f64>,
    pub intervals: QualityIntervals,
}

#[derive(Clone, Copy, Default)]
struct Counts { confusion: [[usize; 4]; 3], loss: f64, abstentions: [usize; 3] }
impl Counts {
    fn add(&mut self, other: Self) -> Result<(), AssayError> {
        // Resampling can repeat a large source cluster many times. Reserve
        // arithmetic headroom for all fixed three-label numerators, including
        // on 32-bit targets, before changing any accumulated cell.
        let rows = self.confusion.iter().flatten().chain(other.confusion.iter().flatten())
            .try_fold(0_usize, |sum, &count| sum.checked_add(count).ok_or(AssayError::InputLimit))?;
        if rows > usize::MAX / 4 { return Err(AssayError::InputLimit); }
        for (target, source) in self.confusion.iter_mut().flatten().zip(other.confusion.iter().flatten()) {
            *target = target.checked_add(*source).ok_or(AssayError::InputLimit)?;
        }
        for (target, source) in self.abstentions.iter_mut().zip(other.abstentions) {
            *target = target.checked_add(source).ok_or(AssayError::InputLimit)?;
        }
        self.loss += other.loss;
        if !self.loss.is_finite() { return Err(AssayError::Observation); }
        Ok(())
    }
    fn rows(self) -> usize { self.confusion.iter().flatten().sum() }
    fn abstained(self) -> usize { self.confusion.iter().map(|row| row[3]).sum() }
    fn exact(self) -> usize { (0..3).map(|i| self.confusion[i][i]).sum() }
    fn macro_f1(self) -> Option<f64> {
        (0..3).try_fold(0.0, |sum, i| {
            let support: usize = self.confusion[i].iter().sum();
            let predicted: usize = self.confusion.iter().map(|row| row[i]).sum();
            ratio(2 * self.confusion[i][i], support + predicted).map(|value| sum + value / 3.0)
        })
    }
    fn values(self) -> [Option<f64>; 5] {
        let rows = self.rows(); let decided = rows - self.abstained();
        [ratio(self.exact(), rows), self.macro_f1(), (rows != 0).then(|| self.loss / rows as f64),
            ratio(decided, rows), ratio(decided - self.exact(), decided)]
    }
    fn quality(self, sources: usize) -> Result<Quality, AssayError> {
        let rows = self.rows(); let abstained = self.abstained(); let decided = rows - abstained;
        if rows == 0 || sources == 0 { return Err(AssayError::Scope); }
        let mut labels = reserved(3)?;
        for (i, label) in RELATIONS.iter().enumerate() {
            let support = self.confusion[i].iter().sum();
            let predicted: usize = self.confusion.iter().map(|row| row[i]).sum();
            let tp = self.confusion[i][i]; let fp = predicted - tp;
            let ap = self.confusion[i][3]; let missed = support - tp; let fn_count = missed - ap;
            labels.push(LabelMetrics { label: (*label).to_owned(), support, predicted, decided, abstained,
                true_positive: tp, false_positive: fp, true_negative: decided - tp - fp - fn_count,
                false_negative: fn_count, abstained_positive: ap, abstained_negative: abstained - ap,
                precision: ratio(tp, predicted), recall: ratio(tp, support), f1: ratio(2 * tp, support + predicted) });
        }
        let exact = self.exact(); let wrong_decided = decided - exact;
        let labels = LabelSetMetrics { mode: LabelMode::Multiclass, rows, label_count: 3,
            decided_rows: decided, abstained_rows: abstained, partially_decided_rows: 0,
            exact_match_accuracy: exact as f64 / rows as f64, decided_exact_match_accuracy: ratio(exact, decided),
            hamming_loss: (2 * wrong_decided + 3 * abstained) as f64 / (3 * rows) as f64,
            label_decision_coverage: decided as f64 / rows as f64,
            micro_precision: ratio(exact, decided), micro_recall: ratio(exact, rows),
            micro_f1: ratio(2 * exact, rows + decided), macro_f1: self.macro_f1(), labels };
        let [ambiguous_distribution, insufficient_evidence, conflicting_evidence] = self.abstentions;
        Ok(Quality { source_clusters: sources, labels, brier: self.loss / rows as f64,
            selective_risk: ratio(wrong_decided, decided), abstentions: Abstentions {
                ambiguous_distribution, insufficient_evidence, conflicting_evidence } })
    }
}
fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}
fn clusters<E: From<AssayError>>(replay: &NativeReplay, checkpoint: &mut impl FnMut() -> Result<(), E>)
    -> Result<(Vec<Sha256Digest>, Vec<Counts>), E> {
    let mut groups: BTreeMap<Sha256Digest, Counts> = BTreeMap::new();
    for (index, row) in replay.rows.iter().enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        if row.gold >= 3 || row.top >= 3 || row.predicted.is_some_and(|p| p >= 3 || p != row.top)
            || row.predicted.is_some() == row.abstention.is_some() { return Err(AssayError::Observation.into()); }
        let positive = std::array::from_fn::<_, 3, _>(|i| i == row.gold);
        let predicted = std::array::from_fn::<_, 3, _>(|i| row.predicted.map(|p| p == i));
        let loss = label_metrics::brier_loss(LabelMode::Multiclass, Observation {
            positive: &positive, probability: &row.probability, predicted: &predicted, top_label: Some(row.top) })?;
        let mut counts = Counts { loss, ..Counts::default() };
        counts.confusion[row.gold][row.predicted.unwrap_or(3)] = 1;
        if let Some(reason) = row.abstention { counts.abstentions[match reason {
            FaithfulnessAbstention::AmbiguousDistribution => 0,
            FaithfulnessAbstention::InsufficientEvidence => 1,
            FaithfulnessAbstention::ConflictingEvidence => 2 }] = 1; }
        groups.entry(row.source).or_default().add(counts)?;
    }
    let mut sources = reserved(groups.len())?; let mut values = reserved(groups.len())?;
    for (source, counts) in groups { sources.push(source); values.push(counts); }
    checkpoint()?; Ok((sources, values))
}
fn total<E: From<AssayError>>(clusters: &[Counts], checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Counts, E> {
    let mut total = Counts::default();
    for (index, &row) in clusters.iter().enumerate() { if index % 64 == 0 { checkpoint()?; } total.add(row)?; }
    Ok(total)
}
pub(super) fn measure<E: From<AssayError>>(candidate: &NativeReplay, baseline: Option<&NativeReplay>,
    options: EvalOptions, limits: Limits, checkpoint: &mut impl FnMut() -> Result<(), E>)
    -> Result<(Measurement, Option<Measurement>, Option<PairedSummary>), E> {
    let (sources, current) = clusters(candidate, checkpoint)?;
    let prior = baseline.map(|replay| clusters(replay, checkpoint)).transpose()?;
    if let Some((prior_sources, _)) = &prior {
        if &sources != prior_sources { return Err(AssayError::Scope.into()); }
        options.validate_paired_rows(sources.len(), limits)?;
    } else { options.validate_rows(sources.len(), limits)?; }
    if sources.len() < options.minimum_samples {
        let insufficient = || Measurement::InsufficientData { claims: candidate.rows.len(),
            source_clusters: sources.len(), minimum_sources: options.minimum_samples };
        return Ok((insufficient(), prior.as_ref().map(|_| insufficient()), None));
    }
    let current_total = total(&current, checkpoint)?;
    let prior_total = prior.as_ref().map(|(_, rows)| total(rows, checkpoint)).transpose()?;
    let intervals = document_bootstrap::measure_with_checkpoint(&current,
        prior.as_ref().map(|(_, rows)| rows.as_slice()), options, limits, Counts::add, Counts::values, checkpoint)?;
    let paired = prior_total.zip(intervals.changes).map(|(prior, intervals)| {
        let a = current_total.values(); let b = prior.values();
        Ok::<_, AssayError>(PairedSummary { source_clusters: sources.len(),
            accuracy_change: a[0].ok_or(AssayError::Scope)? - b[0].ok_or(AssayError::Scope)?,
            macro_f1_change: a[1].zip(b[1]).map(|(a,b)| a-b),
            brier_change: a[2].ok_or(AssayError::Scope)? - b[2].ok_or(AssayError::Scope)?,
            coverage_change: a[3].ok_or(AssayError::Scope)? - b[3].ok_or(AssayError::Scope)?,
            selective_risk_change: a[4].zip(b[4]).map(|(a,b)| a-b), intervals: intervals.into() })
    }).transpose()?;
    let current = Measurement::Measured { quality: current_total.quality(sources.len())?, intervals: intervals.candidate.into() };
    let prior = prior_total.zip(intervals.baseline).map(|(prior, intervals)|
        Ok::<_, AssayError>(Measurement::Measured { quality: prior.quality(sources.len())?, intervals: intervals.into() })).transpose()?;
    checkpoint()?; Ok((current, prior, paired))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repeated_source_count_overflow_refuses_before_mutating_the_cluster() {
        let mut accumulated = Counts::default(); accumulated.confusion[0][0] = usize::MAX / 4;
        let mut one = Counts::default(); one.confusion[0][0] = 1;
        assert_eq!(accumulated.add(one), Err(AssayError::InputLimit));
        assert_eq!(accumulated.confusion[0][0], usize::MAX / 4);
    }
}
