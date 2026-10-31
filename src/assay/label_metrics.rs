//! Document-unit statistics for exclusive and independent-label classifiers.
//!
//! Native replay owns the observations. This module only measures borrowed
//! vectors and grants no execution or calibration authority. Bootstrap draws
//! retain each complete document: correlated labels are never IID samples.

use serde::{Deserialize, Serialize};
use super::{AssayError, ConfidenceInterval, EvalOptions, Limits, LockedTestMetrics, Measurement, Split};

const BINS: usize = 10;
const MAX_LABELS: usize = 128;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelMode { Multiclass, Multilabel }

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LabelMetrics {
    pub label: String,
    pub support: usize,
    pub predicted: usize,
    pub decided: usize,
    pub abstained: usize,
    pub true_positive: usize,
    pub false_positive: usize,
    pub true_negative: usize,
    pub false_negative: usize,
    pub abstained_positive: usize,
    pub abstained_negative: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    /// Abstained positive labels are false negatives for recall and F1.
    /// No value is invented when both support and predictions are absent.
    pub f1: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LabelSetMetrics {
    pub mode: LabelMode,
    pub rows: usize,
    pub label_count: usize,
    pub decided_rows: usize,
    pub abstained_rows: usize,
    pub partially_decided_rows: usize,
    /// Complete correct decisions / all documents; abstentions earn no credit.
    pub exact_match_accuracy: f64,
    pub decided_exact_match_accuracy: Option<f64>,
    /// Incorrect or abstained label decisions / all document-label pairs.
    pub hamming_loss: f64,
    pub label_decision_coverage: f64,
    pub micro_precision: Option<f64>,
    pub micro_recall: Option<f64>,
    pub micro_f1: Option<f64>,
    /// Every declared label must have defined F1 for this value to exist.
    pub macro_f1: Option<f64>,
    pub labels: Vec<LabelMetrics>,
}

#[derive(Clone, Copy)]
pub(crate) struct Observation<'a> {
    pub positive: &'a [bool],
    pub probability: &'a [f64],
    pub predicted: &'a [Option<bool>],
    /// The raw-score winner, including when the native task abstained. Never
    /// reconstruct this from rounded/underflowed candidate weights.
    pub top_label: Option<usize>,
}

#[derive(Clone, Copy, Default)]
struct Bin { count: usize, probability: f64, positive: f64 }
#[derive(Clone, Copy)]
struct Document {
    loss: f64,
    accepted: bool,
    error: bool,
    bins: [Bin; BINS],
}
#[derive(Default)]
struct Aggregate { loss: f64, accepted: usize, errors: usize, bins: [Bin; BINS] }
impl Aggregate {
    fn add(&mut self, document: &Document) {
        self.loss += document.loss;
        self.accepted += usize::from(document.accepted);
        self.errors += usize::from(document.accepted && document.error);
        for (target, source) in self.bins.iter_mut().zip(document.bins) {
            target.count += source.count;
            target.probability += source.probability;
            target.positive += source.positive;
        }
    }
    fn ece(&self) -> f64 {
        let count: usize = self.bins.iter().map(|bin| bin.count).sum();
        // Each row contributes one top-label event or every independent label.
        self.bins.iter().map(|bin| (bin.probability - bin.positive).abs()).sum::<f64>() / count as f64
    }
    fn risk(&self) -> Option<f64> { ratio(self.errors, self.accepted) }
}

fn reserved<T>(count: usize) -> Result<Vec<T>, AssayError> {
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| AssayError::Allocation)?;
    Ok(values)
}
fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}
fn bin(probability: f64) -> usize { ((probability * BINS as f64) as usize).min(BINS - 1) }

fn validate(mode: LabelMode, row: Observation<'_>, labels: usize) -> Result<(), AssayError> {
    if !(1..=MAX_LABELS).contains(&labels) || row.positive.len() != labels
        || row.probability.len() != labels || row.predicted.len() != labels
        || row.probability.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
        return Err(AssayError::Observation);
    }
    match mode {
        LabelMode::Multiclass => {
            let top = row.top_label.filter(|&top| top < labels).ok_or(AssayError::Observation)?;
            let sum: f64 = row.probability.iter().sum();
            let completely_abstained = row.predicted.iter().all(Option::is_none);
            let classified = row.predicted.iter().enumerate().all(|(i, p)| *p == Some(i == top));
            if labels < 2 || row.positive.iter().filter(|&&p| p).count() != 1
                || (sum - 1.0).abs() > 32.0 * f64::EPSILON * labels as f64
                || !(completely_abstained || classified) {
                return Err(AssayError::Observation);
            }
        }
        LabelMode::Multilabel => if row.top_label.is_some() { return Err(AssayError::Observation); },
    }
    Ok(())
}

/// Multiclass uses half the sum of squared one-hot errors (range 0..=1),
/// which agrees with binary-event Brier for two complementary candidates.
/// Multilabel uses the mean independent positive-event squared error.
pub(crate) fn brier_loss(mode: LabelMode, row: Observation<'_>) -> Result<f64, AssayError> {
    validate(mode, row, row.positive.len())?;
    let sum = row.probability.iter().zip(row.positive).map(|(&p, &positive)| {
        let error = p - if positive { 1.0 } else { 0.0 };
        error * error
    }).sum::<f64>();
    Ok(match mode { LabelMode::Multiclass => 0.5 * sum, LabelMode::Multilabel => sum / row.positive.len() as f64 })
}

fn document(mode: LabelMode, row: Observation<'_>, threshold: f64) -> Result<Document, AssayError> {
    let loss = brier_loss(mode, row)?;
    let decided = row.predicted.iter().all(Option::is_some);
    let correct = row.predicted.iter().zip(row.positive).all(|(&p, &gold)| p == Some(gold));
    let mut bins = [Bin::default(); BINS];
    let accepted = match mode {
        LabelMode::Multiclass => {
            let top = row.top_label.ok_or(AssayError::Observation)?;
            let p = row.probability[top];
            bins[bin(p)] = Bin { count: 1, probability: p, positive: if row.positive[top] { 1.0 } else { 0.0 } };
            decided && p >= threshold
        }
        LabelMode::Multilabel => {
            for (&p, &gold) in row.probability.iter().zip(row.positive) {
                let bin = &mut bins[bin(p)];
                bin.count += 1; bin.probability += p; bin.positive += if gold { 1.0 } else { 0.0 };
            }
            // A complete set is accepted only if every actual native decision
            // is present and its own selected-side weight clears the threshold.
            decided && row.predicted.iter().zip(row.probability).all(|(&decision, &p)| match decision {
                Some(true) => p >= threshold, Some(false) => 1.0 - p >= threshold, None => false,
            })
        }
    };
    Ok(Document { loss, accepted, error: !correct, bins })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn measure_with_checkpoint<E: From<AssayError>>(mode: LabelMode, labels: &[&str],
    rows: &[Observation<'_>], options: EvalOptions, limits: Limits,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(Measurement, Option<LabelSetMetrics>), E> {
    options.validate_rows(rows.len(), limits)?;
    if !(1..=MAX_LABELS).contains(&labels.len()) || labels.iter().any(|label| label.is_empty())
        || labels.iter().enumerate().any(|(i, label)| labels[..i].contains(label)) {
        return Err(AssayError::Manifest.into());
    }
    let pairs = rows.len().checked_mul(labels.len()).ok_or(AssayError::InputLimit)?;
    let mut documents = reserved(rows.len())?;
    let mut classes = reserved(labels.len())?;
    for &label in labels {
        classes.push(LabelMetrics { label: label.to_owned(), support: 0, predicted: 0, decided: 0, abstained: 0,
            true_positive: 0, false_positive: 0, true_negative: 0, false_negative: 0,
            abstained_positive: 0, abstained_negative: 0, precision: None, recall: None, f1: None });
    }
    let (mut exact, mut decided_rows, mut partial, mut wrong, mut decided_labels) = (0, 0, 0, 0, 0);
    let mut aggregate = Aggregate::default();
    for (index, &row) in rows.iter().enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        validate(mode, row, labels.len())?;
        let document = document(mode, row, options.acceptance_threshold)?;
        let decided = row.predicted.iter().filter(|p| p.is_some()).count();
        decided_rows += usize::from(decided == labels.len());
        partial += usize::from(decided != 0 && decided != labels.len());
        decided_labels += decided;
        exact += usize::from(!document.error);
        for ((class, &gold), &prediction) in classes.iter_mut().zip(row.positive).zip(row.predicted) {
            class.support += usize::from(gold);
            class.predicted += usize::from(prediction == Some(true));
            class.decided += usize::from(prediction.is_some());
            class.abstained += usize::from(prediction.is_none());
            wrong += usize::from(prediction != Some(gold));
            match (gold, prediction) {
                (true, Some(true)) => class.true_positive += 1,
                (false, Some(true)) => class.false_positive += 1,
                (false, Some(false)) => class.true_negative += 1,
                (true, Some(false)) => class.false_negative += 1,
                (true, None) => class.abstained_positive += 1,
                (false, None) => class.abstained_negative += 1,
            }
        }
        aggregate.add(&document); documents.push(document);
    }
    checkpoint()?;
    if rows.len() < options.minimum_samples {
        return Ok((Measurement::InsufficientData { split: Split::LockedTest, rows: rows.len(),
            minimum_samples: options.minimum_samples }, None));
    }
    let (mut tp, mut fp, mut missed) = (0, 0, 0);
    let mut macro_f1 = Some(0.0);
    for class in &mut classes {
        let false_negative = class.false_negative + class.abstained_positive;
        class.precision = ratio(class.true_positive, class.true_positive + class.false_positive);
        class.recall = ratio(class.true_positive, class.support);
        class.f1 = ratio(2 * class.true_positive, 2 * class.true_positive + class.false_positive + false_negative);
        macro_f1 = macro_f1.zip(class.f1).map(|(sum, f1)| sum + f1 / labels.len() as f64);
        tp += class.true_positive; fp += class.false_positive; missed += false_negative;
    }
    let quality = LabelSetMetrics { mode, rows: rows.len(), label_count: labels.len(), decided_rows,
        abstained_rows: rows.len() - decided_rows, partially_decided_rows: partial,
        exact_match_accuracy: exact as f64 / rows.len() as f64,
        decided_exact_match_accuracy: ratio(exact, decided_rows), hamming_loss: wrong as f64 / pairs as f64,
        label_decision_coverage: decided_labels as f64 / pairs as f64,
        micro_precision: ratio(tp, tp + fp), micro_recall: ratio(tp, tp + missed),
        micro_f1: ratio(2 * tp, 2 * tp + fp + missed), macro_f1, labels: classes };
    let (ece_interval, brier_interval, selective_risk_interval) = bootstrap(&documents, options, checkpoint)?;
    let metrics = LockedTestMetrics { rows: rows.len(), ece: aggregate.ece(),
        brier: aggregate.loss / rows.len() as f64, acceptance_threshold: options.acceptance_threshold,
        accepted: aggregate.accepted, abstained: rows.len() - aggregate.accepted, selective_risk: aggregate.risk(),
        ece_interval, brier_interval, selective_risk_interval };
    checkpoint()?;
    Ok((Measurement::Measured { metrics }, Some(quality)))
}

fn bootstrap<E: From<AssayError>>(documents: &[Document], options: EvalOptions,
    checkpoint: &mut impl FnMut() -> Result<(), E>)
    -> Result<(ConfidenceInterval, ConfidenceInterval, Option<ConfidenceInterval>), E> {
    let mut ece = reserved(options.bootstrap.resamples)?;
    let mut brier = reserved(options.bootstrap.resamples)?;
    let mut risk = reserved(options.bootstrap.resamples)?;
    let mut all_risks_defined = true;
    let mut state = options.bootstrap.seed;
    for _ in 0..options.bootstrap.resamples {
        checkpoint()?;
        let mut aggregate = Aggregate::default();
        for draw in 0..documents.len() {
            if draw % 64 == 0 { checkpoint()?; }
            let index = sampled_index(&mut state, documents.len(), checkpoint)?;
            aggregate.add(&documents[index]);
        }
        ece.push(aggregate.ece()); brier.push(aggregate.loss / documents.len() as f64);
        if let Some(value) = aggregate.risk() { risk.push(value); } else { all_risks_defined = false; }
    }
    checkpoint()?;
    let ece = interval(&mut ece, options); checkpoint()?;
    let brier = interval(&mut brier, options); checkpoint()?;
    let risk = all_risks_defined.then(|| interval(&mut risk, options));
    checkpoint()?; Ok((ece, brier, risk))
}

fn interval(values: &mut [f64], options: EvalOptions) -> ConfidenceInterval {
    // At most 10,000 finite values; in-place sorting has no extra allocation.
    values.sort_unstable_by(f64::total_cmp);
    let tail = (1.0 - options.bootstrap.confidence_level) / 2.0;
    let last = values.len() - 1;
    ConfidenceInterval { lower: values[(tail * last as f64).floor() as usize],
        upper: values[((1.0 - tail) * last as f64).ceil() as usize],
        confidence_level: options.bootstrap.confidence_level, resamples: options.bootstrap.resamples }
}

fn sampled_index<E>(state: &mut u64, upper: usize, checkpoint: &mut impl FnMut() -> Result<(), E>)
    -> Result<usize, E> {
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
