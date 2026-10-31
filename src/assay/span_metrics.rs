//! Exact-boundary span scoring and document-level paired uncertainty.
//!
//! Ambiguous proposals have no selected span and earn no match. Matching is
//! one-to-one even for duplicate predictions; relaxed scoring ignores only
//! the entity type, never the boundaries. No probability is manufactured.

use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};
use super::{AssayError, BootstrapOptions, ConfidenceInterval, Limits, Split};

const MAX_TYPES: usize = 8;
const MAX_SPANS: usize = 4096;
const MAX_OCCURRENCES: usize = 65_536;
const MAX_LANGUAGES: usize = 128;

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
            || !(0.0..1.0).contains(&self.bootstrap.confidence_level)
            || self.bootstrap.confidence_level == 0.0 {
            return Err(AssayError::Options);
        }
        Ok(())
    }
    pub fn validate_rows(self, rows: usize, limits: Limits) -> Result<(), AssayError> {
        self.validate_arms(rows, 1, limits)
    }
    pub fn validate_paired_rows(self, rows: usize, limits: Limits) -> Result<(), AssayError> {
        self.validate_arms(rows, 2, limits)
    }
    fn validate_arms(self, rows: usize, arms: u64, limits: Limits) -> Result<(), AssayError> {
        self.validate(limits)?;
        if rows == 0 || rows > limits.max_rows
            || (rows as u64).checked_mul(self.bootstrap.resamples as u64)
                .and_then(|draws| draws.checked_mul(arms))
                .is_none_or(|draws| draws > limits.max_bootstrap_draws) {
            return Err(AssayError::InputLimit);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Span {
    pub kind: usize,
    pub byte_start: usize,
    pub byte_end: usize,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Prediction {
    pub kind: usize,
    /// None is unresolved ambiguity, not an asserted interval.
    pub span: Option<(usize, usize)>,
    pub candidate_spans: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Counts { tp: usize, fp: usize, missed: usize }
impl Counts {
    fn add(&mut self, other: Self) -> Result<(), AssayError> {
        self.tp = sum(self.tp, other.tp)?;
        self.fp = sum(self.fp, other.fp)?;
        self.missed = sum(self.missed, other.missed)?;
        Ok(())
    }
    fn scores(self) -> SpanScores {
        // The admitted corpus has at most 100,000 * 4,096 spans per side.
        SpanScores { true_positive: self.tp, false_positive: self.fp, false_negative: self.missed,
            support: self.tp + self.missed, predicted: self.tp + self.fp,
            precision: ratio(self.tp, self.tp + self.fp),
            recall: ratio(self.tp, self.tp + self.missed),
            f1: ratio(2 * self.tp, 2 * self.tp + self.fp + self.missed) }
    }
}

/// Fixed-size summaries retain neither mention text nor candidate intervals.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SpanCounts {
    totals: Totals,
    by_type: [Counts; MAX_TYPES],
    anchored: usize,
    ambiguous: usize,
    ambiguous_candidates: usize,
    duplicate_anchored: usize,
}
impl SpanCounts {
    pub(crate) fn observe_with_checkpoint<E: From<AssayError>>(gold: &[Span], predictions: &[Prediction],
        kinds: usize, checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Self, E> {
        checkpoint()?;
        if !(1..=MAX_TYPES).contains(&kinds) || gold.len() > MAX_SPANS || predictions.len() > MAX_SPANS {
            return Err(AssayError::Observation.into());
        }
        let mut truth = reserved(gold.len())?;
        let mut anchored = reserved(predictions.len())?;
        let mut value = Self::default();
        value.totals.rows = 1;
        for (index, span) in gold.iter().enumerate() {
            if index % 64 == 0 { checkpoint()?; }
            if span.kind >= kinds || span.byte_start >= span.byte_end { return Err(AssayError::Observation.into()); }
            truth.push((span.byte_start, span.byte_end, span.kind));
            value.by_type[span.kind].missed += 1;
        }
        truth.sort_unstable();
        for pair in truth.windows(2) {
            if pair[0].0 == pair[1].0 && pair[0].1 == pair[1].1 { return Err(AssayError::Observation.into()); }
        }
        let mut occurrences = 0;
        for (index, prediction) in predictions.iter().enumerate() {
            if index % 64 == 0 { checkpoint()?; }
            if prediction.kind >= kinds { return Err(AssayError::Observation.into()); }
            occurrences = sum(occurrences, prediction.candidate_spans)?;
            if occurrences > MAX_OCCURRENCES { return Err(AssayError::Observation.into()); }
            value.by_type[prediction.kind].fp += 1;
            match prediction.span {
                Some((start, end)) if start < end && prediction.candidate_spans == 1 => {
                    anchored.push((start, end, prediction.kind)); value.anchored += 1;
                }
                None if prediction.candidate_spans >= 2 => {
                    value.ambiguous += 1;
                    value.ambiguous_candidates = sum(value.ambiguous_candidates, prediction.candidate_spans)?;
                }
                _ => return Err(AssayError::Observation.into()),
            }
        }
        anchored.sort_unstable();
        checkpoint()?;
        let (mut gold_index, mut prediction_index, mut relaxed_tp) = (0, 0, 0);
        while prediction_index < anchored.len() {
            if prediction_index % 64 == 0 { checkpoint()?; }
            let first = anchored[prediction_index];
            let interval = (first.0, first.1);
            let mut end = prediction_index + 1;
            while end < anchored.len() && (anchored[end].0, anchored[end].1) == interval {
                if end % 64 == 0 { checkpoint()?; }
                if anchored[end] == anchored[end - 1] { value.duplicate_anchored += 1; }
                end += 1;
            }
            while gold_index < truth.len() && (truth[gold_index].0, truth[gold_index].1) < interval {
                if gold_index % 64 == 0 { checkpoint()?; }
                gold_index += 1;
            }
            if let Some(gold) = truth.get(gold_index).filter(|span| (span.0, span.1) == interval) {
                relaxed_tp += 1;
                // At most one gold entity owns a byte interval. A group of
                // repeated or conflicting predictions cannot earn extra TP.
                if anchored[prediction_index..end].binary_search(&(gold.0, gold.1, gold.2)).is_ok() {
                    let counts = &mut value.by_type[gold.2];
                    counts.tp += 1; counts.fp -= 1; counts.missed -= 1;
                }
            }
            prediction_index = end;
        }
        for counts in &value.by_type[..kinds] { value.totals.exact.add(*counts)?; }
        value.totals.relaxed = Counts { tp: relaxed_tp,
            fp: predictions.len() - relaxed_tp, missed: gold.len() - relaxed_tp };
        value.totals.exact_documents = usize::from(value.totals.exact.fp == 0 && value.totals.exact.missed == 0);
        checkpoint()?; Ok(value)
    }

    fn add(&mut self, other: &Self) -> Result<(), AssayError> {
        self.totals.add(other.totals)?;
        for (current, next) in self.by_type.iter_mut().zip(other.by_type) { current.add(next)?; }
        self.anchored = sum(self.anchored, other.anchored)?;
        self.ambiguous = sum(self.ambiguous, other.ambiguous)?;
        self.ambiguous_candidates = sum(self.ambiguous_candidates, other.ambiguous_candidates)?;
        self.duplicate_anchored = sum(self.duplicate_anchored, other.duplicate_anchored)?;
        Ok(())
    }
    fn anchoring(&self) -> AnchoringMetrics {
        let proposed = self.anchored + self.ambiguous;
        AnchoringMetrics { anchored_predictions: self.anchored, ambiguous_predictions: self.ambiguous,
            ambiguous_candidate_spans: self.ambiguous_candidates, duplicate_anchored_predictions: self.duplicate_anchored,
            anchored_fraction: ratio(self.anchored, proposed), ambiguous_fraction: ratio(self.ambiguous, proposed) }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Totals { rows: usize, exact: Counts, relaxed: Counts, exact_documents: usize }
impl Totals {
    fn add(&mut self, other: Self) -> Result<(), AssayError> {
        self.rows = sum(self.rows, other.rows)?;
        self.exact_documents = sum(self.exact_documents, other.exact_documents)?;
        self.exact.add(other.exact)?; self.relaxed.add(other.relaxed)
    }
    fn accuracy(self) -> f64 { self.exact_documents as f64 / self.rows as f64 }
}
pub(crate) struct Observation<'a> { pub language: &'a str, pub counts: &'a SpanCounts }

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpanScores {
    pub support: usize,
    pub predicted: usize,
    pub true_positive: usize,
    pub false_positive: usize,
    pub false_negative: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    /// Undefined when both gold and predicted span counts are zero.
    pub f1: Option<f64>,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpanIntervals {
    /// An interval is absent if any bootstrap replicate has zero denominator.
    pub precision: Option<ConfidenceInterval>,
    pub recall: Option<ConfidenceInterval>,
    pub f1: Option<ConfidenceInterval>,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnchoringMetrics {
    pub anchored_predictions: usize,
    pub ambiguous_predictions: usize,
    pub ambiguous_candidate_spans: usize,
    /// Extra anchored proposals with the same interval AND entity type.
    pub duplicate_anchored_predictions: usize,
    pub anchored_fraction: Option<f64>,
    pub ambiguous_fraction: Option<f64>,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TypeMetrics { pub entity_type: String, pub exact: SpanScores }
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageMetrics {
    pub language: String,
    pub rows: usize,
    pub exact: SpanScores,
    pub type_relaxed: SpanScores,
    pub document_accuracy: f64,
    pub anchoring: AnchoringMetrics,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpanMetrics {
    pub rows: usize,
    pub exact: SpanScores,
    pub type_relaxed: SpanScores,
    pub document_accuracy: f64,
    pub anchoring: AnchoringMetrics,
    pub exact_interval: SpanIntervals,
    pub type_relaxed_interval: SpanIntervals,
    pub document_accuracy_interval: ConfidenceInterval,
    /// Declared types, including those absent from the locked-test corpus.
    pub by_type: Vec<TypeMetrics>,
    /// Caller-declared row language; this is not language detection evidence.
    pub by_language: Vec<LanguageMetrics>,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum Measurement {
    InsufficientData { split: Split, rows: usize, minimum_samples: usize },
    Measured { metrics: SpanMetrics },
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedSummary {
    pub rows: usize,
    /// Whole-document exact FP + FN determines better/worse/equal counts.
    pub mean_exact_error_change: f64,
    pub candidate_better: usize,
    pub candidate_worse: usize,
    pub equal: usize,
    /// All changes are candidate minus baseline, on identical document draws.
    pub exact_f1_change: Option<f64>,
    pub type_relaxed_f1_change: Option<f64>,
    pub exact_recall_change: Option<f64>,
    pub document_accuracy_change: f64,
    pub exact_f1_change_interval: Option<ConfidenceInterval>,
    pub type_relaxed_f1_change_interval: Option<ConfidenceInterval>,
    pub exact_recall_change_interval: Option<ConfidenceInterval>,
    pub document_accuracy_change_interval: ConfidenceInterval,
}

pub(crate) fn measure_with_checkpoint<E: From<AssayError>>(types: &[&str], rows: &[Observation<'_>],
    options: EvalOptions, limits: Limits, checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Measurement, E> {
    options.validate_rows(rows.len(), limits)?;
    validate_rows(types, rows, checkpoint)?;
    if rows.len() < options.minimum_samples { return Ok(insufficient(rows.len(), options)); }
    let (current, _, _) = bootstrap(rows, None, options, checkpoint)?;
    Ok(Measurement::Measured { metrics: aggregate(types, rows, current, checkpoint)? })
}

/// One bootstrap traversal produces both marginal intervals and paired
/// changes. Sampling whole documents preserves within-document dependence.
pub(crate) fn paired_with_checkpoint<E: From<AssayError>>(types: &[&str], candidate: &[Observation<'_>],
    baseline: &[Observation<'_>], options: EvalOptions, limits: Limits,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(Measurement, Measurement, Option<PairedSummary>), E> {
    options.validate_paired_rows(candidate.len(), limits)?;
    validate_rows(types, candidate, checkpoint)?; validate_rows(types, baseline, checkpoint)?;
    if candidate.len() != baseline.len() { return Err(AssayError::Scope.into()); }
    for (index, (a, b)) in candidate.iter().zip(baseline).enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        if a.language != b.language || a.counts.totals.exact.tp + a.counts.totals.exact.missed
            != b.counts.totals.exact.tp + b.counts.totals.exact.missed {
            return Err(AssayError::Scope.into());
        }
    }
    if candidate.len() < options.minimum_samples {
        return Ok((insufficient(candidate.len(), options), insufficient(candidate.len(), options), None));
    }
    let (current, prior, changes) = bootstrap(candidate, Some(baseline), options, checkpoint)?;
    let current = aggregate(types, candidate, current, checkpoint)?;
    let prior = aggregate(types, baseline, prior.ok_or(AssayError::Scope)?, checkpoint)?;
    let changes = changes.ok_or(AssayError::Scope)?;
    let mut paired = PairedSummary { rows: candidate.len(), mean_exact_error_change: 0.0,
        candidate_better: 0, candidate_worse: 0, equal: 0,
        exact_f1_change: difference(current.exact.f1, prior.exact.f1),
        type_relaxed_f1_change: difference(current.type_relaxed.f1, prior.type_relaxed.f1),
        exact_recall_change: difference(current.exact.recall, prior.exact.recall),
        document_accuracy_change: current.document_accuracy - prior.document_accuracy,
        exact_f1_change_interval: changes.exact_f1, type_relaxed_f1_change_interval: changes.relaxed_f1,
        exact_recall_change_interval: changes.exact_recall,
        document_accuracy_change_interval: changes.accuracy.ok_or(AssayError::Observation)? };
    for (index, (a, b)) in candidate.iter().zip(baseline).enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        let a = a.counts.totals.exact; let b = b.counts.totals.exact;
        let current_errors = a.fp + a.missed; let prior_errors = b.fp + b.missed;
        paired.mean_exact_error_change += current_errors as f64 - prior_errors as f64;
        match current_errors.cmp(&prior_errors) {
            std::cmp::Ordering::Less => paired.candidate_better += 1,
            std::cmp::Ordering::Greater => paired.candidate_worse += 1,
            std::cmp::Ordering::Equal => paired.equal += 1,
        }
    }
    paired.mean_exact_error_change /= candidate.len() as f64;
    checkpoint()?;
    Ok((Measurement::Measured { metrics: current }, Measurement::Measured { metrics: prior }, Some(paired)))
}

fn validate_rows<E: From<AssayError>>(types: &[&str], rows: &[Observation<'_>],
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), E> {
    if types.is_empty() || types.len() > MAX_TYPES { return Err(AssayError::Observation.into()); }
    for (index, kind) in types.iter().enumerate() {
        if kind.is_empty() || kind.len() > 256 || types[..index].contains(kind) { return Err(AssayError::Observation.into()); }
    }
    let mut languages = std::collections::BTreeSet::new();
    for (index, row) in rows.iter().enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        if row.language.is_empty() || row.language.len() > 256 || row.counts.totals.rows != 1
            || row.counts.by_type[types.len()..].iter().any(|counts| *counts != Counts::default()) {
            return Err(AssayError::Observation.into());
        }
        languages.insert(row.language);
        if languages.len() > MAX_LANGUAGES { return Err(AssayError::InputLimit.into()); }
    }
    checkpoint()?; Ok(())
}
fn insufficient(rows: usize, options: EvalOptions) -> Measurement {
    Measurement::InsufficientData { split: Split::LockedTest, rows, minimum_samples: options.minimum_samples }
}
fn aggregate<E: From<AssayError>>(types: &[&str], rows: &[Observation<'_>], intervals: Intervals,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<SpanMetrics, E> {
    let mut all = SpanCounts::default();
    let mut languages = BTreeMap::<&str, SpanCounts>::new();
    for (index, row) in rows.iter().enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        all.add(row.counts)?;
        languages.entry(row.language).or_default().add(row.counts)?;
    }
    let mut by_type = reserved(types.len())?;
    for (index, kind) in types.iter().enumerate() {
        by_type.push(TypeMetrics { entity_type: (*kind).to_owned(), exact: all.by_type[index].scores() });
    }
    let mut by_language = reserved(languages.len())?;
    for (language, counts) in languages {
        checkpoint()?;
        by_language.push(LanguageMetrics { language: language.to_owned(), rows: counts.totals.rows,
            exact: counts.totals.exact.scores(), type_relaxed: counts.totals.relaxed.scores(),
            document_accuracy: counts.totals.accuracy(), anchoring: counts.anchoring() });
    }
    Ok(SpanMetrics { rows: rows.len(), exact: all.totals.exact.scores(), type_relaxed: all.totals.relaxed.scores(),
        document_accuracy: all.totals.accuracy(), anchoring: all.anchoring(),
        exact_interval: intervals.exact, type_relaxed_interval: intervals.relaxed,
        document_accuracy_interval: intervals.accuracy.ok_or(AssayError::Observation)?, by_type, by_language })
}

struct Series { values: Vec<f64>, defined: bool }
impl Series {
    fn new(count: usize) -> Result<Self, AssayError> { Ok(Self { values: reserved(count)?, defined: true }) }
    fn push(&mut self, value: Option<f64>) {
        if let Some(value) = value { self.values.push(value); } else { self.defined = false; }
    }
    fn finish<E>(&mut self, options: EvalOptions, checkpoint: &mut impl FnMut() -> Result<(), E>)
        -> Result<Option<ConfidenceInterval>, E> {
        checkpoint()?;
        if !self.defined || self.values.is_empty() { return Ok(None); }
        self.values.sort_unstable_by(f64::total_cmp);
        let tail = (1.0 - options.bootstrap.confidence_level) / 2.0;
        let last = self.values.len() - 1;
        let result = ConfidenceInterval { lower: self.values[(tail * last as f64).floor() as usize],
            upper: self.values[((1.0 - tail) * last as f64).ceil() as usize],
            confidence_level: options.bootstrap.confidence_level, resamples: options.bootstrap.resamples };
        checkpoint()?; Ok(Some(result))
    }
}
struct ScoreSamples { precision: Series, recall: Series, f1: Series }
impl ScoreSamples {
    fn new(count: usize) -> Result<Self, AssayError> {
        Ok(Self { precision: Series::new(count)?, recall: Series::new(count)?, f1: Series::new(count)? })
    }
    fn push(&mut self, counts: Counts) {
        let scores = counts.scores();
        self.precision.push(scores.precision); self.recall.push(scores.recall); self.f1.push(scores.f1);
    }
    fn finish<E>(&mut self, options: EvalOptions, checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<SpanIntervals, E> {
        Ok(SpanIntervals { precision: self.precision.finish(options, checkpoint)?,
            recall: self.recall.finish(options, checkpoint)?, f1: self.f1.finish(options, checkpoint)? })
    }
}
struct Samples { exact: ScoreSamples, relaxed: ScoreSamples, accuracy: Series }
struct Intervals { exact: SpanIntervals, relaxed: SpanIntervals, accuracy: Option<ConfidenceInterval> }
impl Samples {
    fn new(count: usize) -> Result<Self, AssayError> {
        Ok(Self { exact: ScoreSamples::new(count)?, relaxed: ScoreSamples::new(count)?, accuracy: Series::new(count)? })
    }
    fn push(&mut self, totals: Totals) {
        self.exact.push(totals.exact); self.relaxed.push(totals.relaxed); self.accuracy.push(Some(totals.accuracy()));
    }
    fn finish<E>(&mut self, options: EvalOptions, checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Intervals, E> {
        Ok(Intervals { exact: self.exact.finish(options, checkpoint)?, relaxed: self.relaxed.finish(options, checkpoint)?,
            accuracy: self.accuracy.finish(options, checkpoint)? })
    }
}
struct ChangeSamples { exact_f1: Series, relaxed_f1: Series, exact_recall: Series, accuracy: Series }
struct ChangeIntervals { exact_f1: Option<ConfidenceInterval>, relaxed_f1: Option<ConfidenceInterval>,
    exact_recall: Option<ConfidenceInterval>, accuracy: Option<ConfidenceInterval> }
impl ChangeSamples {
    fn new(count: usize) -> Result<Self, AssayError> {
        Ok(Self { exact_f1: Series::new(count)?, relaxed_f1: Series::new(count)?,
            exact_recall: Series::new(count)?, accuracy: Series::new(count)? })
    }
    fn push(&mut self, a: Totals, b: Totals) {
        let current = a.exact.scores(); let prior = b.exact.scores();
        self.exact_f1.push(difference(current.f1, prior.f1));
        self.relaxed_f1.push(difference(a.relaxed.scores().f1, b.relaxed.scores().f1));
        self.exact_recall.push(difference(current.recall, prior.recall));
        self.accuracy.push(Some(a.accuracy() - b.accuracy()));
    }
    fn finish<E>(&mut self, options: EvalOptions, checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<ChangeIntervals, E> {
        Ok(ChangeIntervals { exact_f1: self.exact_f1.finish(options, checkpoint)?,
            relaxed_f1: self.relaxed_f1.finish(options, checkpoint)?, exact_recall: self.exact_recall.finish(options, checkpoint)?,
            accuracy: self.accuracy.finish(options, checkpoint)? })
    }
}
fn bootstrap<E: From<AssayError>>(candidate: &[Observation<'_>], baseline: Option<&[Observation<'_>]>,
    options: EvalOptions, checkpoint: &mut impl FnMut() -> Result<(), E>)
    -> Result<(Intervals, Option<Intervals>, Option<ChangeIntervals>), E> {
    let mut current = Samples::new(options.bootstrap.resamples)?;
    let mut prior = baseline.map(|_| Samples::new(options.bootstrap.resamples)).transpose()?;
    let mut changes = baseline.map(|_| ChangeSamples::new(options.bootstrap.resamples)).transpose()?;
    let mut state = options.bootstrap.seed;
    for _ in 0..options.bootstrap.resamples {
        checkpoint()?;
        let (mut a, mut b) = (Totals::default(), Totals::default());
        for draw in 0..candidate.len() {
            if draw % 64 == 0 { checkpoint()?; }
            let index = sampled_index(&mut state, candidate.len(), checkpoint)?;
            a.add(candidate[index].counts.totals)?;
            if let Some(rows) = baseline { b.add(rows[index].counts.totals)?; }
        }
        current.push(a);
        if let Some(prior) = prior.as_mut() { prior.push(b); }
        if let Some(changes) = changes.as_mut() { changes.push(a, b); }
    }
    let current = current.finish(options, checkpoint)?;
    let prior = prior.as_mut().map(|values| values.finish(options, checkpoint)).transpose()?;
    let changes = changes.as_mut().map(|values| values.finish(options, checkpoint)).transpose()?;
    checkpoint()?; Ok((current, prior, changes))
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
fn sum(a: usize, b: usize) -> Result<usize, AssayError> { a.checked_add(b).ok_or(AssayError::InputLimit) }
fn ratio(a: usize, b: usize) -> Option<f64> { (b != 0).then(|| a as f64 / b as f64) }
fn difference(a: Option<f64>, b: Option<f64>) -> Option<f64> { a.zip(b).map(|(a, b)| a - b) }
fn reserved<T>(count: usize) -> Result<Vec<T>, AssayError> {
    let mut result = Vec::new(); result.try_reserve_exact(count).map_err(|_| AssayError::Allocation)?; Ok(result)
}

#[cfg(test)] mod tests;
