//! Exact terminal-field scoring without flattened paths or copied values.
//!
//! Objects align by key and arrays by position. Scalars and empty containers
//! are terminal fields. Missing, null, empty array and empty object differ;
//! exact Decimal equality never rounds a value through IEEE floating point.

use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};
use crate::validation::JsonValue;
use super::{AssayError, ConfidenceInterval, Limits, Split};
pub use super::document_bootstrap::EvalOptions;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct FieldCounts {
    true_positive: usize,
    false_positive: usize,
    false_negative: usize,
}
impl FieldCounts {
    pub(crate) fn observe_with_checkpoint<E: From<AssayError>>(gold: &JsonValue, predicted: &JsonValue,
        max_fields: usize, max_nodes: usize, max_depth: usize,
        checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Self, E> {
        let mut walk = Walk::new(max_fields, max_nodes, max_depth)?;
        checkpoint()?; walk.compare(gold, predicted, 0, checkpoint)?; checkpoint()?;
        Ok(walk.counts)
    }
    pub(crate) fn exact_match(self) -> bool { self.false_positive == 0 && self.false_negative == 0 }
    pub(crate) fn errors(self) -> usize { self.false_positive + self.false_negative }
    fn add(&mut self, other: Self) -> Result<(), AssayError> {
        self.true_positive = sum(self.true_positive, other.true_positive)?;
        self.false_positive = sum(self.false_positive, other.false_positive)?;
        self.false_negative = sum(self.false_negative, other.false_negative)?;
        Ok(())
    }
    fn scores(self) -> FieldScores {
        let support = self.true_positive + self.false_negative;
        let predicted = self.true_positive + self.false_positive;
        FieldScores { support, predicted, true_positive: self.true_positive,
            false_positive: self.false_positive, false_negative: self.false_negative,
            precision: ratio(self.true_positive, predicted), recall: ratio(self.true_positive, support),
            f1: ratio(2 * self.true_positive, support + predicted) }
    }
}

/// Bound retained exact gold and decoded trees using the same terminal-field
/// definition as scoring. The caller's parser separately bounds allocations.
pub(crate) fn count_with_checkpoint<E: From<AssayError>>(value: &JsonValue,
    max_fields: usize, max_nodes: usize, max_depth: usize,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<usize, E> {
    let mut walk = Walk::new(max_fields, max_nodes, max_depth)?;
    checkpoint()?; walk.unmatched(value, true, 0, false, checkpoint)?; checkpoint()?;
    Ok(walk.fields[0])
}

struct Walk {
    counts: FieldCounts,
    nodes: [usize; 2],
    fields: [usize; 2],
    max_fields: usize,
    max_nodes: usize,
    max_depth: usize,
}
impl Walk {
    fn new(max_fields: usize, max_nodes: usize, max_depth: usize) -> Result<Self, AssayError> {
        if !(1..=4096).contains(&max_fields) || !(1..=65_536).contains(&max_nodes)
            || !(1..=128).contains(&max_depth) { return Err(AssayError::Limits); }
        Ok(Self { counts: FieldCounts::default(), nodes: [0; 2], fields: [0; 2],
            max_fields, max_nodes, max_depth })
    }
    fn enter<E: From<AssayError>>(&mut self, gold: bool, depth: usize,
        checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), E> {
        let index = usize::from(!gold);
        if self.nodes[index] % 64 == 0 { checkpoint()?; }
        if depth > self.max_depth || self.nodes[index] == self.max_nodes { return Err(AssayError::InputLimit.into()); }
        self.nodes[index] += 1; Ok(())
    }
    fn field(&mut self, gold: bool) -> Result<(), AssayError> {
        let index = usize::from(!gold);
        if self.fields[index] == self.max_fields { return Err(AssayError::InputLimit); }
        self.fields[index] += 1; Ok(())
    }
    fn compare<E: From<AssayError>>(&mut self, gold: &JsonValue, predicted: &JsonValue, depth: usize,
        checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), E> {
        self.enter(true, depth, checkpoint)?; self.enter(false, depth, checkpoint)?;
        match (gold, predicted) {
            (JsonValue::Object(a), JsonValue::Object(b)) if !a.is_empty() && !b.is_empty() => {
                let (mut a, mut b) = (a.iter().peekable(), b.iter().peekable());
                while let (Some((ak, _)), Some((bk, _))) = (a.peek(), b.peek()) {
                    match ak.cmp(bk) {
                        std::cmp::Ordering::Equal => {
                            let (_, av) = a.next().ok_or(AssayError::Observation)?;
                            let (_, bv) = b.next().ok_or(AssayError::Observation)?;
                            self.compare(av, bv, depth + 1, checkpoint)?;
                        }
                        std::cmp::Ordering::Less => {
                            let (_, value) = a.next().ok_or(AssayError::Observation)?;
                            self.unmatched(value, true, depth + 1, false, checkpoint)?;
                        }
                        std::cmp::Ordering::Greater => {
                            let (_, value) = b.next().ok_or(AssayError::Observation)?;
                            self.unmatched(value, false, depth + 1, false, checkpoint)?;
                        }
                    }
                }
                for (_, value) in a { self.unmatched(value, true, depth + 1, false, checkpoint)?; }
                for (_, value) in b { self.unmatched(value, false, depth + 1, false, checkpoint)?; }
            }
            (JsonValue::Array(a), JsonValue::Array(b)) if !a.is_empty() && !b.is_empty() => {
                let common = a.len().min(b.len());
                for (a, b) in a[..common].iter().zip(&b[..common]) { self.compare(a, b, depth + 1, checkpoint)?; }
                for value in &a[common..] { self.unmatched(value, true, depth + 1, false, checkpoint)?; }
                for value in &b[common..] { self.unmatched(value, false, depth + 1, false, checkpoint)?; }
            }
            _ if terminal(gold) && terminal(predicted) => {
                self.field(true)?; self.field(false)?;
                if gold == predicted { self.counts.true_positive += 1; }
                else { self.counts.false_positive += 1; self.counts.false_negative += 1; }
            }
            _ => {
                // Distinct container kinds never share paths. In particular,
                // an object key "0" cannot align with an array's index zero.
                self.unmatched(gold, true, depth, true, checkpoint)?;
                self.unmatched(predicted, false, depth, true, checkpoint)?;
            }
        }
        Ok(())
    }
    fn unmatched<E: From<AssayError>>(&mut self, value: &JsonValue, gold: bool, depth: usize,
        entered: bool, checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), E> {
        if !entered { self.enter(gold, depth, checkpoint)?; }
        match value {
            JsonValue::Object(values) if !values.is_empty() => {
                for value in values.values() { self.unmatched(value, gold, depth + 1, false, checkpoint)?; }
            }
            JsonValue::Array(values) if !values.is_empty() => {
                for value in values { self.unmatched(value, gold, depth + 1, false, checkpoint)?; }
            }
            _ => {
                self.field(gold)?;
                if gold { self.counts.false_negative += 1; } else { self.counts.false_positive += 1; }
            }
        }
        Ok(())
    }
}
fn terminal(value: &JsonValue) -> bool {
    match value { JsonValue::Array(values) => values.is_empty(), JsonValue::Object(values) => values.is_empty(), _ => true }
}
fn sum(a: usize, b: usize) -> Result<usize, AssayError> { a.checked_add(b).ok_or(AssayError::InputLimit) }
fn ratio(a: usize, b: usize) -> Option<f64> { (b != 0).then(|| a as f64 / b as f64) }

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FieldScores {
    pub support: usize,
    pub predicted: usize,
    pub true_positive: usize,
    pub false_positive: usize,
    pub false_negative: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FieldIntervals {
    pub precision: Option<ConfidenceInterval>,
    pub recall: Option<ConfidenceInterval>,
    pub f1: Option<ConfidenceInterval>,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SchemaMetrics {
    pub schema_id: String,
    pub rows: usize,
    pub fields: FieldScores,
    pub exact_documents: usize,
    /// Absent when a declared schema has no locked-test documents.
    pub document_accuracy: Option<f64>,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageMetrics {
    pub language: String,
    pub rows: usize,
    pub fields: FieldScores,
    pub exact_documents: usize,
    pub document_accuracy: Option<f64>,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FieldMetrics {
    pub rows: usize,
    pub fields: FieldScores,
    pub exact_documents: usize,
    pub document_accuracy: f64,
    pub field_interval: FieldIntervals,
    pub document_accuracy_interval: ConfidenceInterval,
    /// All declared schemas; breakdowns are point estimates, not extra draws.
    pub by_schema: Vec<SchemaMetrics>,
    /// Caller-declared metadata, not language detection evidence.
    pub by_language: Vec<LanguageMetrics>,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum Measurement {
    InsufficientData { split: Split, rows: usize, minimum_samples: usize },
    Measured { metrics: FieldMetrics },
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedSummary {
    pub rows: usize,
    /// Per-document FP + FN; negative change is fewer candidate errors.
    pub mean_field_error_change: f64,
    pub candidate_better: usize,
    pub candidate_worse: usize,
    pub equal: usize,
    /// Candidate minus baseline on identical complete-document draws.
    pub field_precision_change: Option<f64>,
    pub field_recall_change: Option<f64>,
    pub field_f1_change: Option<f64>,
    pub document_accuracy_change: f64,
    pub field_precision_change_interval: Option<ConfidenceInterval>,
    pub field_recall_change_interval: Option<ConfidenceInterval>,
    pub field_f1_change_interval: Option<ConfidenceInterval>,
    pub document_accuracy_change_interval: ConfidenceInterval,
}

#[derive(Clone, Copy, Debug, Default)]
struct Totals { rows: usize, fields: FieldCounts, exact_documents: usize }
impl Totals {
    fn row(fields: FieldCounts) -> Self {
        Self { rows: 1, fields, exact_documents: usize::from(fields.exact_match()) }
    }
    fn add(&mut self, other: Self) -> Result<(), AssayError> {
        self.rows = sum(self.rows, other.rows)?;
        self.exact_documents = sum(self.exact_documents, other.exact_documents)?;
        self.fields.add(other.fields)
    }
    fn values(self) -> [Option<f64>; 4] {
        let fields = self.fields.scores();
        [fields.precision, fields.recall, fields.f1, ratio(self.exact_documents, self.rows)]
    }
}
pub(crate) struct Observation<'a> {
    pub schema_id: &'a str,
    pub language: &'a str,
    pub counts: &'a FieldCounts,
}
pub(crate) fn measure_with_checkpoint<E: From<AssayError>>(schemas: &[&str], rows: &[Observation<'_>],
    options: EvalOptions, limits: Limits, checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<Measurement, E> {
    options.validate_rows(rows.len(), limits)?;
    validate_rows(schemas, rows, checkpoint)?;
    if rows.len() < options.minimum_samples { return Ok(insufficient(rows.len(), options)); }
    let summaries = summaries(rows, checkpoint)?;
    let intervals = super::document_bootstrap::measure_with_checkpoint::<4, Totals, E>(
        &summaries, None, options, limits, Totals::add, Totals::values, checkpoint)?;
    Ok(Measurement::Measured { metrics: aggregate(schemas, rows, intervals.candidate, checkpoint)? })
}
pub(crate) fn paired_with_checkpoint<E: From<AssayError>>(schemas: &[&str], candidate: &[Observation<'_>],
    baseline: &[Observation<'_>], options: EvalOptions, limits: Limits,
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(Measurement, Measurement, Option<PairedSummary>), E> {
    options.validate_paired_rows(candidate.len(), limits)?;
    validate_rows(schemas, candidate, checkpoint)?; validate_rows(schemas, baseline, checkpoint)?;
    if candidate.len() != baseline.len() { return Err(AssayError::Scope.into()); }
    for (index, (a, b)) in candidate.iter().zip(baseline).enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        if a.schema_id != b.schema_id || a.language != b.language
            || a.counts.true_positive + a.counts.false_negative != b.counts.true_positive + b.counts.false_negative {
            return Err(AssayError::Scope.into());
        }
    }
    if candidate.len() < options.minimum_samples {
        return Ok((insufficient(candidate.len(), options), insufficient(candidate.len(), options), None));
    }
    let current = summaries(candidate, checkpoint)?; let prior = summaries(baseline, checkpoint)?;
    let intervals = super::document_bootstrap::measure_with_checkpoint::<4, Totals, E>(
        &current, Some(&prior), options, limits, Totals::add, Totals::values, checkpoint)?;
    let current = aggregate(schemas, candidate, intervals.candidate, checkpoint)?;
    let prior = aggregate(schemas, baseline, intervals.baseline.ok_or(AssayError::Scope)?, checkpoint)?;
    let changes = intervals.changes.ok_or(AssayError::Scope)?;
    let difference = |a: Option<f64>, b: Option<f64>| a.zip(b).map(|(a, b)| a - b);
    let mut paired = PairedSummary { rows: candidate.len(), mean_field_error_change: 0.0,
        candidate_better: 0, candidate_worse: 0, equal: 0,
        field_precision_change: difference(current.fields.precision, prior.fields.precision),
        field_recall_change: difference(current.fields.recall, prior.fields.recall),
        field_f1_change: difference(current.fields.f1, prior.fields.f1),
        document_accuracy_change: current.document_accuracy - prior.document_accuracy,
        field_precision_change_interval: changes[0], field_recall_change_interval: changes[1],
        field_f1_change_interval: changes[2], document_accuracy_change_interval: changes[3].ok_or(AssayError::Observation)? };
    for (index, (a, b)) in candidate.iter().zip(baseline).enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        let (a, b) = (a.counts.errors(), b.counts.errors());
        paired.mean_field_error_change += a as f64 - b as f64;
        match a.cmp(&b) {
            std::cmp::Ordering::Less => paired.candidate_better += 1,
            std::cmp::Ordering::Greater => paired.candidate_worse += 1,
            std::cmp::Ordering::Equal => paired.equal += 1,
        }
    }
    paired.mean_field_error_change /= candidate.len() as f64;
    checkpoint()?;
    Ok((Measurement::Measured { metrics: current }, Measurement::Measured { metrics: prior }, Some(paired)))
}
fn validate_rows<E: From<AssayError>>(schemas: &[&str], rows: &[Observation<'_>],
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<(), E> {
    if schemas.is_empty() || schemas.len() > 64 { return Err(AssayError::Observation.into()); }
    let mut known = BTreeSet::new();
    for schema in schemas {
        if schema.is_empty() || schema.len() > 256 || !known.insert(*schema) { return Err(AssayError::Observation.into()); }
    }
    let mut languages = BTreeSet::new();
    for (index, row) in rows.iter().enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        if !known.contains(row.schema_id) || row.language.is_empty() || row.language.len() > 256
            || !(1..=4096).contains(&sum(row.counts.true_positive, row.counts.false_negative)?)
            || !(1..=4096).contains(&sum(row.counts.true_positive, row.counts.false_positive)?) {
            return Err(AssayError::Observation.into());
        }
        languages.insert(row.language);
        if languages.len() > 128 { return Err(AssayError::InputLimit.into()); }
    }
    checkpoint()?; Ok(())
}
fn insufficient(rows: usize, options: EvalOptions) -> Measurement {
    Measurement::InsufficientData { split: Split::LockedTest, rows, minimum_samples: options.minimum_samples }
}
fn summaries<E: From<AssayError>>(rows: &[Observation<'_>], checkpoint: &mut impl FnMut() -> Result<(), E>)
    -> Result<Vec<Totals>, E> {
    let mut result = reserved(rows.len())?;
    for (index, row) in rows.iter().enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        result.push(Totals::row(*row.counts));
    }
    checkpoint()?; Ok(result)
}
fn aggregate<E: From<AssayError>>(schemas: &[&str], rows: &[Observation<'_>], intervals: [Option<ConfidenceInterval>; 4],
    checkpoint: &mut impl FnMut() -> Result<(), E>) -> Result<FieldMetrics, E> {
    let mut all = Totals::default();
    let mut by_schema: BTreeMap<&str, Totals> = schemas.iter().map(|id| (*id, Totals::default())).collect();
    let mut by_language = BTreeMap::<&str, Totals>::new();
    for (index, row) in rows.iter().enumerate() {
        if index % 64 == 0 { checkpoint()?; }
        let counts = Totals::row(*row.counts); all.add(counts)?;
        by_schema.get_mut(row.schema_id).ok_or(AssayError::Scope)?.add(counts)?;
        by_language.entry(row.language).or_default().add(counts)?;
    }
    let mut schemas = reserved(by_schema.len())?;
    for (schema_id, totals) in by_schema {
        checkpoint()?;
        schemas.push(SchemaMetrics { schema_id: schema_id.to_owned(), rows: totals.rows, fields: totals.fields.scores(),
            exact_documents: totals.exact_documents, document_accuracy: ratio(totals.exact_documents, totals.rows) });
    }
    let mut languages = reserved(by_language.len())?;
    for (language, totals) in by_language {
        checkpoint()?;
        languages.push(LanguageMetrics { language: language.to_owned(), rows: totals.rows, fields: totals.fields.scores(),
            exact_documents: totals.exact_documents, document_accuracy: ratio(totals.exact_documents, totals.rows) });
    }
    Ok(FieldMetrics { rows: rows.len(), fields: all.fields.scores(), exact_documents: all.exact_documents,
        document_accuracy: ratio(all.exact_documents, all.rows).ok_or(AssayError::Observation)?,
        field_interval: FieldIntervals { precision: intervals[0], recall: intervals[1], f1: intervals[2] },
        document_accuracy_interval: intervals[3].ok_or(AssayError::Observation)?, by_schema: schemas, by_language: languages })
}
fn reserved<T>(count: usize) -> Result<Vec<T>, AssayError> {
    let mut values = Vec::new(); values.try_reserve_exact(count).map_err(|_| AssayError::Allocation)?; Ok(values)
}

#[cfg(test)] mod tests;
