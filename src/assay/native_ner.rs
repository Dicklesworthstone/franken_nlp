//! Source-verified native NER evaluation over a frozen, user-owned corpus.
//!
//! The native host alone can construct a replay. Only locked-test documents
//! execute; every split and exact input byte remains bound to the scorecard.
//! Unresolved occurrences never become invented span locations or confidence.

use serde::{Deserialize, Serialize};
use super::{AssayError, Limits, Split, SplitBindings, SplitIds, ASSAY_SCHEMA_VERSION};

mod dataset;
mod replay;
mod report;
pub use dataset::{Dataset, GoldSpan, Row, SuiteManifest};
pub use report::{Bindings, Evaluation, EvaluationBody, EvalOptions, Gates, NativeMetadata,
    QualificationBody, QualificationReport, parse_gates, AnchoringMetrics, Decision, LanguageMetrics,
    Measurement, PairedSummary, SpanIntervals, SpanMetrics, SpanScores, TypeMetrics};
pub(crate) use replay::{Collector, NativeReplay};
pub(crate) use report::{compare_with_checkpoint, evaluate_with_checkpoint, validate_scope};

pub const NATIVE_TASK: &str = "named_entity_recognition";
pub const NATIVE_EVIDENCE: &str = "native_int4_ner_replay";
pub const EXECUTION_COLLECTION_FORMAT: &str = "ordered_full_execution_identity_sha256_framed_v1";
pub const OBSERVATION_COLLECTION_FORMAT: &str = "ordered_complete_native_ner_runs_sha256_framed_v1";
pub const MAX_SPANS_PER_ROW: usize = 4096;
pub const MAX_OCCURRENCES_PER_ROW: usize = 65_536;
pub const MAX_LANGUAGES: usize = 128;

/// Geometry and independent source-verification bounds, admitted before
/// source rows or gold vectors are allocated. The scan allowance is shared
/// across the complete hosted operation, including both comparison arms.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NerLimits {
    pub max_spans_per_row: usize,
    pub max_occurrences_per_row: usize,
    pub max_languages: usize,
    pub max_verification_steps: u64,
}
impl Default for NerLimits {
    fn default() -> Self {
        Self { max_spans_per_row: 64, max_occurrences_per_row: 16_384,
            max_languages: 64, max_verification_steps: 64 * 1024 * 1024 }
    }
}
impl NerLimits {
    pub fn validate(self) -> Result<(), AssayError> {
        if !(1..=MAX_SPANS_PER_ROW).contains(&self.max_spans_per_row)
            || !(1..=MAX_OCCURRENCES_PER_ROW).contains(&self.max_occurrences_per_row)
            || !(1..=MAX_LANGUAGES).contains(&self.max_languages)
            || !(1..=4_294_967_296).contains(&self.max_verification_steps) {
            return Err(AssayError::Limits);
        }
        Ok(())
    }
}

/// Retained row summaries have fixed eight-type geometry. Prediction text,
/// token IDs and candidate intervals are hashed and released one row at a
/// time. The host separately admits Dataset gold and current native output.
pub fn required_replay_bytes(limits: Limits, ner_limits: NerLimits, arms: usize)
    -> Result<u64, AssayError> {
    limits.validate()?; ner_limits.validate()?;
    if !(1..=2).contains(&arms) { return Err(AssayError::Limits); }
    let multiply = |a: u64, b: u64| a.checked_mul(b).ok_or(AssayError::Limits);
    let add = |a: u64, b: u64| a.checked_add(b).ok_or(AssayError::Limits);
    let rows = multiply(limits.max_rows as u64, 4096)?;
    let manifests = multiply(limits.max_manifest_bytes as u64, 8)?;
    let per_arm = add(add(rows, manifests)?, 65_536)?;
    let reports = multiply(limits.max_output_bytes as u64, 8)?;
    let bootstrap = multiply(limits.max_bootstrap_resamples as u64, 192)?;
    add(add(multiply(per_arm, arms as u64)?, reports)?, bootstrap)
}

fn identifier(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && !value.chars().any(|ch| ch.is_control() || ch.is_whitespace())
}
fn label(value: &str, cap: usize) -> bool {
    !value.trim().is_empty() && value.trim() == value && value.len() <= cap && !value.chars().any(char::is_control)
}
fn reserved<T>(count: usize) -> Result<Vec<T>, AssayError> {
    let mut out = Vec::new(); out.try_reserve_exact(count).map_err(|_| AssayError::Allocation)?; Ok(out)
}

#[cfg(test)] mod tests;
