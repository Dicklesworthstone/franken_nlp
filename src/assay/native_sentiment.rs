//! Held-out human ratings compared with actual native dimensional sentiment.
//!
//! Each document is one statistical unit, carrying all of its requested axes.
//! Gold ratings use the existing task's normalized coordinate: integer milli-
//! units in -1000..=1000 mean -1..=1, not the separate planned -100..=100 scale.
//! Native candidate weights and descriptive means are never calibration claims.

use super::{AssayError, Limits, Split, SplitBindings, SplitIds, ASSAY_SCHEMA_VERSION};
use crate::tasks::sentiment::SentimentAxis;

mod dataset;
mod replay;
mod metrics;
mod report;
pub use dataset::{Dataset, Gold, Row, SuiteManifest};
pub use metrics::{AxisChange, AxisQuality, EvalOptions, Measurement, PairedSummary, QualityIntervals, SpearmanStatus};
pub use report::{Bindings, Decision, Evaluation, EvaluationBody, Gates, NativeMetadata,
    QualificationBody, QualificationReport, parse_gates};
pub(crate) use replay::{Collector, NativeReplay, ReplayError};
pub(crate) use metrics::validate_replay;
pub(crate) use report::{evaluate_with_checkpoint, compare_with_checkpoint, validate_scope};

pub const NATIVE_TASK: &str = "sentiment";
pub const NATIVE_EVIDENCE: &str = "native_int4_sentiment_replay";
pub const GOLD_SCALE: i32 = 1000;
pub const MAX_AXES: usize = SentimentAxis::ALL.len();
pub const EXECUTION_COLLECTION_FORMAT: &str = "ordered_full_sentiment_execution_identity_sha256_framed_v1";
pub const OBSERVATION_COLLECTION_FORMAT: &str = "ordered_complete_native_sentiment_sha256_framed_v1";

/// Compact outcomes, exact private bindings, rank scratch, bounded bootstrap
/// series and complete report staging. The execution owner separately admits
/// source text, planners, resident weights and the reusable native engine.
pub fn required_replay_bytes(limits: Limits, axes: usize, arms: usize) -> Result<u64, AssayError> {
    limits.validate()?;
    if !(1..=MAX_AXES).contains(&axes) || !(1..=2).contains(&arms) { return Err(AssayError::Limits); }
    let multiply = |a: u64, b: u64| a.checked_mul(b).ok_or(AssayError::Limits);
    let add = |a: u64, b: u64| a.checked_add(b).ok_or(AssayError::Limits);
    let rows = multiply(limits.max_rows as u64, add(1024, multiply(axes as u64, 256)?)?)?;
    let manifests = multiply(limits.max_manifest_bytes as u64, 8)?;
    let per_arm = add(add(rows, manifests)?, 65_536)?;
    let reports = multiply(limits.max_output_bytes as u64, 8)?;
    // Axis bootstraps run sequentially; at most twelve scalar series coexist.
    let bootstrap = multiply(limits.max_bootstrap_resamples as u64, 192)?;
    add(add(multiply(per_arm, arms as u64)?, reports)?, bootstrap)
}

fn identifier(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && !value.chars().any(|ch| ch.is_control() || ch.is_whitespace())
}
fn reserved<T>(count: usize) -> Result<Vec<T>, AssayError> {
    let mut values = Vec::new(); values.try_reserve_exact(count).map_err(|_| AssayError::Allocation)?; Ok(values)
}
fn axis_name(axis: SentimentAxis) -> &'static str {
    match axis { SentimentAxis::Valence => "valence", SentimentAxis::Arousal => "arousal",
        SentimentAxis::Dominance => "dominance", SentimentAxis::Approach => "approach" }
}

#[cfg(test)] mod tests;
