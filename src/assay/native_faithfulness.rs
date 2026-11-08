//! Actual full-source faithfulness replay over a frozen human-labeled suite.
//!
//! Multiple claims about one source remain one statistical cluster. Only
//! held-out rows execute; all splits and exact source/claim bytes are bound.
//! A native judgment is compared with caller-supplied gold, never called truth.

use super::{AssayError, Limits, Split, SplitBindings, SplitIds, ASSAY_SCHEMA_VERSION};

mod dataset;
mod replay;
mod metrics;
mod report;
pub use dataset::{Dataset, Row, SuiteManifest};
pub use report::{Bindings, Decision, Evaluation, EvaluationBody, EvalOptions, Gates,
    Measurement, NativeMetadata, PairedSummary, QualificationBody, QualificationReport,
    Quality, QualityIntervals, parse_gates};
pub(crate) use replay::{Collector, NativeReplay, ReplayError};
pub(crate) use report::{evaluate_with_checkpoint, compare_with_checkpoint, validate_scope};

pub const NATIVE_TASK: &str = "faithfulness";
pub const NATIVE_EVIDENCE: &str = "native_int4_faithfulness_replay";
pub const EXECUTION_COLLECTION_FORMAT: &str = "ordered_full_judge_execution_identity_sha256_framed_v1";
pub const OBSERVATION_COLLECTION_FORMAT: &str = "ordered_complete_native_faithfulness_sha256_framed_v1";
pub const RELATIONS: [&str; 3] = ["contradicted", "entailed", "unsupported"];

/// Retained compact outcomes, source-cluster summaries, exact commitments and
/// complete report staging. Source text, model, native KV and task preparation
/// are admitted separately by the execution owner. No RSS claim is implied.
pub fn required_replay_bytes(limits: Limits, arms: usize) -> Result<u64, AssayError> {
    limits.validate()?;
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
fn reserved<T>(count: usize) -> Result<Vec<T>, AssayError> {
    let mut values = Vec::new(); values.try_reserve_exact(count).map_err(|_| AssayError::Allocation)?; Ok(values)
}

#[cfg(test)] mod tests;
