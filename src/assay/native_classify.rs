//! Actual native binary-classification replay over one frozen labeled corpus.
//!
//! Only the hosted native execution owner can construct a replay. Reports
//! contain no document text, private row IDs or raw private content digests.
//! Complete row identities and actual scored decisions have separate keyed
//! collection commitments; there is no invented document-independent identity.

use super::{AssayError, EvalOptions, Limits, Split, SplitBindings, SplitIds, ASSAY_SCHEMA_VERSION};

mod dataset;
mod replay;
mod report;
pub use dataset::{Dataset, Row, SuiteManifest};
pub use report::{Bindings, ClassMetrics, ClassificationMetrics, Evaluation, EvaluationBody,
    Gates, QualificationBody, QualificationReport, parse_gates};
pub(crate) use replay::{Collector, NativeReplay};
pub(crate) use report::{compare_with_checkpoint, evaluate_with_checkpoint, validate_scope};
#[cfg(test)]
pub(crate) use report::{compare, evaluate};

pub const NATIVE_EVIDENCE: &str = "native_int4_classification_replay";
pub const NATIVE_TASK: &str = "binary_classification";
pub const EXECUTION_COLLECTION_FORMAT: &str = "ordered_full_execution_identity_sha256_framed_v1";
pub const OBSERVATION_COLLECTION_FORMAT: &str = "ordered_native_binary_observations_sha256_framed_v1";

/// Conservative payload/ownership quote for all retained score, row-ID,
/// manifest, split-partition, bootstrap, commitment and report staging.
/// It excludes source Dataset text, compiler/preparation, model and native KV;
/// the host admits those separately. The complete final output has its own
/// delivery guard. This is not an operating-system RSS guarantee.
pub fn required_replay_bytes(limits: Limits, arms: usize) -> Result<u64, AssayError> {
    limits.validate()?;
    if !(1..=2).contains(&arms) { return Err(AssayError::Limits); }
    let multiply = |a: u64, b: u64| a.checked_mul(b).ok_or(AssayError::Limits);
    let add = |a: u64, b: u64| a.checked_add(b).ok_or(AssayError::Limits);
    // IDs are capped at 256 bytes. This prices their retained copies in the
    // manifest, scores, partition sets and canonical commitment staging.
    let rows = multiply(limits.max_rows as u64, 2048)?;
    let manifests = multiply(limits.max_manifest_bytes as u64, 8)?;
    let per_arm = add(add(rows, manifests)?, 65_536)?;
    let all_arms = multiply(per_arm, arms as u64)?;
    let reports = multiply(limits.max_output_bytes as u64, 8)?;
    let bootstrap = multiply(limits.max_bootstrap_resamples as u64, 64)?;
    add(add(all_arms, reports)?, bootstrap)
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
