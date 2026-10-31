//! Native structured-extraction quality over frozen, owner-defined schemas.
//!
//! Exact schema and gold JSON remain private strings at the outer wire
//! boundary. The independent decimal parser owns field equality; the native
//! extraction finalizer owns structural and source-membership validity.

use serde::{Deserialize, Serialize};
use super::{AssayError, Limits, Split, SplitBindings, SplitIds, ASSAY_SCHEMA_VERSION};

mod dataset;
mod replay;
mod report;
pub use dataset::{Dataset, Row, SchemaDefinition, SuiteManifest};
pub use report::{Bindings, Decision, Evaluation, EvaluationBody, EvalOptions, FieldIntervals,
    FieldMetrics, FieldScores, Gates, LanguageMetrics, Measurement, NativeMetadata, PairedSummary,
    QualificationBody, QualificationReport, SchemaMetrics, parse_gates};
pub(crate) use replay::{Collector, NativeReplay};
pub(crate) use report::{compare_with_checkpoint, evaluate_with_checkpoint, validate_scope};

pub const NATIVE_TASK: &str = "structured_extraction";
pub const NATIVE_EVIDENCE: &str = "native_int4_extraction_replay";
pub const EXECUTION_COLLECTION_FORMAT: &str = "ordered_full_execution_identity_sha256_framed_v1";
pub const OBSERVATION_COLLECTION_FORMAT: &str = "ordered_complete_native_extraction_runs_sha256_framed_v1";
pub const MAX_SCHEMAS: usize = 64;
pub const MAX_FIELDS_PER_ROW: usize = 4096;
pub const MAX_JSON_NODES_PER_ROW: usize = 65_536;
pub const MAX_JSON_DEPTH: usize = 128;
pub const MAX_LANGUAGES: usize = 128;
pub const MAX_OCCURRENCES_PER_ROW: usize = 65_536;

/// Tree geometry is admitted before exact gold allocation. The verification
/// allowance covers gold preparation and every native/independent replay
/// source scan across the whole operation, including both comparison arms.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractLimits {
    pub max_schemas: usize,
    pub max_fields_per_row: usize,
    pub max_json_nodes_per_row: usize,
    pub max_json_depth: usize,
    pub max_languages: usize,
    pub max_occurrences_per_row: usize,
    pub max_verification_steps: u64,
}
impl Default for ExtractLimits {
    fn default() -> Self {
        Self { max_schemas: 16, max_fields_per_row: 256, max_json_nodes_per_row: 16_384,
            max_json_depth: 64, max_languages: 64, max_occurrences_per_row: 16_384,
            max_verification_steps: 64 * 1024 * 1024 }
    }
}
impl ExtractLimits {
    pub fn validate(self) -> Result<(), AssayError> {
        if !(1..=MAX_SCHEMAS).contains(&self.max_schemas)
            || !(1..=MAX_FIELDS_PER_ROW).contains(&self.max_fields_per_row)
            || !(1..=MAX_JSON_NODES_PER_ROW).contains(&self.max_json_nodes_per_row)
            || !(1..=MAX_JSON_DEPTH).contains(&self.max_json_depth)
            || !(1..=MAX_LANGUAGES).contains(&self.max_languages)
            || !(1..=MAX_OCCURRENCES_PER_ROW).contains(&self.max_occurrences_per_row)
            || !(1..=4_294_967_296).contains(&self.max_verification_steps) {
            return Err(AssayError::Limits);
        }
        Ok(())
    }
}

/// Replays retain fixed per-document summaries, never generated JSON, field
/// paths, tokens or occurrence lists. Dataset gold and one active native
/// result/compiler are admitted separately by the host and CLI.
pub fn required_replay_bytes(limits: Limits, extract: ExtractLimits, arms: usize)
    -> Result<u64, AssayError> {
    limits.validate()?; extract.validate()?;
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
