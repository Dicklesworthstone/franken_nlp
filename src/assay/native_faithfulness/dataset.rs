//! Exact source/claim cases, with source-disjoint development/calibration/test.
use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};
use crate::{execution_identity::Sha256Digest,
    tasks::{ir::TaskBudget, judge::{FaithfulnessPolicy, FaithfulnessRelation, JudgeRequest}}};
use super::*;

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteManifest {
    pub schema_version: u32,
    pub task: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub named_population: String,
    pub policy: FaithfulnessPolicy,
    pub expected_input_sha256: Sha256Digest,
    pub splits: SplitIds,
}

/// Gold, row IDs and split names never enter the model's request.
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub schema_version: u32,
    pub id: String,
    pub split: Split,
    pub source: String,
    pub claim: String,
    pub relation: FaithfulnessRelation,
}

pub struct Dataset {
    pub(super) manifest: SuiteManifest,
    rows: Vec<Row>,
    pub(super) manifest_digest: Sha256Digest,
    pub(super) input_digest: Sha256Digest,
    pub(super) sizes: InputSizes,
    sources: usize,
}
#[derive(Clone, Copy)]
pub(super) struct InputSizes { pub manifest: usize, pub input: usize, pub line: usize, pub rows: usize }
impl InputSizes {
    pub fn validate(self, limits: Limits) -> Result<(), AssayError> {
        limits.validate()?;
        if self.manifest > limits.max_manifest_bytes || self.input > limits.max_input_bytes
            || self.line > limits.max_line_bytes || self.rows > limits.max_rows { return Err(AssayError::InputLimit); }
        Ok(())
    }
}
impl Dataset {
    pub fn parse(manifest_bytes: &[u8], input: &[u8], limits: Limits) -> Result<Self, AssayError> {
        limits.validate()?;
        if manifest_bytes.is_empty() || manifest_bytes.len() > limits.max_manifest_bytes
            || input.is_empty() || input.len() > limits.max_input_bytes { return Err(AssayError::InputLimit); }
        let mut manifest: SuiteManifest = strict_json(manifest_bytes, 4096)?;
        let total = validate_manifest(&manifest, limits)?;
        let input_digest = Sha256Digest::of_bytes(input);
        if input_digest != manifest.expected_input_sha256 { return Err(AssayError::InputDigest); }
        let text = std::str::from_utf8(input).map_err(|_| AssayError::Json)?;
        let mut count = 0_usize; let mut max_line = 0;
        for line in text.split_terminator('\n') {
            count = count.checked_add(1).ok_or(AssayError::InputLimit)?;
            if count > limits.max_rows || line.len() > limits.max_line_bytes { return Err(AssayError::InputLimit); }
            if line.strip_suffix('\r').unwrap_or(line).trim().is_empty() { return Err(AssayError::Observation); }
            max_line = max_line.max(line.len());
        }
        if count != total { return Err(AssayError::SplitMembership); }
        manifest.splits.development.sort(); manifest.splits.calibration.sort(); manifest.splits.locked_test.sort();
        let manifest_digest = Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&manifest)
            .map_err(|_| AssayError::Manifest)?);
        let mut expected = BTreeMap::new();
        for (split, ids) in split_lists(&manifest.splits) { for id in ids { expected.insert(id.as_str(), split); } }
        let mut seen = BTreeSet::new(); let mut source_splits = BTreeMap::new(); let mut pairs = BTreeSet::new();
        let mut rows = reserved(manifest.splits.locked_test.len())?;
        for line in text.split_terminator('\n') {
            let row: Row = strict_json(line.strip_suffix('\r').unwrap_or(line).as_bytes(), limits.max_line_bytes)?;
            if row.schema_version != ASSAY_SCHEMA_VERSION || !identifier(&row.id)
                || row.source.is_empty() || row.claim.is_empty() { return Err(AssayError::Observation); }
            if expected.get(row.id.as_str()) != Some(&row.split) { return Err(AssayError::SplitMembership); }
            if !seen.insert(row.id.clone()) { return Err(AssayError::DuplicateId); }
            let source = Sha256Digest::of_bytes(row.source.as_bytes());
            if source_splits.insert(source, row.split).is_some_and(|split| split != row.split) {
                return Err(AssayError::SplitMembership);
            }
            if !pairs.insert((source, Sha256Digest::of_bytes(row.claim.as_bytes()))) { return Err(AssayError::DuplicateUnit); }
            if row.split == Split::LockedTest { rows.push(row); }
        }
        if seen.len() != expected.len() { return Err(AssayError::SplitMembership); }
        let sources = source_splits.values().filter(|&&split| split == Split::LockedTest).count();
        rows.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(Self { manifest, rows, manifest_digest, input_digest, sources,
            sizes: InputSizes { manifest: manifest_bytes.len(), input: input.len(), line: max_line, rows: total } })
    }
    pub fn manifest(&self) -> &SuiteManifest { &self.manifest }
    pub fn replay_rows(&self) -> &[Row] { &self.rows }
    pub fn replay_len(&self) -> usize { self.rows.len() }
    pub fn validated_rows(&self) -> usize { self.sizes.rows }
    pub fn locked_test_sources(&self) -> usize { self.sources }
    pub fn validate_limits(&self, limits: Limits) -> Result<(), AssayError> { self.sizes.validate(limits) }
    /// Refuse incomplete evidence-head capacity before loading either model.
    pub fn validate_head_limit(&self, max_heads: usize) -> Result<(), AssayError> {
        if !(1..=32).contains(&max_heads) { return Err(AssayError::Limits); }
        for row in &self.rows {
            let windows = crate::tasks::judge::partition_evidence(&row.source, self.manifest.policy)
                .map_err(|_| AssayError::InputLimit)?;
            let heads = if windows.len() == 1 { 1 } else { windows.len() + 1 };
            if heads > max_heads { return Err(AssayError::InputLimit); }
        }
        Ok(())
    }

    pub fn request(&self, index: usize, budget: TaskBudget) -> Result<JudgeRequest, AssayError> {
        budget.validate().map_err(|_| AssayError::Limits)?;
        let row = self.rows.get(index).ok_or(AssayError::Scope)?;
        Ok(JudgeRequest::Faithfulness { source: row.source.clone(), claim: row.claim.clone(),
            policy: self.manifest.policy, budget })
    }
    pub(crate) fn request_digest(&self, index: usize, budget: TaskBudget) -> Result<Sha256Digest, AssayError> {
        self.request(index, budget)?.request_digest().map_err(|_| AssayError::Identity)
    }
}
fn validate_manifest(manifest: &SuiteManifest, limits: Limits) -> Result<usize, AssayError> {
    if manifest.schema_version != ASSAY_SCHEMA_VERSION || manifest.task != NATIVE_TASK
        || !identifier(&manifest.dataset_id) || !identifier(&manifest.dataset_version)
        || manifest.named_population.trim().is_empty() || manifest.named_population.trim() != manifest.named_population
        || manifest.named_population.len() > 1024 || manifest.named_population.chars().any(char::is_control) {
        return Err(AssayError::Manifest);
    }
    manifest.policy.validate().map_err(|_| AssayError::Manifest)?;
    let mut seen = BTreeSet::new(); let mut total = 0_usize;
    for (_, ids) in split_lists(&manifest.splits) {
        if ids.is_empty() { return Err(AssayError::SplitMembership); }
        total = total.checked_add(ids.len()).filter(|&n| n <= limits.max_rows).ok_or(AssayError::InputLimit)?;
        for id in ids { if !identifier(id) || !seen.insert(id.as_str()) { return Err(AssayError::SplitMembership); } }
    }
    Ok(total)
}
pub(super) fn split_lists(ids: &SplitIds) -> [(Split, &[String]); 3] {
    [(Split::Development, &ids.development), (Split::Calibration, &ids.calibration), (Split::LockedTest, &ids.locked_test)]
}
pub(super) fn strict_json<T: serde::de::DeserializeOwned>(bytes: &[u8], max_string_bytes: usize)
    -> Result<T, AssayError> {
    let text = std::str::from_utf8(bytes).map_err(|_| AssayError::Json)?;
    let value = crate::canonjson::parse_str_with_limits(text,
        crate::canonjson::ParseLimits { max_depth: 16, max_string_bytes }).map_err(|_| AssayError::Json)?;
    serde_json::from_value(value).map_err(|_| AssayError::Json)
}
