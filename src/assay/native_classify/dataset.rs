//! Strict task-shaped source input; scores and execution identities are absent.
use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};
use crate::{execution_identity::Sha256Digest,
    tasks::{classify::{ClassificationLabel, ClassificationMode, ClassificationPolicy, ClassificationRequest},
        ir::TaskBudget}};
use super::*;

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteManifest {
    pub schema_version: u32,
    pub task: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub named_population: String,
    pub positive_label: ClassificationLabel,
    pub negative_label: ClassificationLabel,
    /// Candidate-relative task thresholds; distinct from evaluation acceptance.
    #[serde(default)]
    pub policy: ClassificationPolicy,
    pub expected_input_sha256: Sha256Digest,
    pub splits: SplitIds,
}

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub schema_version: u32,
    pub id: String,
    pub split: Split,
    pub text: String,
    pub positive: bool,
}

/// Parsing seals the exact source pin, complete membership and global source
/// units. Public readers cannot replace rows or supply model predictions.
pub struct Dataset {
    pub(super) manifest: SuiteManifest,
    rows: Vec<Row>,
    pub(super) manifest_digest: Sha256Digest,
    pub(super) input_digest: Sha256Digest,
    pub(super) sizes: InputSizes,
}

#[derive(Clone, Copy)]
pub(super) struct InputSizes { pub manifest: usize, pub input: usize, pub line: usize, pub rows: usize }
impl InputSizes {
    pub fn validate(self, limits: Limits) -> Result<(), AssayError> {
        limits.validate()?;
        if self.manifest > limits.max_manifest_bytes || self.input > limits.max_input_bytes
            || self.line > limits.max_line_bytes || self.rows > limits.max_rows {
            return Err(AssayError::InputLimit);
        }
        Ok(())
    }
}

impl Dataset {
    pub fn parse(manifest_bytes: &[u8], input: &[u8], limits: Limits) -> Result<Self, AssayError> {
        limits.validate()?;
        if manifest_bytes.is_empty() || manifest_bytes.len() > limits.max_manifest_bytes
            || input.is_empty() || input.len() > limits.max_input_bytes {
            return Err(AssayError::InputLimit);
        }
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
        let mut rows = reserved(total)?;
        {
            let mut expected = BTreeMap::new();
            for (split, ids) in split_lists(&manifest.splits) {
                for id in ids { expected.insert(id.as_str(), split); }
            }
            let mut seen = BTreeSet::new(); let mut units = BTreeSet::new();
            for line in text.split_terminator('\n') {
                let row: Row = strict_json(line.strip_suffix('\r').unwrap_or(line).as_bytes(), limits.max_line_bytes)?;
                if row.schema_version != ASSAY_SCHEMA_VERSION || !identifier(&row.id) || row.text.is_empty() {
                    return Err(AssayError::Observation);
                }
                if expected.get(row.id.as_str()) != Some(&row.split) { return Err(AssayError::SplitMembership); }
                if !seen.insert(row.id.clone()) { return Err(AssayError::DuplicateId); }
                // The exact source bytes define the source unit. A new ID,
                // label or split cannot conceal an already observed document.
                if !units.insert(Sha256Digest::of_bytes(row.text.as_bytes())) { return Err(AssayError::DuplicateUnit); }
                rows.push(row);
            }
            if seen.len() != expected.len() { return Err(AssayError::SplitMembership); }
        }
        rows.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(Self { manifest, rows, manifest_digest, input_digest,
            sizes: InputSizes { manifest: manifest_bytes.len(), input: input.len(), line: max_line, rows: total } })
    }
    pub fn manifest(&self) -> &SuiteManifest { &self.manifest }
    pub fn rows(&self) -> &[Row] { &self.rows }
    pub fn len(&self) -> usize { self.rows.len() }
    pub fn is_empty(&self) -> bool { self.rows.is_empty() }
    pub fn locked_test_len(&self) -> usize { self.manifest.splits.locked_test.len() }
    pub fn validate_limits(&self, limits: Limits) -> Result<(), AssayError> { self.sizes.validate(limits) }

    /// The host passes its one admitted task budget. Only source bytes, the
    /// two declared labels and the bounded native task policy enter the model.
    /// Human gold labels and evaluation splits never enter its prompt.
    pub fn classification_request(&self, index: usize, budget: TaskBudget)
        -> Result<ClassificationRequest, AssayError> {
        budget.validate().map_err(|_| AssayError::Limits)?;
        let row = self.rows.get(index).ok_or(AssayError::Scope)?;
        Ok(ClassificationRequest { document: row.text.clone(),
            labels: vec![self.manifest.positive_label.clone(), self.manifest.negative_label.clone()],
            mode: ClassificationMode::Exclusive, policy: self.manifest.policy, budget })
    }
}

fn validate_manifest(manifest: &SuiteManifest, limits: Limits) -> Result<usize, AssayError> {
    if manifest.schema_version != ASSAY_SCHEMA_VERSION || manifest.task != NATIVE_TASK
        || !identifier(&manifest.dataset_id) || !identifier(&manifest.dataset_version)
        || !label(&manifest.named_population, 1024)
        || manifest.positive_label.id == manifest.negative_label.id
        || manifest.policy.minimum_candidate_weight_ppm > 1_000_000
        || manifest.policy.minimum_margin_ppm > 1_000_000 {
        return Err(AssayError::Manifest);
    }
    for item in [&manifest.positive_label, &manifest.negative_label] {
        if !label(&item.id, 256) || item.description.len() > 4096 {
            return Err(AssayError::Manifest);
        }
    }
    let mut total = 0_usize; let mut seen = BTreeSet::new();
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
