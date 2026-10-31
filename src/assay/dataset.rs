//! Strict, pinned observations and disjoint population membership.
use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};
use crate::{
    calibration::{CalibrationPartition, LabeledScore, SplitMembership},
    execution_identity::{ExecutionIdentity, Sha256Digest, ThinkingMode},
    receipt::{CommitmentDomain, CommitmentKey, ContentCommitment},
};
use super::{AssayError, Limits, ASSAY_SCHEMA_VERSION};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Split { Development, Calibration, LockedTest }
impl Split {
    pub const fn name(self) -> &'static str {
        match self { Self::Development => "development", Self::Calibration => "calibration", Self::LockedTest => "locked_test" }
    }
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SplitIds {
    pub development: Vec<String>,
    pub calibration: Vec<String>,
    pub locked_test: Vec<String>,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteManifest {
    pub schema_version: u32,
    pub task: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub named_population: String,
    pub positive_label: String,
    pub negative_label: String,
    pub execution_identity: ExecutionIdentity,
    pub expected_input_sha256: Sha256Digest,
    pub splits: SplitIds,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub schema_version: u32,
    pub id: String,
    pub split: Split,
    pub probability: f64,
    pub positive: bool,
    /// Declared source-unit identity, private owner input only. Reusing a unit
    /// under another row ID never expands the independent sample population.
    pub unit_sha256: Sha256Digest,
}

/// Only parsing can construct a dataset. Identity, pin, complete membership
/// and duplicate-unit checks precede any exposed metric or calibration set.
pub struct Dataset {
    manifest: SuiteManifest,
    rows: Vec<Observation>,
    partition: CalibrationPartition,
    input_digest: Sha256Digest,
    manifest_bytes: usize,
    input_bytes: usize,
    max_line_bytes: usize,
}
impl Dataset {
    pub fn parse(manifest_bytes: &[u8], observations: &[u8], limits: Limits) -> Result<Self, AssayError> {
        limits.validate()?;
        if manifest_bytes.is_empty() || manifest_bytes.len() > limits.max_manifest_bytes
            || observations.is_empty() || observations.len() > limits.max_input_bytes {
            return Err(AssayError::InputLimit);
        }
        let mut manifest: SuiteManifest = strict_json(manifest_bytes, 4096)?;
        let total = validate_manifest(&manifest, limits)?;
        let input_digest = Sha256Digest::of_bytes(observations);
        if input_digest != manifest.expected_input_sha256 { return Err(AssayError::InputDigest); }
        let text = std::str::from_utf8(observations).map_err(|_| AssayError::Json)?;
        // Check all frame sizes and the complete row count before row storage.
        let mut count = 0_usize;
        let mut max_line_bytes = 0_usize;
        for line in text.split_terminator('\n') {
            count = count.checked_add(1).ok_or(AssayError::InputLimit)?;
            if line.len() > limits.max_line_bytes || count > limits.max_rows { return Err(AssayError::InputLimit); }
            if line.strip_suffix('\r').unwrap_or(line).trim().is_empty() { return Err(AssayError::Observation); }
            max_line_bytes = max_line_bytes.max(line.len());
        }
        if count != total { return Err(AssayError::SplitMembership); }
        manifest.splits.development.sort();
        manifest.splits.calibration.sort();
        manifest.splits.locked_test.sort();
        let mut rows = Vec::new();
        rows.try_reserve_exact(total).map_err(|_| AssayError::Allocation)?;
        {
            let mut expected = BTreeMap::new();
            for (split, ids) in split_lists(&manifest.splits) {
                for id in ids {
                    if expected.insert(id.as_str(), split).is_some() { return Err(AssayError::SplitMembership); }
                }
            }
            let mut seen = BTreeSet::new();
            let mut units = BTreeSet::new();
            for line in text.split_terminator('\n') {
                let line = line.strip_suffix('\r').unwrap_or(line);
                let row: Observation = strict_json(line.as_bytes(), 256)?;
                if row.schema_version != ASSAY_SCHEMA_VERSION || !identifier(&row.id)
                    || !row.probability.is_finite() || !(0.0..=1.0).contains(&row.probability) {
                    return Err(AssayError::Observation);
                }
                if expected.get(row.id.as_str()) != Some(&row.split) { return Err(AssayError::SplitMembership); }
                if !seen.insert(row.id.clone()) { return Err(AssayError::DuplicateId); }
                if !units.insert(row.unit_sha256) { return Err(AssayError::DuplicateUnit); }
                rows.push(row);
            }
            if seen.len() != expected.len() { return Err(AssayError::SplitMembership); }
        }
        rows.sort_by(|a, b| a.id.cmp(&b.id));
        let membership = SplitMembership::new(manifest.splits.development.clone(),
            manifest.splits.calibration.clone(), manifest.splits.locked_test.clone())
            .map_err(|_| AssayError::SplitMembership)?;
        let mut development = reserved(manifest.splits.development.len())?;
        let mut calibration = reserved(manifest.splits.calibration.len())?;
        let mut locked_test = reserved(manifest.splits.locked_test.len())?;
        for row in &rows {
            let score = LabeledScore::new(row.id.clone(), row.probability, row.positive)
                .map_err(|_| AssayError::Observation)?;
            match row.split {
                Split::Development => development.push(score),
                Split::Calibration => calibration.push(score),
                Split::LockedTest => locked_test.push(score),
            }
        }
        let partition = membership.partition(development, calibration, locked_test)
            .map_err(|_| AssayError::SplitMembership)?;
        Ok(Self { manifest, rows, partition, input_digest, manifest_bytes: manifest_bytes.len(),
            input_bytes: observations.len(), max_line_bytes })
    }
    pub fn manifest(&self) -> &SuiteManifest { &self.manifest }
    pub fn partition(&self) -> &CalibrationPartition { &self.partition }
    /// A caller may tighten limits between parsing and fitting/reporting.
    pub fn validate_limits(&self, limits: Limits) -> Result<(), AssayError> {
        limits.validate()?;
        if self.manifest_bytes > limits.max_manifest_bytes || self.input_bytes > limits.max_input_bytes
            || self.max_line_bytes > limits.max_line_bytes || self.rows.len() > limits.max_rows {
            return Err(AssayError::InputLimit);
        }
        Ok(())
    }
    /// Stable UTF-8 ID order, regardless of the NDJSON physical row ordering.
    pub fn rows(&self) -> &[Observation] { &self.rows }
    pub fn rows_for_split(&self, split: Split) -> impl Iterator<Item = &Observation> {
        self.rows.iter().filter(move |row| row.split == split)
    }

    /// Public HMAC bindings; the raw private digests and ID lists never leave
    /// this boundary. Input commits to its checked SHA-256 digest's canonical
    /// lowercase hexadecimal bytes using an explicit domain, rather than
    /// retaining a second full input buffer.
    pub fn bindings(&self, key: &CommitmentKey, namespace: &str) -> Result<DatasetBindings, AssayError> {
        if !identifier(namespace) { return Err(AssayError::Commitment); }
        let suite = crate::canonjson::canonical_bytes(&self.manifest).map_err(|_| AssayError::Commitment)?;
        let identity = self.manifest.execution_identity.canonical_json_bytes().map_err(|_| AssayError::Identity)?;
        let split = |name: &str, ids: &[String]| {
            let bytes = crate::canonjson::canonical_bytes(ids).map_err(|_| AssayError::Commitment)?;
            key.commit(CommitmentDomain::Configuration, namespace, name, &bytes).map_err(|_| AssayError::Commitment)
        };
        Ok(DatasetBindings {
            metadata: DatasetMetadata {
                dataset_id: self.manifest.dataset_id.clone(),
                dataset_version: self.manifest.dataset_version.clone(),
                named_population: self.manifest.named_population.clone(),
                positive_label: self.manifest.positive_label.clone(),
                negative_label: self.manifest.negative_label.clone(),
                task_spec: self.manifest.execution_identity.task_spec.clone(),
                model_recipe_id: self.manifest.execution_identity.quant_recipe.clone(),
                prompt_template_sha256: self.manifest.execution_identity.template_digest,
                numerics_profile: self.manifest.execution_identity.numerics_profile.label(),
                thinking_mode: match self.manifest.execution_identity.thinking_mode {
                    ThinkingMode::Disabled => "disabled",
                    ThinkingMode::Enabled => "enabled",
                }.to_owned(),
            },
            suite: key.commit(CommitmentDomain::Configuration, namespace, "binary_event_suite_v1", &suite)
                .map_err(|_| AssayError::Commitment)?,
            input: key.commit(CommitmentDomain::Input, namespace, "observations_sha256_hex_v1", self.input_digest.to_hex().as_bytes())
                .map_err(|_| AssayError::Commitment)?,
            execution_identity: key.commit(CommitmentDomain::ExecutionIdentity, namespace, "binary_event_subject_v1", &identity)
                .map_err(|_| AssayError::Commitment)?,
            splits: SplitBindings {
                development: split("development_ids_v1", &self.manifest.splits.development)?,
                calibration: split("calibration_ids_v1", &self.manifest.splits.calibration)?,
                locked_test: split("locked_test_ids_v1", &self.manifest.splits.locked_test)?,
            },
        })
    }
}

/// Caller-declared public scope accompanies every scorecard. The template hash
/// identifies trusted template data; private prompt, schema, TaskIR and input
/// hashes remain only inside the keyed commitments below.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetMetadata {
    pub dataset_id: String,
    pub dataset_version: String,
    pub named_population: String,
    pub positive_label: String,
    pub negative_label: String,
    pub task_spec: String,
    pub model_recipe_id: String,
    pub prompt_template_sha256: Sha256Digest,
    pub numerics_profile: String,
    pub thinking_mode: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SplitBindings {
    pub development: ContentCommitment,
    pub calibration: ContentCommitment,
    pub locked_test: ContentCommitment,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetBindings {
    pub metadata: DatasetMetadata,
    pub suite: ContentCommitment,
    pub input: ContentCommitment,
    pub execution_identity: ContentCommitment,
    pub splits: SplitBindings,
}

fn validate_manifest(manifest: &SuiteManifest, limits: Limits) -> Result<usize, AssayError> {
    if manifest.schema_version != ASSAY_SCHEMA_VERSION || manifest.task != "binary_event"
        || !identifier(&manifest.dataset_id) || !identifier(&manifest.dataset_version)
        || !label(&manifest.named_population, 1024) || !label(&manifest.positive_label, 256)
        || !label(&manifest.negative_label, 256) || manifest.positive_label == manifest.negative_label {
        return Err(AssayError::Manifest);
    }
    manifest.execution_identity.validate().map_err(|_| AssayError::Identity)?;
    // Identity validation enforces semantic requirements; this user-data
    // boundary additionally excludes controls and unbounded display strings.
    let identity = &manifest.execution_identity;
    for value in [&identity.source_revision, &identity.artifact_format, &identity.quant_recipe,
        &identity.task_spec, &identity.grammar_compiler_version, &identity.kv_dtype,
        &identity.sampler_version, &identity.backend_semantic_version] {
        if !label(value, 256) { return Err(AssayError::Identity); }
    }
    for value in [&identity.host_class, &identity.compiler_identity].into_iter().flatten() {
        if !label(value, 1024) { return Err(AssayError::Identity); }
    }
    let mut total = 0_usize;
    let mut seen = BTreeSet::new();
    for (_, ids) in split_lists(&manifest.splits) {
        if ids.is_empty() { return Err(AssayError::SplitMembership); }
        total = total.checked_add(ids.len()).filter(|&n| n <= limits.max_rows).ok_or(AssayError::InputLimit)?;
        for id in ids {
            if !identifier(id) || !seen.insert(id.as_str()) { return Err(AssayError::SplitMembership); }
        }
    }
    Ok(total)
}
fn split_lists(ids: &SplitIds) -> [(Split, &[String]); 3] {
    [(Split::Development, &ids.development), (Split::Calibration, &ids.calibration), (Split::LockedTest, &ids.locked_test)]
}
fn identifier(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && !value.chars().any(|ch| ch.is_control() || ch.is_whitespace())
}
fn label(value: &str, cap: usize) -> bool {
    !value.trim().is_empty() && value.trim() == value && value.len() <= cap && !value.chars().any(char::is_control)
}
fn strict_json<T: serde::de::DeserializeOwned>(bytes: &[u8], max_string_bytes: usize) -> Result<T, AssayError> {
    let text = std::str::from_utf8(bytes).map_err(|_| AssayError::Json)?;
    let value = crate::canonjson::parse_str_with_limits(text,
        crate::canonjson::ParseLimits { max_depth: 16, max_string_bytes }).map_err(|_| AssayError::Json)?;
    serde_json::from_value(value).map_err(|_| AssayError::Json)
}
fn reserved<T>(count: usize) -> Result<Vec<T>, AssayError> {
    let mut out = Vec::new(); out.try_reserve_exact(count).map_err(|_| AssayError::Allocation)?; Ok(out)
}
