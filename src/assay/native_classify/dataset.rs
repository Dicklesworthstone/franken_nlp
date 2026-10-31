//! Strict task-shaped source input; scores and execution identities are absent.
use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};
use crate::{execution_identity::Sha256Digest,
    tasks::{classify::{ClassificationLabel, ClassificationMode, ClassificationPolicy, ClassificationRequest},
        ir::TaskBudget}};
use super::*;

/// The closed task taxonomy of native classification Assay. Multilabel scores
/// are independent binary heads, never an exclusive distribution over labels.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind { Binary, Multiclass, Multilabel }

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteManifest {
    pub schema_version: u32,
    pub task: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub named_population: String,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "present")]
    pub positive_label: Option<ClassificationLabel>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "present")]
    pub negative_label: Option<ClassificationLabel>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<ClassificationLabel>,
    /// Candidate-relative task thresholds; distinct from evaluation acceptance.
    #[serde(default)]
    pub policy: ClassificationPolicy,
    pub expected_input_sha256: Sha256Digest,
    pub splits: SplitIds,
}
impl SuiteManifest {
    pub fn task_kind(&self) -> Result<TaskKind, AssayError> {
        match self.task.as_str() {
            NATIVE_TASK => Ok(TaskKind::Binary),
            MULTICLASS_TASK => Ok(TaskKind::Multiclass),
            MULTILABEL_TASK => Ok(TaskKind::Multilabel),
            _ => Err(AssayError::Manifest),
        }
    }
    pub fn is_binary(&self) -> bool { self.task == NATIVE_TASK }
    pub fn label_count(&self) -> usize { if self.is_binary() { 2 } else { self.labels.len() } }
    pub fn head_count(&self) -> usize { if self.task == MULTILABEL_TASK { self.labels.len() } else { 1 } }
    /// Binary order is positive then negative. The parser sorts the other two
    /// label spaces once, so replay vectors require no per-row label strings.
    pub fn labels(&self) -> impl Iterator<Item = &ClassificationLabel> {
        self.positive_label.iter().chain(self.negative_label.iter()).chain(self.labels.iter())
    }
}

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub schema_version: u32,
    pub id: String,
    pub split: Split,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "present")]
    pub positive: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "present")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "present")]
    pub labels: Option<Vec<String>>,
}

/// Parsing seals the exact source pin, complete membership and global source
/// units. Public readers cannot replace rows or supply model predictions.
pub struct Dataset {
    pub(super) manifest: SuiteManifest,
    rows: Vec<Row>,
    pub(super) manifest_digest: Sha256Digest,
    pub(super) input_digest: Sha256Digest,
    pub(super) sizes: InputSizes,
    kind: TaskKind,
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
        let value: serde_json::Value = strict_json(manifest_bytes, 4096)?;
        // An empty extra label set is still a foreign task field. Check its
        // presence before serde's default-empty vector erases that distinction.
        if value.get("task").and_then(serde_json::Value::as_str) == Some(NATIVE_TASK)
            && value.get("labels").is_some() { return Err(AssayError::Manifest); }
        let mut manifest: SuiteManifest = serde_json::from_value(value).map_err(|_| AssayError::Json)?;
        let total = validate_manifest(&manifest, limits)?;
        let kind = manifest.task_kind()?;
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
        manifest.labels.sort_by(|a, b| a.id.cmp(&b.id));
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
                let mut row: Row = strict_json(line.strip_suffix('\r').unwrap_or(line).as_bytes(), limits.max_line_bytes)?;
                if row.schema_version != ASSAY_SCHEMA_VERSION || !identifier(&row.id) || row.text.is_empty() {
                    return Err(AssayError::Observation);
                }
                validate_gold(&mut row, &manifest, kind)?;
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
        Ok(Self { manifest, rows, manifest_digest, input_digest, kind,
            sizes: InputSizes { manifest: manifest_bytes.len(), input: input.len(), line: max_line, rows: total } })
    }
    pub fn manifest(&self) -> &SuiteManifest { &self.manifest }
    pub fn rows(&self) -> &[Row] { &self.rows }
    pub fn len(&self) -> usize { self.rows.len() }
    pub fn is_empty(&self) -> bool { self.rows.is_empty() }
    pub fn locked_test_len(&self) -> usize { self.manifest.splits.locked_test.len() }
    pub fn validate_limits(&self, limits: Limits) -> Result<(), AssayError> { self.sizes.validate(limits) }
    pub fn task_kind(&self) -> TaskKind { self.kind }
    pub fn is_binary(&self) -> bool { self.kind == TaskKind::Binary }
    pub fn label_count(&self) -> usize { self.manifest.label_count() }
    pub fn head_count(&self) -> usize { self.manifest.head_count() }

    pub(super) fn gold(&self, index: usize) -> Result<Vec<bool>, AssayError> {
        let row = self.rows.get(index).ok_or(AssayError::Scope)?;
        let mut values = reserved(self.label_count())?;
        match self.kind {
            TaskKind::Binary => {
                let positive = row.positive.ok_or(AssayError::Observation)?;
                values.extend([positive, !positive]);
            }
            TaskKind::Multiclass => {
                let id = row.label.as_deref().ok_or(AssayError::Observation)?;
                values.extend(self.manifest.labels.iter().map(|label| label.id == id));
            }
            TaskKind::Multilabel => {
                let labels = row.labels.as_deref().ok_or(AssayError::Observation)?;
                values.extend(self.manifest.labels.iter().map(|label|
                    labels.binary_search(&label.id).is_ok()));
            }
        }
        Ok(values)
    }

    /// The host passes its one admitted task budget. Only source bytes, the
    /// declared labels and the bounded native task policy enter the model.
    /// Human gold labels and evaluation splits never enter its prompt.
    pub fn classification_request(&self, index: usize, budget: TaskBudget)
        -> Result<ClassificationRequest, AssayError> {
        budget.validate().map_err(|_| AssayError::Limits)?;
        let row = self.rows.get(index).ok_or(AssayError::Scope)?;
        let mut labels = reserved(self.label_count())?;
        labels.extend(self.manifest.labels().cloned());
        Ok(ClassificationRequest { document: row.text.clone(), labels,
            mode: if self.kind == TaskKind::Multilabel { ClassificationMode::MultiLabel }
                else { ClassificationMode::Exclusive }, policy: self.manifest.policy, budget })
    }
}

fn validate_manifest(manifest: &SuiteManifest, limits: Limits) -> Result<usize, AssayError> {
    let kind = manifest.task_kind()?;
    if manifest.schema_version != ASSAY_SCHEMA_VERSION
        || !identifier(&manifest.dataset_id) || !identifier(&manifest.dataset_version)
        || !label(&manifest.named_population, 1024)
        || manifest.policy.minimum_candidate_weight_ppm > 1_000_000
        || manifest.policy.minimum_margin_ppm > 1_000_000 {
        return Err(AssayError::Manifest);
    }
    match kind {
        TaskKind::Binary => {
            if manifest.positive_label.is_none() || manifest.negative_label.is_none()
                || !manifest.labels.is_empty() { return Err(AssayError::Manifest); }
        }
        TaskKind::Multiclass | TaskKind::Multilabel => {
            let minimum = if kind == TaskKind::Multiclass { 2 } else { 1 };
            if manifest.positive_label.is_some() || manifest.negative_label.is_some()
                || !(minimum..=MAX_LABELS).contains(&manifest.labels.len()) {
                return Err(AssayError::Manifest);
            }
        }
    }
    let mut labels = BTreeSet::new();
    for item in manifest.labels() {
        if !label(&item.id, 256) || item.description.len() > 4096 || !labels.insert(item.id.as_str()) {
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

fn validate_gold(row: &mut Row, manifest: &SuiteManifest, kind: TaskKind) -> Result<(), AssayError> {
    let known = |id: &str| manifest.labels.binary_search_by(|label| label.id.as_str().cmp(id)).is_ok();
    match kind {
        TaskKind::Binary => {
            if row.positive.is_none() || row.label.is_some() || row.labels.is_some() {
                return Err(AssayError::Observation);
            }
        }
        TaskKind::Multiclass => {
            if row.positive.is_some() || row.labels.is_some()
                || !row.label.as_deref().is_some_and(known) { return Err(AssayError::Observation); }
        }
        TaskKind::Multilabel => {
            if row.positive.is_some() || row.label.is_some() { return Err(AssayError::Observation); }
            let labels = row.labels.as_mut().ok_or(AssayError::Observation)?;
            if labels.len() > manifest.labels.len() { return Err(AssayError::Observation); }
            labels.sort();
            if labels.iter().any(|id| !known(id)) || labels.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(AssayError::Observation);
            }
        }
    }
    Ok(())
}

/// Missing optional task fields are fine; explicitly present null is not a
/// second spelling for an absent field in this strict task-shaped protocol.
fn present<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where T: Deserialize<'de>, D: serde::Deserializer<'de> {
    T::deserialize(deserializer).map(Some)
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
