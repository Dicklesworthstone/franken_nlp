//! Strict NER gold over exact UTF-8 source bytes, without supplied predictions.
use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};
use crate::{execution_identity::Sha256Digest,
    tasks::{ir::TaskBudget, ner::{EntityType, NerOptions}, source_planning::SourceTaskRequest},
    validation::grounded_fields::VerifiedSourceSpan};
use super::*;

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteManifest {
    pub schema_version: u32,
    pub task: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub named_population: String,
    pub options: NerOptions,
    pub expected_input_sha256: Sha256Digest,
    pub splits: SplitIds,
}

/// Gold uses exact byte intervals. Unicode scalar coordinates are computed
/// from the original source by Dataset, never accepted from caller metadata.
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GoldSpan {
    pub entity_type: EntityType,
    pub byte_start: usize,
    pub byte_end: usize,
    #[serde(skip)]
    scalar_start: usize,
    #[serde(skip)]
    scalar_end: usize,
}
impl GoldSpan {
    pub fn source_span(&self) -> VerifiedSourceSpan {
        VerifiedSourceSpan { byte_start: self.byte_start, byte_end: self.byte_end,
            scalar_start: self.scalar_start, scalar_end: self.scalar_end }
    }
}

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub schema_version: u32,
    pub id: String,
    pub split: Split,
    pub text: String,
    /// Owner-declared language metadata, not a model language diagnosis.
    pub language: String,
    pub spans: Vec<GoldSpan>,
}

/// The complete corpus is validated and pinned. Development and calibration
/// rows remain private binding inputs; this evaluator executes locked test.
pub struct Dataset {
    pub(super) manifest: SuiteManifest,
    rows: Vec<Row>,
    replay_start: usize,
    languages: Vec<String>,
    max_gold_spans: usize,
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
    /// All caller geometry is checked before typed gold vectors are allocated.
    /// A bounded strict JSON tree is used to inspect each vector's length first.
    pub fn parse(manifest_bytes: &[u8], input: &[u8], limits: Limits, ner_limits: NerLimits)
        -> Result<Self, AssayError> {
        limits.validate()?; ner_limits.validate()?;
        if manifest_bytes.is_empty() || manifest_bytes.len() > limits.max_manifest_bytes
            || input.is_empty() || input.len() > limits.max_input_bytes {
            return Err(AssayError::InputLimit);
        }
        let mut manifest: SuiteManifest = strict_json(manifest_bytes, 4096)?;
        let total = validate_manifest(&manifest, limits)?;
        if manifest.options.max_entities > ner_limits.max_spans_per_row {
            return Err(AssayError::InputLimit);
        }
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
        manifest.options.types.sort_by_key(|kind| kind.label());
        let manifest_digest = Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&manifest)
            .map_err(|_| AssayError::Manifest)?);
        let mut rows = reserved(total)?;
        let mut languages = BTreeSet::new(); let mut max_gold_spans = 0;
        {
            let mut expected = BTreeMap::new();
            for (split, ids) in split_lists(&manifest.splits) {
                for id in ids { expected.insert(id.as_str(), split); }
            }
            let mut seen = BTreeSet::new(); let mut units = BTreeSet::new();
            for line in text.split_terminator('\n') {
                let value: serde_json::Value = strict_json(line.strip_suffix('\r').unwrap_or(line).as_bytes(),
                    limits.max_line_bytes)?;
                let spans = value.get("spans").and_then(serde_json::Value::as_array).ok_or(AssayError::Observation)?;
                if spans.len() > ner_limits.max_spans_per_row { return Err(AssayError::InputLimit); }
                // Distinct language geometry is also refused before converting
                // this row into source strings and owned typed gold storage.
                let language = value.get("language").and_then(serde_json::Value::as_str).ok_or(AssayError::Observation)?;
                if !language_tag(language) { return Err(AssayError::Observation); }
                if !languages.contains(language) && languages.len() == ner_limits.max_languages {
                    return Err(AssayError::InputLimit);
                }
                languages.insert(language.to_owned());
                let mut row: Row = serde_json::from_value(value).map_err(|_| AssayError::Json)?;
                if row.schema_version != ASSAY_SCHEMA_VERSION || !identifier(&row.id) || row.text.is_empty() {
                    return Err(AssayError::Observation);
                }
                validate_gold(&mut row, &manifest.options)?;
                if expected.get(row.id.as_str()) != Some(&row.split) { return Err(AssayError::SplitMembership); }
                if !seen.insert(row.id.clone()) { return Err(AssayError::DuplicateId); }
                if !units.insert(Sha256Digest::of_bytes(row.text.as_bytes())) { return Err(AssayError::DuplicateUnit); }
                max_gold_spans = max_gold_spans.max(row.spans.len());
                rows.push(row);
            }
            if seen.len() != expected.len() { return Err(AssayError::SplitMembership); }
        }
        rows.sort_by(|a, b| a.split.cmp(&b.split).then_with(|| a.id.cmp(&b.id)));
        let replay_start = rows.len().checked_sub(manifest.splits.locked_test.len()).ok_or(AssayError::SplitMembership)?;
        let mut language_list = reserved(languages.len())?; language_list.extend(languages);
        Ok(Self { manifest, rows, replay_start, languages: language_list, max_gold_spans,
            manifest_digest, input_digest,
            sizes: InputSizes { manifest: manifest_bytes.len(), input: input.len(), line: max_line, rows: total } })
    }
    pub fn manifest(&self) -> &SuiteManifest { &self.manifest }
    pub fn rows(&self) -> &[Row] { &self.rows }
    pub fn replay_rows(&self) -> &[Row] { &self.rows[self.replay_start..] }
    pub fn len(&self) -> usize { self.rows.len() }
    pub fn is_empty(&self) -> bool { self.rows.is_empty() }
    pub fn replay_len(&self) -> usize { self.rows.len() - self.replay_start }
    pub fn locked_test_len(&self) -> usize { self.replay_len() }
    pub fn languages(&self) -> &[String] { &self.languages }
    pub fn validate_limits(&self, limits: Limits, ner_limits: NerLimits) -> Result<(), AssayError> {
        self.sizes.validate(limits)?; ner_limits.validate()?;
        if self.max_gold_spans > ner_limits.max_spans_per_row
            || self.manifest.options.max_entities > ner_limits.max_spans_per_row
            || self.languages.len() > ner_limits.max_languages { return Err(AssayError::InputLimit); }
        Ok(())
    }
    pub(super) fn kind_index(&self, kind: EntityType) -> Result<usize, AssayError> {
        self.manifest.options.types.binary_search_by_key(&kind.label(), |value| value.label())
            .map_err(|_| AssayError::Observation)
    }
    pub(super) fn request_digest(&self, replay_index: usize, budget: TaskBudget) -> Result<Sha256Digest, AssayError> {
        let row = self.replay_rows().get(replay_index).ok_or(AssayError::Scope)?;
        // This is the exact tagged SourceTaskRequest::Ner wire shape. Borrow
        // source/options during canonicalization instead of cloning them.
        #[derive(Serialize)]
        struct Request<'a> { task: &'static str, document: &'a str, options: &'a NerOptions, budget: TaskBudget }
        let bytes = crate::canonjson::canonical_bytes(&Request { task: "ner", document: &row.text,
            options: &self.manifest.options, budget }).map_err(|_| AssayError::Commitment)?;
        Ok(Sha256Digest::of_bytes(&bytes))
    }
    /// Neither held-out gold, row IDs, language labels nor split membership is
    /// forwarded into the task's trusted instruction or untrusted source.
    pub fn source_request(&self, replay_index: usize, budget: TaskBudget) -> Result<SourceTaskRequest, AssayError> {
        budget.validate().map_err(|_| AssayError::Limits)?;
        let row = self.replay_rows().get(replay_index).ok_or(AssayError::Scope)?;
        Ok(SourceTaskRequest::Ner { document: row.text.clone(), options: self.manifest.options.clone(), budget })
    }
}

fn validate_manifest(manifest: &SuiteManifest, limits: Limits) -> Result<usize, AssayError> {
    if manifest.schema_version != ASSAY_SCHEMA_VERSION || manifest.task != NATIVE_TASK
        || !identifier(&manifest.dataset_id) || !identifier(&manifest.dataset_version)
        || !label(&manifest.named_population, 1024) || manifest.options.validate().is_err() {
        return Err(AssayError::Manifest);
    }
    let mut total = 0_usize; let mut seen = BTreeSet::new();
    for (_, ids) in split_lists(&manifest.splits) {
        if ids.is_empty() { return Err(AssayError::SplitMembership); }
        total = total.checked_add(ids.len()).filter(|&n| n <= limits.max_rows).ok_or(AssayError::InputLimit)?;
        for id in ids { if !identifier(id) || !seen.insert(id.as_str()) { return Err(AssayError::SplitMembership); } }
    }
    Ok(total)
}

fn validate_gold(row: &mut Row, options: &NerOptions) -> Result<(), AssayError> {
    // One source pass supplies every scalar coordinate. Repeatedly scanning
    // source prefixes would make many nested/overlapping gold spans quadratic.
    let mut boundaries = reserved(row.text.len().checked_add(1).ok_or(AssayError::InputLimit)?)?;
    boundaries.extend(row.text.char_indices().map(|(byte, _)| byte)); boundaries.push(row.text.len());
    for span in &mut row.spans {
        if span.byte_start >= span.byte_end || !options.types.contains(&span.entity_type) {
            return Err(AssayError::Observation);
        }
        span.scalar_start = boundaries.binary_search(&span.byte_start).map_err(|_| AssayError::Observation)?;
        span.scalar_end = boundaries.binary_search(&span.byte_end).map_err(|_| AssayError::Observation)?;
    }
    row.spans.sort_by_key(|span| (span.byte_start, span.byte_end, span.entity_type));
    if row.spans.windows(2).any(|pair|
        pair[0].byte_start == pair[1].byte_start && pair[0].byte_end == pair[1].byte_end) {
        return Err(AssayError::Observation);
    }
    Ok(())
}
fn language_tag(value: &str) -> bool {
    !value.is_empty() && value.len() <= 63
        && value.split('-').all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_alphanumeric()))
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
