//! Exact private schema/gold storage, validated before any model is loaded.
use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};
use crate::{assay::field_metrics::count_with_checkpoint,
    batch::{BatchDocument, extract::{ExtractionBatchArgs, ExtractionBatchGrounding}},
    execution_identity::Sha256Digest,
    grammar::{CompileLimits, SchemaNode, SourceAnnotation, runtime::{JsonValidationSchema, SourceRuntimeLimits},
        source_index::SourceLanguage},
    tasks::ir::TaskBudget,
    validation::{JsonLimits, JsonParseErrorKind, JsonValue,
        grounded_fields::{FieldGroundingError, GroundingBudget, SourceFieldEvidence, SourceOccurrence,
            scan_occurrences}}};
use super::*;

#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SchemaDefinition {
    pub id: String,
    /// Exact schema JSON, including decimals outside binary-float precision.
    pub schema_json: String,
    pub grounding: ExtractionBatchGrounding,
}
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteManifest {
    pub schema_version: u32,
    pub task: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub named_population: String,
    pub schemas: Vec<SchemaDefinition>,
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
    pub schema_id: String,
    /// Owner-declared population metadata, never a model diagnosis.
    pub language: String,
    pub expected_json: String,
    #[serde(skip)]
    gold: Option<JsonValue>,
}

pub struct Dataset {
    pub(super) manifest: SuiteManifest,
    rows: Vec<Row>,
    replay_start: usize,
    languages: Vec<String>,
    extract_limits: ExtractLimits,
    compiler: CompileLimits,
    source: SourceRuntimeLimits,
    source_field_bounds: Vec<usize>,
    gold_verification: VerificationCharge,
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
    /// Catalogue caps precede typed schemas; exact JSON nodes are admitted
    /// during parsing. Gold uses one aggregate source-work allowance across
    /// all splits. Only parsed values survive; schema automata do not.
    pub fn parse(manifest_bytes: &[u8], input: &[u8], limits: Limits, extract: ExtractLimits,
        compiler: CompileLimits, source: SourceRuntimeLimits) -> Result<Self, AssayError> {
        validate_configuration(limits, extract, compiler, source)?;
        if manifest_bytes.is_empty() || manifest_bytes.len() > limits.max_manifest_bytes
            || input.is_empty() || input.len() > limits.max_input_bytes { return Err(AssayError::InputLimit); }
        let raw: serde_json::Value = strict_json(manifest_bytes, limits.max_manifest_bytes)?;
        let schemas = raw.get("schemas").and_then(serde_json::Value::as_array).ok_or(AssayError::Manifest)?;
        if schemas.is_empty() || schemas.len() > extract.max_schemas { return Err(AssayError::InputLimit); }
        for schema in schemas {
            let text = schema.get("schema_json").and_then(serde_json::Value::as_str).ok_or(AssayError::Manifest)?;
            if text.is_empty() || text.len() > compiler.max_schema_bytes { return Err(AssayError::InputLimit); }
        }
        let mut manifest: SuiteManifest = serde_json::from_value(raw).map_err(|_| AssayError::Json)?;
        let total = validate_manifest(&manifest, limits)?;
        manifest.schemas.sort_by(|a, b| a.id.cmp(&b.id));
        let mut schemas: Vec<JsonValidationSchema> = reserved(manifest.schemas.len())?;
        let mut source_field_bounds = reserved(manifest.schemas.len())?;
        for definition in &manifest.schemas {
            let compiled = JsonValidationSchema::compile(&definition.schema_json, compiler).map_err(|_| AssayError::Manifest)?;
            if (definition.grounding == ExtractionBatchGrounding::SourceMembership) != compiled.requires_verbatim_source() {
                return Err(AssayError::Manifest);
            }
            source_field_bounds.push(source_field_bound(compiled.root(), source.verification.max_fields));
            schemas.push(compiled);
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
        let manifest_digest = Sha256Digest::of_bytes(&crate::canonjson::canonical_bytes(&manifest)
            .map_err(|_| AssayError::Manifest)?);
        let mut rows = reserved(total)?; let mut languages = BTreeSet::new();
        let mut gold_verification = VerificationCharge::default();
        {
            let mut expected = BTreeMap::new();
            for (split, ids) in split_lists(&manifest.splits) {
                for id in ids { expected.insert(id.as_str(), split); }
            }
            let mut seen = BTreeSet::new(); let mut units = BTreeSet::new();
            for line in text.split_terminator('\n') {
                let raw: serde_json::Value = strict_json(line.strip_suffix('\r').unwrap_or(line).as_bytes(), limits.max_line_bytes)?;
                let language = raw.get("language").and_then(serde_json::Value::as_str).ok_or(AssayError::Observation)?;
                if !language_tag(language) { return Err(AssayError::Observation); }
                if !languages.contains(language) && languages.len() == extract.max_languages { return Err(AssayError::InputLimit); }
                languages.insert(language.to_owned());
                let mut row: Row = serde_json::from_value(raw).map_err(|_| AssayError::Json)?;
                if row.schema_version != ASSAY_SCHEMA_VERSION || !identifier(&row.id) || row.text.is_empty()
                    || !identifier(&row.schema_id) { return Err(AssayError::Observation); }
                if expected.get(row.id.as_str()) != Some(&row.split) { return Err(AssayError::SplitMembership); }
                if !seen.insert(row.id.clone()) { return Err(AssayError::DuplicateId); }
                if !units.insert(Sha256Digest::of_bytes(row.text.as_bytes())) { return Err(AssayError::DuplicateUnit); }
                let schema_index = manifest.schemas.binary_search_by(|schema| schema.id.cmp(&row.schema_id))
                    .map_err(|_| AssayError::Observation)?;
                let gold = parse_value(&row.expected_json, compiler.max_output_bytes, extract)?;
                count_with_checkpoint(&gold, extract.max_fields_per_row, extract.max_json_nodes_per_row,
                    extract.max_json_depth, &mut || Ok::<(), AssayError>(()))?;
                schemas[schema_index].validate_value(&gold).map_err(|_| AssayError::Observation)?;
                if manifest.schemas[schema_index].grounding == ExtractionBatchGrounding::SourceMembership {
                    SourceLanguage::preflight(&row.text, source.index).map_err(|_| AssayError::InputLimit)?;
                    let remaining = extract.max_verification_steps.checked_sub(gold_verification.steps)
                        .ok_or(AssayError::InputLimit)?;
                    let mut verification = GroundingBudget {
                        max_fields: source.verification.max_fields,
                        max_matches: source.verification.max_matches.min(source.index.max_matches),
                        max_scan_steps: source.verification.max_scan_steps.min(remaining),
                    };
                    let before = verification;
                    verify_gold(schemas[schema_index].root(), &gold, &row.text, &mut verification, 0)?;
                    gold_verification = gold_verification.add(VerificationCharge {
                        fields: before.max_fields - verification.max_fields,
                        matches: before.max_matches - verification.max_matches,
                        steps: before.max_scan_steps - verification.max_scan_steps,
                    })?;
                }
                row.gold = Some(gold); rows.push(row);
            }
            if seen.len() != expected.len() { return Err(AssayError::SplitMembership); }
        }
        rows.sort_by(|a, b| a.split.cmp(&b.split).then_with(|| a.id.cmp(&b.id)));
        let replay_start = rows.len().checked_sub(manifest.splits.locked_test.len()).ok_or(AssayError::SplitMembership)?;
        let mut language_list = reserved(languages.len())?; language_list.extend(languages);
        Ok(Self { manifest, rows, replay_start, languages: language_list, extract_limits: extract, compiler, source, source_field_bounds,
            gold_verification, manifest_digest, input_digest,
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
    pub fn gold_verification_steps(&self) -> u64 { self.gold_verification.steps }
    pub fn gold_verification_fields(&self) -> usize { self.gold_verification.fields }
    pub fn gold_verification_matches(&self) -> usize { self.gold_verification.matches }
    pub fn validate_limits(&self, limits: Limits, extract: ExtractLimits, compiler: CompileLimits,
        source: SourceRuntimeLimits) -> Result<(), AssayError> {
        validate_configuration(limits, extract, compiler, source)?; self.sizes.validate(limits)?;
        // Parsed gold and source preflight belong to this exact configuration.
        // A changed configuration must be admitted by parsing the corpus again.
        if extract != self.extract_limits || compiler != self.compiler || source != self.source {
            return Err(AssayError::Limits);
        }
        Ok(())
    }
    pub fn schema_for_row(&self, replay_index: usize) -> Result<&SchemaDefinition, AssayError> {
        let row = self.replay_rows().get(replay_index).ok_or(AssayError::Scope)?;
        let index = self.manifest.schemas.binary_search_by(|schema| schema.id.cmp(&row.schema_id))
            .map_err(|_| AssayError::Manifest)?;
        Ok(&self.manifest.schemas[index])
    }
    /// The exact schema's maximum reachable annotated strings, clamped to
    /// the independently admitted row cap. Optional keys count as present;
    /// array multiplication uses the compiler's finite item bound.
    pub fn max_source_fields(&self, replay_index: usize) -> Result<usize, AssayError> {
        let row = self.replay_rows().get(replay_index).ok_or(AssayError::Scope)?;
        let index = self.manifest.schemas.binary_search_by(|schema| schema.id.cmp(&row.schema_id))
            .map_err(|_| AssayError::Manifest)?;
        self.source_field_bounds.get(index).copied().ok_or(AssayError::Manifest)
    }
    /// Gold, split and language never enter the extraction request. The item
    /// ID associates host work; the batch compiler does not include it in text.
    pub fn extraction_request(&self, index: usize, budget: TaskBudget)
        -> Result<BatchDocument<ExtractionBatchArgs>, AssayError> {
        budget.validate().map_err(|_| AssayError::Limits)?;
        let row = self.replay_rows().get(index).ok_or(AssayError::Scope)?;
        let schema = self.schema_for_row(index)?;
        Ok(BatchDocument { id: row.id.clone(), text: row.text.clone(), task_args: Some(ExtractionBatchArgs {
            schema: schema.schema_json.clone(), grounding: schema.grounding, budget }) })
    }
    pub(super) fn request_digest(&self, index: usize, budget: TaskBudget) -> Result<Sha256Digest, AssayError> {
        let row = self.replay_rows().get(index).ok_or(AssayError::Scope)?; let schema = self.schema_for_row(index)?;
        crate::batch::extract::grouped_int4::extraction_request_digest(&row.text, &schema.schema_json,
            schema.grounding, budget).map_err(|_| AssayError::Commitment)
    }
    pub(super) fn gold(&self, index: usize) -> Result<&JsonValue, AssayError> {
        self.replay_rows().get(index).and_then(|row| row.gold.as_ref()).ok_or(AssayError::Observation)
    }
    pub(super) fn validate_replay(&self, limits: Limits, extract: ExtractLimits) -> Result<(), AssayError> {
        self.validate_limits(limits, extract, self.compiler, self.source)
    }
}

pub(super) fn parse_value(input: &str, max_bytes: usize, limits: ExtractLimits) -> Result<JsonValue, AssayError> {
    crate::validation::json::parse_json_with_node_limit(input, JsonLimits {
        max_input_bytes: max_bytes, max_depth: limits.max_json_depth,
        max_container_entries: limits.max_json_nodes_per_row, max_string_lexeme_bytes: max_bytes,
        max_number_lexeme_bytes: 512,
    }, limits.max_json_nodes_per_row).map_err(|error| match error.kind() {
        JsonParseErrorKind::Resource => AssayError::InputLimit, _ => AssayError::Json,
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct VerificationCharge { pub fields: usize, pub matches: usize, pub steps: u64 }
impl VerificationCharge {
    pub(super) fn add(self, other: Self) -> Result<Self, AssayError> {
        Ok(Self { fields: self.fields.checked_add(other.fields).ok_or(AssayError::InputLimit)?,
            matches: self.matches.checked_add(other.matches).ok_or(AssayError::InputLimit)?,
            steps: self.steps.checked_add(other.steps).ok_or(AssayError::InputLimit)? })
    }
    pub(crate) fn debit(self, budget: &mut GroundingBudget) -> Result<(), AssayError> {
        let fields = budget.max_fields.checked_sub(self.fields).ok_or(AssayError::InputLimit)?;
        let matches = budget.max_matches.checked_sub(self.matches).ok_or(AssayError::InputLimit)?;
        let steps = budget.max_scan_steps.checked_sub(self.steps).ok_or(AssayError::InputLimit)?;
        *budget = GroundingBudget { max_fields: fields, max_matches: matches, max_scan_steps: steps }; Ok(())
    }
}
/// Charge one native finalizer pass without re-running KMP. This consumes
/// actual native evidence; verify_completed independently rechecks its spans.
pub(crate) fn verification_charge(source: &str, fields: &[SourceFieldEvidence], limits: ExtractLimits)
    -> Result<VerificationCharge, AssayError> {
    if fields.len() > limits.max_fields_per_row { return Err(AssayError::InputLimit); }
    let mut charge = VerificationCharge { fields: fields.len(), ..VerificationCharge::default() };
    for field in fields {
        let first = field.spans.first().ok_or(AssayError::Observation)?;
        let length = first.byte_end.checked_sub(first.byte_start).ok_or(AssayError::Observation)?;
        if source.get(first.byte_start..first.byte_end).is_none()
            || field.occurrence != if field.spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous } {
            return Err(AssayError::Observation);
        }
        charge.matches = charge.matches.checked_add(field.spans.len()).ok_or(AssayError::InputLimit)?;
        if charge.matches > limits.max_occurrences_per_row { return Err(AssayError::InputLimit); }
        let steps = (source.len() as u64).checked_add(length as u64).and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(8)).ok_or(AssayError::InputLimit)?;
        charge.steps = charge.steps.checked_add(steps).ok_or(AssayError::InputLimit)?;
    }
    Ok(charge)
}
fn validate_configuration(limits: Limits, extract: ExtractLimits, compiler: CompileLimits,
    source: SourceRuntimeLimits) -> Result<(), AssayError> {
    limits.validate()?; extract.validate()?;
    if compiler.max_schema_bytes == 0 || compiler.max_string_bytes == 0 || compiler.max_array_items == 0
        || compiler.max_output_bytes == 0 || compiler.max_states == 0 || compiler.max_transitions == 0
        || compiler.max_mask_bytes == 0 || source.index.max_source_bytes == 0 || source.index.max_index_bytes == 0
        || source.index.max_build_steps == 0 || source.index.max_matches == 0
        || source.verification.max_fields == 0 || source.verification.max_fields > extract.max_fields_per_row
        || source.verification.max_matches == 0
        || source.verification.max_matches.min(source.index.max_matches) > extract.max_occurrences_per_row
        || source.verification.max_scan_steps == 0 || source.verification.max_scan_steps > extract.max_verification_steps {
        return Err(AssayError::Limits);
    }
    Ok(())
}
fn validate_manifest(manifest: &SuiteManifest, limits: Limits) -> Result<usize, AssayError> {
    if manifest.schema_version != ASSAY_SCHEMA_VERSION || manifest.task != NATIVE_TASK
        || !identifier(&manifest.dataset_id) || !identifier(&manifest.dataset_version)
        || !label(&manifest.named_population, 1024) { return Err(AssayError::Manifest); }
    let mut schemas = BTreeSet::new();
    for schema in &manifest.schemas {
        if !identifier(&schema.id) || !schemas.insert(schema.id.as_str()) { return Err(AssayError::Manifest); }
    }
    let mut total = 0_usize; let mut seen = BTreeSet::new();
    for (_, ids) in split_lists(&manifest.splits) {
        if ids.is_empty() { return Err(AssayError::SplitMembership); }
        total = total.checked_add(ids.len()).filter(|&n| n <= limits.max_rows).ok_or(AssayError::InputLimit)?;
        for id in ids { if !identifier(id) || !seen.insert(id.as_str()) { return Err(AssayError::SplitMembership); } }
    }
    Ok(total)
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
fn grounding_error(error: FieldGroundingError) -> AssayError {
    match error {
        FieldGroundingError::AllocationRefused => AssayError::Allocation,
        FieldGroundingError::FieldBudget | FieldGroundingError::MatchBudget | FieldGroundingError::WorkBudget => AssayError::InputLimit,
        _ => AssayError::Observation,
    }
}

fn source_field_bound(schema: &SchemaNode, cap: usize) -> usize {
    match schema {
        SchemaNode::String { source: SourceAnnotation::Verbatim, .. } => 1.min(cap),
        SchemaNode::Object { properties, .. } => properties.values().fold(0_usize,
            |count, value| count.saturating_add(source_field_bound(value, cap)).min(cap)),
        SchemaNode::Array { items, max_items } => source_field_bound(items, cap).saturating_mul(*max_items).min(cap),
        _ => 0,
    }
}

// Gold needs no published pointer evidence. Stream the independent verifier
// over schema-annotated strings so long shared property paths are never
// multiplied into retained per-field strings or occurrence vectors.
fn verify_gold(schema: &SchemaNode, value: &JsonValue, source: &str,
    verification: &mut GroundingBudget, depth: usize) -> Result<(), AssayError> {
    if depth > MAX_JSON_DEPTH { return Err(AssayError::InputLimit); }
    match (schema, value) {
        (SchemaNode::String { source: SourceAnnotation::Verbatim, .. }, JsonValue::String(text)) => {
            verification.max_fields = verification.max_fields.checked_sub(1).ok_or(AssayError::InputLimit)?;
            scan_occurrences(source, text, verification).map_err(grounding_error)?;
        }
        (SchemaNode::Object { properties, .. }, JsonValue::Object(values)) => {
            for (key, value) in values {
                verify_gold(properties.get(key).ok_or(AssayError::Observation)?, value, source, verification, depth + 1)?;
            }
        }
        (SchemaNode::Array { items, .. }, JsonValue::Array(values)) => {
            for value in values { verify_gold(items, value, source, verification, depth + 1)?; }
        }
        (SchemaNode::String { .. }, JsonValue::String(_))
        | (SchemaNode::Number { .. }, JsonValue::Number(_))
        | (SchemaNode::Boolean { .. }, JsonValue::Boolean(_))
        | (SchemaNode::Null { .. }, JsonValue::Null) => {}
        _ => return Err(AssayError::Observation),
    }
    Ok(())
}
