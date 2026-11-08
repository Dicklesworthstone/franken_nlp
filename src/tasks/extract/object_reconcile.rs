//! Exact reconciliation of explicitly partial ROOT-object extraction maps.
//! Only the root `required` set is relaxed for chunk extraction. Every nested
//! value is atomic and keeps its original schema. Conflicts and missing final
//! required fields refuse the complete document; no precedence or repair is
//! inferred. Original-schema validation and source matching are independent.
use std::{collections::BTreeMap, fmt, mem::size_of};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use crate::{canonjson, execution_identity::Sha256Digest,
    grammar::{CompileLimits, SchemaNode, SourceAnnotation, runtime::JsonValidationSchema},
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl},
    validation::{JsonValue, grounded_fields::{GroundingBudget, SourceFieldEvidence,
        SourceOccurrence, VerifiedSourceSpan, scan_occurrences}},
};
use super::array_union::{self, RootArrayUnionError, RootArrayUnionLimits, Writer};

pub const ROOT_OBJECT_DERIVATION: &str = "root-required-empty-all-child-schemas-unchanged-v1";
pub const ROOT_OBJECT_RECONCILIATION: &str = "exact-atomic-root-properties-refuse-conflicts-v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RootObjectLimits {
    pub max_properties: usize,
    pub max_origins: usize,
    /// Sum of every chunk's JSON bytes, including duplicate values.
    pub max_input_bytes: usize,
    pub max_json_bytes: usize,
    /// Nodes in all chunk values plus the final independent parse.
    pub max_nodes: usize,
    pub max_result_bytes: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RootObjectError {
    InvalidLimits, Schema, Input, Conflict, MissingRequired, Source, Limit,
    Allocation, Serialization, Cancelled(DecodeCancellationKind),
}
impl fmt::Display for RootObjectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid root-object reconciliation limits",
            Self::Schema => "root-object reconciliation requires a supported root object",
            Self::Input => "root-object map or original-schema validation refused",
            Self::Conflict => "root-object properties contain conflicting exact values",
            Self::MissingRequired => "root-object map did not provide every required property",
            Self::Source => "root-object original-source verification refused",
            Self::Limit => "complete root-object reconciliation exceeds its allowance",
            Self::Allocation => "root-object reconciliation allocation refused",
            Self::Serialization => "root-object reconciliation serialization refused",
            Self::Cancelled(_) => "root-object reconciliation cancelled",
        })
    }
}
impl std::error::Error for RootObjectError {}
impl From<RootArrayUnionError> for RootObjectError {
    fn from(error: RootArrayUnionError) -> Self { match error {
        RootArrayUnionError::InvalidLimits => Self::InvalidLimits,
        RootArrayUnionError::Schema => Self::Schema,
        RootArrayUnionError::Input => Self::Input,
        RootArrayUnionError::Limit => Self::Limit,
        RootArrayUnionError::Allocation => Self::Allocation,
        RootArrayUnionError::Serialization => Self::Serialization,
        RootArrayUnionError::Cancelled(cause) => Self::Cancelled(cause),
    } }
}
impl RootObjectLimits {
    /// Both parser/index copies, exact output text, all origins and one global
    /// source matcher coexist with the retained chunk result. No per-chunk
    /// renewal of these limits is permitted.
    pub fn workspace_bytes(self) -> Result<u64, RootObjectError> {
        let base = RootArrayUnionLimits { max_records: self.max_properties,
            max_origins: self.max_origins, max_input_bytes: self.max_input_bytes,
            max_json_bytes: self.max_json_bytes, max_nodes: self.max_nodes,
            max_result_bytes: self.max_result_bytes }.requirements()?.workspace_bytes;
        let extra = (self.max_properties as u64).checked_mul(12_288 + size_of::<RootObjectProperty>() as u64)
            .and_then(|n| n.checked_add((self.max_json_bytes as u64).checked_mul(16)?))
            .ok_or(RootObjectError::InvalidLimits)?;
        base.checked_add(extra).ok_or(RootObjectError::InvalidLimits)
    }
}

/// Model-free, non-deserializable derivation of one exact caller schema.
/// Both declarations are independently compiled; only root required changes.
pub struct RootObjectSchema {
    original: JsonValidationSchema,
    partial: JsonValidationSchema,
    partial_json: String,
    original_digest: Sha256Digest,
    partial_digest: Sha256Digest,
}
impl RootObjectSchema {
    /// Additional simultaneous ORIGINAL and PARTIAL declaration/derivation
    /// storage. Independent of reducer result-byte caps: a tiny legal object
    /// can still have a large optional enum schema. The 64-level schema cap
    /// bounds repeated path text; eight lexical/path copies per level plus
    /// 8192 bytes per compiler state cover both sparse declaration/length maps
    /// and construction overlap. Native chunk programs retain their separate
    /// caller-admitted preparation reserve.
    pub fn workspace_bytes(compiler: CompileLimits) -> Result<u64, RootObjectError> {
        if compiler.max_schema_bytes == 0 || compiler.max_states == 0 {
            return Err(RootObjectError::InvalidLimits);
        }
        (compiler.max_schema_bytes as u64).checked_mul(512)
            .and_then(|n| n.checked_add((compiler.max_states as u64).checked_mul(8192)?))
            .and_then(|n| n.checked_add(4096)).ok_or(RootObjectError::InvalidLimits)
    }
    pub fn derive<C: DecodeStepControl>(schema: &str, compiler: CompileLimits,
        limits: RootObjectLimits, control: &mut C) -> Result<Self, RootObjectError> {
        poll(control)?;
        limits.workspace_bytes()?;
        Self::workspace_bytes(compiler)?;
        if schema.len() > compiler.max_schema_bytes { return Err(RootObjectError::Schema); }
        let original = JsonValidationSchema::compile(schema, CompileLimits {
            max_output_bytes: limits.max_json_bytes, ..compiler }).map_err(|_| RootObjectError::Schema)?;
        let SchemaNode::Object { properties, .. } = original.root() else { return Err(RootObjectError::Schema); };
        if properties.len() > limits.max_properties { return Err(RootObjectError::Limit); }
        array_union::schema_shape(original.root(), 0)?;
        // Schema preparation has its OWN compiler-byte quote. Do not spend
        // or require the later chunk/result node allowance to parse a schema.
        // Every lexical node consumes a source byte; inserting required:[]
        // can add at most one value node.
        let schema_nodes = schema.len().checked_add(2).ok_or(RootObjectError::InvalidLimits)?;
        let mut value = array_union::parse(schema, schema_nodes, control)?;
        let JsonValue::Object(ref mut entries) = value else { return Err(RootObjectError::Schema); };
        entries.insert("required".to_owned(), JsonValue::Array(Vec::new()));
        let mut writer = Writer::new(compiler.max_schema_bytes);
        let mut remaining = schema_nodes;
        writer.value(&value, &mut remaining, control)?;
        let partial_json = writer.finish()?;
        let partial = JsonValidationSchema::compile(&partial_json, compiler).map_err(|_| RootObjectError::Schema)?;
        let SchemaNode::Object { properties: actual, required } = partial.root() else { return Err(RootObjectError::Schema); };
        if actual != properties || !required.is_empty() { return Err(RootObjectError::Schema); }
        poll(control)?;
        Ok(Self { original, partial, original_digest: Sha256Digest::of_bytes(schema.as_bytes()),
            partial_digest: Sha256Digest::of_bytes(partial_json.as_bytes()), partial_json })
    }
    pub fn partial_schema(&self) -> &str { &self.partial_json }
    pub fn requires_source(&self) -> bool { self.original.requires_verbatim_source() }
    pub(crate) fn binding(&self, limits: RootObjectLimits) -> Result<Sha256Digest, RootObjectError> {
        let bytes = canonjson::canonical_bytes(&(ROOT_OBJECT_DERIVATION, ROOT_OBJECT_RECONCILIATION,
            self.original_digest, self.partial_digest, limits)).map_err(|_| RootObjectError::Serialization)?;
        Ok(Sha256Digest::of_bytes(&bytes))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RootObjectOrigin {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    /// Addresses that chunk's unchanged original-coordinate field evidence.
    pub source_field_indices: Vec<usize>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RootObjectProperty {
    pub json_pointer: String,
    pub origins: Vec<RootObjectOrigin>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RootObjectVerification { pub fields: usize, pub matches: usize, pub scan_steps: u64 }
/// Sealed successful reconciliation. A conflicting or incomplete object has
/// no successful value and never exposes a silently selected winning field.
#[derive(Serialize)]
pub struct RootObjectResult {
    schema_version: u32,
    policy: &'static str,
    schema_derivation: &'static str,
    json: String,
    properties: Vec<RootObjectProperty>,
    original_source_fields: Vec<SourceFieldEvidence>,
    input_properties: usize,
    duplicate_properties: usize,
    input_json_bytes: usize,
    visited_nodes: usize,
    verification_used: RootObjectVerification,
    untrusted_fields: [&'static str; 3],
    #[serde(skip)] binding: Sha256Digest,
    #[serde(skip)] mapped_digest: [u8; 32],
}
impl RootObjectResult {
    pub fn json(&self) -> &str { &self.json }
    pub fn properties(&self) -> &[RootObjectProperty] { &self.properties }
    pub fn original_source_fields(&self) -> &[SourceFieldEvidence] { &self.original_source_fields }
    pub fn verification_used(&self) -> RootObjectVerification { self.verification_used }
    pub(crate) fn binding(&self) -> Sha256Digest { self.binding }
    pub(crate) fn matches_chunks<'a>(&self, chunks: impl Iterator<Item = ObjectChunk<'a>>) -> bool {
        let mut hash = map_hasher();
        for chunk in chunks { hash_chunk(&mut hash, &chunk); }
        let actual: [u8; 32] = hash.finalize().into();
        actual == self.mapped_digest
    }
}
pub(crate) struct ObjectChunk<'a> {
    pub chunk_id: usize, pub source_span: VerifiedSourceSpan,
    pub json: &'a str, pub source_fields: &'a [SourceFieldEvidence],
}
struct PropertyValue { canonical: String, origins: Vec<RootObjectOrigin> }

pub(crate) fn reconcile<'a, C: DecodeStepControl>(schema: &RootObjectSchema, source: &str,
    binding: Sha256Digest, limits: RootObjectLimits, mut verification: GroundingBudget,
    source_membership: bool, chunks: impl ExactSizeIterator<Item = ObjectChunk<'a>>, control: &mut C)
    -> Result<RootObjectResult, RootObjectError> {
    poll(control)?;
    limits.workspace_bytes()?;
    if chunks.len() == 0 || source.is_empty() { return Err(RootObjectError::Input); }
    let initial = verification;
    let mut index = BTreeMap::<String, PropertyValue>::new();
    let mut remaining_nodes = limits.max_nodes;
    let mut input_bytes = 0_usize;
    let mut input_properties = 0_usize;
    let mut previous = VerifiedSourceSpan { byte_start: 0, byte_end: 0, scalar_start: 0, scalar_end: 0 };
    let mut hash = map_hasher();
    for (expected_id, chunk) in chunks.enumerate() {
        poll(control)?;
        let slice = source.get(chunk.source_span.byte_start..chunk.source_span.byte_end)
            .ok_or(RootObjectError::Input)?;
        if chunk.chunk_id != expected_id || chunk.source_span.byte_start != previous.byte_end
            || chunk.source_span.scalar_start != previous.scalar_end
            || chunk.source_span.byte_end <= chunk.source_span.byte_start
            || chunk.source_span.scalar_end <= chunk.source_span.scalar_start
            || chunk.source_span.scalar_end.checked_sub(chunk.source_span.scalar_start) != Some(slice.chars().count()) {
            return Err(RootObjectError::Input);
        }
        previous = chunk.source_span;
        hash_chunk(&mut hash, &chunk);
        input_bytes = input_bytes.checked_add(chunk.json.len()).filter(|&n| n <= limits.max_input_bytes)
            .ok_or(RootObjectError::Limit)?;
        let parsed = array_union::parse(chunk.json, remaining_nodes, control)?;
        schema.partial.validate_value(&parsed).map_err(|_| RootObjectError::Input)?;
        let JsonValue::Object(values) = parsed else { return Err(RootObjectError::Input); };
        remaining_nodes = remaining_nodes.checked_sub(1).ok_or(RootObjectError::Limit)?;
        input_properties = input_properties.checked_add(values.len()).filter(|&n| n <= limits.max_origins)
            .ok_or(RootObjectError::Limit)?;
        let mut field_indices = BTreeMap::<String, Vec<usize>>::new();
        for (position, field) in chunk.source_fields.iter().enumerate() {
            poll(control)?;
            let key = pointer_property(&field.json_pointer)?;
            if !values.contains_key(&key) { return Err(RootObjectError::Input); }
            let indices = field_indices.entry(key).or_default();
            indices.try_reserve(1).map_err(|_| RootObjectError::Allocation)?;
            indices.push(position);
        }
        for (name, value) in values {
            poll(control)?;
            let mut canonical = Writer::new(limits.max_json_bytes);
            canonical.value(&value, &mut remaining_nodes, control)?;
            let canonical = canonical.finish()?;
            let origin = RootObjectOrigin { chunk_id: chunk.chunk_id, source_span: chunk.source_span,
                source_field_indices: field_indices.remove(&name).unwrap_or_default() };
            if let Some(previous) = index.get_mut(&name) {
                if previous.canonical != canonical { return Err(RootObjectError::Conflict); }
                previous.origins.try_reserve(1).map_err(|_| RootObjectError::Allocation)?;
                previous.origins.push(origin);
            } else {
                if index.len() == limits.max_properties { return Err(RootObjectError::Limit); }
                let mut origins = Vec::new(); origins.try_reserve_exact(1).map_err(|_| RootObjectError::Allocation)?;
                origins.push(origin); index.insert(name, PropertyValue { canonical, origins });
            }
        }
    }
    if previous.byte_end != source.len() || previous.scalar_end != source.chars().count() {
        return Err(RootObjectError::Input);
    }
    let SchemaNode::Object { required, .. } = schema.original.root() else { return Err(RootObjectError::Schema); };
    if required.iter().any(|key| !index.contains_key(key)) { return Err(RootObjectError::MissingRequired); }
    let mut json = Writer::new(limits.max_json_bytes);
    let mut properties = Vec::new(); properties.try_reserve_exact(index.len()).map_err(|_| RootObjectError::Allocation)?;
    json.push(b"{")?;
    for (name, value) in index {
        poll(control)?;
        if !properties.is_empty() { json.push(b",")?; }
        json.string(&name, control)?; json.push(b":")?; json.push(value.canonical.as_bytes())?;
        properties.push(RootObjectProperty { json_pointer: pointer(&name), origins: value.origins });
    }
    json.push(b"}")?;
    let json = json.finish()?;
    let final_value = array_union::parse(&json, remaining_nodes, control)?;
    schema.original.validate_value(&final_value).map_err(|_| RootObjectError::Input)?;
    count_nodes(&final_value, &mut remaining_nodes, control)?;
    let mut fields = Vec::new();
    if source_membership {
        verify_fields(schema.original.root(), &final_value, source, "$", &mut verification, &mut fields, control)?;
    }
    let result = RootObjectResult { schema_version: 1, policy: ROOT_OBJECT_RECONCILIATION,
        schema_derivation: ROOT_OBJECT_DERIVATION, json,
        duplicate_properties: input_properties.checked_sub(properties.len()).ok_or(RootObjectError::Input)?,
        properties, original_source_fields: fields, input_properties, input_json_bytes: input_bytes,
        visited_nodes: limits.max_nodes - remaining_nodes,
        verification_used: RootObjectVerification { fields: initial.max_fields - verification.max_fields,
            matches: initial.max_matches - verification.max_matches,
            scan_steps: initial.max_scan_steps - verification.max_scan_steps },
        untrusted_fields: ["json", "properties", "original_source_fields"], binding,
        mapped_digest: hash.finalize().into() };
    super::quantized::check_size(&result, limits.max_result_bytes as u64).map_err(|_| RootObjectError::Limit)?;
    poll(control)?;
    Ok(result)
}
fn count_nodes<C: DecodeStepControl>(value: &JsonValue, remaining: &mut usize, control: &mut C)
    -> Result<(), RootObjectError> {
    poll(control)?; *remaining = remaining.checked_sub(1).ok_or(RootObjectError::Limit)?;
    match value { JsonValue::Array(values) => for value in values { count_nodes(value, remaining, control)?; },
        JsonValue::Object(values) => for value in values.values() { count_nodes(value, remaining, control)?; }, _ => {} }
    Ok(())
}
fn verify_fields<C: DecodeStepControl>(schema: &SchemaNode, value: &JsonValue, source: &str,
    path: &str, budget: &mut GroundingBudget, fields: &mut Vec<SourceFieldEvidence>, control: &mut C)
    -> Result<(), RootObjectError> {
    poll(control)?;
    match (schema, value) {
        (SchemaNode::String { source: SourceAnnotation::Verbatim, .. }, JsonValue::String(text)) => {
            budget.max_fields = budget.max_fields.checked_sub(1).ok_or(RootObjectError::Limit)?;
            let spans = scan_occurrences(source, text, budget).map_err(|_| RootObjectError::Source)?;
            poll(control)?;
            let occurrence = if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous };
            fields.try_reserve(1).map_err(|_| RootObjectError::Allocation)?;
            fields.push(SourceFieldEvidence { json_pointer: path.to_owned(), occurrence, spans });
        }
        (SchemaNode::Object { properties, .. }, JsonValue::Object(values)) => for (key, value) in values {
            let next = if path == "$" { pointer(key) } else { format!("{path}{}", pointer(key)) };
            verify_fields(properties.get(key).ok_or(RootObjectError::Input)?, value, source, &next, budget, fields, control)?;
        },
        (SchemaNode::Array { items, .. }, JsonValue::Array(values)) => for (index, value) in values.iter().enumerate() {
            verify_fields(items, value, source, &format!("{path}/{index}"), budget, fields, control)?;
        },
        _ => {}, // The independently validated final value already proves shape.
    }
    Ok(())
}
fn pointer(key: &str) -> String { format!("/{}", key.replace('~', "~0").replace('/', "~1")) }
fn pointer_property(path: &str) -> Result<String, RootObjectError> {
    let first = path.strip_prefix('/').ok_or(RootObjectError::Input)?.split('/').next().ok_or(RootObjectError::Input)?;
    let mut output = String::new(); output.try_reserve_exact(first.len()).map_err(|_| RootObjectError::Allocation)?;
    let mut chars = first.chars();
    while let Some(ch) = chars.next() { output.push(if ch == '~' { match chars.next() {
        Some('0') => '~', Some('1') => '/', _ => return Err(RootObjectError::Input),
    } } else { ch }); }
    Ok(output)
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), RootObjectError> {
    control.prefill_checkpoint(0).map_or(Ok(()), |cause| Err(RootObjectError::Cancelled(cause)))
}
fn map_hasher() -> Sha256 { let mut hash = Sha256::new(); hash.update(b"fnlp-root-object-map-v1\0"); hash }
fn hash_chunk(hash: &mut Sha256, chunk: &ObjectChunk<'_>) {
    fn number(hash: &mut Sha256, value: usize) { hash.update((value as u64).to_le_bytes()); }
    fn text(hash: &mut Sha256, value: &str) { number(hash, value.len()); hash.update(value.as_bytes()); }
    fn span(hash: &mut Sha256, value: VerifiedSourceSpan) {
        for n in [value.byte_start, value.byte_end, value.scalar_start, value.scalar_end] { number(hash, n); }
    }
    number(hash, chunk.chunk_id); span(hash, chunk.source_span); text(hash, chunk.json);
    number(hash, chunk.source_fields.len());
    for field in chunk.source_fields {
        text(hash, &field.json_pointer);
        hash.update([match field.occurrence { SourceOccurrence::Anchored => 1, SourceOccurrence::Ambiguous => 2 }]);
        number(hash, field.spans.len()); for &value in &field.spans { span(hash, value); }
    }
}

#[cfg(test)] mod tests;
