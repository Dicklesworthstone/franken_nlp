//! Exact root-array union after independently verified complete source maps.
//! First occurrence fixes record order; every duplicate keeps its origin.
//! No scalar conflict resolution, approximate matching, truncation or model pass.
use std::{collections::BTreeMap, fmt, mem::size_of};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use crate::{
    canonjson, execution_identity::Sha256Digest, grammar::SchemaNode,
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl},
    validation::{self, JsonLimits, JsonValue, grounded_fields::{SourceFieldEvidence, VerifiedSourceSpan}},
};

pub const ROOT_ARRAY_UNION_POLICY: &str = "exact-root-array-union-source-order-v1";
const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_DEPTH: usize = 64;
const MAX_KEY_BYTES: usize = 1024;
// Any canonical spelling of a permitted schema key fits this lexical cap.
// A no-allocation lexical pass enforces it BEFORE the independent tree parser.
const MAX_KEY_JSON_BYTES: usize = 6 * MAX_KEY_BYTES;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RootArrayUnionLimits {
    pub max_records: usize,
    pub max_origins: usize,
    /// Sum of ALL input chunk JSON bytes, including duplicates and punctuation.
    pub max_input_bytes: usize,
    pub max_json_bytes: usize,
    /// All input value nodes PLUS the final independently parsed array.
    pub max_nodes: usize,
    /// Complete union envelope; retained chunk output is additional.
    pub max_result_bytes: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RootArrayUnionRequirements { pub workspace_bytes: u64 }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RootArrayUnionError {
    InvalidLimits, Schema, Input, Limit, Allocation, Serialization,
    Cancelled(DecodeCancellationKind),
}
impl fmt::Display for RootArrayUnionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid root-array union limits",
            Self::Schema => "root-array union requires a bounded root array and supported key/depth limits",
            Self::Input => "root-array union source result or evidence refused",
            Self::Limit => "root-array union complete result exceeds its allowance",
            Self::Allocation => "root-array union allocation refused",
            Self::Serialization => "root-array union serialization refused",
            Self::Cancelled(_) => "root-array union cancelled",
        })
    }
}
impl std::error::Error for RootArrayUnionError {}

impl RootArrayUnionLimits {
    pub fn requirements(self) -> Result<RootArrayUnionRequirements, RootArrayUnionError> {
        if !(1..=1_000_000).contains(&self.max_records)
            || !(1..=1_000_000).contains(&self.max_origins)
            || !(1..=1_000_000).contains(&self.max_nodes)
            || self.max_records > self.max_origins
            || [self.max_input_bytes, self.max_json_bytes, self.max_result_bytes].iter()
                .any(|n| !(2..=MAX_BYTES).contains(n))
            || self.max_json_bytes > self.max_result_bytes {
            return Err(RootArrayUnionError::InvalidLimits);
        }
        let word = size_of::<usize>() as u64;
        let value = size_of::<JsonValue>() as u64;
        let string = size_of::<String>() as u64;
        // Pinned std B-tree nodes have <=11 entries and <=12 child pointers.
        // One sparse map/set node per value, doubled for split overlap, plus
        // four Vec slots per value. Tree parsing retains no location table.
        let node = add(&[mul(16, word)?, mul(11, add(&[string, value])?)?])?;
        let trees = mul(self.max_nodes as u64, add(&[mul(2, node)?, mul(4, value)?])?)?;
        let path = add(&[mul(MAX_DEPTH as u64, (2 * MAX_KEY_JSON_BYTES + 42) as u64)?, 64])?;
        let paths = mul(path, (MAX_DEPTH + 4) as u64)?;
        // Parsed strings/keys and scanner buffers; one parsed tree is live at
        // a time. Canonical index keys, current key and complete JSON coexist.
        let strings = mul(4, self.max_input_bytes.max(self.max_json_bytes) as u64)?;
        let canonical = mul(4, self.max_json_bytes as u64)?;
        let index_node = add(&[mul(16, word)?, mul(11, add(&[string, word])?)?])?;
        let records = mul(self.max_records as u64,
            add(&[mul(2, index_node)?, mul(4, size_of::<RootArrayUnionRecord>() as u64)?])?)?;
        let origins = mul(self.max_origins as u64,
            add(&[mul(4, size_of::<RootArrayUnionOrigin>() as u64)?, mul(4, size_of::<Vec<usize>>() as u64)?])?)?;
        let field_indices = mul(self.max_nodes as u64, mul(4, word)?)?;
        Ok(RootArrayUnionRequirements { workspace_bytes:
            add(&[trees, paths, strings, canonical, records, origins, field_indices, 4096])? })
    }

    /// Pure schema restriction check; no source, model or executable authority.
    pub fn validate_schema(self, schema: &SchemaNode) -> Result<(), RootArrayUnionError> {
        self.requirements()?;
        if !matches!(schema, SchemaNode::Array { .. }) { return Err(RootArrayUnionError::Schema); }
        schema_shape(schema, 0)
    }

    pub(crate) fn binding(self, schema: Sha256Digest) -> Result<Sha256Digest, RootArrayUnionError> {
        self.requirements()?;
        #[derive(Serialize)]
        struct Binding { policy: &'static str, limits: RootArrayUnionLimits, schema: Sha256Digest }
        let bytes = canonjson::canonical_bytes(&Binding { policy: ROOT_ARRAY_UNION_POLICY, limits: self, schema })
            .map_err(|_| RootArrayUnionError::Serialization)?;
        Ok(Sha256Digest::of_bytes(&bytes))
    }
}
pub(super) fn schema_shape(schema: &SchemaNode, depth: usize) -> Result<(), RootArrayUnionError> {
    if depth > MAX_DEPTH { return Err(RootArrayUnionError::Schema); }
    match schema {
        SchemaNode::Array { items, .. } => schema_shape(items, depth + 1)?,
        SchemaNode::Object { properties, .. } => for (key, child) in properties {
            if key.len() > MAX_KEY_BYTES { return Err(RootArrayUnionError::Schema); }
            schema_shape(child, depth + 1)?;
        },
        _ => {},
    }
    Ok(())
}

/// One occurrence of an exact record in an independently validated source map.
/// `source_field_indices` addresses that chunk's retained `original_fields`.
/// Those entries retain ALL original-coordinate spans and ambiguity. To locate
/// the corresponding union field, replace the pointer's first array index with
/// the containing RootArrayUnionRecord.array_index; the escaped suffix is exact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RootArrayUnionOrigin {
    pub chunk_id: usize,
    pub array_index: usize,
    pub source_span: VerifiedSourceSpan,
    pub source_field_indices: Vec<usize>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RootArrayUnionRecord {
    pub array_index: usize,
    pub origins: Vec<RootArrayUnionOrigin>,
}
/// Sealed successful reducer output, never deserialized as execution evidence.
#[derive(Serialize)]
pub struct RootArrayUnionResult {
    schema_version: u32,
    policy: &'static str,
    /// Exact canonical JSON STRING, retaining the full 38-digit decimal domain.
    json: String,
    records: Vec<RootArrayUnionRecord>,
    input_records: usize,
    duplicate_records: usize,
    input_json_bytes: usize,
    visited_nodes: usize,
    untrusted_fields: [&'static str; 2],
    #[serde(skip)]
    binding: Sha256Digest,
    #[serde(skip)]
    mapped_digest: [u8; 32],
}
impl RootArrayUnionResult {
    pub fn json(&self) -> &str { &self.json }
    pub fn records(&self) -> &[RootArrayUnionRecord] { &self.records }
    pub fn input_records(&self) -> usize { self.input_records }
    pub fn duplicate_records(&self) -> usize { self.duplicate_records }
    pub(crate) fn binding(&self) -> Sha256Digest { self.binding }
    pub(crate) fn matches_chunks<'a>(&self, chunks: impl Iterator<Item = ArrayUnionChunk<'a>>) -> bool {
        let mut digest = map_hasher();
        for chunk in chunks { hash_chunk(&mut digest, &chunk); }
        let actual: [u8; 32] = digest.finalize().into();
        self.mapped_digest == actual
    }
}
pub(crate) struct ArrayUnionChunk<'a> {
    pub chunk_id: usize,
    pub source_span: VerifiedSourceSpan,
    pub json: &'a str,
    pub source_fields: &'a [SourceFieldEvidence],
}

/// Only sealed native-map composition calls this function. Evidence pointers
/// refer to the retained validated map; this function mints no source spans.
pub(crate) fn reduce<'a, C: DecodeStepControl>(schema: &SchemaNode, binding: Sha256Digest,
    limits: RootArrayUnionLimits, chunks: impl ExactSizeIterator<Item = ArrayUnionChunk<'a>>,
    validate_declaration: impl FnOnce(&JsonValue) -> Result<(), RootArrayUnionError>,
    control: &mut C) -> Result<RootArrayUnionResult, RootArrayUnionError> {
    poll(control)?;
    limits.validate_schema(schema)?;
    let SchemaNode::Array { max_items, .. } = schema else { return Err(RootArrayUnionError::Schema); };
    let record_cap = limits.max_records.min(*max_items);
    let mut index = BTreeMap::<String, usize>::new();
    let mut records: Vec<RootArrayUnionRecord> = Vec::new();
    let mut json = Writer::new(limits.max_json_bytes);
    json.push(b"[")?;
    let mut remaining_nodes = limits.max_nodes;
    let mut input_bytes = 0_usize;
    let mut input_records = 0_usize;
    let mut final_nodes = 1_usize;
    let mut previous_span: Option<VerifiedSourceSpan> = None;
    let mut mapped_digest = map_hasher();
    if chunks.len() == 0 { return Err(RootArrayUnionError::Input); }
    for (expected_id, chunk) in chunks.enumerate() {
        poll(control)?;
        if chunk.chunk_id != expected_id || chunk.source_span.byte_start > chunk.source_span.byte_end
            || chunk.source_span.scalar_start > chunk.source_span.scalar_end
            || previous_span.is_some_and(|previous| previous.byte_end != chunk.source_span.byte_start
                || previous.scalar_end != chunk.source_span.scalar_start) {
            return Err(RootArrayUnionError::Input);
        }
        previous_span = Some(chunk.source_span);
        input_bytes = input_bytes.checked_add(chunk.json.len()).filter(|&n| n <= limits.max_input_bytes)
            .ok_or(RootArrayUnionError::Limit)?;
        hash_chunk(&mut mapped_digest, &chunk);
        let parsed = parse(chunk.json, remaining_nodes, control)?;
        validation::validate_value(schema, &parsed).map_err(|_| RootArrayUnionError::Input)?;
        let JsonValue::Array(values) = parsed else { return Err(RootArrayUnionError::Input); };
        remaining_nodes = remaining_nodes.checked_sub(1).ok_or(RootArrayUnionError::Limit)?;
        if chunk.source_fields.len() > remaining_nodes { return Err(RootArrayUnionError::Limit); }
        input_records = input_records.checked_add(values.len()).filter(|&n| n <= limits.max_origins)
            .ok_or(RootArrayUnionError::Limit)?;
        let mut fields: Vec<Vec<usize>> = vector(values.len())?;
        fields.resize_with(values.len(), Vec::new);
        // One linear partition of existing evidence. No per-record source scan
        // or repeated full-field walk and no loss of duplicate-origin spans.
        for (field_index, field) in chunk.source_fields.iter().enumerate() {
            poll(control)?;
            let slot = pointer_index(&field.json_pointer)?;
            let target = fields.get_mut(slot).ok_or(RootArrayUnionError::Input)?;
            target.try_reserve(1).map_err(|_| RootArrayUnionError::Allocation)?;
            target.push(field_index);
        }
        for (array_index, (value, source_field_indices)) in values.iter().zip(fields).enumerate() {
            poll(control)?;
            let before = remaining_nodes;
            let mut canonical = Writer::new(limits.max_json_bytes);
            canonical.value(value, &mut remaining_nodes, control)?;
            let nodes = before - remaining_nodes;
            let key = canonical.finish()?;
            let origin = RootArrayUnionOrigin { chunk_id: chunk.chunk_id, array_index,
                source_span: chunk.source_span, source_field_indices };
            if let Some(&slot) = index.get(&key) {
                records[slot].origins.try_reserve(1).map_err(|_| RootArrayUnionError::Allocation)?;
                records[slot].origins.push(origin);
            } else {
                if records.len() == record_cap { return Err(RootArrayUnionError::Limit); }
                if !records.is_empty() { json.push(b",")?; }
                json.push(key.as_bytes())?;
                final_nodes = final_nodes.checked_add(nodes).ok_or(RootArrayUnionError::Limit)?;
                let array_index = records.len();
                records.try_reserve(1).map_err(|_| RootArrayUnionError::Allocation)?;
                let mut origins = vector(1)?; origins.push(origin);
                records.push(RootArrayUnionRecord { array_index, origins });
                index.insert(key, array_index);
            }
        }
    }
    json.push(b"]")?;
    let json = json.finish()?;
    // Independently parse the FINAL bytes and apply the ORIGINAL schema. In
    // particular, successful per-chunk maxItems checks cannot authorize a
    // larger aggregate. Neither model validity nor canonicalization is reused.
    let final_value = parse(&json, remaining_nodes, control)?;
    validation::validate_value(schema, &final_value).map_err(|_| RootArrayUnionError::Input)?;
    validate_declaration(&final_value)?;
    remaining_nodes = remaining_nodes.checked_sub(final_nodes).ok_or(RootArrayUnionError::Limit)?;
    drop(final_value);
    drop(index);
    poll(control)?;
    let result = RootArrayUnionResult { schema_version: 1, policy: ROOT_ARRAY_UNION_POLICY,
        duplicate_records: input_records.checked_sub(records.len()).ok_or(RootArrayUnionError::Input)?,
        json, records, input_records, input_json_bytes: input_bytes,
        visited_nodes: limits.max_nodes - remaining_nodes, untrusted_fields: ["json", "records"], binding,
        mapped_digest: mapped_digest.finalize().into() };
    super::quantized::check_size(&result, limits.max_result_bytes as u64)
        .map_err(|_| RootArrayUnionError::Limit)?;
    poll(control)?;
    Ok(result)
}

pub(super) fn parse<C: DecodeStepControl>(json: &str, nodes: usize, control: &mut C) -> Result<JsonValue, RootArrayUnionError> {
    // Strings followed by ':' are object keys in valid JSON. The independent
    // parser still checks the complete syntax and duplicate-key semantics.
    // This scan has no allocation and caps path components before tree parsing.
    let bytes = json.as_bytes(); let mut at = 0_usize;
    while at < bytes.len() {
        if at % 4096 == 0 { poll(control)?; }
        if bytes[at] != b'"' { at += 1; continue; }
        at += 1; let start = at;
        while at < bytes.len() && bytes[at] != b'"' {
            if at % 4096 == 0 { poll(control)?; }
            if bytes[at] == b'\\' { at = at.checked_add(1).ok_or(RootArrayUnionError::Limit)?; }
            at = at.checked_add(1).ok_or(RootArrayUnionError::Limit)?;
        }
        if at >= bytes.len() { return Err(RootArrayUnionError::Input); }
        let length = at - start; at += 1;
        while at < bytes.len() && bytes[at].is_ascii_whitespace() {
            if at % 4096 == 0 { poll(control)?; } at += 1;
        }
        if bytes.get(at) == Some(&b':') && length > MAX_KEY_JSON_BYTES { return Err(RootArrayUnionError::Limit); }
    }
    poll(control)?;
    let value = validation::json::parse_json_with_node_limit(json, JsonLimits {
        max_input_bytes: MAX_BYTES, max_depth: MAX_DEPTH, max_container_entries: nodes,
        max_string_lexeme_bytes: MAX_BYTES, max_number_lexeme_bytes: 512,
    }, nodes).map_err(|_| RootArrayUnionError::Input)?;
    poll(control)?;
    Ok(value)
}

fn pointer_index(pointer: &str) -> Result<usize, RootArrayUnionError> {
    let first = pointer.strip_prefix('/').ok_or(RootArrayUnionError::Input)?.split('/').next()
        .ok_or(RootArrayUnionError::Input)?;
    if first.is_empty() || first.len() > 1 && first.starts_with('0') || !first.bytes().all(|b| b.is_ascii_digit()) {
        return Err(RootArrayUnionError::Input);
    }
    first.parse().map_err(|_| RootArrayUnionError::Input)
}
// Private child-set binding, never serialized or used as public telemetry.
// Length framing keeps exact lexical JSON and all source evidence distinct;
// hashing needs no population-sized canonical serialization buffer.
fn map_hasher() -> Sha256 { let mut hash = Sha256::new(); hash.update(b"fnlp-root-array-map-v1\0"); hash }
fn hash_chunk(hash: &mut Sha256, chunk: &ArrayUnionChunk<'_>) {
    fn number(hash: &mut Sha256, n: usize) { hash.update((n as u64).to_le_bytes()); }
    fn text(hash: &mut Sha256, value: &str) { number(hash, value.len()); hash.update(value.as_bytes()); }
    fn span(hash: &mut Sha256, value: VerifiedSourceSpan) {
        for n in [value.byte_start, value.byte_end, value.scalar_start, value.scalar_end] { number(hash, n); }
    }
    number(hash, chunk.chunk_id); span(hash, chunk.source_span); text(hash, chunk.json);
    number(hash, chunk.source_fields.len());
    for field in chunk.source_fields {
        text(hash, &field.json_pointer);
        hash.update([match field.occurrence {
            validation::grounded_fields::SourceOccurrence::Anchored => 1,
            validation::grounded_fields::SourceOccurrence::Ambiguous => 2,
        }]);
        number(hash, field.spans.len());
        for &value in &field.spans { span(hash, value); }
    }
}
pub(super) struct Writer { bytes: Vec<u8>, cap: usize }
impl Writer {
    pub(super) fn new(cap: usize) -> Self { Self { bytes: Vec::new(), cap } }
    pub(super) fn push(&mut self, bytes: &[u8]) -> Result<(), RootArrayUnionError> {
        let needed = self.bytes.len().checked_add(bytes.len()).filter(|&n| n <= self.cap)
            .ok_or(RootArrayUnionError::Limit)?;
        if needed > self.bytes.capacity() {
            let mut capacity = self.bytes.capacity().max(1);
            while capacity < needed { capacity = capacity.checked_mul(2).unwrap_or(self.cap).min(self.cap); }
            self.bytes.try_reserve_exact(capacity - self.bytes.len()).map_err(|_| RootArrayUnionError::Allocation)?;
        }
        self.bytes.extend_from_slice(bytes); Ok(())
    }
    pub(super) fn string<C: DecodeStepControl>(&mut self, text: &str, control: &mut C) -> Result<(), RootArrayUnionError> {
        self.push(b"\"")?;
        let mut start = 0;
        for (i, byte) in text.bytes().enumerate() {
            if i % 4096 == 0 { poll(control)?; }
            let escaped: Option<&[u8]> = match byte { b'"' => Some(b"\\\""), b'\\' => Some(b"\\\\"),
                b'\x08' => Some(b"\\b"), b'\x0c' => Some(b"\\f"), b'\n' => Some(b"\\n"),
                b'\r' => Some(b"\\r"), b'\t' => Some(b"\\t"), _ => None };
            if escaped.is_some() || byte < 32 {
                self.push(&text.as_bytes()[start..i])?;
                if let Some(escaped) = escaped { self.push(escaped)?; }
                else { const HEX: &[u8] = b"0123456789abcdef";
                    self.push(&[b'\\', b'u', b'0', b'0', HEX[(byte >> 4) as usize], HEX[(byte & 15) as usize]])?; }
                start = i + 1;
            }
        }
        self.push(&text.as_bytes()[start..])?; self.push(b"\"")
    }
    pub(super) fn value<C: DecodeStepControl>(&mut self, value: &JsonValue, remaining: &mut usize, control: &mut C)
        -> Result<(), RootArrayUnionError> {
        poll(control)?;
        *remaining = remaining.checked_sub(1).ok_or(RootArrayUnionError::Limit)?;
        match value {
            JsonValue::Null => self.push(b"null")?,
            JsonValue::Boolean(true) => self.push(b"true")?,
            JsonValue::Boolean(false) => self.push(b"false")?,
            JsonValue::String(text) => self.string(text, control)?,
            JsonValue::Number(value) => self.push(value.canonical_spelling().as_bytes())?,
            JsonValue::Array(values) => { self.push(b"[")?;
                for (i, value) in values.iter().enumerate() { if i > 0 { self.push(b",")?; } self.value(value, remaining, control)?; }
                self.push(b"]")?; }
            JsonValue::Object(values) => { self.push(b"{")?;
                for (i, (key, value)) in values.iter().enumerate() { if i > 0 { self.push(b",")?; }
                    self.string(key, control)?; self.push(b":")?; self.value(value, remaining, control)?; }
                self.push(b"}")?; }
        }
        Ok(())
    }
    pub(super) fn finish(self) -> Result<String, RootArrayUnionError> {
        String::from_utf8(self.bytes).map_err(|_| RootArrayUnionError::Serialization)
    }
}
fn vector<T>(capacity: usize) -> Result<Vec<T>, RootArrayUnionError> {
    let mut out = Vec::new(); out.try_reserve_exact(capacity).map_err(|_| RootArrayUnionError::Allocation)?; Ok(out)
}
fn poll(control: &mut impl DecodeStepControl) -> Result<(), RootArrayUnionError> {
    control.prefill_checkpoint(0).map_or(Ok(()), |cause| Err(RootArrayUnionError::Cancelled(cause)))
}
fn add(values: &[u64]) -> Result<u64, RootArrayUnionError> {
    values.iter().try_fold(0_u64, |sum, &n| sum.checked_add(n).ok_or(RootArrayUnionError::InvalidLimits))
}
fn mul(a: u64, b: u64) -> Result<u64, RootArrayUnionError> { a.checked_mul(b).ok_or(RootArrayUnionError::InvalidLimits) }

#[cfg(test)] mod tests;
