//! Executable, bounded canonical JSON byte language for constrained decoding.
//!
//! The descriptive compiler graph is not an execution machine. This module
//! supplies its missing byte transitions. A clone owns only a bounded parser
//! stack; schema/literal data are borrowed, and generated documents are never
//! copied into vocabulary-trie states. Every accepted prefix has a completion
//! within the output-byte bound. Bound source fields additionally advance a
//! request-owned substring cursor; unbound source annotations still refuse.

use crate::validation::{JsonValue as ValidationValue, JsonLimits, parse_json_with_limits, validate_value};

use super::{
    compiler::{CompileLimits, CompiledSchema, SchemaNode, SourceAnnotation, compile_json_schema},
    mask::ByteState,
    source_index::{SourceCursor, SourceLanguage},
    schema::{ScalarValue, SchemaError, escape_json_string},
};

mod source;
pub use source::{SOURCE_JSON_RUNTIME_VERSION, SourceCompileError, SourceRuntimeLimits};
pub(crate) use source::SourceValidationError;

/// Versioned canonical byte-language and code-point length semantics.
pub const JSON_RUNTIME_VERSION: &str = "canonical-json-runtime-v1";
const MAX_DEPTH: usize = 64;
const NUMBER_BYTES: usize = 48;

#[derive(Clone, Debug)]
struct Property {
    key: Vec<u8>,
    child: usize,
    required: bool,
    // Minimum encoded bytes for required fields starting here, including
    // one leading comma per required field and excluding the closing brace.
    required_suffix_bytes: usize,
}

#[derive(Clone, Debug)]
enum Kind {
    Object(Vec<Property>),
    Array { child: usize, maximum: usize },
    String { max_bytes: usize, max_chars: usize, verbatim: bool },
    Number { integer: bool },
    Literals(Vec<Vec<u8>>),
}

#[derive(Clone, Debug)]
struct Node {
    kind: Kind,
    minimum: usize,
}

/// Compile once, then borrow for any number of independent request states.
#[derive(Clone, Debug)]
pub struct JsonProgram {
    schema: CompiledSchema,
    nodes: Vec<Node>,
    root: usize,
    max_output_bytes: usize,
    source: Option<source::SourceBinding>,
}

/// Model-free declaration checks for private, already parsed gold values.
/// This owns no executable states, source binding or acceptance authority.
/// Actual source-bound enum feasibility remains the native planner's check.
pub(crate) struct JsonValidationSchema {
    root: SchemaNode,
    requires_source: bool,
}
impl JsonValidationSchema {
    pub(crate) fn compile(source: &str, limits: CompileLimits) -> Result<Self, SchemaError> {
        let schema = compile_json_schema(source, limits)?;
        let mut nodes = Vec::new();
        // The declaration may contain verbatim enums without an invented
        // document. Only this non-executable validator permits that state.
        let root = build(schema.root(), &mut nodes, limits.max_states, 0, None, true)?
            .ok_or_else(|| invalid("empty schema language"))?;
        if nodes[root].minimum > limits.max_output_bytes { return Err(resource("no JSON value fits output byte limit")); }
        Ok(Self { root: schema.root().clone(), requires_source: schema.requires_verbatim_source() })
    }
    pub(crate) fn root(&self) -> &SchemaNode { &self.root }
    pub(crate) fn requires_verbatim_source(&self) -> bool { self.requires_source }
    pub(crate) fn validate_value(&self, value: &ValidationValue) -> Result<(), SchemaError> {
        validate_value(&self.root, value).map_err(|_| invalid("independent JSON validation failed"))
    }
}

impl JsonProgram {
    /// Compile the existing supported schema subset with standard code-point
    /// `maxLength`, separate from the engine's UTF-8 byte cap.
    pub fn compile(source: &str, limits: CompileLimits) -> Result<Self, SchemaError> {
        Self::compile_bound(source, limits, None)
    }

    fn compile_bound(source: &str, limits: CompileLimits, binding: Option<source::SourceBinding>) -> Result<Self, SchemaError> {
        let schema = compile_json_schema(source, limits)?;
        if schema.requires_verbatim_source() && binding.is_none() {
            return Err(SchemaError::UnsupportedKeyword {
                pointer: "$".to_owned(), keyword: "runtime source-product gate".to_owned(),
            });
        }
        let mut nodes = Vec::new();
        let root = build(schema.root(), &mut nodes, limits.max_states, 0,
            binding.as_ref().map(|b| &b.language), false)?
            .ok_or_else(|| invalid("no schema value satisfies the source constraints"))?;
        if nodes[root].minimum > limits.max_output_bytes {
            return Err(resource("no JSON value fits output byte limit"));
        }
        Ok(Self {
            schema, nodes, root, max_output_bytes: limits.max_output_bytes, source: binding,
        })
    }

    #[must_use]
    pub fn initial_state(&self) -> JsonState<'_> {
        JsonState { program: self, stack: vec![Frame::Value(self.root)], used: 0, failed: false }
    }

    #[must_use]
    pub const fn max_output_bytes(&self) -> usize { self.max_output_bytes }

    #[must_use]
    pub fn node_count(&self) -> usize { self.nodes.len() }

    /// Private addressed-sampling authority over the ACTUAL compiled program,
    /// including Unicode rules, source bytes and independent verification
    /// limits. This never accepts a caller's declared schema digest in place
    /// of executable authority, allocates, or exposes a content fingerprint.
    pub(crate) fn sampling_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        fn number(hash: &mut Sha256, value: usize) { hash.update((value as u64).to_be_bytes()); }
        fn bytes(hash: &mut Sha256, value: &[u8]) { number(hash, value.len()); hash.update(value); }
        // Preserve the v1 sampling digest layout using the schema's length rules.
        fn lengths(hash: &mut Sha256, schema: &SchemaNode) {
            let max_chars = match schema {
                SchemaNode::String { max_chars, .. } => *max_chars,
                _ => None,
            };
            hash.update([u8::from(max_chars.is_some())]);
            number(hash, max_chars.unwrap_or(0));
            if let SchemaNode::Object { properties, .. } = schema {
                number(hash, properties.len());
                for (key, child) in properties { bytes(hash, key.as_bytes()); lengths(hash, child); }
            } else { number(hash, 0); }
            if let SchemaNode::Array { items, .. } = schema {
                hash.update([1]); lengths(hash, items);
            } else { hash.update([0]); }
        }
        // Node literals below bind exact scalar values without allocating
        // decimal/string renderings. Retain declaration distinctions and
        // source annotations that affect independent final validation too.
        fn declaration(hash: &mut Sha256, schema: &SchemaNode) {
            match schema {
                SchemaNode::Object { properties, required } => {
                    hash.update([0]); number(hash, properties.len()); number(hash, required.len());
                    for (key, child) in properties {
                        bytes(hash, key.as_bytes()); hash.update([u8::from(required.contains(key))]);
                        declaration(hash, child);
                    }
                }
                SchemaNode::Array { items, max_items } => {
                    hash.update([1]); number(hash, *max_items); declaration(hash, items);
                }
                SchemaNode::String { max_bytes, allowed, source, .. } => {
                    hash.update([2, u8::from(allowed.is_some()), u8::from(*source == SourceAnnotation::Verbatim)]);
                    number(hash, *max_bytes);
                }
                SchemaNode::Number { integer, allowed } => hash.update([3, u8::from(*integer), u8::from(allowed.is_some())]),
                SchemaNode::Boolean { allowed } => hash.update([4, u8::from(allowed.is_some())]),
                SchemaNode::Null { allowed } => hash.update([5, u8::from(allowed.is_some())]),
            }
        }
        let mut hash = Sha256::new();
        bytes(&mut hash, b"fnlp-compiled-json-sampling-authority-v1");
        bytes(&mut hash, JSON_RUNTIME_VERSION.as_bytes());
        bytes(&mut hash, self.version().as_bytes());
        number(&mut hash, self.root); number(&mut hash, self.max_output_bytes);
        declaration(&mut hash, self.schema.root()); lengths(&mut hash, self.schema.root());
        number(&mut hash, self.nodes.len());
        for node in &self.nodes {
            number(&mut hash, node.minimum);
            match &node.kind {
                Kind::Object(properties) => {
                    hash.update([0]); number(&mut hash, properties.len());
                    for property in properties {
                        bytes(&mut hash, &property.key); number(&mut hash, property.child);
                        hash.update([u8::from(property.required)]);
                    }
                }
                Kind::Array { child, maximum } => {
                    hash.update([1]); number(&mut hash, *child); number(&mut hash, *maximum);
                }
                Kind::String { max_bytes, max_chars, verbatim } => {
                    hash.update([2, u8::from(*verbatim)]); number(&mut hash, *max_bytes); number(&mut hash, *max_chars);
                }
                Kind::Number { integer } => hash.update([3, u8::from(*integer)]),
                Kind::Literals(literals) => {
                    hash.update([4]); number(&mut hash, literals.len());
                    for literal in literals { bytes(&mut hash, literal); }
                }
            }
        }
        hash.update([u8::from(self.source.is_some())]);
        if let Some(source) = &self.source {
            bytes(&mut hash, super::source_index::SOURCE_LANGUAGE_VERSION.as_bytes());
            bytes(&mut hash, source.language.source().as_bytes());
            let limits = source.limits;
            number(&mut hash, limits.index.max_source_bytes); number(&mut hash, limits.index.max_index_bytes);
            hash.update(limits.index.max_build_steps.to_be_bytes()); number(&mut hash, limits.index.max_matches);
            number(&mut hash, limits.verification.max_fields); number(&mut hash, limits.verification.max_matches);
            hash.update(limits.verification.max_scan_steps.to_be_bytes());
        }
        hash.finalize().into()
    }

    /// Produce a shortest canonical witness of this executable grammar. In
    /// source-bound programs, literal choices already reflect the actual
    /// document. Replay the witness and run the independent verifier before
    /// making any bytes available to the caller.
    pub fn sample_json(&self) -> Result<String, SchemaError> {
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(self.nodes[self.root].minimum)
            .map_err(|_| resource("sample allocation refused"))?;
        self.write_minimum(self.root, &mut bytes)?;
        let mut state = self.initial_state();
        if !state.consume_bytes(&bytes) || !state.is_accepting() {
            return Err(invalid("sample failed executable grammar validation"));
        }
        let text = String::from_utf8(bytes).map_err(|_| invalid("sample is not UTF-8"))?;
        self.validate_json(&text)?;
        Ok(text)
    }

    fn write_minimum(&self, node: usize, bytes: &mut Vec<u8>) -> Result<(), SchemaError> {
        match &self.nodes[node].kind {
            Kind::Object(properties) => {
                bytes.push(b'{');
                for (i, field) in properties.iter().filter(|field| field.required).enumerate() {
                    if i != 0 { bytes.push(b','); }
                    bytes.extend_from_slice(&field.key);
                    self.write_minimum(field.child, bytes)?;
                }
                bytes.push(b'}');
            }
            Kind::Array { .. } => bytes.extend_from_slice(b"[]"),
            Kind::String { .. } => bytes.extend_from_slice(b"\"\""),
            Kind::Number { .. } => bytes.push(b'0'),
            Kind::Literals(values) => {
                // Stored in byte order; ties are deterministic.
                let minimum = values.iter().min_by_key(|value| value.len())
                    .ok_or_else(|| invalid("empty literal language"))?;
                bytes.extend_from_slice(minimum);
            }
        }
        Ok(())
    }

    /// Independent whole-value verification, never acceptance by parser state.
    /// The existing validator receives schema data, not transitions or masks.
    pub fn validate_json(&self, text: &str) -> Result<(), SchemaError> {
        match self.validate_json_with_checkpoints(text, || Ok::<(), std::convert::Infallible>(())) {
            Ok(()) => Ok(()),
            Err(SourceValidationError::Schema(error)) => Err(error),
            Err(SourceValidationError::Cancelled(never)) => match never {},
        }
    }

    /// Complete declarative validation for a composed value whose original
    /// chunk evidence is retained separately. No source-membership authority
    /// is minted here and the originating chunk's output byte cap is not a
    /// whole-document cap. Includes runtime-owned Unicode maxLength rules.
    pub(crate) fn validate_declarative_value(&self, value: &ValidationValue) -> Result<(), SchemaError> {
        validate_value(self.schema.root(), value).map_err(|_| invalid("independent JSON validation failed"))
    }

    fn validated_value(&self, text: &str) -> Result<ValidationValue, SchemaError> {
        if text.len() > self.max_output_bytes { return Err(resource("output byte limit exceeded")); }
        let value = parse_json_with_limits(text, JsonLimits {
            max_input_bytes: self.max_output_bytes,
            max_string_lexeme_bytes: self.max_output_bytes,
            max_container_entries: self.max_output_bytes,
            ..JsonLimits::default()
        }).map_err(|_| invalid("independent JSON parsing failed"))?;
        validate_value(self.schema.root(), &value)
            .map_err(|_| invalid("independent JSON validation failed"))?;
        Ok(value)
    }
}

fn resource(reason: &str) -> SchemaError {
    SchemaError::Resource { pointer: "$".to_owned(), reason: reason.to_owned() }
}
fn invalid(reason: &str) -> SchemaError {
    SchemaError::InvalidSchema { pointer: "$".to_owned(), reason: reason.to_owned() }
}
// None denotes a valid declaration with no value for this particular source.
// A caller may omit such a property or emit an empty array. Resource and
// malformed-declaration errors remain errors throughout the tree.
fn build(schema: &SchemaNode, nodes: &mut Vec<Node>,
    cap: usize, depth: usize, source: Option<&SourceLanguage>, declaration_only: bool) -> Result<Option<usize>, SchemaError> {
    if depth > MAX_DEPTH || nodes.len() >= cap { return Err(resource("runtime schema state limit exceeded")); }
    let literals = |values: &[ScalarValue]| {
        let mut bytes: Vec<_> = values.iter().map(|v| v.canonical_json().into_bytes()).collect();
        bytes.sort();
        Kind::Literals(bytes)
    };
    let kind = match schema {
        SchemaNode::Object { properties, required } => {
            let mut fields = Vec::new();
            let mut possible = true;
            for (name, child) in properties {
                let required = required.contains(name);
                match build(child, nodes, cap, depth + 1, source, declaration_only)? {
                    Some(child) => fields.push(Property {
                        key: format!("{}:", escape_json_string(name)).into_bytes(),
                        child,
                        required,
                        required_suffix_bytes: 0,
                    }),
                    None if required => possible = false,
                    None => {}
                }
            }
            if !possible { return Ok(None); }
            // Compute this after source binding has eliminated impossible
            // optional fields. All request states share the compiled suffixes.
            let mut required_bytes = 0_usize;
            for field in fields.iter_mut().rev() {
                if field.required {
                    required_bytes = required_bytes.checked_add(1)
                        .and_then(|n| n.checked_add(field.key.len()))
                        .and_then(|n| n.checked_add(nodes[field.child].minimum))
                        .ok_or_else(|| resource("minimum output overflow"))?;
                }
                field.required_suffix_bytes = required_bytes;
            }
            Kind::Object(fields)
        }
        SchemaNode::Array { items, max_items } => {
            match build(items, nodes, cap, depth + 1, source, declaration_only)? {
                Some(child) => Kind::Array { child, maximum: *max_items },
                None => Kind::Literals(vec![b"[]".to_vec()]),
            }
        }
        SchemaNode::String { max_bytes, max_chars, allowed, source: annotation } => {
            let verbatim = *annotation == SourceAnnotation::Verbatim;
            let max_chars = max_chars.unwrap_or(usize::MAX);
            if let Some(values) = allowed {
                if verbatim && !declaration_only {
                    let index = source.ok_or_else(|| invalid("source annotation has no binding"))?;
                    let filtered: Vec<_> = values.iter().filter(|v| {
                        let ScalarValue::String(text) = v else { return false; };
                        let mut cursor = index.cursor();
                        text.bytes().all(|b| cursor.push_byte(b)) && cursor.is_accepting()
                    }).cloned().collect();
                    if filtered.is_empty() { return Ok(None); }
                    literals(&filtered)
                } else { literals(values) }
            } else { Kind::String { max_bytes: *max_bytes, max_chars, verbatim } }
        }
        SchemaNode::Number { integer, allowed } => match allowed {
            Some(values) => literals(values), None => Kind::Number { integer: *integer },
        },
        SchemaNode::Boolean { allowed } => match allowed {
            Some(values) => literals(values), None => Kind::Literals(vec![b"false".to_vec(), b"true".to_vec()]),
        },
        SchemaNode::Null { allowed } => match allowed {
            Some(values) => literals(values), None => Kind::Literals(vec![b"null".to_vec()]),
        },
    };
    let minimum = match &kind {
        Kind::Object(fields) => {
            let required = fields.first().map_or(0, |field| field.required_suffix_bytes);
            // The opening brace takes the place of the first comma.
            required.max(1).checked_add(1).ok_or_else(|| resource("minimum output overflow"))?
        }
        Kind::Array { .. } | Kind::String { .. } => 2,
        Kind::Number { .. } => 1,
        Kind::Literals(values) => values.iter().map(Vec::len).min().ok_or_else(|| invalid("empty scalar language"))?,
    };
    if nodes.len() >= cap { return Err(resource("runtime schema state limit exceeded")); }
    let id = nodes.len();
    nodes.push(Node { kind, minimum });
    Ok(Some(id))
}

#[derive(Clone, Debug)]
enum Frame<'a> {
    Value(usize),
    ObjectNext { node: usize, next: usize, close: bool },
    ObjectAfter { node: usize, next: usize },
    Key { node: usize, choices: Vec<usize>, position: usize },
    ArrayNext { node: usize, count: usize, close: bool },
    ArrayAfter { node: usize, count: usize },
    Literal { node: usize, choices: Vec<usize>, position: usize },
    String { node: usize, bytes: usize, chars: usize, mode: StringMode, source: Option<SourceCursor<'a>> },
    Number { integer: bool, bytes: [u8; NUMBER_BYTES], len: usize, tail: usize },
}

#[derive(Clone, Copy, Debug)]
enum StringMode {
    Plain,
    Escape,
    Unicode { position: usize, high: u8 },
    Utf8 { remaining: u8, width: u8, low: u8, high: u8 },
}

/// A request-owned exact byte state, suitable for `VocabMaskOracle`.
#[derive(Clone, Debug)]
pub struct JsonState<'a> {
    program: &'a JsonProgram,
    stack: Vec<Frame<'a>>,
    used: usize,
    failed: bool,
}

impl<'a> JsonState<'a> {
    /// Acceptance allows EOS; it does not automatically truncate a number or
    /// an enum whose accepted spelling prefixes another accepted spelling.
    #[must_use]
    pub fn is_accepting(&self) -> bool {
        !self.failed && (self.stack.is_empty() || (self.stack.len() == 1 && self.frame_minimum(&self.stack[0]) == Some(0)))
    }

    #[must_use]
    pub const fn consumed_bytes(&self) -> usize { self.used }

    pub fn consume_bytes(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&byte| self.consume(byte))
    }

    fn fields(&self, node: usize) -> &[Property] {
        let Kind::Object(fields) = &self.program.nodes[node].kind else { unreachable!("object frame") };
        fields
    }
    fn literals(&self, node: usize) -> &[Vec<u8>] {
        let Kind::Literals(values) = &self.program.nodes[node].kind else { unreachable!("literal frame") };
        values
    }
    fn array(&self, node: usize) -> (usize, usize) {
        let Kind::Array { child, maximum } = self.program.nodes[node].kind else { unreachable!("array frame") };
        (child, maximum)
    }
    fn object_tail(&self, node: usize, next: usize, comma: bool) -> usize {
        let required = self.fields(node).get(next).map_or(0, |field| field.required_suffix_bytes);
        required + usize::from(comma || required == 0)
    }
    fn frame_minimum(&self, frame: &Frame<'_>) -> Option<usize> {
        Some(match frame {
            Frame::Value(node) => self.program.nodes[*node].minimum,
            Frame::ObjectAfter { node, next } => self.object_tail(*node, *next, true),
            Frame::ObjectNext { node, next, close } => {
                let fields = self.fields(*node);
                if *close || fields.get(*next).is_some_and(|field| field.required_suffix_bytes != 0) {
                    self.object_tail(*node, *next, false)
                } else {
                    fields[*next..].iter().map(|f| f.key.len() + self.program.nodes[f.child].minimum + 1).min()?
                }
            }
            Frame::Key { node, choices, position } => choices.iter().map(|&i| {
                let field = &self.fields(*node)[i];
                field.key.len() - position + self.program.nodes[field.child].minimum + self.object_tail(*node, i + 1, true)
            }).min()?,
            Frame::ArrayNext { node, close, .. } => if *close { 1 } else { self.program.nodes[self.array(*node).0].minimum + 1 },
            Frame::ArrayAfter { .. } => 1,
            Frame::Literal { node, choices, position } => choices.iter().map(|&i| self.literals(*node)[i].len() - position).min()?,
            Frame::String { mode, source, .. } => 1 + match mode {
                StringMode::Plain => 0,
                StringMode::Escape => match source { Some(cursor) => source::escape_tail(cursor)?, None => 1 },
                StringMode::Unicode { position, .. } => 4 - position,
                StringMode::Utf8 { remaining, .. } => usize::from(*remaining),
            },
            Frame::Number { tail, .. } => *tail,
        })
    }

    fn advance(&mut self, byte: u8) -> bool {
        // Epsilon expansion and scalar-delimiter replay consume no input.
        loop {
            let Some(frame) = self.stack.pop() else { return false; };
            match frame {
                Frame::Value(node) => match &self.program.nodes[node].kind {
                    Kind::Object(_) => {
                        if byte != b'{' { return false; }
                        self.stack.push(Frame::ObjectNext { node, next: 0, close: true }); return true;
                    }
                    Kind::Array { .. } => {
                        if byte != b'[' { return false; }
                        self.stack.push(Frame::ArrayNext { node, count: 0, close: true }); return true;
                    }
                    Kind::String { verbatim, .. } => {
                        if byte != b'"' { return false; }
                        let source = if *verbatim { self.program.source.as_ref().map(|b| b.language.cursor()) } else { None };
                        self.stack.push(Frame::String { node, bytes: 0, chars: 0, mode: StringMode::Plain, source }); return true;
                    }
                    Kind::Number { integer } => self.stack.push(Frame::Number { integer: *integer, bytes: [0; NUMBER_BYTES], len: 0, tail: 1 }),
                    Kind::Literals(values) => self.stack.push(Frame::Literal { node, choices: (0..values.len()).collect(), position: 0 }),
                },
                Frame::ObjectNext { node, next, close } => {
                    let fields = self.fields(node);
                    let suffix_bytes = fields.get(next).map_or(0, |field| field.required_suffix_bytes);
                    if byte == b'}' && close && suffix_bytes == 0 { return true; }
                    // The suffix total drops strictly just after the first
                    // required key. All choices up to and including it are legal.
                    let end = next + fields[next..].partition_point(|field| field.required_suffix_bytes == suffix_bytes);
                    if next == end { return false; }
                    self.stack.push(Frame::Key { node, choices: (next..end).collect(), position: 0 });
                }
                Frame::Key { node, mut choices, position } => {
                    choices.retain(|&i| self.fields(node)[i].key.get(position) == Some(&byte));
                    if choices.is_empty() { return false; }
                    let position = position + 1;
                    if choices.len() == 1 && self.fields(node)[choices[0]].key.len() == position {
                        let i = choices[0]; let child = self.fields(node)[i].child;
                        self.stack.push(Frame::ObjectAfter { node, next: i + 1 });
                        self.stack.push(Frame::Value(child));
                    } else { self.stack.push(Frame::Key { node, choices, position }); }
                    return true;
                }
                Frame::ObjectAfter { node, next } => {
                    let fields = self.fields(node);
                    if byte == b'}' && fields.get(next).is_none_or(|field| field.required_suffix_bytes == 0) { return true; }
                    if byte != b',' || next == fields.len() { return false; }
                    self.stack.push(Frame::ObjectNext { node, next, close: false }); return true;
                }
                Frame::ArrayNext { node, count, close } => {
                    if byte == b']' && close { return true; }
                    let (child, maximum) = self.array(node);
                    if count >= maximum { return false; }
                    self.stack.push(Frame::ArrayAfter { node, count: count + 1 });
                    self.stack.push(Frame::Value(child));
                }
                Frame::ArrayAfter { node, count } => {
                    if byte == b']' { return true; }
                    if byte != b',' || count >= self.array(node).1 { return false; }
                    self.stack.push(Frame::ArrayNext { node, count, close: false }); return true;
                }
                Frame::Literal { node, mut choices, position } => {
                    let accepting = choices.iter().any(|&i| self.literals(node)[i].len() == position);
                    choices.retain(|&i| self.literals(node)[i].get(position) == Some(&byte));
                    if choices.is_empty() { if accepting { continue; } return false; }
                    let position = position + 1;
                    if choices.iter().any(|&i| self.literals(node)[i].len() > position) {
                        self.stack.push(Frame::Literal { node, choices, position });
                    }
                    return true;
                }
                Frame::String { node, mut bytes, mut chars, mode, mut source } => {
                    if source.as_mut().is_some_and(|cursor| !source::advance(cursor, mode, byte)) { return false; }
                    let Kind::String { max_bytes, max_chars, .. } = self.program.nodes[node].kind else { unreachable!("string frame") };
                    let room = bytes < max_bytes && chars < max_chars;
                    let next_mode = match mode {
                        StringMode::Plain => match byte {
                            b'"' => return true,
                            b'\\' if room => StringMode::Escape,
                            0x20..=0x7f if room => { bytes += 1; chars += 1; StringMode::Plain }
                            0xc2..=0xf4 if room => {
                                let (width, low, high) = match byte {
                                    0xc2..=0xdf => (2, 0x80, 0xbf),
                                    0xe0 => (3, 0xa0, 0xbf), 0xed => (3, 0x80, 0x9f),
                                    0xe1..=0xef => (3, 0x80, 0xbf),
                                    0xf0 => (4, 0x90, 0xbf), 0xf4 => (4, 0x80, 0x8f),
                                    _ => (4, 0x80, 0xbf),
                                };
                                if max_bytes - bytes < usize::from(width) { return false; }
                                StringMode::Utf8 { remaining: width - 1, width, low, high }
                            }
                            _ => return false,
                        },
                        StringMode::Escape => match byte {
                            b'"' | b'\\' | b'b' | b'f' | b'n' | b'r' | b't' => { bytes += 1; chars += 1; StringMode::Plain }
                            b'u' => StringMode::Unicode { position: 0, high: 0 },
                            _ => return false,
                        },
                        StringMode::Unicode { position, high } => match (position, byte) {
                            (0 | 1, b'0') => StringMode::Unicode { position: position + 1, high },
                            (2, b'0' | b'1') => StringMode::Unicode { position: 3, high: byte - b'0' },
                            (3, b'0'..=b'9' | b'a'..=b'f') => {
                                let low = if byte <= b'9' { byte - b'0' } else { byte - b'a' + 10 };
                                if [8, 9, 10, 12, 13].contains(&(high * 16 + low)) { return false; }
                                bytes += 1; chars += 1; StringMode::Plain
                            }
                            _ => return false,
                        },
                        StringMode::Utf8 { remaining, width, low, high } => {
                            if byte < low || byte > high { return false; }
                            if remaining == 1 { bytes += usize::from(width); chars += 1; StringMode::Plain }
                            else { StringMode::Utf8 { remaining: remaining - 1, width, low: 0x80, high: 0xbf } }
                        }
                    };
                    self.stack.push(Frame::String { node, bytes, chars, mode: next_mode, source }); return true;
                }
                Frame::Number { integer, mut bytes, len, tail } => {
                    if !matches!(byte, b'0'..=b'9' | b'-' | b'.' | b'e') {
                        if len > 0 && tail == 0 { continue; } return false;
                    }
                    if len == NUMBER_BYTES { return false; }
                    bytes[len] = byte;
                    let Some(tail) = number_tail(&bytes[..len + 1], integer) else { return false; };
                    self.stack.push(Frame::Number { integer, bytes, len: len + 1, tail }); return true;
                }
            }
        }
    }
}

impl ByteState for JsonState<'_> {
    fn consume(&mut self, byte: u8) -> bool {
        if self.failed || self.used == self.program.max_output_bytes { self.failed = true; return false; }
        if !self.advance(byte) { self.failed = true; return false; }
        self.used += 1;
        let needed = self.stack.iter().try_fold(self.used, |total, frame| total.checked_add(self.frame_minimum(frame)?));
        if needed.is_none_or(|n| n > self.program.max_output_bytes) { self.failed = true; return false; }
        true
    }
}

// The finite canonical number language is: i64/u64 integers, or normalized
// scientific decimals with at most 38 significant digits and exponent -308..308.
// Representable integers never use scientific spelling. This exact prefix test
// considers exponent completions, so 1e1 is viable (1e100) but is not accepting.
fn number_tail(bytes: &[u8], integer: bool) -> Option<usize> {
    let negative = bytes.first() == Some(&b'-');
    let body = if negative { &bytes[1..] } else { bytes };
    if body.is_empty() { return Some(1); }
    if body[0] == b'0' { return (body.len() == 1 && !negative).then_some(0); }
    if !matches!(body[0], b'1'..=b'9') { return None; }
    let bound = if negative { 1_u128 << 63 } else { u128::from(u64::MAX) };
    if body.iter().all(u8::is_ascii_digit) {
        let magnitude = body.iter().try_fold(0_u128, |v, b| v.checked_mul(10)?.checked_add(u128::from(b - b'0')))?;
        return (magnitude <= bound).then_some(0);
    }
    if integer { return None; }
    let split = body.iter().position(|&b| b == b'e');
    let mantissa = split.map_or(body, |i| &body[..i]);
    let fraction = if mantissa.len() == 1 { &[][..] }
        else if mantissa.get(1) == Some(&b'.') { &mantissa[2..] }
        else { return None; };
    if !fraction.iter().all(u8::is_ascii_digit) || fraction.len() > 37 { return None; }
    let needs_digit = mantissa.len() > 1 && (fraction.is_empty() || fraction.last() == Some(&b'0'));
    if needs_digit && (split.is_some() || fraction.len() == 37) { return None; }
    let mut coefficient = u128::from(mantissa[0] - b'0');
    for &b in fraction { coefficient = coefficient * 10 + u128::from(b - b'0'); }
    if needs_digit { coefficient = coefficient * 10 + 1; }
    let fractional_digits = fraction.len() + usize::from(needs_digit);
    let mut maximum_integer_shift = None;
    if coefficient <= bound {
        let mut value = coefficient; let mut shift = 0;
        while value <= bound / 10 { value *= 10; shift += 1; }
        maximum_integer_shift = Some(shift);
    }
    let exponent_prefix = split.map_or(&[][..], |i| &body[i + 1..]);
    let plain_integers = maximum_integer_shift.map(|shift| {
        let first = fractional_digits as i32;
        (first, first + shift)
    });
    let tail = exponent_tail(exponent_prefix, plain_integers)?;
    Some(tail + usize::from(split.is_none()) + usize::from(needs_digit))
}

// Shortest completion of a normalized exponent in -308..=308. Exponents
// that would yield an i64/u64 integer require plain output instead. They
// form one contiguous interval of nonnegative exponents for each mantissa.
fn exponent_tail(prefix: &[u8], plain_integers: Option<(i32, i32)>) -> Option<usize> {
    use super::schema::{MAX_ADJUSTED_DECIMAL_EXPONENT, MIN_ADJUSTED_DECIMAL_EXPONENT};

    let all_plain = |low, high| plain_integers.is_some_and(|(first, last)| low >= first && high <= last);
    if prefix.is_empty() {
        // If all single digits are forbidden, "-1" always fits.
        return Some(if all_plain(0, 9) { 2 } else { 1 });
    }
    let negative = prefix[0] == b'-';
    let digits = if negative { &prefix[1..] } else { prefix };
    if digits.is_empty() { return Some(1); }
    if digits[0] == b'0' {
        return (!negative && digits.len() == 1 && !all_plain(0, 0)).then_some(0);
    }
    if digits.len() > 3 || !digits.iter().all(u8::is_ascii_digit) { return None; }
    let magnitude = digits.iter().fold(0_i32, |value, &byte| value * 10 + i32::from(byte - b'0'));
    if negative {
        return (-magnitude >= MIN_ADJUSTED_DECIMAL_EXPONENT).then_some(0);
    }

    // Appending a digit expands [lo, hi] to [10*lo, 10*hi + 9].
    // A completion exists at this width if any bounded endpoint lies
    // outside the excluded interval. A nonzero prefix has at most 3 widths.
    let (mut low, mut high) = (magnitude, magnitude);
    let mut extra = 0;
    while low <= MAX_ADJUSTED_DECIMAL_EXPONENT {
        if !all_plain(low, high.min(MAX_ADJUSTED_DECIMAL_EXPONENT)) { return Some(extra); }
        low *= 10;
        high = high * 10 + 9;
        extra += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(schema: &str) -> JsonProgram { JsonProgram::compile(schema, CompileLimits::default()).unwrap() }
    fn accepts(program: &JsonProgram, text: &str) -> bool {
        let mut state = program.initial_state(); state.consume_bytes(text.as_bytes()) && state.is_accepting()
    }

    #[test]
    fn every_key_prefix_matches_the_budgeted_object_language() {
        let schema = r#"{"type":"object","additionalProperties":false,"properties":{
            "a":{"type":"null"},"aa":{"type":"null"},"aaa":{"type":"null"},
            "b":{"type":"null"},"bb":{"type":"null"},"c":{"type":"null"}
        },"required":["aaa","bb"]}"#;
        let mut documents = Vec::new();
        for subset in 0..16 {
            let fields = [("a", subset & 1 != 0), ("aa", subset & 2 != 0), ("aaa", true),
                ("b", subset & 4 != 0), ("bb", true), ("c", subset & 8 != 0)];
            let entries: Vec<_> = fields.into_iter().filter(|(_, present)| *present)
                .map(|(name, _)| format!(r#""{name}":null"#)).collect();
            documents.push(format!("{{{}}}", entries.join(",")));
        }
        let minimum = documents.iter().map(String::len).min().unwrap();
        let maximum = documents.iter().map(String::len).max().unwrap();
        for cap in minimum..=maximum {
            let program = JsonProgram::compile(schema, CompileLimits {
                max_output_bytes: cap, ..CompileLimits::default()
            }).unwrap();
            for candidate in &documents {
                let mut state = program.initial_state();
                for (i, byte) in candidate.bytes().enumerate() {
                    let viable = documents.iter().any(|doc| doc.len() <= cap && doc.starts_with(&candidate[..=i]));
                    assert_eq!(state.consume(byte), viable, "cap={cap} prefix={}", &candidate[..=i]);
                    if !viable { break; }
                }
                assert_eq!(state.is_accepting(), candidate.len() <= cap);
            }
        }
    }

    #[test]
    fn nested_required_optional_objects_and_arrays_execute() {
        let p = program(r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"array","items":{"type":"integer"},"maxItems":2},"c":{"type":"null"}},"required":["b"],"additionalProperties":false}"#);
        for text in [r#"{"b":[]}"#, r#"{"a":true,"b":[-2,3],"c":null}"#] {
            assert!(accepts(&p, text), "{text}"); p.validate_json(text).unwrap();
        }
        for text in ["{}", r#"{"a":true}"#, r#"{"b":[1,2,3]}"#, r#"{"b":[],"a":true}"#, r#"{"b":[],}"#, r#"{"b":[1,]}"#, r#"{"b":[],"b":[]}"#] { assert!(!accepts(&p, text), "{text}"); }
    }

    #[test]
    fn optional_keys_with_common_prefix_keep_all_choices() {
        let p = program(r#"{"type":"object","properties":{"a":{"type":"null"},"aa":{"type":"null"},"ab":{"type":"null"}},"required":[],"additionalProperties":false}"#);
        for text in ["{}", r#"{"a":null}"#, r#"{"aa":null}"#, r#"{"ab":null}"#, r#"{"a":null,"aa":null,"ab":null}"#] { assert!(accepts(&p, text)); }
    }

    #[test]
    fn unicode_length_is_codepoints_with_a_separate_byte_cap() {
        let p = program(r#"{"type":"string","maxLength":1}"#);
        for text in [r#""é""#, r#""𐀀""#, r#""\n""#] { assert!(accepts(&p, text)); p.validate_json(text).unwrap(); }
        for text in [r#""éa""#, "\"e\u{301}\""] { assert!(!accepts(&p, text)); assert!(p.validate_json(text).is_err()); }
        let p = JsonProgram::compile(r#"{"type":"string"}"#, CompileLimits { max_string_bytes: 1, ..CompileLimits::default() }).unwrap();
        assert!(!accepts(&p, r#""é""#));
    }

    #[test]
    fn invalid_utf8_surrogates_and_noncanonical_escapes_are_excluded() {
        let p = program(r#"{"type":"string"}"#);
        for bytes in [vec![b'"', 0xc0], vec![b'"', 0xed, 0xa0], vec![b'"', 0xf4, 0x90], b"\"\\uD800\"".to_vec(), b"\"\\/\"".to_vec(), b"\"\\u000a\"".to_vec()] {
            assert!(!p.initial_state().consume_bytes(&bytes));
        }
        assert!(accepts(&p, r#""\u0000\u001f\t\\\"""#));
    }

    #[test]
    fn incomplete_utf8_is_viable_but_never_accepting() {
        let p = program(r#"{"type":"string","maxLength":1}"#);
        let mut s = p.initial_state();
        for &b in &[b'"', 0xf0, 0x90, 0x80] { assert!(s.consume(b)); assert!(!s.is_accepting()); }
        assert!(s.consume_bytes(&[0x80, b'"'])); assert!(s.is_accepting());
    }

    #[test]
    fn exponent_completion_matches_all_617_canonical_spellings() {
        use std::collections::{BTreeMap, BTreeSet};

        let spellings: Vec<_> = (-308_i32..=308).map(|e| (e, e.to_string())).collect();
        let mut prefixes = BTreeSet::new();
        for (_, text) in &spellings {
            for n in 0..=text.len() { prefixes.insert(&text[..n]); }
        }
        prefixes.extend(["-0", "00", "01", "+1", "--", "1-", "309", "-309", "3080", "e", "0x"]);
        for plain in [None, Some((0, 0)), Some((0, 19)), Some((1, 18)),
            Some((9, 10)), Some((18, 32)), Some((37, 37))] {
            let mut oracle = BTreeMap::new();
            for (e, text) in &spellings {
                if plain.is_some_and(|(lo, hi)| (lo..=hi).contains(e)) { continue; }
                for n in 0..=text.len() {
                    let remaining = text.len() - n;
                    let best = oracle.entry(&text[..n]).or_insert(remaining);
                    *best = (*best).min(remaining);
                }
            }
            for &prefix in &prefixes {
                assert_eq!(exponent_tail(prefix.as_bytes(), plain), oracle.get(prefix).copied(),
                    "prefix={prefix:?} plain={plain:?}");
            }
        }
    }

    #[test]
    fn numbers_match_exact_canonical_domain_and_bounds() {
        let p = program(r#"{"type":"number"}"#);
        for text in ["0", "-1", "18446744073709551615", "-9223372036854775808", "1.5e0", "1e-308", "1e308", "1e100"] { assert!(accepts(&p, text), "{text}"); p.validate_json(text).unwrap(); }
        for text in ["-0", "01", "1.0", "1e0", "1e309", "1e-309", "18446744073709551616", "-9223372036854775809", "1.0e3", "1e+20", "1e01"] { assert!(!accepts(&p, text), "{text}"); }
        let mut s = p.initial_state(); assert!(s.consume_bytes(b"1e1")); assert!(!s.is_accepting());
        assert!(s.consume_bytes(b"00")); assert!(s.is_accepting());
    }

    #[test]
    fn numeric_enum_prefixes_do_not_force_early_completion() {
        let p = program(r#"{"type":"array","items":{"type":"integer","enum":[1,10]},"maxItems":1}"#);
        assert!(accepts(&p, "[1]")); assert!(accepts(&p, "[10]")); assert!(!accepts(&p, "[11]"));
    }

    #[test]
    fn output_budget_reserves_every_required_suffix() {
        let schema = r#"{"type":"object","properties":{"x":{"type":"string"},"y":{"type":"null"}},"required":["x","y"],"additionalProperties":false}"#;
        let text = r#"{"x":"","y":null}"#;
        let p = JsonProgram::compile(schema, CompileLimits { max_output_bytes: text.len(), ..CompileLimits::default() }).unwrap();
        assert!(accepts(&p, text));
        assert!(!p.initial_state().consume_bytes(br#"{"x":"a"#));
    }

    #[test]
    fn clones_are_independent_and_failed_states_stay_failed() {
        let p = program(r#"{"type":"boolean"}"#); let s = p.initial_state();
        let mut left = s.clone(); let mut right = s.clone();
        assert!(left.consume_bytes(b"true")); assert!(right.consume_bytes(b"false"));
        assert!(!left.consume(b'x')); assert!(!left.consume(b' ')); assert!(right.is_accepting());
        assert_eq!(s.consumed_bytes(), 0);
    }

    #[test]
    fn unknown_keywords_duplicate_keys_and_unbound_source_refuse() {
        for schema in [r#"{"type":"string","type":"string"}"#, r#"{"type":"string","pattern":".*"}"#, r#"{"type":"string","x-fnlp-source":"verbatim"}"#, r#"{"type":"string","maxLength":-1}"#] {
            assert!(JsonProgram::compile(schema, CompileLimits::default()).is_err());
        }
    }

    #[test]
    fn zero_length_string_and_zero_item_array_have_real_transitions() {
        let p = program(r#"{"type":"string","maxLength":0}"#); assert!(accepts(&p, "\"\"")); assert!(!accepts(&p, "\"a\""));
        let p = program(r#"{"type":"array","items":{"type":"boolean"},"maxItems":0}"#); assert!(accepts(&p, "[]")); assert!(!accepts(&p, "[true]"));
    }

    #[test]
    fn every_prefix_of_valid_nested_output_remains_live() {
        let p = program(r#"{"type":"array","items":{"type":"string","maxLength":8},"maxItems":4}"#);
        let text = "[\"é\",\"\\n\",\"𐀀\"]";
        let mut s = p.initial_state();
        for &b in text.as_bytes() { assert!(s.consume(b)); }
        assert!(s.is_accepting()); assert_eq!(s.consumed_bytes(), text.len()); p.validate_json(text).unwrap();
    }

    #[test]
    fn declaration_validation_shares_native_scalar_length_and_enum_semantics() {
        for schema in [r#"{"type":"string","maxLength":2}"#,
            r#"{"type":"string","maxLength":2,"enum":["é","上海"]}"#] {
            let declaration = JsonValidationSchema::compile(schema, CompileLimits::default()).unwrap();
            let native = JsonProgram::compile(schema, CompileLimits::default()).unwrap();
            for text in [r#""é""#, r#""上海""#, r#""é上海""#, "null"] {
                let value = crate::validation::parse_json(text).unwrap();
                assert_eq!(declaration.validate_value(&value).is_ok(), native.validate_json(text).is_ok());
            }
        }
        let invalid = r#"{"type":"string","maxLength":1,"enum":["上海"]}"#;
        assert!(JsonValidationSchema::compile(invalid, CompileLimits::default()).is_err());
        assert!(JsonProgram::compile(invalid, CompileLimits::default()).is_err());
    }

    #[test]
    fn declaration_validation_never_grants_unbound_source_execution() {
        let schema = r#"{"type":"string","maxLength":1,"enum":["é"],"x-fnlp-source":"verbatim"}"#;
        let declaration = JsonValidationSchema::compile(schema, CompileLimits::default()).unwrap();
        assert!(declaration.requires_verbatim_source());
        declaration.validate_value(&crate::validation::parse_json(r#""é""#).unwrap()).unwrap();
        assert!(JsonProgram::compile(schema, CompileLimits::default()).is_err());
        let actual = JsonProgram::compile_with_source(schema, "é", CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
        actual.validate_json(r#""é""#).unwrap();
        assert!(JsonProgram::compile_with_source(schema, "absent", CompileLimits::default(), SourceRuntimeLimits::default()).is_err());
    }

    #[test]
    fn declaration_validation_keeps_native_depth_and_minimum_output_caps() {
        let too_small = CompileLimits { max_output_bytes: 1, ..CompileLimits::default() };
        let schema = r#"{"type":"string"}"#;
        assert!(JsonValidationSchema::compile(schema, too_small).is_err());
        assert!(JsonProgram::compile(schema, too_small).is_err());
        let mut nested = "{\"type\":\"null\"}".to_owned();
        for _ in 0..65 { nested = format!("{{\"type\":\"array\",\"items\":{nested},\"maxItems\":1}}"); }
        assert!(JsonValidationSchema::compile(&nested, CompileLimits::default()).is_err());
        assert!(JsonProgram::compile(&nested, CompileLimits::default()).is_err());
    }

    #[test]
    fn length_rule_storage_never_repeats_long_ancestor_property_names() {
        fn stored_key_bytes(schema: &SchemaNode) -> usize {
            match schema {
                SchemaNode::Object { properties, .. } => properties.iter().map(|(key, child)| key.len() + stored_key_bytes(child)).sum(),
                SchemaNode::Array { items, .. } => stored_key_bytes(items),
                _ => 0,
            }
        }
        let long = "private/~".repeat(256);
        let mut properties = serde_json::Map::new(); let mut values = serde_json::Map::new();
        for index in 0..64 {
            let key = format!("k{index:02}");
            properties.insert(key.clone(), serde_json::json!({"type":"string","maxLength":1}));
            values.insert(key, serde_json::json!("é"));
        }
        let mut outer = serde_json::Map::new();
        outer.insert(long.clone(), serde_json::json!({"type":"object","properties":properties,"additionalProperties":false}));
        let schema = serde_json::to_string(&serde_json::json!({"type":"object","properties":outer,"additionalProperties":false})).unwrap();
        let native = JsonProgram::compile(&schema, CompileLimits::default()).unwrap();
        let declaration = JsonValidationSchema::compile(&schema, CompileLimits::default()).unwrap();
        assert_eq!(stored_key_bytes(native.schema.root()), long.len() + 64 * 3);
        assert_eq!(stored_key_bytes(&declaration.root), long.len() + 64 * 3);
        assert!(stored_key_bytes(native.schema.root()) < schema.len());
        let mut document = serde_json::Map::new(); document.insert(long, serde_json::Value::Object(values));
        let text = serde_json::to_string(&document).unwrap();
        let value = crate::validation::parse_json(&text).unwrap();
        native.validate_json(&text).unwrap(); declaration.validate_value(&value).unwrap();
    }
}
