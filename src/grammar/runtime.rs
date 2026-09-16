//! Executable, bounded canonical JSON byte language for constrained decoding.
//!
//! The descriptive compiler graph is not an execution machine. This module
//! supplies its missing byte transitions. A clone owns only a bounded parser
//! stack; schema/literal data are borrowed, and generated documents are never
//! copied into vocabulary-trie states. Every accepted prefix has a completion
//! within the output-byte bound. Source products remain explicitly refused.

use std::collections::BTreeMap;

use crate::validation::{JsonValue as ValidationValue, JsonLimits, parse_json_with_limits, validate_value};

use super::{
    compiler::{CompileLimits, CompiledSchema, SchemaNode, compile_json_schema},
    mask::ByteState,
    schema::{IntegerValue, JsonValue, ScalarValue, SchemaError, escape_json_string, parse_json},
};

/// Versioned canonical byte-language and code-point length semantics.
pub const JSON_RUNTIME_VERSION: &str = "canonical-json-runtime-v1";
const MAX_DEPTH: usize = 64;
const NUMBER_BYTES: usize = 48;

#[derive(Clone, Debug)]
struct Property {
    key: Vec<u8>,
    child: usize,
    required: bool,
}

#[derive(Clone, Debug)]
enum Kind {
    Object(Vec<Property>),
    Array { child: usize, maximum: usize },
    String { max_bytes: usize, max_chars: usize },
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
    lengths: BTreeMap<String, usize>,
    exponents: Vec<(i32, Vec<u8>)>,
}

impl JsonProgram {
    /// Compile the existing supported schema subset with standard code-point
    /// `maxLength`, separate from the engine's UTF-8 byte cap.
    ///
    /// The legacy compiler interprets `maxLength` as bytes. Remove only that
    /// keyword from a lossless copy, retain its independently checked bounds,
    /// and let the compiler enforce every other keyword and deployment cap.
    /// The original schema, not this normalization, belongs in request identity.
    pub fn compile(source: &str, limits: CompileLimits) -> Result<Self, SchemaError> {
        if source.len() > limits.max_schema_bytes {
            return Err(resource("schema byte limit exceeded"));
        }
        let mut raw = parse_json(source)?;
        let mut lengths = BTreeMap::new();
        take_lengths(&mut raw, "$", &mut lengths, 0)?;
        let normalized = render(&raw);
        // Canonical spelling can expand exact decimals. This internal copy is
        // bounded by the already checked source plus the exact parser domain.
        let mut compiler_limits = limits;
        compiler_limits.max_schema_bytes = normalized.len();
        let schema = compile_json_schema(&normalized, compiler_limits)?;
        if schema.requires_verbatim_source() {
            return Err(SchemaError::UnsupportedKeyword {
                pointer: "$".to_owned(), keyword: "runtime source-product gate".to_owned(),
            });
        }
        let mut nodes = Vec::new();
        let root = build(schema.root(), "$", &lengths, &mut nodes, limits.max_states, 0)?;
        if nodes[root].minimum > limits.max_output_bytes {
            return Err(resource("no JSON value fits output byte limit"));
        }
        Ok(Self {
            schema, nodes, root, max_output_bytes: limits.max_output_bytes, lengths,
            exponents: (-308..=308).map(|e: i32| (e, e.to_string().into_bytes())).collect(),
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

    /// Independent whole-value verification, never acceptance by parser state.
    /// The existing validator receives schema data, not transitions or masks.
    pub fn validate_json(&self, text: &str) -> Result<(), SchemaError> {
        if text.len() > self.max_output_bytes { return Err(resource("output byte limit exceeded")); }
        let value = parse_json_with_limits(text, JsonLimits {
            max_input_bytes: self.max_output_bytes,
            max_string_lexeme_bytes: self.max_output_bytes,
            max_container_entries: self.max_output_bytes,
            ..JsonLimits::default()
        }).map_err(|_| invalid("independent JSON parsing failed"))?;
        validate_value(self.schema.root(), &value)
            .map_err(|_| invalid("independent JSON validation failed"))?;
        validate_lengths(self.schema.root(), &value, "$", &self.lengths)
    }
}

fn resource(reason: &str) -> SchemaError {
    SchemaError::Resource { pointer: "$".to_owned(), reason: reason.to_owned() }
}
fn invalid(reason: &str) -> SchemaError {
    SchemaError::InvalidSchema { pointer: "$".to_owned(), reason: reason.to_owned() }
}
fn child_path(path: &str, key: &str) -> String {
    format!("{path}/properties/{}", key.replace('~', "~0").replace('/', "~1"))
}

fn take_lengths(value: &mut JsonValue, path: &str, lengths: &mut BTreeMap<String, usize>, depth: usize) -> Result<(), SchemaError> {
    if depth > MAX_DEPTH { return Err(resource("schema nesting limit exceeded")); }
    let JsonValue::Object(object) = value else { return Ok(()); };
    if object.get("type").and_then(JsonValue::as_string) == Some("string") {
        if let Some(bound) = object.remove("maxLength") {
            let count = match bound.as_number().and_then(|n| n.integer_value()) {
                Some(IntegerValue::Signed(n)) if n >= 0 => n as u64,
                Some(IntegerValue::Unsigned(n)) => n,
                _ => return Err(invalid("maxLength must be a nonnegative integer")),
            };
            lengths.insert(path.to_owned(), usize::try_from(count).unwrap_or(usize::MAX));
        }
    }
    if let Some(JsonValue::Object(properties)) = object.get_mut("properties") {
        for (name, child) in properties { take_lengths(child, &child_path(path, name), lengths, depth + 1)?; }
    }
    if let Some(items) = object.get_mut("items") {
        take_lengths(items, &format!("{path}/items"), lengths, depth + 1)?;
    }
    Ok(())
}

fn render(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".to_owned(),
        JsonValue::Boolean(v) => v.to_string(),
        JsonValue::String(v) => escape_json_string(v),
        JsonValue::Number(v) => v.canonical_spelling(),
        JsonValue::Array(v) => format!("[{}]", v.iter().map(render).collect::<Vec<_>>().join(",")),
        JsonValue::Object(v) => format!("{{{}}}", v.iter().map(|(k, v)| format!("{}:{}", escape_json_string(k), render(v))).collect::<Vec<_>>().join(",")),
    }
}

fn build(schema: &SchemaNode, path: &str, lengths: &BTreeMap<String, usize>, nodes: &mut Vec<Node>, cap: usize, depth: usize) -> Result<usize, SchemaError> {
    if depth > MAX_DEPTH || nodes.len() >= cap { return Err(resource("runtime schema state limit exceeded")); }
    let literals = |values: &[ScalarValue]| {
        let mut bytes: Vec<_> = values.iter().map(|v| v.canonical_json().into_bytes()).collect();
        bytes.sort();
        Kind::Literals(bytes)
    };
    let kind = match schema {
        SchemaNode::Object { properties, required } => {
            let mut fields = Vec::new();
            for (name, child) in properties {
                fields.push(Property {
                    key: format!("{}:", escape_json_string(name)).into_bytes(),
                    child: build(child, &child_path(path, name), lengths, nodes, cap, depth + 1)?,
                    required: required.contains(name),
                });
            }
            Kind::Object(fields)
        }
        SchemaNode::Array { items, max_items } => Kind::Array {
            child: build(items, &format!("{path}/items"), lengths, nodes, cap, depth + 1)?, maximum: *max_items,
        },
        SchemaNode::String { max_bytes, allowed, .. } => {
            let max_chars = lengths.get(path).copied().unwrap_or(usize::MAX);
            if let Some(values) = allowed {
                if values.iter().any(|v| matches!(v, ScalarValue::String(s) if s.chars().count() > max_chars)) {
                    return Err(invalid("enum/const exceeds maxLength"));
                }
                literals(values)
            } else { Kind::String { max_bytes: *max_bytes, max_chars } }
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
            let mut total = 2_usize;
            let mut count = 0;
            for field in fields.iter().filter(|f| f.required) {
                total = total.checked_add(field.key.len()).and_then(|n| n.checked_add(nodes[field.child].minimum))
                    .and_then(|n| n.checked_add(usize::from(count > 0))).ok_or_else(|| resource("minimum output overflow"))?;
                count += 1;
            }
            total
        }
        Kind::Array { .. } | Kind::String { .. } => 2,
        Kind::Number { .. } => 1,
        Kind::Literals(values) => values.iter().map(Vec::len).min().ok_or_else(|| invalid("empty scalar language"))?,
    };
    if nodes.len() >= cap { return Err(resource("runtime schema state limit exceeded")); }
    let id = nodes.len();
    nodes.push(Node { kind, minimum });
    Ok(id)
}

fn validate_lengths(schema: &SchemaNode, value: &ValidationValue, path: &str, lengths: &BTreeMap<String, usize>) -> Result<(), SchemaError> {
    match (schema, value) {
        (SchemaNode::String { .. }, ValidationValue::String(s)) => {
            if lengths.get(path).is_some_and(|&cap| s.chars().count() > cap) {
                return Err(SchemaError::Validation { pointer: path.to_owned(), reason: "maxLength exceeded".to_owned() });
            }
        }
        (SchemaNode::Object { properties, .. }, ValidationValue::Object(values)) => {
            for (key, child) in properties {
                if let Some(value) = values.get(key) { validate_lengths(child, value, &child_path(path, key), lengths)?; }
            }
        }
        (SchemaNode::Array { items, .. }, ValidationValue::Array(values)) => {
            for value in values { validate_lengths(items, value, &format!("{path}/items"), lengths)?; }
        }
        _ => {}
    }
    Ok(())
}

#[derive(Clone, Debug)]
enum Frame {
    Value(usize),
    ObjectNext { node: usize, next: usize, close: bool },
    ObjectAfter { node: usize, next: usize },
    Key { node: usize, choices: Vec<usize>, position: usize },
    ArrayNext { node: usize, count: usize, close: bool },
    ArrayAfter { node: usize, count: usize },
    Literal { node: usize, choices: Vec<usize>, position: usize },
    String { node: usize, bytes: usize, chars: usize, mode: StringMode },
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
    stack: Vec<Frame>,
    used: usize,
    failed: bool,
}

impl JsonState<'_> {
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
        let mut bytes = 1;
        let mut separator = comma;
        for field in self.fields(node)[next..].iter().filter(|f| f.required) {
            bytes += field.key.len() + self.program.nodes[field.child].minimum + usize::from(separator);
            separator = true;
        }
        bytes
    }
    fn frame_minimum(&self, frame: &Frame) -> Option<usize> {
        Some(match frame {
            Frame::Value(node) => self.program.nodes[*node].minimum,
            Frame::ObjectAfter { node, next } => self.object_tail(*node, *next, true),
            Frame::ObjectNext { node, next, close } => {
                let fields = self.fields(*node);
                if *close || fields[*next..].iter().any(|f| f.required) {
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
            Frame::String { mode, .. } => 1 + match mode {
                StringMode::Plain => 0, StringMode::Escape => 1,
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
                    Kind::String { .. } => {
                        if byte != b'"' { return false; }
                        self.stack.push(Frame::String { node, bytes: 0, chars: 0, mode: StringMode::Plain }); return true;
                    }
                    Kind::Number { integer } => self.stack.push(Frame::Number { integer: *integer, bytes: [0; NUMBER_BYTES], len: 0, tail: 1 }),
                    Kind::Literals(values) => self.stack.push(Frame::Literal { node, choices: (0..values.len()).collect(), position: 0 }),
                },
                Frame::ObjectNext { node, next, close } => {
                    let fields = self.fields(node);
                    if byte == b'}' && close && !fields[next..].iter().any(|f| f.required) { return true; }
                    let end = fields[next..].iter().position(|f| f.required).map_or(fields.len(), |i| next + i + 1);
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
                    if byte == b'}' && !fields[next..].iter().any(|f| f.required) { return true; }
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
                Frame::String { node, mut bytes, mut chars, mode } => {
                    let Kind::String { max_bytes, max_chars } = self.program.nodes[node].kind else { unreachable!("string frame") };
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
                    self.stack.push(Frame::String { node, bytes, chars, mode: next_mode }); return true;
                }
                Frame::Number { integer, mut bytes, len, tail } => {
                    if !matches!(byte, b'0'..=b'9' | b'-' | b'.' | b'e') {
                        if len > 0 && tail == 0 { continue; } return false;
                    }
                    if len == NUMBER_BYTES { return false; }
                    bytes[len] = byte;
                    let Some(tail) = number_tail(&bytes[..len + 1], integer, &self.program.exponents) else { return false; };
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
fn number_tail(bytes: &[u8], integer: bool, exponents: &[(i32, Vec<u8>)]) -> Option<usize> {
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
    let tail = exponents.iter().filter(|(e, spelling)| {
        spelling.starts_with(exponent_prefix) && !maximum_integer_shift.is_some_and(|shift| {
            *e >= fractional_digits as i32 && *e <= fractional_digits as i32 + shift
        })
    }).map(|(_, spelling)| spelling.len() - exponent_prefix.len()).min()?;
    Some(tail + usize::from(split.is_none()) + usize::from(needs_digit))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(schema: &str) -> JsonProgram { JsonProgram::compile(schema, CompileLimits::default()).unwrap() }
    fn accepts(program: &JsonProgram, text: &str) -> bool {
        let mut state = program.initial_state(); state.consume_bytes(text.as_bytes()) && state.is_accepting()
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
}
