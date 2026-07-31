//! Duplicate-key-rejecting parsing and stable canonical JSON writing.
//!
//! This module is the only permitted raw JSON parse chokepoint for
//! authority-bearing input.  Boundary callers select [`ParseLimits`] before
//! parsing; they must not call `serde_json::from_str` or `from_reader`
//! directly.  The writer accepts typed `Serialize` values.  Authority-bearing
//! schemas use structs and `BTreeMap` where a dynamic key set is declared; the
//! writer deliberately does not expose a `Value`-map authority API.
//!
//! Canonical string escaping is pinned as follows: quotation mark and reverse
//! solidus use their short escapes; backspace, tab, line feed, form feed, and
//! carriage return use `\\b`, `\\t`, `\\n`, `\\f`, and `\\r`; every other
//! C0 control character uses a lower-case `\\u00xx` escape; all other Unicode
//! scalar values are emitted as their UTF-8 bytes.  Object keys sort by UTF-8
//! byte lexicographic order.  Output has no insignificant whitespace.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use serde::ser::{
    self, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant, SerializeTuple,
    SerializeTupleStruct, SerializeTupleVariant,
};
use serde::{Serialize, Serializer};
use serde::de::{self, DeserializeSeed, Error as _, MapAccess, SeqAccess, Visitor};
use serde_json::Value;

/// Default maximum number of nested object or array containers accepted by a
/// boundary parser.
pub const DEFAULT_MAX_DEPTH: usize = 64;

/// Default maximum decoded UTF-8 byte length of any JSON string or object key.
pub const DEFAULT_MAX_STRING_BYTES: usize = 1024 * 1024;

/// Caller-selectable input caps for the rejecting parse layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParseLimits {
    /// Maximum number of nested array/object containers.
    pub max_depth: usize,
    /// Maximum decoded UTF-8 byte length of a key or string value.
    pub max_string_bytes: usize,
}

impl Default for ParseLimits {
    fn default() -> Self {
        Self {
            max_depth: DEFAULT_MAX_DEPTH,
            max_string_bytes: DEFAULT_MAX_STRING_BYTES,
        }
    }
}

/// A JSON Pointer path.  The root displays as `$`; all non-root paths use the
/// RFC 6901 slash form, such as `/items/0/name`.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct JsonPath(String);

impl JsonPath {
    fn root() -> Self {
        Self(String::new())
    }

    fn key(&self, key: &str) -> Self {
        let escaped = key.replace('~', "~0").replace('/', "~1");
        Self(format!("{}/{}", self.0, escaped))
    }

    fn index(&self, index: usize) -> Self {
        Self(format!("{}/{}", self.0, index))
    }
}

impl fmt::Display for JsonPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            formatter.write_str("$")
        } else {
            formatter.write_str(&self.0)
        }
    }
}

/// Typed failures from the canonical JSON boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanonJsonError {
    /// Two semantically equal object keys appeared in one object.
    DuplicateKey { path: JsonPath },
    /// Input exceeded the caller's nested-container limit.
    DepthLimit { path: JsonPath, max_depth: usize },
    /// A decoded string or key exceeded the caller's size limit.
    StringLimit {
        path: JsonPath,
        max_string_bytes: usize,
        observed_bytes: usize,
    },
    /// The JSON text was otherwise invalid.
    Parse {
        message: String,
        line: usize,
        column: usize,
    },
    /// A value cannot be represented as JSON without loss.
    Serialize { message: String },
    /// JSON has no representation for IEEE-754 NaN or infinity.
    NonFiniteNumber,
}

impl fmt::Display for CanonJsonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateKey { path } => {
                write!(formatter, "duplicate JSON object key at {path}")
            }
            Self::DepthLimit { path, max_depth } => {
                write!(
                    formatter,
                    "JSON nesting limit {max_depth} exceeded at {path}"
                )
            }
            Self::StringLimit {
                path,
                max_string_bytes,
                observed_bytes,
            } => write!(
                formatter,
                "JSON string limit {max_string_bytes} bytes exceeded at {path} (observed {observed_bytes})"
            ),
            Self::Parse {
                message,
                line,
                column,
            } => write!(formatter, "invalid JSON at {line}:{column}: {message}"),
            Self::Serialize { message } => {
                write!(formatter, "JSON serialization failed: {message}")
            }
            Self::NonFiniteNumber => formatter.write_str("non-finite JSON number is forbidden"),
        }
    }
}

impl Error for CanonJsonError {}

/// Parse JSON while rejecting a duplicate key at any object depth.
pub fn parse_str(input: &str) -> Result<Value, CanonJsonError> {
    parse_str_with_limits(input, ParseLimits::default())
}

/// Parse JSON using explicit boundary limits.
pub fn parse_str_with_limits(input: &str, limits: ParseLimits) -> Result<Value, CanonJsonError> {
    match Preflight::new(input, limits).scan() {
        Ok(()) | Err(PreflightError::Malformed) => {}
        Err(PreflightError::DuplicateKey(path)) => {
            return Err(CanonJsonError::DuplicateKey { path });
        }
        Err(PreflightError::DepthLimit(path)) => {
            return Err(CanonJsonError::DepthLimit {
                path,
                max_depth: limits.max_depth,
            });
        }
        Err(PreflightError::StringLimit {
            path,
            observed_bytes,
        }) => {
            return Err(CanonJsonError::StringLimit {
                path,
                max_string_bytes: limits.max_string_bytes,
                observed_bytes,
            });
        }
    }

    let mut deserializer = serde_json::Deserializer::from_str(input);
    let value = ValueSeed {
        path: JsonPath::root(),
        depth: 0,
        limits,
    }
    .deserialize(&mut deserializer)
    .map_err(parse_error)?;
    deserializer.end().map_err(parse_error)?;
    validate_value_limits(&value, &JsonPath::root(), 0, limits)?;
    Ok(value)
}

/// Write a typed value as canonical JSON bytes.
///
/// The input is converted through Serde only to obtain its schema-shaped JSON
/// tree.  Object ordering and string escaping are emitted below, rather than
/// delegated to a map serializer.  `NaN` and both infinities are rejected.
pub fn canonical_bytes<T>(value: &T) -> Result<Vec<u8>, CanonJsonError>
where
    T: Serialize + ?Sized,
{
    ensure_finite(value)?;
    let value = serde_json::to_value(value).map_err(serialization_error)?;
    let mut output = Vec::new();
    write_value(&value, &mut output);
    Ok(output)
}

/// Write a typed value as canonical UTF-8 JSON text.
pub fn canonical_string<T>(value: &T) -> Result<String, CanonJsonError>
where
    T: Serialize + ?Sized,
{
    let bytes = canonical_bytes(value)?;
    String::from_utf8(bytes).map_err(|error| CanonJsonError::Serialize {
        message: format!("canonical writer emitted invalid UTF-8: {error}"),
    })
}

/// Reject duplicate keys in `input` and return its canonical byte form.
pub fn canonicalize_str(input: &str, limits: ParseLimits) -> Result<Vec<u8>, CanonJsonError> {
    let value = parse_str_with_limits(input, limits)?;
    let mut output = Vec::new();
    write_value(&value, &mut output);
    Ok(output)
}

fn parse_error(error: serde_json::Error) -> CanonJsonError {
    CanonJsonError::Parse {
        message: error.to_string(),
        line: error.line(),
        column: error.column(),
    }
}

fn serialization_error(error: serde_json::Error) -> CanonJsonError {
    // serde_json's serde data-model rejection for NaN and infinity is the
    // stable "not a JSON number" diagnostic.  Classify it rather than turning
    // it into a null or a lossy string representation.
    if error.to_string().contains("not a JSON number") {
        CanonJsonError::NonFiniteNumber
    } else {
        CanonJsonError::Serialize {
            message: error.to_string(),
        }
    }
}

fn validate_value_limits(
    value: &Value,
    path: &JsonPath,
    depth: usize,
    limits: ParseLimits,
) -> Result<(), CanonJsonError> {
    match value {
        Value::String(value) => check_value_string_limit(path, value, limits),
        Value::Array(values) => {
            check_value_depth(path, depth, limits)?;
            for (index, value) in values.iter().enumerate() {
                validate_value_limits(value, &path.index(index), depth + 1, limits)?;
            }
            Ok(())
        }
        Value::Object(values) => {
            check_value_depth(path, depth, limits)?;
            for (key, value) in values {
                let key_path = path.key(key);
                check_value_string_limit(&key_path, key, limits)?;
                validate_value_limits(value, &key_path, depth + 1, limits)?;
            }
            Ok(())
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => Ok(()),
    }
}

fn check_value_depth(
    path: &JsonPath,
    depth: usize,
    limits: ParseLimits,
) -> Result<(), CanonJsonError> {
    if depth >= limits.max_depth {
        Err(CanonJsonError::DepthLimit {
            path: path.clone(),
            max_depth: limits.max_depth,
        })
    } else {
        Ok(())
    }
}

fn check_value_string_limit(
    path: &JsonPath,
    value: &str,
    limits: ParseLimits,
) -> Result<(), CanonJsonError> {
    if value.len() > limits.max_string_bytes {
        Err(CanonJsonError::StringLimit {
            path: path.clone(),
            max_string_bytes: limits.max_string_bytes,
            observed_bytes: value.len(),
        })
    } else {
        Ok(())
    }
}

fn ensure_finite<T>(value: &T) -> Result<(), CanonJsonError>
where
    T: Serialize + ?Sized,
{
    value.serialize(FiniteProbe).map_err(|error| match error {
        FiniteProbeError::NonFiniteNumber => CanonJsonError::NonFiniteNumber,
        FiniteProbeError::Custom(message) => CanonJsonError::Serialize { message },
    })
}

#[derive(Debug)]
enum FiniteProbeError {
    NonFiniteNumber,
    Custom(String),
}

impl fmt::Display for FiniteProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteNumber => formatter.write_str("non-finite JSON number is forbidden"),
            Self::Custom(message) => formatter.write_str(message),
        }
    }
}

impl Error for FiniteProbeError {}

impl ser::Error for FiniteProbeError {
    fn custom<T>(message: T) -> Self
    where
        T: fmt::Display,
    {
        Self::Custom(message.to_string())
    }
}

/// A traversal-only Serde serializer.  serde_json normally converts non-finite
/// floats to `null` in some serializers, so inspect the original typed value
/// before any JSON tree is constructed.
struct FiniteProbe;

impl Serializer for FiniteProbe {
    type Ok = ();
    type Error = FiniteProbeError;
    type SerializeSeq = FiniteProbeCompound;
    type SerializeTuple = FiniteProbeCompound;
    type SerializeTupleStruct = FiniteProbeCompound;
    type SerializeTupleVariant = FiniteProbeCompound;
    type SerializeMap = FiniteProbeCompound;
    type SerializeStruct = FiniteProbeCompound;
    type SerializeStructVariant = FiniteProbeCompound;

    fn serialize_bool(self, _: bool) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_i8(self, _: i8) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_i16(self, _: i16) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_i32(self, _: i32) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_i64(self, _: i64) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_i128(self, _: i128) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_u8(self, _: u8) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_u16(self, _: u16) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_u32(self, _: u32) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_u64(self, _: u64) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_u128(self, _: u128) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_f32(self, value: f32) -> Result<Self::Ok, Self::Error> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(FiniteProbeError::NonFiniteNumber)
        }
    }

    fn serialize_f64(self, value: f64) -> Result<Self::Ok, Self::Error> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(FiniteProbeError::NonFiniteNumber)
        }
    }

    fn serialize_char(self, _: char) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_str(self, _: &str) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_bytes(self, _: &[u8]) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_some<T>(self, value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_unit_struct(self, _: &'static str) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn serialize_newtype_struct<T>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(self)
    }

    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Ok(FiniteProbeCompound)
    }

    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, Self::Error> {
        Ok(FiniteProbeCompound)
    }

    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        Ok(FiniteProbeCompound)
    }

    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        Ok(FiniteProbeCompound)
    }

    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        Ok(FiniteProbeCompound)
    }

    fn serialize_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Ok(FiniteProbeCompound)
    }

    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        Ok(FiniteProbeCompound)
    }
}

struct FiniteProbeCompound;

macro_rules! probe_elements {
    ($trait:ident) => {
        impl ser::$trait for FiniteProbeCompound {
            type Ok = ();
            type Error = FiniteProbeError;

            fn serialize_element<T>(&mut self, value: &T) -> Result<(), Self::Error>
            where
                T: Serialize + ?Sized,
            {
                value.serialize(FiniteProbe)
            }

            fn end(self) -> Result<Self::Ok, Self::Error> {
                Ok(())
            }
        }
    };
}

probe_elements!(SerializeSeq);
probe_elements!(SerializeTuple);

impl SerializeTupleStruct for FiniteProbeCompound {
    type Ok = ();
    type Error = FiniteProbeError;

    fn serialize_field<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(FiniteProbe)
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl SerializeTupleVariant for FiniteProbeCompound {
    type Ok = ();
    type Error = FiniteProbeError;

    fn serialize_field<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(FiniteProbe)
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl SerializeMap for FiniteProbeCompound {
    type Ok = ();
    type Error = FiniteProbeError;

    fn serialize_key<T>(&mut self, key: &T) -> Result<(), Self::Error>
    where
        T: Serialize + ?Sized,
    {
        key.serialize(FiniteProbe)
    }

    fn serialize_value<T>(&mut self, value: &T) -> Result<(), Self::Error>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(FiniteProbe)
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl SerializeStruct for FiniteProbeCompound {
    type Ok = ();
    type Error = FiniteProbeError;

    fn serialize_field<T>(&mut self, _: &'static str, value: &T) -> Result<(), Self::Error>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(FiniteProbe)
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

impl SerializeStructVariant for FiniteProbeCompound {
    type Ok = ();
    type Error = FiniteProbeError;

    fn serialize_field<T>(&mut self, _: &'static str, value: &T) -> Result<(), Self::Error>
    where
        T: Serialize + ?Sized,
    {
        value.serialize(FiniteProbe)
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }
}

fn write_value(value: &Value, output: &mut Vec<u8>) {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(true) => output.extend_from_slice(b"true"),
        Value::Bool(false) => output.extend_from_slice(b"false"),
        Value::Number(number) => output.extend_from_slice(number.to_string().as_bytes()),
        Value::String(string) => write_string(string, output),
        Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_value(value, output);
            }
            output.push(b']');
        }
        Value::Object(object) => {
            let mut entries: Vec<_> = object.iter().collect();
            entries.sort_unstable_by(|(left, _), (right, _)| left.as_bytes().cmp(right.as_bytes()));

            output.push(b'{');
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_string(key, output);
                output.push(b':');
                write_value(value, output);
            }
            output.push(b'}');
        }
    }
}

fn write_string(value: &str, output: &mut Vec<u8>) {
    output.push(b'"');
    for character in value.chars() {
        match character {
            '"' => output.extend_from_slice(br#"\""#),
            '\\' => output.extend_from_slice(br#"\\"#),
            '\u{0008}' => output.extend_from_slice(br#"\b"#),
            '\t' => output.extend_from_slice(br#"\t"#),
            '\n' => output.extend_from_slice(br#"\n"#),
            '\u{000C}' => output.extend_from_slice(br#"\f"#),
            '\r' => output.extend_from_slice(br#"\r"#),
            character if character <= '\u{001F}' => {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                let value = character as u32;
                output.extend_from_slice(b"\\u00");
                output.push(HEX[((value >> 4) & 0x0f) as usize]);
                output.push(HEX[(value & 0x0f) as usize]);
            }
            character => {
                let mut encoded = [0_u8; 4];
                output.extend_from_slice(character.encode_utf8(&mut encoded).as_bytes());
            }
        }
    }
    output.push(b'"');
}

struct ValueSeed {
    path: JsonPath,
    depth: usize,
    limits: ParseLimits,
}

impl<'de> DeserializeSeed<'de> for ValueSeed {
    type Value = Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_any(ValueVisitor {
            path: self.path,
            depth: self.depth,
            limits: self.limits,
        })
    }
}

struct ValueVisitor {
    path: JsonPath,
    depth: usize,
    limits: ParseLimits,
}

impl ValueVisitor {
    fn check_string<E>(&self, value: &str) -> Result<(), E>
    where
        E: de::Error,
    {
        if value.len() > self.limits.max_string_bytes {
            return Err(E::custom(format!(
                "JSON string limit {} bytes exceeded at {} (observed {})",
                self.limits.max_string_bytes,
                self.path,
                value.len()
            )));
        }
        Ok(())
    }

    fn check_container_depth<E>(&self) -> Result<(), E>
    where
        E: de::Error,
    {
        if self.depth >= self.limits.max_depth {
            return Err(E::custom(format!(
                "JSON nesting limit {} exceeded at {}",
                self.limits.max_depth, self.path
            )));
        }
        Ok(())
    }
}

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("non-finite JSON number is forbidden"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.check_string(value)?;
        Ok(Value::String(value.to_owned()))
    }

    fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_str(value)
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.check_string(&value)?;
        Ok(Value::String(value))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Null)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Null)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        ValueSeed {
            path: self.path,
            depth: self.depth,
            limits: self.limits,
        }
        .deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        self.check_container_depth()?;
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(ValueSeed {
            path: self.path.index(values.len()),
            depth: self.depth + 1,
            limits: self.limits,
        })? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        self.check_container_depth()?;
        let mut object = serde_json::Map::new();
        while let Some(key) = map.next_key::<String>()? {
            self.check_string(&key)?;
            let key_path = self.path.key(&key);
            if object.contains_key(&key) {
                return Err(A::Error::custom(format!(
                    "duplicate JSON object key at {key_path}"
                )));
            }
            let value = map.next_value_seed(ValueSeed {
                path: key_path,
                depth: self.depth + 1,
                limits: self.limits,
            })?;
            object.insert(key, value);
        }
        Ok(Value::Object(object))
    }
}

enum PreflightError {
    DuplicateKey(JsonPath),
    DepthLimit(JsonPath),
    StringLimit {
        path: JsonPath,
        observed_bytes: usize,
    },
    Malformed,
}

struct Preflight<'input> {
    input: &'input str,
    bytes: &'input [u8],
    index: usize,
    limits: ParseLimits,
}

impl<'input> Preflight<'input> {
    fn new(input: &'input str, limits: ParseLimits) -> Self {
        Self {
            input,
            bytes: input.as_bytes(),
            index: 0,
            limits,
        }
    }

    fn scan(mut self) -> Result<(), PreflightError> {
        self.skip_whitespace();
        self.scan_value(&JsonPath::root(), 0)?;
        self.skip_whitespace();
        if self.index == self.bytes.len() {
            Ok(())
        } else {
            Err(PreflightError::Malformed)
        }
    }

    fn scan_value(&mut self, path: &JsonPath, depth: usize) -> Result<(), PreflightError> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'{') => self.scan_object(path, depth),
            Some(b'[') => self.scan_array(path, depth),
            Some(b'"') => {
                let string = self.scan_string()?;
                self.check_string_limit(path, &string)
            }
            Some(b't') => self.consume_keyword(b"true"),
            Some(b'f') => self.consume_keyword(b"false"),
            Some(b'n') => self.consume_keyword(b"null"),
            Some(_) => self.scan_scalar(),
            None => Err(PreflightError::Malformed),
        }
    }

    fn scan_object(&mut self, path: &JsonPath, depth: usize) -> Result<(), PreflightError> {
        self.check_depth(path, depth)?;
        self.index += 1;
        self.skip_whitespace();
        if self.consume_if(b'}') {
            return Ok(());
        }

        let mut keys = BTreeSet::new();
        loop {
            self.skip_whitespace();
            let key = self.scan_string()?;
            let key_path = path.key(&key);
            self.check_string_limit(&key_path, &key)?;
            if !keys.insert(key) {
                return Err(PreflightError::DuplicateKey(key_path));
            }
            self.skip_whitespace();
            if !self.consume_if(b':') {
                return Err(PreflightError::Malformed);
            }
            self.scan_value(&key_path, depth + 1)?;
            self.skip_whitespace();
            if self.consume_if(b'}') {
                return Ok(());
            }
            if !self.consume_if(b',') {
                return Err(PreflightError::Malformed);
            }
        }
    }

    fn scan_array(&mut self, path: &JsonPath, depth: usize) -> Result<(), PreflightError> {
        self.check_depth(path, depth)?;
        self.index += 1;
        self.skip_whitespace();
        if self.consume_if(b']') {
            return Ok(());
        }

        let mut item_index = 0;
        loop {
            self.scan_value(&path.index(item_index), depth + 1)?;
            item_index += 1;
            self.skip_whitespace();
            if self.consume_if(b']') {
                return Ok(());
            }
            if !self.consume_if(b',') {
                return Err(PreflightError::Malformed);
            }
        }
    }

    fn scan_string(&mut self) -> Result<String, PreflightError> {
        let start = self.index;
        if !self.consume_if(b'"') {
            return Err(PreflightError::Malformed);
        }

        while let Some(byte) = self.peek() {
            match byte {
                b'"' => {
                    self.index += 1;
                    return serde_json::from_str(&self.input[start..self.index])
                        .map_err(|_| PreflightError::Malformed);
                }
                b'\\' => {
                    self.index += 1;
                    if self.peek().is_none() {
                        return Err(PreflightError::Malformed);
                    }
                    self.index += 1;
                }
                0x00..=0x1f => return Err(PreflightError::Malformed),
                _ => self.index += 1,
            }
        }
        Err(PreflightError::Malformed)
    }

    fn scan_scalar(&mut self) -> Result<(), PreflightError> {
        let start = self.index;
        while let Some(byte) = self.peek() {
            if matches!(byte, b' ' | b'\n' | b'\r' | b'\t' | b',' | b']' | b'}') {
                break;
            }
            self.index += 1;
        }
        if self.index == start {
            Err(PreflightError::Malformed)
        } else {
            Ok(())
        }
    }

    fn consume_keyword(&mut self, keyword: &[u8]) -> Result<(), PreflightError> {
        if self.bytes[self.index..].starts_with(keyword) {
            self.index += keyword.len();
            Ok(())
        } else {
            Err(PreflightError::Malformed)
        }
    }

    fn check_depth(&self, path: &JsonPath, depth: usize) -> Result<(), PreflightError> {
        if depth >= self.limits.max_depth {
            Err(PreflightError::DepthLimit(path.clone()))
        } else {
            Ok(())
        }
    }

    fn check_string_limit(&self, path: &JsonPath, value: &str) -> Result<(), PreflightError> {
        if value.len() > self.limits.max_string_bytes {
            Err(PreflightError::StringLimit {
                path: path.clone(),
                observed_bytes: value.len(),
            })
        } else {
            Ok(())
        }
    }

    fn consume_if(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.index).copied()
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.index += 1;
        }
    }
}
