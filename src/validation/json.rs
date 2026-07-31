//! A bounded, duplicate-key-rejecting JSON parser owned by validation.
//!
//! Deliberately do not route this through the grammar parser: the successful
//! response gate needs an implementation that can disagree with grammar.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Maximum significant decimal digits accepted by the validator.
pub const MAX_SIGNIFICANT_DECIMAL_DIGITS: usize = 38;
/// Minimum adjusted decimal exponent accepted by the validator.
pub const MIN_ADJUSTED_DECIMAL_EXPONENT: i32 = -308;
/// Maximum adjusted decimal exponent accepted by the validator.
pub const MAX_ADJUSTED_DECIMAL_EXPONENT: i32 = 308;
/// Hard lexical cap for one number token.
pub const MAX_NUMBER_LEXEME_BYTES: usize = 512;

/// Bounds applied before parser allocations grow with hostile input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JsonLimits {
    pub max_input_bytes: usize,
    pub max_depth: usize,
    pub max_container_entries: usize,
    pub max_string_lexeme_bytes: usize,
    pub max_number_lexeme_bytes: usize,
}

impl Default for JsonLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 256 * 1024,
            max_depth: 128,
            max_container_entries: 16 * 1024,
            max_string_lexeme_bytes: 64 * 1024,
            max_number_lexeme_bytes: MAX_NUMBER_LEXEME_BYTES,
        }
    }
}

/// A parsed JSON value retaining exact number semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JsonValue {
    Null,
    Boolean(bool),
    String(String),
    Number(Decimal),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    pub const fn kind_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Boolean(_) => "boolean",
            Self::String(_) => "string",
            Self::Number(_) => "number",
            Self::Array(_) => "array",
            Self::Object(_) => "object",
        }
    }
}

/// Exact signed/unsigned 64-bit integer representation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum IntegerValue {
    Signed(i64),
    Unsigned(u64),
}

/// A normalized finite decimal: `sign * coefficient * 10^exponent`.
///
/// Zero is always `(0, 0, 0)`; no nonzero coefficient has trailing zeroes.
/// Therefore `1`, `1.0`, and `1e0` compare equal without any IEEE rounding.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Decimal {
    sign: i8,
    coefficient: u128,
    exponent: i32,
}

impl Decimal {
    /// Parse exactly one RFC-8259 number token without an intermediate float.
    pub fn parse(input: &str) -> Result<Self, DecimalError> {
        if input.is_empty() || input.len() > MAX_NUMBER_LEXEME_BYTES {
            return Err(DecimalError::LexemeLength);
        }

        let bytes = input.as_bytes();
        let mut cursor = 0_usize;
        let mut sign = 1_i8;
        if bytes[cursor] == b'-' {
            sign = -1;
            cursor += 1;
            if cursor == bytes.len() {
                return Err(DecimalError::Syntax);
            }
        }

        let integer_start = cursor;
        match bytes.get(cursor) {
            Some(b'0') => cursor += 1,
            Some(b'1'..=b'9') => {
                cursor += 1;
                while matches!(bytes.get(cursor), Some(b'0'..=b'9')) {
                    cursor += 1;
                }
            }
            _ => return Err(DecimalError::Syntax),
        }
        if bytes[integer_start] == b'0' && matches!(bytes.get(cursor), Some(b'0'..=b'9')) {
            return Err(DecimalError::Syntax);
        }

        let mut digits = input[integer_start..cursor].to_owned();
        let mut fraction_digits = 0_i32;
        if matches!(bytes.get(cursor), Some(b'.')) {
            cursor += 1;
            let fraction_start = cursor;
            while matches!(bytes.get(cursor), Some(b'0'..=b'9')) {
                cursor += 1;
            }
            if fraction_start == cursor {
                return Err(DecimalError::Syntax);
            }
            fraction_digits = i32::try_from(cursor - fraction_start)
                .map_err(|_| DecimalError::ExponentOutOfRange)?;
            digits.push_str(&input[fraction_start..cursor]);
        }

        let mut scientific_exponent = 0_i32;
        if matches!(bytes.get(cursor), Some(b'e' | b'E')) {
            cursor += 1;
            let mut exponent_sign = 1_i32;
            if matches!(bytes.get(cursor), Some(b'+' | b'-')) {
                if bytes[cursor] == b'-' {
                    exponent_sign = -1;
                }
                cursor += 1;
            }
            let exponent_start = cursor;
            while matches!(bytes.get(cursor), Some(b'0'..=b'9')) {
                scientific_exponent = scientific_exponent
                    .checked_mul(10)
                    .and_then(|value| value.checked_add(i32::from(bytes[cursor] - b'0')))
                    .ok_or(DecimalError::ExponentOutOfRange)?;
                cursor += 1;
            }
            if exponent_start == cursor {
                return Err(DecimalError::Syntax);
            }
            scientific_exponent = scientific_exponent
                .checked_mul(exponent_sign)
                .ok_or(DecimalError::ExponentOutOfRange)?;
        }
        if cursor != bytes.len() {
            return Err(DecimalError::Syntax);
        }

        let significant_start = digits
            .bytes()
            .position(|byte| byte != b'0')
            .unwrap_or(digits.len());
        if significant_start == digits.len() {
            return Ok(Self {
                sign: 0,
                coefficient: 0,
                exponent: 0,
            });
        }
        let mut significant = digits[significant_start..].to_owned();
        let trailing_zeroes = significant
            .bytes()
            .rev()
            .take_while(|byte| *byte == b'0')
            .count();
        significant.truncate(significant.len() - trailing_zeroes);
        if significant.len() > MAX_SIGNIFICANT_DECIMAL_DIGITS {
            return Err(DecimalError::TooManySignificantDigits {
                observed: significant.len(),
            });
        }

        let mut coefficient = 0_u128;
        for byte in significant.bytes() {
            coefficient = coefficient
                .checked_mul(10)
                .and_then(|value| value.checked_add(u128::from(byte - b'0')))
                .ok_or(DecimalError::CoefficientOverflow)?;
        }
        let exponent = scientific_exponent
            .checked_sub(fraction_digits)
            .and_then(|value| value.checked_add(i32::try_from(trailing_zeroes).ok()?))
            .ok_or(DecimalError::ExponentOutOfRange)?;
        let adjusted = exponent
            .checked_add(
                i32::try_from(significant.len() - 1)
                    .map_err(|_| DecimalError::ExponentOutOfRange)?,
            )
            .ok_or(DecimalError::ExponentOutOfRange)?;
        if !(MIN_ADJUSTED_DECIMAL_EXPONENT..=MAX_ADJUSTED_DECIMAL_EXPONENT).contains(&adjusted) {
            return Err(DecimalError::AdjustedExponentOutOfRange { adjusted });
        }

        Ok(Self {
            sign,
            coefficient,
            exponent,
        })
    }

    pub const fn is_zero(&self) -> bool {
        self.sign == 0
    }

    /// Return an exact 64-bit integer only when this decimal denotes one.
    pub fn integer_value(&self) -> Option<IntegerValue> {
        if self.is_zero() {
            return Some(IntegerValue::Signed(0));
        }
        if self.exponent < 0 {
            return None;
        }
        let mut magnitude = self.coefficient;
        for _ in 0..self.exponent {
            magnitude = magnitude.checked_mul(10)?;
        }
        if self.sign < 0 {
            let min_magnitude = (i64::MAX as u128) + 1;
            if magnitude > min_magnitude {
                return None;
            }
            if magnitude == min_magnitude {
                return Some(IntegerValue::Signed(i64::MIN));
            }
            return i64::try_from(magnitude)
                .ok()
                .map(|value| IntegerValue::Signed(-value));
        }
        if magnitude <= i64::MAX as u128 {
            i64::try_from(magnitude).ok().map(IntegerValue::Signed)
        } else {
            u64::try_from(magnitude).ok().map(IntegerValue::Unsigned)
        }
    }

    pub fn is_integer_in_64_bit_domain(&self) -> bool {
        self.integer_value().is_some()
    }

    /// The canonical exact spelling used for equality with schema literals.
    pub fn canonical_spelling(&self) -> String {
        if self.is_zero() {
            return "0".to_owned();
        }
        if let Some(integer) = self.integer_value() {
            return match integer {
                IntegerValue::Signed(value) => value.to_string(),
                IntegerValue::Unsigned(value) => value.to_string(),
            };
        }
        let digits = self.coefficient.to_string();
        let adjusted = self.exponent + i32::try_from(digits.len() - 1).unwrap_or(i32::MAX);
        let mut output = String::new();
        if self.sign < 0 {
            output.push('-');
        }
        let mut characters = digits.chars();
        if let Some(first) = characters.next() {
            output.push(first);
        }
        let rest = characters.as_str();
        if !rest.is_empty() {
            output.push('.');
            output.push_str(rest);
        }
        output.push('e');
        output.push_str(&adjusted.to_string());
        output
    }
}

/// Exact-decimal parse failures for the independent validator implementation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecimalError {
    Syntax,
    LexemeLength,
    TooManySignificantDigits { observed: usize },
    CoefficientOverflow,
    ExponentOutOfRange,
    AdjustedExponentOutOfRange { adjusted: i32 },
}

impl fmt::Display for DecimalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax => formatter.write_str("invalid RFC-8259 number syntax"),
            Self::LexemeLength => write!(
                formatter,
                "number lexeme exceeds {MAX_NUMBER_LEXEME_BYTES} bytes"
            ),
            Self::TooManySignificantDigits { observed } => write!(
                formatter,
                "number has {observed} significant digits; maximum is {MAX_SIGNIFICANT_DECIMAL_DIGITS}"
            ),
            Self::CoefficientOverflow => {
                formatter.write_str("checked decimal coefficient overflow")
            }
            Self::ExponentOutOfRange => formatter.write_str("checked decimal exponent overflow"),
            Self::AdjustedExponentOutOfRange { adjusted } => write!(
                formatter,
                "adjusted exponent {adjusted} is outside [{MIN_ADJUSTED_DECIMAL_EXPONENT}, {MAX_ADJUSTED_DECIMAL_EXPONENT}]"
            ),
        }
    }
}

impl std::error::Error for DecimalError {}

/// Classification for a JSON-boundary rejection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JsonParseErrorKind {
    Syntax,
    DuplicateKey,
    Numeric,
    Resource,
}

/// Typed parser error with a byte location but never a private input echo.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JsonParseError {
    kind: JsonParseErrorKind,
    pointer: String,
    byte_offset: usize,
    reason: String,
}

impl JsonParseError {
    pub const fn kind(&self) -> JsonParseErrorKind {
        self.kind
    }

    pub fn pointer(&self) -> &str {
        &self.pointer
    }

    pub const fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl fmt::Display for JsonParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "validation JSON {:?} at {} (byte {}): {}",
            self.kind, self.pointer, self.byte_offset, self.reason
        )
    }
}

impl std::error::Error for JsonParseError {}

/// An internal source location retained by the independent parser so the
/// schema walker can report a safe byte and Unicode-scalar coordinate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct JsonLocation {
    pub byte_offset: usize,
    pub scalar_offset: usize,
}

pub(crate) struct ParsedJson {
    pub value: JsonValue,
    pub locations: BTreeMap<String, JsonLocation>,
}

/// Parse one JSON document under the ordinary product bounds.
pub fn parse_json(input: &str) -> Result<JsonValue, JsonParseError> {
    parse_json_with_limits(input, JsonLimits::default())
}

/// Parse one JSON document under caller-supplied bounded limits.
pub fn parse_json_with_limits(
    input: &str,
    limits: JsonLimits,
) -> Result<JsonValue, JsonParseError> {
    parse_json_with_locations_and_limits(input, limits).map(|document| document.value)
}

pub(crate) fn parse_json_with_locations(input: &str) -> Result<ParsedJson, JsonParseError> {
    parse_json_with_locations_and_limits(input, JsonLimits::default())
}

fn parse_json_with_locations_and_limits(
    input: &str,
    limits: JsonLimits,
) -> Result<ParsedJson, JsonParseError> {
    if input.len() > limits.max_input_bytes {
        return Err(JsonParseError {
            kind: JsonParseErrorKind::Resource,
            pointer: "$".to_owned(),
            byte_offset: limits.max_input_bytes,
            reason: format!(
                "JSON input is {} bytes and exceeds cap {}",
                input.len(),
                limits.max_input_bytes
            ),
        });
    }
    Parser {
        input,
        offset: 0,
        limits,
        locations: BTreeMap::new(),
    }
    .parse_document()
}

struct Parser<'a> {
    input: &'a str,
    offset: usize,
    limits: JsonLimits,
    locations: BTreeMap<String, JsonLocation>,
}

impl<'a> Parser<'a> {
    fn parse_document(mut self) -> Result<ParsedJson, JsonParseError> {
        self.skip_whitespace();
        let value = self.parse_value(0, "$")?;
        self.skip_whitespace();
        if self.offset != self.input.len() {
            return Err(self.error(
                JsonParseErrorKind::Syntax,
                "$",
                "trailing bytes after JSON value",
            ));
        }
        Ok(ParsedJson {
            value,
            locations: self.locations,
        })
    }

    fn parse_value(&mut self, depth: usize, pointer: &str) -> Result<JsonValue, JsonParseError> {
        if depth > self.limits.max_depth {
            return Err(self.error(
                JsonParseErrorKind::Resource,
                pointer,
                format!("JSON nesting exceeds {} containers", self.limits.max_depth),
            ));
        }
        self.skip_whitespace();
        self.locations
            .entry(pointer.to_owned())
            .or_insert(JsonLocation {
                byte_offset: self.offset,
                scalar_offset: self.input[..self.offset].chars().count(),
            });
        match self.peek() {
            Some(b'{') => self.parse_object(depth + 1, pointer),
            Some(b'[') => self.parse_array(depth + 1, pointer),
            Some(b'"') => self.parse_string(pointer).map(JsonValue::String),
            Some(b't') => self.parse_literal("true", JsonValue::Boolean(true), pointer),
            Some(b'f') => self.parse_literal("false", JsonValue::Boolean(false), pointer),
            Some(b'n') => self.parse_literal("null", JsonValue::Null, pointer),
            Some(b'-' | b'0'..=b'9') => self.parse_number(pointer).map(JsonValue::Number),
            Some(_) => {
                Err(self.error(JsonParseErrorKind::Syntax, pointer, "expected a JSON value"))
            }
            None => Err(self.error(
                JsonParseErrorKind::Syntax,
                pointer,
                "unexpected end of JSON input",
            )),
        }
    }

    fn parse_object(&mut self, depth: usize, pointer: &str) -> Result<JsonValue, JsonParseError> {
        self.expect_byte(b'{', pointer)?;
        self.skip_whitespace();
        let mut entries = BTreeMap::new();
        let mut seen = BTreeSet::new();
        if self.consume_byte(b'}') {
            return Ok(JsonValue::Object(entries));
        }
        loop {
            if entries.len() == self.limits.max_container_entries {
                return Err(self.error(
                    JsonParseErrorKind::Resource,
                    pointer,
                    format!(
                        "object entries exceed cap {}",
                        self.limits.max_container_entries
                    ),
                ));
            }
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return Err(self.error(
                    JsonParseErrorKind::Syntax,
                    pointer,
                    "object key must be a JSON string",
                ));
            }
            let key = self.parse_string(pointer)?;
            let child_pointer = pointer_key(pointer, &key);
            if !seen.insert(key.clone()) {
                return Err(self.error(
                    JsonParseErrorKind::DuplicateKey,
                    &child_pointer,
                    "duplicate object key",
                ));
            }
            self.skip_whitespace();
            self.expect_byte(b':', &child_pointer)?;
            let value = self.parse_value(depth, &child_pointer)?;
            entries.insert(key, value);
            self.skip_whitespace();
            if self.consume_byte(b'}') {
                break;
            }
            self.expect_byte(b',', pointer)?;
        }
        Ok(JsonValue::Object(entries))
    }

    fn parse_array(&mut self, depth: usize, pointer: &str) -> Result<JsonValue, JsonParseError> {
        self.expect_byte(b'[', pointer)?;
        self.skip_whitespace();
        let mut values = Vec::new();
        if self.consume_byte(b']') {
            return Ok(JsonValue::Array(values));
        }
        loop {
            if values.len() == self.limits.max_container_entries {
                return Err(self.error(
                    JsonParseErrorKind::Resource,
                    pointer,
                    format!(
                        "array entries exceed cap {}",
                        self.limits.max_container_entries
                    ),
                ));
            }
            let child_pointer = pointer_index(pointer, values.len());
            values.push(self.parse_value(depth, &child_pointer)?);
            self.skip_whitespace();
            if self.consume_byte(b']') {
                break;
            }
            self.expect_byte(b',', pointer)?;
        }
        Ok(JsonValue::Array(values))
    }

    fn parse_literal(
        &mut self,
        expected: &str,
        value: JsonValue,
        pointer: &str,
    ) -> Result<JsonValue, JsonParseError> {
        if self.input[self.offset..].starts_with(expected) {
            self.offset += expected.len();
            Ok(value)
        } else {
            Err(self.error(JsonParseErrorKind::Syntax, pointer, "invalid JSON literal"))
        }
    }

    fn parse_number(&mut self, pointer: &str) -> Result<Decimal, JsonParseError> {
        let start = self.offset;
        if self.consume_byte(b'-') && self.offset == self.input.len() {
            return Err(self.error(
                JsonParseErrorKind::Syntax,
                pointer,
                "incomplete JSON number",
            ));
        }
        match self.peek() {
            Some(b'0') => {
                self.offset += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.error(
                        JsonParseErrorKind::Syntax,
                        pointer,
                        "JSON numbers may not have leading zeroes",
                    ));
                }
            }
            Some(b'1'..=b'9') => {
                self.offset += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.offset += 1;
                }
            }
            _ => {
                return Err(self.error(JsonParseErrorKind::Syntax, pointer, "invalid JSON number"));
            }
        }
        if self.consume_byte(b'.') {
            let fraction_start = self.offset;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.offset += 1;
            }
            if fraction_start == self.offset {
                return Err(self.error(
                    JsonParseErrorKind::Syntax,
                    pointer,
                    "JSON fraction requires digits",
                ));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.offset += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.offset += 1;
            }
            let exponent_start = self.offset;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.offset += 1;
            }
            if exponent_start == self.offset {
                return Err(self.error(
                    JsonParseErrorKind::Syntax,
                    pointer,
                    "JSON exponent requires digits",
                ));
            }
        }
        let lexeme_len = self.offset - start;
        if lexeme_len > self.limits.max_number_lexeme_bytes {
            return Err(self.error(
                JsonParseErrorKind::Resource,
                pointer,
                format!(
                    "number lexeme is {lexeme_len} bytes and exceeds cap {}",
                    self.limits.max_number_lexeme_bytes
                ),
            ));
        }
        Decimal::parse(&self.input[start..self.offset])
            .map_err(|error| self.error(JsonParseErrorKind::Numeric, pointer, error.to_string()))
    }

    fn parse_string(&mut self, pointer: &str) -> Result<String, JsonParseError> {
        let start = self.offset;
        self.expect_byte(b'"', pointer)?;
        let mut output = String::new();
        loop {
            let Some(byte) = self.peek() else {
                return Err(self.error(
                    JsonParseErrorKind::Syntax,
                    pointer,
                    "unterminated JSON string",
                ));
            };
            match byte {
                b'"' => {
                    self.offset += 1;
                    if self.offset - start > self.limits.max_string_lexeme_bytes {
                        return Err(self.error(
                            JsonParseErrorKind::Resource,
                            pointer,
                            format!(
                                "JSON string lexeme exceeds cap {}",
                                self.limits.max_string_lexeme_bytes
                            ),
                        ));
                    }
                    return Ok(output);
                }
                b'\\' => {
                    self.offset += 1;
                    let escaped = self.peek().ok_or_else(|| {
                        self.error(
                            JsonParseErrorKind::Syntax,
                            pointer,
                            "incomplete JSON escape",
                        )
                    })?;
                    self.offset += 1;
                    match escaped {
                        b'"' => output.push('"'),
                        b'\\' => output.push('\\'),
                        b'/' => output.push('/'),
                        b'b' => output.push('\u{0008}'),
                        b'f' => output.push('\u{000c}'),
                        b'n' => output.push('\n'),
                        b'r' => output.push('\r'),
                        b't' => output.push('\t'),
                        b'u' => output.push(self.parse_unicode_escape(pointer)?),
                        _ => {
                            return Err(self.error(
                                JsonParseErrorKind::Syntax,
                                pointer,
                                "unsupported JSON escape",
                            ));
                        }
                    }
                }
                0x00..=0x1f => {
                    return Err(self.error(
                        JsonParseErrorKind::Syntax,
                        pointer,
                        "unescaped control character in JSON string",
                    ));
                }
                _ => {
                    let character = self.input[self.offset..].chars().next().ok_or_else(|| {
                        self.error(
                            JsonParseErrorKind::Syntax,
                            pointer,
                            "invalid UTF-8 boundary",
                        )
                    })?;
                    output.push(character);
                    self.offset += character.len_utf8();
                }
            }
            if self.offset - start > self.limits.max_string_lexeme_bytes {
                return Err(self.error(
                    JsonParseErrorKind::Resource,
                    pointer,
                    format!(
                        "JSON string lexeme exceeds cap {}",
                        self.limits.max_string_lexeme_bytes
                    ),
                ));
            }
        }
    }

    fn parse_unicode_escape(&mut self, pointer: &str) -> Result<char, JsonParseError> {
        let first = self.parse_u16_escape(pointer)?;
        if (0xdc00..=0xdfff).contains(&first) {
            return Err(self.error(
                JsonParseErrorKind::Syntax,
                pointer,
                "unpaired low surrogate in JSON string",
            ));
        }
        if !(0xd800..=0xdbff).contains(&first) {
            return char::from_u32(u32::from(first)).ok_or_else(|| {
                self.error(
                    JsonParseErrorKind::Syntax,
                    pointer,
                    "invalid unicode escape",
                )
            });
        }
        if !self.input[self.offset..].starts_with("\\u") {
            return Err(self.error(
                JsonParseErrorKind::Syntax,
                pointer,
                "high surrogate requires a low surrogate",
            ));
        }
        self.offset += 2;
        let second = self.parse_u16_escape(pointer)?;
        if !(0xdc00..=0xdfff).contains(&second) {
            return Err(self.error(
                JsonParseErrorKind::Syntax,
                pointer,
                "high surrogate requires a low surrogate",
            ));
        }
        let scalar = 0x10000 + ((u32::from(first) - 0xd800) << 10) + (u32::from(second) - 0xdc00);
        char::from_u32(scalar).ok_or_else(|| {
            self.error(
                JsonParseErrorKind::Syntax,
                pointer,
                "invalid surrogate pair",
            )
        })
    }

    fn parse_u16_escape(&mut self, pointer: &str) -> Result<u16, JsonParseError> {
        if self.offset + 4 > self.input.len() {
            return Err(self.error(
                JsonParseErrorKind::Syntax,
                pointer,
                "incomplete unicode escape",
            ));
        }
        let mut value = 0_u16;
        for byte in self.input.as_bytes()[self.offset..self.offset + 4]
            .iter()
            .copied()
        {
            let digit = match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => {
                    return Err(self.error(
                        JsonParseErrorKind::Syntax,
                        pointer,
                        "invalid unicode escape digit",
                    ));
                }
            };
            value = (value << 4) | u16::from(digit);
        }
        self.offset += 4;
        Ok(value)
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.offset += 1;
        }
    }

    fn expect_byte(&mut self, expected: u8, pointer: &str) -> Result<(), JsonParseError> {
        if self.consume_byte(expected) {
            Ok(())
        } else {
            Err(self.error(
                JsonParseErrorKind::Syntax,
                pointer,
                format!("expected byte {:?}", char::from(expected)),
            ))
        }
    }

    fn consume_byte(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.offset += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.as_bytes().get(self.offset).copied()
    }

    fn error(
        &self,
        kind: JsonParseErrorKind,
        pointer: &str,
        reason: impl Into<String>,
    ) -> JsonParseError {
        JsonParseError {
            kind,
            pointer: pointer.to_owned(),
            byte_offset: self.offset,
            reason: reason.into(),
        }
    }
}

fn pointer_key(parent: &str, key: &str) -> String {
    let escaped = key.replace('~', "~0").replace('/', "~1");
    if parent == "$" {
        format!("/{escaped}")
    } else {
        format!("{parent}/{escaped}")
    }
}

fn pointer_index(parent: &str, index: usize) -> String {
    if parent == "$" {
        format!("/{index}")
    } else {
        format!("{parent}/{index}")
    }
}
