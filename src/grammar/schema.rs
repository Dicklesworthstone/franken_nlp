//! Lossless JSON values and exact bounded-decimal primitives for schemas.
//!
//! This is deliberately a small JSON reader rather than a `serde_json::Value`
//! adapter: schema number literals must never acquire `f64` semantics.  The
//! grammar compiler consumes these values immediately after duplicate-key
//! rejection, keeping the user-schema boundary independent of parser-specific
//! number rounding.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// The maximum significant decimal digits in the normative v1 number domain.
pub const MAX_SIGNIFICANT_DECIMAL_DIGITS: usize = 38;
/// The largest permitted adjusted exponent in the v1 number domain.
pub const MAX_ADJUSTED_DECIMAL_EXPONENT: i32 = 308;
/// The smallest permitted adjusted exponent in the v1 number domain.
pub const MIN_ADJUSTED_DECIMAL_EXPONENT: i32 = -308;
/// A hard lexical cap prevents adversarial number tokens from consuming
/// unbounded parser time even when most digits are zero.
pub const MAX_NUMBER_LEXEME_BYTES: usize = 512;

/// A parsed JSON value whose number variant retains an exact decimal value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum JsonValue {
    Null,
    Boolean(bool),
    String(String),
    Number(ExactDecimal),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    pub(crate) const fn kind_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Boolean(_) => "boolean",
            Self::String(_) => "string",
            Self::Number(_) => "number",
            Self::Array(_) => "array",
            Self::Object(_) => "object",
        }
    }

    pub(crate) fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn as_boolean(&self) -> Option<bool> {
        match self {
            Self::Boolean(value) => Some(*value),
            _ => None,
        }
    }

    pub(crate) fn as_number(&self) -> Option<&ExactDecimal> {
        match self {
            Self::Number(value) => Some(value),
            _ => None,
        }
    }
}

/// JSON scalar values used by `const` and `enum` constraints.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ScalarValue {
    Null,
    Boolean(bool),
    String(String),
    Number(ExactDecimal),
}

impl ScalarValue {
    pub(crate) fn from_json(value: &JsonValue) -> Option<Self> {
        match value {
            JsonValue::Null => Some(Self::Null),
            JsonValue::Boolean(value) => Some(Self::Boolean(*value)),
            JsonValue::String(value) => Some(Self::String(value.clone())),
            JsonValue::Number(value) => Some(Self::Number(value.clone())),
            JsonValue::Array(_) | JsonValue::Object(_) => None,
        }
    }

    pub(crate) fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Boolean(_) => "boolean",
            Self::String(_) => "string",
            Self::Number(value) if value.integer_value().is_some() => "integer",
            Self::Number(_) => "number",
        }
    }

    /// Return the canonical JSON spelling required by constrained output.
    pub fn canonical_json(&self) -> String {
        match self {
            Self::Null => "null".to_owned(),
            Self::Boolean(value) => value.to_string(),
            Self::String(value) => escape_json_string(value),
            Self::Number(value) => value.canonical_spelling(),
        }
    }
}

/// Exact signed/unsigned 64-bit integer representation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum IntegerValue {
    Signed(i64),
    Unsigned(u64),
}

impl IntegerValue {
    pub const fn canonical_spelling(self) -> i128 {
        match self {
            Self::Signed(value) => value as i128,
            Self::Unsigned(value) => value as i128,
        }
    }
}

/// A normalized finite decimal: `sign × coefficient × 10^exponent`.
///
/// Zero is always held as `(0, 0, 0)`.  Nonzero coefficients have no leading
/// or trailing zeros, which makes equality mathematical rather than lexical:
/// `1`, `1.0`, and `1e0` compare equal.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ExactDecimal {
    sign: i8,
    coefficient: u128,
    exponent: i32,
}

impl ExactDecimal {
    /// Parse one RFC-8259 number token without ever creating an IEEE float.
    pub fn parse(input: &str) -> Result<Self, DecimalError> {
        if input.is_empty() || input.len() > MAX_NUMBER_LEXEME_BYTES {
            return Err(DecimalError::LexemeLength);
        }
        let bytes = input.as_bytes();
        let mut cursor = 0;
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
            fraction_digits = (cursor - fraction_start)
                .try_into()
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
                let digit = i32::from(bytes[cursor] - b'0');
                scientific_exponent = scientific_exponent
                    .checked_mul(10)
                    .and_then(|value| value.checked_add(digit))
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
        let retained = significant.len() - trailing_zeroes;
        significant.truncate(retained);
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
            .and_then(|value| value.checked_add(trailing_zeroes as i32))
            .ok_or(DecimalError::ExponentOutOfRange)?;
        let adjusted = exponent
            .checked_add((significant.len() - 1) as i32)
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

    /// Convert only exact integral values in the signed/unsigned i64 domain.
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

    /// Render the one canonical constrained-decoding spelling.
    pub fn canonical_spelling(&self) -> String {
        if self.is_zero() {
            return "0".to_owned();
        }
        if let Some(integer) = self.integer_value() {
            return integer.canonical_spelling().to_string();
        }
        let digits = self.coefficient.to_string();
        let adjusted = self.exponent + (digits.len() as i32 - 1);
        let mut output = String::new();
        if self.sign < 0 {
            output.push('-');
        }
        let mut characters = digits.chars();
        let Some(first) = characters.next() else {
            return "0".to_owned();
        };
        output.push(first);
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

/// Exact-decimal parse failures.
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

/// Parse the schema/instance JSON boundary and reject duplicate object keys.
pub(crate) fn parse_json(input: &str) -> Result<JsonValue, SchemaError> {
    RawJsonParser::new(input).parse()
}

/// The compiler's typed failure channel.  It always carries a JSON pointer or
/// `$` root marker and never echoes a complete potentially private document.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchemaError {
    Parse { offset: usize, reason: String },
    DuplicateKey { pointer: String },
    UnsupportedKeyword { pointer: String, keyword: String },
    InvalidSchema { pointer: String, reason: String },
    Numeric { pointer: String, reason: String },
    Resource { pointer: String, reason: String },
    Validation { pointer: String, reason: String },
}

impl SchemaError {
    pub fn pointer(&self) -> &str {
        match self {
            Self::Parse { .. } => "$",
            Self::DuplicateKey { pointer }
            | Self::UnsupportedKeyword { pointer, .. }
            | Self::InvalidSchema { pointer, .. }
            | Self::Numeric { pointer, .. }
            | Self::Resource { pointer, .. }
            | Self::Validation { pointer, .. } => pointer,
        }
    }

    pub fn keyword(&self) -> Option<&str> {
        match self {
            Self::UnsupportedKeyword { keyword, .. } => Some(keyword),
            _ => None,
        }
    }
}

impl fmt::Display for SchemaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse { offset, reason } => {
                write!(
                    formatter,
                    "schema JSON parse failure at byte {offset}: {reason}"
                )
            }
            Self::DuplicateKey { pointer } => {
                write!(formatter, "duplicate object key at {pointer}")
            }
            Self::UnsupportedKeyword { pointer, keyword } => {
                write!(formatter, "unsupported keyword {keyword:?} at {pointer}")
            }
            Self::InvalidSchema { pointer, reason }
            | Self::Numeric { pointer, reason }
            | Self::Resource { pointer, reason }
            | Self::Validation { pointer, reason } => write!(formatter, "{reason} at {pointer}"),
        }
    }
}

impl std::error::Error for SchemaError {}

pub(crate) fn pointer_key(parent: &str, key: &str) -> String {
    let escaped = key.replace('~', "~0").replace('/', "~1");
    if parent == "$" {
        format!("/{escaped}")
    } else {
        format!("{parent}/{escaped}")
    }
}

pub(crate) fn pointer_index(parent: &str, index: usize) -> String {
    if parent == "$" {
        format!("/{index}")
    } else {
        format!("{parent}/{index}")
    }
}

pub(crate) fn escape_json_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str(r#"\""#),
            '\\' => output.push_str(r#"\\"#),
            '\u{0008}' => output.push_str(r#"\b"#),
            '\t' => output.push_str(r#"\t"#),
            '\n' => output.push_str(r#"\n"#),
            '\u{000C}' => output.push_str(r#"\f"#),
            '\r' => output.push_str(r#"\r"#),
            value if value <= '\u{001F}' => {
                use std::fmt::Write as _;
                let _ = write!(output, "\\u{:04x}", value as u32);
            }
            value => output.push(value),
        }
    }
    output.push('"');
    output
}

struct RawJsonParser<'a> {
    input: &'a str,
    offset: usize,
}

impl<'a> RawJsonParser<'a> {
    const MAX_DEPTH: usize = 128;

    const fn new(input: &'a str) -> Self {
        Self { input, offset: 0 }
    }

    fn parse(mut self) -> Result<JsonValue, SchemaError> {
        self.skip_whitespace();
        let value = self.parse_value(0, "$")?;
        self.skip_whitespace();
        if self.offset != self.input.len() {
            return Err(self.parse_error("trailing bytes after JSON value"));
        }
        Ok(value)
    }

    fn parse_value(&mut self, depth: usize, pointer: &str) -> Result<JsonValue, SchemaError> {
        if depth > Self::MAX_DEPTH {
            return Err(SchemaError::Resource {
                pointer: pointer.to_owned(),
                reason: format!("JSON nesting exceeds {} containers", Self::MAX_DEPTH),
            });
        }
        self.skip_whitespace();
        match self.peek() {
            Some(b'{') => self.parse_object(depth + 1, pointer),
            Some(b'[') => self.parse_array(depth + 1, pointer),
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b't') => self.parse_literal("true", JsonValue::Boolean(true)),
            Some(b'f') => self.parse_literal("false", JsonValue::Boolean(false)),
            Some(b'n') => self.parse_literal("null", JsonValue::Null),
            Some(b'-' | b'0'..=b'9') => self.parse_number(pointer).map(JsonValue::Number),
            Some(_) => Err(self.parse_error("expected a JSON value")),
            None => Err(self.parse_error("unexpected end of JSON input")),
        }
    }

    fn parse_object(&mut self, depth: usize, pointer: &str) -> Result<JsonValue, SchemaError> {
        self.expect_byte(b'{')?;
        self.skip_whitespace();
        let mut entries = BTreeMap::new();
        let mut seen = BTreeSet::new();
        if self.consume_byte(b'}') {
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return Err(self.parse_error("object key must be a JSON string"));
            }
            let key = self.parse_string()?;
            let child_pointer = pointer_key(pointer, &key);
            if !seen.insert(key.clone()) {
                return Err(SchemaError::DuplicateKey {
                    pointer: child_pointer,
                });
            }
            self.skip_whitespace();
            self.expect_byte(b':')?;
            let value = self.parse_value(depth, &child_pointer)?;
            entries.insert(key, value);
            self.skip_whitespace();
            if self.consume_byte(b'}') {
                break;
            }
            self.expect_byte(b',')?;
        }
        Ok(JsonValue::Object(entries))
    }

    fn parse_array(&mut self, depth: usize, pointer: &str) -> Result<JsonValue, SchemaError> {
        self.expect_byte(b'[')?;
        self.skip_whitespace();
        let mut values = Vec::new();
        if self.consume_byte(b']') {
            return Ok(JsonValue::Array(values));
        }
        loop {
            let child_pointer = pointer_index(pointer, values.len());
            values.push(self.parse_value(depth, &child_pointer)?);
            self.skip_whitespace();
            if self.consume_byte(b']') {
                break;
            }
            self.expect_byte(b',')?;
        }
        Ok(JsonValue::Array(values))
    }

    fn parse_literal(
        &mut self,
        expected: &str,
        value: JsonValue,
    ) -> Result<JsonValue, SchemaError> {
        if self.input[self.offset..].starts_with(expected) {
            self.offset += expected.len();
            Ok(value)
        } else {
            Err(self.parse_error("invalid JSON literal"))
        }
    }

    fn parse_number(&mut self, pointer: &str) -> Result<ExactDecimal, SchemaError> {
        let start = self.offset;
        if self.consume_byte(b'-') && self.offset == self.input.len() {
            return Err(self.parse_error("incomplete JSON number"));
        }
        match self.peek() {
            Some(b'0') => {
                self.offset += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.parse_error("JSON numbers may not have leading zeroes"));
                }
            }
            Some(b'1'..=b'9') => {
                self.offset += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.offset += 1;
                }
            }
            _ => return Err(self.parse_error("invalid JSON number")),
        }
        if self.consume_byte(b'.') {
            let fraction_start = self.offset;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.offset += 1;
            }
            if fraction_start == self.offset {
                return Err(self.parse_error("JSON fraction requires digits"));
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
                return Err(self.parse_error("JSON exponent requires digits"));
            }
        }
        ExactDecimal::parse(&self.input[start..self.offset]).map_err(|error| SchemaError::Numeric {
            pointer: pointer.to_owned(),
            reason: error.to_string(),
        })
    }

    fn parse_string(&mut self) -> Result<String, SchemaError> {
        self.expect_byte(b'"')?;
        let mut output = String::new();
        loop {
            let Some(byte) = self.peek() else {
                return Err(self.parse_error("unterminated JSON string"));
            };
            match byte {
                b'"' => {
                    self.offset += 1;
                    return Ok(output);
                }
                b'\\' => {
                    self.offset += 1;
                    let escaped = self
                        .peek()
                        .ok_or_else(|| self.parse_error("incomplete JSON escape"))?;
                    self.offset += 1;
                    match escaped {
                        b'"' => output.push('"'),
                        b'\\' => output.push('\\'),
                        b'/' => output.push('/'),
                        b'b' => output.push('\u{0008}'),
                        b'f' => output.push('\u{000C}'),
                        b'n' => output.push('\n'),
                        b'r' => output.push('\r'),
                        b't' => output.push('\t'),
                        b'u' => output.push(self.parse_unicode_escape()?),
                        _ => return Err(self.parse_error("unsupported JSON escape")),
                    }
                }
                0x00..=0x1f => {
                    return Err(self.parse_error("unescaped control character in JSON string"));
                }
                _ => {
                    let rest = &self.input[self.offset..];
                    let character = rest
                        .chars()
                        .next()
                        .ok_or_else(|| self.parse_error("invalid UTF-8 boundary"))?;
                    output.push(character);
                    self.offset += character.len_utf8();
                }
            }
        }
    }

    fn parse_unicode_escape(&mut self) -> Result<char, SchemaError> {
        let first = self.parse_u16_escape()?;
        if (0xdc00..=0xdfff).contains(&first) {
            return Err(self.parse_error("unpaired low surrogate in JSON string"));
        }
        if !(0xd800..=0xdbff).contains(&first) {
            return char::from_u32(u32::from(first))
                .ok_or_else(|| self.parse_error("invalid unicode escape"));
        }
        if !self.input[self.offset..].starts_with("\\u") {
            return Err(self.parse_error("high surrogate requires a low surrogate"));
        }
        self.offset += 2;
        let second = self.parse_u16_escape()?;
        if !(0xdc00..=0xdfff).contains(&second) {
            return Err(self.parse_error("high surrogate requires a low surrogate"));
        }
        let scalar = 0x10000 + ((u32::from(first) - 0xd800) << 10) + (u32::from(second) - 0xdc00);
        char::from_u32(scalar).ok_or_else(|| self.parse_error("invalid surrogate pair"))
    }

    fn parse_u16_escape(&mut self) -> Result<u16, SchemaError> {
        if self.offset + 4 > self.input.len() {
            return Err(self.parse_error("incomplete unicode escape"));
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
                _ => return Err(self.parse_error("invalid unicode escape digit")),
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

    fn expect_byte(&mut self, expected: u8) -> Result<(), SchemaError> {
        if self.consume_byte(expected) {
            Ok(())
        } else {
            Err(self.parse_error(&format!("expected byte {:?}", char::from(expected))))
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

    fn parse_error(&self, reason: impl Into<String>) -> SchemaError {
        SchemaError::Parse {
            offset: self.offset,
            reason: reason.into(),
        }
    }
}
