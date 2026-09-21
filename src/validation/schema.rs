//! Independent structured-output schema validation.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::grammar::{ScalarValue, SchemaNode, SourceAnnotation};

use super::json::{JsonLocation, JsonParseError, JsonValue, ParsedJson, parse_json_with_locations};
use super::offsets::SourceSpan;
use super::source_membership::{GroundingError, verify_source_membership};

/// The category of a validation failure reported at the product boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidationErrorKind {
    Parse,
    Constraint,
    Grounding,
}

/// A validation failure with safe default formatting and explicit access to
/// private pointer metadata. Property names must not escape through logging.
#[derive(Clone, Eq, PartialEq)]
pub struct ValidationError {
    kind: ValidationErrorKind,
    pointer: String,
    byte_offset: Option<usize>,
    scalar_offset: Option<usize>,
    expected: String,
}

impl ValidationError {
    pub const fn kind(&self) -> ValidationErrorKind {
        self.kind
    }

    /// Explicit diagnostic access. This may contain untrusted property names
    /// or caller-supplied grounding pointers, including terminal controls.
    pub fn pointer(&self) -> &str {
        &self.pointer
    }

    pub const fn byte_offset(&self) -> Option<usize> {
        self.byte_offset
    }

    pub const fn scalar_offset(&self) -> Option<usize> {
        self.scalar_offset
    }

    pub fn expected(&self) -> &str {
        &self.expected
    }

    fn constraint(pointer: &str, expected: impl Into<String>) -> Self {
        Self {
            kind: ValidationErrorKind::Constraint,
            pointer: pointer.to_owned(),
            byte_offset: None,
            scalar_offset: None,
            expected: expected.into(),
        }
    }

    fn grounding(
        pointer: &str,
        byte_offset: Option<usize>,
        scalar_offset: Option<usize>,
        expected: impl Into<String>,
    ) -> Self {
        Self {
            kind: ValidationErrorKind::Grounding,
            pointer: pointer.to_owned(),
            byte_offset,
            scalar_offset,
            expected: expected.into(),
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "validation {:?}", self.kind)?;
        if let Some(byte_offset) = self.byte_offset {
            write!(formatter, " byte {byte_offset}")?;
        }
        if let Some(scalar_offset) = self.scalar_offset {
            write!(formatter, " scalar {scalar_offset}")?;
        }
        write!(formatter, ": {}", self.expected)
    }
}

impl fmt::Debug for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl std::error::Error for ValidationError {}

impl From<JsonParseError> for ValidationError {
    fn from(value: JsonParseError) -> Self {
        Self {
            kind: ValidationErrorKind::Parse,
            pointer: value.pointer().to_owned(),
            byte_offset: Some(value.byte_offset()),
            scalar_offset: None,
            expected: value.reason().to_owned(),
        }
    }
}

/// Grounding evidence for one `x-fnlp-source: verbatim` string node.
#[derive(Clone, Copy, Debug)]
pub struct GroundedValue<'a> {
    pub json_pointer: &'a str,
    pub source: &'a str,
    pub span: SourceSpan,
}

/// Parse and validate a structured response through the independent parser.
pub fn validate_json(schema: &SchemaNode, input: &str) -> Result<(), ValidationError> {
    let document = parse_json_with_locations(input)?;
    validate_document(schema, &document)
}

/// Validate a previously parsed value against only immutable schema data.
pub fn validate_value(schema: &SchemaNode, value: &JsonValue) -> Result<(), ValidationError> {
    validate_node(schema, value, "$")
}

/// Validate a response and prove every source-annotated string against its
/// independently verified source span.
pub fn validate_with_grounding(
    schema: &SchemaNode,
    input: &str,
    groundings: &[GroundedValue<'_>],
) -> Result<(), ValidationError> {
    let document = parse_json_with_locations(input)?;
    validate_document(schema, &document)?;
    let value = &document.value;

    let mut required = BTreeMap::new();
    collect_grounded_strings(schema, &value, "$", &mut required)?;
    let mut supplied = BTreeSet::new();
    for grounding in groundings {
        if !supplied.insert(grounding.json_pointer) {
            return Err(ValidationError::grounding(
                grounding.json_pointer,
                Some(grounding.span.byte_start),
                Some(grounding.span.scalar_start),
                "one grounding record per JSON pointer",
            ));
        }
        let Some(expected_text) = required.get(grounding.json_pointer) else {
            return Err(ValidationError::grounding(
                grounding.json_pointer,
                Some(grounding.span.byte_start),
                Some(grounding.span.scalar_start),
                "grounding record for a source-annotated string",
            ));
        };
        verify_source_membership(grounding.source, expected_text, grounding.span)
            .map_err(|error| grounding_error(grounding.json_pointer, grounding.span, error))?;
    }
    for pointer in required.keys() {
        if !supplied.contains(pointer.as_str()) {
            return Err(ValidationError::grounding(
                pointer,
                None,
                None,
                "source-membership evidence for verbatim output",
            ));
        }
    }
    Ok(())
}

fn validate_document(schema: &SchemaNode, document: &ParsedJson) -> Result<(), ValidationError> {
    validate_value(schema, &document.value)
        .map_err(|error| attach_location(error, &document.locations))
}

fn attach_location(
    mut error: ValidationError,
    locations: &BTreeMap<String, JsonLocation>,
) -> ValidationError {
    if error.byte_offset.is_some() {
        return error;
    }
    let mut pointer = error.pointer.clone();
    loop {
        if let Some(location) = locations.get(&pointer) {
            error.byte_offset = Some(location.byte_offset);
            error.scalar_offset = Some(location.scalar_offset);
            return error;
        }
        if pointer == "$" {
            return error;
        }
        pointer = parent_pointer(&pointer);
    }
}

fn validate_node(
    schema: &SchemaNode,
    value: &JsonValue,
    pointer: &str,
) -> Result<(), ValidationError> {
    match schema {
        SchemaNode::Object {
            properties,
            required,
        } => {
            let JsonValue::Object(object) = value else {
                return Err(type_error(pointer, "object", value));
            };
            for name in required {
                if !object.contains_key(name) {
                    return Err(ValidationError::constraint(
                        &pointer_key(pointer, name),
                        "required property must be present",
                    ));
                }
            }
            for (name, child) in object {
                let child_pointer = pointer_key(pointer, name);
                let Some(child_schema) = properties.get(name) else {
                    return Err(ValidationError::constraint(
                        &child_pointer,
                        "additional properties are forbidden",
                    ));
                };
                validate_node(child_schema, child, &child_pointer)?;
            }
            Ok(())
        }
        SchemaNode::Array { items, max_items } => {
            let JsonValue::Array(values) = value else {
                return Err(type_error(pointer, "array", value));
            };
            if values.len() > *max_items {
                return Err(ValidationError::constraint(
                    pointer,
                    format!("array length must be at most {max_items}"),
                ));
            }
            for (index, child) in values.iter().enumerate() {
                validate_node(items, child, &pointer_index(pointer, index))?;
            }
            Ok(())
        }
        SchemaNode::String {
            max_bytes, allowed, ..
        } => {
            let JsonValue::String(text) = value else {
                return Err(type_error(pointer, "string", value));
            };
            if text.len() > *max_bytes {
                return Err(ValidationError::constraint(
                    pointer,
                    format!("string UTF-8 bytes must be at most {max_bytes}"),
                ));
            }
            ensure_allowed(allowed.as_deref(), value, pointer)
        }
        SchemaNode::Number {
            integer, allowed, ..
        } => {
            let JsonValue::Number(number) = value else {
                return Err(type_error(
                    pointer,
                    if *integer { "integer" } else { "number" },
                    value,
                ));
            };
            if *integer && !number.is_integer_in_64_bit_domain() {
                return Err(ValidationError::constraint(
                    pointer,
                    "exact signed or unsigned 64-bit integer",
                ));
            }
            ensure_allowed(allowed.as_deref(), value, pointer)
        }
        SchemaNode::Boolean { allowed } => {
            if !matches!(value, JsonValue::Boolean(_)) {
                return Err(type_error(pointer, "boolean", value));
            }
            ensure_allowed(allowed.as_deref(), value, pointer)
        }
        SchemaNode::Null { allowed } => {
            if !matches!(value, JsonValue::Null) {
                return Err(type_error(pointer, "null", value));
            }
            ensure_allowed(allowed.as_deref(), value, pointer)
        }
    }
}

fn ensure_allowed(
    allowed: Option<&[ScalarValue]>,
    value: &JsonValue,
    pointer: &str,
) -> Result<(), ValidationError> {
    let Some(allowed) = allowed else {
        return Ok(());
    };
    if allowed
        .iter()
        .any(|candidate| scalar_equal(candidate, value))
    {
        Ok(())
    } else {
        Err(ValidationError::constraint(
            pointer,
            "value equal to one declared enum or const scalar",
        ))
    }
}

fn scalar_equal(schema_value: &ScalarValue, value: &JsonValue) -> bool {
    match (schema_value, value) {
        (ScalarValue::Null, JsonValue::Null) => true,
        (ScalarValue::Boolean(expected), JsonValue::Boolean(actual)) => expected == actual,
        (ScalarValue::String(expected), JsonValue::String(actual)) => expected == actual,
        (ScalarValue::Number(expected), JsonValue::Number(actual)) => {
            expected.canonical_spelling() == actual.canonical_spelling()
        }
        _ => false,
    }
}

fn collect_grounded_strings(
    schema: &SchemaNode,
    value: &JsonValue,
    pointer: &str,
    output: &mut BTreeMap<String, String>,
) -> Result<(), ValidationError> {
    match (schema, value) {
        (SchemaNode::Object { properties, .. }, JsonValue::Object(values)) => {
            for (name, child) in values {
                if let Some(child_schema) = properties.get(name) {
                    let child_pointer = pointer_key(pointer, name);
                    collect_grounded_strings(child_schema, child, &child_pointer, output)?;
                }
            }
        }
        (SchemaNode::Array { items, .. }, JsonValue::Array(values)) => {
            for (index, child) in values.iter().enumerate() {
                collect_grounded_strings(items, child, &pointer_index(pointer, index), output)?;
            }
        }
        (SchemaNode::String { source, .. }, JsonValue::String(text))
            if matches!(source, SourceAnnotation::Verbatim) =>
        {
            output.insert(pointer.to_owned(), text.clone());
        }
        _ => {}
    }
    Ok(())
}

fn grounding_error(pointer: &str, span: SourceSpan, error: GroundingError) -> ValidationError {
    ValidationError::grounding(
        pointer,
        Some(span.byte_start),
        Some(span.scalar_start),
        error.to_string(),
    )
}

fn type_error(pointer: &str, expected: &str, actual: &JsonValue) -> ValidationError {
    ValidationError::constraint(
        pointer,
        format!("{expected}; received {}", actual.kind_name()),
    )
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

fn parent_pointer(pointer: &str) -> String {
    let Some(index) = pointer.rfind('/') else {
        return "$".to_owned();
    };
    if index == 0 {
        "$".to_owned()
    } else {
        pointer[..index].to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::{CompileLimits, compile_json_schema};

    fn assert_private(error: &ValidationError) {
        assert!(error.pointer().contains("PRIVATE"));
        for rendered in [format!("{error}"), format!("{error:?}"), format!("{error:#?}")] {
            assert!(!rendered.contains("PRIVATE"));
            assert!(!rendered.chars().any(char::is_control));
            assert!(rendered.contains(error.expected()));
        }
    }

    #[test]
    fn unexpected_property_errors_keep_locations_but_do_not_log_names() {
        let schema = compile_json_schema(
            r#"{"type":"object","additionalProperties":false,"properties":{}}"#,
            CompileLimits::default(),
        ).unwrap();
        let input = r#"{"PRIVATE\n\u001b[31m/é~":0}"#;
        let error = validate_json(schema.root(), input).unwrap_err();
        let offset = input.rfind('0').unwrap();
        assert_eq!(error.kind(), ValidationErrorKind::Constraint);
        assert_eq!(error.pointer(), "/PRIVATE\n\u{001b}[31m~1é~0");
        assert_eq!(error.byte_offset(), Some(offset));
        assert_eq!(error.scalar_offset(), Some(input[..offset].chars().count()));
        assert_private(&error);
    }

    #[test]
    fn parser_error_conversion_does_not_reintroduce_private_names() {
        let original = super::super::parse_json(r#"{"PRIVATE\n\u001b":tru}"#).unwrap_err();
        let pointer = original.pointer().to_owned();
        let offset = original.byte_offset();
        let error = ValidationError::from(original);
        assert_eq!(error.kind(), ValidationErrorKind::Parse);
        assert_eq!(error.pointer(), pointer);
        assert_eq!(error.byte_offset(), Some(offset));
        assert_private(&error);
    }

    #[test]
    fn caller_supplied_grounding_pointers_are_private_diagnostics_too() {
        let schema = compile_json_schema(
            r#"{"type":"string","x-fnlp-source":"verbatim"}"#,
            CompileLimits::default(),
        ).unwrap();
        let grounding = GroundedValue {
            json_pointer: "/PRIVATE\n\u{001b}[31m",
            source: "PRIVATE_SOURCE",
            span: SourceSpan::new(0, 1, 0, 1),
        };
        let error = validate_with_grounding(schema.root(), r#""a""#, &[grounding]).unwrap_err();
        assert_eq!(error.kind(), ValidationErrorKind::Grounding);
        assert_eq!(error.pointer(), grounding.json_pointer);
        assert_private(&error);
    }
}
