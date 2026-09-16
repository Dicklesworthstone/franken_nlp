//! Independent source membership and occurrence recovery for structured output.
//!
//! This module consumes only immutable schema data, the validator's parsed JSON,
//! and original UTF-8 text. It never uses a grammar cursor, suffix index, mask,
//! or acceptance flag. KMP finds overlapping matches; byte/scalar coordinates
//! come from a separate scan. Limits fail the whole result, never truncate it.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::grammar::{SchemaNode, SourceAnnotation};
use super::JsonValue;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GroundingBudget {
    pub max_fields: usize,
    pub max_matches: usize,
    pub max_scan_steps: u64,
}
impl Default for GroundingBudget {
    fn default() -> Self {
        Self { max_fields: 4096, max_matches: 16_384, max_scan_steps: 64 * 1024 * 1024 }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerifiedSourceSpan {
    pub byte_start: usize,
    pub byte_end: usize,
    pub scalar_start: usize,
    pub scalar_end: usize,
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceOccurrence { Anchored, Ambiguous }

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceFieldEvidence {
    pub json_pointer: String,
    pub occurrence: SourceOccurrence,
    pub spans: Vec<VerifiedSourceSpan>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldGroundingError { Shape, Absent, FieldBudget, MatchBudget, WorkBudget, AllocationRefused }
impl fmt::Display for FieldGroundingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Shape => "source verification requires a schema-valid value",
            Self::Absent => "verbatim field is absent from the original source",
            Self::FieldBudget => "source field count exceeds verification budget",
            Self::MatchBudget => "source occurrence count exceeds verification budget",
            Self::WorkBudget => "source verification exceeds scan work budget",
            Self::AllocationRefused => "source verification allocation refused",
        })
    }
}
impl Error for FieldGroundingError {}

/// Independently check every present source annotation, including annotations
/// below optional object keys and arrays. Returned JSON pointers locate values,
/// not schema nodes. Absence of an optional field requires no invented span.
pub fn verify_source_fields(
    schema: &SchemaNode, value: &JsonValue, source: &str, budget: GroundingBudget,
) -> Result<Vec<SourceFieldEvidence>, FieldGroundingError> {
    let mut state = Verification { source, budget, fields: Vec::new() };
    state.walk(schema, value, "$", 0)?;
    Ok(state.fields)
}

struct Verification<'a> {
    source: &'a str,
    budget: GroundingBudget,
    fields: Vec<SourceFieldEvidence>,
}
impl Verification<'_> {
    fn walk(&mut self, schema: &SchemaNode, value: &JsonValue, path: &str, depth: usize) -> Result<(), FieldGroundingError> {
        if depth > 128 { return Err(FieldGroundingError::Shape); }
        match (schema, value) {
            (SchemaNode::String { source: SourceAnnotation::Verbatim, .. }, JsonValue::String(text)) => {
                if self.fields.len() == self.budget.max_fields { return Err(FieldGroundingError::FieldBudget); }
                let spans = scan_occurrences(self.source, text, &mut self.budget)?;
                let occurrence = if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous };
                self.fields.try_reserve(1).map_err(|_| FieldGroundingError::AllocationRefused)?;
                self.fields.push(SourceFieldEvidence { json_pointer: path.to_owned(), occurrence, spans });
            }
            (SchemaNode::Object { properties, .. }, JsonValue::Object(values)) => {
                for (name, value) in values {
                    let child = properties.get(name).ok_or(FieldGroundingError::Shape)?;
                    self.walk(child, value, &pointer(path, &name.replace('~', "~0").replace('/', "~1")), depth + 1)?;
                }
            }
            (SchemaNode::Array { items, .. }, JsonValue::Array(values)) => {
                for (i, value) in values.iter().enumerate() { self.walk(items, value, &pointer(path, &i.to_string()), depth + 1)?; }
            }
            (SchemaNode::String { .. }, JsonValue::String(_))
            | (SchemaNode::Number { .. }, JsonValue::Number(_))
            | (SchemaNode::Boolean { .. }, JsonValue::Boolean(_))
            | (SchemaNode::Null { .. }, JsonValue::Null) => {}
            _ => return Err(FieldGroundingError::Shape),
        }
        Ok(())
    }
}
fn pointer(parent: &str, key: &str) -> String {
    if parent == "$" { format!("/{key}") } else { format!("{parent}/{key}") }
}

/// Independent exact matching, also useful for NER mention finalization.
/// Consumes aggregate budgets shared by all fields in a response.
pub fn scan_occurrences(source: &str, text: &str, budget: &mut GroundingBudget) -> Result<Vec<VerifiedSourceSpan>, FieldGroundingError> {
    let cost = (source.len() as u64).checked_add(text.len() as u64)
        .and_then(|n| n.checked_add(1)).and_then(|n| n.checked_mul(8)).ok_or(FieldGroundingError::WorkBudget)?;
    budget.max_scan_steps = budget.max_scan_steps.checked_sub(cost).ok_or(FieldGroundingError::WorkBudget)?;
    let mut output = Vec::new();
    let mut push = |span| -> Result<(), FieldGroundingError> {
        if output.len() == budget.max_matches { return Err(FieldGroundingError::MatchBudget); }
        output.try_reserve(1).map_err(|_| FieldGroundingError::AllocationRefused)?;
        output.push(span); Ok(())
    };
    if text.is_empty() {
        for (scalar, byte) in source.char_indices().map(|(i, _)| i).chain(std::iter::once(source.len())).enumerate() {
            push(VerifiedSourceSpan { byte_start: byte, byte_end: byte, scalar_start: scalar, scalar_end: scalar })?;
        }
    } else {
        let needle = text.as_bytes();
        let mut prefix = Vec::new();
        prefix.try_reserve_exact(needle.len()).map_err(|_| FieldGroundingError::AllocationRefused)?;
        prefix.resize(needle.len(), 0_usize);
        let mut matched = 0;
        for i in 1..needle.len() {
            while matched > 0 && needle[i] != needle[matched] { matched = prefix[matched - 1]; }
            if needle[i] == needle[matched] { matched += 1; }
            prefix[i] = matched;
        }
        let text_scalars = text.chars().count();
        let mut scalars = 0_usize;
        matched = 0;
        for (i, byte) in source.bytes().enumerate() {
            if byte & 0xc0 != 0x80 { scalars += 1; }
            while matched > 0 && byte != needle[matched] { matched = prefix[matched - 1]; }
            if byte == needle[matched] { matched += 1; }
            if matched == needle.len() {
                let end = i + 1; let start = end - needle.len();
                if source.get(start..end) != Some(text) { return Err(FieldGroundingError::Shape); }
                push(VerifiedSourceSpan { byte_start: start, byte_end: end,
                    scalar_start: scalars.checked_sub(text_scalars).ok_or(FieldGroundingError::Shape)?, scalar_end: scalars })?;
                matched = prefix[matched - 1]; // retain overlapping matches
            }
        }
    }
    drop(push);
    if output.is_empty() { return Err(FieldGroundingError::Absent); }
    budget.max_matches -= output.len();
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammar::{compile_json_schema, CompileLimits}, validation::{parse_json, validate_source_span, SourceSpan}};
    #[test]
    fn overlapping_occurrences_and_unicode_offsets_are_verified_independently() {
        let source = "éaaaa上海上海";
        for text in ["aa", "上海", "é", "", "aaaa上海"] {
            let spans = scan_occurrences(source, text, &mut GroundingBudget::default()).unwrap();
            for s in &spans {
                validate_source_span(source, text, SourceSpan::new(s.byte_start, s.byte_end, s.scalar_start, s.scalar_end)).unwrap();
            }
            let expected = source.char_indices().map(|(i, _)| i).chain(std::iter::once(source.len()))
                .filter(|&i| source[i..].starts_with(text)).count();
            assert_eq!(spans.len(), expected);
        }
    }
    #[test]
    fn nested_arrays_and_escaped_property_names_have_value_pointers() {
        let schema = compile_json_schema(r#"{"type":"object","additionalProperties":false,"properties":{"a/b":{"type":"array","items":{"type":"string","x-fnlp-source":"verbatim"},"maxItems":2}}}"#, CompileLimits::default()).unwrap();
        let v = parse_json(r#"{"a/b":["Alice","Bob"]}"#).unwrap();
        let e = verify_source_fields(schema.root(), &v, "Alice Bob Alice", GroundingBudget::default()).unwrap();
        assert_eq!(e[0].json_pointer, "/a~1b/0"); assert_eq!(e[0].occurrence, SourceOccurrence::Ambiguous);
        assert_eq!(e[1].occurrence, SourceOccurrence::Anchored);
    }
    #[test]
    fn aggregate_matches_cannot_be_reset_between_fields() {
        let schema = compile_json_schema(r#"{"type":"array","items":{"type":"string","x-fnlp-source":"verbatim"},"maxItems":2}"#, CompileLimits::default()).unwrap();
        let v = parse_json(r#"["a","a"]"#).unwrap();
        assert_eq!(verify_source_fields(schema.root(), &v, "a a", GroundingBudget { max_matches: 3, ..GroundingBudget::default() }), Err(FieldGroundingError::MatchBudget));
    }
    #[test]
    fn invented_or_normalized_strings_are_rejected() {
        for text in ["é", "Alice"] {
            assert_eq!(scan_occurrences("e\u{301} alice", text, &mut GroundingBudget::default()), Err(FieldGroundingError::Absent));
        }
    }
    #[test]
    fn work_limit_refuses_before_scan() {
        assert_eq!(scan_occurrences("secret", "secret", &mut GroundingBudget { max_scan_steps: 0, ..GroundingBudget::default() }), Err(FieldGroundingError::WorkBudget));
    }
    #[test]
    fn empty_source_has_one_empty_boundary() {
        assert_eq!(scan_occurrences("", "", &mut GroundingBudget::default()).unwrap().len(), 1);
        assert_eq!(scan_occurrences("", "x", &mut GroundingBudget::default()), Err(FieldGroundingError::Absent));
    }
}
