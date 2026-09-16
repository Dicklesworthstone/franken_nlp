//! Schema-aware, exact-value claim rendering. No generated prompts, inferred
//! field meanings, floating-point coercions, or executable template language.
use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};
use crate::{canonjson, grammar::{SchemaNode, SourceAnnotation}, validation::JsonValue};
use super::{SemanticError, SemanticLimits, SemanticNotChecked};

pub const CLAIM_RENDER_VERSION: &str = "extract-schema-scalar-json-insertion-v1";

/// Typed paths distinguish array items from a property literally named '*'.
#[derive(Clone, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case", deny_unknown_fields)]
pub enum ClaimPathStep { Property(String), EachItem }
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimValueKind { String, Integer, Number, Boolean }
/// The single substitution is prefix + canonical JSON scalar + suffix.
/// String quotes and exact decimal spellings are retained in the model claim.
/// Text fields deliberately omit Debug; all enter the judge as untrusted data.
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimRule {
    pub id: String,
    pub path: Vec<ClaimPathStep>,
    pub value_kind: ClaimValueKind,
    pub prefix: String,
    pub suffix: String,
}

pub(super) struct RenderedField {
    pub pointer: String,
    pub rule_id: Option<String>,
    pub claim: Option<String>,
    pub not_checked: Option<SemanticNotChecked>,
}

pub(super) fn render_fields(schema: &SchemaNode, value: &JsonValue, rules: &[ClaimRule], limits: SemanticLimits)
    -> Result<Vec<RenderedField>, SemanticError> {
    if rules.len() > limits.max_fields { return Err(SemanticError::Limit("claim_rules")); }
    let mut by_path = BTreeMap::new(); let mut ids = std::collections::BTreeSet::new();
    let mut template_bytes = 0_usize;
    for rule in rules {
        check_id(&rule.id)?;
        if rule.path.len() > 64 || (rule.prefix.trim().is_empty() && rule.suffix.trim().is_empty()) {
            return Err(SemanticError::Contract("claim rule path or proposition is empty/overdeep"));
        }
        for n in [rule.id.len(), rule.prefix.len(), rule.suffix.len()] {
            template_bytes = template_bytes.checked_add(n).ok_or(SemanticError::Limit("claim_templates"))?;
        }
        let mut node = schema;
        for step in &rule.path {
            node = match (node, step) {
                (SchemaNode::Object { properties, .. }, ClaimPathStep::Property(key)) => {
                    template_bytes = template_bytes.checked_add(key.len()).ok_or(SemanticError::Limit("claim_templates"))?;
                    properties.get(key).ok_or(SemanticError::Contract("claim path absent from schema"))?
                }
                (SchemaNode::Array { items, .. }, ClaimPathStep::EachItem) => items,
                _ => return Err(SemanticError::Contract("claim path kind differs from schema")),
            };
        }
        if template_bytes > limits.max_template_bytes { return Err(SemanticError::Limit("claim_templates")); }
        if is_verbatim(node) || kind(node) != Some(rule.value_kind) {
            return Err(SemanticError::Contract("claim rule requires a matching non-verbatim scalar"));
        }
        if !ids.insert(&rule.id) || by_path.insert(rule.path.clone(), rule).is_some() {
            return Err(SemanticError::Contract("duplicate claim rule identifier or schema path"));
        }
    }
    let mut walk = Walker { rules: by_path, limits, fields: Vec::new(), visited: 0, claim_bytes: 0 };
    walk.walk(schema, value, "$", &mut Vec::new())?;
    Ok(walk.fields)
}
fn kind(node: &SchemaNode) -> Option<ClaimValueKind> {
    match node { SchemaNode::String { .. } => Some(ClaimValueKind::String),
        SchemaNode::Number { integer: true, .. } => Some(ClaimValueKind::Integer),
        SchemaNode::Number { integer: false, .. } => Some(ClaimValueKind::Number),
        SchemaNode::Boolean { .. } => Some(ClaimValueKind::Boolean), _ => None }
}
fn is_verbatim(node: &SchemaNode) -> bool { matches!(node, SchemaNode::String { source: SourceAnnotation::Verbatim, .. }) }
pub(super) fn check_id(id: &str) -> Result<(), SemanticError> {
    if id.is_empty() || id.len() > 64 || !id.bytes().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, b'-' | b'_')) {
        return Err(SemanticError::Contract("invalid semantic revision or claim identifier"));
    }
    Ok(())
}
struct Walker<'a> {
    rules: BTreeMap<Vec<ClaimPathStep>, &'a ClaimRule>,
    limits: SemanticLimits,
    fields: Vec<RenderedField>,
    visited: usize,
    claim_bytes: usize,
}
impl Walker<'_> {
    fn walk(&mut self, schema: &SchemaNode, value: &JsonValue, pointer: &str, path: &mut Vec<ClaimPathStep>) -> Result<(), SemanticError> {
        self.visited = self.visited.checked_add(1).ok_or(SemanticError::Limit("claim_walk"))?;
        if self.visited > self.limits.max_walk_nodes || path.len() > 64 { return Err(SemanticError::Limit("claim_walk")); }
        match (schema, value) {
            (SchemaNode::Object { properties, .. }, JsonValue::Object(values)) if !values.is_empty() => {
                for (key, value) in values {
                    let child = properties.get(key).ok_or(SemanticError::Contract("invalid extraction shape"))?;
                    path.push(ClaimPathStep::Property(key.clone()));
                    let key = key.replace('~', "~0").replace('/', "~1");
                    self.walk(child, value, &child_pointer(pointer, &key), path)?;
                    path.pop();
                }
                return Ok(());
            }
            (SchemaNode::Array { items, .. }, JsonValue::Array(values)) if !values.is_empty() => {
                path.push(ClaimPathStep::EachItem);
                for (index, value) in values.iter().enumerate() { self.walk(items, value, &child_pointer(pointer, &index.to_string()), path)?; }
                path.pop();
                return Ok(());
            }
            _ => {}
        }
        if self.fields.len() == self.limits.max_fields { return Err(SemanticError::Limit("semantic_fields")); }
        let reason = if is_verbatim(schema) { Some(SemanticNotChecked::VerbatimMembershipOnly) }
            else if matches!(value, JsonValue::Null) { Some(SemanticNotChecked::NullValue) }
            else if matches!(value, JsonValue::Array(_) | JsonValue::Object(_)) { Some(SemanticNotChecked::EmptyContainer) }
            else if !self.rules.contains_key(path) { Some(SemanticNotChecked::NoClaimRule) } else { None };
        let (rule_id, claim) = if reason.is_none() {
            let rule = self.rules.get(path).ok_or(SemanticError::Contract("missing compiled claim rule"))?;
            let scalar = scalar_json(value, self.limits.max_claim_bytes)?;
            let len = rule.prefix.len().checked_add(scalar.len()).and_then(|n| n.checked_add(rule.suffix.len()))
                .ok_or(SemanticError::Limit("claim_bytes"))?;
            self.claim_bytes = self.claim_bytes.checked_add(len).ok_or(SemanticError::Limit("claim_bytes"))?;
            if len > self.limits.max_claim_bytes || self.claim_bytes > self.limits.max_total_claim_bytes {
                return Err(SemanticError::Limit("claim_bytes"));
            }
            let mut claim = String::new(); claim.try_reserve_exact(len).map_err(|_| SemanticError::AllocationRefused)?;
            claim.push_str(&rule.prefix); claim.push_str(&scalar); claim.push_str(&rule.suffix);
            (Some(rule.id.clone()), Some(claim))
        } else { (None, None) };
        self.fields.try_reserve(1).map_err(|_| SemanticError::AllocationRefused)?;
        self.fields.push(RenderedField { pointer: pointer.to_owned(), rule_id, claim, not_checked: reason });
        Ok(())
    }
}
fn child_pointer(parent: &str, key: &str) -> String {
    if parent == "$" { format!("/{key}") } else { format!("{parent}/{key}") }
}
fn scalar_json(value: &JsonValue, max_bytes: usize) -> Result<String, SemanticError> {
    let out = match value {
        JsonValue::String(text) => {
            // Conservative preallocation bound for JSON control escaping;
            // UTF-8 remains bytes, never locale/numeric normalization.
            let upper = text.len().checked_mul(6).and_then(|n| n.checked_add(2)).ok_or(SemanticError::Limit("claim_bytes"))?;
            if upper > max_bytes { return Err(SemanticError::Limit("claim_bytes")); }
            canonjson::canonical_string(text).map_err(|_| SemanticError::Serialization)?
        }
        JsonValue::Number(number) => number.canonical_spelling(),
        JsonValue::Boolean(value) => value.to_string(),
        _ => return Err(SemanticError::Contract("semantic claim value is not an admitted scalar")),
    };
    if out.len() > max_bytes { return Err(SemanticError::Limit("claim_bytes")); }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammar::{CompileLimits, compile_json_schema}, validation::parse_json};
    fn rule(path: Vec<ClaimPathStep>, value_kind: ClaimValueKind) -> ClaimRule {
        ClaimRule { id: "field-v1".to_owned(), path, value_kind, prefix: "The amount is ".to_owned(), suffix: ".".to_owned() }
    }
    fn fields(schema: &str, json: &str, rules: &[ClaimRule]) -> Vec<RenderedField> {
        let schema = compile_json_schema(schema, CompileLimits::default()).unwrap();
        render_fields(schema.root(), &parse_json(json).unwrap(), rules, SemanticLimits::default()).unwrap()
    }
    #[test]
    fn thirty_eight_digit_values_never_round_through_f64() {
        let out = fields(r#"{"type":"number"}"#, "12345678901234567890123456789012345678", &[rule(vec![], ClaimValueKind::Number)]);
        assert_eq!(out[0].claim.as_deref(), Some("The amount is 1.2345678901234567890123456789012345678e37."));
    }
    #[test]
    fn nested_arrays_and_literal_star_properties_have_distinct_typed_paths() {
        let schema = r#"{"type":"object","additionalProperties":false,"properties":{"a/b~":{"type":"array","maxItems":2,"items":{"type":"number"}},"*":{"type":"boolean"}}}"#;
        let rules = [rule(vec![ClaimPathStep::Property("a/b~".to_owned()), ClaimPathStep::EachItem], ClaimValueKind::Number)];
        let out = fields(schema, r#"{"a/b~":[1,2],"*":true}"#, &rules);
        assert_eq!(out[0].pointer, "/*"); assert_eq!(out[0].not_checked, Some(SemanticNotChecked::NoClaimRule));
        assert_eq!(out[1].pointer, "/a~1b~0/0"); assert_eq!(out[2].pointer, "/a~1b~0/1");
        assert_eq!(out[2].claim.as_deref(), Some("The amount is 2."));
    }
    #[test]
    fn source_membership_is_never_promoted_to_semantic_support() {
        let out = fields(r#"{"type":"string","x-fnlp-source":"verbatim"}"#, r#""Alice""#, &[]);
        assert_eq!(out[0].not_checked, Some(SemanticNotChecked::VerbatimMembershipOnly)); assert!(out[0].claim.is_none());
    }
    #[test]
    fn unknown_paths_wrong_types_and_duplicate_rules_fail_before_judging() {
        let schema = compile_json_schema(r#"{"type":"number"}"#, CompileLimits::default()).unwrap();
        let value = parse_json("1").unwrap(); let rule = rule(vec![], ClaimValueKind::String);
        assert!(render_fields(schema.root(), &value, &[rule], SemanticLimits::default()).is_err());
        let rule = super::tests::rule(vec![], ClaimValueKind::Number);
        assert!(render_fields(schema.root(), &value, &[rule.clone(), rule], SemanticLimits::default()).is_err());
    }
    #[test]
    fn null_empty_containers_and_unmapped_values_are_explicitly_unchecked() {
        assert_eq!(fields(r#"{"type":"null"}"#, "null", &[])[0].not_checked, Some(SemanticNotChecked::NullValue));
        assert_eq!(fields(r#"{"type":"array","maxItems":1,"items":{"type":"number"}}"#, "[]", &[])[0].not_checked, Some(SemanticNotChecked::EmptyContainer));
    }
    #[test]
    fn exceeding_field_budget_refuses_the_entire_walk() {
        let schema = compile_json_schema(r#"{"type":"array","maxItems":2,"items":{"type":"number"}}"#, CompileLimits::default()).unwrap();
        let value = parse_json("[1,2]").unwrap();
        let limits = SemanticLimits { max_fields: 1, ..SemanticLimits::default() };
        assert!(render_fields(schema.root(), &value, &[], limits).is_err());
    }
}
