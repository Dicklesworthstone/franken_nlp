//! Bounded compiler for the normative v1 JSON-Schema subset.
//!
//! This compiler produces a typed JSON automaton plan, not a general JSON
//! Schema implementation.  Unsupported keywords reject at the schema
//! boundary, before a model or tokenizer is loaded.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::schema::{
    IntegerValue, JsonValue, ScalarValue, SchemaError, escape_json_string, parse_json,
    pointer_index, pointer_key,
};

/// Nanbeige's vocabulary mask footprint: 166,144 legal bits / 8.
pub const MASK_BYTES_PER_STATE: u64 = 20_768;

/// Checked per-request grammar resource limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompileLimits {
    /// Maximum UTF-8 bytes accepted for the schema source itself.
    pub max_schema_bytes: usize,
    /// Engine cap for an individual generated string's UTF-8 bytes.
    pub max_string_bytes: usize,
    /// Engine cap used when an array omits `maxItems`.
    pub max_array_items: usize,
    /// Maximum generated output bytes, including JSON punctuation.
    pub max_output_bytes: usize,
    /// Maximum logical automaton states before automaton allocation.
    pub max_states: usize,
    /// Maximum logical automaton transitions before automaton allocation.
    pub max_transitions: usize,
    /// Maximum cached legal-mask bytes before mask/cache allocation.
    pub max_mask_bytes: usize,
}

impl Default for CompileLimits {
    fn default() -> Self {
        Self {
            max_schema_bytes: 256 * 1024,
            max_string_bytes: 64 * 1024,
            max_array_items: 256,
            max_output_bytes: 256 * 1024,
            max_states: 4_096,
            max_transitions: 16_384,
            max_mask_bytes: 64 * 1024 * 1024,
        }
    }
}

/// The source-grounding annotation understood by v1 string schemas.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceAnnotation {
    None,
    /// This marks a source product requirement.  It does not claim the
    /// OQ-17-gated product is currently enabled; execution checks that gate.
    Verbatim,
}

/// A type-preserving v1 schema node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchemaNode {
    Object {
        properties: BTreeMap<String, SchemaNode>,
        required: BTreeSet<String>,
    },
    Array {
        items: Box<SchemaNode>,
        max_items: usize,
    },
    String {
        max_bytes: usize,
        allowed: Option<Vec<ScalarValue>>,
        source: SourceAnnotation,
    },
    Number {
        integer: bool,
        allowed: Option<Vec<ScalarValue>>,
    },
    Boolean {
        allowed: Option<Vec<ScalarValue>>,
    },
    Null {
        allowed: Option<Vec<ScalarValue>>,
    },
}

impl SchemaNode {
    const fn kind_name(&self) -> &'static str {
        match self {
            Self::Object { .. } => "object",
            Self::Array { .. } => "array",
            Self::String { .. } => "string",
            Self::Number { integer: true, .. } => "integer",
            Self::Number { integer: false, .. } => "number",
            Self::Boolean { .. } => "boolean",
            Self::Null { .. } => "null",
        }
    }

    fn allowed(&self) -> Option<&[ScalarValue]> {
        match self {
            Self::String { allowed, .. }
            | Self::Number { allowed, .. }
            | Self::Boolean { allowed }
            | Self::Null { allowed } => allowed.as_deref(),
            Self::Object { .. } | Self::Array { .. } => None,
        }
    }

    fn requires_verbatim_source(&self) -> bool {
        match self {
            Self::Object { properties, .. } => {
                properties.values().any(Self::requires_verbatim_source)
            }
            Self::Array { items, .. } => items.requires_verbatim_source(),
            Self::String { source, .. } => *source == SourceAnnotation::Verbatim,
            Self::Number { .. } | Self::Boolean { .. } | Self::Null { .. } => false,
        }
    }
}

/// Preflight resource report retained with every compiled schema.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutomatonEstimate {
    pub state_count: usize,
    pub transition_count: usize,
    pub mask_cache_bytes: usize,
    pub enum_trie_nodes: usize,
    pub number_lexers: usize,
    pub minimum_output_bytes: usize,
}

/// A logical typed-JSON automaton, intentionally independent of tokenization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypedJsonAutomaton {
    states: Vec<AutomatonState>,
    transitions: Vec<AutomatonTransition>,
    enum_tries: Vec<EnumTrie>,
    number_lexers: Vec<CanonicalNumberLexer>,
}

impl TypedJsonAutomaton {
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    pub fn enum_trie_count(&self) -> usize {
        self.enum_tries.len()
    }

    pub fn number_lexer_count(&self) -> usize {
        self.number_lexers.len()
    }

    /// Verify the no-dead-state condition on the logical graph.
    pub fn every_state_reaches_acceptance(&self) -> bool {
        let mut reverse = vec![Vec::new(); self.states.len()];
        for transition in &self.transitions {
            if let Some(edges) = reverse.get_mut(transition.to) {
                edges.push(transition.from);
            } else {
                return false;
            }
        }
        let mut reaches_acceptance = vec![false; self.states.len()];
        let mut pending = VecDeque::new();
        for state in &self.states {
            if state.accepting {
                reaches_acceptance[state.id] = true;
                pending.push_back(state.id);
            }
        }
        while let Some(state) = pending.pop_front() {
            for predecessor in &reverse[state] {
                if !reaches_acceptance[*predecessor] {
                    reaches_acceptance[*predecessor] = true;
                    pending.push_back(*predecessor);
                }
            }
        }
        reaches_acceptance.into_iter().all(|reachable| reachable)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AutomatonState {
    id: usize,
    accepting: bool,
    minimum_remaining_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AutomatonTransition {
    from: usize,
    to: usize,
    label: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct EnumTrie {
    canonical_scalars: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CanonicalNumberLexer {
    integer_only: bool,
    maximum_significant_digits: usize,
    minimum_adjusted_exponent: i32,
    maximum_adjusted_exponent: i32,
}

/// A compiled schema and its model-free validation/sample API.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledSchema {
    root: SchemaNode,
    automaton: TypedJsonAutomaton,
    estimate: AutomatonEstimate,
    limits: CompileLimits,
}

impl CompiledSchema {
    pub fn root(&self) -> &SchemaNode {
        &self.root
    }

    pub fn automaton(&self) -> &TypedJsonAutomaton {
        &self.automaton
    }

    pub const fn estimate(&self) -> AutomatonEstimate {
        self.estimate
    }

    /// Whether execution must consult the separately gated source product.
    pub fn requires_verbatim_source(&self) -> bool {
        self.root.requires_verbatim_source()
    }

    /// Validate an instance using exact decimal equality and v1 bounds.
    pub fn validate_json(&self, input: &str) -> Result<(), SchemaError> {
        let value = parse_json(input)?;
        validate_node(&self.root, &value, "$")
    }

    /// Emit one canonical valid instance without loading model weights.
    pub fn sample_json(&self) -> Result<String, SchemaError> {
        let output = sample_node(&self.root, "$")?;
        if output.len() > self.limits.max_output_bytes {
            return Err(SchemaError::Resource {
                pointer: "$".to_owned(),
                reason: format!(
                    "sample is {} bytes and exceeds output cap {}",
                    output.len(),
                    self.limits.max_output_bytes
                ),
            });
        }
        self.validate_json(&output)?;
        Ok(output)
    }
}

/// Compile one user schema into the bounded v1 typed-JSON automaton.
pub fn compile_json_schema(
    input: &str,
    limits: CompileLimits,
) -> Result<CompiledSchema, SchemaError> {
    if input.len() > limits.max_schema_bytes {
        return Err(SchemaError::Resource {
            pointer: "$".to_owned(),
            reason: format!(
                "schema is {} bytes and exceeds schema cap {}",
                input.len(),
                limits.max_schema_bytes
            ),
        });
    }
    let raw = parse_json(input)?;
    let root = compile_node(&raw, "$", limits)?;
    let estimate = estimate_node(&root, "$", limits)?;
    preflight(estimate, limits)?;
    let automaton = build_automaton(&root, estimate)?;
    if !automaton.every_state_reaches_acceptance() {
        return Err(SchemaError::InvalidSchema {
            pointer: "$".to_owned(),
            reason: "compiler produced a reachable state without an acceptance path".to_owned(),
        });
    }
    Ok(CompiledSchema {
        root,
        automaton,
        estimate,
        limits,
    })
}

const SUPPORTED_KEYWORDS: &[&str] = &[
    "type",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "maxItems",
    "maxLength",
    "x-fnlp-source",
];

fn compile_node(
    value: &JsonValue,
    pointer: &str,
    limits: CompileLimits,
) -> Result<SchemaNode, SchemaError> {
    let object = value
        .as_object()
        .ok_or_else(|| SchemaError::InvalidSchema {
            pointer: pointer.to_owned(),
            reason: "schema node must be an object".to_owned(),
        })?;
    for key in object.keys() {
        if !SUPPORTED_KEYWORDS.contains(&key.as_str()) {
            return Err(SchemaError::UnsupportedKeyword {
                pointer: pointer_key(pointer, key),
                keyword: key.clone(),
            });
        }
    }
    let type_value = object
        .get("type")
        .ok_or_else(|| SchemaError::InvalidSchema {
            pointer: pointer.to_owned(),
            reason: "every v1 schema node requires a scalar type keyword".to_owned(),
        })?;
    let schema_type = parse_schema_type(type_value, &pointer_key(pointer, "type"))?;
    let allowed = parse_scalar_constraint(object, schema_type, pointer)?;

    match schema_type {
        SchemaType::Object => {
            reject_irrelevant(
                object,
                pointer,
                &["type", "properties", "required", "additionalProperties"],
            )?;
            let additional_properties = object
                .get("additionalProperties")
                .and_then(JsonValue::as_boolean)
                .ok_or_else(|| SchemaError::InvalidSchema {
                    pointer: pointer_key(pointer, "additionalProperties"),
                    reason: "object schemas require additionalProperties:false".to_owned(),
                })?;
            if additional_properties {
                return Err(SchemaError::UnsupportedKeyword {
                    pointer: pointer_key(pointer, "additionalProperties"),
                    keyword: "additionalProperties:true".to_owned(),
                });
            }
            let mut properties = BTreeMap::new();
            if let Some(value) = object.get("properties") {
                let property_map = value
                    .as_object()
                    .ok_or_else(|| SchemaError::InvalidSchema {
                        pointer: pointer_key(pointer, "properties"),
                        reason: "properties must be an object".to_owned(),
                    })?;
                for (name, child) in property_map {
                    properties.insert(
                        name.clone(),
                        compile_node(
                            child,
                            &pointer_key(&pointer_key(pointer, "properties"), name),
                            limits,
                        )?,
                    );
                }
            }
            let required = parse_required(object.get("required"), pointer)?;
            for name in &required {
                if !properties.contains_key(name) {
                    return Err(SchemaError::InvalidSchema {
                        pointer: pointer_key(pointer, "required"),
                        reason: format!("required property {name:?} is not declared in properties"),
                    });
                }
            }
            if allowed.is_some() {
                return Err(SchemaError::InvalidSchema {
                    pointer: pointer.to_owned(),
                    reason: "enum and const only accept JSON scalars in v1".to_owned(),
                });
            }
            Ok(SchemaNode::Object {
                properties,
                required,
            })
        }
        SchemaType::Array => {
            reject_irrelevant(object, pointer, &["type", "items", "maxItems"])?;
            let item_value = object
                .get("items")
                .ok_or_else(|| SchemaError::InvalidSchema {
                    pointer: pointer.to_owned(),
                    reason: "array schemas require items".to_owned(),
                })?;
            let items = compile_node(item_value, &pointer_key(pointer, "items"), limits)?;
            let max_items = match object.get("maxItems") {
                Some(value) => bounded_count(value, &pointer_key(pointer, "maxItems"), "maxItems")?
                    .min(limits.max_array_items),
                None => limits.max_array_items,
            };
            if allowed.is_some() {
                return Err(SchemaError::InvalidSchema {
                    pointer: pointer.to_owned(),
                    reason: "enum and const only accept JSON scalars in v1".to_owned(),
                });
            }
            Ok(SchemaNode::Array {
                items: Box::new(items),
                max_items,
            })
        }
        SchemaType::String => {
            reject_irrelevant(
                object,
                pointer,
                &["type", "enum", "const", "maxLength", "x-fnlp-source"],
            )?;
            let max_bytes = match object.get("maxLength") {
                Some(value) => {
                    bounded_count(value, &pointer_key(pointer, "maxLength"), "maxLength")?
                        .min(limits.max_string_bytes)
                }
                None => limits.max_string_bytes,
            };
            let source = match object.get("x-fnlp-source") {
                None => SourceAnnotation::None,
                Some(JsonValue::String(value)) if value == "verbatim" => SourceAnnotation::Verbatim,
                Some(JsonValue::String(value)) => {
                    return Err(SchemaError::UnsupportedKeyword {
                        pointer: pointer_key(pointer, "x-fnlp-source"),
                        keyword: format!("x-fnlp-source:{value}"),
                    });
                }
                Some(_) => {
                    return Err(SchemaError::InvalidSchema {
                        pointer: pointer_key(pointer, "x-fnlp-source"),
                        reason: "x-fnlp-source must be the string verbatim".to_owned(),
                    });
                }
            };
            if let Some(values) = &allowed {
                for value in values {
                    let ScalarValue::String(value) = value else {
                        return Err(SchemaError::InvalidSchema {
                            pointer: pointer.to_owned(),
                            reason: "string enum/const contains a non-string scalar".to_owned(),
                        });
                    };
                    if value.len() > max_bytes {
                        return Err(SchemaError::Resource {
                            pointer: pointer.to_owned(),
                            reason: format!(
                                "string enum/const value is {} bytes and exceeds effective maxLength {max_bytes}",
                                value.len()
                            ),
                        });
                    }
                }
            }
            Ok(SchemaNode::String {
                max_bytes,
                allowed,
                source,
            })
        }
        SchemaType::Number | SchemaType::Integer => {
            reject_irrelevant(object, pointer, &["type", "enum", "const"])?;
            Ok(SchemaNode::Number {
                integer: schema_type == SchemaType::Integer,
                allowed,
            })
        }
        SchemaType::Boolean => {
            reject_irrelevant(object, pointer, &["type", "enum", "const"])?;
            Ok(SchemaNode::Boolean { allowed })
        }
        SchemaType::Null => {
            reject_irrelevant(object, pointer, &["type", "enum", "const"])?;
            Ok(SchemaNode::Null { allowed })
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SchemaType {
    Object,
    Array,
    String,
    Number,
    Integer,
    Boolean,
    Null,
}

fn parse_schema_type(value: &JsonValue, pointer: &str) -> Result<SchemaType, SchemaError> {
    match value.as_string() {
        Some("object") => Ok(SchemaType::Object),
        Some("array") => Ok(SchemaType::Array),
        Some("string") => Ok(SchemaType::String),
        Some("number") => Ok(SchemaType::Number),
        Some("integer") => Ok(SchemaType::Integer),
        Some("boolean") => Ok(SchemaType::Boolean),
        Some("null") => Ok(SchemaType::Null),
        Some(value) => Err(SchemaError::InvalidSchema {
            pointer: pointer.to_owned(),
            reason: format!("unsupported v1 type {value:?}"),
        }),
        None => Err(SchemaError::InvalidSchema {
            pointer: pointer.to_owned(),
            reason: "type must be one v1 type string, never a union".to_owned(),
        }),
    }
}

fn parse_scalar_constraint(
    object: &BTreeMap<String, JsonValue>,
    schema_type: SchemaType,
    pointer: &str,
) -> Result<Option<Vec<ScalarValue>>, SchemaError> {
    if object.contains_key("enum") && object.contains_key("const") {
        return Err(SchemaError::InvalidSchema {
            pointer: pointer.to_owned(),
            reason: "enum and const cannot be combined in v1".to_owned(),
        });
    }
    let mut values = if let Some(value) = object.get("const") {
        vec![scalar_at(value, &pointer_key(pointer, "const"))?]
    } else if let Some(value) = object.get("enum") {
        let array = value.as_array().ok_or_else(|| SchemaError::InvalidSchema {
            pointer: pointer_key(pointer, "enum"),
            reason: "enum must be a nonempty array of JSON scalars".to_owned(),
        })?;
        if array.is_empty() {
            return Err(SchemaError::InvalidSchema {
                pointer: pointer_key(pointer, "enum"),
                reason: "empty enum has no path to acceptance".to_owned(),
            });
        }
        let mut values = Vec::with_capacity(array.len());
        let mut seen = BTreeSet::new();
        for (index, value) in array.iter().enumerate() {
            let value_pointer = pointer_index(&pointer_key(pointer, "enum"), index);
            let scalar = scalar_at(value, &value_pointer)?;
            if !seen.insert(scalar.clone()) {
                return Err(SchemaError::InvalidSchema {
                    pointer: value_pointer,
                    reason: "enum contains a duplicate mathematically equal scalar".to_owned(),
                });
            }
            values.push(scalar);
        }
        values
    } else {
        return Ok(None);
    };

    for value in &values {
        if !scalar_matches_type(value, schema_type) {
            return Err(SchemaError::InvalidSchema {
                pointer: pointer.to_owned(),
                reason: format!(
                    "scalar {} is incompatible with declared type {}",
                    value.type_name(),
                    schema_type_name(schema_type)
                ),
            });
        }
    }
    values.sort();
    Ok(Some(values))
}

fn scalar_at(value: &JsonValue, pointer: &str) -> Result<ScalarValue, SchemaError> {
    ScalarValue::from_json(value).ok_or_else(|| SchemaError::InvalidSchema {
        pointer: pointer.to_owned(),
        reason: "enum and const only accept JSON scalars in v1".to_owned(),
    })
}

fn scalar_matches_type(value: &ScalarValue, schema_type: SchemaType) -> bool {
    match (schema_type, value) {
        (SchemaType::String, ScalarValue::String(_))
        | (SchemaType::Number, ScalarValue::Number(_))
        | (SchemaType::Boolean, ScalarValue::Boolean(_))
        | (SchemaType::Null, ScalarValue::Null) => true,
        (SchemaType::Integer, ScalarValue::Number(value)) => value.integer_value().is_some(),
        (SchemaType::Object, _) | (SchemaType::Array, _) => false,
        _ => false,
    }
}

const fn schema_type_name(value: SchemaType) -> &'static str {
    match value {
        SchemaType::Object => "object",
        SchemaType::Array => "array",
        SchemaType::String => "string",
        SchemaType::Number => "number",
        SchemaType::Integer => "integer",
        SchemaType::Boolean => "boolean",
        SchemaType::Null => "null",
    }
}

fn reject_irrelevant(
    object: &BTreeMap<String, JsonValue>,
    pointer: &str,
    accepted: &[&str],
) -> Result<(), SchemaError> {
    for key in object.keys() {
        if !accepted.contains(&key.as_str()) {
            return Err(SchemaError::InvalidSchema {
                pointer: pointer_key(pointer, key),
                reason: format!("keyword {key:?} is invalid for this declared type"),
            });
        }
    }
    Ok(())
}

fn parse_required(
    value: Option<&JsonValue>,
    pointer: &str,
) -> Result<BTreeSet<String>, SchemaError> {
    let Some(value) = value else {
        return Ok(BTreeSet::new());
    };
    let array = value.as_array().ok_or_else(|| SchemaError::InvalidSchema {
        pointer: pointer_key(pointer, "required"),
        reason: "required must be an array of property names".to_owned(),
    })?;
    let mut required = BTreeSet::new();
    for (index, value) in array.iter().enumerate() {
        let name = value
            .as_string()
            .ok_or_else(|| SchemaError::InvalidSchema {
                pointer: pointer_index(&pointer_key(pointer, "required"), index),
                reason: "required entries must be strings".to_owned(),
            })?;
        if !required.insert(name.to_owned()) {
            return Err(SchemaError::InvalidSchema {
                pointer: pointer_index(&pointer_key(pointer, "required"), index),
                reason: "required contains a duplicate property name".to_owned(),
            });
        }
    }
    Ok(required)
}

fn bounded_count(value: &JsonValue, pointer: &str, keyword: &str) -> Result<usize, SchemaError> {
    let decimal = value
        .as_number()
        .ok_or_else(|| SchemaError::InvalidSchema {
            pointer: pointer.to_owned(),
            reason: format!("{keyword} must be a nonnegative integer"),
        })?;
    let integer = decimal
        .integer_value()
        .ok_or_else(|| SchemaError::Numeric {
            pointer: pointer.to_owned(),
            reason: format!("{keyword} must be an exact signed/unsigned 64-bit integer"),
        })?;
    let value = match integer {
        IntegerValue::Signed(value) if value >= 0 => value as u64,
        IntegerValue::Unsigned(value) => value,
        IntegerValue::Signed(_) => {
            return Err(SchemaError::InvalidSchema {
                pointer: pointer.to_owned(),
                reason: format!("{keyword} must not be negative"),
            });
        }
    };
    usize::try_from(value).map_err(|_| SchemaError::Resource {
        pointer: pointer.to_owned(),
        reason: format!("{keyword} does not fit this platform's checked bound"),
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct RawEstimate {
    states: u64,
    transitions: u64,
    enum_trie_nodes: u64,
    number_lexers: u64,
}

fn estimate_node(
    node: &SchemaNode,
    pointer: &str,
    limits: CompileLimits,
) -> Result<AutomatonEstimate, SchemaError> {
    let mut raw = RawEstimate::default();
    let minimum_output_bytes = estimate_node_inner(node, pointer, &mut raw)?;
    let mask_cache_bytes = raw
        .states
        .checked_mul(MASK_BYTES_PER_STATE)
        .ok_or_else(|| resource_overflow(pointer, "state × mask-byte estimate"))?;
    let estimate = AutomatonEstimate {
        state_count: checked_usize(raw.states, pointer, "state estimate")?,
        transition_count: checked_usize(raw.transitions, pointer, "transition estimate")?,
        mask_cache_bytes: checked_usize(mask_cache_bytes, pointer, "mask-byte estimate")?,
        enum_trie_nodes: checked_usize(raw.enum_trie_nodes, pointer, "enum-trie estimate")?,
        number_lexers: checked_usize(raw.number_lexers, pointer, "number-lexer estimate")?,
        minimum_output_bytes: checked_usize(
            minimum_output_bytes,
            pointer,
            "minimum output byte estimate",
        )?,
    };
    if estimate.minimum_output_bytes > limits.max_output_bytes {
        return Err(SchemaError::Resource {
            pointer: pointer.to_owned(),
            reason: format!(
                "shortest accepting output is {} bytes and exceeds output cap {}",
                estimate.minimum_output_bytes, limits.max_output_bytes
            ),
        });
    }
    Ok(estimate)
}

fn estimate_node_inner(
    node: &SchemaNode,
    pointer: &str,
    raw: &mut RawEstimate,
) -> Result<u64, SchemaError> {
    add_checked(&mut raw.states, 1, pointer, "state estimate")?;
    add_checked(
        &mut raw.transitions,
        1,
        pointer,
        "finish transition estimate",
    )?;
    match node {
        SchemaNode::Object {
            properties,
            required,
        } => {
            add_checked(
                &mut raw.states,
                1,
                pointer,
                "object delimiter state estimate",
            )?;
            let mut output = 2_u64;
            let mut emitted = 0_u64;
            for (name, child) in properties {
                let child_pointer = pointer_key(&pointer_key(pointer, "properties"), name);
                let child_minimum = estimate_node_inner(child, &child_pointer, raw)?;
                add_checked(
                    &mut raw.transitions,
                    1,
                    pointer,
                    "object property transition estimate",
                )?;
                if required.contains(name) {
                    if emitted != 0 {
                        add_checked(&mut output, 1, pointer, "object comma byte estimate")?;
                    }
                    add_checked(
                        &mut output,
                        (escape_json_string(name).len() + 1) as u64,
                        pointer,
                        "object key byte estimate",
                    )?;
                    add_checked(
                        &mut output,
                        child_minimum,
                        pointer,
                        "object child byte estimate",
                    )?;
                    emitted += 1;
                }
            }
            Ok(output)
        }
        SchemaNode::Array { items, max_items } => {
            let counter_states = u64::try_from(*max_items)
                .ok()
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| resource_overflow(pointer, "array counter state estimate"))?;
            add_checked(
                &mut raw.states,
                counter_states,
                pointer,
                "array counter state estimate",
            )?;
            add_checked(
                &mut raw.transitions,
                counter_states,
                pointer,
                "array item transition estimate",
            )?;
            let _ = estimate_node_inner(items, &pointer_key(pointer, "items"), raw)?;
            Ok(2)
        }
        SchemaNode::String { allowed, .. }
        | SchemaNode::Boolean { allowed }
        | SchemaNode::Null { allowed }
        | SchemaNode::Number { allowed, .. } => {
            if let Some(values) = allowed {
                let mut trie_nodes = 1_u64;
                let mut minimum = None;
                for value in values {
                    let canonical = value.canonical_json();
                    trie_nodes = trie_nodes
                        .checked_add(canonical.len() as u64)
                        .ok_or_else(|| resource_overflow(pointer, "enum trie node estimate"))?;
                    minimum = Some(minimum.map_or(canonical.len() as u64, |current: u64| {
                        current.min(canonical.len() as u64)
                    }));
                }
                add_checked(
                    &mut raw.enum_trie_nodes,
                    trie_nodes,
                    pointer,
                    "enum trie node estimate",
                )?;
                add_checked(
                    &mut raw.states,
                    trie_nodes,
                    pointer,
                    "enum trie state estimate",
                )?;
                add_checked(
                    &mut raw.transitions,
                    trie_nodes,
                    pointer,
                    "enum trie transition estimate",
                )?;
                return Ok(minimum.unwrap_or(0));
            }
            match node {
                SchemaNode::String { .. } => Ok(2),
                SchemaNode::Boolean { .. } | SchemaNode::Null { .. } => Ok(4),
                SchemaNode::Number { .. } => {
                    add_checked(&mut raw.number_lexers, 1, pointer, "number lexer estimate")?;
                    add_checked(&mut raw.states, 3, pointer, "number lexer state estimate")?;
                    add_checked(
                        &mut raw.transitions,
                        4,
                        pointer,
                        "number lexer transition estimate",
                    )?;
                    Ok(1)
                }
                SchemaNode::Object { .. } | SchemaNode::Array { .. } => {
                    Err(SchemaError::InvalidSchema {
                        pointer: pointer.to_owned(),
                        reason: "internal scalar-estimate type mismatch".to_owned(),
                    })
                }
            }
        }
    }
}

fn preflight(estimate: AutomatonEstimate, limits: CompileLimits) -> Result<(), SchemaError> {
    let checks = [
        (estimate.state_count, limits.max_states, "state count"),
        (
            estimate.transition_count,
            limits.max_transitions,
            "transition count",
        ),
        (
            estimate.mask_cache_bytes,
            limits.max_mask_bytes,
            "mask-cache bytes",
        ),
    ];
    for (observed, cap, name) in checks {
        if observed > cap {
            return Err(SchemaError::Resource {
                pointer: "$".to_owned(),
                reason: format!(
                    "{name} estimate {observed} exceeds request cap {cap} before allocation"
                ),
            });
        }
    }
    Ok(())
}

fn build_automaton(
    root: &SchemaNode,
    estimate: AutomatonEstimate,
) -> Result<TypedJsonAutomaton, SchemaError> {
    let mut automaton = TypedJsonAutomaton {
        states: Vec::with_capacity(estimate.state_count),
        transitions: Vec::with_capacity(estimate.transition_count),
        enum_tries: Vec::new(),
        number_lexers: Vec::with_capacity(estimate.number_lexers),
    };
    let accept = push_state(&mut automaton, true, 0);
    let root_state = build_node(&mut automaton, root, accept, estimate.minimum_output_bytes)?;
    automaton.transitions.push(AutomatonTransition {
        from: root_state,
        to: accept,
        label: "accept".to_owned(),
    });
    Ok(automaton)
}

fn build_node(
    automaton: &mut TypedJsonAutomaton,
    node: &SchemaNode,
    accept: usize,
    minimum_remaining_bytes: usize,
) -> Result<usize, SchemaError> {
    let state = push_state(automaton, false, minimum_remaining_bytes);
    match node {
        SchemaNode::Object { properties, .. } => {
            for (name, child) in properties {
                let child_state = build_node(automaton, child, accept, 0)?;
                automaton.transitions.push(AutomatonTransition {
                    from: state,
                    to: child_state,
                    label: format!("key:{name}"),
                });
            }
        }
        SchemaNode::Array { items, .. } => {
            let item_state = build_node(automaton, items, accept, 0)?;
            automaton.transitions.push(AutomatonTransition {
                from: state,
                to: item_state,
                label: "item".to_owned(),
            });
        }
        SchemaNode::Number { integer, allowed } => {
            if let Some(values) = allowed {
                automaton.enum_tries.push(enum_trie(values));
            } else {
                automaton.number_lexers.push(CanonicalNumberLexer {
                    integer_only: *integer,
                    maximum_significant_digits: super::schema::MAX_SIGNIFICANT_DECIMAL_DIGITS,
                    minimum_adjusted_exponent: super::schema::MIN_ADJUSTED_DECIMAL_EXPONENT,
                    maximum_adjusted_exponent: super::schema::MAX_ADJUSTED_DECIMAL_EXPONENT,
                });
            }
        }
        SchemaNode::String { allowed, .. }
        | SchemaNode::Boolean { allowed }
        | SchemaNode::Null { allowed } => {
            if let Some(values) = allowed {
                automaton.enum_tries.push(enum_trie(values));
            }
        }
    }
    automaton.transitions.push(AutomatonTransition {
        from: state,
        to: accept,
        label: node.kind_name().to_owned(),
    });
    Ok(state)
}

fn enum_trie(values: &[ScalarValue]) -> EnumTrie {
    EnumTrie {
        canonical_scalars: values.iter().map(ScalarValue::canonical_json).collect(),
    }
}

fn push_state(
    automaton: &mut TypedJsonAutomaton,
    accepting: bool,
    minimum_remaining_bytes: usize,
) -> usize {
    let id = automaton.states.len();
    automaton.states.push(AutomatonState {
        id,
        accepting,
        minimum_remaining_bytes,
    });
    id
}

fn validate_node(node: &SchemaNode, value: &JsonValue, pointer: &str) -> Result<(), SchemaError> {
    match node {
        SchemaNode::Object {
            properties,
            required,
        } => {
            let object = value
                .as_object()
                .ok_or_else(|| expected_type(pointer, "object", value.kind_name()))?;
            for name in required {
                if !object.contains_key(name) {
                    return Err(SchemaError::Validation {
                        pointer: pointer.to_owned(),
                        reason: format!("missing required property {name:?}"),
                    });
                }
            }
            for (name, value) in object {
                let child = properties
                    .get(name)
                    .ok_or_else(|| SchemaError::Validation {
                        pointer: pointer_key(pointer, name),
                        reason: "property rejected by additionalProperties:false".to_owned(),
                    })?;
                validate_node(child, value, &pointer_key(pointer, name))?;
            }
        }
        SchemaNode::Array { items, max_items } => {
            let array = value
                .as_array()
                .ok_or_else(|| expected_type(pointer, "array", value.kind_name()))?;
            if array.len() > *max_items {
                return Err(SchemaError::Validation {
                    pointer: pointer.to_owned(),
                    reason: format!(
                        "array length {} exceeds effective maxItems {max_items}",
                        array.len()
                    ),
                });
            }
            for (index, value) in array.iter().enumerate() {
                validate_node(items, value, &pointer_index(pointer, index))?;
            }
        }
        SchemaNode::String {
            max_bytes, allowed, ..
        } => {
            let string = value
                .as_string()
                .ok_or_else(|| expected_type(pointer, "string", value.kind_name()))?;
            if string.len() > *max_bytes {
                return Err(SchemaError::Validation {
                    pointer: pointer.to_owned(),
                    reason: format!(
                        "string is {} bytes and exceeds effective maxLength {max_bytes}",
                        string.len()
                    ),
                });
            }
            validate_allowed(
                allowed.as_deref(),
                ScalarValue::String(string.to_owned()),
                pointer,
            )?;
        }
        SchemaNode::Number { integer, allowed } => {
            let number = value
                .as_number()
                .ok_or_else(|| expected_type(pointer, node.kind_name(), value.kind_name()))?;
            if *integer && number.integer_value().is_none() {
                return Err(SchemaError::Validation {
                    pointer: pointer.to_owned(),
                    reason: "number is not an exact signed/unsigned 64-bit integer".to_owned(),
                });
            }
            validate_allowed(
                allowed.as_deref(),
                ScalarValue::Number(number.clone()),
                pointer,
            )?;
        }
        SchemaNode::Boolean { allowed } => {
            let boolean = value
                .as_boolean()
                .ok_or_else(|| expected_type(pointer, "boolean", value.kind_name()))?;
            validate_allowed(allowed.as_deref(), ScalarValue::Boolean(boolean), pointer)?;
        }
        SchemaNode::Null { allowed } => {
            if value != &JsonValue::Null {
                return Err(expected_type(pointer, "null", value.kind_name()));
            }
            validate_allowed(allowed.as_deref(), ScalarValue::Null, pointer)?;
        }
    }
    Ok(())
}

fn expected_type(pointer: &str, expected: &str, observed: &str) -> SchemaError {
    SchemaError::Validation {
        pointer: pointer.to_owned(),
        reason: format!("expected {expected}, observed {observed}"),
    }
}

fn validate_allowed(
    allowed: Option<&[ScalarValue]>,
    value: ScalarValue,
    pointer: &str,
) -> Result<(), SchemaError> {
    if let Some(allowed) = allowed {
        if !allowed.contains(&value) {
            return Err(SchemaError::Validation {
                pointer: pointer.to_owned(),
                reason: "value is outside enum/const constraint".to_owned(),
            });
        }
    }
    Ok(())
}

fn sample_node(node: &SchemaNode, pointer: &str) -> Result<String, SchemaError> {
    if let Some(allowed) = node.allowed() {
        let value = allowed.first().ok_or_else(|| SchemaError::InvalidSchema {
            pointer: pointer.to_owned(),
            reason: "empty enum has no sample".to_owned(),
        })?;
        return Ok(value.canonical_json());
    }
    match node {
        SchemaNode::Object {
            properties,
            required,
        } => {
            let mut output = String::from("{");
            for (index, name) in required.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                let child = properties
                    .get(name)
                    .ok_or_else(|| SchemaError::InvalidSchema {
                        pointer: pointer.to_owned(),
                        reason: format!("required property {name:?} is not declared"),
                    })?;
                output.push_str(&escape_json_string(name));
                output.push(':');
                output.push_str(&sample_node(child, &pointer_key(pointer, name))?);
            }
            output.push('}');
            Ok(output)
        }
        SchemaNode::Array { .. } => Ok("[]".to_owned()),
        SchemaNode::String { .. } => Ok("\"\"".to_owned()),
        SchemaNode::Number { .. } => Ok("0".to_owned()),
        SchemaNode::Boolean { .. } => Ok("false".to_owned()),
        SchemaNode::Null { .. } => Ok("null".to_owned()),
    }
}

fn checked_usize(value: u64, pointer: &str, context: &str) -> Result<usize, SchemaError> {
    usize::try_from(value).map_err(|_| SchemaError::Resource {
        pointer: pointer.to_owned(),
        reason: format!("{context} exceeds this platform's checked usize range"),
    })
}

fn add_checked(
    target: &mut u64,
    value: u64,
    pointer: &str,
    context: &str,
) -> Result<(), SchemaError> {
    *target = target
        .checked_add(value)
        .ok_or_else(|| resource_overflow(pointer, context))?;
    Ok(())
}

fn resource_overflow(pointer: &str, context: &str) -> SchemaError {
    SchemaError::Resource {
        pointer: pointer.to_owned(),
        reason: format!("checked arithmetic overflow while computing {context}"),
    }
}
