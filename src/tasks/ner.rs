//! Source-constrained named-entity recognition on the native eager task path.
//!
//! Surface forms are generated from the exact source language, never relocated
//! by fuzzy matching after free generation. Every occurrence is retained when
//! a surface repeats. This proves byte/scalar coordinates and source membership,
//! not that the model assigned the correct entity type or found every entity.

use std::{collections::BTreeSet, error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{
    canonjson,
    execution_identity::ExecutionIdentity,
    grammar::{CompileLimits, runtime::SourceRuntimeLimits},
    native_engine::{
        constrained::{JsonDecodeOptions, JsonWorkBudget},
        decode::DecodeStepControl,
        hf_bf16_eager::HfBf16EagerEngine,
    },
    tokenizer::specials::TemplateControlIds,
    validation::{JsonLimits, JsonValue, parse_json_with_limits,
        grounded_fields::{SourceOccurrence, VerifiedSourceSpan}},
};
use super::{
    extract::{ExtractError, ExtractPlan, ExtractResult, ExtractionGrounding, ExtractionVocabulary, SourceDocument},
    ir::{ScoreSpace, TaskPlan},
};

pub const NER_TASK_VERSION: &str = "ner-v1";
const MAX_ENTITIES: usize = 4096;
const MAX_MENTION_SCALARS: usize = 65_536;

/// Closed type vocabulary. Unknown labels are not coerced into a known type.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType { Person, Organization, Location, Date, Time, Money, Product, Event }
impl EntityType {
    pub const ALL: [Self; 8] = [Self::Person, Self::Organization, Self::Location,
        Self::Date, Self::Time, Self::Money, Self::Product, Self::Event];
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Person => "person", Self::Organization => "organization", Self::Location => "location",
            Self::Date => "date", Self::Time => "time", Self::Money => "money", Self::Product => "product", Self::Event => "event",
        }
    }
    pub fn from_label(label: &str) -> Result<Self, NerError> {
        Self::ALL.into_iter().find(|kind| kind.label() == label).ok_or(NerError::UnknownType)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NerOptions {
    pub types: Vec<EntityType>,
    pub max_entities: usize,
    /// Unicode scalar values, not UTF-8 bytes or UTF-16 code units.
    pub max_mention_scalars: usize,
}
impl Default for NerOptions {
    fn default() -> Self {
        Self { types: vec![EntityType::Person, EntityType::Organization, EntityType::Location],
            max_entities: 64, max_mention_scalars: 256 }
    }
}
impl NerOptions {
    pub fn validate(&self) -> Result<(), NerError> {
        if self.types.is_empty() || self.types.len() > EntityType::ALL.len()
            || self.max_entities == 0 || self.max_entities > MAX_ENTITIES
            || self.max_mention_scalars == 0 || self.max_mention_scalars > MAX_MENTION_SCALARS
        { return Err(NerError::InvalidOptions); }
        let unique: BTreeSet<_> = self.types.iter().copied().collect();
        if unique.len() != self.types.len() { return Err(NerError::InvalidOptions); }
        Ok(())
    }

    /// Exact schema bytes to put into the TaskIR grammar digest and trusted
    /// task instruction. Type order is canonical; type membership and both
    /// caps are part of the schema identity. No model needs to be loaded.
    pub fn schema_source(&self) -> Result<String, NerError> {
        self.validate()?;
        let mut labels: Vec<_> = self.types.iter().map(|kind| kind.label()).collect();
        labels.sort_unstable();
        let schema = NerSchema {
            kind: "array", max_items: self.max_entities,
            items: EntitySchema { kind: "object", additional_properties: false, required: ["text", "type"],
                properties: EntityProperties {
                    text: MentionSchema { kind: "string", max_length: self.max_mention_scalars, source: "verbatim" },
                    entity_type: TypeSchema { kind: "string", labels },
                },
            },
        };
        canonjson::canonical_string(&schema).map_err(|_| NerError::Serialization)
    }
}

#[derive(Serialize)]
struct NerSchema { #[serde(rename = "type")] kind: &'static str, #[serde(rename = "maxItems")] max_items: usize, items: EntitySchema }
#[derive(Serialize)]
struct EntitySchema {
    #[serde(rename = "type")] kind: &'static str,
    #[serde(rename = "additionalProperties")] additional_properties: bool,
    required: [&'static str; 2], properties: EntityProperties,
}
#[derive(Serialize)]
struct EntityProperties { text: MentionSchema, #[serde(rename = "type")] entity_type: TypeSchema }
#[derive(Serialize)]
struct MentionSchema {
    #[serde(rename = "type")] kind: &'static str,
    #[serde(rename = "maxLength")] max_length: usize,
    #[serde(rename = "x-fnlp-source")] source: &'static str,
}
#[derive(Serialize)]
struct TypeSchema { #[serde(rename = "type")] kind: &'static str, #[serde(rename = "enum")] labels: Vec<&'static str> }

/// One model-proposed surface/type pair with its complete source occurrences.
/// Ambiguous mentions intentionally have no privileged "selected" offset.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NamedEntity {
    pub text: String,
    pub entity_type: EntityType,
    pub occurrence: SourceOccurrence,
    pub spans: Vec<VerifiedSourceSpan>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NerResult {
    pub schema_version: u32,
    pub task_spec_version: String,
    pub numerics_profile: String,
    pub score_space: ScoreSpace,
    pub grounding: ExtractionGrounding,
    pub entities: Vec<NamedEntity>,
    /// Exact generated token sequence including terminal EOS, not prompt ids.
    pub generated_token_ids: Vec<u32>,
    pub forward_positions: u64,
    pub projected_logits: u64,
    pub mask_node_visit_charge: u64,
}

#[derive(Debug)]
pub enum NerError {
    InvalidOptions, UnknownType, InvalidResult, OutputBudgetExceeded, AllocationRefused,
    Serialization, Extraction(ExtractError),
}
impl fmt::Display for NerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions => f.write_str("NER requires unique types and bounded nonzero mention/count limits"),
            Self::UnknownType => f.write_str("unknown NER type; expected person, organization, location, date, time, money, product, or event"),
            Self::InvalidResult => f.write_str("NER has no result: invalid entity or incomplete source evidence"),
            Self::OutputBudgetExceeded => f.write_str("NER complete result exceeds its byte budget"),
            Self::AllocationRefused => f.write_str("NER result allocation refused"),
            Self::Serialization => f.write_str("NER result serialization failed"),
            Self::Extraction(error) => write!(f, "NER has no result: {error}"),
        }
    }
}
impl Error for NerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Extraction(error) => Some(error), _ => None }
    }
}
impl From<ExtractError> for NerError { fn from(error: ExtractError) -> Self { Self::Extraction(error) } }

/// A request-owned plan, deliberately neither Debug nor serializable. Source
/// bytes and prompt identities stay private in the shared extraction core.
pub struct NerPlan {
    extraction: ExtractPlan,
    options: NerOptions,
    max_result_bytes: u64,
}
impl NerPlan {
    /// The caller supplies exact trusted instruction/scaffold tokens and the
    /// SourceDocument's document tokens in a ner-v1 TaskPlan. Its grammar must
    /// bind options.schema_source() and SOURCE_JSON_RUNTIME_VERSION, and its
    /// postconditions must require SourceSpansVerified. No prompt is silently
    /// rewritten here and no model/artifact loader is added.
    #[allow(clippy::too_many_arguments)]
    pub fn from_task_plan(
        task: &TaskPlan, document: &SourceDocument, options: NerOptions,
        decode_options: JsonDecodeOptions, compiler_limits: CompileLimits,
        controls: &TemplateControlIds, source_limits: SourceRuntimeLimits,
    ) -> Result<Self, NerError> {
        let schema = options.schema_source()?;
        let extraction = ExtractPlan::from_builtin(task, &schema, decode_options, compiler_limits,
            controls, Some((document, source_limits)), NER_TASK_VERSION)?;
        Ok(Self { extraction, options, max_result_bytes: task.ir().budget().max_output_bytes })
    }
    pub fn bind_identity(&self, identity: ExecutionIdentity) -> Result<ExecutionIdentity, NerError> {
        self.extraction.bind_identity(identity).map_err(Into::into)
    }
    pub fn verify_identity(&self, identity: &ExecutionIdentity) -> Result<(), NerError> {
        self.extraction.verify_identity(identity).map_err(Into::into)
    }
    pub fn execute_eager<C: DecodeStepControl>(
        &self, engine: &mut HfBf16EagerEngine, identity: &ExecutionIdentity,
        vocabulary: &ExtractionVocabulary, work: JsonWorkBudget, control: &mut C,
    ) -> Result<NerResult, NerError> {
        let raw = self.extraction.execute_eager(engine, identity, vocabulary, work, control)?;
        finalize(raw, &self.options, self.max_result_bytes)
    }
}

// Private conversion: only the extraction core's independently verified result
// reaches this from production. No public deserialize-to-authority shortcut.
fn finalize(raw: ExtractResult, options: &NerOptions, max_bytes: u64) -> Result<NerResult, NerError> {
    if raw.schema_version != 2 || raw.task_spec_version != NER_TASK_VERSION
        || raw.score_space != ScoreSpace::NotComputed || raw.grounding != ExtractionGrounding::SourceMembership
    { return Err(NerError::InvalidResult); }
    let cap = usize::try_from(max_bytes).unwrap_or(usize::MAX);
    let parsed = parse_json_with_limits(&raw.output.json, JsonLimits {
        max_input_bytes: cap, max_string_lexeme_bytes: cap,
        max_container_entries: options.max_entities.max(2), max_depth: 4,
        ..JsonLimits::default()
    }).map_err(|_| NerError::InvalidResult)?;
    let JsonValue::Array(items) = parsed else { return Err(NerError::InvalidResult); };
    if items.len() > options.max_entities { return Err(NerError::InvalidResult); }
    let mut evidence = raw.source_fields.into_iter();
    let mut entities = Vec::new();
    entities.try_reserve_exact(items.len()).map_err(|_| NerError::AllocationRefused)?;
    for (index, value) in items.into_iter().enumerate() {
        let JsonValue::Object(mut fields) = value else { return Err(NerError::InvalidResult); };
        if fields.len() != 2 { return Err(NerError::InvalidResult); }
        let Some(JsonValue::String(text)) = fields.remove("text") else { return Err(NerError::InvalidResult); };
        let Some(JsonValue::String(label)) = fields.remove("type") else { return Err(NerError::InvalidResult); };
        let kind = EntityType::from_label(&label)?;
        let scalars = text.chars().count();
        if text.is_empty() || scalars > options.max_mention_scalars || !options.types.contains(&kind) {
            return Err(NerError::InvalidResult);
        }
        let proof = evidence.next().ok_or(NerError::InvalidResult)?;
        if proof.json_pointer != format!("/{index}/text") || proof.spans.is_empty()
            || proof.occurrence != if proof.spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }
            || proof.spans.iter().any(|s| s.byte_end.checked_sub(s.byte_start) != Some(text.len())
                || s.scalar_end.checked_sub(s.scalar_start) != Some(scalars))
            || proof.spans.windows(2).any(|pair| pair[0].byte_start >= pair[1].byte_start
                || pair[0].scalar_start >= pair[1].scalar_start)
        { return Err(NerError::InvalidResult); }
        entities.push(NamedEntity { text, entity_type: kind, occurrence: proof.occurrence, spans: proof.spans });
    }
    if evidence.next().is_some() { return Err(NerError::InvalidResult); }
    let result = NerResult {
        schema_version: 1, task_spec_version: NER_TASK_VERSION.to_owned(),
        numerics_profile: raw.output.numerics_profile, score_space: ScoreSpace::NotComputed,
        grounding: ExtractionGrounding::SourceMembership, entities, generated_token_ids: raw.output.token_ids,
        forward_positions: raw.output.forward_positions, projected_logits: raw.output.projected_logits,
        mask_node_visit_charge: raw.output.mask_node_visit_charge,
    };
    if canonjson::canonical_bytes(&result).map_err(|_| NerError::Serialization)?.len() as u64 > max_bytes {
        return Err(NerError::OutputBudgetExceeded);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammar::runtime::{JsonProgram, SOURCE_JSON_RUNTIME_VERSION},
        native_engine::{constrained::JsonDecodeOutput, hf_bf16_eager::HF_BF16_EAGER_PROFILE},
        validation::{SourceSpan, validate_source_span}};
    fn raw(source: &str, json: &str, options: &NerOptions) -> ExtractResult {
        let p = JsonProgram::compile_with_source(&options.schema_source().unwrap(), source,
            CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
        ExtractResult { schema_version: 2, task_spec_version: NER_TASK_VERSION.to_owned(), score_space: ScoreSpace::NotComputed,
            grounding: ExtractionGrounding::SourceMembership, source_fields: p.source_fields(json).unwrap(),
            output: JsonDecodeOutput { schema_version: 1, numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(),
                token_ids: vec![1, 2, 0], json: json.to_owned(), forward_positions: 8, projected_logits: 8 * 166_144,
                mask_node_visit_charge: 20 },
        }
    }
    const ALICE: &str = r#"[{"text":"Alice","type":"person"}]"#;

    #[test]
    fn configured_type_order_does_not_change_schema_identity() {
        let a = NerOptions::default(); let mut b = a.clone(); b.types.reverse();
        assert_eq!(a.schema_source().unwrap(), b.schema_source().unwrap());
        b.types.pop(); assert_ne!(a.schema_source().unwrap(), b.schema_source().unwrap());
    }
    #[test]
    fn invalid_type_sets_and_unbounded_options_refuse() {
        let mut o = NerOptions::default(); o.types.push(EntityType::Person); assert!(o.validate().is_err());
        o.types.clear(); assert!(o.validate().is_err());
        o = NerOptions::default(); o.max_entities = usize::MAX; assert!(o.validate().is_err());
        o = NerOptions::default(); o.max_mention_scalars = 0; assert!(o.validate().is_err());
        assert!(EntityType::from_label("PERSON").is_err()); assert!(EntityType::from_label("unknown").is_err());
    }
    #[test]
    fn typed_schema_executes_verbatim_mentions_and_rejects_invented_text() {
        let options = NerOptions::default();
        let p = JsonProgram::compile_with_source(&options.schema_source().unwrap(), "Alice",
            CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
        let mut state = p.initial_state(); assert!(state.consume_bytes(ALICE.as_bytes())); assert!(state.is_accepting());
        assert_eq!(p.version(), SOURCE_JSON_RUNTIME_VERSION);
        assert!(!p.initial_state().consume_bytes(br#"[{"text":"Carol"#));
        assert!(!p.initial_state().consume_bytes(br#"[{"text":"Alice","type":"event"#));
    }
    #[test]
    fn repeated_entities_preserve_every_occurrence_and_explicit_ambiguity() {
        let options = NerOptions::default();
        let result = finalize(raw("Alice Bob Alice", ALICE, &options), &options, 8192).unwrap();
        assert_eq!(result.entities[0].occurrence, SourceOccurrence::Ambiguous);
        assert_eq!(result.entities[0].spans.iter().map(|s| s.byte_start).collect::<Vec<_>>(), vec![0, 10]);
        let serialized = canonjson::canonical_string(&result).unwrap();
        for absent in ["confidence", "prompt_digest", "selected_offset"] { assert!(!serialized.contains(absent)); }
    }
    #[test]
    fn unicode_byte_and_scalar_offsets_round_trip_independently() {
        let options = NerOptions::default(); let source = "é上海 and 上海";
        let json = r#"[{"text":"上海","type":"location"}]"#;
        let result = finalize(raw(source, json, &options), &options, 8192).unwrap();
        for e in result.entities { for s in e.spans {
            validate_source_span(source, &e.text, SourceSpan::new(s.byte_start, s.byte_end, s.scalar_start, s.scalar_end)).unwrap();
        } }
    }
    #[test]
    fn empty_entity_set_is_a_success_not_abstention_or_failure() {
        let options = NerOptions::default();
        assert!(finalize(raw("plain text", "[]", &options), &options, 8192).unwrap().entities.is_empty());
    }
    #[test]
    fn empty_mentions_and_unselected_types_do_not_finalize() {
        let options = NerOptions::default();
        let bad = raw("Alice", r#"[{"text":"","type":"person"}]"#, &options);
        assert!(matches!(finalize(bad, &options, 8192), Err(NerError::InvalidResult)));
        let mut narrow = options.clone(); narrow.types = vec![EntityType::Organization];
        assert!(finalize(raw("Alice", ALICE, &options), &narrow, 8192).is_err());
    }
    #[test]
    fn missing_swapped_extra_or_inconsistent_evidence_is_rejected() {
        let options = NerOptions::default();
        for corruption in 0..5 {
            let mut r = raw("Alice", ALICE, &options);
            match corruption {
                0 => r.source_fields.clear(),
                1 => r.source_fields[0].json_pointer = "/1/text".to_owned(),
                2 => r.source_fields.push(r.source_fields[0].clone()),
                3 => r.source_fields[0].occurrence = SourceOccurrence::Ambiguous,
                _ => r.source_fields[0].spans[0].scalar_end += 1,
            }
            assert!(matches!(finalize(r, &options, 8192), Err(NerError::InvalidResult)));
        }
    }
    #[test]
    fn complete_ner_envelope_is_byte_bounded() {
        let options = NerOptions::default();
        let result = finalize(raw("Alice", ALICE, &options), &options, 8192).unwrap();
        let bytes = canonjson::canonical_bytes(&result).unwrap().len() as u64;
        assert!(finalize(raw("Alice", ALICE, &options), &options, bytes).is_ok());
        assert!(matches!(finalize(raw("Alice", ALICE, &options), &options, bytes - 1), Err(NerError::OutputBudgetExceeded)));
    }
    #[test]
    fn cancellation_preserves_its_typed_cause() {
        use crate::native_engine::{constrained::JsonDecodeError, decode::DecodeCancellationKind};
        let error: NerError = ExtractError::Decode(JsonDecodeError::Cancelled(DecodeCancellationKind::Deadline)).into();
        assert!(matches!(error, NerError::Extraction(ExtractError::Decode(JsonDecodeError::Cancelled(DecodeCancellationKind::Deadline)))));
    }
}
