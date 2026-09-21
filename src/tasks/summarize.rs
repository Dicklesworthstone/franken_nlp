//! Cited summaries on the native constrained-json execution path.
//!
//! Every emitted bullet has byte-verified quotes from the supplied source.
//! This is a STRUCTURAL citation contract. A quote's existence does not establish
//! that it entails the bullet, and the same model is not its own truth oracle.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{
    canonjson,
    execution_identity::ExecutionIdentity,
    grammar::{CompileLimits, runtime::SourceRuntimeLimits},
    native_engine::{constrained::{JsonDecodeOptions, JsonWorkBudget},
        decode::DecodeStepControl, hf_bf16_eager::HfBf16EagerEngine},
    tokenizer::specials::TemplateControlIds,
    validation::{JsonLimits, JsonValue, parse_json_with_limits,
        grounded_fields::{SourceFieldEvidence, SourceOccurrence, VerifiedSourceSpan}},
};
use super::{
    extract::{ExtractError, ExtractPlan, ExtractResult, ExtractionGrounding, ExtractionVocabulary, SourceDocument},
    ir::{ScoreSpace, TaskPlan},
};

pub const SUMMARIZE_TASK_VERSION: &str = "summarize-v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SummaryOptions {
    pub max_bullets: usize,
    /// Unicode scalar limits. These are not linguistic word counts.
    pub max_bullet_scalars: usize,
    pub max_citations_per_bullet: usize,
    pub max_quote_scalars: usize,
}
impl Default for SummaryOptions {
    fn default() -> Self {
        Self { max_bullets: 8, max_bullet_scalars: 512, max_citations_per_bullet: 4, max_quote_scalars: 256 }
    }
}
impl SummaryOptions {
    pub fn validate(self) -> Result<(), SummaryError> {
        if !(1..=1024).contains(&self.max_bullets) || !(1..=65_536).contains(&self.max_bullet_scalars)
            || !(1..=128).contains(&self.max_citations_per_bullet) || !(1..=4096).contains(&self.max_quote_scalars)
        { return Err(SummaryError::InvalidOptions); }
        Ok(())
    }
    /// All output-language options are reflected in these canonical bytes and
    /// therefore in the TaskIR schema digest. No unbound post-hoc length policy.
    pub fn schema_source(self) -> Result<String, SummaryError> {
        self.validate()?;
        #[derive(Serialize)]
        struct TextSchema {
            #[serde(rename = "type")] kind: &'static str,
            #[serde(rename = "maxLength")] max_length: usize,
        }
        #[derive(Serialize)]
        struct QuoteSchema {
            #[serde(rename = "type")] kind: &'static str,
            #[serde(rename = "maxLength")] max_length: usize,
            #[serde(rename = "x-fnlp-source")] source: &'static str,
        }
        #[derive(Serialize)]
        struct QuoteList {
            #[serde(rename = "type")] kind: &'static str,
            #[serde(rename = "maxItems")] max_items: usize, items: QuoteSchema,
        }
        #[derive(Serialize)]
        struct Properties { citations: QuoteList, text: TextSchema }
        #[derive(Serialize)]
        struct BulletSchema {
            #[serde(rename = "type")] kind: &'static str,
            #[serde(rename = "additionalProperties")] additional_properties: bool,
            required: [&'static str; 2], properties: Properties,
        }
        #[derive(Serialize)]
        struct SummarySchema {
            #[serde(rename = "type")] kind: &'static str,
            #[serde(rename = "maxItems")] max_items: usize, items: BulletSchema,
        }
        canonjson::canonical_string(&SummarySchema { kind: "array", max_items: self.max_bullets,
            items: BulletSchema { kind: "object", additional_properties: false, required: ["citations", "text"],
                properties: Properties {
                    citations: QuoteList { kind: "array", max_items: self.max_citations_per_bullet,
                        items: QuoteSchema { kind: "string", max_length: self.max_quote_scalars, source: "verbatim" } },
                    text: TextSchema { kind: "string", max_length: self.max_bullet_scalars },
                },
            },
        }).map_err(|_| SummaryError::Serialization)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceCitation {
    pub quote: String,
    pub occurrence: SourceOccurrence,
    /// Every exact occurrence. Ambiguous quotes get no guessed selected offset.
    pub spans: Vec<VerifiedSourceSpan>,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CitedBullet {
    pub text: String,
    pub citations: Vec<SourceCitation>,
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CitationGuarantee { StructuralSourceMembership }
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SummarySemanticSupport { NotAssessed }

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SummaryResult {
    pub schema_version: u32,
    pub task_spec_version: String,
    pub numerics_profile: String,
    pub citation_guarantee: CitationGuarantee,
    pub semantic_support: SummarySemanticSupport,
    pub score_space: ScoreSpace,
    pub bullets: Vec<CitedBullet>,
    pub generated_token_ids: Vec<u32>,
    pub forward_positions: u64,
    pub projected_logits: u64,
    pub mask_node_visit_charge: u64,
}
#[derive(Debug)]
pub enum SummaryError {
    InvalidOptions, InvalidResult, OutputBudgetExceeded, AllocationRefused, Serialization,
    Extraction(ExtractError),
}
impl fmt::Display for SummaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions => f.write_str("summary requires bounded nonzero bullet and citation limits"),
            Self::InvalidResult => f.write_str("summary has no result: invalid bullet or incomplete citation evidence"),
            Self::OutputBudgetExceeded => f.write_str("complete cited summary exceeds its byte budget"),
            Self::AllocationRefused => f.write_str("cited summary allocation refused"),
            Self::Serialization => f.write_str("cited summary serialization failed"),
            Self::Extraction(error) => write!(f, "summary execution failed: {error}"),
        }
    }
}
impl Error for SummaryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Extraction(error) => Some(error), _ => None }
    }
}
impl From<ExtractError> for SummaryError {
    fn from(error: ExtractError) -> Self { Self::Extraction(error) }
}

/// Request-owned executable plan, deliberately not Debug/Serialize. Trusted
/// prompt construction remains the caller's responsibility; source encoding,
/// grammar identity, model identity and native resource bounds are checked.
pub struct SummaryPlan {
    extraction: ExtractPlan,
    options: SummaryOptions,
    max_result_bytes: u64,
}
impl SummaryPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn from_task_plan(task: &TaskPlan, document: &SourceDocument, options: SummaryOptions,
        decode: JsonDecodeOptions, compiler: CompileLimits, controls: &TemplateControlIds,
        source_limits: SourceRuntimeLimits) -> Result<Self, SummaryError> {
        let schema = options.schema_source()?;
        let extraction = ExtractPlan::from_builtin(task, &schema, decode, compiler,
            controls, Some((document, source_limits)), SUMMARIZE_TASK_VERSION)?;
        Ok(Self { extraction, options, max_result_bytes: task.ir().budget().max_output_bytes })
    }
    pub fn bind_identity(&self, identity: ExecutionIdentity) -> Result<ExecutionIdentity, SummaryError> {
        self.extraction.bind_identity(identity).map_err(Into::into)
    }
    pub fn verify_identity(&self, identity: &ExecutionIdentity) -> Result<(), SummaryError> {
        self.extraction.verify_identity(identity).map_err(Into::into)
    }
    pub fn execute_eager<C: DecodeStepControl>(&self, engine: &mut HfBf16EagerEngine,
        identity: &ExecutionIdentity, vocabulary: &ExtractionVocabulary, work: JsonWorkBudget,
        control: &mut C) -> Result<SummaryResult, SummaryError> {
        let raw = self.extraction.execute_eager(engine, identity, vocabulary, work, control)?;
        finalize(raw, self.options, self.max_result_bytes)
    }
}

fn citation(quote: String, pointer: &str, proof: SourceFieldEvidence, max_scalars: usize) -> Result<SourceCitation, SummaryError> {
    let scalars = quote.chars().count();
    if quote.is_empty() || scalars > max_scalars || proof.json_pointer != pointer || proof.spans.is_empty()
        || proof.occurrence != if proof.spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }
        || proof.spans.iter().any(|span| span.byte_end.checked_sub(span.byte_start) != Some(quote.len())
            || span.scalar_end.checked_sub(span.scalar_start) != Some(scalars))
        || proof.spans.windows(2).any(|pair| pair[0].byte_start >= pair[1].byte_start
            || pair[0].scalar_start >= pair[1].scalar_start)
    { return Err(SummaryError::InvalidResult); }
    Ok(SourceCitation { quote, occurrence: proof.occurrence, spans: proof.spans })
}

// Only the shared decoder's independently verified source result can reach this
// private conversion in production. Parsing never promotes caller-provided JSON.
pub(super) fn finalize(raw: ExtractResult, options: SummaryOptions, max_bytes: u64) -> Result<SummaryResult, SummaryError> {
    options.validate()?;
    if raw.schema_version != 2 || raw.task_spec_version != SUMMARIZE_TASK_VERSION
        || raw.score_space != ScoreSpace::NotComputed || raw.grounding != ExtractionGrounding::SourceMembership
    { return Err(SummaryError::InvalidResult); }
    let cap = usize::try_from(max_bytes).unwrap_or(usize::MAX);
    let parsed = parse_json_with_limits(&raw.output.json, JsonLimits { max_input_bytes: cap,
        max_string_lexeme_bytes: cap, max_depth: 6,
        max_container_entries: options.max_bullets.max(options.max_citations_per_bullet).max(2),
        ..JsonLimits::default()
    }).map_err(|_| SummaryError::InvalidResult)?;
    let JsonValue::Array(items) = parsed else { return Err(SummaryError::InvalidResult); };
    if items.len() > options.max_bullets { return Err(SummaryError::InvalidResult); }
    let mut proofs = raw.source_fields.into_iter(); let mut bullets = Vec::new();
    bullets.try_reserve_exact(items.len()).map_err(|_| SummaryError::AllocationRefused)?;
    for (index, value) in items.into_iter().enumerate() {
        let JsonValue::Object(mut fields) = value else { return Err(SummaryError::InvalidResult); };
        if fields.len() != 2 { return Err(SummaryError::InvalidResult); }
        let Some(JsonValue::String(text)) = fields.remove("text") else { return Err(SummaryError::InvalidResult); };
        if !text.chars().any(|c| !c.is_whitespace()) || text.chars().count() > options.max_bullet_scalars {
            return Err(SummaryError::InvalidResult);
        }
        let Some(JsonValue::Array(quotes)) = fields.remove("citations") else { return Err(SummaryError::InvalidResult); };
        if quotes.is_empty() || quotes.len() > options.max_citations_per_bullet { return Err(SummaryError::InvalidResult); }
        let mut citations = Vec::new();
        citations.try_reserve_exact(quotes.len()).map_err(|_| SummaryError::AllocationRefused)?;
        for (q, quote) in quotes.into_iter().enumerate() {
            let JsonValue::String(quote) = quote else { return Err(SummaryError::InvalidResult); };
            citations.push(citation(quote, &format!("/{index}/citations/{q}"),
                proofs.next().ok_or(SummaryError::InvalidResult)?, options.max_quote_scalars)?);
        }
        bullets.push(CitedBullet { text, citations });
    }
    if proofs.next().is_some() { return Err(SummaryError::InvalidResult); }
    let result = SummaryResult { schema_version: 1, task_spec_version: SUMMARIZE_TASK_VERSION.to_owned(),
        numerics_profile: raw.output.numerics_profile, citation_guarantee: CitationGuarantee::StructuralSourceMembership,
        semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
        bullets, generated_token_ids: raw.output.token_ids, forward_positions: raw.output.forward_positions,
        projected_logits: raw.output.projected_logits, mask_node_visit_charge: raw.output.mask_node_visit_charge,
    };
    if canonjson::canonical_bytes(&result).map_err(|_| SummaryError::Serialization)?.len() as u64 > max_bytes {
        return Err(SummaryError::OutputBudgetExceeded);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammar::runtime::{JsonProgram, SOURCE_JSON_RUNTIME_VERSION},
        native_engine::{constrained::JsonDecodeOutput, hf_bf16_eager::HF_BF16_EAGER_PROFILE},
        validation::{SourceSpan, validate_source_span}};
    const VALID: &str = r#"[{"citations":["Alice"],"text":"Alice is named."}]"#;
    fn raw(source: &str, json: &str, options: SummaryOptions) -> ExtractResult {
        let program = JsonProgram::compile_with_source(&options.schema_source().unwrap(), source,
            CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
        ExtractResult { schema_version: 2, task_spec_version: SUMMARIZE_TASK_VERSION.to_owned(),
            score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
            source_fields: program.source_fields(json).unwrap(),
            output: JsonDecodeOutput { schema_version: 1, numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(),
                token_ids: vec![1, 2, 0], json: json.to_owned(), forward_positions: 8,
                projected_logits: 100, mask_node_visit_charge: 20 },
        }
    }
    #[test]
    fn all_language_limits_are_validated_and_bound_in_schema_identity() {
        let base = SummaryOptions::default();
        for axis in 0..4 {
            let mut changed = base;
            match axis { 0 => changed.max_bullets = 2, 1 => changed.max_bullet_scalars = 2,
                2 => changed.max_citations_per_bullet = 2, _ => changed.max_quote_scalars = 2 }
            assert_ne!(base.schema_source().unwrap(), changed.schema_source().unwrap());
            match axis { 0 => changed.max_bullets = 0, 1 => changed.max_bullet_scalars = 0,
                2 => changed.max_citations_per_bullet = 0, _ => changed.max_quote_scalars = 0 }
            assert!(changed.schema_source().is_err());
        }
    }
    #[test]
    fn native_language_allows_summary_prose_but_masks_off_source_quotes() {
        let o = SummaryOptions::default();
        let p = JsonProgram::compile_with_source(&o.schema_source().unwrap(), "Alice",
            CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
        assert_eq!(p.version(), SOURCE_JSON_RUNTIME_VERSION);
        let mut state = p.initial_state(); assert!(state.consume_bytes(VALID.as_bytes())); assert!(state.is_accepting());
        assert!(!p.initial_state().consume_bytes(br#"[{"citations":["Carol"#));
    }
    #[test]
    fn every_bullet_requires_at_least_one_nonempty_citation() {
        let o = SummaryOptions::default();
        for json in [r#"[{"citations":[],"text":"Alice is named."}]"#,
            r#"[{"citations":[""],"text":"Alice is named."}]"#,
            r#"[{"citations":["Alice"],"text":""}]"#,
            r#"[{"citations":["Alice"],"text":" \t"}]"#] {
            assert!(matches!(finalize(raw("Alice", json, o), o, 8192), Err(SummaryError::InvalidResult)));
        }
    }
    #[test]
    fn multiple_bullets_and_quotes_keep_exact_provenance() {
        let o = SummaryOptions::default(); let source = "Alice met Bob in 上海. Alice";
        let json = r#"[{"citations":["Alice","Bob"],"text":"Two people met."},{"citations":["上海"],"text":"A location is named."}]"#;
        let result = finalize(raw(source, json, o), o, 16384).unwrap();
        assert_eq!(result.bullets.len(), 2);
        assert_eq!(result.bullets[0].citations[0].occurrence, SourceOccurrence::Ambiguous);
        for bullet in result.bullets { for citation in bullet.citations { for s in citation.spans {
            validate_source_span(source, &citation.quote, SourceSpan::new(s.byte_start, s.byte_end,
                s.scalar_start, s.scalar_end)).unwrap();
        } } }
    }
    #[test]
    fn structural_citations_are_not_misrepresented_as_entailment() {
        let o = SummaryOptions::default();
        let json = r#"[{"citations":["Alice"],"text":"The moon is made of cheese."}]"#;
        let result = finalize(raw("Alice", json, o), o, 8192).unwrap();
        assert_eq!(result.citation_guarantee, CitationGuarantee::StructuralSourceMembership);
        assert_eq!(result.semantic_support, SummarySemanticSupport::NotAssessed);
        let serialized = canonjson::canonical_string(&result).unwrap();
        for absent in ["confidence", "prompt_digest", "document_digest", "entailed"] { assert!(!serialized.contains(absent)); }
    }
    #[test]
    fn missing_swapped_extra_and_inconsistent_proofs_fail_closed() {
        let o = SummaryOptions::default();
        for mode in 0..7 {
            let mut value = raw("Alice Alice", VALID, o);
            match mode { 0 => value.source_fields.clear(),
                1 => value.source_fields[0].json_pointer = "/0/text".to_owned(),
                2 => value.source_fields.push(value.source_fields[0].clone()),
                3 => value.source_fields[0].spans.clear(),
                4 => value.source_fields[0].occurrence = SourceOccurrence::Anchored,
                5 => value.source_fields[0].spans[0].scalar_end += 1,
                _ => value.source_fields[0].spans.reverse() }
            assert!(matches!(finalize(value, o, 8192), Err(SummaryError::InvalidResult)));
        }
    }
    #[test]
    fn finalizer_independently_enforces_every_output_cap() {
        let o = SummaryOptions::default();
        let json = r#"[{"citations":["Alice","Bob"],"text":"Two people met."},{"citations":["Bob"],"text":"Bob is named."}]"#;
        for axis in 0..4 {
            let mut narrow = o;
            match axis { 0 => narrow.max_bullets = 1, 1 => narrow.max_bullet_scalars = 1,
                2 => narrow.max_citations_per_bullet = 1, _ => narrow.max_quote_scalars = 1 }
            assert!(finalize(raw("Alice Bob", json, o), narrow, 16384).is_err());
        }
    }
    #[test]
    fn wrong_task_unbound_source_and_duplicate_keys_are_not_accepted() {
        let o = SummaryOptions::default();
        for mode in 0..4 {
            let mut value = raw("Alice", VALID, o);
            match mode { 0 => value.task_spec_version = "extract-v1".to_owned(),
                1 => value.schema_version = 1,
                2 => value.grounding = ExtractionGrounding::NotRequested,
                _ => value.output.json = r#"[{"citations":["Alice"],"text":"a","text":"b"}]"#.to_owned() }
            assert!(matches!(finalize(value, o, 8192), Err(SummaryError::InvalidResult)));
        }
    }
    #[test]
    fn complete_summary_envelope_has_an_exact_byte_limit() {
        let o = SummaryOptions::default();
        let result = finalize(raw("Alice", VALID, o), o, 8192).unwrap();
        let bytes = canonjson::canonical_bytes(&result).unwrap().len() as u64;
        assert!(finalize(raw("Alice", VALID, o), o, bytes).is_ok());
        assert!(matches!(finalize(raw("Alice", VALID, o), o, bytes - 1), Err(SummaryError::OutputBudgetExceeded)));
        assert!(finalize(raw("plain", "[]", o), o, 8192).unwrap().bullets.is_empty());
    }
    #[test]
    fn replay_is_canonical_and_cancellation_is_not_abstention() {
        let o = SummaryOptions::default();
        let a = finalize(raw("Alice", VALID, o), o, 8192).unwrap();
        let b = finalize(raw("Alice", VALID, o), o, 8192).unwrap();
        assert_eq!(canonjson::canonical_bytes(&a).unwrap(), canonjson::canonical_bytes(&b).unwrap());
        use crate::native_engine::{constrained::JsonDecodeError, decode::DecodeCancellationKind};
        let error: SummaryError = ExtractError::Decode(JsonDecodeError::Cancelled(DecodeCancellationKind::Deadline)).into();
        assert!(matches!(error, SummaryError::Extraction(ExtractError::Decode(
            JsonDecodeError::Cancelled(DecodeCancellationKind::Deadline)))));
    }
}
