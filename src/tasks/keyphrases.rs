//! Ranked, byte-exact keyphrases through the shared native source decoder.
//!
//! Ranking is the model's proposal order, not a probability or calibrated
//! relevance score. Repeated proposals keep their first rank. Every occurrence
//! of each retained phrase is preserved; ambiguous text gets no invented offset.

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
    extract::{ExtractError, ExtractPlan, ExtractResult, ExtractionGrounding,
        ExtractionVocabulary, SourceDocument},
    ir::{ScoreSpace, TaskPlan},
};

pub const KEYPHRASES_TASK_VERSION: &str = "keyphrases-v1";
pub const KEYPHRASES_RANKING: &str = "model-order-first-exact-v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KeyphraseOptions {
    /// Maximum model proposals. Deduplication may return fewer phrases.
    pub max_phrases: usize,
    /// Unicode scalar values, not bytes, tokens, or UTF-16 code units.
    pub max_phrase_scalars: usize,
}
impl Default for KeyphraseOptions {
    fn default() -> Self { Self { max_phrases: 16, max_phrase_scalars: 256 } }
}
impl KeyphraseOptions {
    pub fn validate(self) -> Result<(), KeyphraseError> {
        if !(1..=4096).contains(&self.max_phrases)
            || !(1..=4096).contains(&self.max_phrase_scalars)
        { return Err(KeyphraseError::InvalidOptions); }
        Ok(())
    }
    /// Put these exact bytes in the trusted instruction and grammar identity.
    /// The source runtime excludes off-document strings during decoding.
    pub fn schema_source(self) -> Result<String, KeyphraseError> {
        self.validate()?;
        #[derive(Serialize)]
        struct PhraseSchema {
            #[serde(rename = "type")] kind: &'static str,
            #[serde(rename = "maxLength")] max_length: usize,
            #[serde(rename = "x-fnlp-source")] source: &'static str,
        }
        #[derive(Serialize)]
        struct ListSchema {
            #[serde(rename = "type")] kind: &'static str,
            #[serde(rename = "maxItems")] max_items: usize,
            items: PhraseSchema,
        }
        canonjson::canonical_string(&ListSchema {
            kind: "array", max_items: self.max_phrases,
            items: PhraseSchema { kind: "string", max_length: self.max_phrase_scalars, source: "verbatim" },
        }).map_err(|_| KeyphraseError::Serialization)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RankedKeyphrase {
    /// One-based, contiguous rank after exact duplicate removal.
    pub rank: usize,
    pub text: String,
    pub occurrence: SourceOccurrence,
    pub spans: Vec<VerifiedSourceSpan>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KeyphraseResult {
    pub schema_version: u32,
    pub task_spec_version: String,
    pub numerics_profile: String,
    pub ranking_policy: String,
    pub score_space: ScoreSpace,
    pub grounding: ExtractionGrounding,
    pub phrases: Vec<RankedKeyphrase>,
    pub generated_token_ids: Vec<u32>,
    pub forward_positions: u64,
    pub projected_logits: u64,
    pub mask_node_visit_charge: u64,
}

#[derive(Debug)]
pub enum KeyphraseError {
    InvalidOptions,
    InvalidResult,
    OutputBudgetExceeded,
    AllocationRefused,
    Serialization,
    Extraction(ExtractError),
}
impl fmt::Display for KeyphraseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions => f.write_str("keyphrases require bounded nonzero phrase/count limits"),
            Self::InvalidResult => f.write_str("keyphrases have no result: invalid phrases or source evidence"),
            Self::OutputBudgetExceeded => f.write_str("complete keyphrase result exceeds its byte budget"),
            Self::AllocationRefused => f.write_str("keyphrase allocation refused"),
            Self::Serialization => f.write_str("keyphrase serialization failed"),
            Self::Extraction(error) => write!(f, "keyphrase extraction failed: {error}"),
        }
    }
}
impl Error for KeyphraseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Extraction(error) => Some(error), _ => None }
    }
}
impl From<ExtractError> for KeyphraseError {
    fn from(error: ExtractError) -> Self { Self::Extraction(error) }
}

/// No Debug or serialization: exact document and prompt identities stay private.
pub struct KeyphrasePlan {
    extraction: ExtractPlan,
    options: KeyphraseOptions,
    max_result_bytes: u64,
}
impl KeyphrasePlan {
    /// Requires a keyphrases-v1 constrained-json plan, exact source tokens,
    /// source-runtime schema identity, and SourceSpansVerified postcondition.
    /// The caller retains model, runtime, tokenizer and template admission.
    #[allow(clippy::too_many_arguments)]
    pub fn from_task_plan(
        task: &TaskPlan, document: &SourceDocument, options: KeyphraseOptions,
        decode: JsonDecodeOptions, compiler: CompileLimits,
        controls: &TemplateControlIds, source_limits: SourceRuntimeLimits,
    ) -> Result<Self, KeyphraseError> {
        let schema = options.schema_source()?;
        let extraction = ExtractPlan::from_builtin(task, &schema, decode, compiler,
            controls, Some((document, source_limits)), KEYPHRASES_TASK_VERSION)?;
        Ok(Self { extraction, options, max_result_bytes: task.ir().budget().max_output_bytes })
    }
    pub fn bind_identity(&self, identity: ExecutionIdentity) -> Result<ExecutionIdentity, KeyphraseError> {
        self.extraction.bind_identity(identity).map_err(Into::into)
    }
    pub fn verify_identity(&self, identity: &ExecutionIdentity) -> Result<(), KeyphraseError> {
        self.extraction.verify_identity(identity).map_err(Into::into)
    }
    pub fn execute_eager<C: DecodeStepControl>(
        &self, engine: &mut HfBf16EagerEngine, identity: &ExecutionIdentity,
        vocabulary: &ExtractionVocabulary, work: JsonWorkBudget, control: &mut C,
    ) -> Result<KeyphraseResult, KeyphraseError> {
        let raw = self.extraction.execute_eager(engine, identity, vocabulary, work, control)?;
        finalize(raw, self.options, self.max_result_bytes)
    }
}

// Only independently source-verified decoder output reaches this in production.
// Deserializing a KeyphraseResult never grants execution or evidence authority.
pub(super) fn finalize(raw: ExtractResult, options: KeyphraseOptions, max_bytes: u64) -> Result<KeyphraseResult, KeyphraseError> {
    options.validate()?;
    if raw.schema_version != 2 || raw.task_spec_version != KEYPHRASES_TASK_VERSION
        || raw.score_space != ScoreSpace::NotComputed || raw.grounding != ExtractionGrounding::SourceMembership
    { return Err(KeyphraseError::InvalidResult); }
    let cap = usize::try_from(max_bytes).unwrap_or(usize::MAX);
    let parsed = parse_json_with_limits(&raw.output.json, JsonLimits {
        max_input_bytes: cap, max_string_lexeme_bytes: cap,
        max_container_entries: options.max_phrases, max_depth: 2,
        ..JsonLimits::default()
    }).map_err(|_| KeyphraseError::InvalidResult)?;
    let JsonValue::Array(items) = parsed else { return Err(KeyphraseError::InvalidResult); };
    if items.len() > options.max_phrases { return Err(KeyphraseError::InvalidResult); }
    let mut evidence = raw.source_fields.into_iter();
    let mut seen = BTreeSet::new();
    let mut phrases = Vec::new();
    phrases.try_reserve_exact(items.len()).map_err(|_| KeyphraseError::AllocationRefused)?;
    for (index, value) in items.iter().enumerate() {
        let JsonValue::String(text) = value else { return Err(KeyphraseError::InvalidResult); };
        let scalars = text.chars().count();
        if text.is_empty() || scalars > options.max_phrase_scalars { return Err(KeyphraseError::InvalidResult); }
        let proof = evidence.next().ok_or(KeyphraseError::InvalidResult)?;
        if proof.json_pointer != format!("/{index}") || proof.spans.is_empty()
            || proof.occurrence != if proof.spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }
            || proof.spans.iter().any(|span| span.byte_end.checked_sub(span.byte_start) != Some(text.len())
                || span.scalar_end.checked_sub(span.scalar_start) != Some(scalars))
            || proof.spans.windows(2).any(|pair| pair[0].byte_start >= pair[1].byte_start
                || pair[0].scalar_start >= pair[1].scalar_start)
        { return Err(KeyphraseError::InvalidResult); }
        // Consume/check duplicate evidence too; a duplicate cannot hide a
        // malformed proof or shift the pointer of the following phrase.
        if seen.insert(text.as_str()) {
            let mut owned = String::new();
            owned.try_reserve_exact(text.len()).map_err(|_| KeyphraseError::AllocationRefused)?;
            owned.push_str(text);
            phrases.push(RankedKeyphrase { rank: phrases.len() + 1, text: owned,
                occurrence: proof.occurrence, spans: proof.spans });
        }
    }
    if evidence.next().is_some() { return Err(KeyphraseError::InvalidResult); }
    let result = KeyphraseResult {
        schema_version: 1, task_spec_version: KEYPHRASES_TASK_VERSION.to_owned(),
        numerics_profile: raw.output.numerics_profile, ranking_policy: KEYPHRASES_RANKING.to_owned(),
        score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
        phrases, generated_token_ids: raw.output.token_ids, forward_positions: raw.output.forward_positions,
        projected_logits: raw.output.projected_logits, mask_node_visit_charge: raw.output.mask_node_visit_charge,
    };
    if canonjson::canonical_bytes(&result).map_err(|_| KeyphraseError::Serialization)?.len() as u64 > max_bytes {
        return Err(KeyphraseError::OutputBudgetExceeded);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammar::runtime::{JsonProgram, SOURCE_JSON_RUNTIME_VERSION},
        native_engine::{constrained::JsonDecodeOutput, hf_bf16_eager::HF_BF16_EAGER_PROFILE},
        validation::{SourceSpan, validate_source_span}};

    fn raw(source: &str, json: &str, options: KeyphraseOptions) -> ExtractResult {
        let program = JsonProgram::compile_with_source(&options.schema_source().unwrap(), source,
            CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
        ExtractResult { schema_version: 2, task_spec_version: KEYPHRASES_TASK_VERSION.to_owned(),
            score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
            source_fields: program.source_fields(json).unwrap(),
            output: JsonDecodeOutput { schema_version: 1, numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(),
                token_ids: vec![1, 2, 0], json: json.to_owned(), forward_positions: 8,
                projected_logits: 100, mask_node_visit_charge: 20 },
        }
    }

    #[test]
    fn bounded_options_and_schema_identity() {
        let options = KeyphraseOptions::default();
        for invalid in [KeyphraseOptions { max_phrases: 0, ..options },
            KeyphraseOptions { max_phrase_scalars: 0, ..options },
            KeyphraseOptions { max_phrases: usize::MAX, ..options },
            KeyphraseOptions { max_phrase_scalars: usize::MAX, ..options }] {
            assert!(invalid.schema_source().is_err());
        }
        assert_ne!(options.schema_source().unwrap(),
            KeyphraseOptions { max_phrases: 2, ..options }.schema_source().unwrap());
        assert_ne!(options.schema_source().unwrap(),
            KeyphraseOptions { max_phrase_scalars: 2, ..options }.schema_source().unwrap());
    }
    #[test]
    fn runtime_masks_invented_phrases_before_generation() {
        let options = KeyphraseOptions::default();
        let program = JsonProgram::compile_with_source(&options.schema_source().unwrap(), "Rust compiler",
            CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
        assert_eq!(program.version(), SOURCE_JSON_RUNTIME_VERSION);
        let mut state = program.initial_state();
        assert!(state.consume_bytes(br#"["compiler","Rust"]"#));
        assert!(state.is_accepting());
        assert!(!program.initial_state().consume_bytes(br#"["Python"#));
    }
    #[test]
    fn ranking_preserves_model_order_not_lexical_order() {
        let o = KeyphraseOptions::default();
        let result = finalize(raw("Rust compiler", r#"["compiler","Rust"]"#, o), o, 8192).unwrap();
        assert_eq!(result.phrases.iter().map(|p| (p.rank, p.text.as_str())).collect::<Vec<_>>(),
            vec![(1, "compiler"), (2, "Rust")]);
        assert_eq!(result.ranking_policy, KEYPHRASES_RANKING);
    }
    #[test]
    fn exact_dedup_keeps_first_rank_and_all_occurrences() {
        let o = KeyphraseOptions::default();
        let result = finalize(raw("Rust rust Rust", r#"["Rust","rust","Rust"]"#, o), o, 8192).unwrap();
        assert_eq!(result.phrases.len(), 2);
        assert_eq!(result.phrases[0].occurrence, SourceOccurrence::Ambiguous);
        assert_eq!(result.phrases[0].spans.iter().map(|s| s.byte_start).collect::<Vec<_>>(), vec![0, 10]);
        assert_eq!(result.phrases[1].rank, 2);
        assert_eq!(result.phrases[1].text, "rust");
    }
    #[test]
    fn unicode_and_overlapping_occurrences_round_trip() {
        let o = KeyphraseOptions::default();
        let source = "é上海 and 上海 ababa";
        let result = finalize(raw(source, r#"["上海","aba"]"#, o), o, 16384).unwrap();
        assert_eq!(result.phrases[1].spans.len(), 2);
        for phrase in result.phrases { for span in phrase.spans {
            validate_source_span(source, &phrase.text, SourceSpan::new(span.byte_start, span.byte_end,
                span.scalar_start, span.scalar_end)).unwrap();
        } }
    }
    #[test]
    fn empty_list_is_valid_but_empty_phrase_is_not() {
        let o = KeyphraseOptions::default();
        assert!(finalize(raw("plain", "[]", o), o, 8192).unwrap().phrases.is_empty());
        assert!(matches!(finalize(raw("plain", r#"[""]"#, o), o, 8192), Err(KeyphraseError::InvalidResult)));
    }
    #[test]
    fn incomplete_and_corrupt_proofs_fail_closed() {
        let o = KeyphraseOptions::default();
        for corrupt in 0..7 {
            let mut value = raw("Rust Rust", r#"["Rust","Rust"]"#, o);
            match corrupt {
                0 => value.source_fields.clear(),
                1 => value.source_fields[0].json_pointer = "/1".to_owned(),
                2 => value.source_fields.push(value.source_fields[0].clone()),
                3 => value.source_fields[1].spans.clear(),
                4 => value.source_fields[0].occurrence = SourceOccurrence::Anchored,
                5 => value.source_fields[0].spans[0].scalar_end += 1,
                _ => value.source_fields[0].spans.reverse(),
            }
            assert!(matches!(finalize(value, o, 8192), Err(KeyphraseError::InvalidResult)));
        }
    }
    #[test]
    fn task_and_grounding_cannot_be_relabeled() {
        let o = KeyphraseOptions::default();
        for field in 0..3 {
            let mut value = raw("Rust", r#"["Rust"]"#, o);
            match field { 0 => value.schema_version = 1,
                1 => value.task_spec_version = "ner-v1".to_owned(),
                _ => value.grounding = ExtractionGrounding::NotRequested }
            assert!(matches!(finalize(value, o, 8192), Err(KeyphraseError::InvalidResult)));
        }
    }
    #[test]
    fn finalized_list_respects_options_independently() {
        let o = KeyphraseOptions::default();
        assert!(finalize(raw("Rust compiler", r#"["Rust","compiler"]"#, o),
            KeyphraseOptions { max_phrases: 1, ..o }, 8192).is_err());
        assert!(finalize(raw("Rust", r#"["Rust"]"#, o),
            KeyphraseOptions { max_phrase_scalars: 3, ..o }, 8192).is_err());
    }
    #[test]
    fn result_budget_includes_offsets_tokens_and_metadata() {
        let o = KeyphraseOptions::default();
        let result = finalize(raw("Rust", r#"["Rust"]"#, o), o, 8192).unwrap();
        let bytes = canonjson::canonical_bytes(&result).unwrap().len() as u64;
        assert!(finalize(raw("Rust", r#"["Rust"]"#, o), o, bytes).is_ok());
        assert!(matches!(finalize(raw("Rust", r#"["Rust"]"#, o), o, bytes - 1), Err(KeyphraseError::OutputBudgetExceeded)));
        let json = canonjson::canonical_string(&result).unwrap();
        for absent in ["confidence", "prompt_digest", "document_digest"] { assert!(!json.contains(absent)); }
    }
    #[test]
    fn replay_is_canonical_and_cancellation_keeps_its_cause() {
        let o = KeyphraseOptions::default();
        let a = finalize(raw("Rust", r#"["Rust"]"#, o), o, 8192).unwrap();
        let b = finalize(raw("Rust", r#"["Rust"]"#, o), o, 8192).unwrap();
        assert_eq!(canonjson::canonical_bytes(&a).unwrap(), canonjson::canonical_bytes(&b).unwrap());
        use crate::native_engine::{constrained::JsonDecodeError, decode::DecodeCancellationKind};
        let error: KeyphraseError = ExtractError::Decode(JsonDecodeError::Cancelled(DecodeCancellationKind::Deadline)).into();
        assert!(matches!(error, KeyphraseError::Extraction(ExtractError::Decode(
            JsonDecodeError::Cancelled(DecodeCancellationKind::Deadline)))));
    }
}
