//! Native context-limited QA over supplied passages; retrieval is not performed.
//!
//! The question and passage manifest bind private prompt identity but are not
//! evidence. Citations must occur wholly inside an original passage, never in
//! a synthetic join. Source membership is structural, not semantic support.
//! Model-declared abstention is explicitly UNCALIBRATED in this task version.

use std::{collections::BTreeSet, error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{
    canonjson,
    execution_identity::ExecutionIdentity,
    grammar::{CompileLimits, runtime::SourceRuntimeLimits},
    native_engine::{constrained::{JsonDecodeOptions, JsonWorkBudget},
        decode::DecodeStepControl, hf_bf16_eager::{HfBf16EagerEngine, HF_BF16_EAGER_PROFILE}},
    tokenizer::specials::TemplateControlIds,
    validation::{JsonLimits, JsonValue, parse_json_with_limits,
        grounded_fields::{SourceFieldEvidence, SourceOccurrence, VerifiedSourceSpan}},
};
use super::{
    extract::{ExtractError, ExtractPlan, ExtractResult, ExtractionGrounding, ExtractionVocabulary,
        SourceDocument, SourceDocumentEncoder},
    ir::{ScoreSpace, TaskPlan},
    summarize::{CitationGuarantee, SummarySemanticSupport},
};

pub const ANSWER_TASK_VERSION: &str = "answer-v1";
pub const ANSWER_PASSAGE_LAYOUT_VERSION: &str = "ordered-exact-passages-double-newline-v1";
const JOIN: &str = "\n\n";
const MAX_BYTES: usize = 64 * 1024 * 1024;

/// Supplied retrieval data. Neither passage ids nor text are trusted prose.
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerPassage { pub id: String, pub text: String }

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerInputLimits {
    pub max_passages: usize,
    /// Aggregate question, joined passage, and canonical manifest bytes.
    pub max_input_bytes: usize,
    /// Aggregate untrusted tokens, excluding trusted template scaffolding.
    pub max_input_tokens: usize,
}
impl Default for AnswerInputLimits {
    fn default() -> Self {
        Self { max_passages: 32, max_input_bytes: 1024 * 1024, max_input_tokens: 8192 }
    }
}
impl AnswerInputLimits {
    pub fn validate(self) -> Result<(), AnswerError> {
        if !(1..=1024).contains(&self.max_passages)
            || !(1..=MAX_BYTES).contains(&self.max_input_bytes)
            || !(1..=MAX_BYTES).contains(&self.max_input_tokens)
        { return Err(AnswerError::InvalidOptions); }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerOptions {
    pub max_answer_scalars: usize,
    pub max_citations: usize,
    pub max_quote_scalars: usize,
}
impl Default for AnswerOptions {
    fn default() -> Self { Self { max_answer_scalars: 2048, max_citations: 8, max_quote_scalars: 256 } }
}
impl AnswerOptions {
    pub fn validate(self) -> Result<(), AnswerError> {
        if !(1..=65_536).contains(&self.max_answer_scalars) || !(1..=128).contains(&self.max_citations)
            || !(1..=4096).contains(&self.max_quote_scalars)
        { return Err(AnswerError::InvalidOptions); }
        Ok(())
    }
    /// Only validated numeric limits enter this code-owned schema. Caller
    /// question/text/ids are never interpolated into trusted schema prose.
    pub fn schema_source(self) -> Result<String, AnswerError> {
        self.validate()?;
        canonjson::canonical_string(&serde_json::json!({
            "type": "object", "additionalProperties": false,
            "required": ["answer", "answerable", "citations"],
            "properties": {
                "answer": {"type": "string", "maxLength": self.max_answer_scalars},
                "answerable": {"type": "boolean"},
                "citations": {"type": "array", "maxItems": self.max_citations,
                    "items": {"type": "string", "maxLength": self.max_quote_scalars, "x-fnlp-source": "verbatim"}}
            }
        })).map_err(|_| AnswerError::Serialization)
    }
}

#[derive(Serialize)]
struct PassageLayout { id: String, span: VerifiedSourceSpan }

/// A sealed question/passage binding. The two context documents are the
/// question and the exact canonical passage manifest, followed by the source.
/// The manifest prevents distinct passage partitions/ids over identical joined
/// source bytes from sharing a TaskIR/prompt identity. No raw content digest is
/// exported, and no duplicate source authority can be deserialized into this.
pub struct AnswerContext {
    document: SourceDocument,
    passages: Vec<PassageLayout>,
}
impl AnswerContext {
    pub fn encode(encoder: &SourceDocumentEncoder, question: &str, passages: &[AnswerPassage],
        limits: AnswerInputLimits) -> Result<Self, AnswerError> {
        limits.validate()?;
        if passages.is_empty() { return Err(AnswerError::MissingPassages); }
        if passages.len() > limits.max_passages || !question.chars().any(|c| !c.is_whitespace()) {
            return Err(AnswerError::InvalidInput);
        }
        let mut ids = BTreeSet::new();
        let mut joined_bytes = JOIN.len().checked_mul(passages.len() - 1).ok_or(AnswerError::InputBudgetExceeded)?;
        let mut id_bytes = 0_usize;
        for passage in passages {
            if passage.id.is_empty() || passage.id.len() > 128 || !ids.insert(passage.id.as_str())
                || !passage.text.chars().any(|c| !c.is_whitespace())
            { return Err(AnswerError::InvalidInput); }
            joined_bytes = joined_bytes.checked_add(passage.text.len()).ok_or(AnswerError::InputBudgetExceeded)?;
            id_bytes = id_bytes.checked_add(passage.id.len()).ok_or(AnswerError::InputBudgetExceeded)?;
        }
        let minimum = joined_bytes.checked_add(question.len()).and_then(|n| n.checked_add(id_bytes))
            .ok_or(AnswerError::InputBudgetExceeded)?;
        if minimum > limits.max_input_bytes || minimum > limits.max_input_tokens {
            return Err(AnswerError::InputBudgetExceeded);
        }
        let mut source = String::new();
        source.try_reserve_exact(joined_bytes).map_err(|_| AnswerError::AllocationRefused)?;
        let mut layout = Vec::new();
        layout.try_reserve_exact(passages.len()).map_err(|_| AnswerError::AllocationRefused)?;
        let mut scalar = 0_usize;
        for (index, passage) in passages.iter().enumerate() {
            if index != 0 { source.push_str(JOIN); scalar += JOIN.len(); }
            let byte_start = source.len(); let scalar_start = scalar;
            source.push_str(&passage.text); scalar += passage.text.chars().count();
            layout.push(PassageLayout { id: copy_text(&passage.id)?,
                span: VerifiedSourceSpan { byte_start, byte_end: source.len(), scalar_start, scalar_end: scalar } });
        }
        let manifest = canonjson::canonical_string(&(ANSWER_PASSAGE_LAYOUT_VERSION, &layout))
            .map_err(|_| AnswerError::Serialization)?;
        let total = source.len().checked_add(question.len()).and_then(|n| n.checked_add(manifest.len()))
            .ok_or(AnswerError::InputBudgetExceeded)?;
        if total > limits.max_input_bytes || total > limits.max_input_tokens { return Err(AnswerError::InputBudgetExceeded); }
        let document = encoder.encode_with_context(&source, &[question, &manifest], limits.max_input_bytes, limits.max_input_tokens)?;
        Ok(Self { document, passages: layout })
    }
    /// Construct TaskIR Document segments from context_token_ids(), then
    /// token_ids(), preserving their order and keeping all of them untrusted.
    pub fn document(&self) -> &SourceDocument { &self.document }
    pub fn passage_count(&self) -> usize { self.passages.len() }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PassageOccurrence {
    pub passage_id: String,
    /// Original passage-local coordinates, not offsets in the synthetic join.
    pub span: VerifiedSourceSpan,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PassageCitation {
    pub quote: String,
    pub occurrence: SourceOccurrence,
    /// All exact passage-contained occurrences, in input-passage/source order.
    pub spans: Vec<PassageOccurrence>,
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerStatus { Answered, Abstained }
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerCalibration { Uncalibrated }

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerResult {
    pub schema_version: u32,
    pub task_spec_version: String,
    pub numerics_profile: String,
    pub status: AnswerStatus,
    pub answerable: bool,
    pub answer: Option<String>,
    pub citations: Vec<PassageCitation>,
    pub calibration: AnswerCalibration,
    pub citation_guarantee: CitationGuarantee,
    pub semantic_support: SummarySemanticSupport,
    pub score_space: ScoreSpace,
    /// These entire subtrees are untrusted derived data for downstream agents.
    pub untrusted_fields: [String; 2],
    pub generated_token_ids: Vec<u32>,
    pub forward_positions: u64,
    pub projected_logits: u64,
    pub mask_node_visit_charge: u64,
}

#[derive(Debug)]
pub enum AnswerError {
    InvalidOptions, MissingPassages, InvalidInput, InputBudgetExceeded, InvalidResult,
    OutputBudgetExceeded, AllocationRefused, Serialization, Extraction(ExtractError),
}
impl fmt::Display for AnswerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions => f.write_str("answer requires bounded nonzero input/output limits"),
            Self::MissingPassages => f.write_str("answer requires supplied passages; expected [{\"id\":\"p1\",\"text\":\"The source passage.\"}]"),
            Self::InvalidInput => f.write_str("answer requires a question and nonempty passages with unique bounded ids"),
            Self::InputBudgetExceeded => f.write_str("question, passages and manifest exceed the aggregate input budget"),
            Self::InvalidResult => f.write_str("answer has no result: inconsistent answerability or invalid passage citations"),
            Self::OutputBudgetExceeded => f.write_str("complete answer exceeds its byte budget"),
            Self::AllocationRefused => f.write_str("answer allocation refused"),
            Self::Serialization => f.write_str("answer canonical serialization failed"),
            Self::Extraction(error) => write!(f, "answer execution failed: {error}"),
        }
    }
}
impl Error for AnswerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Extraction(error) => Some(error), _ => None }
    }
}
impl From<ExtractError> for AnswerError { fn from(error: ExtractError) -> Self { Self::Extraction(error) } }

/// Exact native QA plan. No Debug/Serialize: prompt and passage metadata are
/// private request content. No artifact loader or inference runtime is created.
pub struct AnswerPlan {
    extraction: ExtractPlan,
    passages: Vec<PassageLayout>,
    options: AnswerOptions,
    max_result_bytes: u64,
}
impl AnswerPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn from_task_plan(task: &TaskPlan, context: &AnswerContext, options: AnswerOptions,
        decode: JsonDecodeOptions, compiler: CompileLimits, controls: &TemplateControlIds,
        source_limits: SourceRuntimeLimits) -> Result<Self, AnswerError> {
        let schema = options.schema_source()?;
        let extraction = ExtractPlan::from_builtin(task, &schema, decode, compiler, controls,
            Some((&context.document, source_limits)), ANSWER_TASK_VERSION)?;
        let mut passages = Vec::new();
        passages.try_reserve_exact(context.passages.len()).map_err(|_| AnswerError::AllocationRefused)?;
        for passage in &context.passages {
            passages.push(PassageLayout { id: copy_text(&passage.id)?, span: passage.span });
        }
        Ok(Self { extraction, passages, options, max_result_bytes: task.ir().budget().max_output_bytes })
    }
    pub fn bind_identity(&self, identity: ExecutionIdentity) -> Result<ExecutionIdentity, AnswerError> {
        self.extraction.bind_identity(identity).map_err(Into::into)
    }
    pub fn verify_identity(&self, identity: &ExecutionIdentity) -> Result<(), AnswerError> {
        self.extraction.verify_identity(identity).map_err(Into::into)
    }
    pub fn execute_eager<C: DecodeStepControl>(&self, engine: &mut HfBf16EagerEngine,
        identity: &ExecutionIdentity, vocabulary: &ExtractionVocabulary, work: JsonWorkBudget,
        control: &mut C) -> Result<AnswerResult, AnswerError> {
        let raw = self.extraction.execute_eager(engine, identity, vocabulary, work, control)?;
        finalize(raw, &self.passages, self.options, self.max_result_bytes)
    }
}

fn copy_text(text: &str) -> Result<String, AnswerError> {
    let mut copy = String::new();
    copy.try_reserve_exact(text.len()).map_err(|_| AnswerError::AllocationRefused)?;
    copy.push_str(text); Ok(copy)
}

// Shared extraction independently proves complete exact-source occurrences.
// Only private passage boundaries are used to project them back to originals.
fn citation(quote: String, pointer: &str, proof: SourceFieldEvidence, passages: &[PassageLayout],
    max_scalars: usize) -> Result<PassageCitation, AnswerError> {
    let scalars = quote.chars().count();
    if !quote.chars().any(|c| !c.is_whitespace()) || scalars > max_scalars
        || proof.json_pointer != pointer || proof.spans.is_empty()
        || proof.occurrence != if proof.spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }
        || proof.spans.windows(2).any(|p| p[0].byte_start >= p[1].byte_start || p[0].scalar_start >= p[1].scalar_start)
    { return Err(AnswerError::InvalidResult); }
    let mut spans = Vec::new();
    spans.try_reserve_exact(proof.spans.len()).map_err(|_| AnswerError::AllocationRefused)?;
    for span in proof.spans {
        if span.byte_end.checked_sub(span.byte_start) != Some(quote.len())
            || span.scalar_end.checked_sub(span.scalar_start) != Some(scalars)
        { return Err(AnswerError::InvalidResult); }
        let position = passages.partition_point(|p| p.span.byte_start <= span.byte_start);
        let Some(passage) = position.checked_sub(1).and_then(|p| passages.get(p)) else {
            return Err(AnswerError::InvalidResult);
        };
        if span.byte_start >= passage.span.byte_end || span.byte_end > passage.span.byte_end {
            // A match in a separator or crossing a join is not an original
            // passage occurrence. It is NEVER published as citation evidence.
            continue;
        }
        if span.scalar_start < passage.span.scalar_start || span.scalar_end > passage.span.scalar_end {
            return Err(AnswerError::InvalidResult);
        }
        spans.push(PassageOccurrence { passage_id: copy_text(&passage.id)?, span: VerifiedSourceSpan {
            byte_start: span.byte_start - passage.span.byte_start, byte_end: span.byte_end - passage.span.byte_start,
            scalar_start: span.scalar_start - passage.span.scalar_start, scalar_end: span.scalar_end - passage.span.scalar_start,
        } });
    }
    if spans.is_empty() { return Err(AnswerError::InvalidResult); }
    Ok(PassageCitation { quote, occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans })
}

// No public deserialize-to-authority entrypoint: this consumes only the native
// decoder's independently verified ExtractResult in production.
fn finalize(raw: ExtractResult, passages: &[PassageLayout], options: AnswerOptions,
    max_bytes: u64) -> Result<AnswerResult, AnswerError> {
    options.validate()?;
    if passages.is_empty() || raw.schema_version != 2 || raw.task_spec_version != ANSWER_TASK_VERSION
        || raw.output.schema_version != 1 || raw.output.numerics_profile != HF_BF16_EAGER_PROFILE
        || raw.score_space != ScoreSpace::NotComputed || raw.grounding != ExtractionGrounding::SourceMembership
    { return Err(AnswerError::InvalidResult); }
    let cap = usize::try_from(max_bytes).unwrap_or(usize::MAX);
    let value = parse_json_with_limits(&raw.output.json, JsonLimits {
        max_input_bytes: cap, max_string_lexeme_bytes: cap, max_depth: 4,
        max_container_entries: options.max_citations.max(3), ..JsonLimits::default()
    }).map_err(|_| AnswerError::InvalidResult)?;
    let JsonValue::Object(mut fields) = value else { return Err(AnswerError::InvalidResult); };
    if fields.len() != 3 { return Err(AnswerError::InvalidResult); }
    let Some(JsonValue::Boolean(answerable)) = fields.remove("answerable") else { return Err(AnswerError::InvalidResult); };
    let Some(JsonValue::String(answer)) = fields.remove("answer") else { return Err(AnswerError::InvalidResult); };
    let Some(JsonValue::Array(quotes)) = fields.remove("citations") else { return Err(AnswerError::InvalidResult); };
    if answer.chars().count() > options.max_answer_scalars || quotes.len() > options.max_citations
        || (answerable && (!answer.chars().any(|c| !c.is_whitespace()) || quotes.is_empty()))
        || (!answerable && (!answer.is_empty() || !quotes.is_empty()))
    { return Err(AnswerError::InvalidResult); }
    let mut proofs = raw.source_fields.into_iter(); let mut citations = Vec::new();
    citations.try_reserve_exact(quotes.len()).map_err(|_| AnswerError::AllocationRefused)?;
    for (index, value) in quotes.into_iter().enumerate() {
        let JsonValue::String(quote) = value else { return Err(AnswerError::InvalidResult); };
        citations.push(citation(quote, &format!("/citations/{index}"),
            proofs.next().ok_or(AnswerError::InvalidResult)?, passages, options.max_quote_scalars)?);
    }
    if proofs.next().is_some() { return Err(AnswerError::InvalidResult); }
    let result = AnswerResult {
        schema_version: 1, task_spec_version: ANSWER_TASK_VERSION.to_owned(), numerics_profile: raw.output.numerics_profile,
        status: if answerable { AnswerStatus::Answered } else { AnswerStatus::Abstained }, answerable,
        answer: answerable.then_some(answer), citations, calibration: AnswerCalibration::Uncalibrated,
        citation_guarantee: CitationGuarantee::StructuralSourceMembership, semantic_support: SummarySemanticSupport::NotAssessed,
        score_space: ScoreSpace::NotComputed, untrusted_fields: ["answer".to_owned(), "citations".to_owned()],
        generated_token_ids: raw.output.token_ids, forward_positions: raw.output.forward_positions,
        projected_logits: raw.output.projected_logits, mask_node_visit_charge: raw.output.mask_node_visit_charge,
    };
    if canonjson::canonical_bytes(&result).map_err(|_| AnswerError::Serialization)?.len() as u64 > max_bytes {
        return Err(AnswerError::OutputBudgetExceeded);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammar::runtime::JsonProgram, native_engine::constrained::JsonDecodeOutput,
        validation::{SourceSpan, validate_source_span}, tokenizer::specials::ArchivedControlRegistries};
    fn encoder() -> SourceDocumentEncoder {
        let registry = ArchivedControlRegistries::from_archived_json(
            r#"{"schema_version":1,"registry":"TokenizerSpecialIds","entries":[{"id":0,"special":true,"surface":"<eos>"}]}"#,
            r#"{"schema_version":1,"registry":"TemplateControlIds","entries":[{"id":0,"special":true,"surface":"<eos>"},{"id":3,"special":false,"surface":"<think>"}]}"#,
        ).unwrap();
        SourceDocumentEncoder::pinned(registry.template_controls()).unwrap()
    }
    fn options() -> AnswerOptions { AnswerOptions { max_answer_scalars: 128, max_citations: 4, max_quote_scalars: 32 } }
    fn passages(values: &[(&str, &str)]) -> Vec<AnswerPassage> {
        values.iter().map(|(id, text)| AnswerPassage { id: (*id).to_owned(), text: (*text).to_owned() }).collect()
    }
    fn context(values: &[(&str, &str)]) -> AnswerContext {
        AnswerContext::encode(&encoder(), "Who is named?", &passages(values), AnswerInputLimits::default()).unwrap()
    }
    fn raw(context: &AnswerContext, json: &str) -> ExtractResult {
        let program = JsonProgram::compile_with_source(&options().schema_source().unwrap(), context.document.text(),
            CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
        ExtractResult { schema_version: 2, task_spec_version: ANSWER_TASK_VERSION.to_owned(),
            score_space: ScoreSpace::NotComputed, grounding: ExtractionGrounding::SourceMembership,
            source_fields: program.source_fields(json).unwrap(), output: JsonDecodeOutput {
                schema_version: 1, numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(), token_ids: vec![1, 2, 0],
                json: json.to_owned(), forward_positions: 8, projected_logits: 100, mask_node_visit_charge: 20,
            } }
    }
    const ANSWER: &str = r#"{"answer":"Alice is named.","answerable":true,"citations":["Alice"]}"#;
    const ABSTAIN: &str = r#"{"answer":"","answerable":false,"citations":[]}"#;
    #[test]
    fn answered_and_abstained_are_distinct_successful_results() {
        let c = context(&[("p1", "Alice")]);
        let answer = finalize(raw(&c, ANSWER), &c.passages, options(), 16384).unwrap();
        assert!(answer.answerable); assert_eq!(answer.status, AnswerStatus::Answered);
        assert_eq!(answer.calibration, AnswerCalibration::Uncalibrated);
        assert_eq!(answer.semantic_support, SummarySemanticSupport::NotAssessed);
        let abstain = finalize(raw(&c, ABSTAIN), &c.passages, options(), 16384).unwrap();
        assert!(!abstain.answerable); assert_eq!(abstain.status, AnswerStatus::Abstained);
        assert!(abstain.answer.is_none()); assert!(abstain.citations.is_empty());
    }
    #[test]
    fn answerability_cannot_hide_uncited_or_partial_answers() {
        let c = context(&[("p1", "Alice")]);
        for json in [r#"{"answer":"","answerable":true,"citations":["Alice"]}"#,
            r#"{"answer":"Alice","answerable":true,"citations":[]}"#,
            r#"{"answer":"Alice","answerable":false,"citations":[]}"#,
            r#"{"answer":"","answerable":false,"citations":["Alice"]}"#,
            r#"{"answer":" \t","answerable":true,"citations":["Alice"]}"#] {
            assert!(matches!(finalize(raw(&c, json), &c.passages, options(), 16384), Err(AnswerError::InvalidResult)));
        }
    }
    #[test]
    fn absent_empty_duplicate_and_unbounded_passages_are_usage_errors() {
        let e = encoder(); let limits = AnswerInputLimits::default();
        assert!(matches!(AnswerContext::encode(&e, "Who?", &[], limits), Err(AnswerError::MissingPassages)));
        for p in [passages(&[("", "Alice")]), passages(&[("p", "")]),
            passages(&[("p", "Alice"), ("p", "Bob")]), passages(&[("p", " \n")])] {
            assert!(AnswerContext::encode(&e, "Who?", &p, limits).is_err());
        }
        assert!(AnswerContext::encode(&e, " \t", &passages(&[("p", "Alice")]), limits).is_err());
        assert!(AnswerContext::encode(&e, "Who?", &passages(&[("p", "Alice"), ("q", "Bob")]),
            AnswerInputLimits { max_passages: 1, ..limits }).is_err());
    }
    #[test]
    fn metadata_and_question_are_not_part_of_the_evidence_language() {
        let c = AnswerContext::encode(&encoder(), "Carol?", &passages(&[("Bob", "Alice")]), AnswerInputLimits::default()).unwrap();
        assert_eq!(c.document.context_token_ids().len(), 2);
        assert_eq!(c.document.text(), "Alice");
        let program = JsonProgram::compile_with_source(&options().schema_source().unwrap(), c.document.text(),
            CompileLimits::default(), SourceRuntimeLimits::default()).unwrap();
        for quote in ["Bob", "Carol"] {
            let json = serde_json::json!({"answer":"A name.","answerable":true,"citations":[quote]}).to_string();
            assert!(program.source_fields(&json).is_err());
        }
    }
    #[test]
    fn identical_joined_text_different_passage_identity_has_different_manifest_tokens() {
        let a = context(&[("a", "Alice"), ("b", "Bob")]);
        let b = context(&[("a", "Alice\n\nBob")]);
        let c = context(&[("renamed", "Alice"), ("b", "Bob")]);
        assert_eq!(a.document.token_ids(), b.document.token_ids());
        assert_eq!(a.document.token_ids(), c.document.token_ids());
        assert_ne!(a.document.context_token_ids().nth(1), b.document.context_token_ids().nth(1));
        assert_ne!(a.document.context_token_ids().nth(1), c.document.context_token_ids().nth(1));
    }
    #[test]
    fn original_unicode_offsets_and_every_ambiguous_occurrence_are_preserved() {
        let p = passages(&[("one", "é上海 上海 ababa"), ("two", "上海 ababa")]);
        let c = AnswerContext::encode(&encoder(), "What is named?", &p, AnswerInputLimits::default()).unwrap();
        let json = r#"{"answer":"A location.","answerable":true,"citations":["上海","aba"]}"#;
        let result = finalize(raw(&c, json), &c.passages, options(), 16384).unwrap();
        assert_eq!(result.citations[0].spans.len(), 3); assert_eq!(result.citations[1].spans.len(), 4);
        for citation in result.citations {
            assert_eq!(citation.occurrence, SourceOccurrence::Ambiguous);
            for occurrence in citation.spans {
                let original = p.iter().find(|p| p.id == occurrence.passage_id).unwrap(); let s = occurrence.span;
                validate_source_span(&original.text, &citation.quote,
                    SourceSpan::new(s.byte_start, s.byte_end, s.scalar_start, s.scalar_end)).unwrap();
            }
        }
    }
    #[test]
    fn fabricated_cross_passage_quotes_are_never_published() {
        let c = context(&[("a", "ab"), ("b", "cd")]);
        let json = r#"{"answer":"A claim.","answerable":true,"citations":["b\n\nc"]}"#;
        assert!(matches!(finalize(raw(&c, json), &c.passages, options(), 16384), Err(AnswerError::InvalidResult)));
        let c = context(&[("real", "b\n\nc"), ("a", "b"), ("b", "c")]);
        let result = finalize(raw(&c, json), &c.passages, options(), 16384).unwrap();
        assert_eq!(result.citations[0].spans.len(), 1);
        assert_eq!(result.citations[0].spans[0].passage_id, "real");
        assert_eq!(result.citations[0].occurrence, SourceOccurrence::Anchored);
    }
    #[test]
    fn input_budget_includes_question_manifest_and_source() {
        let p = passages(&[("p", "Alice")]); let c = context(&[("p", "Alice")]);
        let exact = c.document.total_token_count(); let e = encoder();
        let limits = AnswerInputLimits { max_input_bytes: exact, max_input_tokens: exact, ..AnswerInputLimits::default() };
        assert!(AnswerContext::encode(&e, "Who is named?", &p, limits).is_ok());
        assert!(AnswerContext::encode(&e, "Who is named?", &p, AnswerInputLimits { max_input_bytes: exact - 1, ..limits }).is_err());
        assert!(AnswerContext::encode(&e, "Who is named?", &p, AnswerInputLimits { max_input_tokens: exact - 1, ..limits }).is_err());
    }
    #[test]
    fn missing_swapped_extra_and_corrupt_proofs_refuse() {
        let c = context(&[("p", "Alice Alice")]);
        for mode in 0..7 {
            let mut value = raw(&c, ANSWER);
            match mode { 0 => value.source_fields.clear(), 1 => value.source_fields[0].json_pointer = "/answer".to_owned(),
                2 => value.source_fields.push(value.source_fields[0].clone()), 3 => value.source_fields[0].spans.clear(),
                4 => value.source_fields[0].spans[0].scalar_end += 1,
                5 => value.source_fields[0].spans.reverse(), _ => value.source_fields[0].occurrence = SourceOccurrence::Anchored }
            assert!(finalize(value, &c.passages, options(), 16384).is_err());
        }
    }
    #[test]
    fn schema_and_finalizer_both_bound_answer_and_citations() {
        let c = context(&[("p", "Alice Bob")]); let o = options();
        for axis in 0..3 {
            let mut narrow = o;
            match axis { 0 => narrow.max_answer_scalars = 1, 1 => narrow.max_citations = 1, _ => narrow.max_quote_scalars = 1 }
            assert_ne!(o.schema_source().unwrap(), narrow.schema_source().unwrap());
            let json = r#"{"answer":"Two names.","answerable":true,"citations":["Alice","Bob"]}"#;
            assert!(finalize(raw(&c, json), &c.passages, narrow, 16384).is_err());
        }
        assert!(AnswerOptions { max_citations: 0, ..o }.schema_source().is_err());
    }
    #[test]
    fn whole_result_not_only_generated_json_is_byte_bounded() {
        let c = context(&[("p", "Alice")]);
        let a = finalize(raw(&c, ANSWER), &c.passages, options(), 16384).unwrap();
        let exact = canonjson::canonical_bytes(&a).unwrap().len() as u64;
        assert!(finalize(raw(&c, ANSWER), &c.passages, options(), exact).is_ok());
        assert!(matches!(finalize(raw(&c, ANSWER), &c.passages, options(), exact - 1), Err(AnswerError::OutputBudgetExceeded)));
        let b = finalize(raw(&c, ANSWER), &c.passages, options(), 16384).unwrap();
        assert_eq!(canonjson::canonical_bytes(&a).unwrap(), canonjson::canonical_bytes(&b).unwrap());
        let json = canonjson::canonical_string(&a).unwrap();
        for absent in ["confidence", "prompt_digest", "document_digest"] { assert!(!json.contains(absent)); }
    }
    #[test]
    fn wrong_task_duplicate_keys_and_cancellation_cannot_become_abstention() {
        let c = context(&[("p", "Alice")]);
        for mode in 0..3 {
            let mut value = raw(&c, ABSTAIN);
            match mode { 0 => value.task_spec_version = "summarize-v1".to_owned(),
                1 => value.grounding = ExtractionGrounding::NotRequested,
                _ => value.output.json = r#"{"answer":"","answerable":true,"answerable":false,"citations":[]}"#.to_owned() }
            assert!(finalize(value, &c.passages, options(), 16384).is_err());
        }
        use crate::native_engine::{constrained::JsonDecodeError, decode::DecodeCancellationKind};
        let error: AnswerError = ExtractError::Decode(JsonDecodeError::Cancelled(DecodeCancellationKind::Deadline)).into();
        assert!(matches!(error, AnswerError::Extraction(ExtractError::Decode(JsonDecodeError::Cancelled(DecodeCancellationKind::Deadline)))));
    }
}
