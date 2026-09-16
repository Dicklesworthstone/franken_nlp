//! Pinned, byte-preserving source admission and source-bound extraction.
//! SourceDocument is not deserializable: its private token/text binding can
//! only be minted by the pinned encoder. Plans compare that exact sequence
//! with the TaskIR document segment before trusting any source occurrence.

use super::*;
use crate::{
    grammar::source_index::SOURCE_LANGUAGE_VERSION,
    tokenizer::untrusted::{UntrustedDocument, UntrustedDocumentEncoder},
};

/// Retain this model-free encoder across corpus items. It emits no privileged
/// template controls, even when their literal spellings occur in the document.
pub struct SourceDocumentEncoder {
    tokenizer: EmbeddedTokenizer,
    controls: TemplateControlIds,
    control_digest: Sha256Digest,
}
impl SourceDocumentEncoder {
    pub fn pinned(controls: &TemplateControlIds) -> Result<Self, ExtractError> {
        if controls.ids().iter().any(|&id| id as usize >= NANBEIGE_VOCAB_SIZE) {
            return Err(ExtractError::Contract("source control id outside vocabulary"));
        }
        Ok(Self {
            tokenizer: EmbeddedTokenizer::pinned().map_err(|_| ExtractError::Tokenizer)?,
            controls: controls.clone(), control_digest: control_digest(controls)?,
        })
    }
    /// The currently audited untrusted encoder uses one byte-fallback token
    /// per byte. Check both ceilings before token or source copies allocate.
    pub fn encode(&self, text: &str, max_bytes: usize, max_tokens: usize) -> Result<SourceDocument, ExtractError> {
        if text.len() > max_bytes || text.len() > max_tokens {
            return Err(ExtractError::Contract("source document exceeds byte or token budget"));
        }
        let document = UntrustedDocumentEncoder::new(self.tokenizer.tokenizer(), &self.controls)
            .encode(text.as_bytes()).map_err(|_| ExtractError::Contract("source byte-preserving encoding refused"))?;
        if document.ids().len() > max_tokens { return Err(ExtractError::Contract("source token budget exceeded")); }
        Ok(SourceDocument { document, control_digest: self.control_digest })
    }
}

/// Original UTF-8 text and its exact, pinned untrusted token sequence.
/// No Debug/Serialize implementation: this is private per-request content.
pub struct SourceDocument {
    document: UntrustedDocument,
    control_digest: Sha256Digest,
}
impl SourceDocument {
    pub fn text(&self) -> &str {
        std::str::from_utf8(self.document.bytes()).expect("SourceDocument was constructed from str")
    }
    pub fn token_ids(&self) -> &[u32] { self.document.ids() }
    pub(super) fn verify_task(&self, ir: &TaskIR, controls: &TemplateControlIds) -> Result<(), ExtractError> {
        let mut docs = ir.prompt_segments().iter().filter(|s| s.kind() == PromptSegmentKind::Document);
        if docs.next().is_none_or(|s| s.token_ids() != self.token_ids()) || docs.next().is_some() {
            return Err(ExtractError::Contract("source bytes do not bind the single TaskIR document"));
        }
        if self.control_digest != control_digest(controls)? {
            return Err(ExtractError::Contract("source encoding control registry differs from task"));
        }
        Ok(())
    }
}
fn control_digest(controls: &TemplateControlIds) -> Result<Sha256Digest, ExtractError> {
    Ok(Sha256Digest::of_bytes(&canonjson::canonical_bytes(controls.ids()).map_err(|_| ExtractError::Serialization)?))
}

impl ExtractPlan {
    /// Source-bound counterpart of from_task_plan. The TaskIR must explicitly
    /// name SOURCE_JSON_RUNTIME_VERSION and SourceSpansVerified. No unsupported
    /// source annotation is stripped, approximated, or checked only after a
    /// free-generation attempt. Native execution uses the same bounded runner.
    pub fn from_task_plan_with_source(
        task: &TaskPlan, schema: &str, options: JsonDecodeOptions, limits: CompileLimits,
        controls: &TemplateControlIds, source: &SourceDocument, source_limits: SourceRuntimeLimits,
    ) -> Result<Self, ExtractError> {
        Self::from_builtin(task, schema, options, limits, controls, Some((source, source_limits)), "extract-v1")
    }
}

pub(super) fn check_source_postconditions(ir: &TaskIR) -> Result<(), ExtractError> {
    let value = serde_json::to_value(ir).map_err(|_| ExtractError::Serialization)?;
    let view: PostconditionView = serde_json::from_value(value).map_err(|_| ExtractError::Serialization)?;
    if !view.postconditions.contains(&FinitePostcondition::SourceSpansVerified) || view.postconditions.iter().any(|p| !matches!(p,
        FinitePostcondition::JsonValid | FinitePostcondition::MatchesGrammar
        | FinitePostcondition::OutputWithinBudget | FinitePostcondition::SourceSpansVerified
    )) { return Err(ExtractError::Contract("source plan requires supported source-verification postconditions")); }
    Ok(())
}

#[derive(Serialize)]
struct SourcePolicy {
    runtime: &'static str,
    source_language: &'static str,
    extraction_policy: Sha256Digest,
    limits: SourceRuntimeLimits,
}
pub(super) fn source_policy_digest(base: Sha256Digest, limits: SourceRuntimeLimits) -> Result<Sha256Digest, ExtractError> {
    let policy = SourcePolicy { runtime: SOURCE_JSON_RUNTIME_VERSION, source_language: SOURCE_LANGUAGE_VERSION, extraction_policy: base, limits };
    Ok(Sha256Digest::of_bytes(&canonjson::canonical_bytes(&policy).map_err(|_| ExtractError::Serialization)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{BuiltInTask, ir::{TaskBudget, PlanContext, PromptSegment}};
    use super::super::tests::{identity, registry, options, output};
    const SCHEMA: &str = r#"{"type":"string","x-fnlp-source":"verbatim"}"#;
    fn task(document: &SourceDocument, source_postcondition: bool) -> TaskPlan {
        let id = identity();
        let budget = TaskBudget { max_input_tokens: 256, max_output_tokens: 8, max_output_bytes: 8192, max_grammar_states: 4096, max_kv_bytes: 1 << 30 };
        let mut postconditions = vec![FinitePostcondition::JsonValid, FinitePostcondition::OutputWithinBudget];
        if source_postcondition { postconditions.push(FinitePostcondition::SourceSpansVerified); }
        let ir = TaskIR::new(vec![PromptSegment::new(PromptSegmentKind::Document, document.token_ids().to_vec()),
            PromptSegment::new(PromptSegmentKind::AnswerScaffold, vec![2])], DecodeStrategy::ConstrainedJson,
            GrammarReference::json_schema(Sha256Digest::of_bytes(SCHEMA.as_bytes()), SOURCE_JSON_RUNTIME_VERSION),
            None, postconditions, budget, DependencyScope::ItemLocal).unwrap();
        TaskPlan::new(BuiltInTask::Extract.spec(), &PlanContext::new(&id, budget).unwrap(), ir).unwrap()
    }
    fn document(text: &str) -> SourceDocument {
        SourceDocumentEncoder::pinned(registry().template_controls()).unwrap().encode(text, 256, 256).unwrap()
    }
    fn plan(task: &TaskPlan, document: &SourceDocument) -> Result<ExtractPlan, ExtractError> {
        ExtractPlan::from_task_plan_with_source(task, SCHEMA, options(), CompileLimits::default(), registry().template_controls(), document, SourceRuntimeLimits::default())
    }
    #[test]
    fn source_and_prompt_cannot_be_swapped() {
        let a = document("Alice"); let b = document("Bob");
        assert!(plan(&task(&a, true), &b).is_err());
    }
    #[test]
    fn source_verification_is_an_explicit_required_postcondition() {
        let d = document("Alice"); assert!(plan(&task(&d, false), &d).is_err());
    }
    #[test]
    fn repeated_sources_emit_every_verified_occurrence_without_confidence() {
        let d = document("Alice Bob Alice"); let p = plan(&task(&d, true), &d).unwrap();
        let result = p.finalize(output("\"Alice\"")).unwrap();
        assert_eq!(result.schema_version, 2); assert_eq!(result.grounding, ExtractionGrounding::SourceMembership);
        assert_eq!(result.source_fields[0].spans.len(), 2);
        let bytes = canonjson::canonical_string(&result).unwrap();
        assert!(!bytes.contains("confidence")); assert!(!bytes.contains("prompt_digest"));
        assert!(p.finalize(output("\"Carol\"")).is_err());
    }
    #[test]
    fn source_runtime_and_limits_are_identity_bound() {
        let d = document("Alice"); let t = task(&d, true); let p = plan(&t, &d).unwrap();
        let mut id = p.bind_identity(identity()).unwrap();
        assert_eq!(id.grammar_compiler_version, SOURCE_JSON_RUNTIME_VERSION);
        id.grammar_compiler_version = JSON_RUNTIME_VERSION.to_owned(); assert!(p.verify_identity(&id).is_err());
        let base = Sha256Digest::of_bytes(b"policy"); let a = SourceRuntimeLimits::default(); let mut b = a;
        b.verification.max_matches -= 1;
        assert_ne!(source_policy_digest(base, a).unwrap(), source_policy_digest(base, b).unwrap());
    }
    #[test]
    fn source_encode_caps_are_checked_before_tokenization() {
        let e = SourceDocumentEncoder::pinned(registry().template_controls()).unwrap();
        assert!(e.encode("Alice", 4, 100).is_err()); assert!(e.encode("Alice", 100, 4).is_err());
    }
    #[test]
    fn source_evidence_is_included_in_the_result_byte_limit() {
        let d = document("Alice Alice"); let mut p = plan(&task(&d, true), &d).unwrap();
        let result = p.finalize(output("\"Alice\"")).unwrap();
        p.max_result_bytes = canonjson::canonical_bytes(&result).unwrap().len() as u64 - 1;
        assert!(matches!(p.finalize(output("\"Alice\"")), Err(ExtractError::OutputBudgetExceeded)));
    }
}
