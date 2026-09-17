//! Pinned, byte-preserving source admission and source-bound extraction.
//! SourceDocument is not deserializable: its private token/text binding can
//! only be minted by the pinned encoder. All untrusted context segments bind
//! the TaskIR, but only the designated source belongs to the citation language.

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
        self.encode_with_context(text, &[], max_bytes, max_tokens)
    }

    /// Encode a designated evidence source and at most 16 auxiliary contexts
    /// (for example a question and a passage manifest). TaskIR must contain
    /// their Document segments in exactly this order: contexts, then source.
    /// Trusted separators may occur between segments. Contexts are not added
    /// to text(), so they can never authorize a source-constrained quotation.
    /// The byte and token ceilings apply to the SUM, not freshly to each part.
    pub fn encode_with_context(&self, text: &str, contexts: &[&str], max_bytes: usize,
        max_tokens: usize) -> Result<SourceDocument, ExtractError> {
        if contexts.len() > 16 { return Err(ExtractError::Contract("too many source context segments")); }
        let total = contexts.iter().try_fold(text.len(), |sum, part| sum.checked_add(part.len()))
            .ok_or(ExtractError::AllocationRefused)?;
        if total > max_bytes || total > max_tokens {
            return Err(ExtractError::Contract("source and context exceed aggregate byte or token budget"));
        }
        let encoder = UntrustedDocumentEncoder::new(self.tokenizer.tokenizer(), &self.controls);
        let encode = |part: &str| encoder.encode(part.as_bytes())
            .map_err(|_| ExtractError::Contract("source byte-preserving encoding refused"));
        let mut context_documents = Vec::new();
        context_documents.try_reserve_exact(contexts.len()).map_err(|_| ExtractError::AllocationRefused)?;
        let mut token_count = 0_usize;
        for context in contexts {
            let encoded = encode(context)?;
            token_count = token_count.checked_add(encoded.ids().len()).ok_or(ExtractError::AllocationRefused)?;
            if token_count > max_tokens { return Err(ExtractError::Contract("source context token budget exceeded")); }
            context_documents.push(encoded);
        }
        let document = encode(text)?;
        token_count = token_count.checked_add(document.ids().len()).ok_or(ExtractError::AllocationRefused)?;
        if token_count > max_tokens { return Err(ExtractError::Contract("source token budget exceeded")); }
        Ok(SourceDocument { document, context_documents, token_count, control_digest: self.control_digest })
    }
}

/// Original evidence text, optional auxiliary contexts, and their exact pinned
/// untrusted token sequences. No Debug/Serialize: this is private request data.
pub struct SourceDocument {
    document: UntrustedDocument,
    context_documents: Vec<UntrustedDocument>,
    token_count: usize,
    control_digest: Sha256Digest,
}
impl SourceDocument {
    /// Evidence text only; questions and metadata are deliberately excluded.
    pub fn text(&self) -> &str {
        std::str::from_utf8(self.document.bytes()).expect("SourceDocument was constructed from str")
    }
    pub fn token_ids(&self) -> &[u32] { self.document.ids() }
    pub fn context_token_ids(&self) -> impl ExactSizeIterator<Item = &[u32]> + '_ {
        self.context_documents.iter().map(|document| document.ids())
    }
    pub fn total_token_count(&self) -> usize { self.token_count }
    pub(super) fn verify_task(&self, ir: &TaskIR, controls: &TemplateControlIds) -> Result<(), ExtractError> {
        let mut docs = ir.prompt_segments().iter().filter(|s| s.kind() == PromptSegmentKind::Document);
        for expected in self.context_token_ids().chain(std::iter::once(self.token_ids())) {
            if docs.next().is_none_or(|s| s.token_ids() != expected) {
                return Err(ExtractError::Contract("source/context bytes differ from ordered TaskIR documents"));
            }
        }
        if docs.next().is_some() { return Err(ExtractError::Contract("unbound extra TaskIR document")); }
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
        let mut segments: Vec<_> = document.context_token_ids()
            .map(|ids| PromptSegment::new(PromptSegmentKind::Document, ids.to_vec())).collect();
        segments.push(PromptSegment::new(PromptSegmentKind::Document, document.token_ids().to_vec()));
        segments.push(PromptSegment::new(PromptSegmentKind::AnswerScaffold, vec![2]));
        let ir = TaskIR::new(segments, DecodeStrategy::ConstrainedJson,
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
    #[test]
    fn context_only_text_cannot_be_cited() {
        let e = SourceDocumentEncoder::pinned(registry().template_controls()).unwrap();
        let d = e.encode_with_context("Alice", &["Carol?", "passage-id-Bob"], 256, 256).unwrap();
        let p = plan(&task(&d, true), &d).unwrap();
        assert_eq!(d.text(), "Alice");
        assert!(p.finalize(output("\"Alice\"")).is_ok());
        assert!(p.finalize(output("\"Carol\"")).is_err());
        assert!(p.finalize(output("\"Bob\"")).is_err());
    }
    #[test]
    fn swapping_omitting_or_inserting_context_fails_exact_binding() {
        let e = SourceDocumentEncoder::pinned(registry().template_controls()).unwrap();
        let d = e.encode_with_context("Alice", &["one", "two"], 256, 256).unwrap();
        let t = task(&d, true);
        for contexts in [vec!["two", "one"], vec!["one"], vec!["one", "two", "extra"], vec!["changed", "two"]] {
            let other = e.encode_with_context("Alice", &contexts, 256, 256).unwrap();
            assert!(plan(&t, &other).is_err());
        }
        assert!(plan(&t, &document("Alice")).is_err());
    }
    #[test]
    fn context_byte_and_token_limits_are_aggregate() {
        let e = SourceDocumentEncoder::pinned(registry().template_controls()).unwrap();
        assert!(e.encode_with_context("Alice", &["Bob"], 7, 100).is_err());
        assert!(e.encode_with_context("Alice", &["Bob"], 100, 7).is_err());
        let d = e.encode_with_context("Alice", &["Bob"], 8, 8).unwrap();
        assert_eq!(d.total_token_count(), 8);
        assert!(e.encode_with_context("", &[""; 17], 100, 100).is_err());
    }
    #[test]
    fn every_context_excludes_privileged_control_ids() {
        let r = registry(); let e = SourceDocumentEncoder::pinned(r.template_controls()).unwrap();
        let d = e.encode_with_context("Alice <think>", &["<think> Carol", "<eos>"], 256, 256).unwrap();
        for ids in d.context_token_ids().chain(std::iter::once(d.token_ids())) {
            assert!(ids.iter().all(|&id| !r.template_controls().contains(id)));
        }
        assert!(plan(&task(&d, true), &d).is_ok());
    }
}
