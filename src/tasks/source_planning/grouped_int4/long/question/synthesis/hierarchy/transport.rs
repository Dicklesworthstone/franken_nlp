//! Original-occurrence transport and complete question/manifest byte geometry.
use super::*;
use evidence::Collection;
use std::collections::BTreeMap;

impl Collection {
    pub(super) fn empty() -> Self { Self { passages: Vec::new(), origins: Vec::new(), bytes: 0 } }
    pub(super) fn window<C: DecodeStepControl>(&self, range: Range<usize>, control: &mut C)
        -> Result<Self, Int4SourceMapError> {
        checkpoint(control)?;
        if range.start >= range.end || range.end > self.passages.len() || self.passages.len() != self.origins.len() {
            return Err(invalid());
        }
        let mut out = Self { passages: vector(range.len())?, origins: vector(range.len())?, bytes: 0 };
        for index in range {
            checkpoint(control)?;
            let passage = &self.passages[index];
            let mut origins = vector(self.origins[index].len())?; origins.extend_from_slice(&self.origins[index]);
            out.bytes = out.bytes.checked_add(passage.text.len()).ok_or(Int4SourceMapError::WorkLimit)?;
            out.passages.push(AnswerPassage { id: copy(&passage.id)?, text: copy(&passage.text)? });
            out.origins.push(origins);
        }
        checkpoint(control)?;
        Ok(out)
    }
    /// Only independently lifted citations enter this private path. Equal
    /// quotations from different groups remain distinct provenance passages.
    pub(super) fn append_verified<C: DecodeStepControl>(&mut self, source: &str, answer: &SynthesisAnswer,
        expected: Int4QuestionHierarchyPreflight, remaining: &mut GroundingBudget, control: &mut C)
        -> Result<(), Int4SourceMapError> {
        let mut seen = BTreeMap::<&str, &[VerifiedSourceSpan]>::new();
        for citation in &answer.citations {
            checkpoint(control)?;
            evidence::check_original_citation(citation, expected.base.options, expected.discovery().source_span())?;
            remaining.max_fields = remaining.max_fields.checked_sub(1).ok_or(Int4SourceMapError::WorkLimit)?;
            for span in &citation.spans {
                checkpoint(control)?;
                remaining.max_matches = remaining.max_matches.checked_sub(1).ok_or(Int4SourceMapError::WorkLimit)?;
                remaining.max_scan_steps = remaining.max_scan_steps.checked_sub(citation.quote.len() as u64)
                    .ok_or(Int4SourceMapError::WorkLimit)?;
                if source.get(span.byte_start..span.byte_end) != Some(citation.quote.as_str()) { return Err(invalid()); }
            }
            if let Some(old) = seen.insert(citation.quote.as_str(), &citation.spans) {
                if old != citation.spans.as_slice() { return Err(invalid()); }
                continue;
            }
            if self.passages.len() >= expected.base.limits.max_evidence_passages { return Err(Int4SourceMapError::WorkLimit); }
            self.bytes = self.bytes.checked_add(citation.quote.len()).filter(|&n| n <= expected.base.limits.max_evidence_bytes)
                .ok_or(Int4SourceMapError::WorkLimit)?;
            self.passages.try_reserve(1).map_err(|_| Int4SourceMapError::Allocation)?;
            self.origins.try_reserve(1).map_err(|_| Int4SourceMapError::Allocation)?;
            let mut origins = vector(citation.spans.len())?; origins.extend_from_slice(&citation.spans);
            self.passages.push(AnswerPassage { id: format!("evidence_{}", self.passages.len()), text: copy(&citation.quote)? });
            self.origins.push(origins);
        }
        checkpoint(control)?;
        Ok(())
    }
}
/// This is the exact declarative manifest layout used by AnswerContext, not a
/// source-only token estimate. The actual pinned encoder still admits every
/// candidate context independently before its compiled plan is authorized.
pub(super) fn context_bytes<'a>(question_bytes: usize, passages: impl Iterator<Item = (&'a str, &'a str)>)
    -> Result<usize, Int4SourceMapError> {
    #[derive(Serialize)]
    struct Layout<'a> { id: &'a str, span: VerifiedSourceSpan }
    let mut layout = Vec::new(); let mut bytes = 0_usize; let mut scalars = 0_usize;
    for (id, text) in passages {
        if !layout.is_empty() {
            bytes = bytes.checked_add(2).ok_or(Int4SourceMapError::WorkLimit)?;
            scalars = scalars.checked_add(2).ok_or(Int4SourceMapError::WorkLimit)?;
        }
        let byte_start = bytes; let scalar_start = scalars;
        bytes = bytes.checked_add(text.len()).ok_or(Int4SourceMapError::WorkLimit)?;
        scalars = scalars.checked_add(text.chars().count()).ok_or(Int4SourceMapError::WorkLimit)?;
        layout.try_reserve(1).map_err(|_| Int4SourceMapError::Allocation)?;
        layout.push(Layout { id, span: VerifiedSourceSpan { byte_start, byte_end: bytes, scalar_start, scalar_end: scalars } });
    }
    if layout.is_empty() { return Err(invalid()); }
    let manifest = canonjson::canonical_bytes(&(ANSWER_PASSAGE_LAYOUT_VERSION, layout))
        .map_err(|_| Int4SourceError::from(SourcePlanningError::Serialization))?;
    bytes.checked_add(question_bytes).and_then(|n| n.checked_add(manifest.len())).ok_or(Int4SourceMapError::WorkLimit)
}
pub(super) fn selected_quotes(passes: &[QuestionHierarchyPass]) -> Result<Vec<&str>, Int4SourceMapError> {
    let mut quotes = Vec::new();
    for pass in passes {
        let mut seen = BTreeMap::<&str, &[VerifiedSourceSpan]>::new();
        for citation in &pass.answer.citations {
            if let Some(old) = seen.insert(citation.quote.as_str(), &citation.spans) {
                if old != citation.spans.as_slice() { return Err(invalid()); }
                continue;
            }
            quotes.try_reserve(1).map_err(|_| Int4SourceMapError::Allocation)?;
            quotes.push(citation.quote.as_str());
        }
    }
    Ok(quotes)
}
pub(super) fn quote_size(quotes: &[&str]) -> Result<EvidenceSize, Int4SourceMapError> {
    Ok(EvidenceSize { passages: quotes.len(), bytes: quotes.iter().try_fold(0_usize,
        |sum, quote| sum.checked_add(quote.len()).ok_or_else(invalid))? })
}
