//! Original-occurrence transport and complete question/manifest byte geometry.
use super::*;
use evidence::Collection;
use std::collections::BTreeMap;

impl Collection {
    pub(super) fn empty() -> Self { Self { passages: Vec::new(), origins: Vec::new(), bytes: 0 } }
    pub(super) fn window<C: DecodeStepControl>(&self, range: Range<usize>, control: &mut C)
        -> Result<Self, Int8SourceMapError> {
        checkpoint(control)?;
        if range.start >= range.end || range.end > self.passages.len() || self.passages.len() != self.origins.len() {
            return Err(invalid());
        }
        let mut out = Self { passages: vector(range.len())?, origins: vector(range.len())?, bytes: 0 };
        for index in range {
            checkpoint(control)?;
            let passage = &self.passages[index];
            let mut origins = vector(self.origins[index].len())?;
            for &origin in &self.origins[index] { checkpoint(control)?; origins.push(origin); }
            out.bytes = out.bytes.checked_add(passage.text.len()).ok_or(Int8SourceMapError::WorkLimit)?;
            out.passages.push(AnswerPassage { id: copy(&passage.id)?, text: copy(&passage.text)? });
            out.origins.push(origins);
        }
        checkpoint(control)?;
        Ok(out)
    }
    /// Only independently lifted citations enter this path. Equal quotes from
    /// different groups remain separate passages with their original provenance.
    pub(super) fn append_verified<C: DecodeStepControl>(&mut self, source: &str, answer: &SynthesisAnswer,
        expected: Int8QuestionHierarchyPreflight, remaining: &mut GroundingBudget, control: &mut C)
        -> Result<(), Int8SourceMapError> {
        let mut seen = BTreeMap::<&str, &[VerifiedSourceSpan]>::new();
        for citation in &answer.citations {
            checkpoint(control)?;
            evidence::check_original_citation(citation, expected.options, expected.discovery().source_span(), control)?;
            remaining.max_fields = remaining.max_fields.checked_sub(1).ok_or(Int8SourceMapError::WorkLimit)?;
            for span in &citation.spans {
                checkpoint(control)?;
                remaining.max_matches = remaining.max_matches.checked_sub(1).ok_or(Int8SourceMapError::WorkLimit)?;
                remaining.max_scan_steps = remaining.max_scan_steps.checked_sub(citation.quote.len() as u64)
                    .ok_or(Int8SourceMapError::WorkLimit)?;
                if source.get(span.byte_start..span.byte_end) != Some(citation.quote.as_str()) { return Err(invalid()); }
            }
            if let Some(old) = seen.insert(citation.quote.as_str(), &citation.spans) {
                if old != citation.spans.as_slice() { return Err(invalid()); }
                continue;
            }
            if self.passages.len() >= expected.base.limits.max_evidence_passages { return Err(Int8SourceMapError::WorkLimit); }
            self.bytes = self.bytes.checked_add(citation.quote.len()).filter(|&n| n <= expected.base.limits.max_evidence_bytes)
                .ok_or(Int8SourceMapError::WorkLimit)?;
            self.passages.try_reserve(1).map_err(|_| Int8SourceMapError::Allocation)?;
            self.origins.try_reserve(1).map_err(|_| Int8SourceMapError::Allocation)?;
            let mut origins = vector(citation.spans.len())?;
            for &span in &citation.spans { checkpoint(control)?; origins.push(span); }
            self.passages.push(AnswerPassage { id: format!("evidence_{}", self.passages.len()), text: copy(&citation.quote)? });
            self.origins.push(origins);
        }
        checkpoint(control)?;
        Ok(())
    }
}
/// The exact declarative manifest layout used by AnswerContext. The actual
/// pinned encoder still admits every complete question/context independently.
pub(super) fn context_bytes<'a, C: DecodeStepControl>(question_bytes: usize,
    passages: impl Iterator<Item = (&'a str, &'a str)>, control: &mut C) -> Result<usize, Int8SourceMapError> {
    #[derive(Serialize)]
    struct Layout<'a> { id: &'a str, span: VerifiedSourceSpan }
    let mut layout = Vec::new(); let mut bytes = 0_usize; let mut scalars = 0_usize;
    for (id, text) in passages {
        checkpoint(control)?;
        if !layout.is_empty() {
            bytes = bytes.checked_add(2).ok_or(Int8SourceMapError::WorkLimit)?;
            scalars = scalars.checked_add(2).ok_or(Int8SourceMapError::WorkLimit)?;
        }
        let byte_start = bytes; let scalar_start = scalars;
        bytes = bytes.checked_add(text.len()).ok_or(Int8SourceMapError::WorkLimit)?;
        let count = mapreduce::count_scalars_with_checkpoints(text, &mut || checkpoint(control))?;
        scalars = scalars.checked_add(count).ok_or(Int8SourceMapError::WorkLimit)?;
        layout.try_reserve(1).map_err(|_| Int8SourceMapError::Allocation)?;
        layout.push(Layout { id, span: VerifiedSourceSpan { byte_start, byte_end: bytes, scalar_start, scalar_end: scalars } });
    }
    if layout.is_empty() { return Err(invalid()); }
    let manifest = canonjson::canonical_bytes(&(ANSWER_PASSAGE_LAYOUT_VERSION, layout))
        .map_err(|_| Int8SourceError::from(SourcePlanningError::Serialization))?;
    checkpoint(control)?;
    bytes.checked_add(question_bytes).and_then(|n| n.checked_add(manifest.len())).ok_or(Int8SourceMapError::WorkLimit)
}
pub(super) fn selected_quotes<'a, C: DecodeStepControl>(passes: &'a [QuestionHierarchyPass], control: &mut C)
    -> Result<Vec<&'a str>, Int8SourceMapError> {
    let mut quotes = Vec::new();
    for pass in passes {
        checkpoint(control)?;
        let mut seen = BTreeMap::<&str, &[VerifiedSourceSpan]>::new();
        for citation in &pass.answer.citations {
            checkpoint(control)?;
            if let Some(old) = seen.insert(citation.quote.as_str(), &citation.spans) {
                if old != citation.spans.as_slice() { return Err(invalid()); }
                continue;
            }
            quotes.try_reserve(1).map_err(|_| Int8SourceMapError::Allocation)?;
            quotes.push(citation.quote.as_str());
        }
    }
    Ok(quotes)
}
pub(super) fn quote_size<C: DecodeStepControl>(quotes: &[&str], control: &mut C) -> Result<EvidenceSize, Int8SourceMapError> {
    let mut bytes = 0_usize;
    for quote in quotes { checkpoint(control)?; bytes = bytes.checked_add(quote.len()).ok_or_else(invalid)?; }
    Ok(EvidenceSize { passages: quotes.len(), bytes })
}
