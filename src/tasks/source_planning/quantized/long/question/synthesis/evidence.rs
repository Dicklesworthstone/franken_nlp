//! Recheck evidence before deduplication, then lift final subquotes through the
//! original source occurrences. Never search the whole document to invent support.
use super::*;
use std::collections::BTreeSet;
use crate::validation::grounded_fields::FieldGroundingError;

pub(super) struct Collection {
    pub passages: Vec<AnswerPassage>,
    origins: Vec<Vec<VerifiedSourceSpan>>,
    pub bytes: usize,
}
impl Collection {
    pub(super) fn copy_passages(&self) -> Result<Vec<AnswerPassage>, Int8SourceMapError> {
        let mut out = vector(self.passages.len())?;
        for passage in &self.passages { out.push(AnswerPassage { id: copy(&passage.id)?, text: copy(&passage.text)? }); }
        Ok(out)
    }
}
pub(super) fn collect<C: DecodeStepControl>(source: &str, value: &QuestionValue, limits: QuestionSynthesisLimits,
    remaining: &mut GroundingBudget, control: &mut C) -> Result<Collection, Int8SourceMapError> {
    let mut result = Collection { passages: Vec::new(), origins: Vec::new(), bytes: 0 };
    let mut seen = BTreeSet::new();
    let (mut byte, mut scalar) = (0_usize, 0_usize);
    for (index, chunk) in value.chunks().enumerate() {
        checkpoint(control)?;
        let extent = chunk.source_span;
        if chunk.chunk_id != index || extent.byte_start != byte || extent.scalar_start != scalar
            || extent.byte_end <= byte { return Err(invalid()); }
        let text = source.get(extent.byte_start..extent.byte_end).ok_or_else(invalid)?;
        remaining.max_scan_steps = remaining.max_scan_steps.checked_sub(text.len() as u64).ok_or(Int8SourceMapError::WorkLimit)?;
        scalar = scalar.checked_add(text.chars().count()).ok_or_else(invalid)?;
        if scalar != extent.scalar_end { return Err(invalid()); }
        byte = extent.byte_end;
        match chunk.status {
            QuestionChunkStatus::Answered => {
                if !chunk.answer.as_deref().is_some_and(has_text) || chunk.citations.is_empty() { return Err(invalid()); }
            }
            QuestionChunkStatus::Abstained | QuestionChunkStatus::WhitespaceOnly => {
                if chunk.answer.is_some() || !chunk.citations.is_empty()
                    || (chunk.status == QuestionChunkStatus::WhitespaceOnly) == has_text(text) { return Err(invalid()); }
                continue;
            }
        }
        for citation in &chunk.citations {
            checkpoint(control)?;
            if !has_text(&citation.quote) { return Err(invalid()); }
            remaining.max_fields = remaining.max_fields.checked_sub(1).ok_or(Int8SourceMapError::WorkLimit)?;
            let local = scan(text, &citation.quote, remaining)?;
            if local.is_empty() || citation.spans.len() != local.len() || citation.occurrence != occurrence(local.len()) {
                return Err(invalid());
            }
            let mut original = vector(local.len())?;
            for (span, reported) in local.into_iter().zip(&citation.spans) {
                checkpoint(control)?;
                let lifted = lift(span, extent)?;
                if &lifted != reported { return Err(invalid()); }
                original.push(lifted);
            }
            // Duplicates cannot hide bad offsets or replenish verification work.
            // Equal quotes from different chunks remain separate source passages.
            if !seen.insert((chunk.chunk_id, citation.quote.as_str())) { continue; }
            if result.passages.len() == limits.max_evidence_passages { return Err(Int8SourceMapError::WorkLimit); }
            result.bytes = result.bytes.checked_add(citation.quote.len()).filter(|&n| n <= limits.max_evidence_bytes)
                .ok_or(Int8SourceMapError::WorkLimit)?;
            result.passages.try_reserve(1).map_err(|_| Int8SourceMapError::Allocation)?;
            result.origins.try_reserve(1).map_err(|_| Int8SourceMapError::Allocation)?;
            // Code-owned, bounded, source-order IDs. No model answer, question,
            // document fingerprint or caller-controlled instruction enters them.
            result.passages.push(AnswerPassage { id: format!("evidence_{}", result.passages.len()), text: copy(&citation.quote)? });
            result.origins.push(original);
        }
    }
    if byte != source.len() { return Err(invalid()); }
    checkpoint(control)?;
    Ok(result)
}
pub(super) fn final_answer<C: DecodeStepControl>(source: &str, raw: AnswerResult, evidence: &Collection,
    options: AnswerOptions, remaining: &mut GroundingBudget, control: &mut C) -> Result<SynthesisAnswer, Int8SourceMapError> {
    checkpoint(control)?;
    if raw.schema_version != 1 || raw.task_spec_version != ANSWER_TASK_VERSION || raw.numerics_profile != STRICT_INT8_PROFILE
        || raw.calibration != AnswerCalibration::Uncalibrated || raw.score_space != ScoreSpace::NotComputed
        || raw.citation_guarantee != CitationGuarantee::StructuralSourceMembership
        || raw.semantic_support != SummarySemanticSupport::NotAssessed
        || raw.untrusted_fields != ["answer", "citations"] || raw.citations.len() > options.max_citations
        || evidence.passages.is_empty() || evidence.passages.len() != evidence.origins.len() { return Err(invalid()); }
    let status = match raw.status {
        AnswerStatus::Answered => {
            if !raw.answerable || raw.citations.is_empty() || !raw.answer.as_ref().is_some_and(|text|
                has_text(text) && text.chars().count() <= options.max_answer_scalars) { return Err(invalid()); }
            SynthesisStatus::Answered
        }
        AnswerStatus::Abstained => {
            if raw.answerable || raw.answer.is_some() || !raw.citations.is_empty() { return Err(invalid()); }
            SynthesisStatus::Abstained
        }
    };
    let mut citations = vector(raw.citations.len())?;
    for citation in raw.citations {
        checkpoint(control)?;
        if citation.quote.is_empty() || citation.quote.chars().count() > options.max_quote_scalars { return Err(invalid()); }
        remaining.max_fields = remaining.max_fields.checked_sub(1).ok_or(Int8SourceMapError::WorkLimit)?;
        let mut spans = Vec::new(); let mut next = 0_usize;
        for (passage, origins) in evidence.passages.iter().zip(&evidence.origins) {
            checkpoint(control)?;
            // Answer's source grammar sees a join, but citations must occur
            // WHOLLY in an admitted evidence passage. Join-spanning text fails.
            for local in scan(&passage.text, &citation.quote, remaining)? {
                let reported = citation.spans.get(next).ok_or_else(invalid)?;
                if reported.passage_id != passage.id || reported.span != local { return Err(invalid()); }
                next = next.checked_add(1).ok_or_else(invalid)?;
                for &origin in origins {
                    checkpoint(control)?;
                    // Charge fanout BEFORE lifting, even duplicate coordinates.
                    remaining.max_matches = remaining.max_matches.checked_sub(1).ok_or(Int8SourceMapError::WorkLimit)?;
                    let span = lift(local, origin)?;
                    if source.get(span.byte_start..span.byte_end) != Some(citation.quote.as_str()) { return Err(invalid()); }
                    spans.try_reserve(1).map_err(|_| Int8SourceMapError::Allocation)?;
                    spans.push(span);
                }
            }
        }
        if next == 0 || next != citation.spans.len() || citation.occurrence != occurrence(next) || spans.is_empty() {
            return Err(invalid());
        }
        // Overlapping evidence or repeated proposals may address the SAME
        // original occurrence. Do not label that as multiple source occurrences.
        spans.sort_unstable_by_key(|s| (s.byte_start, s.byte_end, s.scalar_start, s.scalar_end));
        spans.dedup();
        citations.push(SourceCitation { quote: citation.quote, occurrence: occurrence(spans.len()), spans });
    }
    checkpoint(control)?;
    Ok(SynthesisAnswer { status, answer: raw.answer, citations })
}
fn occurrence(count: usize) -> SourceOccurrence {
    if count == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }
}
fn scan(source: &str, quote: &str, budget: &mut GroundingBudget) -> Result<Vec<VerifiedSourceSpan>, Int8SourceMapError> {
    scan_occurrences(source, quote, budget).map_err(|e| match e {
        FieldGroundingError::AllocationRefused => Int8SourceMapError::Allocation,
        FieldGroundingError::WorkBudget | FieldGroundingError::MatchBudget | FieldGroundingError::FieldBudget => Int8SourceMapError::WorkLimit,
        _ => invalid(),
    })
}
fn lift(local: VerifiedSourceSpan, origin: VerifiedSourceSpan) -> Result<VerifiedSourceSpan, Int8SourceMapError> {
    if local.byte_start >= local.byte_end || local.scalar_start >= local.scalar_end { return Err(invalid()); }
    let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(invalid);
    let span = VerifiedSourceSpan { byte_start: add(origin.byte_start, local.byte_start)?,
        byte_end: add(origin.byte_start, local.byte_end)?, scalar_start: add(origin.scalar_start, local.scalar_start)?,
        scalar_end: add(origin.scalar_start, local.scalar_end)? };
    if span.byte_end > origin.byte_end || span.scalar_end > origin.scalar_end { return Err(invalid()); }
    Ok(span)
}
