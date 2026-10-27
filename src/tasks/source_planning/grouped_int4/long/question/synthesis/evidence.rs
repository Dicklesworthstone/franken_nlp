//! Reverify every discovery quote before deduplication, then lift final subquotes
//! through only the selected original occurrences. Synthetic manifest bytes and
//! generated passage answers cannot become factual source evidence.
use super::*;
use std::collections::BTreeSet;

pub(super) struct Collection {
    pub(super) passages: Vec<AnswerPassage>,
    pub(super) origins: Vec<Vec<VerifiedSourceSpan>>,
    pub(super) bytes: usize,
}
impl Collection {
    pub(super) fn copy_passages(&self) -> Result<Vec<AnswerPassage>, Int4SourceMapError> {
        let mut out = vector(self.passages.len())?;
        for passage in &self.passages { out.push(AnswerPassage { id: copy(&passage.id)?, text: copy(&passage.text)? }); }
        Ok(out)
    }
}
pub(super) fn collect<C: DecodeStepControl>(source: &str, value: &QuestionValue, limits: QuestionSynthesisLimits,
    remaining: &mut GroundingBudget, control: &mut C) -> Result<Collection, Int4SourceMapError> {
    let mut result = Collection { passages: Vec::new(), origins: Vec::new(), bytes: 0 };
    let mut seen = BTreeSet::new();
    let (mut byte, mut scalar) = (0_usize, 0_usize);
    for (index, chunk) in value.chunks().enumerate() {
        checkpoint(control)?;
        let extent = chunk.source_span;
        if chunk.chunk_id != index || extent.byte_start != byte || extent.scalar_start != scalar
            || extent.byte_end <= byte { return Err(invalid()); }
        let text = source.get(extent.byte_start..extent.byte_end).ok_or_else(invalid)?;
        charge_scan(remaining, text.len())?;
        scalar = scalar.checked_add(text.chars().count()).ok_or_else(invalid)?;
        if scalar != extent.scalar_end { return Err(invalid()); }
        byte = extent.byte_end;
        match chunk.status {
            QuestionChunkStatus::Answered => {
                if !chunk.answer.as_deref().is_some_and(has_text) || chunk.citations.is_empty() || !has_text(text) {
                    return Err(invalid());
                }
            }
            QuestionChunkStatus::Abstained | QuestionChunkStatus::WhitespaceOnly => {
                if chunk.answer.is_some() || !chunk.citations.is_empty()
                    || (chunk.status == QuestionChunkStatus::WhitespaceOnly) == has_text(text)
                    || (chunk.status == QuestionChunkStatus::WhitespaceOnly
                        && (chunk.model_work != Int4Work::default() || chunk.mask_node_visit_charge != 0)) {
                    return Err(invalid());
                }
                continue;
            }
        }
        for citation in &chunk.citations {
            checkpoint(control)?;
            if citation.quote.is_empty() { return Err(invalid()); }
            remaining.max_fields = remaining.max_fields.checked_sub(1).ok_or(Int4SourceMapError::WorkLimit)?;
            let local = scan_occurrences(text, &citation.quote, remaining).map_err(scan_error)?;
            if local.is_empty() || citation.spans.len() != local.len() || citation.occurrence != occurrence(local.len()) {
                return Err(invalid());
            }
            let mut origins = vector(local.len())?;
            for (span, reported) in local.into_iter().zip(&citation.spans) {
                checkpoint(control)?;
                let lifted = lift(span, extent)?;
                if &lifted != reported { return Err(invalid()); }
                origins.push(lifted);
            }
            // Every duplicate is checked above; only storage deduplicates it.
            if !seen.insert((chunk.chunk_id, citation.quote.as_str())) { continue; }
            if result.passages.len() >= limits.max_evidence_passages { return Err(Int4SourceMapError::WorkLimit); }
            result.bytes = result.bytes.checked_add(citation.quote.len()).filter(|&n| n <= limits.max_evidence_bytes)
                .ok_or(Int4SourceMapError::WorkLimit)?;
            result.passages.try_reserve(1).map_err(|_| Int4SourceMapError::Allocation)?;
            result.origins.try_reserve(1).map_err(|_| Int4SourceMapError::Allocation)?;
            result.passages.push(AnswerPassage { id: format!("evidence_{}", result.passages.len()), text: copy(&citation.quote)? });
            result.origins.push(origins);
        }
    }
    if byte != source.len() { return Err(invalid()); }
    checkpoint(control)?;
    Ok(result)
}
pub(super) fn final_answer<C: DecodeStepControl>(source: &str, raw: AnswerResult, evidence: &Collection,
    options: AnswerOptions, remaining: &mut GroundingBudget, control: &mut C) -> Result<SynthesisAnswer, Int4SourceMapError> {
    lift_answer(source, &raw, evidence, options, remaining, control)
}
/// Borrow the native receipt so hierarchical callers can retain every pass.
pub(super) fn lift_answer<C: DecodeStepControl>(source: &str, raw: &AnswerResult, evidence: &Collection,
    options: AnswerOptions, remaining: &mut GroundingBudget, control: &mut C) -> Result<SynthesisAnswer, Int4SourceMapError> {
    checkpoint(control)?;
    let status = check_answer(raw, options)?;
    if evidence.passages.is_empty() || evidence.passages.len() != evidence.origins.len() { return Err(invalid()); }
    let mut citations = vector(raw.citations.len())?;
    for citation in &raw.citations {
        checkpoint(control)?;
        remaining.max_fields = remaining.max_fields.checked_sub(1).ok_or(Int4SourceMapError::WorkLimit)?;
        let mut spans = Vec::new(); let mut next = 0_usize;
        for (passage, origins) in evidence.passages.iter().zip(&evidence.origins) {
            checkpoint(control)?;
            let local_spans = match scan_occurrences(&passage.text, &citation.quote, remaining) {
                Ok(spans) => spans,
                // A quote need not occur in EVERY passage. The no-match scan is
                // still charged, and later passages must still be inspected.
                Err(FieldGroundingError::Absent) => continue,
                Err(error) => return Err(scan_error(error)),
            };
            for local in local_spans {
                let reported = citation.spans.get(next).ok_or_else(invalid)?;
                if reported.passage_id != passage.id || reported.span != local { return Err(invalid()); }
                next = next.checked_add(1).ok_or_else(invalid)?;
                for &origin in origins {
                    checkpoint(control)?;
                    remaining.max_matches = remaining.max_matches.checked_sub(1).ok_or(Int4SourceMapError::WorkLimit)?;
                    charge_scan(remaining, citation.quote.len())?;
                    let span = lift(local, origin)?;
                    if source.get(span.byte_start..span.byte_end) != Some(citation.quote.as_str()) { return Err(invalid()); }
                    spans.try_reserve(1).map_err(|_| Int4SourceMapError::Allocation)?;
                    spans.push(span);
                }
            }
        }
        if next == 0 || next != citation.spans.len() || citation.occurrence != occurrence(next) || spans.is_empty() {
            return Err(invalid());
        }
        spans.sort_unstable_by_key(|s| (s.byte_start, s.byte_end, s.scalar_start, s.scalar_end));
        spans.dedup();
        citations.push(SourceCitation { quote: copy(&citation.quote)?, occurrence: occurrence(spans.len()), spans });
    }
    let answer = raw.answer.as_deref().map(copy).transpose()?;
    checkpoint(control)?;
    Ok(SynthesisAnswer { status, answer, citations })
}
pub(super) fn check_answer(raw: &AnswerResult, options: AnswerOptions) -> Result<SynthesisStatus, Int4SourceMapError> {
    if raw.schema_version != 1 || raw.task_spec_version != ANSWER_TASK_VERSION || raw.numerics_profile != STRICT_INT4_PROFILE
        || raw.calibration != AnswerCalibration::Uncalibrated || raw.score_space != ScoreSpace::NotComputed
        || raw.citation_guarantee != CitationGuarantee::StructuralSourceMembership
        || raw.semantic_support != SummarySemanticSupport::NotAssessed
        || raw.untrusted_fields != ["answer", "citations"] || raw.citations.len() > options.max_citations { return Err(invalid()); }
    let status = match raw.status {
        AnswerStatus::Answered => {
            if !raw.answerable || raw.citations.is_empty() || !raw.answer.as_ref().is_some_and(|answer|
                has_text(answer) && answer.chars().count() <= options.max_answer_scalars) { return Err(invalid()); }
            SynthesisStatus::Answered
        }
        AnswerStatus::Abstained => {
            if raw.answerable || raw.answer.is_some() || !raw.citations.is_empty() { return Err(invalid()); }
            SynthesisStatus::Abstained
        }
    };
    for citation in &raw.citations {
        if citation.quote.is_empty() || citation.quote.chars().count() > options.max_quote_scalars { return Err(invalid()); }
    }
    Ok(status)
}
pub(super) fn check_original_citation(citation: &SourceCitation, options: AnswerOptions, source: VerifiedSourceSpan)
    -> Result<(), Int4SourceMapError> {
    let scalars = citation.quote.chars().count();
    if citation.quote.is_empty() || scalars > options.max_quote_scalars || citation.spans.is_empty()
        || citation.occurrence != occurrence(citation.spans.len())
        || citation.spans.iter().any(|s| s.byte_end.checked_sub(s.byte_start) != Some(citation.quote.len())
            || s.scalar_end.checked_sub(s.scalar_start) != Some(scalars)
            || s.byte_start < source.byte_start || s.byte_end > source.byte_end
            || s.scalar_start < source.scalar_start || s.scalar_end > source.scalar_end)
        || citation.spans.windows(2).any(|p| p[0].byte_start >= p[1].byte_start || p[0].scalar_start >= p[1].scalar_start) {
        return Err(invalid());
    }
    Ok(())
}
fn occurrence(count: usize) -> SourceOccurrence { if count == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous } }
fn charge_scan(budget: &mut GroundingBudget, bytes: usize) -> Result<(), Int4SourceMapError> {
    budget.max_scan_steps = budget.max_scan_steps.checked_sub(bytes as u64).ok_or(Int4SourceMapError::WorkLimit)?; Ok(())
}
fn scan_error(error: FieldGroundingError) -> Int4SourceMapError {
    match error { FieldGroundingError::AllocationRefused => Int4SourceMapError::Allocation,
        FieldGroundingError::WorkBudget | FieldGroundingError::MatchBudget | FieldGroundingError::FieldBudget => Int4SourceMapError::WorkLimit,
        _ => invalid() }
}
fn lift(local: VerifiedSourceSpan, origin: VerifiedSourceSpan) -> Result<VerifiedSourceSpan, Int4SourceMapError> {
    if local.byte_start >= local.byte_end || local.scalar_start >= local.scalar_end { return Err(invalid()); }
    let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(invalid);
    let span = VerifiedSourceSpan { byte_start: add(origin.byte_start, local.byte_start)?, byte_end: add(origin.byte_start, local.byte_end)?,
        scalar_start: add(origin.scalar_start, local.scalar_start)?, scalar_end: add(origin.scalar_start, local.scalar_end)? };
    if span.byte_end > origin.byte_end || span.scalar_end > origin.scalar_end { return Err(invalid()); }
    Ok(span)
}

#[cfg(test)] mod tests;
