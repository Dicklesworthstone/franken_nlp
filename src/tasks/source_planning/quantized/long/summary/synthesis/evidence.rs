//! Exact source-quote collection and provenance-preserving final citation lift.
//! A quote crossing a synthetic evidence boundary is refused, even if it occurs
//! elsewhere in the document. Repeated evidence never invents extra support.
use super::*;
use std::collections::BTreeSet;

struct Segment { extent: VerifiedSourceSpan, origins: Vec<VerifiedSourceSpan> }
pub(super) struct Collection { pub text: String, segments: Vec<Segment> }
impl Collection { pub fn segment_count(&self) -> usize { self.segments.len() } }

pub(super) fn collect<C: DecodeStepControl>(source: &str, value: &SourceMapValue, options: SummaryOptions,
    limits: SummarySynthesisLimits, remaining: &mut GroundingBudget, control: &mut C)
    -> Result<Collection, Int8SourceMapError> {
    let mut collection = Collection { text: String::new(), segments: Vec::new() };
    let mut seen = BTreeSet::new();
    let (mut byte, mut scalar, mut evidence_scalars) = (0_usize, 0_usize, 0_usize);
    for (index, chunk) in value.chunks().enumerate() {
        checkpoint(control)?;
        let extent = chunk.source_span;
        if chunk.chunk_id != index || extent.byte_start != byte || extent.scalar_start != scalar
            || extent.byte_end <= byte { return Err(invalid()); }
        let text = source.get(byte..extent.byte_end).ok_or_else(invalid)?;
        charge_scan(remaining, text.len())?;
        scalar = scalar.checked_add(text.chars().count()).ok_or_else(invalid)?;
        if scalar != extent.scalar_end { return Err(invalid()); }
        byte = extent.byte_end;
        let SourceTaskResult::Summarize(summary) = &chunk.native.result else { return Err(invalid()); };
        check_summary(summary, options)?;
        let mut field = 0_usize;
        for (b, bullet) in summary.bullets.iter().enumerate() {
            for (q, citation) in bullet.citations.iter().enumerate() {
                checkpoint(control)?;
                remaining.max_fields = remaining.max_fields.checked_sub(1).ok_or(Int8SourceMapError::WorkLimit)?;
                let local = scan_occurrences(text, &citation.quote, remaining).map_err(scan_error)?;
                if local != citation.spans || citation.occurrence != occurrence(local.len()) { return Err(invalid()); }
                let original = chunk.original_spans.get(field).ok_or_else(invalid)?;
                field = field.checked_add(1).ok_or_else(invalid)?;
                match original.field {
                    SourceMapField::SummaryCitation { bullet, citation } if bullet == b && citation == q => (),
                    _ => return Err(invalid()),
                }
                if original.spans.len() != local.len() { return Err(invalid()); }
                let mut origins = vector(local.len())?;
                for (local, reported) in local.into_iter().zip(&original.spans) {
                    checkpoint(control)?;
                    let lifted = lift(local, extent)?;
                    if &lifted != reported { return Err(invalid()); }
                    origins.push(lifted);
                }
                // Verify every duplicate before deduplicating it. Keep equal
                // quotes from distinct chunks as distinct provenance segments.
                if !seen.insert((chunk.chunk_id, citation.quote.as_str())) { continue; }
                if collection.segments.len() >= limits.max_evidence_segments { return Err(Int8SourceMapError::WorkLimit); }
                let separator = if collection.segments.is_empty() { "" } else { "\n\n" };
                let added = separator.len().checked_add(citation.quote.len()).ok_or(Int8SourceMapError::WorkLimit)?;
                let end = collection.text.len().checked_add(added).filter(|&n| n <= limits.max_evidence_bytes)
                    .ok_or(Int8SourceMapError::WorkLimit)?;
                collection.text.try_reserve(added).map_err(|_| Int8SourceMapError::Allocation)?;
                collection.segments.try_reserve(1).map_err(|_| Int8SourceMapError::Allocation)?;
                collection.text.push_str(separator);
                evidence_scalars = evidence_scalars.checked_add(separator.len()).ok_or_else(invalid)?;
                let start = collection.text.len(); let scalar_start = evidence_scalars;
                evidence_scalars = evidence_scalars.checked_add(citation.quote.chars().count()).ok_or_else(invalid)?;
                collection.text.push_str(&citation.quote);
                collection.segments.push(Segment { extent: VerifiedSourceSpan { byte_start: start, byte_end: end,
                    scalar_start, scalar_end: evidence_scalars }, origins });
            }
        }
        if field != chunk.original_spans.len() { return Err(invalid()); }
    }
    if byte != source.len() { return Err(invalid()); }
    checkpoint(control)?;
    Ok(collection)
}

pub(super) fn lift_summary<C: DecodeStepControl>(source: &str, raw: &SummaryResult, evidence: &Collection,
    options: SummaryOptions, remaining: &mut GroundingBudget, control: &mut C)
    -> Result<Vec<CitedBullet>, Int8SourceMapError> {
    checkpoint(control)?;
    check_summary(raw, options)?;
    if evidence.segments.is_empty() || evidence.text.is_empty() { return Err(invalid()); }
    let mut bullets = vector(raw.bullets.len())?;
    for bullet in &raw.bullets {
        checkpoint(control)?;
        let mut citations = vector(bullet.citations.len())?;
        for citation in &bullet.citations {
            checkpoint(control)?;
            remaining.max_fields = remaining.max_fields.checked_sub(1).ok_or(Int8SourceMapError::WorkLimit)?;
            // Reconstruct the native evidence-document occurrence census, not
            // just the reported spans. Omitted or invented matches are fatal.
            let locals = scan_occurrences(&evidence.text, &citation.quote, remaining).map_err(scan_error)?;
            if locals != citation.spans || citation.occurrence != occurrence(locals.len()) { return Err(invalid()); }
            let mut spans = Vec::new();
            for local in locals {
                checkpoint(control)?;
                // Segments are disjoint and source-ordered. Every native match
                // must be wholly inside ONE segment, never separators or joins.
                let index = evidence.segments.partition_point(|s| s.extent.byte_start <= local.byte_start);
                let segment = index.checked_sub(1).and_then(|i| evidence.segments.get(i)).ok_or_else(invalid)?;
                let e = segment.extent;
                if local.byte_end > e.byte_end || local.scalar_start < e.scalar_start || local.scalar_end > e.scalar_end {
                    return Err(invalid());
                }
                let relative = VerifiedSourceSpan { byte_start: local.byte_start - e.byte_start,
                    byte_end: local.byte_end - e.byte_start, scalar_start: local.scalar_start - e.scalar_start,
                    scalar_end: local.scalar_end - e.scalar_start };
                for &origin in &segment.origins {
                    checkpoint(control)?;
                    remaining.max_matches = remaining.max_matches.checked_sub(1).ok_or(Int8SourceMapError::WorkLimit)?;
                    charge_scan(remaining, citation.quote.len())?;
                    let span = lift(relative, origin)?;
                    if source.get(span.byte_start..span.byte_end) != Some(citation.quote.as_str()) { return Err(invalid()); }
                    spans.try_reserve(1).map_err(|_| Int8SourceMapError::Allocation)?;
                    spans.push(span);
                }
            }
            // Overlapping/repeated map evidence may reach the same original
            // occurrence more than once. Charge fanout, then deduplicate offsets.
            spans.sort_unstable_by_key(|s| (s.byte_start, s.byte_end, s.scalar_start, s.scalar_end));
            spans.dedup();
            if spans.is_empty() { return Err(invalid()); }
            citations.push(SourceCitation { quote: copy(&citation.quote)?, occurrence: occurrence(spans.len()), spans });
        }
        bullets.push(CitedBullet { text: copy(&bullet.text)?, citations });
    }
    checkpoint(control)?;
    Ok(bullets)
}
pub(super) fn check_summary(raw: &SummaryResult, options: SummaryOptions) -> Result<(), Int8SourceMapError> {
    options.validate().map_err(|_| Int8SourceMapError::InvalidLimits)?;
    if raw.schema_version != 1 || raw.task_spec_version != SUMMARIZE_TASK_VERSION || raw.numerics_profile != STRICT_INT8_PROFILE
        || raw.citation_guarantee != CitationGuarantee::StructuralSourceMembership
        || raw.semantic_support != SummarySemanticSupport::NotAssessed || raw.score_space != ScoreSpace::NotComputed
        || raw.bullets.len() > options.max_bullets { return Err(invalid()); }
    for bullet in &raw.bullets {
        if !bullet.text.chars().any(|c| !c.is_whitespace()) || bullet.text.chars().count() > options.max_bullet_scalars
            || bullet.citations.is_empty() || bullet.citations.len() > options.max_citations_per_bullet { return Err(invalid()); }
        for citation in &bullet.citations { check_citation(citation, options.max_quote_scalars)?; }
    }
    Ok(())
}
pub(super) fn check_citation(citation: &SourceCitation, max_scalars: usize) -> Result<(), Int8SourceMapError> {
    let count = citation.quote.chars().count();
    if citation.quote.is_empty() || count > max_scalars || citation.spans.is_empty()
        || citation.occurrence != occurrence(citation.spans.len())
        || citation.spans.iter().any(|s| s.byte_end.checked_sub(s.byte_start) != Some(citation.quote.len())
            || s.scalar_end.checked_sub(s.scalar_start) != Some(count))
        || citation.spans.windows(2).any(|p| p[0].byte_start >= p[1].byte_start || p[0].scalar_start >= p[1].scalar_start) {
        return Err(invalid());
    }
    Ok(())
}
fn occurrence(n: usize) -> SourceOccurrence { if n == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous } }
fn charge_scan(budget: &mut GroundingBudget, n: usize) -> Result<(), Int8SourceMapError> {
    budget.max_scan_steps = budget.max_scan_steps.checked_sub(n as u64).ok_or(Int8SourceMapError::WorkLimit)?; Ok(())
}
fn scan_error(e: FieldGroundingError) -> Int8SourceMapError {
    match e { FieldGroundingError::AllocationRefused => Int8SourceMapError::Allocation,
        FieldGroundingError::WorkBudget | FieldGroundingError::MatchBudget | FieldGroundingError::FieldBudget => Int8SourceMapError::WorkLimit,
        _ => invalid() }
}
fn lift(local: VerifiedSourceSpan, origin: VerifiedSourceSpan) -> Result<VerifiedSourceSpan, Int8SourceMapError> {
    if local.byte_start >= local.byte_end || local.scalar_start >= local.scalar_end { return Err(invalid()); }
    let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(invalid);
    let span = VerifiedSourceSpan { byte_start: add(origin.byte_start, local.byte_start)?, byte_end: add(origin.byte_start, local.byte_end)?,
        scalar_start: add(origin.scalar_start, local.scalar_start)?, scalar_end: add(origin.scalar_start, local.scalar_end)? };
    if span.byte_end > origin.byte_end || span.scalar_end > origin.scalar_end { return Err(invalid()); }
    Ok(span)
}
#[cfg(test)] mod tests;
