//! Private evidence transport: rebase windows and carry verified quote origins.
use super::*;
use evidence::{Collection, Segment};
use std::collections::BTreeMap;

impl Collection {
    pub(super) fn empty() -> Self { Self { text: String::new(), segments: Vec::new() } }
    pub(super) fn range_text(&self, range: Range<usize>) -> Result<&str, Int8SourceMapError> {
        if range.start >= range.end || range.end > self.segments.len() { return Err(invalid()); }
        let start = self.segments[range.start].extent.byte_start;
        let end = self.segments[range.end - 1].extent.byte_end;
        self.text.get(start..end).ok_or_else(invalid)
    }
    pub(super) fn window<C: DecodeStepControl>(&self, range: Range<usize>, control: &mut C)
        -> Result<Self, Int8SourceMapError> {
        checkpoint(control)?;
        let text = copy(self.range_text(range.clone())?)?;
        let base = self.segments[range.start].extent;
        let mut segments = vector(range.len())?;
        for segment in &self.segments[range] {
            checkpoint(control)?;
            let e = segment.extent;
            let mut origins = vector(segment.origins.len())?;
            origins.extend_from_slice(&segment.origins);
            segments.push(Segment { extent: VerifiedSourceSpan {
                byte_start: e.byte_start.checked_sub(base.byte_start).ok_or_else(invalid)?,
                byte_end: e.byte_end.checked_sub(base.byte_start).ok_or_else(invalid)?,
                scalar_start: e.scalar_start.checked_sub(base.scalar_start).ok_or_else(invalid)?,
                scalar_end: e.scalar_end.checked_sub(base.scalar_start).ok_or_else(invalid)? }, origins });
        }
        Ok(Self { text, segments })
    }
    /// Called once per completed native group, after independent citation lift.
    /// Equal quotes in different groups remain separate provenance segments.
    pub(super) fn append_verified<C: DecodeStepControl>(&mut self, source: &str, bullets: &[CitedBullet],
        options: SummaryOptions, limits: SummarySynthesisLimits, remaining: &mut GroundingBudget, control: &mut C)
        -> Result<(), Int8SourceMapError> {
        let mut seen = BTreeMap::<&str, &[VerifiedSourceSpan]>::new();
        for bullet in bullets {
            for citation in &bullet.citations {
                checkpoint(control)?;
                evidence::check_citation(citation, options.max_quote_scalars)?;
                remaining.max_fields = remaining.max_fields.checked_sub(1).ok_or(Int8SourceMapError::WorkLimit)?;
                // Charge/recheck every proposal before deduplication. Origins
                // are reachable via lift_summary, never a whole-source search.
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
                if self.segments.len() >= limits.max_evidence_segments { return Err(Int8SourceMapError::WorkLimit); }
                let separator = if self.segments.is_empty() { "" } else { "\n\n" };
                let added = separator.len().checked_add(citation.quote.len()).ok_or(Int8SourceMapError::WorkLimit)?;
                let end = self.text.len().checked_add(added).filter(|&n| n <= limits.max_evidence_bytes)
                    .ok_or(Int8SourceMapError::WorkLimit)?;
                let scalar_start = self.segments.last().map_or(0, |s| s.extent.scalar_end)
                    .checked_add(separator.len()).ok_or_else(invalid)?;
                let scalar_end = scalar_start.checked_add(citation.quote.chars().count()).ok_or_else(invalid)?;
                self.text.try_reserve(added).map_err(|_| Int8SourceMapError::Allocation)?;
                self.segments.try_reserve(1).map_err(|_| Int8SourceMapError::Allocation)?;
                let mut origins = vector(citation.spans.len())?;
                origins.extend_from_slice(&citation.spans);
                self.text.push_str(separator);
                let start = self.text.len();
                self.text.push_str(&citation.quote);
                self.segments.push(Segment { extent: VerifiedSourceSpan { byte_start: start, byte_end: end,
                    scalar_start, scalar_end }, origins });
            }
        }
        checkpoint(control)?;
        Ok(())
    }
}

/// Reconstruct the next frontier's geometry from retained, verified group
/// citations. Used only for completed-receipt checks, not as an execution API.
pub(super) fn selected_size(passes: &[SummaryHierarchyPass]) -> Result<EvidenceSize, Int8SourceMapError> {
    let mut size = EvidenceSize { segments: 0, bytes: 0 };
    for pass in passes {
        let mut seen = BTreeMap::<&str, &[VerifiedSourceSpan]>::new();
        for bullet in &pass.bullets { for citation in &bullet.citations {
            if let Some(old) = seen.insert(citation.quote.as_str(), &citation.spans) {
                if old != citation.spans.as_slice() { return Err(invalid()); }
                continue;
            }
            size.bytes = size.bytes.checked_add(if size.segments == 0 { 0 } else { 2 })
                .and_then(|n| n.checked_add(citation.quote.len())).ok_or(Int8SourceMapError::WorkLimit)?;
            size.segments = size.segments.checked_add(1).ok_or(Int8SourceMapError::WorkLimit)?;
        } }
    }
    Ok(size)
}
