//! Cited long-document summaries with deterministic, evidence-preserving reduction.
//!
//! Exact bullet text is the ONLY deduplication key. Every independently checked
//! citation survives intermediate reductions; top-k is applied only once, after
//! the complete child set is known. No reducer rephrases a bullet, invents an
//! evidence span, reconciles contradictions, or treats frequency as confidence.

use std::{collections::BTreeMap, error::Error, fmt, io, mem::size_of};
use serde::{Deserialize, Serialize};
use crate::{canonjson, native_engine::hf_bf16_eager::HF_BF16_EAGER_PROFILE,
    tasks::{ir::ScoreSpace, mapreduce::{MapOutput, MapReduceTask, ReduceInput, ReductionPolicy, SourceChunk},
        summarize::{CitationGuarantee, SourceCitation, SummaryError, SummaryOptions, SummaryResult,
            SummarySemanticSupport, SUMMARIZE_TASK_VERSION}},
    validation::grounded_fields::{GroundingBudget, SourceOccurrence, VerifiedSourceSpan, scan_occurrences}};

pub const SUMMARY_REDUCTION_POLICY: &str = "exact-bullet-union-chunk-support-rank-sum-source-v1";

/// Static composition boundary. A pass must bound its own model work and keep
/// its admission guards alive until its native work and result delivery finish.
/// Public/deserialized SummaryResult values are independently reverified here.
pub trait SummaryPass {
    type Error;
    fn options(&self) -> SummaryOptions;
    fn run(&mut self, chunk: &SourceChunk<'_>) -> Result<SummaryResult, Self::Error>;
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusSummaryLimits {
    pub max_unique_bullets: usize,
    pub max_citations: usize,
    pub max_evidence_spans: usize,
    /// Canonical value bytes AND a conservative stored-field charge. Neither
    /// is an allocator-observed heap or peak-memory measurement.
    pub max_value_bytes: usize,
    /// Nonrenewable independent verification work across every map call.
    pub max_scan_steps: u64,
}
impl Default for CorpusSummaryLimits {
    fn default() -> Self {
        Self { max_unique_bullets: 4096, max_citations: 16_384, max_evidence_spans: 65_536,
            max_value_bytes: 4 * 1024 * 1024, max_scan_steps: 512 * 1024 * 1024 }
    }
}
impl CorpusSummaryLimits {
    pub fn validate(self) -> Result<(), SummaryError> {
        if !(1..=65_536).contains(&self.max_unique_bullets)
            || !(1..=1_000_000).contains(&self.max_citations)
            || !(1..=1_000_000).contains(&self.max_evidence_spans)
            || !(1..=64 * 1024 * 1024).contains(&self.max_value_bytes) || self.max_scan_steps == 0
        { return Err(SummaryError::InvalidOptions); }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ChunkBulletEvidence {
    pub chunk_id: usize,
    /// Earliest one-based local rank, regardless of duplicate occurrences of
    /// the same bullet within this map. One supporting vote per chunk.
    pub local_rank: usize,
    /// All exact occurrences WITHIN THIS CHUNK, lifted to ORIGINAL-document
    /// byte/scalar coordinates. Not an exhaustive whole-document quote search.
    pub citations: Vec<SourceCitation>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CorpusBullet {
    pub text: String,
    pub rank_sum: u64,
    /// Strictly increasing chunk ids; citations within each entry sort by quote.
    pub evidence: Vec<ChunkBulletEvidence>,
}

/// Private construction prevents deserialized values becoming reduction authority.
#[derive(Serialize)]
pub struct SummaryAggregate {
    first_chunk: usize,
    end_chunk: usize,
    bullets: Vec<CorpusBullet>,
    forward_positions: u64,
    projected_logits: u64,
    mask_node_visit_charge: u64,
}
impl SummaryAggregate {
    pub fn bullets(&self) -> &[CorpusBullet] { &self.bullets }
    pub fn mapped_chunks(&self) -> usize { self.end_chunk - self.first_chunk }
    /// Most distinct supporting chunks, then smallest sum of local ranks,
    /// earliest source occurrence, then byte-lexical bullet text. These are
    /// deterministic selection heuristics, not calibrated importance scores.
    pub fn into_ranked(mut self, max_bullets: usize, max_bytes: usize) -> Result<CorpusSummaryResult, SummaryError> {
        if !(1..=1024).contains(&max_bullets) || !(1..=64 * 1024 * 1024).contains(&max_bytes) {
            return Err(SummaryError::InvalidOptions);
        }
        self.bullets.sort_by(|a, b| b.evidence.len().cmp(&a.evidence.len())
            .then_with(|| a.rank_sum.cmp(&b.rank_sum))
            .then_with(|| first_byte(a).cmp(&first_byte(b)))
            .then_with(|| a.text.as_bytes().cmp(b.text.as_bytes())));
        let omitted_bullets = self.bullets.len().saturating_sub(max_bullets);
        self.bullets.truncate(max_bullets);
        let result = CorpusSummaryResult { schema_version: 1,
            task_spec_version: SUMMARIZE_TASK_VERSION.to_owned(), reduction_policy: SUMMARY_REDUCTION_POLICY.to_owned(),
            citation_guarantee: CitationGuarantee::StructuralSourceMembership,
            semantic_support: SummarySemanticSupport::NotAssessed, score_space: ScoreSpace::NotComputed,
            mapped_chunks: self.end_chunk - self.first_chunk, requested_max_bullets: max_bullets,
            omitted_bullets, bullets: self.bullets, forward_positions: self.forward_positions,
            projected_logits: self.projected_logits, mask_node_visit_charge: self.mask_node_visit_charge,
            untrusted_fields: ["bullets".to_owned()], warnings: [
                SummaryWarning::LossyMapAndFinalSelection, SummaryWarning::SingleContextEquivalenceNotEstablished,
                SummaryWarning::SemanticConflictsNotReconciled, SummaryWarning::ChunkBoundariesMaySplitContext,
                SummaryWarning::SupportFrequencyIsNotImportanceOrConfidence] };
        check_bytes(&result, max_bytes)?;
        Ok(result)
    }
}
fn first_byte(bullet: &CorpusBullet) -> usize {
    bullet.evidence.iter().flat_map(|e| &e.citations).flat_map(|c| &c.spans)
        .map(|s| s.byte_start).min().unwrap_or(usize::MAX)
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryWarning {
    LossyMapAndFinalSelection, SingleContextEquivalenceNotEstablished,
    SemanticConflictsNotReconciled, ChunkBoundariesMaySplitContext,
    SupportFrequencyIsNotImportanceOrConfidence,
}
#[derive(Serialize)]
pub struct CorpusSummaryResult {
    pub schema_version: u32,
    pub task_spec_version: String,
    pub reduction_policy: String,
    pub citation_guarantee: CitationGuarantee,
    pub semantic_support: SummarySemanticSupport,
    pub score_space: ScoreSpace,
    pub mapped_chunks: usize,
    pub requested_max_bullets: usize,
    pub omitted_bullets: usize,
    pub bullets: Vec<CorpusBullet>,
    pub forward_positions: u64,
    pub projected_logits: u64,
    pub mask_node_visit_charge: u64,
    pub untrusted_fields: [String; 1],
    pub warnings: [SummaryWarning; 5],
}

#[derive(Debug)]
pub enum CorpusSummaryError<E> { Pass(E), Summary(SummaryError), Poisoned }
impl<E> fmt::Display for CorpusSummaryError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self { Self::Pass(_) => "corpus summary map failed",
            Self::Summary(_) => "corpus summary validation or reduction failed",
            Self::Poisoned => "corpus summary cannot resume a failed operation" })
    }
}
impl<E: Error + 'static> Error for CorpusSummaryError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Pass(e) => Some(e), Self::Summary(e) => Some(e), Self::Poisoned => None }
    }
}

/// Feed to tasks::mapreduce::execute. All intermediate candidates survive or
/// execution fails, so successful final selection is independent of fan-in.
/// A failed map/reduce (including an unwind) permanently poisons this instance.
pub struct CorpusSummaryTask<P> {
    pass: P,
    options: SummaryOptions,
    limits: CorpusSummaryLimits,
    scan_remaining: u64,
    next_chunk: usize,
    failed: bool,
}
impl<P: SummaryPass> CorpusSummaryTask<P> {
    pub fn new(pass: P, limits: CorpusSummaryLimits) -> Result<Self, SummaryError> {
        limits.validate()?;
        let options = pass.options(); options.validate()?;
        Ok(Self { pass, options, limits, scan_remaining: limits.max_scan_steps, next_chunk: 0, failed: false })
    }
    pub fn scan_steps_remaining(&self) -> u64 { self.scan_remaining }
    pub fn pass(&self) -> &P { &self.pass }
    pub fn into_pass(self) -> P { self.pass }
}
impl<P: SummaryPass> MapReduceTask for CorpusSummaryTask<P> {
    type Value = SummaryAggregate;
    type Error = CorpusSummaryError<P::Error>;
    fn policy(&self) -> ReductionPolicy {
        ReductionPolicy { id: SUMMARY_REDUCTION_POLICY, may_discard_information: true }
    }
    fn map_batch(&mut self, chunks: &[SourceChunk<'_>]) -> Result<Vec<MapOutput<Self::Value>>, Self::Error> {
        if self.failed { return Err(CorpusSummaryError::Poisoned); }
        self.failed = true;
        if chunks.is_empty() || self.pass.options() != self.options {
            return Err(CorpusSummaryError::Summary(SummaryError::InvalidOptions));
        }
        let mut outputs = Vec::new();
        outputs.try_reserve_exact(chunks.len()).map_err(|_| CorpusSummaryError::Summary(SummaryError::AllocationRefused))?;
        for chunk in chunks {
            if chunk.id() != self.next_chunk { return Err(CorpusSummaryError::Summary(SummaryError::InvalidResult)); }
            let raw = self.pass.run(chunk).map_err(CorpusSummaryError::Pass)?;
            let value = map_value(chunk, raw, self.options, self.limits, &mut self.scan_remaining)
                .map_err(CorpusSummaryError::Summary)?;
            self.next_chunk = self.next_chunk.checked_add(1).ok_or(CorpusSummaryError::Summary(SummaryError::InvalidResult))?;
            outputs.push(MapOutput { chunk_id: chunk.id(), value });
        }
        self.failed = false; Ok(outputs)
    }
    fn reduce(&mut self, input: ReduceInput<'_, Self::Value>) -> Result<Self::Value, Self::Error> {
        if self.failed { return Err(CorpusSummaryError::Poisoned); }
        self.failed = true;
        let value = merge_values(input.children.iter().map(|n| n.value()), self.limits)
            .map_err(CorpusSummaryError::Summary)?;
        self.failed = false; Ok(value)
    }
}

fn map_value(chunk: &SourceChunk<'_>, raw: SummaryResult, options: SummaryOptions,
    limits: CorpusSummaryLimits, remaining: &mut u64) -> Result<SummaryAggregate, SummaryError> {
    if raw.schema_version != 1 || raw.task_spec_version != SUMMARIZE_TASK_VERSION
        || raw.numerics_profile != HF_BF16_EAGER_PROFILE || raw.score_space != ScoreSpace::NotComputed
        || raw.citation_guarantee != CitationGuarantee::StructuralSourceMembership
        || raw.semantic_support != SummarySemanticSupport::NotAssessed || raw.bullets.len() > options.max_bullets
    { return Err(SummaryError::InvalidResult); }
    check_bytes(&raw, limits.max_value_bytes)?;
    let mut grouped: BTreeMap<String, CorpusBullet> = BTreeMap::new();
    let mut budget = GroundingBudget { max_fields: limits.max_citations,
        max_matches: limits.max_evidence_spans, max_scan_steps: *remaining };
    let mut citation_count = 0_usize;
    for (index, mut bullet) in raw.bullets.into_iter().enumerate() {
        if !bullet.text.chars().any(|c| !c.is_whitespace()) || bullet.text.chars().count() > options.max_bullet_scalars
            || bullet.citations.is_empty() || bullet.citations.len() > options.max_citations_per_bullet
        { return Err(SummaryError::InvalidResult); }
        citation_count = limited_add(citation_count, bullet.citations.len(), limits.max_citations)?;
        for citation in &mut bullet.citations {
            if citation.quote.is_empty() || citation.quote.chars().count() > options.max_quote_scalars
                || citation.spans.len() > limits.max_evidence_spans { return Err(SummaryError::InvalidResult); }
            // Independent linear matching checks completeness as well as validity,
            // including overlapping matches; model-provided offsets are not authority.
            let checked = scan_occurrences(chunk.text(), &citation.quote, &mut budget);
            *remaining = budget.max_scan_steps;
            let mut spans = checked.map_err(|e| match e {
                crate::validation::grounded_fields::FieldGroundingError::AllocationRefused => SummaryError::AllocationRefused,
                crate::validation::grounded_fields::FieldGroundingError::WorkBudget
                | crate::validation::grounded_fields::FieldGroundingError::MatchBudget => SummaryError::OutputBudgetExceeded,
                _ => SummaryError::InvalidResult,
            })?;
            if spans != citation.spans || citation.occurrence != if spans.len() == 1 {
                SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }
            { return Err(SummaryError::InvalidResult); }
            let origin = chunk.span();
            for span in &mut spans {
                span.byte_start = span.byte_start.checked_add(origin.byte_start).ok_or(SummaryError::InvalidResult)?;
                span.byte_end = span.byte_end.checked_add(origin.byte_start).ok_or(SummaryError::InvalidResult)?;
                span.scalar_start = span.scalar_start.checked_add(origin.scalar_start).ok_or(SummaryError::InvalidResult)?;
                span.scalar_end = span.scalar_end.checked_add(origin.scalar_start).ok_or(SummaryError::InvalidResult)?;
            }
            citation.spans = spans;
        }
        bullet.citations.sort_by(|a, b| a.quote.cmp(&b.quote));
        bullet.citations.dedup_by(|a, b| a.quote == b.quote);
        if let Some(existing) = grouped.get_mut(&bullet.text) {
            let citations = &mut existing.evidence[0].citations;
            citations.try_reserve(bullet.citations.len()).map_err(|_| SummaryError::AllocationRefused)?;
            citations.extend(bullet.citations);
            citations.sort_by(|a, b| a.quote.cmp(&b.quote));
            citations.dedup_by(|a, b| a.quote == b.quote);
        } else {
            if grouped.len() == limits.max_unique_bullets { return Err(SummaryError::OutputBudgetExceeded); }
            grouped.insert(copy_text(&bullet.text)?, CorpusBullet { text: bullet.text, rank_sum: (index + 1) as u64,
                evidence: vec![ChunkBulletEvidence { chunk_id: chunk.id(), local_rank: index + 1, citations: bullet.citations }] });
        }
    }
    let mut bullets = Vec::new(); bullets.try_reserve_exact(grouped.len()).map_err(|_| SummaryError::AllocationRefused)?;
    bullets.extend(grouped.into_values());
    let value = SummaryAggregate { first_chunk: chunk.id(), end_chunk: chunk.id().checked_add(1).ok_or(SummaryError::InvalidResult)?,
        bullets, forward_positions: raw.forward_positions, projected_logits: raw.projected_logits,
        mask_node_visit_charge: raw.mask_node_visit_charge };
    let mut charge = Charge::default();
    for bullet in &value.bullets { charge.bullet(&bullet.text, limits)?; for e in &bullet.evidence { charge.evidence(e, limits)?; } }
    check_bytes(&value, limits.max_value_bytes)?; Ok(value)
}

fn merge_values<'a>(values: impl IntoIterator<Item = &'a SummaryAggregate>, limits: CorpusSummaryLimits)
    -> Result<SummaryAggregate, SummaryError> {
    let mut merged: BTreeMap<&str, CorpusBullet> = BTreeMap::new();
    let mut total = SummaryAggregate { first_chunk: 0, end_chunk: 0, bullets: Vec::new(),
        forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 0 };
    let mut first = true; let mut charge = Charge::default();
    for value in values {
        if value.first_chunk >= value.end_chunk || (!first && total.end_chunk != value.first_chunk) {
            return Err(SummaryError::InvalidResult);
        }
        if first { total.first_chunk = value.first_chunk; first = false; }
        total.end_chunk = value.end_chunk;
        total.forward_positions = total.forward_positions.checked_add(value.forward_positions).ok_or(SummaryError::OutputBudgetExceeded)?;
        total.projected_logits = total.projected_logits.checked_add(value.projected_logits).ok_or(SummaryError::OutputBudgetExceeded)?;
        total.mask_node_visit_charge = total.mask_node_visit_charge.checked_add(value.mask_node_visit_charge).ok_or(SummaryError::OutputBudgetExceeded)?;
        for bullet in &value.bullets {
            if !merged.contains_key(bullet.text.as_str()) {
                charge.bullet(&bullet.text, limits)?;
                merged.insert(&bullet.text, CorpusBullet { text: copy_text(&bullet.text)?, rank_sum: 0, evidence: Vec::new() });
            }
            let target = merged.get_mut(bullet.text.as_str()).ok_or(SummaryError::InvalidResult)?;
            if target.evidence.last().zip(bullet.evidence.first()).is_some_and(|(a, b)| a.chunk_id >= b.chunk_id) {
                return Err(SummaryError::InvalidResult);
            }
            target.rank_sum = target.rank_sum.checked_add(bullet.rank_sum).ok_or(SummaryError::OutputBudgetExceeded)?;
            for evidence in &bullet.evidence {
                charge.evidence(evidence, limits)?; // before allocating the copied evidence
                target.evidence.try_reserve(1).map_err(|_| SummaryError::AllocationRefused)?;
                target.evidence.push(copy_evidence(evidence)?);
            }
        }
    }
    if first { return Err(SummaryError::InvalidResult); }
    total.bullets.try_reserve_exact(merged.len()).map_err(|_| SummaryError::AllocationRefused)?;
    total.bullets.extend(merged.into_values()); check_bytes(&total, limits.max_value_bytes)?; Ok(total)
}
fn copy_text(text: &str) -> Result<String, SummaryError> {
    let mut copy = String::new(); copy.try_reserve_exact(text.len()).map_err(|_| SummaryError::AllocationRefused)?;
    copy.push_str(text); Ok(copy)
}
fn copy_evidence(source: &ChunkBulletEvidence) -> Result<ChunkBulletEvidence, SummaryError> {
    let mut citations = Vec::new(); citations.try_reserve_exact(source.citations.len()).map_err(|_| SummaryError::AllocationRefused)?;
    for citation in &source.citations {
        let mut spans = Vec::new(); spans.try_reserve_exact(citation.spans.len()).map_err(|_| SummaryError::AllocationRefused)?;
        spans.extend_from_slice(&citation.spans);
        citations.push(SourceCitation { quote: copy_text(&citation.quote)?, occurrence: citation.occurrence, spans });
    }
    Ok(ChunkBulletEvidence { chunk_id: source.chunk_id, local_rank: source.local_rank, citations })
}
fn limited_add(a: usize, b: usize, cap: usize) -> Result<usize, SummaryError> {
    a.checked_add(b).filter(|&n| n <= cap).ok_or(SummaryError::OutputBudgetExceeded)
}
#[derive(Default)]
struct Charge { bullets: usize, citations: usize, spans: usize, bytes: usize }
impl Charge {
    fn bullet(&mut self, text: &str, limits: CorpusSummaryLimits) -> Result<(), SummaryError> {
        self.bullets = limited_add(self.bullets, 1, limits.max_unique_bullets)?;
        self.bytes = limited_add(self.bytes, size_of::<CorpusBullet>(), limits.max_value_bytes)?;
        self.bytes = limited_add(self.bytes, text.len(), limits.max_value_bytes)?; Ok(())
    }
    fn evidence(&mut self, e: &ChunkBulletEvidence, limits: CorpusSummaryLimits) -> Result<(), SummaryError> {
        self.bytes = limited_add(self.bytes, size_of::<ChunkBulletEvidence>(), limits.max_value_bytes)?;
        self.citations = limited_add(self.citations, e.citations.len(), limits.max_citations)?;
        for c in &e.citations {
            self.spans = limited_add(self.spans, c.spans.len(), limits.max_evidence_spans)?;
            let bytes = c.spans.len().checked_mul(size_of::<VerifiedSourceSpan>()).ok_or(SummaryError::OutputBudgetExceeded)?;
            for n in [bytes, c.quote.len(), size_of::<SourceCitation>()] {
                self.bytes = limited_add(self.bytes, n, limits.max_value_bytes)?;
            }
        }
        Ok(())
    }
}
/// No-allocation size refusal before canonicalization. Fixed integer/string
/// data only; canonical bytes are still checked independently before success.
pub(crate) fn check_bytes(value: &impl Serialize, cap: usize) -> Result<(), SummaryError> {
    struct Counter { remaining: usize, overflow: bool }
    impl io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            match self.remaining.checked_sub(bytes.len()) {
                Some(n) => { self.remaining = n; Ok(bytes.len()) },
                None => { self.overflow = true; Err(io::Error::other("summary byte budget")) },
            }
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    let mut counter = Counter { remaining: cap, overflow: false };
    if serde_json::to_writer(&mut counter, value).is_err() {
        return Err(if counter.overflow { SummaryError::OutputBudgetExceeded } else { SummaryError::Serialization });
    }
    if canonjson::canonical_bytes(value).map_err(|_| SummaryError::Serialization)?.len() > cap {
        return Err(SummaryError::OutputBudgetExceeded);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
