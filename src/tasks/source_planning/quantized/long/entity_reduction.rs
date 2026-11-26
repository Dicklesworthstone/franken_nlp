//! Exact union of typed surface proposals from an authenticated native document map.
//! Occurrences are scoped to the chunks which proposed that surface/type. Two
//! equal surfaces can name different people; this reducer does no coreference.
use super::*;
use crate::{tasks::ner::{EntityType, NamedEntity}, validation::grounded_fields::SourceOccurrence};
use std::collections::BTreeMap;

pub const INT8_ENTITY_REDUCTION_EXECUTION: &str = "portable-int8-ner-document-union-v1";
pub const ENTITY_REDUCTION_POLICY: &str = "exact-surface-type-contributing-chunks-v1";

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Int8EntityLimits {
    pub max_proposals: usize,
    pub max_evidence_spans: usize,
    pub max_scan_work: u64,
    pub max_result_bytes: usize,
}
impl Default for Int8EntityLimits {
    fn default() -> Self {
        Self { max_proposals: 16_384, max_evidence_spans: 65_536,
            max_scan_work: 64 * 1024 * 1024, max_result_bytes: 4 * 1024 * 1024 }
    }
}
impl Int8EntityLimits {
    /// Additional modeled storage alongside the already charged complete map.
    pub fn workspace_bytes(self) -> Result<u64, Int8SourceMapError> {
        if !(1..=65_536).contains(&self.max_proposals)
            || !(1..=1_048_576).contains(&self.max_evidence_spans)
            || !(1..=1_000_000_000_000).contains(&self.max_scan_work)
            || !(1..=64 * 1024 * 1024).contains(&self.max_result_bytes) {
            return Err(Int8SourceMapError::InvalidLimits);
        }
        Ok(self.max_proposals as u64 * 512 + self.max_evidence_spans as u64 * 64
            + self.max_result_bytes as u64)
    }
    pub fn validate(self, mapping: Int8SourceMapLimits) -> Result<(), Int8SourceMapError> {
        self.workspace_bytes()?;
        if self.max_result_bytes > mapping.reduction.max_result_bytes {
            return Err(Int8SourceMapError::InvalidLimits);
        }
        Ok(())
    }
}

#[derive(Serialize)]
pub struct EntityChunkEvidence {
    pub chunk_id: usize,
    /// Ambiguity applies to this native proposal in its own source chunk.
    pub occurrence: SourceOccurrence,
    /// All matching occurrences in this chunk, in original document coordinates.
    pub spans: Vec<VerifiedSourceSpan>,
}
#[derive(Serialize)]
pub struct DocumentEntity {
    pub text: String,
    pub entity_type: EntityType,
    pub contributing_chunks: Vec<EntityChunkEvidence>,
}
#[derive(Serialize)]
pub struct Int8DocumentEntityRun {
    pub schema_version: u32,
    pub execution: &'static str,
    pub reduction_policy: &'static str,
    pub numerics_profile: &'static str,
    pub source_span: VerifiedSourceSpan,
    pub mapped_chunks: usize,
    pub planned_model_work: Int8Work,
    pub model_work: Int8Work,
    pub reserved_mask_node_visits: u64,
    pub mask_node_visit_charge: u64,
    pub proposal_count: usize,
    pub evidence_span_count: usize,
    pub verification_scan_work: u64,
    pub entities: Vec<DocumentEntity>,
}

impl PreparedInt8SourceMap<'_> {
    pub fn check_entities(&self, limits: Int8EntityLimits) -> Result<(), Int8SourceMapError> {
        limits.validate(self.limits)?;
        if self.expected.task != BuiltInTask::Ner { return Err(Int8SourceMapError::InvalidLimits); }
        Ok(())
    }
}
impl Int8SourceMapPreflight {
    pub fn reduce_entities_with_control<C: DecodeStepControl>(&self, map: &Int8SourceMapRun,
        limits: Int8EntityLimits, control: &mut C) -> Result<Int8DocumentEntityRun, Int8SourceMapError> {
        checkpoint(control)?;
        limits.workspace_bytes()?;
        if self.task != BuiltInTask::Ner || limits.max_result_bytes as u64 > self.max_result_bytes {
            return Err(Int8SourceMapError::InvalidLimits);
        }
        // This checks the private request and entire map payload commitment.
        // The bound native map already compared every local slice with the
        // original UTF-8 bytes and verified both scalar endpoints when lifting.
        self.verify_completed_with_control(map, control)?;
        let mut union = Union::new(limits);
        for chunk in map.mapped.root().value().chunks() {
            checkpoint(control)?;
            let SourceTaskResult::Ner(ner) = &chunk.native.result else { return Err(invalid()); };
            if ner.entities.len() != chunk.original_spans.len() { return Err(invalid()); }
            for (index, (entity, field)) in ner.entities.iter().zip(&chunk.original_spans).enumerate() {
                if !matches!(field.field, SourceMapField::Entity { index: field } if field == index) {
                    return Err(invalid());
                }
                union.add(chunk.chunk_id, chunk.source_span, entity, &field.spans, control)?;
            }
        }
        let mut result = Int8DocumentEntityRun { schema_version: 1, execution: INT8_ENTITY_REDUCTION_EXECUTION,
            reduction_policy: ENTITY_REDUCTION_POLICY, numerics_profile: STRICT_INT8_PROFILE,
            source_span: self.source_span, mapped_chunks: self.chunks, planned_model_work: map.planned_model_work,
            model_work: map.model_work, reserved_mask_node_visits: map.reserved_mask_node_visits,
            mask_node_visit_charge: map.mask_node_visit_charge, proposal_count: union.proposals,
            evidence_span_count: union.spans, verification_scan_work: union.work, entities: Vec::new() };
        result.entities.try_reserve_exact(union.rows.len()).map_err(|_| Int8SourceMapError::Allocation)?;
        for ((entity_type, text), contributing_chunks) in union.rows {
            checkpoint(control)?;
            let mut owned = String::new();
            owned.try_reserve_exact(text.len()).map_err(|_| Int8SourceMapError::Allocation)?;
            owned.push_str(text);
            result.entities.push(DocumentEntity { text: owned, entity_type, contributing_chunks });
        }
        // Stream the final JSON through the same checked writer used by the
        // map. No serialized document-sized buffer is needed to enforce the cap.
        question::binding::composed_result(b"fnlp-int8-entity-document-result-v1", self.request_binding,
            &[map.result_binding], &result, limits.max_result_bytes as u64, control)?;
        Ok(result)
    }
}

fn invalid() -> Int8SourceMapError { Int8SourceError::InvalidResult.into() }
struct Union<'a> {
    limits: Int8EntityLimits,
    rows: BTreeMap<(EntityType, &'a str), Vec<EntityChunkEvidence>>,
    proposals: usize,
    spans: usize,
    work: u64,
    text_bytes: usize,
}
impl<'a> Union<'a> {
    fn new(limits: Int8EntityLimits) -> Self {
        Self { limits, rows: BTreeMap::new(), proposals: 0, spans: 0, work: 0, text_bytes: 0 }
    }
    fn add<C: DecodeStepControl>(&mut self, chunk_id: usize, chunk: VerifiedSourceSpan,
        entity: &'a NamedEntity, original: &[VerifiedSourceSpan], control: &mut C) -> Result<(), Int8SourceMapError> {
        checkpoint(control)?;
        if self.proposals == self.limits.max_proposals { return Err(Int8SourceMapError::WorkLimit); }
        self.proposals += 1;
        self.work = self.work.checked_add(entity.text.len() as u64)
            .and_then(|n| n.checked_add(original.len() as u64)).filter(|&n| n <= self.limits.max_scan_work)
            .ok_or(Int8SourceMapError::WorkLimit)?;
        let occurrence = if original.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous };
        if entity.text.is_empty() || original.is_empty() || entity.spans.len() != original.len()
            || entity.occurrence != occurrence { return Err(invalid()); }

        let key = (entity.entity_type, entity.text.as_str());
        let row = self.rows.entry(key).or_default();
        if row.is_empty() {
            self.text_bytes = self.text_bytes.checked_add(entity.text.len())
                .filter(|&n| n <= self.limits.max_result_bytes).ok_or(Int8SourceMapError::WorkLimit)?;
        }
        let repeated = row.last().filter(|last| last.chunk_id == chunk_id);
        if repeated.is_some_and(|last| last.spans.len() != original.len()) { return Err(invalid()); }
        if repeated.is_none() {
            self.spans = self.spans.checked_add(original.len())
                .filter(|&n| n <= self.limits.max_evidence_spans).ok_or(Int8SourceMapError::WorkLimit)?;
        }
        for (i, (local, global)) in entity.spans.iter().zip(original).enumerate() {
            checkpoint(control)?;
            if local.byte_end.checked_sub(local.byte_start) != Some(entity.text.len())
                || global.scalar_end <= global.scalar_start || global.byte_end > chunk.byte_end
                || global.scalar_end > chunk.scalar_end
                || chunk.byte_start.checked_add(local.byte_start) != Some(global.byte_start)
                || chunk.byte_start.checked_add(local.byte_end) != Some(global.byte_end)
                || chunk.scalar_start.checked_add(local.scalar_start) != Some(global.scalar_start)
                || chunk.scalar_start.checked_add(local.scalar_end) != Some(global.scalar_end)
                || (i > 0 && original[i - 1].byte_start >= global.byte_start)
                || repeated.is_some_and(|last| last.spans[i] != *global) {
                return Err(invalid());
            }
        }
        if repeated.is_none() {
            let mut spans = Vec::new();
            spans.try_reserve_exact(original.len()).map_err(|_| Int8SourceMapError::Allocation)?;
            for page in original.chunks(1024) {
                checkpoint(control)?;
                spans.extend_from_slice(page);
            }
            row.try_reserve(1).map_err(|_| Int8SourceMapError::Allocation)?;
            row.push(EntityChunkEvidence { chunk_id, occurrence, spans });
        }
        Ok(())
    }
}

#[cfg(test)] mod tests;
