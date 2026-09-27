//! Detector constructor for one COMPLETE original source. NER evidence remains
//! chunk-local until independently rescanned; rule detectors never see chunks.
use std::collections::BTreeSet;
use crate::{native_engine::strict_int8::STRICT_INT8_PROFILE,
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace, ner::{NerOptions, NerResult, NER_TASK_VERSION}},
    validation::grounded_fields::{GroundingBudget, SourceOccurrence, VerifiedSourceSpan, scan_occurrences}};
use super::super::{Detection, Detector, RedactError, RedactionRequest,
    detectors::{self, RuleSet}, union::{self, DetectedDocument}};

pub(super) struct Detections<'s> {
    source: &'s str,
    hits: Vec<Detection>,
    rules: RuleSet,
    options: NerOptions,
    max_detections: usize,
    next_chunk: usize,
    byte_end: usize,
    scalar_end: usize,
    failed: bool,
}
impl<'s> Detections<'s> {
    pub fn new(source: &'s str, request: &RedactionRequest, options: &NerOptions) -> Result<Self, RedactError> {
        options.validate().map_err(|_| RedactError::InvalidOptions)?;
        let hits = detectors::detect(source, &request.rules, request.rule_budget)?;
        Ok(Self { source, hits, rules: request.rules.clone(), options: options.clone(),
            max_detections: request.rule_budget.max_detections, next_chunk: 0, byte_end: 0, scalar_end: 0, failed: false })
    }
    pub fn push(&mut self, chunk_id: usize, origin: VerifiedSourceSpan, result: &NerResult,
        grounding: &mut GroundingBudget) -> Result<(), RedactError> {
        if self.failed { return Err(RedactError::InvalidNerEvidence); }
        self.failed = true;
        if chunk_id != self.next_chunk || origin.byte_start != self.byte_end || origin.scalar_start != self.scalar_end
            || origin.byte_start >= origin.byte_end { return Err(RedactError::InvalidSpan); }
        let text = self.source.get(origin.byte_start..origin.byte_end).ok_or(RedactError::InvalidSpan)?;
        if origin.scalar_start.checked_add(text.chars().count()) != Some(origin.scalar_end) { return Err(RedactError::InvalidSpan); }
        if result.schema_version != 1 || result.task_spec_version != NER_TASK_VERSION || result.numerics_profile != STRICT_INT8_PROFILE
            || result.score_space != ScoreSpace::NotComputed || result.grounding != ExtractionGrounding::SourceMembership
            || result.entities.len() > self.options.max_entities { return Err(RedactError::InvalidNerEvidence); }
        grounding.max_fields = grounding.max_fields.checked_sub(result.entities.len()).ok_or(RedactError::DetectionBudget)?;
        for entity in &result.entities {
            if entity.text.is_empty() || entity.text.len() > text.len()
                || entity.text.chars().count() > self.options.max_mention_scalars
                || !self.options.types.contains(&entity.entity_type) { return Err(RedactError::InvalidNerEvidence); }
            let spans = scan_occurrences(text, &entity.text, grounding).map_err(|error| match error {
                crate::validation::grounded_fields::FieldGroundingError::AllocationRefused => RedactError::AllocationRefused,
                crate::validation::grounded_fields::FieldGroundingError::WorkBudget => RedactError::WorkBudget,
                crate::validation::grounded_fields::FieldGroundingError::MatchBudget
                | crate::validation::grounded_fields::FieldGroundingError::FieldBudget => RedactError::DetectionBudget,
                _ => RedactError::InvalidNerEvidence,
            })?;
            if spans.is_empty() || spans != entity.spans || entity.occurrence != (if spans.len() == 1 {
                SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }) { return Err(RedactError::InvalidNerEvidence); }
            if self.hits.len().checked_add(spans.len()).is_none_or(|n| n > self.max_detections) {
                return Err(RedactError::DetectionBudget);
            }
            self.hits.try_reserve_exact(spans.len()).map_err(|_| RedactError::AllocationRefused)?;
            // Every duplicate proposal is verified before the common overlap
            // union deduplicates its edit. Match/field work is never refunded.
            for span in spans {
                let span = VerifiedSourceSpan {
                    byte_start: origin.byte_start.checked_add(span.byte_start).ok_or(RedactError::InvalidSpan)?,
                    byte_end: origin.byte_start.checked_add(span.byte_end).ok_or(RedactError::InvalidSpan)?,
                    scalar_start: origin.scalar_start.checked_add(span.scalar_start).ok_or(RedactError::InvalidSpan)?,
                    scalar_end: origin.scalar_start.checked_add(span.scalar_end).ok_or(RedactError::InvalidSpan)?,
                };
                self.hits.push(Detection { kind: union::kind_for(entity.entity_type), detector: Detector::NerSourceV1, span });
            }
        }
        self.next_chunk = self.next_chunk.checked_add(1).ok_or(RedactError::DetectionBudget)?;
        self.byte_end = origin.byte_end; self.scalar_end = origin.scalar_end; self.failed = false;
        Ok(())
    }
    pub fn finish(self, expected_chunks: usize) -> Result<DetectedDocument<'s>, RedactError> {
        if self.failed || expected_chunks == 0 || self.next_chunk != expected_chunks || self.byte_end != self.source.len()
            || self.scalar_end != self.source.chars().count() { return Err(RedactError::InvalidNerEvidence); }
        Ok(DetectedDocument { source: self.source, regions: union::merge(&self.hits)?, detection_count: self.hits.len(),
            rule_set: self.rules, model_types: self.options.types.into_iter().collect::<BTreeSet<_>>() })
    }
}
