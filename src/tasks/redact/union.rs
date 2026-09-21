//! Recall-first overlap union and independently checked model evidence.
//! An overlapping component covers its complete union, never merely the
//! longest member (which could leave the tail of another detection exposed).

use std::collections::BTreeSet;
use serde::{Deserialize, Serialize};
use crate::{
    tasks::{extract::ExtractionGrounding, ir::ScoreSpace, ner::{EntityType, NerResult, NER_TASK_VERSION}},
    native_engine::hf_bf16_eager::HF_BF16_EAGER_PROFILE,
    validation::grounded_fields::{GroundingBudget, SourceOccurrence, VerifiedSourceSpan, scan_occurrences},
};
use super::{Detection, Detector, PiiKind, RedactError, detectors::{RuleSet, RuleBudget, detect}};

pub const OVERLAP_POLICY: &str = "overlap-connected-union-v1";

/// Selected by a typed pipeline, never inferred from a caller's receipt.
#[derive(Clone, Copy)]
pub(super) enum NerProfile { Eager, Int8 }
impl NerProfile {
    fn label(self) -> &'static str {
        match self {
            Self::Eager => HF_BF16_EAGER_PROFILE,
            Self::Int8 => crate::native_engine::strict_int8::STRICT_INT8_PROFILE,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RedactionRegion {
    pub span: VerifiedSourceSpan,
    pub kinds: BTreeSet<PiiKind>,
    pub detectors: BTreeSet<Detector>,
}

/// The original source binding is borrowed and never serialized or logged.
/// Only the detector constructors mint this value; callers cannot deserialize
/// arbitrary offsets into an editable document.
pub struct DetectedDocument<'a> {
    pub(crate) source: &'a str,
    pub(crate) regions: Vec<RedactionRegion>,
    pub(crate) rule_set: RuleSet,
    pub(crate) model_types: BTreeSet<EntityType>,
    pub(crate) detection_count: usize,
}
impl<'a> DetectedDocument<'a> {
    pub fn rules_only(source: &'a str, rules: &RuleSet, budget: RuleBudget) -> Result<Self, RedactError> {
        let hits = detect(source, rules, budget)?;
        Ok(Self { source, regions: merge(&hits)?, detection_count: hits.len(),
            rule_set: rules.clone(), model_types: BTreeSet::new() })
    }

    /// Use every verified occurrence, including overlapping repeats. Model
    /// guesses with missing/altered evidence are no-result, never offsets that
    /// can delete unrelated bytes. An empty entity set is a legitimate result,
    /// not proof of complete PII recall.
    pub fn with_ner(
        source: &'a str, rules: &RuleSet, budget: RuleBudget, result: &NerResult,
        model_types: &BTreeSet<EntityType>, grounding: GroundingBudget,
    ) -> Result<Self, RedactError> {
        Self::with_ner_profile(source, rules, budget, result, model_types, grounding, NerProfile::Eager)
    }

    pub(super) fn with_ner_profile(
        source: &'a str, rules: &RuleSet, budget: RuleBudget, result: &NerResult,
        model_types: &BTreeSet<EntityType>, mut grounding: GroundingBudget,
        profile: NerProfile,
    ) -> Result<Self, RedactError> {
        if model_types.is_empty() || result.schema_version != 1
            || result.task_spec_version != NER_TASK_VERSION
            || result.numerics_profile != profile.label()
            || result.score_space != ScoreSpace::NotComputed
            || result.grounding != ExtractionGrounding::SourceMembership
            || result.entities.len() > grounding.max_fields {
            return Err(RedactError::InvalidNerEvidence);
        }
        let mut hits = detect(source, rules, budget)?;
        for entity in &result.entities {
            if entity.text.is_empty() || entity.text.len() > budget.max_input_bytes
                || !model_types.contains(&entity.entity_type) { return Err(RedactError::InvalidNerEvidence); }
            let spans = scan_occurrences(source, &entity.text, &mut grounding)
                .map_err(|_| RedactError::InvalidNerEvidence)?;
            let occurrence = if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous };
            if spans != entity.spans || entity.occurrence != occurrence { return Err(RedactError::InvalidNerEvidence); }
            if hits.len().checked_add(spans.len()).is_none_or(|n| n > budget.max_detections) {
                return Err(RedactError::DetectionBudget);
            }
            hits.try_reserve_exact(spans.len()).map_err(|_| RedactError::AllocationRefused)?;
            for span in spans { hits.push(Detection { kind: kind_for(entity.entity_type), detector: Detector::NerSourceV1, span }); }
        }
        Ok(Self { source, regions: merge(&hits)?, detection_count: hits.len(),
            rule_set: rules.clone(), model_types: model_types.clone() })
    }

    pub fn regions(&self) -> &[RedactionRegion] { &self.regions }
    pub fn detection_count(&self) -> usize { self.detection_count }
    pub fn model_types(&self) -> &BTreeSet<EntityType> { &self.model_types }
}

pub const fn kind_for(kind: EntityType) -> PiiKind {
    match kind {
        EntityType::Person => PiiKind::Person, EntityType::Organization => PiiKind::Organization,
        EntityType::Location => PiiKind::Location, EntityType::Date => PiiKind::Date,
        EntityType::Time => PiiKind::Time, EntityType::Money => PiiKind::Money,
        EntityType::Product => PiiKind::Product, EntityType::Event => PiiKind::Event,
    }
}

fn merge(detections: &[Detection]) -> Result<Vec<RedactionRegion>, RedactError> {
    let mut order = Vec::new();
    order.try_reserve_exact(detections.len()).map_err(|_| RedactError::AllocationRefused)?;
    order.extend(detections);
    order.sort_unstable_by_key(|d| (d.span.byte_start, d.span.byte_end, d.kind, d.detector));
    let mut regions: Vec<RedactionRegion> = Vec::new();
    regions.try_reserve_exact(order.len()).map_err(|_| RedactError::AllocationRefused)?;
    for hit in order {
        let span = hit.span;
        if let Some(last) = regions.last_mut() {
            if span.byte_start < last.span.byte_end {
                if span.byte_end > last.span.byte_end {
                    last.span.byte_end = span.byte_end; last.span.scalar_end = span.scalar_end;
                }
                last.kinds.insert(hit.kind); last.detectors.insert(hit.detector);
                continue;
            }
        }
        regions.push(RedactionRegion { span, kinds: [hit.kind].into_iter().collect(),
            detectors: [hit.detector].into_iter().collect() });
    }
    Ok(regions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::ner::NamedEntity;
    fn ner(source: &str, text: &str) -> NerResult {
        let spans = scan_occurrences(source, text, &mut GroundingBudget::default()).unwrap();
        NerResult { schema_version: 1, task_spec_version: NER_TASK_VERSION.to_owned(),
            numerics_profile: HF_BF16_EAGER_PROFILE.to_owned(), score_space: ScoreSpace::NotComputed,
            grounding: ExtractionGrounding::SourceMembership,
            entities: vec![NamedEntity { text: text.to_owned(), entity_type: EntityType::Person,
                occurrence: if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous }, spans }],
            generated_token_ids: Vec::new(), forward_positions: 0, projected_logits: 0, mask_node_visit_charge: 0 }
    }
    fn union<'a>(source: &'a str, result: &NerResult) -> Result<DetectedDocument<'a>, RedactError> {
        DetectedDocument::with_ner(source, &RuleSet::default(), RuleBudget::default(), result,
            &[EntityType::Person].into_iter().collect(), GroundingBudget::default())
    }
    fn hit(start: usize, end: usize, kind: PiiKind) -> Detection {
        Detection { kind, detector: Detector::NerSourceV1,
            span: VerifiedSourceSpan { byte_start: start, byte_end: end, scalar_start: start, scalar_end: end } }
    }
    #[test]
    fn connected_union_covers_tails_and_retains_all_types() {
        let out = merge(&[hit(0, 5, PiiKind::Person), hit(3, 8, PiiKind::Organization), hit(7, 12, PiiKind::Location)]).unwrap();
        assert_eq!(out.len(), 1); assert_eq!(out[0].span.byte_end, 12); assert_eq!(out[0].kinds.len(), 3);
    }
    #[test]
    fn touching_spans_remain_separate_and_order_is_irrelevant() {
        let mut hits = vec![hit(0, 2, PiiKind::Person), hit(2, 4, PiiKind::Person), hit(5, 8, PiiKind::Location)];
        let expected = merge(&hits).unwrap(); hits.reverse();
        assert_eq!(merge(&hits).unwrap(), expected); assert_eq!(expected.len(), 3);
    }
    #[test]
    fn ambiguity_redacts_every_occurrence_not_only_the_first() {
        let source = "é Alice Alice"; let result = ner(source, "Alice");
        let document = union(source, &result).unwrap();
        assert_eq!(document.regions().len(), 2); assert_eq!(document.regions()[0].span.byte_start, 3);
        assert_eq!(document.regions()[0].span.scalar_start, 2);
    }
    #[test]
    fn forged_or_partial_model_coordinates_cannot_become_edits() {
        let source = "Alice Alice";
        let mut result = ner(source, "Alice"); result.entities[0].spans.pop(); assert!(union(source, &result).is_err());
        let mut result = ner(source, "Alice"); result.entities[0].spans[0].scalar_end += 1; assert!(union(source, &result).is_err());
        assert!(union("Bob", &ner(source, "Alice")).is_err());
    }
    #[test]
    fn rule_and_model_detections_both_survive() {
        let source = "Alice: a@example.org";
        let d = union(source, &ner(source, "Alice")).unwrap();
        assert_eq!(d.regions.len(), 2);
        assert!(d.regions.iter().any(|r| r.kinds.contains(&PiiKind::Email)));
        assert!(d.regions.iter().any(|r| r.kinds.contains(&PiiKind::Person)));
    }
    #[test]
    fn duplicates_do_not_duplicate_edits() {
        let h = hit(1, 5, PiiKind::Person);
        assert_eq!(merge(&[h.clone(), h.clone()]).unwrap(), merge(&[h]).unwrap());
    }
    #[test]
    fn model_contract_and_type_drift_refuse() {
        let mut result = ner("Alice", "Alice"); result.grounding = ExtractionGrounding::NotRequested;
        assert!(union("Alice", &result).is_err());
        let mut result = ner("Alice", "Alice"); result.entities[0].entity_type = EntityType::Organization;
        assert!(union("Alice", &result).is_err());
    }
    #[test]
    fn total_detection_budget_includes_model_occurrences() {
        let r = ner("Alice Alice", "Alice");
        let result = DetectedDocument::with_ner("Alice Alice", &RuleSet::default(),
            RuleBudget { max_detections: 1, ..RuleBudget::default() }, &r,
            &[EntityType::Person].into_iter().collect(), GroundingBudget::default());
        assert!(matches!(result, Err(RedactError::DetectionBudget)));
    }
}
