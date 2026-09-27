//! Strict-profile occurrence recovery. No eager result is relabeled, and no
//! serialized receipt alone proves that a model executed. Used only after the
//! private native-run checker in the consumed entity pipeline.
use super::*;
use std::collections::BTreeSet;
use crate::{tasks::{extract::ExtractionGrounding, ir::ScoreSpace},
    validation::grounded_fields::{FieldGroundingError, SourceOccurrence, scan_occurrences}};
use crate::corpus::resolve::MentionInput;

pub(super) fn expand<C: DecodeStepControl>(document: EntityDocument, ner: NerResult, config: &Int8EntityConfig,
    verification: &mut GroundingBudget, control: &mut C) -> Result<ResolutionDocument, Int8EntityError> {
    resolve::checkpoint(control)?;
    if ner.schema_version != 1 || ner.task_spec_version != NER_TASK_VERSION || ner.numerics_profile != STRICT_INT8_PROFILE
        || ner.score_space != ScoreSpace::NotComputed || ner.grounding != ExtractionGrounding::SourceMembership
        || ner.entities.len() > config.ner.max_entities { return Err(Int8EntityError::Accounting); }
    let mut bytes = document.id.len().checked_add(document.text.len())
        .filter(|&n| n <= config.graph.max_input_bytes).ok_or(Int8EntityError::WorkBudget)?;
    let mut seen = BTreeSet::new(); let mut mentions = Vec::new();
    for entity in ner.entities {
        resolve::checkpoint(control)?;
        if !config.ner.types.contains(&entity.entity_type) || entity.text.is_empty()
            || entity.text.len() > config.graph.max_surface_bytes
            || entity.text.chars().count() > config.ner.max_mention_scalars { return Err(Int8EntityError::Accounting); }
        verification.max_fields = verification.max_fields.checked_sub(1).ok_or(ResolveError::InputBudget)?;
        let spans = scan_occurrences(&document.text, &entity.text, verification).map_err(|e| match e {
            FieldGroundingError::AllocationRefused => ResolveError::AllocationRefused,
            FieldGroundingError::WorkBudget => ResolveError::ScanBudget,
            FieldGroundingError::MatchBudget | FieldGroundingError::FieldBudget => ResolveError::InputBudget,
            _ => ResolveError::InvalidAnchor,
        })?;
        if spans.is_empty() || spans != entity.spans
            || entity.occurrence != if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous } {
            return Err(ResolveError::InvalidAnchor.into());
        }
        // Validate even duplicated proposals BEFORE deduplication. Equal names
        // at different offsets remain separate mentions and may resolve apart.
        for span in spans {
            resolve::checkpoint(control)?;
            if !seen.insert((entity.entity_type, span.byte_start, span.byte_end)) { continue; }
            if mentions.len() == config.graph.max_mentions { return Err(Int8EntityError::WorkBudget); }
            let kind = entity.entity_type.label();
            bytes = bytes.checked_add(kind.len()).and_then(|n| n.checked_add(entity.text.len()))
                .filter(|&n| n <= config.graph.max_input_bytes).ok_or(Int8EntityError::WorkBudget)?;
            mentions.try_reserve(1).map_err(|_| Int8EntityError::Allocation)?;
            mentions.push(MentionInput { entity_type: copy(kind)?, surface: copy(&entity.text)?, span });
        }
    }
    Ok(ResolutionDocument { id: document.id, text: document.text, mentions })
}
pub(super) fn charge(document: &ResolutionDocument, bytes: &mut usize, mentions: &mut usize, limits: ResolveLimits)
    -> Result<(), Int8EntityError> {
    let count = mentions.checked_add(document.mentions.len()).filter(|&n| n <= limits.max_mentions)
        .ok_or(Int8EntityError::WorkBudget)?;
    let mut next = bytes.checked_add(document.id.len()).and_then(|n| n.checked_add(document.text.len()))
        .ok_or(Int8EntityError::WorkBudget)?;
    for mention in &document.mentions {
        next = next.checked_add(mention.entity_type.len()).and_then(|n| n.checked_add(mention.surface.len()))
            .ok_or(Int8EntityError::WorkBudget)?;
    }
    if next > limits.max_input_bytes { return Err(Int8EntityError::WorkBudget); }
    *bytes = next; *mentions = count; Ok(())
}
