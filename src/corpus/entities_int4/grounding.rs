//! Independent occurrence recovery and exact original-coordinate lifting.
//! Proposed repeated surfaces expand to EVERY verified local occurrence;
//! duplicate proposals are validated before deduplication, never trusted.
use super::*;
use std::collections::BTreeSet;
use crate::{tasks::{extract::ExtractionGrounding, ir::ScoreSpace},
    validation::grounded_fields::{FieldGroundingError, SourceOccurrence, scan_occurrences},
    corpus::resolve::MentionInput};

pub(super) fn expand<C: DecodeStepControl>(source: &str, ner: NerResult, config: &Int4EntityConfig,
    verification: &mut GroundingBudget, control: &mut C) -> Result<Vec<MentionInput>, Int4EntityError> {
    resolve::checkpoint(control)?;
    if ner.schema_version != 1 || ner.task_spec_version != NER_TASK_VERSION || ner.numerics_profile != STRICT_INT4_PROFILE
        || ner.score_space != ScoreSpace::NotComputed || ner.grounding != ExtractionGrounding::SourceMembership
        || ner.entities.len() > config.ner.max_entities { return Err(Int4EntityError::Accounting); }
    let mut bytes = source.len(); let mut seen = BTreeSet::new(); let mut mentions = Vec::new();
    for entity in ner.entities {
        resolve::checkpoint(control)?;
        if !config.ner.types.contains(&entity.entity_type) || entity.text.is_empty()
            || entity.text.len() > config.graph.max_surface_bytes
            || entity.text.chars().count() > config.ner.max_mention_scalars { return Err(Int4EntityError::Accounting); }
        verification.max_fields = verification.max_fields.checked_sub(1).ok_or(ResolveError::InputBudget)?;
        let spans = scan_occurrences(source, &entity.text, verification).map_err(|error| match error {
            FieldGroundingError::AllocationRefused => ResolveError::AllocationRefused,
            FieldGroundingError::WorkBudget => ResolveError::ScanBudget,
            FieldGroundingError::MatchBudget | FieldGroundingError::FieldBudget => ResolveError::InputBudget,
            _ => ResolveError::InvalidAnchor,
        })?;
        if spans.is_empty() || spans != entity.spans
            || entity.occurrence != if spans.len() == 1 { SourceOccurrence::Anchored } else { SourceOccurrence::Ambiguous } {
            return Err(ResolveError::InvalidAnchor.into());
        }
        for span in spans {
            resolve::checkpoint(control)?;
            if !seen.insert((entity.entity_type, span.byte_start, span.byte_end)) { continue; }
            if mentions.len() == config.graph.max_mentions { return Err(Int4EntityError::WorkBudget); }
            let kind = entity.entity_type.label();
            bytes = bytes.checked_add(kind.len()).and_then(|n| n.checked_add(entity.text.len()))
                .filter(|&n| n <= config.graph.max_input_bytes).ok_or(Int4EntityError::WorkBudget)?;
            mentions.try_reserve(1).map_err(|_| Int4EntityError::Allocation)?;
            mentions.push(MentionInput { entity_type: copy(kind)?, surface: copy(&entity.text)?, span });
        }
    }
    Ok(mentions)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append<C: DecodeStepControl>(document: &mut ResolutionDocument, local: Vec<MentionInput>,
    extent: VerifiedSourceSpan, bytes: &mut usize, count: &mut usize, limits: ResolveLimits, control: &mut C)
    -> Result<(), Int4EntityError> {
    for mut mention in local {
        resolve::checkpoint(control)?;
        let offset = |n: usize, base: usize| n.checked_add(base).ok_or(Int4EntityError::Accounting);
        let span = VerifiedSourceSpan { byte_start: offset(mention.span.byte_start, extent.byte_start)?,
            byte_end: offset(mention.span.byte_end, extent.byte_start)?,
            scalar_start: offset(mention.span.scalar_start, extent.scalar_start)?,
            scalar_end: offset(mention.span.scalar_end, extent.scalar_start)? };
        if span.byte_end > extent.byte_end || span.scalar_end > extent.scalar_end
            || span.byte_start >= span.byte_end || span.scalar_start >= span.scalar_end
            || document.text.get(span.byte_start..span.byte_end) != Some(mention.surface.as_str()) {
            return Err(Int4EntityError::Accounting);
        }
        let next_count = count.checked_add(1).filter(|&n| n <= limits.max_mentions).ok_or(Int4EntityError::WorkBudget)?;
        let next_bytes = bytes.checked_add(mention.entity_type.len()).and_then(|n| n.checked_add(mention.surface.len()))
            .filter(|&n| n <= limits.max_input_bytes).ok_or(Int4EntityError::WorkBudget)?;
        document.mentions.try_reserve(1).map_err(|_| Int4EntityError::Allocation)?;
        mention.span = span; document.mentions.push(mention); *count = next_count; *bytes = next_bytes;
    }
    Ok(())
}
