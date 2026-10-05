//! Delivery checks against the CLI's independently sized complete document.
use super::*;
use crate::{native_engine::strict_int8::STRICT_INT8_PROFILE,
    tasks::{corpus_keyphrases::{CORPUS_KEYPHRASE_POLICY, CorpusKeyphraseWarning},
        extract::ExtractionGrounding, ir::ScoreSpace, keyphrases::KEYPHRASES_TASK_VERSION,
        mapreduce::CHUNK_PROFILE,
        source_planning::quantized::long::keyphrase_reduction::{Int8CorpusKeyphraseRun, Int8KeyphraseLimits,
            INT8_CORPUS_KEYPHRASE_EXECUTION, INT8_CORPUS_KEYPHRASE_SEMANTICS}}};

pub(super) fn check_completed(expected: &Preflight, result: &Int8CorpusKeyphraseRun, limits: Int8KeyphraseLimits)
    -> Result<(), CandidateError> {
    let span = result.source_span;
    let phrases = &result.keyphrases;
    if result.schema_version != 1 || result.execution != INT8_CORPUS_KEYPHRASE_EXECUTION
        || result.numerics_profile != STRICT_INT8_PROFILE || result.chunk_profile != CHUNK_PROFILE
        || result.semantics != INT8_CORPUS_KEYPHRASE_SEMANTICS
        || phrases.schema_version != 1 || phrases.task_spec_version != KEYPHRASES_TASK_VERSION
        || phrases.ranking_policy != CORPUS_KEYPHRASE_POLICY || phrases.score_space != ScoreSpace::NotComputed
        || phrases.grounding != ExtractionGrounding::SourceMembership
        || phrases.warnings != [CorpusKeyphraseWarning::ChunkBoundariesMaySplitPhrases,
            CorpusKeyphraseWarning::SingleContextEquivalenceNotEstablished,
            CorpusKeyphraseWarning::ModelSelectionIsNotRecallGuarantee]
        || result.planned_model_work != expected.work || result.reserved_mask_node_visits != expected.masks
        || phrases.mapped_chunks != expected.chunks || result.requested_max_phrases != limits.max_phrases
        || phrases.phrases.len() > limits.max_phrases
        || phrases.phrases.len().checked_add(phrases.omitted_candidates)
            .is_none_or(|n| n > limits.aggregation.max_unique_phrases)
        || span.byte_start != 0 || span.byte_end != expected.source_bytes
        || span.scalar_start != 0 || span.scalar_end != expected.source_scalars
        || result.model_work.forward_positions > expected.work.forward_positions
        || result.model_work.projected_logits > expected.work.projected_logits
        || result.model_work.attention_pairs > expected.work.attention_pairs
        || !result.model_work.projections.fits(expected.work.projections)
        || result.mask_node_visit_charge > expected.masks
        || result.verification_scan_work > limits.aggregation.max_scan_work
        || phrases.forward_positions != result.model_work.forward_positions
        || phrases.projected_logits != result.model_work.projected_logits
        || phrases.mask_node_visit_charge != result.mask_node_visit_charge {
        return Err(CandidateError::Execution);
    }
    Ok(())
}

#[cfg(test)] mod tests;
