//! Complete-summary delivery checks against the CLI's independently sized source.
use super::*;
use crate::{native_engine::strict_int8::STRICT_INT8_PROFILE,
    tasks::{summarize::SUMMARIZE_TASK_VERSION,
        source_planning::quantized::long::summary::{Int8CorpusSummaryRun, INT8_CORPUS_SUMMARY_EXECUTION}}};

pub(super) fn check_completed(expected: &Preflight, result: &Int8CorpusSummaryRun, max_bullets: usize)
    -> Result<(), CandidateError> {
    let span = result.source_span;
    if result.schema_version != 1 || result.execution != INT8_CORPUS_SUMMARY_EXECUTION
        || result.numerics_profile != STRICT_INT8_PROFILE || result.summary.task_spec_version != SUMMARIZE_TASK_VERSION
        || result.planned_model_work != expected.work || result.reserved_mask_node_visits != expected.masks
        || result.summary.mapped_chunks != expected.chunks || result.summary.requested_max_bullets != max_bullets
        || result.summary.bullets.len() > max_bullets
        || span.byte_start != 0 || span.byte_end != expected.source_bytes
        || span.scalar_start != 0 || span.scalar_end != expected.source_scalars
        || result.model_work.forward_positions > expected.work.forward_positions
        || result.model_work.projected_logits > expected.work.projected_logits
        || result.model_work.attention_pairs > expected.work.attention_pairs
        || !result.model_work.projections.fits(expected.work.projections)
        || result.mask_node_visit_charge > expected.masks
        || result.summary.forward_positions != result.model_work.forward_positions
        || result.summary.projected_logits != result.model_work.projected_logits
        || result.summary.mask_node_visit_charge != result.mask_node_visit_charge {
        return Err(CandidateError::Execution);
    }
    Ok(())
}

#[cfg(test)] mod tests;
