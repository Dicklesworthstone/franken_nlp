//! Shared, checked payload quote for the complete answer hierarchy.
//! Input/grammar preparation, native KV/scratch and delivered output are
//! separately owned by the host. This quote covers their coexisting reduction
//! state; it is an admission bound, not a measured process-RSS claim.
use super::{Int8SourceMapError, Int8SourceMapLimits, QuestionHierarchyLimits,
    QuestionSynthesisLimits, TaskBudget, GroundingBudget};
use crate::{tasks::{answer::{PassageCitation, PassageOccurrence}, summarize::SourceCitation},
    validation::grounded_fields::VerifiedSourceSpan};

/// Retain every discovery result and every permitted native/lifted answer,
/// both evidence frontiers, the current question/manifest window, original
/// occurrence transport, and the complete level's compact planning witnesses.
/// Grammars are dropped after level admission and reconstructed one at a time.
#[allow(clippy::too_many_arguments)]
pub fn workspace_bytes(budget: TaskBudget, mapping: Int8SourceMapLimits, chunks: usize,
    reduction_reserve_bytes: u64, synthesis: QuestionSynthesisLimits,
    verification: GroundingBudget, hierarchy: QuestionHierarchyLimits)
    -> Result<u64, Int8SourceMapError> {
    hierarchy.validate()?;
    if chunks == 0 || chunks > mapping.chunks.max_chunks || chunks > 256
        || reduction_reserve_bytes == 0 || budget.max_output_bytes == 0 || budget.max_output_tokens == 0
        || synthesis.max_evidence_bytes == 0 || !(1..=1024).contains(&synthesis.max_evidence_passages) {
        return Err(Int8SourceMapError::InvalidLimits);
    }
    let native = multiply(budget.max_output_bytes, chunks as u64)?;
    let map_staging = multiply(sum(&[native, mapping.reduction.max_live_value_bytes as u64])?, 4)?;
    let map_tokens = multiply(multiply(u64::from(budget.max_output_tokens), chunks as u64)?, 8)?;
    let fields = multiply(verification.max_fields as u64,
        (std::mem::size_of::<PassageCitation>() + std::mem::size_of::<SourceCitation>()) as u64)?;
    let spans = multiply(verification.max_matches as u64,
        (std::mem::size_of::<PassageOccurrence>() + 2 * std::mem::size_of::<VerifiedSourceSpan>()) as u64)?;
    let per_pass = sum(&[multiply(budget.max_output_bytes, 4)?,
        multiply(u64::from(budget.max_output_tokens), 8)?])?;
    sum(&[map_staging, map_tokens, reduction_reserve_bytes, fields, spans,
        multiply(synthesis.max_evidence_bytes as u64, 64)?,
        multiply(synthesis.max_evidence_passages as u64, 1024)?,
        multiply(verification.max_fields as u64, 128)?,
        multiply(verification.max_matches as u64, 128)?,
        multiply(per_pass, hierarchy.max_passes as u64)?,
        multiply(hierarchy.max_passes as u64, 8192 + 256)?,
        multiply(hierarchy.max_levels as u64, 128)?])
}

fn multiply(a: u64, b: u64) -> Result<u64, Int8SourceMapError> {
    a.checked_mul(b).ok_or(Int8SourceMapError::WorkLimit)
}
fn sum(values: &[u64]) -> Result<u64, Int8SourceMapError> {
    values.iter().try_fold(0_u64, |total, value| total.checked_add(*value).ok_or(Int8SourceMapError::WorkLimit))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{native_engine::strict_int8::Int8Work,
        tasks::mapreduce::{ChunkLimits, ExecutionLimits}};
    fn inputs() -> (TaskBudget, Int8SourceMapLimits, QuestionSynthesisLimits, GroundingBudget) {
        (TaskBudget { max_input_tokens: 64, max_output_tokens: 16, max_output_bytes: 4096,
            max_grammar_states: 4096, max_kv_bytes: 1024 },
        Int8SourceMapLimits { chunks: ChunkLimits { max_chunks: 16, ..ChunkLimits::default() },
            reduction: ExecutionLimits { max_live_value_bytes: 8192, ..ExecutionLimits::default() },
            max_model_work: Int8Work::default(), mask_limits: Default::default(),
            mask_visits_per_chunk: 1000, max_mask_visits: 16_000 },
        QuestionSynthesisLimits { max_evidence_passages: 4, max_evidence_bytes: 1024 },
        GroundingBudget { max_fields: 64, max_matches: 256, max_scan_steps: 1_000_000 })
    }
    #[test]
    fn every_retained_pass_and_discovery_result_has_additional_storage() {
        let (b, m, s, v) = inputs();
        let h = QuestionHierarchyLimits { max_passes: 4, max_levels: 3, ..Default::default() };
        let base = workspace_bytes(b, m, 3, 4096, s, v, h).unwrap();
        let next_pass = workspace_bytes(b, m, 3, 4096, s, v,
            QuestionHierarchyLimits { max_passes: 5, ..h }).unwrap();
        assert_eq!(next_pass - base, 4 * b.max_output_bytes + 8 * u64::from(b.max_output_tokens) + 8192 + 256);
        let next_chunk = workspace_bytes(b, m, 4, 4096, s, v, h).unwrap();
        assert_eq!(next_chunk - base, 4 * b.max_output_bytes + 8 * u64::from(b.max_output_tokens));
        assert_eq!(workspace_bytes(b, m, 3, 8192, s, v, h).unwrap() - base, 4096);
        let wider = GroundingBudget { max_matches: v.max_matches * 2, ..v };
        assert!(workspace_bytes(b, m, 3, 4096, s, wider, h).unwrap() - base >= v.max_matches as u64 * 128);
    }
    #[test]
    fn invalid_cardinality_and_overflow_refuse_instead_of_underquoting() {
        let (mut b, m, s, v) = inputs(); let h = QuestionHierarchyLimits::default();
        for chunks in [0, 17, usize::MAX] { assert!(workspace_bytes(b, m, chunks, 4096, s, v, h).is_err()); }
        assert!(workspace_bytes(b, m, 1, 0, s, v, h).is_err());
        assert!(workspace_bytes(b, m, 1, u64::MAX, s, v, h).is_err());
        assert!(workspace_bytes(b, m, 1, 4096, s, v, QuestionHierarchyLimits { max_passes: 0, ..h }).is_err());
        b.max_output_bytes = u64::MAX / 8;
        assert!(workspace_bytes(b, m, 1, 4096, s, v, h).is_err());
    }
}
