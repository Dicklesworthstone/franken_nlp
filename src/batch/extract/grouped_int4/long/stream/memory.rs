//! Model-free ownership quote for a complete-document extraction stream.
//! The CLI and owned host use this same calculation before private input IO.
use super::super::{Int4ExtractionMapError, Int4ExtractionMapLimits,
    cohort::{Int4ExtractionMapCohortMemoryLimits, Int4ExtractionMapCohortRequirement}};
use super::EXTRACTION_DOCUMENT_CHUNK_METADATA_BYTES;
use crate::{batch::BatchLimits, native_engine::strict_int4::Int4MemoryRequirement,
    tasks::ir::TaskBudget, validation::grounded_fields::{SourceFieldEvidence, VerifiedSourceSpan}};

/// Caller-owned compiler, schema, vocabulary and IO allocations must fit the
/// explicit preparation/IO declarations. Native, complete reduction, frozen
/// identities and result staging are additionally priced here. These are
/// modeled reservations, not allocator interception or an observed RSS bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ExtractionDocumentMemoryLimits {
    pub context_tokens: usize,
    pub allocator_reserve_bytes: u64,
    pub preparation_reserve_bytes: u64,
    pub io_reserve_bytes: u64,
    pub reduction_reserve_bytes: u64,
    /// Explicit physical schedule; serial execution remains one KV row.
    pub cohort_chunks: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ExtractionDocumentMemoryRequirement {
    pub native: Int4MemoryRequirement,
    /// Native RoPE/scratch and caller-declared allocator headroom.
    pub workspace_bytes: u64,
    /// Transport, compiler/vocabulary, IO and all frozen chunk identities.
    pub input_bytes: u64,
    /// Complete map frontier, evidence, token storage and cohort decoder state.
    pub document_workspace_bytes: u64,
    /// Semantic output plus canonical serialization and token vectors.
    pub output_reservation_bytes: u64,
    pub max_output_bytes: u64,
}

impl Int4ExtractionDocumentMemoryLimits {
    pub fn requirements(self, width: usize, budget: TaskBudget, document: Int4ExtractionMapLimits,
        transport: BatchLimits) -> Result<Int4ExtractionDocumentMemoryRequirement, Int4ExtractionMapError> {
        budget.validate().map_err(|_| Int4ExtractionMapError::InvalidLimits)?;
        transport.validate().map_err(|_| Int4ExtractionMapError::InvalidLimits)?;
        let m = document.mapping; let r = m.reduction; let v = document.verification;
        m.chunks.effective_token_limit().map_err(|_| Int4ExtractionMapError::InvalidLimits)?;
        if self.allocator_reserve_bytes == 0 || self.preparation_reserve_bytes == 0
            || self.io_reserve_bytes == 0 || self.reduction_reserve_bytes == 0
            || m.chunks.context_tokens != self.context_tokens || m.chunks.max_input_bytes == 0
            || !(1..=256).contains(&m.chunks.max_chunks)
            || !(1..=1024).contains(&r.map_batch_chunks) || !(2..=1024).contains(&r.reduce_fan_in)
            || r.max_reduction_levels > 64 || !(1..=1_000_000).contains(&r.max_task_calls)
            || [r.max_value_bytes, r.max_live_value_bytes, r.max_total_value_bytes, r.max_result_bytes]
                .iter().any(|n| !(1..=64 * 1024 * 1024).contains(n))
            || r.max_value_bytes > r.max_live_value_bytes
            || r.max_result_bytes > transport.max_output_line_bytes
            || m.mask_limits.max_trie_node_visits == 0 || m.mask_limits.checkpoint_interval_nodes == 0
            || m.mask_visits_per_chunk == 0 || m.max_mask_visits < m.mask_visits_per_chunk
            || !(1..=1_000_000).contains(&v.max_fields) || !(1..=1_000_000).contains(&v.max_matches)
            || v.max_scan_steps == 0
            || [m.max_model_work.forward_positions, m.max_model_work.projected_logits,
                m.max_model_work.attention_pairs, m.max_model_work.projections.dot_products,
                m.max_model_work.projections.multiply_accumulates].contains(&0) {
            return Err(Int4ExtractionMapError::InvalidLimits);
        }
        let row = Int4MemoryRequirement::for_context(self.context_tokens, width)
            .map_err(|_| Int4ExtractionMapError::InvalidLimits)?;
        // Scheduling cannot widen the immutable per-chunk task authority.
        if row.kv_bytes != budget.max_kv_bytes { return Err(Int4ExtractionMapError::Admission); }
        let cohort = self.cohort_requirements(width, budget, document)?;
        let native = cohort.map_or(row, |quote| quote.native);
        let chunks = m.chunks.max_chunks as u64;
        let transport_bytes = sum(&[mul(transport.max_line_bytes as u64, 8)?,
            mul(transport.max_output_line_bytes as u64, 4)?, mul(transport.max_epoch_ids as u64, 128)?,
            transport.max_epoch_id_bytes as u64, self.preparation_reserve_bytes, self.io_reserve_bytes])?;
        let input_bytes = sum(&[transport_bytes, mul(chunks, EXTRACTION_DOCUMENT_CHUNK_METADATA_BYTES)?])?;
        let staged = mul(sum(&[mul(budget.max_output_bytes, chunks)?, r.max_live_value_bytes as u64])?, 4)?;
        let tokens = mul(u64::from(budget.max_output_tokens), chunks)?;
        let fields = mul(v.max_fields as u64, std::mem::size_of::<SourceFieldEvidence>() as u64)?;
        let spans = mul(mul(v.max_matches as u64, std::mem::size_of::<VerifiedSourceSpan>() as u64)?, 2)?;
        let document_workspace_bytes = sum(&[staged, mul(tokens, 8)?, fields, spans,
            self.reduction_reserve_bytes, cohort.map_or(0, |quote| quote.workspace_bytes),
            document.root_array_workspace_bytes()?])?;
        let max_output_bytes = r.max_result_bytes as u64;
        let required = Int4ExtractionDocumentMemoryRequirement { native,
            workspace_bytes: sum(&[native.rope_bytes, native.scratch_payload_bound, self.allocator_reserve_bytes])?,
            input_bytes, document_workspace_bytes,
            output_reservation_bytes: sum(&[mul(max_output_bytes, 4)?, mul(tokens, 8)?, 4096])?,
            max_output_bytes };
        // Individually representable components must also coexist in one
        // representable invocation, before any allocation or stream read.
        sum(&[required.native.kv_bytes, required.workspace_bytes, required.input_bytes,
            required.document_workspace_bytes, required.output_reservation_bytes])?;
        Ok(required)
    }

    pub(crate) fn cohort_requirements(self, width: usize, budget: TaskBudget,
        document: Int4ExtractionMapLimits) -> Result<Option<Int4ExtractionMapCohortRequirement>, Int4ExtractionMapError> {
        self.cohort_chunks.map(|chunks| {
            if chunks != document.mapping.reduction.map_batch_chunks
                || chunks > document.mapping.chunks.max_chunks {
                return Err(Int4ExtractionMapError::InvalidLimits);
            }
            Int4ExtractionMapCohortMemoryLimits { max_chunks: chunks, context_tokens: self.context_tokens,
                max_output_tokens_per_chunk: u64::from(budget.max_output_tokens),
                max_result_bytes_per_chunk: budget.max_output_bytes }.requirements(width)
        }).transpose()
    }
}

fn sum(values: &[u64]) -> Result<u64, Int4ExtractionMapError> {
    values.iter().try_fold(0_u64, |n, value| n.checked_add(*value).ok_or(Int4ExtractionMapError::InvalidLimits))
}
fn mul(left: u64, right: u64) -> Result<u64, Int4ExtractionMapError> {
    left.checked_mul(right).ok_or(Int4ExtractionMapError::InvalidLimits)
}

#[cfg(test)] mod tests;
