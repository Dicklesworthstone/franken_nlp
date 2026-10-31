//! Shared, checked stream quotation before model or document IO.
use super::Int4ExtractionBatchLimits;
use crate::{batch::{BatchCode, BatchFault, BatchLimits},
    native_engine::{constrained_int4::cohort::required_workspace_bytes,
        lmhead::NANBEIGE_VOCAB_SIZE,
        strict_int4::{Int4MemoryRequirement, cohort::MAX_INT4_SEQUENCES}}};

pub const MAX_COHORT_RESULT_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ExtractionCohortMemoryLimits {
    pub max_documents: usize,
    pub context_tokens: usize,
    /// One complete prepared/source/grammar item, its parser choices and mask
    /// traversal state, plus compiler/vocabulary owners. One lookahead is priced.
    pub preparation_reserve_bytes: u64,
    pub io_reserve_bytes: u64,
    pub allocator_reserve_bytes: u64,
}

/// Add native KV, workspace, input and output reservations for the complete
/// transient peak. Resident weights, outer preparation and cleanup are separate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ExtractionCohortMemoryRequirement {
    pub native: Int4MemoryRequirement,
    pub workspace_bytes: u64,
    pub input_bytes: u64,
    pub output_reservation_bytes: u64,
    /// Included in input_bytes; additional cursors/masks/results beyond each
    /// live item's preparation estimate and native full-vocabulary head rails.
    pub decode_workspace_bytes: u64,
    /// Includes complete retained task results and the finite cohort envelope.
    pub max_result_bytes: u64,
}
impl Int4ExtractionCohortMemoryLimits {
    pub fn requirements(self, width: usize, work: Int4ExtractionBatchLimits, transport: BatchLimits)
        -> Result<Int4ExtractionCohortMemoryRequirement, BatchFault> {
        work.validate()?; transport.validate()?;
        if !(1..=MAX_INT4_SEQUENCES).contains(&self.max_documents)
            || self.preparation_reserve_bytes == 0 || self.io_reserve_bytes == 0
            || self.allocator_reserve_bytes == 0 {
            return Err(BatchCode::InvalidLimits.into());
        }
        let contexts = [self.context_tokens; MAX_INT4_SEQUENCES];
        let native = Int4MemoryRequirement::for_cohort(&contexts[..self.max_documents], width)
            .map_err(|_| BatchCode::InvalidLimits)?;
        let count = self.max_documents as u64;
        let rows = multiply(count, transport.max_output_line_bytes as u64)?;
        let max_result_bytes = add(&[rows, 4096, count])?;
        if max_result_bytes > MAX_COHORT_RESULT_BYTES { return Err(BatchCode::InvalidLimits.into()); }
        // Every full-vocabulary projection yields one generated token, and
        // prompt>=1 makes a row's token ceiling no larger than its context.
        // Quota-limited windows can contain fewer executable rows than N.
        let tokens = (work.max_model_work.projected_logits / NANBEIGE_VOCAB_SIZE as u64)
            .min(multiply(count, self.context_tokens as u64)?);
        if tokens == 0 { return Err(BatchCode::InvalidLimits.into()); }
        let decode_workspace_bytes = required_workspace_bytes(self.max_documents, tokens, max_result_bytes)
            .map_err(|_| BatchCode::InvalidLimits)?;
        let workspace_bytes = add(&[native.rope_bytes, native.scratch_payload_bound, self.allocator_reserve_bytes])?;
        let preparations = count.checked_add(1).ok_or(BatchCode::InvalidLimits)?;
        let input_bytes = add(&[
            multiply(multiply(transport.max_line_bytes as u64, 8)?, preparations)?,
            multiply(transport.max_output_line_bytes as u64, 4)?,
            multiply(transport.max_epoch_ids as u64, 128)?, transport.max_epoch_id_bytes as u64,
            multiply(self.preparation_reserve_bytes, preparations)?, self.io_reserve_bytes, decode_workspace_bytes,
        ])?;
        // Exactly the host's per-document output_claim model, retained for
        // every result until the last physical writer flush in this window.
        let output_reservation_bytes = add(&[multiply(rows, 4)?, multiply(tokens, 8)?, multiply(count, 4096)?])?;
        Ok(Int4ExtractionCohortMemoryRequirement { native, workspace_bytes, input_bytes,
            output_reservation_bytes, decode_workspace_bytes, max_result_bytes })
    }
}
fn multiply(value: u64, count: u64) -> Result<u64, BatchFault> {
    value.checked_mul(count).ok_or_else(|| BatchCode::InvalidLimits.into())
}
fn add(values: &[u64]) -> Result<u64, BatchFault> {
    values.iter().try_fold(0_u64, |sum, &value| sum.checked_add(value)
        .ok_or_else(|| BatchCode::InvalidLimits.into()))
}

#[cfg(test)] mod tests;
