//! Pure whole-window quotation shared by the native host and command line.
//! These are modeled reservations, not measured or allocator-enforced RSS.
use super::{BatchCode, BatchFault, Int4ScoredBatchLimits, Int4MemoryRequirement};
use crate::{batch::{BatchLimits, grouped::MAX_GROUP_RECORDS},
    native_engine::strict_int4::{cohort::MAX_INT4_SEQUENCES,
        scoring::cohort::required_task_workspace_bytes}};

pub const MAX_COHORT_OUTPUT_BYTES: u64 = 64 * 1024 * 1024;

/// Every prepared document is bounded separately. One extra preparation slot
/// holds the lookahead that can cause an already queued cohort to drain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ScoredCohortMemoryLimits {
    pub max_documents: usize,
    pub max_heads: usize,
    pub context_tokens: usize,
    /// Compiler/defaults and one prepared document, including task arguments.
    pub preparation_reserve_bytes: u64,
    /// All buffers owned by the supplied reader and writer.
    pub io_reserve_bytes: u64,
    pub allocator_reserve_bytes: u64,
}

/// Add KV, workspace, input and output reservations for the complete transient
/// peak. Resident model, caller preparation and run cleanup remain separate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ScoredCohortMemoryRequirement {
    pub native: Int4MemoryRequirement,
    pub workspace_bytes: u64,
    pub input_bytes: u64,
    pub output_reservation_bytes: u64,
    /// Included in input_bytes; pass this ceiling to the native processor.
    pub task_workspace_bytes: u64,
    /// Complete semantic results retained in one window, before serialization.
    pub max_output_bytes: u64,
}

impl Int4ScoredCohortMemoryLimits {
    /// Quotes the declared worst case before any input/model access. Configs
    /// exceeding the scoring layer's 64 MiB aggregate result ceiling refuse;
    /// no implicit document-count reduction or unpriced lookahead is used.
    pub fn requirements(self, group_width: usize, work: Int4ScoredBatchLimits,
        transport: BatchLimits) -> Result<Int4ScoredCohortMemoryRequirement, BatchFault> {
        work.validate()?;
        transport.validate()?;
        if !(1..=MAX_GROUP_RECORDS).contains(&self.max_documents)
            || !(1..=MAX_INT4_SEQUENCES).contains(&self.max_heads)
            || self.preparation_reserve_bytes == 0 || self.io_reserve_bytes == 0
            || self.allocator_reserve_bytes == 0 {
            return Err(BatchCode::InvalidLimits.into());
        }
        let contexts = [self.context_tokens; MAX_INT4_SEQUENCES];
        let native = Int4MemoryRequirement::for_cohort(&contexts[..self.max_heads], group_width)
            .map_err(|_| BatchCode::InvalidLimits)?;
        let documents = self.max_documents as u64;
        let live_preparations = documents.checked_add(1).ok_or(BatchCode::InvalidLimits)?;
        let max_output_bytes = multiply(documents, transport.max_output_line_bytes as u64)?;
        if max_output_bytes > MAX_COHORT_OUTPUT_BYTES { return Err(BatchCode::InvalidLimits.into()); }
        let task_workspace_bytes = required_task_workspace_bytes(work.max_model_work,
            self.max_heads, max_output_bytes).map_err(|_| BatchCode::InvalidLimits)?;
        let workspace_bytes = add(&[native.rope_bytes, native.scratch_payload_bound,
            self.allocator_reserve_bytes])?;
        let input_bytes = add(&[
            multiply(multiply(transport.max_line_bytes as u64, 8)?, live_preparations)?,
            multiply(transport.max_output_line_bytes as u64, 4)?,
            multiply(transport.max_epoch_ids as u64, 128)?, transport.max_epoch_id_bytes as u64,
            multiply(self.preparation_reserve_bytes, live_preparations)?,
            self.io_reserve_bytes, task_workspace_bytes,
        ])?;
        // Each successful item owns output_claim(bytes, 0) until the last
        // cohort result is written AND flushed. Price every such guard.
        let output_reservation_bytes = add(&[multiply(max_output_bytes, 4)?,
            multiply(documents, 4096)?])?;
        Ok(Int4ScoredCohortMemoryRequirement { native, workspace_bytes, input_bytes,
            output_reservation_bytes, task_workspace_bytes, max_output_bytes })
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
