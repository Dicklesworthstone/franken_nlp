//! Pure declared-memory quote shared by CLI and hosted classification maps.
//! This accounts for simultaneous owners, not a measured process RSS cap.
use super::{Int4ClassificationMapError, Int4ClassificationMapLimits};
use crate::{native_engine::strict_int4::Int4MemoryRequirement,
    tasks::{classify::ClassificationLimits, ir::TaskBudget}};

pub const MIN_CLASSIFICATION_MAP_PREPARATION_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct Int4ClassificationMapMemoryLimits {
    pub context_tokens: usize,
    pub preparation_reserve_bytes: u64,
    pub allocator_reserve_bytes: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4ClassificationMapMemoryRequirement {
    pub native: Int4MemoryRequirement,
    /// RoPE, native scratch and allocator reserve; excludes KV.
    pub workspace_bytes: u64,
    /// Source, labels, compiler, all retained plans and the result frontier.
    pub input_bytes: u64,
    /// Result remains charged through complete caller publication.
    pub output_reservation_bytes: u64,
    pub max_output_bytes: u64,
}
impl Int4ClassificationMapMemoryLimits {
    pub fn requirements(self, width: usize, task: TaskBudget, planning: ClassificationLimits,
        mapping: Int4ClassificationMapLimits) -> Result<Int4ClassificationMapMemoryRequirement, Int4ClassificationMapError> {
        mapping.validate(task)?; planning.validate()?;
        let scoring = planning.scoring;
        if self.preparation_reserve_bytes < MIN_CLASSIFICATION_MAP_PREPARATION_BYTES
            || self.allocator_reserve_bytes == 0 || mapping.chunks.context_tokens > self.context_tokens
            || mapping.chunks.max_input_bytes > planning.max_input_bytes
            || self.context_tokens > planning.max_context_tokens
            || scoring.max_candidates == 0 || scoring.max_total_tokens == 0 || scoring.max_nodes == 0
            || scoring.max_depth == 0 || scoring.max_candidate_id_bytes == 0 || scoring.max_projected_logits == 0 {
            return Err(Int4ClassificationMapError::Limits);
        }
        let native = Int4MemoryRequirement::for_context(self.context_tokens, width)?;
        if native.kv_bytes > task.max_kv_bytes { return Err(Int4ClassificationMapError::Admission); }
        let chunks = mapping.chunks.max_chunks as u64;
        let labels = planning.max_labels as u64;
        // The source is repeated once per independent yes/no head; exclusive
        // classification has just one head. Price the larger valid shape.
        let per_chunk_prompt = mul(u64::from(task.max_input_tokens), labels)?
            .min(planning.max_total_prompt_tokens as u64);
        let declared_tokens = mul(per_chunk_prompt, chunks)?;
        // A new complete plan can exist before its whole-document work check
        // refuses it, in addition to all previously accepted plans.
        let plan_tokens = declared_tokens.min(sum(&[
            mapping.max_model_work.forward_positions.min(declared_tokens), per_chunk_prompt])?);
        let heads = mul(labels, chunks)?;
        // An exclusive language can use its full trie allowance. Each
        // independent binary head has two one-token continuations plus EOS,
        // hence at most five nodes; eight allows its fixed cursor metadata.
        let nodes = (scoring.max_nodes as u64).max(mul(labels, 8)?);
        let node_bytes = sum(&[192, mul(scoring.max_depth.min(task.max_output_tokens as usize) as u64, 8)?])?;
        let plans = sum(&[mul(plan_tokens, 64)?, mul(heads, 16_384)?,
            mul(mul(nodes, node_bytes)?, chunks)?,
            mul(mul(planning.max_total_label_bytes as u64, 4)?, chunks)?])?;
        let per_chunk_result = task.max_output_bytes;
        let retained = mul(sum(&[per_chunk_result, 4096])?, chunks)?.min(mapping.max_retained_result_bytes);
        // The next complete chunk and its serialization exist before the
        // cumulative retained-result cap can reject it.
        let results = mul(sum(&[retained, per_chunk_result, 4096])?, 8)?;
        let input_bytes = sum(&[self.preparation_reserve_bytes,
            mul(mapping.chunks.max_input_bytes as u64, 4)?,
            mul(planning.max_total_label_bytes as u64, 8)?, plans, results])?;
        let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound, self.allocator_reserve_bytes])?;
        let max_output_bytes = mapping.max_result_bytes;
        let output_reservation_bytes = sum(&[mul(max_output_bytes, 8)?, 4096])?;
        sum(&[native.kv_bytes, workspace_bytes, input_bytes, output_reservation_bytes])?;
        Ok(Int4ClassificationMapMemoryRequirement { native, workspace_bytes, input_bytes,
            output_reservation_bytes, max_output_bytes })
    }
}
fn mul(a: u64, b: u64) -> Result<u64, Int4ClassificationMapError> {
    a.checked_mul(b).ok_or(Int4ClassificationMapError::Limits)
}
fn sum(values: &[u64]) -> Result<u64, Int4ClassificationMapError> {
    values.iter().try_fold(0_u64, |a, &b| a.checked_add(b).ok_or(Int4ClassificationMapError::Limits))
}

#[cfg(test)] mod tests;
