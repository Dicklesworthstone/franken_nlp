//! Pure simultaneous-owner quote for complete-source faithfulness. Native
//! execution reserves one serial cache or an explicitly requested ragged
//! cohort; all source/claim plans and complete results remain owned.
use super::{Int4FaithfulnessMapError, Int4FaithfulnessMapLimits, MAX_FAITHFULNESS_MAP_HEADS,
    cohort::COHORT_WRAPPER_BYTES};
use crate::{native_engine::strict_int4::{Int4MemoryRequirement,
        cohort::MAX_INT4_SEQUENCES, scoring::cohort::required_task_workspace_bytes},
    tasks::{judge::JudgeLimits, ir::TaskBudget}, tokenizer::untrusted::UntrustedEncoding};

pub const MIN_FAITHFULNESS_MAP_PREPARATION_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct Int4FaithfulnessMapMemoryLimits {
    pub context_tokens: usize,
    pub preparation_reserve_bytes: u64,
    pub allocator_reserve_bytes: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4FaithfulnessMapMemoryRequirement {
    pub native: Int4MemoryRequirement,
    pub workspace_bytes: u64,
    pub input_bytes: u64,
    pub output_reservation_bytes: u64,
    pub max_output_bytes: u64,
}
impl Int4FaithfulnessMapMemoryRequirement {
    /// Add one transient encoder workspace while retaining all document and
    /// evidence plans. Probes, heads and cohorts reuse this same workspace.
    pub fn with_input_encoding(mut self, encoding: &UntrustedEncoding)
        -> Result<Self, Int4FaithfulnessMapError> {
        let workspace = match encoding {
            UntrustedEncoding::ByteFallback => 0,
            UntrustedEncoding::SafeBpeV1(limits) => limits.requirements()
                .map_err(|_| Int4FaithfulnessMapError::Limits)?.workspace_bytes,
        };
        self.input_bytes = sum(&[self.input_bytes, workspace])?;
        sum(&[self.native.kv_bytes, self.workspace_bytes, self.input_bytes, self.output_reservation_bytes])?;
        Ok(self)
    }
}
/// `memory.input_bytes` already includes `task_workspace_bytes`; callers do
/// not add the serial quote or task workspace again.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4FaithfulnessMapCohortMemoryRequirement {
    pub memory: Int4FaithfulnessMapMemoryRequirement,
    pub task_workspace_bytes: u64,
    pub max_cohort_output_bytes: u64,
}
impl Int4FaithfulnessMapMemoryLimits {
    pub fn requirements(self, width: usize, task: TaskBudget, planning: JudgeLimits,
        mapping: Int4FaithfulnessMapLimits) -> Result<Int4FaithfulnessMapMemoryRequirement, Int4FaithfulnessMapError> {
        self.requirements_with_max_heads(width, task, planning, mapping, MAX_FAITHFULNESS_MAP_HEADS)
    }

    /// Quote complete plan ownership under a declared per-chunk head cap.
    /// Callers must bind that cap to the fixed evidence policy and reject
    /// any prepared chunk that exceeds it before using this reservation.
    /// Serial execution still owns only one physical KV row. Result and
    /// evidence limits are unchanged; no source window may be omitted.
    pub fn requirements_with_max_heads(self, width: usize, task: TaskBudget, planning: JudgeLimits,
        mapping: Int4FaithfulnessMapLimits, max_heads: usize)
        -> Result<Int4FaithfulnessMapMemoryRequirement, Int4FaithfulnessMapError> {
        mapping.validate(task)?;
        let scoring = planning.per_head;
        if !(1..=MAX_FAITHFULNESS_MAP_HEADS).contains(&max_heads)
            || self.preparation_reserve_bytes < MIN_FAITHFULNESS_MAP_PREPARATION_BYTES
            || self.allocator_reserve_bytes == 0 || mapping.chunks.context_tokens > self.context_tokens
            || planning.max_total_prompt_tokens == 0 || planning.max_total_candidate_tokens == 0
            || planning.max_total_projected_logits == 0 || planning.max_output_bytes == 0
            || scoring.max_candidates == 0 || scoring.max_total_tokens == 0 || scoring.max_nodes == 0
            || scoring.max_depth == 0 || scoring.max_candidate_id_bytes == 0 || scoring.max_projected_logits == 0 {
            return Err(Int4FaithfulnessMapError::Limits);
        }
        let native = Int4MemoryRequirement::for_context(self.context_tokens, width)?;
        if native.kv_bytes > task.max_kv_bytes { return Err(Int4FaithfulnessMapError::Admission); }
        let chunks = mapping.chunks.max_chunks as u64;
        let max_heads = max_heads as u64;
        let per_chunk_prompt = mul(u64::from(task.max_input_tokens), max_heads)?
            .min(planning.max_total_prompt_tokens as u64);
        let declared_tokens = mul(per_chunk_prompt, chunks)?;
        // The next full chunk can be prepared before its aggregate work check
        // rejects it. All earlier accepted plans still coexist at that point.
        let plan_tokens = declared_tokens.min(sum(&[
            mapping.max_model_work.forward_positions.min(declared_tokens), per_chunk_prompt])?);
        let heads = mul(max_heads, chunks)?;
        // The E/C/U language is fixed: three one-byte continuations and scored
        // EOS. Generic caller trie ceilings cannot enlarge this closed task.
        let plans = sum(&[mul(plan_tokens, 64)?, mul(heads, 16_384)?,
            mul(mul(8, 256)?, heads)?, mul(mapping.chunks.max_input_bytes as u64, 4)?])?;
        let per_chunk_result = task.max_output_bytes.min(planning.max_output_bytes);
        let retained = mul(sum(&[per_chunk_result, 4096])?, chunks)?.min(mapping.max_retained_result_bytes);
        // The next complete chunk, its serialization, and lifted aggregate
        // witnesses coexist with the retained frontier. No evidence is pruned.
        let results = mul(sum(&[retained, per_chunk_result, 4096])?, 8)?;
        let input_bytes = sum(&[self.preparation_reserve_bytes,
            mul(mapping.chunks.max_input_bytes as u64, 4)?, plans, results])?;
        let workspace_bytes = sum(&[native.rope_bytes, native.scratch_payload_bound, self.allocator_reserve_bytes])?;
        let max_output_bytes = mapping.max_result_bytes;
        let output_reservation_bytes = sum(&[mul(max_output_bytes, 8)?, 4096])?;
        sum(&[native.kv_bytes, workspace_bytes, input_bytes, output_reservation_bytes])?;
        Ok(Int4FaithfulnessMapMemoryRequirement { native, workspace_bytes, input_bytes,
            output_reservation_bytes, max_output_bytes })
    }

    /// Conservative maximum of an admissible ragged window. There can be at
    /// most 64 simultaneous native heads, regardless of the per-chunk bound.
    /// Pricing min(max_heads * chunks, 64) never changes execution: preparation
    /// separately rejects any actual requested window with more than 64 heads.
    pub fn cohort_requirements(self, width: usize, task: TaskBudget, planning: JudgeLimits,
        mapping: Int4FaithfulnessMapLimits, max_heads: usize, cohort_chunks: usize)
        -> Result<Int4FaithfulnessMapCohortMemoryRequirement, Int4FaithfulnessMapError> {
        let mut memory = self.requirements_with_max_heads(width, task, planning, mapping, max_heads)?;
        if max_heads == 0 || max_heads > MAX_FAITHFULNESS_MAP_HEADS || cohort_chunks == 0
            || cohort_chunks > mapping.chunks.max_chunks || cohort_chunks > MAX_INT4_SEQUENCES {
            return Err(Int4FaithfulnessMapError::Limits);
        }
        let heads = max_heads.checked_mul(cohort_chunks).ok_or(Int4FaithfulnessMapError::Limits)?
            .min(MAX_INT4_SEQUENCES);
        if mul(memory.native.kv_bytes, max_heads as u64)? > task.max_kv_bytes {
            return Err(Int4FaithfulnessMapError::Admission);
        }
        let contexts = [self.context_tokens; MAX_INT4_SEQUENCES];
        memory.native = Int4MemoryRequirement::for_cohort(&contexts[..heads], width)?;
        memory.workspace_bytes = sum(&[memory.native.rope_bytes, memory.native.scratch_payload_bound,
            self.allocator_reserve_bytes])?;
        let per_chunk_result = task.max_output_bytes.min(planning.max_output_bytes);
        let max_cohort_output_bytes = sum(&[mul(per_chunk_result, cohort_chunks as u64)?, COHORT_WRAPPER_BYTES])?;
        if max_cohort_output_bytes > super::MAX_BYTES { return Err(Int4FaithfulnessMapError::Limits); }
        let task_workspace_bytes = required_task_workspace_bytes(mapping.max_model_work, heads, max_cohort_output_bytes)?;
        let additional_results = mul(max_cohort_output_bytes.checked_sub(per_chunk_result)
            .ok_or(Int4FaithfulnessMapError::Limits)?, 8)?;
        memory.input_bytes = sum(&[memory.input_bytes, task_workspace_bytes, additional_results])?;
        memory.max_output_bytes = sum(&[mapping.max_result_bytes, COHORT_WRAPPER_BYTES])?;
        memory.output_reservation_bytes = sum(&[mul(memory.max_output_bytes, 8)?, 4096])?;
        sum(&[memory.native.kv_bytes, memory.workspace_bytes, memory.input_bytes, memory.output_reservation_bytes])?;
        Ok(Int4FaithfulnessMapCohortMemoryRequirement { memory, task_workspace_bytes, max_cohort_output_bytes })
    }
}
fn mul(a: u64, b: u64) -> Result<u64, Int4FaithfulnessMapError> {
    a.checked_mul(b).ok_or(Int4FaithfulnessMapError::Limits)
}
fn sum(values: &[u64]) -> Result<u64, Int4FaithfulnessMapError> {
    values.iter().try_fold(0_u64, |a, &b| a.checked_add(b).ok_or(Int4FaithfulnessMapError::Limits))
}

#[cfg(test)] mod tests;
