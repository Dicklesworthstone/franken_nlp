//! Pure admission for complete-document INT8 sentiment.
//!
//! This quote covers declared maxima before source/tokenizer/model IO. It
//! models simultaneous ownership, not measured RSS. Immutable weights,
//! caller-owned parsing/publication and runtime cleanup are separate charges.
use super::{Int8SentimentMapError, Int8SentimentMapLimits};
use crate::{
    native_engine::strict_int8::Int8MemoryRequirement,
    tasks::{ir::TaskBudget, sentiment::SentimentLimits},
    tokenizer::untrusted::UntrustedEncoding,
};

/// Pinned tokenizer/compiler construction and fixed metadata are admitted
/// separately from the document-dependent payloads below.
pub const MIN_SENTIMENT_MAP_PREPARATION_BYTES: u64 = 128 * 1024 * 1024;
const MAX_CHUNK_OUTPUT_BYTES: u64 = 64 * 1024 * 1024;
const WRAPPER_BYTES: u64 = 4096;
const PLAN_BYTES_PER_TOKEN: u64 = 64;
const PLAN_BYTES_PER_AXIS: u64 = 65_536;

#[derive(Clone, Copy, Debug)]
pub struct Int8SentimentMapMemoryLimits {
    pub context_tokens: usize,
    pub preparation_reserve_bytes: u64,
    pub allocator_reserve_bytes: u64,
    /// Before request IO, four prices every supported axis selection.
    pub max_axes: usize,
}

/// Each field has one owner; nested subtotals must not be added twice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8SentimentMapMemoryRequirement {
    /// One native allocation is reused across every independent chunk/head.
    pub native: Int8MemoryRequirement,
    /// RoPE, native scratch and allocator reserve; excludes KV.
    pub workspace_bytes: u64,
    /// Compiler reserve, source, all retained plans/results and task scratch.
    /// Already includes `task_workspace_bytes`.
    pub input_bytes: u64,
    /// Complete typed output, canonical tree/byte staging and framing.
    pub output_reservation_bytes: u64,
    /// Flattened prompt, score cursor and the current chunk's complete heads.
    pub task_workspace_bytes: u64,
    /// Complete hosted result ceiling, including bounded envelope overhead.
    pub max_output_bytes: u64,
}

impl Int8SentimentMapMemoryRequirement {
    /// One transient encoder workspace lives beside the retained plans and
    /// is reused across all source probes and axis prompts.
    pub fn with_input_encoding(
        mut self,
        encoding: &UntrustedEncoding,
    ) -> Result<Self, Int8SentimentMapError> {
        let workspace = match encoding {
            UntrustedEncoding::ByteFallback => 0,
            UntrustedEncoding::SafeBpeV1(limits) => {
                limits
                    .requirements()
                    .map_err(|_| Int8SentimentMapError::Limits)?
                    .workspace_bytes
            }
        };
        self.input_bytes = add(&[self.input_bytes, workspace])?;
        self.checked_total()?;
        Ok(self)
    }

    fn checked_total(self) -> Result<u64, Int8SentimentMapError> {
        add(&[
            self.native.kv_bytes,
            self.workspace_bytes,
            self.input_bytes,
            self.output_reservation_bytes,
        ])
    }
}

impl Int8SentimentMapMemoryLimits {
    /// Checked arithmetic only: no tokenizer, private source, native engine,
    /// runtime or heap-sized context vector is constructed by this method.
    pub fn requirements(
        self,
        task: TaskBudget,
        planning: SentimentLimits,
        mapping: Int8SentimentMapLimits,
    ) -> Result<Int8SentimentMapMemoryRequirement, Int8SentimentMapError> {
        mapping.validate(task)?;
        if !(1..=4).contains(&self.max_axes)
            || self.preparation_reserve_bytes < MIN_SENTIMENT_MAP_PREPARATION_BYTES
            || self.allocator_reserve_bytes == 0
            || mapping.chunks.max_input_bytes == 0
            || mapping.chunks.context_tokens > self.context_tokens
            || !valid_planning(planning)
        {
            return Err(Int8SentimentMapError::Limits);
        }
        let native = Int8MemoryRequirement::for_context(self.context_tokens)?;
        // The per-chunk TaskBudget continues to cover every independent axis;
        // the physical serial executor nevertheless owns only ONE KV cache.
        if multiply(native.kv_bytes, self.max_axes as u64)? > task.max_kv_bytes {
            return Err(Int8SentimentMapError::Admission);
        }
        let per_chunk_output = task.max_output_bytes.min(planning.max_output_bytes);
        if !(1..=MAX_CHUNK_OUTPUT_BYTES).contains(&per_chunk_output) {
            return Err(Int8SentimentMapError::Limits);
        }
        // The same scalar CandidateScorer owns INT4 and INT8 finite-language
        // cursors. Use its established conservative prompt/cursor/result
        // envelope without importing any INT4 work or allocation authority.
        // Whole-document forwards bound every one-at-a-time prompt. Complete
        // chunk results remain live while later heads and canonical staging
        // are constructed, even though their native caches are sequential.
        let task_workspace_bytes = add(&[
            multiply(mapping.max_model_work.forward_positions, 64)?,
            multiply(per_chunk_output, 8)?,
            multiply(self.max_axes as u64, PLAN_BYTES_PER_AXIS)?,
        ])?;
        let chunks = mapping.chunks.max_chunks as u64;
        let per_chunk_prompt = multiply(u64::from(task.max_input_tokens), self.max_axes as u64)?
            .min(planning.max_total_prompt_tokens as u64);
        let declared_plan_tokens = multiply(per_chunk_prompt, chunks)?;
        // Accepted plans already fit the whole-document forward ceiling.
        // One additional complete plan can exist before that ceiling refuses
        // it, so admission must include that transient rejected plan too.
        let plan_tokens = declared_plan_tokens.min(add(&[
            mapping
                .max_model_work
                .forward_positions
                .min(declared_plan_tokens),
            per_chunk_prompt,
        ])?);
        let plan_axes = multiply(self.max_axes as u64, chunks)?;
        let plans = add(&[
            multiply(plan_tokens, PLAN_BYTES_PER_TOKEN)?,
            multiply(plan_axes, PLAN_BYTES_PER_AXIS)?,
            multiply(chunks, 1024)?,
        ])?;
        let retained = multiply(add(&[per_chunk_output, WRAPPER_BYTES])?, chunks)?
            .min(mapping.max_retained_result_bytes);
        // The next complete chunk precedes cumulative-output admission.
        // It lives beside the accepted frontier and cannot borrow its cap.
        let results = multiply(add(&[retained, per_chunk_output, WRAPPER_BYTES])?, 8)?;
        let source = multiply(mapping.chunks.max_input_bytes as u64, 4)?;
        let input_bytes = add(&[
            self.preparation_reserve_bytes,
            source,
            plans,
            results,
            task_workspace_bytes,
        ])?;
        let workspace_bytes = add(&[
            native.rope_bytes,
            native.scratch_payload_bound,
            self.allocator_reserve_bytes,
        ])?;
        let max_output_bytes = add(&[mapping.max_result_bytes, WRAPPER_BYTES])?;
        let output_reservation_bytes = add(&[multiply(max_output_bytes, 4)?, WRAPPER_BYTES])?;
        let required = Int8SentimentMapMemoryRequirement {
            native,
            workspace_bytes,
            input_bytes,
            output_reservation_bytes,
            task_workspace_bytes,
            max_output_bytes,
        };
        required.checked_total()?;
        Ok(required)
    }
}

fn valid_planning(limits: SentimentLimits) -> bool {
    let axis = limits.per_axis;
    axis.max_candidates != 0
        && axis.max_total_tokens != 0
        && axis.max_nodes != 0
        && axis.max_depth != 0
        && axis.max_candidate_id_bytes != 0
        && axis.max_projected_logits != 0
        && limits.max_total_prompt_tokens != 0
        && limits.max_total_candidate_tokens != 0
        && limits.max_total_nodes != 0
        && limits.max_total_projected_logits != 0
        && limits.max_output_bytes != 0
}
fn multiply(a: u64, b: u64) -> Result<u64, Int8SentimentMapError> {
    a.checked_mul(b).ok_or(Int8SentimentMapError::Limits)
}
fn add(values: &[u64]) -> Result<u64, Int8SentimentMapError> {
    values.iter().try_fold(0_u64, |total, &value| {
        total
            .checked_add(value)
            .ok_or(Int8SentimentMapError::Limits)
    })
}

#[cfg(test)]
mod tests;
