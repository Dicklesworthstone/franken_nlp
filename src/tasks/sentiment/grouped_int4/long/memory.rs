//! Pure, complete-document memory admission shared by hosted and CLI callers.
//!
//! The quote covers declared maxima before source parsing or model IO. It is
//! a conservative ownership model, not an RSS measurement. Native weights,
//! caller-owned input parsing/publication and runtime cleanup are separate.
use super::{Int4SentimentMapError, Int4SentimentMapLimits};
use crate::{native_engine::strict_int4::{Int4MemoryRequirement,
        cohort::MAX_INT4_SEQUENCES, scoring::cohort::required_task_workspace_bytes},
    tasks::{ir::TaskBudget, sentiment::SentimentLimits}};

/// Pinned compiler/tokenizer construction and fixed metadata are admitted
/// independently of the explicit document-dependent payloads below.
pub const MIN_SENTIMENT_MAP_PREPARATION_BYTES: u64 = 128 * 1024 * 1024;
const MAX_WINDOW_BYTES: u64 = 64 * 1024 * 1024;
const WRAPPER_BYTES: u64 = 4096;
const PLAN_BYTES_PER_TOKEN: u64 = 64;
const PLAN_BYTES_PER_AXIS: u64 = 65_536;

#[derive(Clone, Copy, Debug)]
pub struct Int4SentimentMapMemoryLimits {
    pub context_tokens: usize,
    pub preparation_reserve_bytes: u64,
    pub allocator_reserve_bytes: u64,
    /// Before request IO, four prices every supported axis selection.
    pub max_axes: usize,
    /// None reuses one native cache. Some(N) prices N complete chunks.
    pub cohort_chunks: Option<usize>,
}

/// Each field has one owner; do not add either nested subtotal twice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int4SentimentMapMemoryRequirement {
    pub native: Int4MemoryRequirement,
    /// RoPE, native scratch and allocator reserve; excludes KV.
    pub workspace_bytes: u64,
    /// Compiler reserve, source, all plans/results and task workspace.
    /// Already includes `task_workspace_bytes`.
    pub input_bytes: u64,
    /// Complete typed output plus canonical tree/byte staging and framing.
    pub output_reservation_bytes: u64,
    /// Simultaneous flattened prompts, cursors and current window results.
    pub task_workspace_bytes: u64,
    /// Maximum hosted value, including optional physical cohort statistics.
    pub max_output_bytes: u64,
    /// Per-window native task output ceiling, not the complete document cap.
    pub max_cohort_output_bytes: u64,
}

impl Int4SentimentMapMemoryLimits {
    /// Checked calculation only: no tokenizer, source read, model load,
    /// runtime, heap-sized context vector or forward pass is constructed.
    pub fn requirements(self, group_width: usize, task: TaskBudget,
        planning: SentimentLimits, mapping: Int4SentimentMapLimits)
        -> Result<Int4SentimentMapMemoryRequirement, Int4SentimentMapError> {
        mapping.validate(task)?;
        if !(1..=4).contains(&self.max_axes)
            || self.preparation_reserve_bytes < MIN_SENTIMENT_MAP_PREPARATION_BYTES
            || self.allocator_reserve_bytes == 0
            || mapping.chunks.context_tokens > self.context_tokens
            || !valid_planning(planning) {
            return Err(Int4SentimentMapError::Limits);
        }
        let window = self.cohort_chunks.unwrap_or(1);
        if window == 0 || window > MAX_INT4_SEQUENCES || window > mapping.chunks.max_chunks {
            return Err(Int4SentimentMapError::Limits);
        }
        let row = Int4MemoryRequirement::for_context(self.context_tokens, group_width)?;
        // One immutable per-chunk TaskBudget permits every declared axis in
        // either schedule. The physical serial allocation still uses one KV.
        if multiply(row.kv_bytes, self.max_axes as u64)? > task.max_kv_bytes {
            return Err(Int4SentimentMapError::Admission);
        }
        // This is the largest VALID input geometry before its axes are known,
        // not permission to clamp an actual oversized request. Both the CLI
        // and hosted executor reject actual_axes * window > 64 separately.
        let heads = self.max_axes.checked_mul(window).ok_or(Int4SentimentMapError::Limits)?
            .min(MAX_INT4_SEQUENCES);
        let native = if self.cohort_chunks.is_some() {
            let contexts = [self.context_tokens; MAX_INT4_SEQUENCES];
            Int4MemoryRequirement::for_cohort(&contexts[..heads], group_width)?
        } else { row };
        let per_chunk_output = task.max_output_bytes.min(planning.max_output_bytes);
        let max_cohort_output_bytes = multiply(per_chunk_output, window as u64)?;
        if max_cohort_output_bytes == 0 || max_cohort_output_bytes > MAX_WINDOW_BYTES {
            return Err(Int4SentimentMapError::Limits);
        }
        // Whole-document forward allowance is also a safe upper bound for
        // every individual window. The exact ragged windows are checked by
        // the sealed executor before it allocates its first native engine.
        let task_workspace_bytes = required_task_workspace_bytes(mapping.max_model_work,
            if self.cohort_chunks.is_some() { heads } else { self.max_axes },
            max_cohort_output_bytes)?;
        let chunks = mapping.chunks.max_chunks as u64;
        let per_chunk_prompt = multiply(u64::from(task.max_input_tokens), self.max_axes as u64)?
            .min(planning.max_total_prompt_tokens as u64);
        let declared_plan_tokens = multiply(per_chunk_prompt, chunks)?;
        // Accepted plans already fit the complete forward ceiling. One more
        // full plan may exist transiently before that ceiling rejects it.
        let plan_tokens = declared_plan_tokens.min(add(&[
            mapping.max_model_work.forward_positions.min(declared_plan_tokens), per_chunk_prompt])?);
        let plan_axes = multiply(self.max_axes as u64, chunks)?;
        // Retained TaskIR prompts and transient cloning/canonical bindings,
        // plus the fixed five-bin scorer, schedules and admitted identities.
        let plans = add(&[multiply(plan_tokens, PLAN_BYTES_PER_TOKEN)?,
            multiply(plan_axes, PLAN_BYTES_PER_AXIS)?, multiply(chunks, 1024)?])?;
        let retained = multiply(add(&[per_chunk_output, WRAPPER_BYTES])?, chunks)?
            .min(mapping.max_retained_result_bytes);
        // The current window exists before cumulative-result admission can
        // refuse it. Price it in addition to the accepted complete frontier.
        let results = multiply(add(&[retained, max_cohort_output_bytes, WRAPPER_BYTES])?, 8)?;
        let source = multiply(mapping.chunks.max_input_bytes as u64, 4)?;
        let input_bytes = add(&[self.preparation_reserve_bytes, source, plans, results,
            task_workspace_bytes])?;
        let workspace_bytes = add(&[native.rope_bytes, native.scratch_payload_bound,
            self.allocator_reserve_bytes])?;
        let max_output_bytes = add(&[mapping.max_result_bytes, WRAPPER_BYTES])?;
        let output_reservation_bytes = add(&[multiply(max_output_bytes, 4)?, WRAPPER_BYTES])?;
        // Check the same total the hosted ownership path will reserve.
        add(&[native.kv_bytes, workspace_bytes, input_bytes, output_reservation_bytes])?;
        Ok(Int4SentimentMapMemoryRequirement { native, workspace_bytes, input_bytes,
            output_reservation_bytes, task_workspace_bytes, max_output_bytes,
            max_cohort_output_bytes })
    }
}

fn valid_planning(limits: SentimentLimits) -> bool {
    let axis = limits.per_axis;
    axis.max_candidates != 0 && axis.max_total_tokens != 0 && axis.max_nodes != 0
        && axis.max_depth != 0 && axis.max_candidate_id_bytes != 0 && axis.max_projected_logits != 0
        && limits.max_total_prompt_tokens != 0 && limits.max_total_candidate_tokens != 0
        && limits.max_total_nodes != 0 && limits.max_total_projected_logits != 0
        && limits.max_output_bytes != 0
}
fn multiply(a: u64, b: u64) -> Result<u64, Int4SentimentMapError> {
    a.checked_mul(b).ok_or(Int4SentimentMapError::Limits)
}
fn add(values: &[u64]) -> Result<u64, Int4SentimentMapError> {
    values.iter().try_fold(0_u64, |total, &value|
        total.checked_add(value).ok_or(Int4SentimentMapError::Limits))
}

#[cfg(test)] mod tests;
