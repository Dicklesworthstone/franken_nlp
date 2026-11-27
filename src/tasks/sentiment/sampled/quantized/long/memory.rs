//! Pure process-ledger bounds for complete sampled INT8 documents.
use super::*;
use crate::native_engine::strict_int8::Int8MemoryRequirement;
use super::super::super::long::memory::{prepared_bytes_bound, MIN_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES};

pub const MIN_INT8_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES: u64 = MIN_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES;

#[derive(Clone, Copy, Debug)]
pub struct Int8SampledSentimentMapMemoryLimits {
    pub context_tokens: usize,
    pub preparation_reserve_bytes: u64,
    pub allocator_reserve_bytes: u64,
    pub max_axes: usize,
    pub max_sampler_bytes: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Int8SampledSentimentMapMemoryRequirement {
    pub native: Int8MemoryRequirement,
    /// Excludes KV and additional physical prefill/team buffers.
    pub workspace_bytes: u64,
    /// Shared compiler/vocabulary, source, all retained prompts/grammars/results,
    /// construction staging and one cumulative encoding workspace.
    pub input_bytes: u64,
    pub output_reservation_bytes: u64,
    pub max_output_bytes: u64,
    pub max_output_tokens: u64,
}
impl Int8SampledSentimentMapMemoryLimits {
    pub fn requirements(self, task: TaskBudget, planning: SampledSentimentLimits,
        mapping: Int8SampledSentimentMapLimits, options: &SampledSentimentOptions, encoding: &UntrustedEncoding)
        -> Result<Int8SampledSentimentMapMemoryRequirement, Int8SampledSentimentMapError> {
        mapping.validate(task)?; planning.validate()?; options.validate()?;
        encoding.validate().map_err(|_| Int8SampledSentimentMapError::Limits)?;
        if !(1..=SentimentAxis::ALL.len()).contains(&self.max_axes)
            || self.preparation_reserve_bytes < MIN_INT8_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES
            || self.allocator_reserve_bytes == 0 || mapping.chunks.context_tokens > self.context_tokens
            || self.max_sampler_bytes < json_sampling_workspace_bytes().map_err(Int8SampledSentimentError::from)? {
            return Err(Int8SampledSentimentMapError::Limits);
        }
        let native = Int8MemoryRequirement::for_context(self.context_tokens)?;
        if native.kv_bytes > task.max_kv_bytes { return Err(Int8SampledSentimentMapError::Admission); }
        let samples = multiply(self.max_axes as u64, options.samples as u64)?;
        let per_chunk_output_tokens = multiply(samples, options.max_sample_tokens as u64)?;
        if per_chunk_output_tokens > u64::from(task.max_output_tokens) { return Err(Int8SampledSentimentMapError::Limits); }
        let chunks = mapping.chunks.max_chunks as u64;
        let max_output_tokens = multiply(per_chunk_output_tokens, chunks)?;
        let per_chunk_prompts = multiply(u64::from(task.max_input_tokens), self.max_axes as u64)?
            .min((planning.max_total_prompt_tokens / options.samples) as u64);
        // One transient next plan can coexist with retained plans before the
        // complete native-work cap refuses it; price that staging too.
        let declared = multiply(per_chunk_prompts, chunks)?;
        let prompt_tokens = declared.min(add(&[
            (mapping.max_model_work.forward_positions / options.samples as u64).min(declared), per_chunk_prompts])?);
        let encoder = match encoding {
            UntrustedEncoding::ByteFallback => 0,
            UntrustedEncoding::SafeBpeV1(limits) => limits.requirements()
                .map_err(|_| Int8SampledSentimentMapError::Limits)?.workspace_bytes,
        };
        let input_bytes = add(&[
            prepared_bytes_bound(planning, mapping.chunks, mapping.max_retained_result_bytes,
                mapping.chunks.max_chunks, prompt_tokens, encoder)?,
            self.preparation_reserve_bytes - MIN_INT8_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES])?;
        let workspace_bytes = add(&[native.rope_bytes, native.scratch_payload_bound,
            self.allocator_reserve_bytes, self.max_sampler_bytes])?;
        let output_reservation_bytes = add(&[multiply(mapping.max_result_bytes, 4)?,
            multiply(max_output_tokens, 8)?, 4096])?;
        add(&[native.kv_bytes, workspace_bytes, input_bytes, output_reservation_bytes])?;
        Ok(Int8SampledSentimentMapMemoryRequirement { native, workspace_bytes, input_bytes,
            output_reservation_bytes, max_output_bytes: mapping.max_result_bytes, max_output_tokens })
    }
}
fn multiply(a: u64, b: u64) -> Result<u64, Int8SampledSentimentMapError> {
    a.checked_mul(b).ok_or(Int8SampledSentimentMapError::Limits)
}
fn add(values: &[u64]) -> Result<u64, Int8SampledSentimentMapError> {
    values.iter().try_fold(0_u64, |total, &value| total.checked_add(value).ok_or(Int8SampledSentimentMapError::Limits))
}
