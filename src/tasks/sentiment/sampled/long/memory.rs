//! Pure bounds for one compiler/vocabulary and all retained chunk plans.
//! No source, model, tokenizer or native allocation is needed for this quote.
use super::*;
use crate::native_engine::strict_int4::Int4MemoryRequirement;

pub const MIN_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES: u64 = 512 * MIB;
const PLAN_BYTES_PER_TOKEN: u64 = 64;
const CHUNK_METADATA_BYTES: u64 = 65_536;

#[derive(Clone, Copy, Debug)]
pub struct SampledSentimentMapMemoryLimits {
    pub context_tokens: usize,
    pub preparation_reserve_bytes: u64,
    pub allocator_reserve_bytes: u64,
    pub max_axes: usize,
    pub max_sampler_bytes: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampledSentimentMapMemoryRequirement {
    pub native: Int4MemoryRequirement,
    /// RoPE, native scratch, allocator and sampler; excludes KV and optional
    /// physical prefill/team scratch, which the host must add independently.
    pub workspace_bytes: u64,
    /// Shared compiler/vocabulary, source, all prompts/grammars/results and
    /// one shared encoder workspace. Do not add its components twice.
    pub input_bytes: u64,
    pub output_reservation_bytes: u64,
    pub max_output_bytes: u64,
    pub max_output_tokens: u64,
}
impl SampledSentimentMapMemoryLimits {
    pub fn requirements(self, group_width: usize, task: TaskBudget,
        planning: SampledSentimentLimits, mapping: SampledSentimentMapLimits,
        options: &SampledSentimentOptions, encoding: &UntrustedEncoding)
        -> Result<SampledSentimentMapMemoryRequirement, SampledSentimentMapError> {
        mapping.validate(task)?; planning.validate()?; options.validate()?;
        encoding.validate().map_err(|_| SampledSentimentMapError::Limits)?;
        if !(1..=SentimentAxis::ALL.len()).contains(&self.max_axes)
            || self.preparation_reserve_bytes < MIN_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES
            || self.allocator_reserve_bytes == 0 || mapping.chunks.context_tokens > self.context_tokens
            || self.max_sampler_bytes < json_sampling_workspace_bytes().map_err(SampledSentimentError::from)? {
            return Err(SampledSentimentMapError::Limits);
        }
        let native = Int4MemoryRequirement::for_context(self.context_tokens, group_width)?;
        if native.kv_bytes > task.max_kv_bytes { return Err(SampledSentimentMapError::Admission); }
        let samples = multiply(self.max_axes as u64, options.samples as u64)?;
        let per_chunk_output_tokens = multiply(samples, options.max_sample_tokens as u64)?;
        if per_chunk_output_tokens > u64::from(task.max_output_tokens) {
            return Err(SampledSentimentMapError::Limits);
        }
        let chunks = mapping.chunks.max_chunks as u64;
        let max_output_tokens = multiply(per_chunk_output_tokens, chunks)?;
        let per_chunk_prompts = multiply(u64::from(task.max_input_tokens), self.max_axes as u64)?
            .min((planning.max_total_prompt_tokens / options.samples) as u64);
        // Accepted plans fit the complete cold forward cap. One further plan
        // can be live when that cap refuses it; price this transient as well.
        let declared = multiply(per_chunk_prompts, chunks)?;
        let prompt_tokens = declared.min(add(&[
            (mapping.max_model_work.forward_positions / options.samples as u64).min(declared), per_chunk_prompts])?);
        let encoding_workspace = match encoding {
            UntrustedEncoding::ByteFallback => 0,
            UntrustedEncoding::SafeBpeV1(limits) => limits.requirements()
                .map_err(|_| SampledSentimentMapError::Limits)?.workspace_bytes,
        };
        let mut input_bytes = prepared_bytes(planning, mapping, mapping.chunks.max_chunks,
            prompt_tokens, encoding_workspace)?;
        input_bytes = add(&[input_bytes,
            self.preparation_reserve_bytes - MIN_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES])?;
        let workspace_bytes = add(&[native.rope_bytes, native.scratch_payload_bound,
            self.allocator_reserve_bytes, self.max_sampler_bytes])?;
        let output_reservation_bytes = add(&[multiply(mapping.max_result_bytes, 4)?,
            multiply(max_output_tokens, 8)?, 4096])?;
        add(&[native.kv_bytes, workspace_bytes, input_bytes, output_reservation_bytes])?;
        Ok(SampledSentimentMapMemoryRequirement { native, workspace_bytes, input_bytes,
            output_reservation_bytes, max_output_bytes: mapping.max_result_bytes, max_output_tokens })
    }
}

/// The fixed sampled schema is recompiled per sealed child budget. Reserve
/// those retained programs separately from the single pinned compiler/trie.
/// Bounds include schema/length-rule trees and strings, automaton states and
/// labelled transitions, runtime nodes/literals/exponents, and compile staging.
/// Legal-token masks are transient in the shared native workspace, not cached
/// once per retained chunk. This is declared payload admission, not RSS.
pub(super) fn grammar_bytes(planning: SampledSentimentLimits) -> Result<u64, SampledSentimentMapError> {
    let compile = planning.compile;
    add(&[multiply(compile.max_schema_bytes as u64, 64)?,
        multiply(compile.max_states as u64, 256)?, multiply(compile.max_transitions as u64, 128)?,
        multiply(planning.max_sample_json_bytes as u64, 16)?, 128 * 1024])
}

pub(super) fn prepared_bytes(planning: SampledSentimentLimits, mapping: SampledSentimentMapLimits,
    chunks: usize, prompt_tokens: u64, encoder: u64) -> Result<u64, SampledSentimentMapError> {
    prepared_bytes_bound(planning, mapping.chunks, mapping.max_retained_result_bytes,
        chunks, prompt_tokens, encoder)
}

/// Model-independent retained grammar, source, prompt and result storage.
/// Both numerical profiles use the same compiler and document representation.
pub(crate) fn prepared_bytes_bound(planning: SampledSentimentLimits, chunk_limits: ChunkLimits,
    max_retained_result_bytes: u64, chunks: usize, prompt_tokens: u64, encoder: u64)
    -> Result<u64, SampledSentimentMapError> {
    let chunks = chunks as u64;
    let programs = multiply(chunks, add(&[grammar_bytes(planning)?, CHUNK_METADATA_BYTES])?)?;
    // Retained complete results coexist with the next complete chunk, its
    // canonical validation staging, and the document reducer's bounded copies.
    let result_bytes = add(&[max_retained_result_bytes, planning.max_result_bytes, 4096])?;
    add(&[MIN_SAMPLED_SENTIMENT_MAP_PREPARATION_BYTES,
        multiply(chunk_limits.max_input_bytes as u64, 4)?,
        multiply(prompt_tokens, PLAN_BYTES_PER_TOKEN)?, programs,
        multiply(result_bytes, 8)?, encoder])
}
fn multiply(a: u64, b: u64) -> Result<u64, SampledSentimentMapError> {
    a.checked_mul(b).ok_or(SampledSentimentMapError::Limits)
}
fn add(values: &[u64]) -> Result<u64, SampledSentimentMapError> {
    values.iter().try_fold(0_u64, |total, &value| total.checked_add(value).ok_or(SampledSentimentMapError::Limits))
}

#[cfg(test)] mod tests;
