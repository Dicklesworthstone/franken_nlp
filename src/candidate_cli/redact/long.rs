//! Explicit long-document mode. Short native redaction keeps its old contract.
use super::*;
use crate::{native_engine::{portable_int8::ProjectionWork, strict_int8::Int8Work},
    tasks::{mapreduce::{ChunkLimits, ExecutionLimits},
        redact::long::LongRedactionConfig, source_planning::quantized::long::Int8SourceMapLimits}};

#[derive(Args)]
pub(in crate::candidate_cli) struct LongArgs {
    /// Run source-aligned NER chunks, whole-document rules and fresh verification.
    /// NER chunk boundaries may split entities; a clean scan is not anonymity.
    #[arg(long)]
    pub chunked: bool,
    /// Maximum NER chunks in EACH original/transformed stage (default 64).
    #[arg(long, requires = "chunked")]
    max_ner_chunks: Option<usize>,
    /// Additional per-chunk byte cap; actual pinned context fit is still checked.
    #[arg(long, requires = "chunked")]
    max_ner_chunk_bytes: Option<usize>,
    #[arg(long, requires = "chunked")]
    max_ner_tokenizer_calls: Option<usize>,
    /// Complete intermediate map/result cap per stage (default 16 MiB).
    /// Live and cumulative value bounds are respectively twice and four times this.
    #[arg(long, requires = "chunked")]
    max_ner_map_bytes: Option<usize>,
    /// BOTH stages together; no renewal per chunk (default 64000000000).
    #[arg(long, requires = "chunked")]
    max_total_mask_node_visits: Option<u64>,
    #[arg(long, requires = "chunked")]
    max_forward_positions: Option<u64>,
    #[arg(long, requires = "chunked")]
    max_projected_logits: Option<u64>,
    #[arg(long, requires = "chunked")]
    max_attention_pairs: Option<u64>,
    #[arg(long, requires = "chunked")]
    max_dot_products: Option<u64>,
    #[arg(long, requires = "chunked")]
    max_multiply_accumulates: Option<u64>,
}
impl LongArgs {
    fn chunks(&self) -> usize { self.max_ner_chunks.unwrap_or(64) }
    fn map_bytes(&self) -> usize { self.max_ner_map_bytes.unwrap_or(16 * 1024 * 1024) }
    fn mask_visits(&self) -> u64 { self.max_total_mask_node_visits.unwrap_or(64_000_000_000) }
    fn work(&self) -> Int8Work {
        Int8Work { forward_positions: self.max_forward_positions.unwrap_or(262_144),
            projected_logits: self.max_projected_logits.unwrap_or(10_000_000_000),
            attention_pairs: self.max_attention_pairs.unwrap_or(1_000_000_000_000),
            projections: ProjectionWork { dot_products: self.max_dot_products.unwrap_or(1_000_000_000_000),
                multiply_accumulates: self.max_multiply_accumulates.unwrap_or(10_000_000_000_000_000) } }
    }
    pub(super) fn validate(&self, host: &SourceHostArgs, limits: Limits) -> Result<(), CandidateError> {
        if !self.chunked {
            if self.max_ner_chunks.is_some() || self.max_ner_chunk_bytes.is_some() || self.max_ner_tokenizer_calls.is_some()
                || self.max_ner_map_bytes.is_some() || self.max_total_mask_node_visits.is_some()
                || self.max_forward_positions.is_some() || self.max_projected_logits.is_some() || self.max_attention_pairs.is_some()
                || self.max_dot_products.is_some() || self.max_multiply_accumulates.is_some() { return Err(CandidateError::Arguments); }
            return Ok(());
        }
        let work = self.work();
        if host.max_input_bytes < 4 || !(1..=256).contains(&self.chunks())
            || self.max_ner_chunk_bytes.is_some_and(|n| !(4..=MAX_INPUT_BYTES).contains(&n))
            || !(1..=1_000_000).contains(&self.max_ner_tokenizer_calls.unwrap_or(8192))
            || !(1..=16 * 1024 * 1024).contains(&self.map_bytes()) || host.max_result_bytes > self.map_bytes()
            || self.mask_visits() < host.max_mask_node_visits || self.mask_visits() > 1_000_000_000_000_000
            || work.forward_positions == 0 || work.forward_positions > 16 * 1024 * 1024
            || work.projected_logits == 0 || work.attention_pairs == 0
            || work.projections.dot_products == 0 || work.projections.multiply_accumulates == 0 {
            return Err(CandidateError::Arguments);
        }
        // All exact native grammars/prompts are prepared before each stage's
        // first forward. Reserve the complete set, not a single chunk's planner.
        let per_chunk = (host.context_tokens as u64).checked_mul(32)
            .and_then(|n| n.checked_add(u64::from(host.max_grammar_states).checked_mul(512)?))
            .ok_or(CandidateError::Arguments)?;
        let floor = per_chunk.checked_mul(self.chunks() as u64)
            .and_then(|n| n.checked_add(256 * MIB))
            .and_then(|n| n.checked_add(host.max_input_bytes.max(host.max_result_bytes) as u64 * 64))
            .and_then(|n| n.checked_add(self.map_bytes() as u64 * 2)).ok_or(CandidateError::Arguments)?;
        if limits.preparation_bytes < floor { return Err(CandidateError::Arguments); }
        Ok(())
    }
    pub(in crate::candidate_cli) fn config(&self, host: &SourceHostArgs, limits: Limits, ner: NerOptions)
        -> Result<LongRedactionConfig, CandidateError> {
        if !self.chunked { return Err(CandidateError::Arguments); }
        self.validate(host, limits)?;
        let max_input = host.max_input_bytes.max(host.max_result_bytes);
        let mut planning = host.planning(); planning.max_input_bytes = max_input;
        let chunks = ChunkLimits { max_input_bytes: max_input,
            max_chunk_bytes: self.max_ner_chunk_bytes.unwrap_or(host.context_tokens).min(max_input),
            max_chunk_tokens: host.context_tokens, context_tokens: host.context_tokens,
            reserved_tokens: host.max_new_tokens, max_chunks: self.chunks(),
            max_tokenizer_calls: self.max_ner_tokenizer_calls.unwrap_or(8192) };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        let bytes = self.map_bytes();
        Ok(LongRedactionConfig { ner, per_chunk: host.task_budget(limits), planning,
            mapping: Int8SourceMapLimits { chunks,
                reduction: ExecutionLimits { map_batch_chunks: 1, reduce_fan_in: 8,
                    max_reduction_levels: 8, max_task_calls: 1024, max_value_bytes: bytes,
                    max_live_value_bytes: bytes * 2, max_total_value_bytes: bytes * 4, max_result_bytes: bytes },
                max_model_work: self.work(), mask_limits: host.masks(), mask_visits_per_chunk: host.max_mask_node_visits,
                max_mask_visits: self.mask_visits() }, max_result_bytes: host.max_result_bytes as u64 })
    }
}

#[cfg(test)] mod tests;
