//! Explicit long-document discovery; supplied-mention resolution is unchanged.
use super::*;
use crate::{corpus::entities_int8::long::Int8DocumentEntityConfig, tasks::mapreduce::ChunkLimits};

#[derive(Args)]
pub(in crate::candidate_cli) struct DiscoveryChunkArgs {
    /// Discover NER in source-aligned chunks, then resolve the ORIGINAL snapshot.
    /// NER boundaries may split entities. No per-chunk clustering or recall claim.
    #[arg(long, requires = "discover_entities")]
    pub chunked: bool,
    /// Maximum NER chunks in each original document (default 64, maximum 256).
    #[arg(long, requires = "chunked")]
    max_ner_chunks: Option<usize>,
    /// Additional chunk ceiling across the entire snapshot (default 1024).
    #[arg(long, requires = "chunked")]
    max_snapshot_ner_chunks: Option<usize>,
    /// Chunk byte ceiling (default 4096); actual pinned token/context fit is checked.
    #[arg(long, requires = "chunked")]
    max_ner_chunk_bytes: Option<usize>,
    /// Pinned tokenizer calls per document partition (default 8192).
    #[arg(long, requires = "chunked")]
    max_ner_tokenizer_calls: Option<usize>,
}
impl DiscoveryChunkArgs {
    fn per_document(&self) -> usize { self.max_ner_chunks.unwrap_or(64) }
    fn snapshot(&self) -> usize { self.max_snapshot_ner_chunks.unwrap_or(1024) }
    pub(super) fn extra_graph_bytes(&self, discover: bool) -> Result<u64, CandidateError> {
        if !self.chunked {
            if self.max_ner_chunks.is_some() || self.max_snapshot_ner_chunks.is_some()
                || self.max_ner_chunk_bytes.is_some() || self.max_ner_tokenizer_calls.is_some() {
                return Err(CandidateError::Arguments);
            }
            return Ok(0);
        }
        if !discover || !(1..=256).contains(&self.per_document()) || !(1..=16_384).contains(&self.snapshot())
            || self.max_ner_chunk_bytes.is_some_and(|n| !(4..=MAX_INPUT_BYTES).contains(&n))
            || !(1..=1_000_000).contains(&self.max_ner_tokenizer_calls.unwrap_or(8192)) {
            return Err(CandidateError::Arguments);
        }
        // Compact witnesses/extents and geometry, not N simultaneous grammars.
        // Exact retained capacities are independently priced by the native host.
        (self.snapshot() as u64).checked_mul(1024).ok_or(CandidateError::Arguments)
    }
    pub(in crate::candidate_cli) fn configuration(&self, command: &ResolveCommand, entities: Int8EntityConfig)
        -> Result<Int8DocumentEntityConfig, CandidateError> {
        self.extra_graph_bytes(command.discovery.discover_entities)?;
        if !self.chunked || command.host.max_input_bytes < 4 { return Err(CandidateError::Arguments); }
        let chunks = ChunkLimits { max_input_bytes: command.host.max_input_bytes,
            max_chunk_bytes: self.max_ner_chunk_bytes.unwrap_or(4096).min(command.host.max_input_bytes),
            max_chunk_tokens: command.host.context_tokens, context_tokens: command.host.context_tokens,
            reserved_tokens: entities.ner_budget.max_output_tokens as usize,
            max_chunks: self.per_document(), max_tokenizer_calls: self.max_ner_tokenizer_calls.unwrap_or(8192) };
        chunks.effective_token_limit().map_err(|_| CandidateError::Arguments)?;
        // No multiplication or fresh pair budget: entities already contains
        // the single invocation's five native axes and whole-snapshot masks.
        Ok(Int8DocumentEntityConfig { entities, chunks, max_snapshot_chunks: self.snapshot() })
    }
}

#[cfg(test)] mod tests;
