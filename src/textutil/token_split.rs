//! Exact per-chunk inspection with the same BOS/EOS-free encoding as
//! `fnlp tokens --no-bos`. Each candidate is encoded independently.
use super::{ReportSplitOptions, SplitOptions, SplitText, TextBudget, TextChunk, TextError};
use crate::{
    native_engine::decode::DecodeCancellationKind,
    tasks::mapreduce::{ChunkLimits, ChunkPlan, MapReduceError},
    tokenizer::{bpe::EncodeOptions, embedded::EmbeddedTokenizer},
};

pub const TOKEN_SPLIT_PROFILE: &str = "utf8-partition-pinned-sp-bpe-no-bos-eos-v1";
// Keep uninterruptible encoder calls small. The cumulative limits include retries.
const MAX_PROBE_BYTES: usize = 4096;
const MAX_PROBES: usize = 65_536;

fn admit(source: &str, options: SplitOptions, max_chunk_tokens: usize, budget: TextBudget) -> Result<(), TextError> {
    budget.admit(source)?;
    if !(4..=super::MAX_BYTES).contains(&options.max_chunk_bytes) || max_chunk_tokens == 0 {
        return Err(TextError::InvalidOptions);
    }
    if source.len() > budget.max_output_bytes { return Err(TextError::OutputBudget); }
    if budget.max_items == 0 && !source.is_empty() { return Err(TextError::ItemBudget); }
    Ok(())
}

/// Lossless partition into independently encoded chunks. Counts recognize the
/// pinned added tokens and exclude inserted BOS/EOS, scaffolding and output.
/// Splits may underfill a budget. All limits apply before result publication.
pub fn split_by_tokens(source: &str, options: SplitOptions, max_chunk_tokens: usize, budget: TextBudget)
    -> Result<SplitText<'_>, TextError> {
    admit(source, options, max_chunk_tokens, budget)?;
    let tokenizer = EmbeddedTokenizer::pinned().map_err(|_| TextError::Tokenization)?;
    split_by_tokens_cached(source, options, max_chunk_tokens, budget, &tokenizer, || None)
}

pub(crate) fn split_by_tokens_cached<'s>(
    source: &'s str, options: SplitOptions, max_chunk_tokens: usize, budget: TextBudget,
    tokenizer: &EmbeddedTokenizer, mut cancelled: impl FnMut() -> Option<DecodeCancellationKind>,
) -> Result<SplitText<'s>, TextError> {
    admit(source, options, max_chunk_tokens, budget)?;
    // A tiny token limit must not repeatedly tokenize a 4 KiB suffix to emit
    // a single token. This bounds probe geometry; actual counts decide fit.
    let max_candidate = options.max_chunk_bytes.min(MAX_PROBE_BYTES);
    let mut probe_cap = max_chunk_tokens.saturating_mul(8).clamp(4, max_candidate);
    let mut probe_bytes = source.len().saturating_mul(32);
    let mut max_probes = MAX_PROBES;
    let mut whole_tokens = None;
    if source.len() > probe_cap && source.len() <= max_candidate {
        // The geometric starting width is a work heuristic, not a token bound.
        // A longer added token (or ordinary merged piece) may fit intact.
        if let Some(cause) = cancelled() { return Err(TextError::Cancelled(cause)); }
        probe_bytes -= source.len();
        max_probes -= 1;
        let count = tokenizer.tokenizer().encode_ids_with_options(source, EncodeOptions { add_bos: false, add_eos: false })
            .map_err(|_| TextError::Tokenization)?.len();
        if let Some(cause) = cancelled() { return Err(TextError::Cancelled(cause)); }
        if count <= max_chunk_tokens {
            probe_cap = source.len();
            whole_tokens = Some(count);
        }
    }
    let limits = ChunkLimits {
        max_input_bytes: budget.max_input_bytes, max_chunk_bytes: options.max_chunk_bytes,
        max_chunk_tokens, context_tokens: max_chunk_tokens, reserved_tokens: 0,
        max_chunks: budget.max_items.max(1), max_tokenizer_calls: max_probes,
    };
    let mut cause = None;
    let plan = ChunkPlan::build_with_checkpoints_and_byte_cap(source, limits, probe_cap, |text| {
        if text.len() == source.len() && let Some(count) = whole_tokens { return Ok(count); }
        probe_bytes = probe_bytes.checked_sub(text.len()).ok_or(MapReduceError::TokenizerBudget)?;
        tokenizer.tokenizer().encode_ids_with_options(text, EncodeOptions { add_bos: false, add_eos: false })
            .map(|ids| ids.len()).map_err(|_| MapReduceError::Tokenizer)
    }, || {
        cause = cancelled();
        if cause.is_some() { Err(MapReduceError::Cancelled) } else { Ok(()) }
    }).map_err(|error| match error {
        MapReduceError::Cancelled => cause.map_or(TextError::Tokenization, TextError::Cancelled),
        MapReduceError::InvalidLimits => TextError::InvalidOptions,
        MapReduceError::InputBudget => TextError::InputBudget,
        MapReduceError::ChunkBudget => TextError::ItemBudget,
        MapReduceError::TokenBudget => TextError::TokenBudget,
        MapReduceError::TokenizerBudget => TextError::TokenizerWorkBudget,
        MapReduceError::AllocationRefused => TextError::AllocationRefused,
        MapReduceError::Tokenizer | MapReduceError::InvalidTokenCount | MapReduceError::InvalidSpan => TextError::Tokenization,
    })?;
    let mut chunks = Vec::new();
    chunks.try_reserve_exact(plan.chunks().len()).map_err(|_| TextError::AllocationRefused)?;
    let mut remaining = budget.max_output_bytes;
    for candidate in plan.chunks() {
        if let Some(cause) = cancelled() { return Err(TextError::Cancelled(cause)); }
        let chunk = TextChunk { span: candidate.span(), text: candidate.text(), tokens: Some(candidate.tokens()) };
        let size = crate::canonjson::canonical_bytes(&chunk).map_err(|_| TextError::Serialization)?.len();
        remaining = remaining.checked_sub(size).ok_or(TextError::OutputBudget)?;
        chunks.push(chunk);
    }
    let original_scalars = chunks.last().map_or(0, |chunk| chunk.span.scalar_end);
    let result = SplitText { schema_version: 1, profile: TOKEN_SPLIT_PROFILE,
        options: ReportSplitOptions { max_chunk_bytes: options.max_chunk_bytes, max_chunk_tokens: Some(max_chunk_tokens) },
        original_bytes: source.len(), original_scalars, chunks };
    budget.result(&result)?;
    if let Some(cause) = cancelled() { return Err(TextError::Cancelled(cause)); }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::embedded::PINNED_ADDED_TOKENS_BYTES;

    #[test]
    fn every_intact_pinned_added_token_fits_a_one_token_chunk() {
        let tokenizer = EmbeddedTokenizer::pinned().unwrap();
        let registry: std::collections::BTreeMap<String, u32> =
            serde_json::from_slice(PINNED_ADDED_TOKENS_BYTES).unwrap();
        for (marker, id) in registry {
            let ids = tokenizer.tokenizer().encode_ids_with_options(&marker, EncodeOptions { add_bos: false, add_eos: false }).unwrap();
            assert_eq!(ids, vec![id]);
            let result = split_by_tokens_cached(&marker, SplitOptions::default(), 1, TextBudget::default(), &tokenizer, || None).unwrap();
            assert_eq!(result.chunks().len(), 1);
            assert_eq!(result.chunks()[0].text, marker);
            assert_eq!(result.chunks()[0].tokens, Some(1));
        }
    }
}
