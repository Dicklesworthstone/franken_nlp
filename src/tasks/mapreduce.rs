//! Long-document planning with exact original-source coordinates.
//!
//! Chunks are a lossless partition, not linguistic sentences. Token counts
//! come from the caller's admitted source encoder, never a byte/token guess.
//! This module neither loads a model nor grants runtime/artifact authority.

use std::{error::Error, fmt, ops::Range};

use serde::{Deserialize, Serialize};

use crate::validation::grounded_fields::VerifiedSourceSpan;

pub mod execution;
pub use execution::{
    ExecutionError, ExecutionLimits, MapOutput, MapReduceResult, MapReduceTask,
    ReduceInput, ReductionNode, ReductionPolicy, ReductionWarning, TaskStage, execute,
};

pub const CHUNK_PROFILE: &str = "source-partition-token-shrink-v1";
const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_ITEMS: usize = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MapReduceError {
    InvalidLimits,
    InputBudget,
    ChunkBudget,
    TokenBudget,
    TokenizerBudget,
    InvalidTokenCount,
    Tokenizer,
    InvalidSpan,
    AllocationRefused,
    Cancelled,
}

impl fmt::Display for MapReduceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid map/reduce limits",
            Self::InputBudget => "map/reduce source byte budget exceeded",
            Self::ChunkBudget => "map/reduce chunk count budget exceeded",
            Self::TokenBudget => "one indivisible source unit exceeds the token budget",
            Self::TokenizerBudget => "map/reduce tokenizer work budget exceeded",
            Self::InvalidTokenCount => "source encoder returned zero tokens for nonempty text",
            Self::Tokenizer => "map/reduce source tokenization failed",
            Self::InvalidSpan => "map/reduce span is not a nonempty source-aligned range",
            Self::AllocationRefused => "map/reduce allocation refused",
            Self::Cancelled => "map/reduce cancelled",
        })
    }
}

impl Error for MapReduceError {}

/// All limits are checked before tokenization. `reserved_tokens` covers the
/// task scaffold AND its maximum output; chunk tokens cannot consume them.
/// The source counter must use the same separately encoded source segment as
/// the eventual TaskIR. This is not a claim about arbitrary prompt concatenation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkLimits {
    pub max_input_bytes: usize,
    pub max_chunk_bytes: usize,
    pub max_chunk_tokens: usize,
    pub context_tokens: usize,
    pub reserved_tokens: usize,
    pub max_chunks: usize,
    pub max_tokenizer_calls: usize,
}

impl Default for ChunkLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 1024 * 1024,
            max_chunk_bytes: 4096,
            max_chunk_tokens: 1024,
            context_tokens: 8192,
            reserved_tokens: 512,
            max_chunks: 4096,
            max_tokenizer_calls: 65_536,
        }
    }
}

impl ChunkLimits {
    pub fn effective_token_limit(self) -> Result<usize, MapReduceError> {
        let available = self.context_tokens.checked_sub(self.reserved_tokens)
            .filter(|&n| n != 0).ok_or(MapReduceError::InvalidLimits)?;
        if self.max_input_bytes > MAX_BYTES
            || !(4..=MAX_BYTES).contains(&self.max_chunk_bytes)
            || self.max_chunk_tokens == 0
            || self.max_chunks == 0 || self.max_chunks > MAX_ITEMS
            || self.max_tokenizer_calls == 0 || self.max_tokenizer_calls > MAX_ITEMS
        {
            return Err(MapReduceError::InvalidLimits);
        }
        Ok(available.min(self.max_chunk_tokens))
    }
}

/// Metadata is serializable; source bytes are deliberately not. Coordinates
/// and ids are minted by ChunkPlan and cannot be mutated through its API.
#[derive(Serialize)]
pub struct SourceChunk<'a> {
    id: usize,
    span: VerifiedSourceSpan,
    tokens: usize,
    #[serde(skip)]
    text: &'a str,
}

impl<'a> SourceChunk<'a> {
    pub fn id(&self) -> usize { self.id }
    pub fn span(&self) -> VerifiedSourceSpan { self.span }
    pub fn tokens(&self) -> usize { self.tokens }
    pub fn text(&self) -> &'a str { self.text }
}

/// Borrows the exact original document. No normalization, source copy, raw
/// content digest, cache authority, or implicit persistence is introduced.
pub struct ChunkPlan<'a> {
    source: &'a str,
    chunks: Vec<SourceChunk<'a>>,
    limits: ChunkLimits,
    tokenizer_calls: usize,
}

impl<'a> ChunkPlan<'a> {
    pub fn build<F>(source: &'a str, limits: ChunkLimits, count: F) -> Result<Self, MapReduceError>
    where
        F: FnMut(&str) -> Result<usize, MapReduceError>,
    {
        Self::build_with_checkpoints(source, limits, count, || Ok(()))
    }

    /// A failed/cancelled plan never returns a truncated prefix. Checkpoints
    /// surround encoder calls and publication; the encoder must independently
    /// honor its own allocation/deadline/cancellation limits while running.
    ///
    /// Tokenization is not assumed prefix-monotone. Oversized candidates are
    /// shrunk geometrically and each emitted chunk is counted afresh. This is
    /// deterministic and bounded, not a promise to find the longest fit.
    pub fn build_with_checkpoints<F, C>(
        source: &'a str,
        limits: ChunkLimits,
        mut count: F,
        mut checkpoint: C,
    ) -> Result<Self, MapReduceError>
    where
        F: FnMut(&str) -> Result<usize, MapReduceError>,
        C: FnMut() -> Result<(), MapReduceError>,
    {
        let token_limit = limits.effective_token_limit()?;
        if source.len() > limits.max_input_bytes { return Err(MapReduceError::InputBudget); }
        checkpoint()?;
        let mut chunks = Vec::new();
        let (mut start, mut scalar, mut calls) = (0, 0, 0);
        while start < source.len() {
            if chunks.len() == limits.max_chunks { return Err(MapReduceError::ChunkBudget); }
            let ceiling = start + limits.max_chunk_bytes.min(source.len() - start);
            let mut end = preferred_end(source, start, aligned_end(source, ceiling));
            let tokens = loop {
                checkpoint()?;
                if calls == limits.max_tokenizer_calls { return Err(MapReduceError::TokenizerBudget); }
                calls += 1;
                let tokens = count(&source[start..end])?;
                checkpoint()?;
                if tokens == 0 { return Err(MapReduceError::InvalidTokenCount); }
                if tokens <= token_limit { break tokens; }
                let mut smaller = aligned_end(source, start + (end - start) / 2);
                if smaller <= start {
                    let first = source[start..].chars().next().ok_or(MapReduceError::InvalidSpan)?;
                    smaller = start + first.len_utf8();
                    if first == '\r' && source.as_bytes().get(smaller) == Some(&b'\n') { smaller += 1; }
                }
                if smaller >= end { return Err(MapReduceError::TokenBudget); }
                end = preferred_end(source, start, smaller);
            };
            let text = &source[start..end];
            let next_scalar = scalar + text.chars().count();
            chunks.try_reserve(1).map_err(|_| MapReduceError::AllocationRefused)?;
            chunks.push(SourceChunk {
                id: chunks.len(),
                span: VerifiedSourceSpan { byte_start: start, byte_end: end, scalar_start: scalar, scalar_end: next_scalar },
                tokens,
                text,
            });
            start = end;
            scalar = next_scalar;
        }
        checkpoint()?;
        Ok(Self { source, chunks, limits, tokenizer_calls: calls })
    }

    pub fn chunks(&self) -> &[SourceChunk<'a>] { &self.chunks }
    pub fn limits(&self) -> ChunkLimits { self.limits }
    pub fn tokenizer_calls(&self) -> usize { self.tokenizer_calls }
    pub fn source_bytes(&self) -> usize { self.source.len() }

    /// Lift a chunk-local citation to exact original byte/scalar coordinates.
    /// This proves membership and UTF-8 boundaries, NOT semantic entailment.
    pub fn lift_span(&self, chunk_id: usize, local: Range<usize>) -> Result<VerifiedSourceSpan, MapReduceError> {
        let chunk = self.chunks.get(chunk_id).ok_or(MapReduceError::InvalidSpan)?;
        if local.start >= local.end || chunk.text.get(local.clone()).is_none() {
            return Err(MapReduceError::InvalidSpan);
        }
        Ok(VerifiedSourceSpan {
            byte_start: chunk.span.byte_start + local.start,
            byte_end: chunk.span.byte_start + local.end,
            scalar_start: chunk.span.scalar_start + chunk.text[..local.start].chars().count(),
            scalar_end: chunk.span.scalar_start + chunk.text[..local.end].chars().count(),
        })
    }
}

fn aligned_end(source: &str, mut end: usize) -> usize {
    while !source.is_char_boundary(end) { end -= 1; }
    if end > 0 && source.as_bytes().get(end - 1) == Some(&b'\r')
        && source.as_bytes().get(end) == Some(&b'\n')
    {
        end -= 1;
    }
    end
}

fn preferred_end(source: &str, start: usize, end: usize) -> usize {
    if end == source.len() { return end; }
    source.as_bytes()[start..end].iter().enumerate().rev().find_map(|(index, byte)| {
        let point = start + index + 1;
        (point - start >= (end - start) / 2
            && matches!(*byte, b' ' | b'\t' | b'\n' | b'\r')
            && !(*byte == b'\r' && source.as_bytes().get(point) == Some(&b'\n')))
            .then_some(point)
    }).unwrap_or(end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(text: &str) -> Result<usize, MapReduceError> { Ok(text.chars().count()) }
    fn limits() -> ChunkLimits {
        ChunkLimits { max_chunk_bytes: 8, max_chunk_tokens: 3, ..ChunkLimits::default() }
    }

    #[test]
    fn partition_preserves_unicode_whitespace_crlf_and_all_coordinates() {
        let source = "éAB\r\n😀 e\u{301}\tZ\r\n尾";
        let plan = ChunkPlan::build(source, limits(), count).unwrap();
        assert_eq!(plan.chunks().iter().map(SourceChunk::text).collect::<String>(), source);
        let (mut byte, mut scalar) = (0, 0);
        for (id, chunk) in plan.chunks().iter().enumerate() {
            assert_eq!(chunk.id(), id);
            let span = chunk.span();
            assert_eq!((span.byte_start, span.scalar_start), (byte, scalar));
            assert_eq!(&source[span.byte_start..span.byte_end], chunk.text());
            assert!(chunk.tokens() <= 3 && chunk.text().len() <= 8);
            assert!(!(source.as_bytes().get(span.byte_end.wrapping_sub(1)) == Some(&b'\r')
                && source.as_bytes().get(span.byte_end) == Some(&b'\n')));
            byte = span.byte_end; scalar += chunk.text().chars().count();
            assert_eq!(span.scalar_end, scalar);
        }
        assert_eq!((byte, scalar), (source.len(), source.chars().count()));
    }

    #[test]
    fn context_reservation_is_not_consumed_by_document_tokens() {
        let options = ChunkLimits { context_tokens: 10, reserved_tokens: 8, ..limits() };
        let plan = ChunkPlan::build("abcdef", options, count).unwrap();
        assert!(plan.chunks().iter().all(|chunk| chunk.tokens() <= 2));
        assert!(matches!(ChunkPlan::build("a", ChunkLimits { reserved_tokens: 10, ..options }, count),
            Err(MapReduceError::InvalidLimits)));
    }

    #[test]
    fn every_candidate_is_measured_without_a_monotonic_tokenizer_assumption() {
        let options = ChunkLimits { max_chunk_bytes: 4, max_chunk_tokens: 3, ..limits() };
        let plan = ChunkPlan::build("abcdefgh", options, |text| {
            Ok(if text.len() == 4 { 9 } else { text.len() })
        }).unwrap();
        assert_eq!(plan.chunks().len(), 4);
        assert!(plan.chunks().iter().all(|chunk| chunk.tokens() == 2));
        assert_eq!(plan.tokenizer_calls(), 7);
    }

    #[test]
    fn indivisible_scalar_and_crlf_fail_instead_of_disappearing() {
        let options = ChunkLimits { max_chunk_tokens: 1, ..limits() };
        for text in ["😀", "\r\n"] {
            assert!(matches!(ChunkPlan::build(text, options, |_| Ok(2)), Err(MapReduceError::TokenBudget)));
        }
    }

    #[test]
    fn empty_input_is_empty_and_never_calls_the_encoder() {
        let plan = ChunkPlan::build("", limits(), |_| panic!("unexpected encoder call")).unwrap();
        assert!(plan.chunks().is_empty());
        assert_eq!(plan.tokenizer_calls(), 0);
    }

    #[test]
    fn allocation_work_and_chunk_limits_fail_the_complete_plan() {
        assert!(matches!(ChunkPlan::build("abcdefgh", ChunkLimits { max_input_bytes: 7, ..limits() }, count),
            Err(MapReduceError::InputBudget)));
        assert!(matches!(ChunkPlan::build("abcdefgh", ChunkLimits { max_chunks: 1, ..limits() }, count),
            Err(MapReduceError::ChunkBudget)));
        assert!(matches!(ChunkPlan::build("abcdefgh", ChunkLimits { max_tokenizer_calls: 1, ..limits() }, count),
            Err(MapReduceError::TokenizerBudget)));
        assert!(matches!(ChunkPlan::build("a", limits(), |_| Ok(0)), Err(MapReduceError::InvalidTokenCount)));
    }

    #[test]
    fn cancellation_after_tokenization_does_not_publish_a_chunk() {
        let mut checks = 0;
        let result = ChunkPlan::build_with_checkpoints("abc", limits(), count, || {
            checks += 1;
            if checks == 3 { Err(MapReduceError::Cancelled) } else { Ok(()) }
        });
        assert!(matches!(result, Err(MapReduceError::Cancelled)));
    }

    #[test]
    fn citations_lift_exactly_and_reject_mid_scalar_or_cross_chunk_offsets() {
        let plan = ChunkPlan::build("éABC😀", ChunkLimits { max_chunk_bytes: 4, ..limits() }, count).unwrap();
        let span = plan.lift_span(0, 0..2).unwrap();
        assert_eq!((span.byte_start, span.byte_end, span.scalar_start, span.scalar_end), (0, 2, 0, 1));
        assert!(matches!(plan.lift_span(0, 1..2), Err(MapReduceError::InvalidSpan)));
        assert!(matches!(plan.lift_span(0, 0..5), Err(MapReduceError::InvalidSpan)));
        assert!(matches!(plan.lift_span(0, 0..0), Err(MapReduceError::InvalidSpan)));
        assert!(matches!(plan.lift_span(99, 0..1), Err(MapReduceError::InvalidSpan)));
        let metadata = serde_json::to_string(&plan.chunks()[0]).unwrap();
        assert!(!metadata.contains("éAB"));
    }
}
