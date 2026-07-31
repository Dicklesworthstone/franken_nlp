//! Eager 48:8 grouped-query attention for `hf-bf16-eager`.

use super::rope::NANBEIGE_HEAD_DIM;
use super::tensor::{Bf16, cast_f32_to_bf16};

/// Query heads in Nanbeige4.2-3B.
pub const QUERY_HEAD_COUNT: usize = 48;
/// KV heads in Nanbeige4.2-3B.
pub const KV_HEAD_COUNT: usize = 8;
/// Query heads served by each KV head.
pub const QUERY_HEADS_PER_KV_HEAD: usize = QUERY_HEAD_COUNT / KV_HEAD_COUNT;
/// Eager attention scale, `1 / sqrt(128)`.
pub const ATTENTION_SCALE: f32 = 1.0 / (NANBEIGE_HEAD_DIM as f32).sqrt();

/// Eager-attention shape or grouping refusal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AttentionError {
    /// A query head outside 0..48 was requested.
    QueryHeadOutOfRange { query_head: usize },
    /// A head vector did not have the explicit 128-element width.
    HeadDimension { expected: usize, actual: usize },
    /// Key/value cache shape did not match its sequence length.
    SequenceShape {
        sequence_len: usize,
        expected: usize,
        actual: usize,
    },
    /// Attention has no defined reduction over an empty sequence.
    EmptySequence,
}

/// Maps one of 48 query heads onto one of 8 KV heads without materializing repeats.
pub fn kv_head_for_query(query_head: usize) -> Result<usize, AttentionError> {
    if query_head >= QUERY_HEAD_COUNT {
        return Err(AttentionError::QueryHeadOutOfRange { query_head });
    }
    Ok(query_head / QUERY_HEADS_PER_KV_HEAD)
}

/// Computes softmax in f32 and narrows its probability activation to bf16.
pub fn softmax_f32_cast_back(scores: &[f32]) -> Result<Vec<Bf16>, AttentionError> {
    let Some(maximum) = scores.iter().copied().reduce(f32::max) else {
        return Err(AttentionError::EmptySequence);
    };
    let exponentials = scores
        .iter()
        .map(|score| (*score - maximum).exp())
        .collect::<Vec<_>>();
    let denominator = exponentials.iter().sum::<f32>();
    Ok(cast_f32_to_bf16(
        &exponentials
            .iter()
            .map(|value| value / denominator)
            .collect::<Vec<_>>(),
    ))
}

/// Computes one eager causal-attention output head.
///
/// `keys` and `values` contain exactly `sequence_len` contiguous 128-wide
/// vectors from the KV head selected by [`kv_head_for_query`]. The caller's
/// cache/runner supplies only positions at or before the causal position.
pub fn eager_attention_head(
    query: &[Bf16],
    keys: &[Bf16],
    values: &[Bf16],
    sequence_len: usize,
) -> Result<Vec<Bf16>, AttentionError> {
    if query.len() != NANBEIGE_HEAD_DIM {
        return Err(AttentionError::HeadDimension {
            expected: NANBEIGE_HEAD_DIM,
            actual: query.len(),
        });
    }
    if sequence_len == 0 {
        return Err(AttentionError::EmptySequence);
    }
    let expected = sequence_len * NANBEIGE_HEAD_DIM;
    if keys.len() != expected {
        return Err(AttentionError::SequenceShape {
            sequence_len,
            expected,
            actual: keys.len(),
        });
    }
    if values.len() != expected {
        return Err(AttentionError::SequenceShape {
            sequence_len,
            expected,
            actual: values.len(),
        });
    }

    let scores = keys
        .chunks_exact(NANBEIGE_HEAD_DIM)
        .map(|key| {
            query
                .iter()
                .zip(key)
                .map(|(query_value, key_value)| query_value.to_f32() * key_value.to_f32())
                .sum::<f32>()
                * ATTENTION_SCALE
        })
        .collect::<Vec<_>>();
    let probabilities = softmax_f32_cast_back(&scores)?;
    let mut output = vec![0.0_f32; NANBEIGE_HEAD_DIM];
    for (probability, value) in probabilities.iter().zip(values.chunks_exact(NANBEIGE_HEAD_DIM)) {
        let probability = probability.to_f32();
        for (destination, source) in output.iter_mut().zip(value) {
            *destination += probability * source.to_f32();
        }
    }
    Ok(cast_f32_to_bf16(&output))
}
