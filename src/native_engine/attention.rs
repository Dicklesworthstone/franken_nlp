//! Eager 48:8 grouped-query attention for `hf-bf16-eager`.

use super::{
    kv::{KV_ELEMENTS_PER_POSITION, KvCache, KvCacheError},
    rope::NANBEIGE_HEAD_DIM,
};
use super::tensor::{Bf16, cast_f32_to_bf16};

/// Query heads in Nanbeige4.2-3B.
pub const QUERY_HEAD_COUNT: usize = 48;
/// KV heads in Nanbeige4.2-3B.
pub const KV_HEAD_COUNT: usize = 8;
/// Query heads served by each KV head.
pub const QUERY_HEADS_PER_KV_HEAD: usize = QUERY_HEAD_COUNT / KV_HEAD_COUNT;
/// Eager attention scale, `1 / sqrt(128)`.
pub const ATTENTION_SCALE: f32 = 0.088_388_346;

/// Eager-attention shape or grouping refusal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AttentionError {
    /// The full query projection did not contain all 48 explicit heads.
    QueryVectorLength { expected: usize, actual: usize },
    /// A prefill query buffer was not an integral sequence of 48-head rows.
    QuerySequenceShape { query_width: usize, actual: usize },
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
    /// The fixed-shape forty-four-slot cache refused a read.
    Kv(KvCacheError),
    /// A causal prefix extended beyond the resident K/V positions.
    CausalPrefixUnavailable {
        requested_positions: usize,
        available_positions: usize,
    },
    /// Attention has no defined reduction over an empty sequence.
    EmptySequence,
}

impl From<KvCacheError> for AttentionError {
    fn from(value: KvCacheError) -> Self {
        Self::Kv(value)
    }
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
    for (probability, value) in probabilities
        .iter()
        .zip(values.chunks_exact(NANBEIGE_HEAD_DIM))
    {
        let probability = probability.to_f32();
        for (destination, source) in output.iter_mut().zip(value) {
            *destination += probability * source.to_f32();
        }
    }
    Ok(cast_f32_to_bf16(&output))
}

/// Computes eager 48:8 GQA directly from one logical KV cache slot.
///
/// The traversal is KV-head-major: every resident K/V head services its six
/// query heads before moving to the next K/V head.  It reads 128-wide K and V
/// views directly from [`KvCache`]; unlike a dense-SDPA reference expansion,
/// it never creates a repeated-KV buffer or per-query gathered cache copy.
/// The score, softmax, probability cast-back, and value-reduction order stay
/// identical to [`eager_attention_head`].
pub fn eager_gqa_attention_from_cache(
    query: &[Bf16],
    cache: &KvCache,
    slot: usize,
) -> Result<Vec<Bf16>, AttentionError> {
    let sequence_len = cache.len_for_slot(slot)?;
    eager_gqa_attention_from_cache_prefix(query, cache, slot, sequence_len)
}

/// Computes one causal GQA row against a resident prefix of a logical slot.
///
/// This is the common decode/prefill primitive.  Decode selects the complete
/// resident slot; prefill selects `position + 1` for each query row.  The
/// prefix length is explicit so a prefilled cache cannot accidentally expose
/// future K/V positions to an earlier causal query.
pub fn eager_gqa_attention_from_cache_prefix(
    query: &[Bf16],
    cache: &KvCache,
    slot: usize,
    sequence_len: usize,
) -> Result<Vec<Bf16>, AttentionError> {
    let expected_query_width = QUERY_HEAD_COUNT * NANBEIGE_HEAD_DIM;
    if query.len() != expected_query_width {
        return Err(AttentionError::QueryVectorLength {
            expected: expected_query_width,
            actual: query.len(),
        });
    }
    let available_positions = cache.len_for_slot(slot)?;
    if sequence_len > available_positions {
        return Err(AttentionError::CausalPrefixUnavailable {
            requested_positions: sequence_len,
            available_positions,
        });
    }
    if sequence_len == 0 {
        return Err(AttentionError::EmptySequence);
    }

    let mut output = vec![Bf16::from_bits(0); expected_query_width];
    for kv_head in 0..KV_HEAD_COUNT {
        let kv_start = kv_head * NANBEIGE_HEAD_DIM;
        let kv_end = kv_start + NANBEIGE_HEAD_DIM;
        for query_offset in 0..QUERY_HEADS_PER_KV_HEAD {
            let query_head = kv_head * QUERY_HEADS_PER_KV_HEAD + query_offset;
            let query_start = query_head * NANBEIGE_HEAD_DIM;
            let query_end = query_start + NANBEIGE_HEAD_DIM;
            let query_head_values = &query[query_start..query_end];

            let mut scores = Vec::with_capacity(sequence_len);
            for position in 0..sequence_len {
                let key = cache.key_at(slot, position)?;
                scores.push(
                    query_head_values
                        .iter()
                        .zip(&key[kv_start..kv_end])
                        .map(|(query_value, key_value)| {
                            query_value.to_f32() * Bf16::from_bits(*key_value).to_f32()
                        })
                        .sum::<f32>()
                        * ATTENTION_SCALE,
                );
            }

            let probabilities = softmax_f32_cast_back(&scores)?;
            let destination = &mut output[query_start..query_end];
            let mut accumulated = vec![0.0_f32; NANBEIGE_HEAD_DIM];
            for (position, probability) in probabilities.iter().enumerate() {
                let value = cache.value_at(slot, position)?;
                let probability = probability.to_f32();
                for (destination, source) in accumulated.iter_mut().zip(&value[kv_start..kv_end]) {
                    *destination += probability * Bf16::from_bits(*source).to_f32();
                }
            }
            destination.copy_from_slice(&cast_f32_to_bf16(&accumulated));
        }
    }
    Ok(output)
}

/// Computes native causal GQA for every row in an already-populated prefill.
///
/// `queries` has one contiguous 48 × 128 row per cache position.  Row `n`
/// observes exactly K/V positions `0..=n`, so the last row is the same native
/// operation as decode after that cache position has been appended.  No 6x
/// expanded K/V storage is built at any point.
pub fn eager_gqa_prefill_from_cache(
    queries: &[Bf16],
    cache: &KvCache,
    slot: usize,
) -> Result<Vec<Bf16>, AttentionError> {
    let query_width = QUERY_HEAD_COUNT * NANBEIGE_HEAD_DIM;
    if queries.len() % query_width != 0 {
        return Err(AttentionError::QuerySequenceShape {
            query_width,
            actual: queries.len(),
        });
    }
    let sequence_len = cache.len_for_slot(slot)?;
    if sequence_len == 0 {
        return Err(AttentionError::EmptySequence);
    }
    let query_rows = queries.len() / query_width;
    if query_rows != sequence_len {
        return Err(AttentionError::CausalPrefixUnavailable {
            requested_positions: query_rows,
            available_positions: sequence_len,
        });
    }

    let mut output = Vec::with_capacity(queries.len());
    for (position, query) in queries.chunks_exact(query_width).enumerate() {
        output.extend(eager_gqa_attention_from_cache_prefix(
            query,
            cache,
            slot,
            position + 1,
        )?);
    }
    Ok(output)
}

const _: () = assert!(KV_ELEMENTS_PER_POSITION == KV_HEAD_COUNT * NANBEIGE_HEAD_DIM);

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bits(value: f32) -> u16 {
        Bf16::from_f32(value).to_bits()
    }

    fn cache_vector(position: usize, value_scale: f32) -> Vec<u16> {
        (0..KV_ELEMENTS_PER_POSITION)
            .map(|index| {
                let head = index / NANBEIGE_HEAD_DIM;
                let dimension = index % NANBEIGE_HEAD_DIM;
                bf16_bits(
                    value_scale
                        * (position as f32 + 1.0)
                        * (head as f32 + 1.0)
                        * (dimension as f32 + 1.0),
                )
            })
            .collect()
    }

    #[test]
    fn cache_scanning_gqa_matches_gathered_head_reference_for_all_48_heads() {
        let sequence_len = 3;
        let slot = 43;
        let mut cache = KvCache::try_with_capacity(sequence_len).expect("small cache reserves");
        for position in 0..sequence_len {
            cache
                .append(
                    slot,
                    position,
                    &cache_vector(position, 0.000_1),
                    &cache_vector(position, 0.000_01),
                )
                .expect("append valid 8x128 KV vectors");
        }
        let query = (0..QUERY_HEAD_COUNT * NANBEIGE_HEAD_DIM)
            .map(|index| bf16_bits((index % NANBEIGE_HEAD_DIM) as f32 * 0.000_1 + 1.0))
            .map(Bf16::from_bits)
            .collect::<Vec<_>>();

        let observed = eager_gqa_attention_from_cache(&query, &cache, slot)
            .expect("valid native cache-scanning GQA");
        for query_head in 0..QUERY_HEAD_COUNT {
            let kv_head = kv_head_for_query(query_head).expect("all 48 query heads map");
            let kv_start = kv_head * NANBEIGE_HEAD_DIM;
            let kv_end = kv_start + NANBEIGE_HEAD_DIM;
            let mut keys = Vec::with_capacity(sequence_len * NANBEIGE_HEAD_DIM);
            let mut values = Vec::with_capacity(sequence_len * NANBEIGE_HEAD_DIM);
            for position in 0..sequence_len {
                keys.extend(
                    cache.key_at(slot, position).expect("resident key")[kv_start..kv_end]
                        .iter()
                        .copied()
                        .map(Bf16::from_bits),
                );
                values.extend(
                    cache.value_at(slot, position).expect("resident value")[kv_start..kv_end]
                        .iter()
                        .copied()
                        .map(Bf16::from_bits),
                );
            }
            let query_start = query_head * NANBEIGE_HEAD_DIM;
            let expected = eager_attention_head(
                &query[query_start..query_start + NANBEIGE_HEAD_DIM],
                &keys,
                &values,
                sequence_len,
            )
            .expect("gathered reference head");
            assert_eq!(
                &observed[query_start..query_start + NANBEIGE_HEAD_DIM],
                expected.as_slice(),
                "query head {query_head} must read KV head {kv_head} without a repeat buffer"
            );
        }
    }

    #[test]
    fn cache_scanning_gqa_rejects_a_non_48_head_query() {
        let cache = KvCache::try_with_capacity(1).expect("small cache reserves");
        assert_eq!(
            eager_gqa_attention_from_cache(&[], &cache, 0),
            Err(AttentionError::QueryVectorLength {
                expected: QUERY_HEAD_COUNT * NANBEIGE_HEAD_DIM,
                actual: 0,
            })
        );
    }

    #[test]
    fn prefill_rows_match_the_same_cache_prefix_used_by_decode() {
        let sequence_len = 3;
        let slot = 0;
        let mut cache = KvCache::try_with_capacity(sequence_len).expect("small cache reserves");
        for position in 0..sequence_len {
            cache
                .append(
                    slot,
                    position,
                    &cache_vector(position, 0.000_1),
                    &cache_vector(position, 0.000_01),
                )
                .expect("append valid 8x128 KV vectors");
        }
        let query_width = QUERY_HEAD_COUNT * NANBEIGE_HEAD_DIM;
        let queries = (0..sequence_len * query_width)
            .map(|index| bf16_bits((index % NANBEIGE_HEAD_DIM) as f32 * 0.000_1 + 1.0))
            .map(Bf16::from_bits)
            .collect::<Vec<_>>();

        let prefill = eager_gqa_prefill_from_cache(&queries, &cache, slot)
            .expect("valid causal prefill cache scan");
        for position in 0..sequence_len {
            let start = position * query_width;
            let decode_prefix = eager_gqa_attention_from_cache_prefix(
                &queries[start..start + query_width],
                &cache,
                slot,
                position + 1,
            )
            .expect("valid decode cache prefix");
            assert_eq!(
                &prefill[start..start + query_width],
                decode_prefix.as_slice(),
                "prefill position {position} must use its matching decode prefix"
            );
        }
    }
}
