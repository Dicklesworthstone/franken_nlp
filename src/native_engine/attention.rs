//! Eager 48:8 grouped-query attention for `hf-bf16-eager`.

use super::tensor::{Bf16, cast_f32_to_bf16};
use super::{
    kv::{KV_ELEMENTS_PER_POSITION, KvCache, KvCacheError},
    rope::NANBEIGE_HEAD_DIM,
};

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

/// Computes one eager QK score with the pinned BF16 output boundaries.
///
/// The remote eager implementation first materializes the BF16 QK matmul
/// result, then divides that BF16 tensor and materializes the BF16 scaled
/// score before the softmax upcast.  This scalar reference preserves those
/// two observable boundaries; the reduction microarchitecture remains a
/// separately fixture-gated conformance question.
pub fn qk_score_bf16_cast_points(query: &[Bf16], key: &[Bf16]) -> Result<Bf16, AttentionError> {
    if query.len() != NANBEIGE_HEAD_DIM {
        return Err(AttentionError::HeadDimension {
            expected: NANBEIGE_HEAD_DIM,
            actual: query.len(),
        });
    }
    if key.len() != NANBEIGE_HEAD_DIM {
        return Err(AttentionError::HeadDimension {
            expected: NANBEIGE_HEAD_DIM,
            actual: key.len(),
        });
    }
    Ok(qk_score_bf16_cast_points_from_values(
        query.iter().map(|value| value.to_f32()),
        key.iter().map(|value| value.to_f32()),
    ))
}

fn qk_score_bf16_cast_points_from_values(
    query: impl Iterator<Item = f32>,
    key: impl Iterator<Item = f32>,
) -> Bf16 {
    let qk_matmul = Bf16::from_f32(query.zip(key).map(|(left, right)| left * right).sum());
    Bf16::from_f32(qk_matmul.to_f32() * ATTENTION_SCALE)
}

/// Fixed-order streaming softmax state for one value vector width.
///
/// Each observed score rescales the prior maximum, denominator, and weighted
/// value sum before absorbing the next K/V position.  It retains only one
/// 128-wide value accumulator, so decode does not need a score or probability
/// buffer proportional to context length.  This is a distinct float reduction
/// surface: [`eager_gqa_attention_from_cache_prefix`] remains the
/// `hf-bf16-eager` route because that profile has a named bf16 probability
/// cast-back site.
#[derive(Clone, Debug)]
pub struct OnlineSoftmaxF32 {
    maximum: f32,
    normalizer: f32,
    weighted_values: Vec<f32>,
    observations: usize,
}

impl OnlineSoftmaxF32 {
    /// Starts an empty streaming reduction for `value_width` scalar values.
    #[must_use]
    pub fn new(value_width: usize) -> Self {
        Self {
            maximum: f32::NEG_INFINITY,
            normalizer: 0.0,
            weighted_values: vec![0.0; value_width],
            observations: 0,
        }
    }

    /// Absorbs a cache-resident bf16 value vector at one score position.
    pub fn observe_bf16_bits(&mut self, score: f32, values: &[u16]) -> Result<(), AttentionError> {
        if values.len() != self.weighted_values.len() {
            return Err(AttentionError::HeadDimension {
                expected: self.weighted_values.len(),
                actual: values.len(),
            });
        }
        let next_maximum = self.maximum.max(score);
        let existing_scale = if self.observations == 0 {
            0.0
        } else {
            (self.maximum - next_maximum).exp()
        };
        let next_scale = (score - next_maximum).exp();
        self.normalizer = self.normalizer * existing_scale + next_scale;
        for (weighted, value) in self.weighted_values.iter_mut().zip(values) {
            let value = Bf16::from_bits(*value).to_f32();
            *weighted = *weighted * existing_scale + next_scale * value;
        }
        self.maximum = next_maximum;
        self.observations += 1;
        Ok(())
    }

    /// Returns the fixed-order f32 weighted average after at least one input.
    pub fn finish(self) -> Result<Vec<f32>, AttentionError> {
        if self.observations == 0 {
            return Err(AttentionError::EmptySequence);
        }
        Ok(self
            .weighted_values
            .into_iter()
            .map(|value| value / self.normalizer)
            .collect())
    }
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
        .map(|key| qk_score_bf16_cast_points(query, key).map(Bf16::to_f32))
        .collect::<Result<Vec<_>, _>>()?;
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
                    qk_score_bf16_cast_points_from_values(
                        query_head_values.iter().map(|value| value.to_f32()),
                        key[kv_start..kv_end]
                            .iter()
                            .map(|value| Bf16::from_bits(*value).to_f32()),
                    )
                    .to_f32(),
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

/// Computes one bounded-state online-softmax GQA decode row from a KV prefix.
///
/// This function uses no per-context score/probability or repeated-KV buffer.
/// Its f32 streaming reduction is intentionally not wired into the exact
/// `hf-bf16-eager` profile: promotion requires the named float metric contract
/// and two-pass-comparator evidence described by the GQA bead.
pub fn online_gqa_attention_from_cache_prefix(
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
            let mut reduction = OnlineSoftmaxF32::new(NANBEIGE_HEAD_DIM);
            for position in 0..sequence_len {
                let key = cache.key_at(slot, position)?;
                let score = qk_score_bf16_cast_points_from_values(
                    query_head_values.iter().map(|value| value.to_f32()),
                    key[kv_start..kv_end]
                        .iter()
                        .map(|value| Bf16::from_bits(*value).to_f32()),
                )
                .to_f32();
                let value = cache.value_at(slot, position)?;
                reduction.observe_bf16_bits(score, &value[kv_start..kv_end])?;
            }
            output[query_start..query_end].copy_from_slice(&cast_f32_to_bf16(&reduction.finish()?));
        }
    }
    Ok(output)
}

const _: () = assert!(KV_ELEMENTS_PER_POSITION == KV_HEAD_COUNT * NANBEIGE_HEAD_DIM);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_engine::kv::KV_SLOT_COUNT;

    fn bf16_bits(value: f32) -> u16 {
        Bf16::from_f32(value).to_bits()
    }

    fn cache_vector(slot: usize, position: usize, value_scale: f32) -> Vec<u16> {
        (0..KV_ELEMENTS_PER_POSITION)
            .map(|index| {
                let head = index / NANBEIGE_HEAD_DIM;
                let dimension = index % NANBEIGE_HEAD_DIM;
                bf16_bits(
                    value_scale
                        * (slot as f32 + 1.0)
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
        let mut cache = KvCache::try_with_capacity(sequence_len).expect("small cache reserves");
        for slot in 0..KV_SLOT_COUNT {
            for position in 0..sequence_len {
                cache
                    .append(
                        slot,
                        position,
                        &cache_vector(slot, position, 0.000_1),
                        &cache_vector(slot, position, 0.000_01),
                    )
                    .expect("append valid 8x128 KV vectors");
            }
        }
        let query = (0..QUERY_HEAD_COUNT * NANBEIGE_HEAD_DIM)
            .map(|index| bf16_bits((index % NANBEIGE_HEAD_DIM) as f32 * 0.000_1 + 1.0))
            .map(Bf16::from_bits)
            .collect::<Vec<_>>();

        for slot in 0..KV_SLOT_COUNT {
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
                    "slot {slot}, query head {query_head} must read KV head {kv_head} without a repeat buffer"
                );
            }
        }
    }

    #[test]
    fn cache_scanning_gqa_preserves_each_causal_context_edge() {
        let slot = 0;
        let maximum_context = 65;
        let mut cache =
            KvCache::try_with_capacity(maximum_context).expect("edge-matrix cache reserves");
        for position in 0..maximum_context {
            cache
                .append(
                    slot,
                    position,
                    &cache_vector(slot, position, 0.000_1),
                    &cache_vector(slot, position, 0.000_01),
                )
                .expect("append valid 8x128 KV edge-matrix position");
        }
        let query = (0..QUERY_HEAD_COUNT * NANBEIGE_HEAD_DIM)
            .map(|index| bf16_bits((index % NANBEIGE_HEAD_DIM) as f32 * 0.000_1 + 1.0))
            .map(Bf16::from_bits)
            .collect::<Vec<_>>();

        for sequence_len in [1, 2, 15, 16, 17, 63, 64, 65] {
            let observed =
                eager_gqa_attention_from_cache_prefix(&query, &cache, slot, sequence_len)
                    .expect("valid causal cache prefix");
            for query_head in 0..QUERY_HEAD_COUNT {
                let kv_head = kv_head_for_query(query_head).expect("all 48 query heads map");
                let kv_start = kv_head * NANBEIGE_HEAD_DIM;
                let kv_end = kv_start + NANBEIGE_HEAD_DIM;
                let mut keys = Vec::with_capacity(sequence_len * NANBEIGE_HEAD_DIM);
                let mut values = Vec::with_capacity(sequence_len * NANBEIGE_HEAD_DIM);
                for position in 0..sequence_len {
                    keys.extend(
                        cache.key_at(slot, position).expect("resident edge key")[kv_start..kv_end]
                            .iter()
                            .copied()
                            .map(Bf16::from_bits),
                    );
                    values.extend(
                        cache.value_at(slot, position).expect("resident edge value")
                            [kv_start..kv_end]
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
                .expect("gathered causal-prefix reference head");
                assert_eq!(
                    &observed[query_start..query_start + NANBEIGE_HEAD_DIM],
                    expected.as_slice(),
                    "context {sequence_len}, query head {query_head} must exclude future KV positions",
                );
            }
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
    fn online_softmax_uses_bounded_state_for_equal_two_position_mass() {
        let mut reduction = OnlineSoftmaxF32::new(2);
        reduction
            .observe_bf16_bits(0.0, &[bf16_bits(2.0), bf16_bits(6.0)])
            .expect("first 128-wide analogue value is accepted");
        reduction
            .observe_bf16_bits(0.0, &[bf16_bits(4.0), bf16_bits(10.0)])
            .expect("second 128-wide analogue value is accepted");
        assert_eq!(
            reduction.finish().expect("nonempty reduction"),
            vec![3.0, 8.0]
        );
    }

    #[test]
    fn online_softmax_refuses_an_empty_or_wrong_width_reduction() {
        assert_eq!(
            OnlineSoftmaxF32::new(1).finish(),
            Err(AttentionError::EmptySequence)
        );
        let mut reduction = OnlineSoftmaxF32::new(2);
        assert_eq!(
            reduction.observe_bf16_bits(0.0, &[bf16_bits(1.0)]),
            Err(AttentionError::HeadDimension {
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn online_softmax_matches_two_pass_nonfinite_and_signed_zero_behavior() {
        let two_pass_positive_infinity = softmax_f32_cast_back(&[f32::INFINITY, 0.0])
            .expect("nonempty two-pass positive-infinity vector");
        assert!(
            two_pass_positive_infinity
                .iter()
                .all(|value| value.to_f32().is_nan())
        );
        let mut online_positive_infinity = OnlineSoftmaxF32::new(1);
        online_positive_infinity
            .observe_bf16_bits(f32::INFINITY, &[bf16_bits(2.0)])
            .expect("positive-infinity score is representable");
        online_positive_infinity
            .observe_bf16_bits(0.0, &[bf16_bits(4.0)])
            .expect("finite follower is representable");
        assert!(
            online_positive_infinity
                .finish()
                .expect("nonempty online positive-infinity reduction")[0]
                .is_nan()
        );

        let two_pass_negative_infinity = softmax_f32_cast_back(&[f32::NEG_INFINITY])
            .expect("nonempty two-pass negative-infinity vector");
        assert!(two_pass_negative_infinity[0].to_f32().is_nan());
        let mut online_negative_infinity = OnlineSoftmaxF32::new(1);
        online_negative_infinity
            .observe_bf16_bits(f32::NEG_INFINITY, &[bf16_bits(2.0)])
            .expect("negative-infinity score is representable");
        assert!(
            online_negative_infinity
                .finish()
                .expect("nonempty online negative-infinity reduction")[0]
                .is_nan()
        );

        let mut online_negative_zero = OnlineSoftmaxF32::new(1);
        online_negative_zero
            .observe_bf16_bits(-0.0, &[bf16_bits(5.0)])
            .expect("negative-zero score is representable");
        assert_eq!(
            online_negative_zero
                .finish()
                .expect("nonempty online negative-zero reduction"),
            vec![5.0],
        );
    }

    #[test]
    fn online_gqa_matches_the_eager_route_for_exact_equal_score_mass() {
        let slot = 0;
        let mut cache = KvCache::try_with_capacity(2).expect("small cache reserves");
        let keys = vec![0_u16; KV_ELEMENTS_PER_POSITION];
        cache
            .append(
                slot,
                0,
                &keys,
                &vec![bf16_bits(2.0); KV_ELEMENTS_PER_POSITION],
            )
            .expect("first uniform KV position is valid");
        cache
            .append(
                slot,
                1,
                &keys,
                &vec![bf16_bits(4.0); KV_ELEMENTS_PER_POSITION],
            )
            .expect("second uniform KV position is valid");
        let query = vec![Bf16::from_f32(1.0); QUERY_HEAD_COUNT * NANBEIGE_HEAD_DIM];

        let eager = eager_gqa_attention_from_cache(&query, &cache, slot)
            .expect("two-position eager GQA is valid");
        let online = online_gqa_attention_from_cache_prefix(&query, &cache, slot, 2)
            .expect("two-position online GQA is valid");
        assert_eq!(online, eager);
        assert!(online.iter().all(|value| *value == Bf16::from_f32(3.0)));
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
                    &cache_vector(slot, position, 0.000_1),
                    &cache_vector(slot, position, 0.000_01),
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
