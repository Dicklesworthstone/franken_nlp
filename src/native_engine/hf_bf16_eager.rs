//! The pinned eager bf16 reference forward for Nanbeige4.2-3B.
//!
//! This is deliberately the semantic, not performance, implementation.  It
//! shares the single 44-binding loop runner with every other profile, keeps
//! activations and K/V in bf16 at the named cast boundaries, and exports only
//! the final untied lm-head logits as f32.

use super::{
    attention::{
        AttentionError, KV_HEAD_COUNT, QUERY_HEAD_COUNT, eager_attention_head, kv_head_for_query,
    },
    kv::{
        KV_ELEMENTS_PER_POSITION, KV_SLOT_COUNT, KvCache, KvCacheError, LOOP_COUNT,
        PHYSICAL_LAYER_COUNT, slot_for,
    },
    layer::{
        HfBf16EagerLayerWeights, HfBf16LayerError, NANBEIGE_HIDDEN_SIZE,
        NANBEIGE_KV_PROJECTION_SIZE, NANBEIGE_Q_PROJECTION_SIZE,
    },
    lmhead::{NANBEIGE_VOCAB_SIZE, export_logits_f32, greedy_argmax},
    looprun::{LayerBinding, LayerExecutor, LoopRunner, PositionContext},
    nn::{RMS_NORM_EPSILON, embedding_row_stays_bf16, rms_norm_f32_reduce_cast_back},
    rope::{NANBEIGE_HEAD_DIM, RopeError, RopeTablesF32},
    tensor::Bf16,
    weights::{Bf16Matrix, WeightShapeError},
};

/// The execution-identity label selected by this exact cast schedule.
pub const HF_BF16_EAGER_PROFILE: &str = "hf-bf16-eager";

/// Shape-checked bf16 model tensors used by the eager semantic oracle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HfBf16EagerWeights {
    /// Vocabulary embedding table. Rows remain bf16 at the embedding boundary.
    pub embeddings: Bf16Matrix,
    /// The 22 physical layers reused in both logical loop passes.
    pub layers: [HfBf16EagerLayerWeights; PHYSICAL_LAYER_COUNT],
    /// The same final RMSNorm applied after each 22-layer pass.
    pub final_norm: Vec<Bf16>,
    /// Untied output projection with f32 logit export.
    pub lm_head: Bf16Matrix,
}

/// One completed eager forward position and its replay-relevant boundaries.
#[derive(Clone, Debug, PartialEq)]
pub struct HfBf16EagerForward {
    /// The common cache/RoPE/causal position used in both loop passes.
    pub position: usize,
    /// One post-MLP hidden state from each of the 44 logical layer executions.
    /// These are the profile-owned L2 taps; the two loop norms remain distinct.
    pub layer_outputs: Vec<HfBf16EagerLayerOutput>,
    /// Post-loop states; index zero is precisely the input to loop two.
    pub post_loop_norms: [Vec<Bf16>; LOOP_COUNT],
    /// Untied lm-head output, exported in f32 without an activation cast-back.
    pub logits: Vec<f32>,
    /// Deterministic first-index-wins greedy selection over `logits`.
    pub greedy_token: usize,
}

/// One L2-visible post-layer hidden state from the shared two-pass schedule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HfBf16EagerLayerOutput {
    /// Zero-based pass index.
    pub loop_index: usize,
    /// Zero-based physical layer index.
    pub layer_index: usize,
    /// Resolved logical K/V slot, always `layer + loop * 22`.
    pub kv_slot: usize,
    /// Post-MLP residual hidden state in the profile's bf16 activation dtype.
    pub hidden: Vec<Bf16>,
}

/// A typed refusal from the eager bf16 profile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HfBf16EagerError {
    /// A top-level model tensor disagreed with the one-model wedge shape.
    ModelMatrixShape {
        name: &'static str,
        rows: usize,
        columns: usize,
        expected_rows: usize,
        expected_columns: usize,
    },
    /// A top-level norm vector disagreed with the hidden width.
    ModelVectorShape {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A physical decoder layer failed its own no-bias shape check.
    Layer(HfBf16LayerError),
    /// A projection or embedding-row lookup failed.
    Weights(WeightShapeError),
    /// A 44-slot K/V append/read invariant failed.
    Kv(KvCacheError),
    /// Eager GQA grouping or attention failed.
    Attention(AttentionError),
    /// Admitted-cap RoPE lookup/application failed.
    Rope(RopeError),
    /// The requested input id is outside the fixed vocabulary.
    TokenOutOfRange { token_id: u32, vocab_size: usize },
    /// Prior decoding left one of the 44 logical K/V slots at another length.
    DivergentCacheLength {
        slot: usize,
        expected_positions: usize,
        actual_positions: usize,
    },
    /// The fixed loop runner delivered an impossible boundary callback order.
    InvalidRunnerBoundary,
    /// The untied lm_head unexpectedly contained no output rows.
    EmptyLogits,
    /// Prefill must contain at least one token position to execute.
    EmptyPrefill,
    /// A cache sequence length could not be expanded to one 128-wide head.
    AttentionCapacityOverflow { sequence_len: usize },
}

impl From<HfBf16LayerError> for HfBf16EagerError {
    fn from(value: HfBf16LayerError) -> Self {
        Self::Layer(value)
    }
}

impl From<WeightShapeError> for HfBf16EagerError {
    fn from(value: WeightShapeError) -> Self {
        Self::Weights(value)
    }
}

impl From<KvCacheError> for HfBf16EagerError {
    fn from(value: KvCacheError) -> Self {
        Self::Kv(value)
    }
}

impl From<AttentionError> for HfBf16EagerError {
    fn from(value: AttentionError) -> Self {
        Self::Attention(value)
    }
}

impl From<RopeError> for HfBf16EagerError {
    fn from(value: RopeError) -> Self {
        Self::Rope(value)
    }
}

impl HfBf16EagerWeights {
    fn validate(&self) -> Result<(), HfBf16EagerError> {
        validate_matrix(
            "embeddings",
            &self.embeddings,
            NANBEIGE_VOCAB_SIZE,
            NANBEIGE_HIDDEN_SIZE,
        )?;
        for layer in &self.layers {
            layer.validate()?;
        }
        if self.final_norm.len() != NANBEIGE_HIDDEN_SIZE {
            return Err(HfBf16EagerError::ModelVectorShape {
                name: "final_norm",
                expected: NANBEIGE_HIDDEN_SIZE,
                actual: self.final_norm.len(),
            });
        }
        validate_matrix(
            "lm_head",
            &self.lm_head,
            NANBEIGE_VOCAB_SIZE,
            NANBEIGE_HIDDEN_SIZE,
        )
    }
}

/// The one-model eager semantic reference engine.
#[derive(Clone, Debug)]
pub struct HfBf16EagerEngine {
    weights: HfBf16EagerWeights,
    kv_cache: KvCache,
    rope: RopeTablesF32,
}

impl HfBf16EagerEngine {
    /// Validates every fixed tensor shape and provisions all bf16 K/V slots
    /// and RoPE rows before entering the token loop.
    pub fn new(
        weights: HfBf16EagerWeights,
        admitted_context_cap: usize,
    ) -> Result<Self, HfBf16EagerError> {
        weights.validate()?;
        Ok(Self {
            kv_cache: KvCache::try_with_capacity(admitted_context_cap)?,
            rope: RopeTablesF32::nanbeige(admitted_context_cap)?,
            weights,
        })
    }

    /// Returns the profile selected by this engine.
    #[must_use]
    pub const fn profile(&self) -> &'static str {
        HF_BF16_EAGER_PROFILE
    }

    /// Exposes the shared cache for slot-depth conformance assertions.
    #[must_use]
    pub const fn kv_cache(&self) -> &KvCache {
        &self.kv_cache
    }

    /// Decodes one token through `22 layers -> norm -> 22 layers -> norm`.
    pub fn decode(&mut self, token_id: u32) -> Result<HfBf16EagerForward, HfBf16EagerError> {
        let position = self.sequence_len()?;
        let token = usize::try_from(token_id).map_err(|_| HfBf16EagerError::TokenOutOfRange {
            token_id,
            vocab_size: NANBEIGE_VOCAB_SIZE,
        })?;
        if token >= NANBEIGE_VOCAB_SIZE {
            return Err(HfBf16EagerError::TokenOutOfRange {
                token_id,
                vocab_size: NANBEIGE_VOCAB_SIZE,
            });
        }

        let mut hidden = embedding_row_stays_bf16(self.weights.embeddings.row(token)?);
        let runner = LoopRunner::from_layer_weights(&self.weights.layers);
        let mut layer_outputs = Vec::with_capacity(KV_SLOT_COUNT);
        let mut post_loop_norms = [Vec::new(), Vec::new()];
        let mut executor = HfBf16EagerExecutor {
            final_norm: &self.weights.final_norm,
            rope: &self.rope,
            completed_layers: 0,
            layer_outputs: &mut layer_outputs,
            post_loop_norms: &mut post_loop_norms,
        };
        runner.run_token(
            &mut executor,
            &mut hidden,
            PositionContext::at(position),
            &mut self.kv_cache,
        )?;
        let logits = export_logits_f32(&hidden, &self.weights.lm_head)?;
        let greedy_token = greedy_argmax(&logits).ok_or(HfBf16EagerError::EmptyLogits)?;
        Ok(HfBf16EagerForward {
            position,
            layer_outputs,
            post_loop_norms,
            logits,
            greedy_token,
        })
    }

    /// Prefills a nonempty prefix in strict token order, retaining all 44 K/V
    /// slots between positions exactly as decode does.
    pub fn prefill(
        &mut self,
        token_ids: &[u32],
    ) -> Result<Vec<HfBf16EagerForward>, HfBf16EagerError> {
        if token_ids.is_empty() {
            return Err(HfBf16EagerError::EmptyPrefill);
        }
        token_ids
            .iter()
            .map(|&token_id| self.decode(token_id))
            .collect()
    }

    fn sequence_len(&self) -> Result<usize, HfBf16EagerError> {
        let expected_positions = self.kv_cache.len_for_slot(0)?;
        for slot in 1..KV_SLOT_COUNT {
            let actual_positions = self.kv_cache.len_for_slot(slot)?;
            if actual_positions != expected_positions {
                return Err(HfBf16EagerError::DivergentCacheLength {
                    slot,
                    expected_positions,
                    actual_positions,
                });
            }
        }
        Ok(expected_positions)
    }
}

struct HfBf16EagerExecutor<'a> {
    final_norm: &'a [Bf16],
    rope: &'a RopeTablesF32,
    completed_layers: usize,
    layer_outputs: &'a mut Vec<HfBf16EagerLayerOutput>,
    post_loop_norms: &'a mut [Vec<Bf16>; LOOP_COUNT],
}

impl LayerExecutor<HfBf16EagerLayerWeights> for HfBf16EagerExecutor<'_> {
    type Hidden = Vec<Bf16>;
    type Error = HfBf16EagerError;

    fn layer_forward(
        &mut self,
        binding: &LayerBinding<'_, HfBf16EagerLayerWeights>,
        hidden: &mut Self::Hidden,
        positions: PositionContext,
        kv_cache: &mut KvCache,
    ) -> Result<(), Self::Error> {
        *hidden = run_hf_bf16_layer(
            binding.weights(),
            binding.loop_index(),
            binding.layer_index(),
            binding.kv_slot(),
            hidden,
            positions,
            self.rope,
            kv_cache,
        )?;
        self.layer_outputs.push(HfBf16EagerLayerOutput {
            loop_index: binding.loop_index(),
            layer_index: binding.layer_index(),
            kv_slot: binding.kv_slot(),
            hidden: hidden.clone(),
        });
        self.completed_layers += 1;
        Ok(())
    }

    fn final_rms_norm(
        &mut self,
        hidden: &mut Self::Hidden,
        _positions: PositionContext,
    ) -> Result<(), Self::Error> {
        let loop_index = self
            .completed_layers
            .checked_div(PHYSICAL_LAYER_COUNT)
            .and_then(|completed| completed.checked_sub(1))
            .filter(|&loop_index| loop_index < LOOP_COUNT)
            .ok_or(HfBf16EagerError::InvalidRunnerBoundary)?;
        *hidden = rms_norm_f32_reduce_cast_back(hidden, self.final_norm, RMS_NORM_EPSILON)
            .map_err(HfBf16LayerError::from)
            .map_err(HfBf16EagerError::from)?;
        self.post_loop_norms[loop_index] = hidden.clone();
        Ok(())
    }
}

fn run_hf_bf16_layer(
    layer: &HfBf16EagerLayerWeights,
    loop_index: usize,
    layer_index: usize,
    kv_slot: usize,
    hidden: &[Bf16],
    positions: PositionContext,
    rope: &RopeTablesF32,
    cache: &mut KvCache,
) -> Result<Vec<Bf16>, HfBf16EagerError> {
    if slot_for(loop_index, layer_index) != Some(kv_slot) {
        return Err(HfBf16EagerError::InvalidRunnerBoundary);
    }
    let attention_norm = layer.input_rms_norm(hidden)?;
    let mut query = layer
        .q_proj
        .project_f32_accumulate_cast_back(&attention_norm)?;
    let mut key = layer
        .k_proj
        .project_f32_accumulate_cast_back(&attention_norm)?;
    let value = layer
        .v_proj
        .project_f32_accumulate_cast_back(&attention_norm)?;
    for head in query.chunks_exact_mut(NANBEIGE_HEAD_DIM) {
        rope.apply_split_half(positions.rope_position, head)?;
    }
    for head in key.chunks_exact_mut(NANBEIGE_HEAD_DIM) {
        rope.apply_split_half(positions.rope_position, head)?;
    }
    cache.append(
        kv_slot,
        positions.cache_position,
        &key.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        &value
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
    )?;
    let attention = eager_gqa_attention(&query, cache, kv_slot)?;
    let attention_output = layer.o_proj.project_f32_accumulate_cast_back(&attention)?;
    Ok(layer.finish_attention_and_mlp(hidden, &attention_output)?)
}

fn eager_gqa_attention(
    query: &[Bf16],
    cache: &KvCache,
    slot: usize,
) -> Result<Vec<Bf16>, HfBf16EagerError> {
    if query.len() != NANBEIGE_Q_PROJECTION_SIZE {
        return Err(HfBf16EagerError::ModelVectorShape {
            name: "attention_query",
            expected: NANBEIGE_Q_PROJECTION_SIZE,
            actual: query.len(),
        });
    }
    let sequence_len = cache.len_for_slot(slot)?;
    let head_elements = sequence_len
        .checked_mul(NANBEIGE_HEAD_DIM)
        .ok_or(HfBf16EagerError::AttentionCapacityOverflow { sequence_len })?;
    let mut output = vec![Bf16::from_bits(0); NANBEIGE_Q_PROJECTION_SIZE];
    for query_head in 0..QUERY_HEAD_COUNT {
        let kv_head = kv_head_for_query(query_head)?;
        let query_start = query_head * NANBEIGE_HEAD_DIM;
        let key_start = kv_head * NANBEIGE_HEAD_DIM;
        let mut keys = Vec::with_capacity(head_elements);
        let mut values = Vec::with_capacity(head_elements);
        for position in 0..sequence_len {
            let cached_key = cache.key_at(slot, position)?;
            let cached_value = cache.value_at(slot, position)?;
            keys.extend(
                cached_key[key_start..key_start + NANBEIGE_HEAD_DIM]
                    .iter()
                    .copied()
                    .map(Bf16::from_bits),
            );
            values.extend(
                cached_value[key_start..key_start + NANBEIGE_HEAD_DIM]
                    .iter()
                    .copied()
                    .map(Bf16::from_bits),
            );
        }
        let head = eager_attention_head(
            &query[query_start..query_start + NANBEIGE_HEAD_DIM],
            &keys,
            &values,
            sequence_len,
        )?;
        output[query_start..query_start + NANBEIGE_HEAD_DIM].copy_from_slice(&head);
    }
    Ok(output)
}

fn validate_matrix(
    name: &'static str,
    matrix: &Bf16Matrix,
    expected_rows: usize,
    expected_columns: usize,
) -> Result<(), HfBf16EagerError> {
    if matrix.rows() != expected_rows || matrix.columns() != expected_columns {
        return Err(HfBf16EagerError::ModelMatrixShape {
            name,
            rows: matrix.rows(),
            columns: matrix.columns(),
            expected_rows,
            expected_columns,
        });
    }
    Ok(())
}

const _: () = assert!(NANBEIGE_KV_PROJECTION_SIZE == KV_ELEMENTS_PER_POSITION);
const _: () = assert!(QUERY_HEAD_COUNT / KV_HEAD_COUNT == 6);
