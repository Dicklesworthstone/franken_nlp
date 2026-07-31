//! Deliberately simple scalar-f32 model specification for parity tests.
//!
//! This is test-only code.  It is not exported from the library, cannot be a
//! production fallback, uses no packed layout, and shares no implementation
//! with the optimized engine.  HF remains the semantic authority; agreement
//! here localizes a discrepancy but is not an oracle-parity claim by itself.

use std::error::Error;
use std::fmt;

pub const PHYSICAL_LAYERS: usize = 22;
pub const LOGICAL_LOOPS: usize = 2;
pub const LOGICAL_KV_SLOTS: usize = PHYSICAL_LAYERS * LOGICAL_LOOPS;
pub const NANBEIGE_HIDDEN: usize = 3072;
pub const NANBEIGE_QUERY_HEADS: usize = 48;
pub const NANBEIGE_KV_HEADS: usize = 8;
pub const NANBEIGE_HEAD_DIM: usize = 128;
pub const NANBEIGE_INTERMEDIATE: usize = 10_752;
pub const NANBEIGE_VOCAB: usize = 166_144;
pub const NANBEIGE_RMS_EPSILON: f32 = 1.0e-5;
pub const NANBEIGE_ROPE_THETA: f32 = 70_000_000.0;

/// Test-only model dimensions.  `tiny_for_tests` keeps the 22-by-2 schedule
/// while reducing vector widths; `nanbeige` spells out the pinned real shape.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecConfig {
    pub hidden: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub rms_epsilon: f32,
    pub rope_theta: f32,
}

impl SpecConfig {
    pub fn nanbeige() -> Self {
        Self {
            hidden: NANBEIGE_HIDDEN,
            query_heads: NANBEIGE_QUERY_HEADS,
            kv_heads: NANBEIGE_KV_HEADS,
            head_dim: NANBEIGE_HEAD_DIM,
            intermediate: NANBEIGE_INTERMEDIATE,
            vocab: NANBEIGE_VOCAB,
            rms_epsilon: NANBEIGE_RMS_EPSILON,
            rope_theta: NANBEIGE_ROPE_THETA,
        }
    }

    pub fn tiny_for_tests() -> Self {
        Self {
            hidden: 4,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            intermediate: 6,
            vocab: 7,
            rms_epsilon: NANBEIGE_RMS_EPSILON,
            rope_theta: NANBEIGE_ROPE_THETA,
        }
    }

    pub fn query_width(&self) -> usize {
        self.query_heads * self.head_dim
    }

    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    pub fn kv_repeat(&self) -> usize {
        self.query_heads / self.kv_heads
    }

    pub fn validate(&self) -> Result<(), SpecError> {
        if self.hidden == 0
            || self.query_heads == 0
            || self.kv_heads == 0
            || self.head_dim == 0
            || self.intermediate == 0
            || self.vocab == 0
        {
            return Err(SpecError::InvalidConfig(
                "all logical dimensions must be non-zero",
            ));
        }
        if self.query_heads % self.kv_heads != 0 {
            return Err(SpecError::InvalidConfig(
                "query_heads must be an exact multiple of kv_heads for dense GQA expansion",
            ));
        }
        if self.head_dim % 2 != 0 {
            return Err(SpecError::InvalidConfig(
                "split-half RoPE requires an even head_dim",
            ));
        }
        if !self.rms_epsilon.is_finite() || self.rms_epsilon <= 0.0 {
            return Err(SpecError::InvalidConfig(
                "rms_epsilon must be finite and positive",
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 1.0 {
            return Err(SpecError::InvalidConfig(
                "rope_theta must be finite and greater than one",
            ));
        }
        Ok(())
    }
}

/// A transparent row-major matrix.  `data[row * cols + col]` is the only
/// layout rule; all matrix-vector products use scalar index loops.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, SpecError> {
        let expected = rows.checked_mul(cols).ok_or(SpecError::SizeOverflow)?;
        if data.len() != expected {
            return Err(SpecError::TensorDataLength {
                rows,
                cols,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { rows, cols, data })
    }

    pub fn zeros(rows: usize, cols: usize) -> Result<Self, SpecError> {
        let len = rows.checked_mul(cols).ok_or(SpecError::SizeOverflow)?;
        Self::new(rows, cols, vec![0.0; len])
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn get(&self, row: usize, column: usize) -> Result<f32, SpecError> {
        if row >= self.rows || column >= self.cols {
            return Err(SpecError::TensorIndex {
                row,
                column,
                rows: self.rows,
                cols: self.cols,
            });
        }
        Ok(self.data[row * self.cols + column])
    }

    pub fn set(&mut self, row: usize, column: usize, value: f32) -> Result<(), SpecError> {
        if row >= self.rows || column >= self.cols {
            return Err(SpecError::TensorIndex {
                row,
                column,
                rows: self.rows,
                cols: self.cols,
            });
        }
        self.data[row * self.cols + column] = value;
        Ok(())
    }

    fn row(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    fn matvec(&self, input: &[f32], name: &'static str) -> Result<Vec<f32>, SpecError> {
        if input.len() != self.cols {
            return Err(SpecError::VectorLength {
                op: name,
                expected: self.cols,
                actual: input.len(),
            });
        }
        let mut output = vec![0.0; self.rows];
        for row in 0..self.rows {
            output[row] = stable_dot(self.row(row), input, name)?;
        }
        Ok(output)
    }
}

/// All linear maps for one physical decoder layer.  Weights are row-major
/// `[out, in]`; no bias exists in the pinned model.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerWeights {
    pub input_norm: Vec<f32>,
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    pub post_attention_norm: Vec<f32>,
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
}

/// Logical f32 weights for the full decoder and untied output projection.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecWeights {
    pub embeddings: Tensor,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: Tensor,
}

impl SpecWeights {
    /// Convenience only for tiny, hand-authored test fixtures.  Real widened
    /// weights should be supplied explicitly through the public structs.
    pub fn zeroed(config: &SpecConfig) -> Result<Self, SpecError> {
        config.validate()?;
        let mut layers = Vec::with_capacity(PHYSICAL_LAYERS);
        for _ in 0..PHYSICAL_LAYERS {
            layers.push(LayerWeights {
                input_norm: vec![1.0; config.hidden],
                q_proj: Tensor::zeros(config.query_width(), config.hidden)?,
                k_proj: Tensor::zeros(config.kv_width(), config.hidden)?,
                v_proj: Tensor::zeros(config.kv_width(), config.hidden)?,
                o_proj: Tensor::zeros(config.hidden, config.query_width())?,
                post_attention_norm: vec![1.0; config.hidden],
                gate_proj: Tensor::zeros(config.intermediate, config.hidden)?,
                up_proj: Tensor::zeros(config.intermediate, config.hidden)?,
                down_proj: Tensor::zeros(config.hidden, config.intermediate)?,
            });
        }
        Ok(Self {
            embeddings: Tensor::zeros(config.vocab, config.hidden)?,
            layers,
            final_norm: vec![1.0; config.hidden],
            lm_head: Tensor::zeros(config.vocab, config.hidden)?,
        })
    }

    fn validate(&self, config: &SpecConfig) -> Result<(), SpecError> {
        require_shape("embeddings", &self.embeddings, config.vocab, config.hidden)?;
        require_length("final_norm", &self.final_norm, config.hidden)?;
        require_shape("lm_head", &self.lm_head, config.vocab, config.hidden)?;
        if self.layers.len() != PHYSICAL_LAYERS {
            return Err(SpecError::LayerCount {
                expected: PHYSICAL_LAYERS,
                actual: self.layers.len(),
            });
        }
        for (layer_index, layer) in self.layers.iter().enumerate() {
            require_length_named("input_norm", layer_index, &layer.input_norm, config.hidden)?;
            require_shape_named(
                "q_proj",
                layer_index,
                &layer.q_proj,
                config.query_width(),
                config.hidden,
            )?;
            require_shape_named(
                "k_proj",
                layer_index,
                &layer.k_proj,
                config.kv_width(),
                config.hidden,
            )?;
            require_shape_named(
                "v_proj",
                layer_index,
                &layer.v_proj,
                config.kv_width(),
                config.hidden,
            )?;
            require_shape_named(
                "o_proj",
                layer_index,
                &layer.o_proj,
                config.hidden,
                config.query_width(),
            )?;
            require_length_named(
                "post_attention_norm",
                layer_index,
                &layer.post_attention_norm,
                config.hidden,
            )?;
            require_shape_named(
                "gate_proj",
                layer_index,
                &layer.gate_proj,
                config.intermediate,
                config.hidden,
            )?;
            require_shape_named(
                "up_proj",
                layer_index,
                &layer.up_proj,
                config.intermediate,
                config.hidden,
            )?;
            require_shape_named(
                "down_proj",
                layer_index,
                &layer.down_proj,
                config.hidden,
                config.intermediate,
            )?;
        }
        Ok(())
    }
}

/// KV data for one position in one logical decoder slot.
#[derive(Clone, Debug, PartialEq)]
pub struct KvPosition {
    pub position: usize,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
struct KvSlot {
    positions: Vec<KvPosition>,
}

/// A deliberately plain 44-slot cache.  Slot computation is visible in
/// [`kv_slot`]: `layer + loop * 22`.
#[derive(Clone, Debug, PartialEq)]
pub struct KvCache {
    slots: Vec<KvSlot>,
    kv_width: usize,
}

impl KvCache {
    pub fn new(config: &SpecConfig) -> Self {
        Self {
            slots: vec![KvSlot { positions: vec![] }; LOGICAL_KV_SLOTS],
            kv_width: config.kv_width(),
        }
    }

    pub fn slot_len(&self, slot: usize) -> Result<usize, SpecError> {
        self.slots
            .get(slot)
            .map(|slot| slot.positions.len())
            .ok_or(SpecError::KvSlotOutOfRange(slot))
    }

    pub fn sequence_len(&self) -> Result<usize, SpecError> {
        let Some(first) = self.slots.first() else {
            return Err(SpecError::CacheInvariant("cache has no slots"));
        };
        let expected = first.positions.len();
        for (slot, entry) in self.slots.iter().enumerate() {
            if entry.positions.len() != expected {
                return Err(SpecError::CacheSlotLengthMismatch {
                    slot,
                    expected,
                    actual: entry.positions.len(),
                });
            }
        }
        Ok(expected)
    }

    fn push(
        &mut self,
        slot: usize,
        position: usize,
        key: Vec<f32>,
        value: Vec<f32>,
    ) -> Result<(), SpecError> {
        if slot >= LOGICAL_KV_SLOTS {
            return Err(SpecError::KvSlotOutOfRange(slot));
        }
        if key.len() != self.kv_width || value.len() != self.kv_width {
            return Err(SpecError::KvWidth {
                expected: self.kv_width,
                key_actual: key.len(),
                value_actual: value.len(),
            });
        }
        let positions = &mut self.slots[slot].positions;
        if positions.len() != position {
            return Err(SpecError::CachePosition {
                slot,
                expected: positions.len(),
                actual: position,
            });
        }
        positions.push(KvPosition {
            position,
            key,
            value,
        });
        Ok(())
    }

    fn positions(&self, slot: usize) -> Result<&[KvPosition], SpecError> {
        self.slots
            .get(slot)
            .map(|slot| slot.positions.as_slice())
            .ok_or(SpecError::KvSlotOutOfRange(slot))
    }
}

/// Return the logical KV slot for one physical layer execution.
pub const fn kv_slot(loop_index: usize, layer_index: usize) -> usize {
    layer_index + loop_index * PHYSICAL_LAYERS
}

/// Per-layer bisect taps.  Each field is logical f32 data in the scalar order.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerTap {
    pub loop_index: usize,
    pub layer_index: usize,
    pub input: Vec<f32>,
    pub attention_norm: Vec<f32>,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub attention_output: Vec<f32>,
    pub post_attention_residual: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub swiglu: Vec<f32>,
    pub output: Vec<f32>,
}

/// Taps for one forward position: all 44 layer outputs, two post-loop norms,
/// final logits, and the deterministic lowest-id greedy choice.
#[derive(Clone, Debug, PartialEq)]
pub struct ForwardTaps {
    pub layer_taps: Vec<LayerTap>,
    pub post_loop_norms: [Vec<f32>; LOGICAL_LOOPS],
    pub logits: Vec<f32>,
    pub greedy_token: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForwardPosition {
    pub position: usize,
    pub taps: ForwardTaps,
}

/// The test-only reference engine.  It only owns logical tensors and cache
/// inputs supplied by tests; it has no dependency on a product kernel path.
#[derive(Clone, Debug)]
pub struct SpecEngine {
    config: SpecConfig,
    weights: SpecWeights,
}

impl SpecEngine {
    pub fn new(config: SpecConfig, weights: SpecWeights) -> Result<Self, SpecError> {
        config.validate()?;
        weights.validate(&config)?;
        Ok(Self { config, weights })
    }

    pub fn config(&self) -> &SpecConfig {
        &self.config
    }

    /// Run a prefill in token order.  Every input token emits a complete tap
    /// bundle; cache positions become visible only through the explicit slots.
    pub fn prefill(
        &self,
        token_ids: &[u32],
        cache: &mut KvCache,
    ) -> Result<Vec<ForwardPosition>, SpecError> {
        if token_ids.is_empty() {
            return Err(SpecError::EmptyPrefill);
        }
        let mut outputs = Vec::with_capacity(token_ids.len());
        for &token_id in token_ids {
            outputs.push(self.decode(token_id, cache)?);
        }
        Ok(outputs)
    }

    /// Run exactly one decode position against the supplied explicit KV cache.
    pub fn decode(&self, token_id: u32, cache: &mut KvCache) -> Result<ForwardPosition, SpecError> {
        let position = cache.sequence_len()?;
        let token = usize::try_from(token_id).map_err(|_| SpecError::TokenOutOfRange {
            token_id,
            vocab: self.config.vocab,
        })?;
        if token >= self.config.vocab {
            return Err(SpecError::TokenOutOfRange {
                token_id,
                vocab: self.config.vocab,
            });
        }
        let mut hidden = self.weights.embeddings.row(token).to_vec();
        let mut layer_taps = Vec::with_capacity(LOGICAL_KV_SLOTS);
        let mut post_loop_norms = [Vec::new(), Vec::new()];

        // The schedule is intentionally literal and visibly not fused.
        for loop_index in 0..2 {
            for layer_index in 0..22 {
                let (next_hidden, tap) =
                    self.run_layer(&hidden, loop_index, layer_index, position, cache)?;
                hidden = next_hidden;
                layer_taps.push(tap);
            }
            hidden = rms_norm(
                &hidden,
                &self.weights.final_norm,
                self.config.rms_epsilon,
                "post_loop_rmsnorm",
            )?;
            post_loop_norms[loop_index] = hidden.clone();
        }

        let logits = self.weights.lm_head.matvec(&hidden, "lm_head")?;
        let greedy_token = greedy_argmax(&logits).ok_or(SpecError::EmptyLogits)?;
        Ok(ForwardPosition {
            position,
            taps: ForwardTaps {
                layer_taps,
                post_loop_norms,
                logits,
                greedy_token,
            },
        })
    }

    fn run_layer(
        &self,
        hidden: &[f32],
        loop_index: usize,
        layer_index: usize,
        position: usize,
        cache: &mut KvCache,
    ) -> Result<(Vec<f32>, LayerTap), SpecError> {
        let layer = &self.weights.layers[layer_index];
        let input = hidden.to_vec();
        let attention_norm = rms_norm(
            hidden,
            &layer.input_norm,
            self.config.rms_epsilon,
            "attention_rmsnorm",
        )?;
        let mut query = layer.q_proj.matvec(&attention_norm, "q_proj")?;
        let mut key = layer.k_proj.matvec(&attention_norm, "k_proj")?;
        let value = layer.v_proj.matvec(&attention_norm, "v_proj")?;
        apply_rope_split_half(
            &mut query,
            self.config.query_heads,
            self.config.head_dim,
            position,
            self.config.rope_theta,
        )?;
        apply_rope_split_half(
            &mut key,
            self.config.kv_heads,
            self.config.head_dim,
            position,
            self.config.rope_theta,
        )?;

        let slot = kv_slot(loop_index, layer_index);
        cache.push(slot, position, key.clone(), value.clone())?;
        let positions = cache.positions(slot)?;
        let mut keys = Vec::with_capacity(positions.len());
        let mut values = Vec::with_capacity(positions.len());
        let mut key_positions = Vec::with_capacity(positions.len());
        for entry in positions {
            keys.push(entry.key.clone());
            values.push(entry.value.clone());
            key_positions.push(entry.position);
        }
        let dense_attention = dense_gqa_attention(
            &self.config,
            &query,
            &keys,
            &values,
            &key_positions,
            position,
        )?;
        let attention_output = layer.o_proj.matvec(&dense_attention.output, "o_proj")?;
        let post_attention_residual = add(&input, &attention_output, "attention_residual")?;
        let ffn_norm = rms_norm(
            &post_attention_residual,
            &layer.post_attention_norm,
            self.config.rms_epsilon,
            "ffn_rmsnorm",
        )?;
        let gate = layer.gate_proj.matvec(&ffn_norm, "gate_proj")?;
        let up = layer.up_proj.matvec(&ffn_norm, "up_proj")?;
        let mut swiglu = vec![0.0; self.config.intermediate];
        for index in 0..self.config.intermediate {
            swiglu[index] = silu(gate[index]) * up[index];
        }
        let down = layer.down_proj.matvec(&swiglu, "down_proj")?;
        let output = add(&post_attention_residual, &down, "ffn_residual")?;

        Ok((
            output.clone(),
            LayerTap {
                loop_index,
                layer_index,
                input,
                attention_norm,
                query,
                key,
                value,
                attention_output,
                post_attention_residual,
                ffn_norm,
                gate,
                up,
                swiglu,
                output,
            },
        ))
    }
}

/// A dense-GQA result.  `probabilities[q_head][key_position]` is represented
/// in row-major flattened form for direct, stable scalar inspection.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseAttention {
    pub output: Vec<f32>,
    pub probabilities: Vec<f32>,
}

/// Expand each of the K/V heads to its six (or config-selected) query heads,
/// then compute dense scaled dot-product attention in fixed loop order.
pub fn dense_gqa_attention(
    config: &SpecConfig,
    query: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    key_positions: &[usize],
    query_position: usize,
) -> Result<DenseAttention, SpecError> {
    if query.len() != config.query_width() {
        return Err(SpecError::VectorLength {
            op: "attention_query",
            expected: config.query_width(),
            actual: query.len(),
        });
    }
    if keys.len() != values.len() || keys.len() != key_positions.len() {
        return Err(SpecError::AttentionSequenceLength {
            keys: keys.len(),
            values: values.len(),
            positions: key_positions.len(),
        });
    }
    if keys.is_empty() {
        return Err(SpecError::EmptyAttention);
    }
    for (index, key) in keys.iter().enumerate() {
        if key.len() != config.kv_width() {
            return Err(SpecError::VectorLength {
                op: "attention_key",
                expected: config.kv_width(),
                actual: key.len(),
            });
        }
        if values[index].len() != config.kv_width() {
            return Err(SpecError::VectorLength {
                op: "attention_value",
                expected: config.kv_width(),
                actual: values[index].len(),
            });
        }
    }

    let sequence = keys.len();
    let mut expanded_keys = vec![0.0; sequence * config.query_width()];
    let mut expanded_values = vec![0.0; sequence * config.query_width()];
    for token_index in 0..sequence {
        for query_head in 0..config.query_heads {
            // Explicit repetition, not a grouped-head shortcut.
            let kv_head = query_head / config.kv_repeat();
            for dimension in 0..config.head_dim {
                let expanded_index =
                    token_index * config.query_width() + query_head * config.head_dim + dimension;
                let kv_index = kv_head * config.head_dim + dimension;
                expanded_keys[expanded_index] = keys[token_index][kv_index];
                expanded_values[expanded_index] = values[token_index][kv_index];
            }
        }
    }

    let scale = 1.0 / (config.head_dim as f32).sqrt();
    let mut probabilities = vec![0.0; config.query_heads * sequence];
    for query_head in 0..config.query_heads {
        let query_start = query_head * config.head_dim;
        let mut scores = vec![0.0; sequence];
        let mut max_score = f32::NEG_INFINITY;
        for token_index in 0..sequence {
            let key_start = token_index * config.query_width() + query_start;
            let raw_score = stable_dot(
                &query[query_start..query_start + config.head_dim],
                &expanded_keys[key_start..key_start + config.head_dim],
                "attention_qk_dot",
            )?;
            // The mask is explicit even though an autoregressive cache has no
            // future entries in ordinary use.
            let mask = if key_positions[token_index] > query_position {
                f32::NEG_INFINITY
            } else {
                0.0
            };
            scores[token_index] = raw_score * scale + mask;
            if scores[token_index] > max_score {
                max_score = scores[token_index];
            }
        }
        let mut normalizer = 0.0;
        for token_index in 0..sequence {
            let probability = (scores[token_index] - max_score).exp();
            probabilities[query_head * sequence + token_index] = probability;
            normalizer += probability;
        }
        for token_index in 0..sequence {
            probabilities[query_head * sequence + token_index] /= normalizer;
        }
    }

    let mut output = vec![0.0; config.query_width()];
    for query_head in 0..config.query_heads {
        for dimension in 0..config.head_dim {
            let mut sum = 0.0;
            for token_index in 0..sequence {
                let probability = probabilities[query_head * sequence + token_index];
                let value_index =
                    token_index * config.query_width() + query_head * config.head_dim + dimension;
                sum += probability * expanded_values[value_index];
            }
            output[query_head * config.head_dim + dimension] = sum;
        }
    }
    Ok(DenseAttention {
        output,
        probabilities,
    })
}

/// Scalar sequential dot product.  The `for` loop is the canonical reduction
/// order cited by downstream differential harnesses.
pub fn stable_dot(left: &[f32], right: &[f32], op: &'static str) -> Result<f32, SpecError> {
    if left.len() != right.len() {
        return Err(SpecError::PairLength {
            op,
            left: left.len(),
            right: right.len(),
        });
    }
    let mut sum = 0.0;
    for index in 0..left.len() {
        sum += left[index] * right[index];
    }
    Ok(sum)
}

/// Scalar RMSNorm with the pinned epsilon.  Sum-of-squares is sequential.
pub fn rms_norm(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    op: &'static str,
) -> Result<Vec<f32>, SpecError> {
    if input.len() != weight.len() {
        return Err(SpecError::PairLength {
            op,
            left: input.len(),
            right: weight.len(),
        });
    }
    if input.is_empty() {
        return Err(SpecError::EmptyVector(op));
    }
    let mut sum_of_squares = 0.0;
    for value in input {
        sum_of_squares += value * value;
    }
    let inverse_rms = 1.0 / (sum_of_squares / input.len() as f32 + epsilon).sqrt();
    let mut output = vec![0.0; input.len()];
    for index in 0..input.len() {
        output[index] = input[index] * inverse_rms * weight[index];
    }
    Ok(output)
}

/// In-place split-half RoPE for one Q or K projection.  Pair `i` with
/// `i + head_dim / 2` and use theta^(-i/(head_dim/2)).
pub fn apply_rope_split_half(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
) -> Result<(), SpecError> {
    if head_dim == 0 || head_dim % 2 != 0 || values.len() != heads * head_dim {
        return Err(SpecError::RopeShape {
            values: values.len(),
            heads,
            head_dim,
        });
    }
    let half = head_dim / 2;
    for head in 0..heads {
        let base = head * head_dim;
        for dimension in 0..half {
            let frequency = theta.powf(-(dimension as f32) / half as f32);
            let angle = position as f32 * frequency;
            let cosine = angle.cos();
            let sine = angle.sin();
            let first = values[base + dimension];
            let second = values[base + half + dimension];
            values[base + dimension] = first * cosine - second * sine;
            values[base + half + dimension] = first * sine + second * cosine;
        }
    }
    Ok(())
}

pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// Deterministic greedy selection: strict `>` leaves the first (lowest token
/// id) as the winner when logits tie.
pub fn greedy_argmax(logits: &[f32]) -> Option<usize> {
    let (&first, rest) = logits.split_first()?;
    let mut best_index = 0;
    let mut best_value = first;
    for (offset, value) in rest.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best_index = offset + 1;
        }
    }
    Some(best_index)
}

/// A readable per-coordinate discrepancy suitable for L1/L2 test logs.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarDiff {
    pub op: &'static str,
    pub loop_index: usize,
    pub layer_index: usize,
    pub index: usize,
    pub expected: f32,
    pub got: f32,
    pub absolute_delta: f32,
    pub relative_delta: f32,
    pub ulp_delta: u32,
}

impl ScalarDiff {
    pub fn between(
        op: &'static str,
        loop_index: usize,
        layer_index: usize,
        index: usize,
        expected: f32,
        got: f32,
    ) -> Self {
        let absolute_delta = (expected - got).abs();
        let relative_delta = absolute_delta / expected.abs().max(got.abs()).max(f32::MIN_POSITIVE);
        Self {
            op,
            loop_index,
            layer_index,
            index,
            expected,
            got,
            absolute_delta,
            relative_delta,
            ulp_delta: ulp_distance(expected, got),
        }
    }
}

impl fmt::Display for ScalarDiff {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "op={} tap=({}, {}) index={} expected={} got={} abs={} rel={} ulp={}",
            self.op,
            self.loop_index,
            self.layer_index,
            self.index,
            self.expected,
            self.got,
            self.absolute_delta,
            self.relative_delta,
            self.ulp_delta
        )
    }
}

pub fn ulp_distance(left: f32, right: f32) -> u32 {
    if left == right {
        return 0;
    }
    let left = ordered_float_bits(left);
    let right = ordered_float_bits(right);
    left.abs_diff(right)
}

fn ordered_float_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

fn add(left: &[f32], right: &[f32], op: &'static str) -> Result<Vec<f32>, SpecError> {
    if left.len() != right.len() {
        return Err(SpecError::PairLength {
            op,
            left: left.len(),
            right: right.len(),
        });
    }
    let mut output = vec![0.0; left.len()];
    for index in 0..left.len() {
        output[index] = left[index] + right[index];
    }
    Ok(output)
}

fn require_length(name: &'static str, values: &[f32], expected: usize) -> Result<(), SpecError> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(SpecError::VectorLength {
            op: name,
            expected,
            actual: values.len(),
        })
    }
}

fn require_length_named(
    name: &'static str,
    layer_index: usize,
    values: &[f32],
    expected: usize,
) -> Result<(), SpecError> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(SpecError::LayerVectorLength {
            name,
            layer_index,
            expected,
            actual: values.len(),
        })
    }
}

fn require_shape(
    name: &'static str,
    tensor: &Tensor,
    rows: usize,
    cols: usize,
) -> Result<(), SpecError> {
    if tensor.rows == rows && tensor.cols == cols {
        Ok(())
    } else {
        Err(SpecError::TensorShape {
            name,
            expected_rows: rows,
            expected_cols: cols,
            actual_rows: tensor.rows,
            actual_cols: tensor.cols,
        })
    }
}

fn require_shape_named(
    name: &'static str,
    layer_index: usize,
    tensor: &Tensor,
    rows: usize,
    cols: usize,
) -> Result<(), SpecError> {
    if tensor.rows == rows && tensor.cols == cols {
        Ok(())
    } else {
        Err(SpecError::LayerTensorShape {
            name,
            layer_index,
            expected_rows: rows,
            expected_cols: cols,
            actual_rows: tensor.rows,
            actual_cols: tensor.cols,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpecError {
    InvalidConfig(&'static str),
    SizeOverflow,
    TensorDataLength {
        rows: usize,
        cols: usize,
        expected: usize,
        actual: usize,
    },
    TensorIndex {
        row: usize,
        column: usize,
        rows: usize,
        cols: usize,
    },
    TensorShape {
        name: &'static str,
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    LayerTensorShape {
        name: &'static str,
        layer_index: usize,
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    VectorLength {
        op: &'static str,
        expected: usize,
        actual: usize,
    },
    LayerVectorLength {
        name: &'static str,
        layer_index: usize,
        expected: usize,
        actual: usize,
    },
    PairLength {
        op: &'static str,
        left: usize,
        right: usize,
    },
    LayerCount {
        expected: usize,
        actual: usize,
    },
    TokenOutOfRange {
        token_id: u32,
        vocab: usize,
    },
    KvSlotOutOfRange(usize),
    KvWidth {
        expected: usize,
        key_actual: usize,
        value_actual: usize,
    },
    CacheInvariant(&'static str),
    CacheSlotLengthMismatch {
        slot: usize,
        expected: usize,
        actual: usize,
    },
    CachePosition {
        slot: usize,
        expected: usize,
        actual: usize,
    },
    AttentionSequenceLength {
        keys: usize,
        values: usize,
        positions: usize,
    },
    EmptyAttention,
    EmptyPrefill,
    EmptyLogits,
    EmptyVector(&'static str),
    RopeShape {
        values: usize,
        heads: usize,
        head_dim: usize,
    },
}

impl fmt::Display for SpecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(formatter, "invalid spec config: {message}"),
            Self::SizeOverflow => formatter.write_str("logical tensor element count overflow"),
            Self::TensorDataLength {
                rows,
                cols,
                expected,
                actual,
            } => write!(
                formatter,
                "tensor [{rows}, {cols}] needs {expected} elements, got {actual}"
            ),
            Self::TensorIndex {
                row,
                column,
                rows,
                cols,
            } => write!(
                formatter,
                "tensor index ({row}, {column}) outside [{rows}, {cols}]"
            ),
            Self::TensorShape {
                name,
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            } => write!(
                formatter,
                "{name} shape [{actual_rows}, {actual_cols}] != [{expected_rows}, {expected_cols}]"
            ),
            Self::LayerTensorShape {
                name,
                layer_index,
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            } => write!(
                formatter,
                "layer {layer_index} {name} shape [{actual_rows}, {actual_cols}] != [{expected_rows}, {expected_cols}]"
            ),
            Self::VectorLength {
                op,
                expected,
                actual,
            } => write!(formatter, "{op} vector length {actual} != {expected}"),
            Self::LayerVectorLength {
                name,
                layer_index,
                expected,
                actual,
            } => write!(
                formatter,
                "layer {layer_index} {name} vector length {actual} != {expected}"
            ),
            Self::PairLength { op, left, right } => {
                write!(formatter, "{op} vector lengths differ: {left} != {right}")
            }
            Self::LayerCount { expected, actual } => {
                write!(formatter, "layer count {actual} != {expected}")
            }
            Self::TokenOutOfRange { token_id, vocab } => {
                write!(formatter, "token {token_id} outside vocabulary {vocab}")
            }
            Self::KvSlotOutOfRange(slot) => write!(formatter, "KV slot {slot} outside 0..44"),
            Self::KvWidth {
                expected,
                key_actual,
                value_actual,
            } => write!(
                formatter,
                "KV width key={key_actual}, value={value_actual}, expected={expected}"
            ),
            Self::CacheInvariant(message) => write!(formatter, "KV cache invariant: {message}"),
            Self::CacheSlotLengthMismatch {
                slot,
                expected,
                actual,
            } => write!(
                formatter,
                "KV slot {slot} length {actual} != expected {expected}"
            ),
            Self::CachePosition {
                slot,
                expected,
                actual,
            } => write!(
                formatter,
                "KV slot {slot} expected position {expected}, got {actual}"
            ),
            Self::AttentionSequenceLength {
                keys,
                values,
                positions,
            } => write!(
                formatter,
                "attention sequence lengths keys={keys}, values={values}, positions={positions}"
            ),
            Self::EmptyAttention => {
                formatter.write_str("attention requires at least one KV position")
            }
            Self::EmptyPrefill => formatter.write_str("prefill requires at least one token"),
            Self::EmptyLogits => {
                formatter.write_str("greedy selection requires at least one logit")
            }
            Self::EmptyVector(op) => write!(formatter, "{op} requires a non-empty vector"),
            Self::RopeShape {
                values,
                heads,
                head_dim,
            } => write!(
                formatter,
                "RoPE values={values} incompatible with heads={heads}, head_dim={head_dim}"
            ),
        }
    }
}

impl Error for SpecError {}
