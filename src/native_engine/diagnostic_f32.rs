//! Structural f32 forward profile for Nanbeige parity diagnosis.
//!
//! `diagnostic-f32` widens each bf16 weight/embedding scalar once at model
//! loading, then keeps activations, RoPE, K/V cache, attention, residuals, and
//! logits in f32.  It is a structural bisect oracle, not an HF-bf16 parity
//! claim: expected greedy-token differences are recorded as token-flip fixtures.

use super::{
    kv::{KV_SLOT_COUNT, LOOP_COUNT, PHYSICAL_LAYER_COUNT, slot_for},
    looprun::{LayerBinding, LoopRunner, PositionContext, StructuralLayerExecutor},
    tensor::Bf16,
};

/// The execution-identity label selected by this implementation.
pub const DIAGNOSTIC_F32_PROFILE: &str = "diagnostic-f32";

/// Model dimensions and numeric constants consumed by the diagnostic profile.
///
/// The public constructor also permits miniature dimensions for scalar
/// conformance fixtures, while requiring the production loop depth of 22.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticF32Config {
    /// Hidden activation width.
    pub hidden_size: usize,
    /// Number of query heads.
    pub query_heads: usize,
    /// Number of key/value heads.
    pub kv_heads: usize,
    /// Explicit head width; never inferred from hidden/query heads.
    pub head_dim: usize,
    /// SwiGLU intermediate width.
    pub intermediate_size: usize,
    /// Embedding and untied lm-head row count.
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    pub rms_epsilon: f32,
    /// Split-half RoPE theta.
    pub rope_theta: f32,
}

impl DiagnosticF32Config {
    /// The fixed Nanbeige4.2-3B production configuration.
    #[must_use]
    pub const fn nanbeige() -> Self {
        Self {
            hidden_size: 3_072,
            query_heads: 48,
            kv_heads: 8,
            head_dim: 128,
            intermediate_size: 10_752,
            vocab_size: 166_144,
            rms_epsilon: 1.0e-5,
            rope_theta: 70_000_000.0,
        }
    }

    /// The f32 query projection width.
    #[must_use]
    pub const fn query_width(&self) -> usize {
        self.query_heads * self.head_dim
    }

    /// The f32 key/value projection width.
    #[must_use]
    pub const fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    fn validate(&self) -> Result<(), DiagnosticF32Error> {
        if self.hidden_size == 0
            || self.query_heads == 0
            || self.kv_heads == 0
            || self.head_dim == 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
        {
            return Err(DiagnosticF32Error::InvalidConfig(
                "all model widths must be nonzero",
            ));
        }
        if self.query_heads % self.kv_heads != 0 {
            return Err(DiagnosticF32Error::InvalidConfig(
                "query_heads must divide evenly into kv_heads groups",
            ));
        }
        if self.head_dim % 2 != 0 {
            return Err(DiagnosticF32Error::InvalidConfig(
                "split-half RoPE requires an even head_dim",
            ));
        }
        if !self.rms_epsilon.is_finite() || self.rms_epsilon <= 0.0 {
            return Err(DiagnosticF32Error::InvalidConfig(
                "rms_epsilon must be finite and positive",
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(DiagnosticF32Error::InvalidConfig(
                "rope_theta must be finite and positive",
            ));
        }
        Ok(())
    }
}

/// Shape-checked f32 matrix after the diagnostic profile's one-time widening.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticF32Matrix {
    rows: usize,
    columns: usize,
    values: Vec<f32>,
}

impl DiagnosticF32Matrix {
    /// Builds a matrix from already-widened f32 values.
    pub fn new(rows: usize, columns: usize, values: Vec<f32>) -> Result<Self, DiagnosticF32Error> {
        let expected = rows
            .checked_mul(columns)
            .ok_or(DiagnosticF32Error::MatrixStorage {
                rows,
                columns,
                expected: usize::MAX,
                actual: values.len(),
            })?;
        if values.len() != expected {
            return Err(DiagnosticF32Error::MatrixStorage {
                rows,
                columns,
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    /// Widens bf16 source values exactly once at diagnostic-profile load/entry.
    pub fn widen_from_bf16(
        rows: usize,
        columns: usize,
        values: &[Bf16],
    ) -> Result<Self, DiagnosticF32Error> {
        Self::new(
            rows,
            columns,
            values.iter().map(|value| value.to_f32()).collect(),
        )
    }

    /// Number of output rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of input columns.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Returns one f32 row without any dtype conversion.
    pub fn row(&self, row: usize) -> Result<&[f32], DiagnosticF32Error> {
        let start = row
            .checked_mul(self.columns)
            .ok_or(DiagnosticF32Error::RowOutOfRange {
                row,
                rows: self.rows,
            })?;
        let end = start + self.columns;
        self.values
            .get(start..end)
            .ok_or(DiagnosticF32Error::RowOutOfRange {
                row,
                rows: self.rows,
            })
    }

    /// Applies a bias-free f32 projection in deterministic row-major order.
    pub fn matvec(&self, input: &[f32], op: &'static str) -> Result<Vec<f32>, DiagnosticF32Error> {
        if input.len() != self.columns {
            return Err(DiagnosticF32Error::ProjectionInput {
                op,
                expected: self.columns,
                actual: input.len(),
            });
        }
        Ok(self
            .values
            .chunks_exact(self.columns)
            .map(|row| {
                row.iter()
                    .zip(input)
                    .map(|(weight, value)| weight * value)
                    .sum()
            })
            .collect())
    }
}

/// A physical layer's already-widened f32 tensors.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticF32LayerWeights {
    pub input_norm: Vec<f32>,
    pub q_proj: DiagnosticF32Matrix,
    pub k_proj: DiagnosticF32Matrix,
    pub v_proj: DiagnosticF32Matrix,
    pub o_proj: DiagnosticF32Matrix,
    pub post_attention_norm: Vec<f32>,
    pub gate_proj: DiagnosticF32Matrix,
    pub up_proj: DiagnosticF32Matrix,
    pub down_proj: DiagnosticF32Matrix,
}

impl DiagnosticF32LayerWeights {
    fn validate(&self, config: &DiagnosticF32Config) -> Result<(), DiagnosticF32Error> {
        validate_vector("input_norm", &self.input_norm, config.hidden_size)?;
        validate_matrix(
            "q_proj",
            &self.q_proj,
            config.query_width(),
            config.hidden_size,
        )?;
        validate_matrix(
            "k_proj",
            &self.k_proj,
            config.kv_width(),
            config.hidden_size,
        )?;
        validate_matrix(
            "v_proj",
            &self.v_proj,
            config.kv_width(),
            config.hidden_size,
        )?;
        validate_matrix(
            "o_proj",
            &self.o_proj,
            config.hidden_size,
            config.query_width(),
        )?;
        validate_vector(
            "post_attention_norm",
            &self.post_attention_norm,
            config.hidden_size,
        )?;
        validate_matrix(
            "gate_proj",
            &self.gate_proj,
            config.intermediate_size,
            config.hidden_size,
        )?;
        validate_matrix(
            "up_proj",
            &self.up_proj,
            config.intermediate_size,
            config.hidden_size,
        )?;
        validate_matrix(
            "down_proj",
            &self.down_proj,
            config.hidden_size,
            config.intermediate_size,
        )
    }
}

/// Complete f32 model state, with 22 reusable physical decoder-layer weights.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticF32Weights {
    pub embeddings: DiagnosticF32Matrix,
    pub layers: [DiagnosticF32LayerWeights; PHYSICAL_LAYER_COUNT],
    pub final_norm: Vec<f32>,
    pub lm_head: DiagnosticF32Matrix,
}

impl DiagnosticF32Weights {
    fn validate(&self, config: &DiagnosticF32Config) -> Result<(), DiagnosticF32Error> {
        validate_matrix(
            "embeddings",
            &self.embeddings,
            config.vocab_size,
            config.hidden_size,
        )?;
        for layer in &self.layers {
            layer.validate(config)?;
        }
        validate_vector("final_norm", &self.final_norm, config.hidden_size)?;
        validate_matrix(
            "lm_head",
            &self.lm_head,
            config.vocab_size,
            config.hidden_size,
        )
    }
}

/// Typed f32 K/V cache with the shared 44 logical slot mapping.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticF32KvCache {
    slots: Vec<DiagnosticF32KvSlot>,
    capacity_positions: usize,
    vector_width: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct DiagnosticF32KvSlot {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl DiagnosticF32KvCache {
    /// Reserves f32 K/V storage before execution; later append calls are
    /// shape- and order-checked against `slot = layer + loop * 22`.
    pub fn try_with_capacity(
        capacity_positions: usize,
        vector_width: usize,
    ) -> Result<Self, DiagnosticF32Error> {
        if capacity_positions == 0 || vector_width == 0 {
            return Err(DiagnosticF32Error::InvalidConfig(
                "diagnostic f32 K/V capacity and width must be nonzero",
            ));
        }
        let element_capacity = capacity_positions.checked_mul(vector_width).ok_or(
            DiagnosticF32Error::CacheCapacityOverflow {
                positions: capacity_positions,
                width: vector_width,
            },
        )?;
        let mut slots = Vec::new();
        slots.try_reserve_exact(KV_SLOT_COUNT).map_err(|_| {
            DiagnosticF32Error::CacheAllocationRefused {
                positions: capacity_positions,
            }
        })?;
        for _ in 0..KV_SLOT_COUNT {
            let mut keys = Vec::new();
            let mut values = Vec::new();
            keys.try_reserve_exact(element_capacity).map_err(|_| {
                DiagnosticF32Error::CacheAllocationRefused {
                    positions: capacity_positions,
                }
            })?;
            values.try_reserve_exact(element_capacity).map_err(|_| {
                DiagnosticF32Error::CacheAllocationRefused {
                    positions: capacity_positions,
                }
            })?;
            slots.push(DiagnosticF32KvSlot { keys, values });
        }
        Ok(Self {
            slots,
            capacity_positions,
            vector_width,
        })
    }

    /// The number of complete token positions, refusing any divergent slot.
    pub fn sequence_len(&self) -> Result<usize, DiagnosticF32Error> {
        let Some(first) = self.slots.first() else {
            return Err(DiagnosticF32Error::InvalidConfig(
                "diagnostic f32 K/V has no slots",
            ));
        };
        let expected = first.keys.len() / self.vector_width;
        for (slot, entry) in self.slots.iter().enumerate() {
            let actual = entry.keys.len() / self.vector_width;
            if actual != expected || entry.values.len() / self.vector_width != actual {
                return Err(DiagnosticF32Error::DivergentCacheLength {
                    slot,
                    expected,
                    actual,
                });
            }
        }
        Ok(expected)
    }

    /// Appends one f32 K/V pair to a logical slot.
    pub fn append(
        &mut self,
        slot: usize,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), DiagnosticF32Error> {
        if key.len() != self.vector_width {
            return Err(DiagnosticF32Error::CacheVectorLength {
                vector: "key",
                expected: self.vector_width,
                actual: key.len(),
            });
        }
        if value.len() != self.vector_width {
            return Err(DiagnosticF32Error::CacheVectorLength {
                vector: "value",
                expected: self.vector_width,
                actual: value.len(),
            });
        }
        let entry = self
            .slots
            .get_mut(slot)
            .ok_or(DiagnosticF32Error::CacheSlotOutOfRange { slot })?;
        if position >= self.capacity_positions {
            return Err(DiagnosticF32Error::CacheCapacityExceeded {
                slot,
                capacity_positions: self.capacity_positions,
            });
        }
        let expected_position = entry.keys.len() / self.vector_width;
        if position != expected_position {
            return Err(DiagnosticF32Error::CacheNonAppendPosition {
                slot,
                expected_position,
                received_position: position,
            });
        }
        entry.keys.extend_from_slice(key);
        entry.values.extend_from_slice(value);
        Ok(())
    }

    /// Number of committed positions in one logical slot.
    pub fn len_for_slot(&self, slot: usize) -> Result<usize, DiagnosticF32Error> {
        self.slots
            .get(slot)
            .map(|entry| entry.keys.len() / self.vector_width)
            .ok_or(DiagnosticF32Error::CacheSlotOutOfRange { slot })
    }

    /// Reads a key vector from a logical slot.
    pub fn key_at(&self, slot: usize, position: usize) -> Result<&[f32], DiagnosticF32Error> {
        self.vector_at(slot, position, "key")
    }

    /// Reads a value vector from a logical slot.
    pub fn value_at(&self, slot: usize, position: usize) -> Result<&[f32], DiagnosticF32Error> {
        self.vector_at(slot, position, "value")
    }

    fn vector_at(
        &self,
        slot: usize,
        position: usize,
        vector: &'static str,
    ) -> Result<&[f32], DiagnosticF32Error> {
        let entry = self
            .slots
            .get(slot)
            .ok_or(DiagnosticF32Error::CacheSlotOutOfRange { slot })?;
        let source = if vector == "key" {
            &entry.keys
        } else {
            &entry.values
        };
        let start = position
            .checked_mul(self.vector_width)
            .ok_or(DiagnosticF32Error::CacheSlotOutOfRange { slot })?;
        let end = start + self.vector_width;
        source
            .get(start..end)
            .ok_or(DiagnosticF32Error::CachePositionOutOfRange { slot, position })
    }
}

/// All named structural taps emitted for one logical layer execution.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticF32LayerTap {
    pub loop_index: usize,
    pub layer_index: usize,
    pub kv_slot: usize,
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

/// Complete structural evidence for one f32 forward position.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticF32Taps {
    pub layer_taps: Vec<DiagnosticF32LayerTap>,
    pub post_loop_norms: [Vec<f32>; LOOP_COUNT],
    pub logits: Vec<f32>,
    pub greedy_token: usize,
}

/// One completed diagnostic-f32 forward position.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticF32Forward {
    pub position: usize,
    pub taps: DiagnosticF32Taps,
}

/// Per-vector metrics emitted by a parity/bisect harness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiagnosticF32Metrics {
    pub max_abs: f32,
    pub max_rel: f32,
    pub max_ulp: u32,
    pub cosine: f32,
}

/// A first structural mismatch identified by the diagnostic tap stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FirstStructuralDivergence {
    pub loop_index: usize,
    pub layer_index: Option<usize>,
    pub point: &'static str,
}

/// Expected f32-vs-bf16 greedy divergence fixture identity.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenFlip {
    pub prompt_id: String,
    pub position: usize,
    pub bf16_token: usize,
    pub f32_token: usize,
    pub logit_gap: f32,
}

/// Returns any observed f32 token flip absent from the frozen fixture set.
#[must_use]
pub fn unlisted_token_flips(expected: &[TokenFlip], observed: &[TokenFlip]) -> Vec<TokenFlip> {
    observed
        .iter()
        .filter(|candidate| {
            !expected.iter().any(|known| {
                known.prompt_id == candidate.prompt_id
                    && known.position == candidate.position
                    && known.bf16_token == candidate.bf16_token
                    && known.f32_token == candidate.f32_token
                    && known.logit_gap.to_bits() == candidate.logit_gap.to_bits()
            })
        })
        .cloned()
        .collect()
}

/// Computes the named per-tap metric vector without changing either profile.
pub fn diagnostic_f32_metrics(
    reference: &[f32],
    observed: &[f32],
) -> Result<DiagnosticF32Metrics, DiagnosticF32Error> {
    if reference.len() != observed.len() {
        return Err(DiagnosticF32Error::ProjectionInput {
            op: "diagnostic_f32_metrics",
            expected: reference.len(),
            actual: observed.len(),
        });
    }
    if reference.is_empty() {
        return Err(DiagnosticF32Error::EmptyActivation {
            op: "diagnostic_f32_metrics",
        });
    }
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut max_ulp = 0_u32;
    let mut dot = 0.0_f32;
    let mut reference_norm = 0.0_f32;
    let mut observed_norm = 0.0_f32;
    for (&left, &right) in reference.iter().zip(observed) {
        max_abs = max_abs.max((left - right).abs());
        max_rel = max_rel.max((left - right).abs() / left.abs().max(f32::MIN_POSITIVE));
        max_ulp = max_ulp.max(ordered_f32_bits(left).abs_diff(ordered_f32_bits(right)));
        dot += left * right;
        reference_norm += left * left;
        observed_norm += right * right;
    }
    let cosine = if reference_norm == 0.0 && observed_norm == 0.0 {
        1.0
    } else if reference_norm == 0.0 || observed_norm == 0.0 {
        0.0
    } else {
        dot / (reference_norm.sqrt() * observed_norm.sqrt())
    };
    Ok(DiagnosticF32Metrics {
        max_abs,
        max_rel,
        max_ulp,
        cosine,
    })
}

/// Locates the first named f32 structural tap differing beyond `max_abs`.
#[must_use]
pub fn first_structural_divergence(
    expected: &DiagnosticF32Taps,
    observed: &DiagnosticF32Taps,
    max_abs: f32,
) -> Option<FirstStructuralDivergence> {
    if expected.layer_taps.len() != observed.layer_taps.len() {
        let first_missing = expected
            .layer_taps
            .get(observed.layer_taps.len())
            .or_else(|| observed.layer_taps.get(expected.layer_taps.len()))?;
        return Some(FirstStructuralDivergence {
            loop_index: first_missing.loop_index,
            layer_index: Some(first_missing.layer_index),
            point: "layer_count",
        });
    }
    for (expected_layer, observed_layer) in expected.layer_taps.iter().zip(&observed.layer_taps) {
        for (point, left, right) in [
            (
                "pre_attention",
                &expected_layer.input,
                &observed_layer.input,
            ),
            (
                "post_attention",
                &expected_layer.post_attention_residual,
                &observed_layer.post_attention_residual,
            ),
            ("post_mlp", &expected_layer.output, &observed_layer.output),
        ] {
            if left.len() != right.len()
                || left
                    .iter()
                    .zip(right)
                    .any(|(&left, &right)| (left - right).abs() > max_abs)
            {
                return Some(FirstStructuralDivergence {
                    loop_index: expected_layer.loop_index,
                    layer_index: Some(expected_layer.layer_index),
                    point,
                });
            }
        }
    }
    for loop_index in 0..LOOP_COUNT {
        let left = &expected.post_loop_norms[loop_index];
        let right = &observed.post_loop_norms[loop_index];
        if left.len() != right.len()
            || left
                .iter()
                .zip(right)
                .any(|(&left, &right)| (left - right).abs() > max_abs)
        {
            return Some(FirstStructuralDivergence {
                loop_index,
                layer_index: None,
                point: "post_loop_norm",
            });
        }
    }
    None
}

/// The diagnostic f32 engine.  It owns an f32 K/V cache, never a bf16 cache.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticF32Engine {
    config: DiagnosticF32Config,
    weights: DiagnosticF32Weights,
    kv_cache: DiagnosticF32KvCache,
}

impl DiagnosticF32Engine {
    /// Validates all shapes and reserves f32 K/V storage for the requested
    /// number of token positions before a forward execution begins.
    pub fn new(
        config: DiagnosticF32Config,
        weights: DiagnosticF32Weights,
        max_positions: usize,
    ) -> Result<Self, DiagnosticF32Error> {
        config.validate()?;
        weights.validate(&config)?;
        let kv_cache = DiagnosticF32KvCache::try_with_capacity(max_positions, config.kv_width())?;
        Ok(Self {
            config,
            weights,
            kv_cache,
        })
    }

    /// Returns the selected profile label for receipts and logging.
    #[must_use]
    pub const fn profile(&self) -> &'static str {
        DIAGNOSTIC_F32_PROFILE
    }

    /// Provides the f32 K/V cache for structural-slot assertions.
    #[must_use]
    pub const fn kv_cache(&self) -> &DiagnosticF32KvCache {
        &self.kv_cache
    }

    /// Runs one token through `22 layers -> norm -> 22 layers -> norm`.
    pub fn decode(&mut self, token_id: u32) -> Result<DiagnosticF32Forward, DiagnosticF32Error> {
        let position = self.kv_cache.sequence_len()?;
        let token = usize::try_from(token_id).map_err(|_| DiagnosticF32Error::TokenOutOfRange {
            token_id,
            vocab_size: self.config.vocab_size,
        })?;
        if token >= self.config.vocab_size {
            return Err(DiagnosticF32Error::TokenOutOfRange {
                token_id,
                vocab_size: self.config.vocab_size,
            });
        }
        let mut hidden = self.weights.embeddings.row(token)?.to_vec();
        let mut layer_taps = Vec::with_capacity(KV_SLOT_COUNT);
        let mut post_loop_norms = [Vec::new(), Vec::new()];
        let runner = LoopRunner::from_layer_weights(&self.weights.layers);
        let mut executor = DiagnosticF32Executor {
            config: &self.config,
            final_norm: &self.weights.final_norm,
            kv_cache: &mut self.kv_cache,
            layer_taps: &mut layer_taps,
            post_loop_norms: &mut post_loop_norms,
        };
        runner.run_token_structural(&mut executor, &mut hidden, PositionContext::at(position))?;
        let logits = self.weights.lm_head.matvec(&hidden, "lm_head")?;
        let greedy_token = greedy_argmax(&logits).ok_or(DiagnosticF32Error::EmptyLogits)?;
        Ok(DiagnosticF32Forward {
            position,
            taps: DiagnosticF32Taps {
                layer_taps,
                post_loop_norms,
                logits,
                greedy_token,
            },
        })
    }

    /// Runs a token prefix in order, retaining f32 K/V activations between positions.
    pub fn prefill(
        &mut self,
        token_ids: &[u32],
    ) -> Result<Vec<DiagnosticF32Forward>, DiagnosticF32Error> {
        if token_ids.is_empty() {
            return Err(DiagnosticF32Error::EmptyPrefill);
        }
        token_ids
            .iter()
            .map(|&token_id| self.decode(token_id))
            .collect()
    }
}

struct DiagnosticF32Executor<'a> {
    config: &'a DiagnosticF32Config,
    final_norm: &'a [f32],
    kv_cache: &'a mut DiagnosticF32KvCache,
    layer_taps: &'a mut Vec<DiagnosticF32LayerTap>,
    post_loop_norms: &'a mut [Vec<f32>; LOOP_COUNT],
}

impl StructuralLayerExecutor<DiagnosticF32LayerWeights> for DiagnosticF32Executor<'_> {
    type Hidden = Vec<f32>;
    type Error = DiagnosticF32Error;

    fn layer_forward(
        &mut self,
        binding: &LayerBinding<'_, DiagnosticF32LayerWeights>,
        hidden: &mut Self::Hidden,
        positions: PositionContext,
    ) -> Result<(), Self::Error> {
        let tap = run_f32_layer(
            self.config,
            binding.weights(),
            binding.loop_index(),
            binding.layer_index(),
            binding.kv_slot(),
            hidden,
            positions.cache_position,
            self.kv_cache,
        )?;
        *hidden = tap.output.clone();
        self.layer_taps.push(tap);
        Ok(())
    }

    fn final_rms_norm(
        &mut self,
        hidden: &mut Self::Hidden,
        _positions: PositionContext,
    ) -> Result<(), Self::Error> {
        let loop_index = self
            .layer_taps
            .len()
            .checked_div(PHYSICAL_LAYER_COUNT)
            .and_then(|completed| completed.checked_sub(1))
            .ok_or(DiagnosticF32Error::InvalidRunnerBoundary)?;
        if loop_index >= LOOP_COUNT {
            return Err(DiagnosticF32Error::InvalidRunnerBoundary);
        }
        *hidden = rms_norm_f32(
            hidden,
            self.final_norm,
            self.config.rms_epsilon,
            "final_rms_norm",
        )?;
        self.post_loop_norms[loop_index] = hidden.clone();
        Ok(())
    }
}

fn run_f32_layer(
    config: &DiagnosticF32Config,
    layer: &DiagnosticF32LayerWeights,
    loop_index: usize,
    layer_index: usize,
    kv_slot: usize,
    hidden: &[f32],
    position: usize,
    cache: &mut DiagnosticF32KvCache,
) -> Result<DiagnosticF32LayerTap, DiagnosticF32Error> {
    if slot_for(loop_index, layer_index) != Some(kv_slot) {
        return Err(DiagnosticF32Error::InvalidKvSlot {
            loop_index,
            layer_index,
            kv_slot,
        });
    }
    let input = hidden.to_vec();
    let attention_norm = rms_norm_f32(
        hidden,
        &layer.input_norm,
        config.rms_epsilon,
        "attention_rms_norm",
    )?;
    let mut query = layer.q_proj.matvec(&attention_norm, "q_proj")?;
    let mut key = layer.k_proj.matvec(&attention_norm, "k_proj")?;
    let value = layer.v_proj.matvec(&attention_norm, "v_proj")?;
    let rope = DiagnosticF32RopeTables::new(position + 1, config.head_dim, config.rope_theta)?;
    rope.apply_all_heads(position, &mut query)?;
    rope.apply_all_heads(position, &mut key)?;
    cache.append(kv_slot, position, &key, &value)?;
    let attention = dense_gqa_attention_f32(config, &query, cache, kv_slot)?;
    let attention_output = layer.o_proj.matvec(&attention, "o_proj")?;
    let post_attention_residual =
        residual_add_f32(&input, &attention_output, "attention_residual")?;
    let ffn_norm = rms_norm_f32(
        &post_attention_residual,
        &layer.post_attention_norm,
        config.rms_epsilon,
        "ffn_rms_norm",
    )?;
    let gate = layer.gate_proj.matvec(&ffn_norm, "gate_proj")?;
    let up = layer.up_proj.matvec(&ffn_norm, "up_proj")?;
    let swiglu = gate
        .iter()
        .zip(&up)
        .map(|(&gate, &up)| silu_f32(gate) * up)
        .collect::<Vec<_>>();
    let mlp_output = layer.down_proj.matvec(&swiglu, "down_proj")?;
    let output = residual_add_f32(&post_attention_residual, &mlp_output, "ffn_residual")?;
    Ok(DiagnosticF32LayerTap {
        loop_index,
        layer_index,
        kv_slot,
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
    })
}

/// f32 split-half RoPE tables; no activation is narrowed at application.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticF32RopeTables {
    head_dim: usize,
    positions: usize,
    cosine: Vec<f32>,
    sine: Vec<f32>,
}

impl DiagnosticF32RopeTables {
    /// Builds f32 cosine/sine tables for every requested position.
    pub fn new(positions: usize, head_dim: usize, theta: f32) -> Result<Self, DiagnosticF32Error> {
        if positions == 0 || head_dim == 0 || head_dim % 2 != 0 {
            return Err(DiagnosticF32Error::InvalidConfig(
                "f32 RoPE requires nonzero positions and even head_dim",
            ));
        }
        let half_dim = head_dim / 2;
        let total = positions
            .checked_mul(half_dim)
            .ok_or(DiagnosticF32Error::InvalidConfig(
                "f32 RoPE table size overflow",
            ))?;
        let mut cosine = Vec::with_capacity(total);
        let mut sine = Vec::with_capacity(total);
        for position in 0..positions {
            for pair in 0..half_dim {
                let inverse_frequency = theta.powf(-(2.0 * pair as f32) / head_dim as f32);
                let phase = position as f32 * inverse_frequency;
                cosine.push(phase.cos());
                sine.push(phase.sin());
            }
        }
        Ok(Self {
            head_dim,
            positions,
            cosine,
            sine,
        })
    }

    /// Applies the f32 table to each contiguous head, preserving f32 values.
    pub fn apply_all_heads(
        &self,
        position: usize,
        activation: &mut [f32],
    ) -> Result<(), DiagnosticF32Error> {
        if position >= self.positions {
            return Err(DiagnosticF32Error::RopePositionOutOfRange {
                position,
                table_positions: self.positions,
            });
        }
        if activation.len() % self.head_dim != 0 {
            return Err(DiagnosticF32Error::ProjectionInput {
                op: "rope_activation",
                expected: self.head_dim,
                actual: activation.len(),
            });
        }
        let half_dim = self.head_dim / 2;
        let offset = position * half_dim;
        for head in activation.chunks_exact_mut(self.head_dim) {
            for pair in 0..half_dim {
                let left = head[pair];
                let right = head[pair + half_dim];
                let cosine = self.cosine[offset + pair];
                let sine = self.sine[offset + pair];
                head[pair] = left * cosine - right * sine;
                head[pair + half_dim] = right * cosine + left * sine;
            }
        }
        Ok(())
    }
}

fn dense_gqa_attention_f32(
    config: &DiagnosticF32Config,
    query: &[f32],
    cache: &DiagnosticF32KvCache,
    slot: usize,
) -> Result<Vec<f32>, DiagnosticF32Error> {
    if query.len() != config.query_width() {
        return Err(DiagnosticF32Error::ProjectionInput {
            op: "attention_query",
            expected: config.query_width(),
            actual: query.len(),
        });
    }
    let sequence_len = cache.len_for_slot(slot)?;
    if sequence_len == 0 {
        return Err(DiagnosticF32Error::EmptyAttention);
    }
    let queries_per_kv = config.query_heads / config.kv_heads;
    let mut output = vec![0.0_f32; config.query_width()];
    let scale = (config.head_dim as f32).sqrt().recip();
    for query_head in 0..config.query_heads {
        let kv_head = query_head / queries_per_kv;
        let query_start = query_head * config.head_dim;
        let query_vector = &query[query_start..query_start + config.head_dim];
        let mut scores = Vec::with_capacity(sequence_len);
        for position in 0..sequence_len {
            let key = cache.key_at(slot, position)?;
            let key_start = kv_head * config.head_dim;
            scores.push(
                query_vector
                    .iter()
                    .zip(&key[key_start..key_start + config.head_dim])
                    .map(|(&query, &key)| query * key)
                    .sum::<f32>()
                    * scale,
            );
        }
        let probabilities = softmax_f32(&scores)?;
        let destination = &mut output[query_start..query_start + config.head_dim];
        for (position, probability) in probabilities.into_iter().enumerate() {
            let value = cache.value_at(slot, position)?;
            let value_start = kv_head * config.head_dim;
            for (destination, source) in destination
                .iter_mut()
                .zip(&value[value_start..value_start + config.head_dim])
            {
                *destination += probability * source;
            }
        }
    }
    Ok(output)
}

/// F32 softmax used by the diagnostic profile, with no cast-back site.
pub fn softmax_f32(scores: &[f32]) -> Result<Vec<f32>, DiagnosticF32Error> {
    let maximum = scores
        .iter()
        .copied()
        .reduce(f32::max)
        .ok_or(DiagnosticF32Error::EmptyAttention)?;
    let exponentials = scores
        .iter()
        .map(|score| (*score - maximum).exp())
        .collect::<Vec<_>>();
    let denominator = exponentials.iter().sum::<f32>();
    Ok(exponentials
        .iter()
        .map(|value| value / denominator)
        .collect())
}

/// Deterministic first-index-wins f32 greedy argmax.
#[must_use]
pub fn greedy_argmax(logits: &[f32]) -> Option<usize> {
    let mut best = None;
    for (index, value) in logits.iter().copied().enumerate() {
        if best
            .map(|(_, current)| value.total_cmp(&current).is_gt())
            .unwrap_or(true)
        {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index)
}

fn rms_norm_f32(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    op: &'static str,
) -> Result<Vec<f32>, DiagnosticF32Error> {
    if input.is_empty() {
        return Err(DiagnosticF32Error::EmptyActivation { op });
    }
    if input.len() != weight.len() {
        return Err(DiagnosticF32Error::ProjectionInput {
            op,
            expected: input.len(),
            actual: weight.len(),
        });
    }
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inverse_rms = (mean_square + epsilon).sqrt().recip();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(&value, &scale)| value * inverse_rms * scale)
        .collect())
}

fn residual_add_f32(
    left: &[f32],
    right: &[f32],
    op: &'static str,
) -> Result<Vec<f32>, DiagnosticF32Error> {
    if left.len() != right.len() {
        return Err(DiagnosticF32Error::ProjectionInput {
            op,
            expected: left.len(),
            actual: right.len(),
        });
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(&left, &right)| left + right)
        .collect())
}

fn silu_f32(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn validate_vector(
    name: &'static str,
    values: &[f32],
    expected: usize,
) -> Result<(), DiagnosticF32Error> {
    if values.len() != expected {
        return Err(DiagnosticF32Error::WeightVectorShape {
            name,
            expected,
            actual: values.len(),
        });
    }
    Ok(())
}

fn validate_matrix(
    name: &'static str,
    matrix: &DiagnosticF32Matrix,
    expected_rows: usize,
    expected_columns: usize,
) -> Result<(), DiagnosticF32Error> {
    if matrix.rows() != expected_rows || matrix.columns() != expected_columns {
        return Err(DiagnosticF32Error::WeightMatrixShape {
            name,
            rows: matrix.rows(),
            columns: matrix.columns(),
            expected_rows,
            expected_columns,
        });
    }
    Ok(())
}

fn ordered_f32_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

/// Construction, shape, cache, or execution refusal from diagnostic-f32.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiagnosticF32Error {
    InvalidConfig(&'static str),
    MatrixStorage {
        rows: usize,
        columns: usize,
        expected: usize,
        actual: usize,
    },
    RowOutOfRange {
        row: usize,
        rows: usize,
    },
    ProjectionInput {
        op: &'static str,
        expected: usize,
        actual: usize,
    },
    WeightVectorShape {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    WeightMatrixShape {
        name: &'static str,
        rows: usize,
        columns: usize,
        expected_rows: usize,
        expected_columns: usize,
    },
    TokenOutOfRange {
        token_id: u32,
        vocab_size: usize,
    },
    CacheSlotOutOfRange {
        slot: usize,
    },
    CachePositionOutOfRange {
        slot: usize,
        position: usize,
    },
    CacheVectorLength {
        vector: &'static str,
        expected: usize,
        actual: usize,
    },
    CacheNonAppendPosition {
        slot: usize,
        expected_position: usize,
        received_position: usize,
    },
    CacheCapacityExceeded {
        slot: usize,
        capacity_positions: usize,
    },
    CacheCapacityOverflow {
        positions: usize,
        width: usize,
    },
    CacheAllocationRefused {
        positions: usize,
    },
    DivergentCacheLength {
        slot: usize,
        expected: usize,
        actual: usize,
    },
    InvalidKvSlot {
        loop_index: usize,
        layer_index: usize,
        kv_slot: usize,
    },
    InvalidRunnerBoundary,
    RopePositionOutOfRange {
        position: usize,
        table_positions: usize,
    },
    EmptyActivation {
        op: &'static str,
    },
    EmptyAttention,
    EmptyLogits,
    EmptyPrefill,
}
