//! The pinned eager bf16 reference forward for Nanbeige4.2-3B.
//!
//! This is deliberately the semantic, not performance, implementation.  It
//! shares the single 44-binding loop runner with every other profile, keeps
//! activations and K/V in bf16 at the named cast boundaries, and exports only
//! the final untied lm-head logits as f32.

use std::{error::Error, fmt, path::Path};

use crate::artifact::safetensors::{
    DEFAULT_MAX_RANGE_BYTES, RowPanel, SafetensorDtype, SafetensorsRangeIndex,
};

use super::{
    attention::{AttentionError, eager_gqa_attention_from_cache},
    kv::{
        KV_ELEMENTS_PER_POSITION, KV_SLOT_COUNT, KvCache, KvCacheError, LOOP_COUNT,
        PHYSICAL_LAYER_COUNT, slot_for,
    },
    layer::{
        HfBf16EagerLayerWeights, HfBf16LayerError, NANBEIGE_HIDDEN_SIZE,
        NANBEIGE_INTERMEDIATE_SIZE, NANBEIGE_KV_PROJECTION_SIZE, NANBEIGE_Q_PROJECTION_SIZE,
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

/// A refusal while materializing the explicit test/oracle source closure.
///
/// Production activation remains artifact-led; this source loader exists only
/// for the model-gated `hf-bf16-eager` conformance route and is never reached
/// from the CLI or ordinary model activation path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HfBf16EagerSourceError {
    /// The pinned source closure failed authenticated range-index construction.
    SourceClosure { detail: String },
    /// A named source tensor was absent or disagreed with its fixed bf16 shape.
    TensorContract { tensor: String, detail: String },
    /// A bounded, authenticated source panel could not be read.
    TensorRead { tensor: String, detail: String },
    /// A source tensor could not form its declared row-major matrix.
    TensorMatrix { tensor: String, detail: String },
    /// The assembled tensors failed the eager engine's one-model validation.
    ModelContract { detail: String },
}

impl fmt::Display for HfBf16EagerSourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceClosure { detail } => {
                write!(formatter, "hf-bf16 eager source closure rejected: {detail}")
            }
            Self::TensorContract { tensor, detail } => {
                write!(
                    formatter,
                    "hf-bf16 eager source tensor {tensor} rejected: {detail}"
                )
            }
            Self::TensorRead { tensor, detail } => {
                write!(
                    formatter,
                    "hf-bf16 eager source tensor {tensor} unreadable: {detail}"
                )
            }
            Self::TensorMatrix { tensor, detail } => {
                write!(
                    formatter,
                    "hf-bf16 eager source tensor {tensor} has invalid matrix: {detail}"
                )
            }
            Self::ModelContract { detail } => {
                write!(
                    formatter,
                    "hf-bf16 eager assembled model rejected: {detail}"
                )
            }
        }
    }
}

impl Error for HfBf16EagerSourceError {}

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
    /// Materialize the authenticated pinned source closure for the
    /// model-gated semantic-oracle route.
    ///
    /// This intentionally reads one bounded row panel at a time: the source
    /// index verifies the complete closure before exposing a byte range, and
    /// this adapter never maps a shard or opens an unchecked tensor path.
    pub fn from_pinned_source(
        source_dir: impl AsRef<Path>,
    ) -> Result<Self, HfBf16EagerSourceError> {
        let source =
            SafetensorsRangeIndex::open_pinned_nanbeige42(source_dir).map_err(|error| {
                HfBf16EagerSourceError::SourceClosure {
                    detail: error.to_string(),
                }
            })?;
        let embeddings = read_pinned_matrix(
            &source,
            "model.embed_tokens.weight",
            NANBEIGE_VOCAB_SIZE,
            NANBEIGE_HIDDEN_SIZE,
        )?;
        let final_norm = read_pinned_vector(&source, "model.norm.weight", NANBEIGE_HIDDEN_SIZE)?;
        let lm_head = read_pinned_matrix(
            &source,
            "lm_head.weight",
            NANBEIGE_VOCAB_SIZE,
            NANBEIGE_HIDDEN_SIZE,
        )?;
        let layer_values = (0..PHYSICAL_LAYER_COUNT)
            .map(|layer_index| read_pinned_layer(&source, layer_index))
            .collect::<Result<Vec<_>, _>>()?;
        let layers =
            layer_values
                .try_into()
                .map_err(|observed: Vec<HfBf16EagerLayerWeights>| {
                    HfBf16EagerSourceError::ModelContract {
                        detail: format!(
                            "expected {PHYSICAL_LAYER_COUNT} physical layers, materialized {}",
                            observed.len()
                        ),
                    }
                })?;
        let weights = Self {
            embeddings,
            layers,
            final_norm,
            lm_head,
        };
        weights
            .validate()
            .map_err(|error| HfBf16EagerSourceError::ModelContract {
                detail: format!("{error:?}"),
            })?;
        Ok(weights)
    }

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

fn read_pinned_layer(
    source: &SafetensorsRangeIndex,
    layer_index: usize,
) -> Result<HfBf16EagerLayerWeights, HfBf16EagerSourceError> {
    let prefix = format!("model.layers.{layer_index}");
    Ok(HfBf16EagerLayerWeights {
        input_norm: read_pinned_vector(
            source,
            &format!("{prefix}.input_layernorm.weight"),
            NANBEIGE_HIDDEN_SIZE,
        )?,
        q_proj: read_pinned_matrix(
            source,
            &format!("{prefix}.self_attn.q_proj.weight"),
            NANBEIGE_Q_PROJECTION_SIZE,
            NANBEIGE_HIDDEN_SIZE,
        )?,
        k_proj: read_pinned_matrix(
            source,
            &format!("{prefix}.self_attn.k_proj.weight"),
            NANBEIGE_KV_PROJECTION_SIZE,
            NANBEIGE_HIDDEN_SIZE,
        )?,
        v_proj: read_pinned_matrix(
            source,
            &format!("{prefix}.self_attn.v_proj.weight"),
            NANBEIGE_KV_PROJECTION_SIZE,
            NANBEIGE_HIDDEN_SIZE,
        )?,
        o_proj: read_pinned_matrix(
            source,
            &format!("{prefix}.self_attn.o_proj.weight"),
            NANBEIGE_HIDDEN_SIZE,
            NANBEIGE_Q_PROJECTION_SIZE,
        )?,
        post_attention_norm: read_pinned_vector(
            source,
            &format!("{prefix}.post_attention_layernorm.weight"),
            NANBEIGE_HIDDEN_SIZE,
        )?,
        gate_proj: read_pinned_matrix(
            source,
            &format!("{prefix}.mlp.gate_proj.weight"),
            NANBEIGE_INTERMEDIATE_SIZE,
            NANBEIGE_HIDDEN_SIZE,
        )?,
        up_proj: read_pinned_matrix(
            source,
            &format!("{prefix}.mlp.up_proj.weight"),
            NANBEIGE_INTERMEDIATE_SIZE,
            NANBEIGE_HIDDEN_SIZE,
        )?,
        down_proj: read_pinned_matrix(
            source,
            &format!("{prefix}.mlp.down_proj.weight"),
            NANBEIGE_HIDDEN_SIZE,
            NANBEIGE_INTERMEDIATE_SIZE,
        )?,
    })
}

fn read_pinned_vector(
    source: &SafetensorsRangeIndex,
    tensor: &str,
    length: usize,
) -> Result<Vec<Bf16>, HfBf16EagerSourceError> {
    check_pinned_tensor(source, tensor, &[length])?;
    let bytes = source
        .read_range(tensor, RowPanel::WholeTensor)
        .map_err(|error| HfBf16EagerSourceError::TensorRead {
            tensor: tensor.to_owned(),
            detail: error.to_string(),
        })?;
    let mut values = Vec::with_capacity(length);
    append_bf16_values(tensor, &mut values, &bytes, length)?;
    Ok(values)
}

fn read_pinned_matrix(
    source: &SafetensorsRangeIndex,
    tensor: &str,
    rows: usize,
    columns: usize,
) -> Result<Bf16Matrix, HfBf16EagerSourceError> {
    check_pinned_tensor(source, tensor, &[rows, columns])?;
    let expected_values =
        rows.checked_mul(columns)
            .ok_or_else(|| HfBf16EagerSourceError::TensorContract {
                tensor: tensor.to_owned(),
                detail: "matrix element count overflowed usize".to_owned(),
            })?;
    let row_bytes = columns
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| HfBf16EagerSourceError::TensorContract {
            tensor: tensor.to_owned(),
            detail: "matrix row byte count overflowed usize".to_owned(),
        })?;
    let panel_cap = usize::try_from(DEFAULT_MAX_RANGE_BYTES).map_err(|_| {
        HfBf16EagerSourceError::TensorContract {
            tensor: tensor.to_owned(),
            detail: "bounded source panel cap does not fit usize".to_owned(),
        }
    })?;
    let rows_per_panel = panel_cap / row_bytes;
    if rows_per_panel == 0 {
        return Err(HfBf16EagerSourceError::TensorContract {
            tensor: tensor.to_owned(),
            detail: format!(
                "one {row_bytes}-byte row exceeds bounded source panel cap {panel_cap}"
            ),
        });
    }

    let mut values = Vec::with_capacity(expected_values);
    let mut start_row = 0_usize;
    while start_row < rows {
        let row_count = rows_per_panel.min(rows - start_row);
        let source_start =
            u64::try_from(start_row).map_err(|_| HfBf16EagerSourceError::TensorContract {
                tensor: tensor.to_owned(),
                detail: "row start does not fit source range index".to_owned(),
            })?;
        let source_count =
            u64::try_from(row_count).map_err(|_| HfBf16EagerSourceError::TensorContract {
                tensor: tensor.to_owned(),
                detail: "row count does not fit source range index".to_owned(),
            })?;
        let bytes = source
            .read_range(
                tensor,
                RowPanel::Rows {
                    start_row: source_start,
                    row_count: source_count,
                },
            )
            .map_err(|error| HfBf16EagerSourceError::TensorRead {
                tensor: tensor.to_owned(),
                detail: error.to_string(),
            })?;
        append_bf16_values(tensor, &mut values, &bytes, row_count * columns)?;
        start_row += row_count;
    }
    Bf16Matrix::new(rows, columns, values).map_err(|error| HfBf16EagerSourceError::TensorMatrix {
        tensor: tensor.to_owned(),
        detail: format!("{error:?}"),
    })
}

fn check_pinned_tensor(
    source: &SafetensorsRangeIndex,
    tensor: &str,
    expected_shape: &[usize],
) -> Result<(), HfBf16EagerSourceError> {
    let expected_shape = expected_shape
        .iter()
        .copied()
        .map(u64::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| HfBf16EagerSourceError::TensorContract {
            tensor: tensor.to_owned(),
            detail: "expected shape does not fit source index dimensions".to_owned(),
        })?;
    let expected_values = expected_shape
        .iter()
        .try_fold(1_u64, |total, dimension| total.checked_mul(*dimension));
    let expected_bytes = expected_values
        .and_then(|values| values.checked_mul(SafetensorDtype::Bf16.byte_width()))
        .ok_or_else(|| HfBf16EagerSourceError::TensorContract {
            tensor: tensor.to_owned(),
            detail: "expected bf16 byte count overflowed u64".to_owned(),
        })?;
    let observed = source
        .census()
        .into_iter()
        .find(|entry| entry.name == tensor)
        .ok_or_else(|| HfBf16EagerSourceError::TensorContract {
            tensor: tensor.to_owned(),
            detail: "tensor is absent from the verified source census".to_owned(),
        })?;
    if observed.dtype != SafetensorDtype::Bf16
        || observed.shape != expected_shape
        || observed.len != expected_bytes
    {
        return Err(HfBf16EagerSourceError::TensorContract {
            tensor: tensor.to_owned(),
            detail: format!(
                "expected dtype={} shape={expected_shape:?} bytes={expected_bytes}; observed dtype={} shape={:?} bytes={}",
                SafetensorDtype::Bf16.as_str(),
                observed.dtype.as_str(),
                observed.shape,
                observed.len,
            ),
        });
    }
    Ok(())
}

fn append_bf16_values(
    tensor: &str,
    destination: &mut Vec<Bf16>,
    bytes: &[u8],
    expected_values: usize,
) -> Result<(), HfBf16EagerSourceError> {
    let expected_bytes = expected_values
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| HfBf16EagerSourceError::TensorRead {
            tensor: tensor.to_owned(),
            detail: "expected bf16 panel byte count overflowed usize".to_owned(),
        })?;
    if bytes.len() != expected_bytes {
        return Err(HfBf16EagerSourceError::TensorRead {
            tensor: tensor.to_owned(),
            detail: format!(
                "expected {expected_bytes} bytes for {expected_values} bf16 values, read {}",
                bytes.len()
            ),
        });
    }
    destination.extend(
        bytes
            .chunks_exact(std::mem::size_of::<u16>())
            .map(|pair| Bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]]))),
    );
    Ok(())
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
    let attention = eager_gqa_attention_from_cache(&query, cache, kv_slot)?;
    let attention_output = layer.o_proj.project_f32_accumulate_cast_back(&attention)?;
    Ok(layer.finish_attention_and_mlp(hidden, &attention_output)?)
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
