//! One physical `hf-bf16-eager` decoder-layer execution surface.

use super::nn::{
    RMS_NORM_EPSILON, ReferencePrimitiveError, residual_add_f32_cast_back,
    rms_norm_f32_reduce_cast_back, swiglu_f32_cast_back,
};
use super::tensor::Bf16;
use super::weights::{Bf16Matrix, WeightShapeError};

/// Nanbeige's hidden width.
pub const NANBEIGE_HIDDEN_SIZE: usize = 3_072;
/// Query projection width: 48 × 128.
pub const NANBEIGE_Q_PROJECTION_SIZE: usize = 6_144;
/// Key/value projection width: 8 × 128.
pub const NANBEIGE_KV_PROJECTION_SIZE: usize = 1_024;
/// SwiGLU gate/up intermediate width.
pub const NANBEIGE_INTERMEDIATE_SIZE: usize = 10_752;

/// The fixed shapes consumed by an unbiased physical decoder layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HfBf16EagerLayerWeights {
    /// Input RMSNorm scale.
    pub input_norm: Vec<Bf16>,
    /// Attention Q projection, 3072 → 6144.
    pub q_proj: Bf16Matrix,
    /// Attention K projection, 3072 → 1024.
    pub k_proj: Bf16Matrix,
    /// Attention V projection, 3072 → 1024.
    pub v_proj: Bf16Matrix,
    /// Attention output projection, 6144 → 3072.
    pub o_proj: Bf16Matrix,
    /// Post-attention RMSNorm scale.
    pub post_attention_norm: Vec<Bf16>,
    /// SwiGLU gate projection, 3072 → 10752.
    pub gate_proj: Bf16Matrix,
    /// SwiGLU up projection, 3072 → 10752.
    pub up_proj: Bf16Matrix,
    /// SwiGLU down projection, 10752 → 3072.
    pub down_proj: Bf16Matrix,
}

/// Layer construction or reference-forward error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HfBf16LayerError {
    /// A fixed model dimension disagreed with the truth-pack model shape.
    ProjectionShape {
        name: &'static str,
        rows: usize,
        columns: usize,
        expected_rows: usize,
        expected_columns: usize,
    },
    /// A norm scale did not have the 3072 hidden entries.
    NormShape {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A lower-level primitive rejected an activation shape.
    Primitive(ReferencePrimitiveError),
    /// A lower-level matrix rejected a projection shape.
    Projection(WeightShapeError),
}

impl From<ReferencePrimitiveError> for HfBf16LayerError {
    fn from(value: ReferencePrimitiveError) -> Self {
        Self::Primitive(value)
    }
}

impl From<WeightShapeError> for HfBf16LayerError {
    fn from(value: WeightShapeError) -> Self {
        Self::Projection(value)
    }
}

impl HfBf16EagerLayerWeights {
    /// Refuses a layer whose no-bias tensor shapes do not match Nanbeige.
    pub fn validate(&self) -> Result<(), HfBf16LayerError> {
        validate_norm("input_norm", &self.input_norm)?;
        validate_matrix(
            "q_proj",
            &self.q_proj,
            NANBEIGE_Q_PROJECTION_SIZE,
            NANBEIGE_HIDDEN_SIZE,
        )?;
        validate_matrix(
            "k_proj",
            &self.k_proj,
            NANBEIGE_KV_PROJECTION_SIZE,
            NANBEIGE_HIDDEN_SIZE,
        )?;
        validate_matrix(
            "v_proj",
            &self.v_proj,
            NANBEIGE_KV_PROJECTION_SIZE,
            NANBEIGE_HIDDEN_SIZE,
        )?;
        validate_matrix(
            "o_proj",
            &self.o_proj,
            NANBEIGE_HIDDEN_SIZE,
            NANBEIGE_Q_PROJECTION_SIZE,
        )?;
        validate_norm("post_attention_norm", &self.post_attention_norm)?;
        validate_matrix(
            "gate_proj",
            &self.gate_proj,
            NANBEIGE_INTERMEDIATE_SIZE,
            NANBEIGE_HIDDEN_SIZE,
        )?;
        validate_matrix(
            "up_proj",
            &self.up_proj,
            NANBEIGE_INTERMEDIATE_SIZE,
            NANBEIGE_HIDDEN_SIZE,
        )?;
        validate_matrix(
            "down_proj",
            &self.down_proj,
            NANBEIGE_HIDDEN_SIZE,
            NANBEIGE_INTERMEDIATE_SIZE,
        )
    }

    /// Applies the first f32-reduce/bf16-cast RMSNorm before eager attention.
    pub fn input_rms_norm(&self, hidden: &[Bf16]) -> Result<Vec<Bf16>, HfBf16LayerError> {
        Ok(rms_norm_f32_reduce_cast_back(
            hidden,
            &self.input_norm,
            RMS_NORM_EPSILON,
        )?)
    }

    /// Applies attention residual, post-attention norm, SwiGLU, and MLP residual.
    pub fn finish_attention_and_mlp(
        &self,
        hidden: &[Bf16],
        attention_output: &[Bf16],
    ) -> Result<Vec<Bf16>, HfBf16LayerError> {
        let after_attention = residual_add_f32_cast_back(hidden, attention_output)?;
        let normalized = rms_norm_f32_reduce_cast_back(
            &after_attention,
            &self.post_attention_norm,
            RMS_NORM_EPSILON,
        )?;
        let gate = self
            .gate_proj
            .project_f32_accumulate_cast_back(&normalized)?;
        let up = self.up_proj.project_f32_accumulate_cast_back(&normalized)?;
        let activated = swiglu_f32_cast_back(&gate, &up)?;
        let mlp_output = self
            .down_proj
            .project_f32_accumulate_cast_back(&activated)?;
        Ok(residual_add_f32_cast_back(&after_attention, &mlp_output)?)
    }
}

fn validate_norm(name: &'static str, values: &[Bf16]) -> Result<(), HfBf16LayerError> {
    if values.len() != NANBEIGE_HIDDEN_SIZE {
        return Err(HfBf16LayerError::NormShape {
            name,
            expected: NANBEIGE_HIDDEN_SIZE,
            actual: values.len(),
        });
    }
    Ok(())
}

fn validate_matrix(
    name: &'static str,
    matrix: &Bf16Matrix,
    expected_rows: usize,
    expected_columns: usize,
) -> Result<(), HfBf16LayerError> {
    if matrix.rows() != expected_rows || matrix.columns() != expected_columns {
        return Err(HfBf16LayerError::ProjectionShape {
            name,
            rows: matrix.rows(),
            columns: matrix.columns(),
            expected_rows,
            expected_columns,
        });
    }
    Ok(())
}
