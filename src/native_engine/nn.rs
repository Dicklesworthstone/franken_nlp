//! Model-specific neural-network primitives for `hf-bf16-eager`.

use super::tensor::{Bf16, cast_f32_to_bf16};

/// The pinned Nanbeige RMSNorm epsilon.
pub const RMS_NORM_EPSILON: f32 = 1.0e-5;

/// Errors from shape-checked reference primitives.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReferencePrimitiveError {
    /// A paired tensor did not have the requested common length.
    LengthMismatch {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    /// Reduction has no defined RMS value for an empty activation.
    EmptyActivation { operation: &'static str },
}

/// Pinned cast sites used by the HF eager semantic profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HfBf16EagerCastSite {
    /// `embed_tokens(input_ids)` rows remain bf16.
    EmbeddingRowStaysBf16,
    /// RMSNorm reduces in f32 and narrows its output back to bf16.
    RmsNormF32ReduceCastBack,
    /// QK matmul output narrows to bf16 before score scaling.
    AttentionQkMatmulCastBack,
    /// The scaled QK score narrows to bf16 before softmax upcasts it.
    AttentionScaleCastBack,
    /// Attention softmax computes in f32 and narrows probabilities to bf16.
    SoftmaxF32CastBack,
    /// RoPE tables are f32 and rotate into bf16 q/k activations.
    RopeF32TableCastAtApplication,
    /// The bf16 lm_head result widens to f32 at the public logits boundary.
    LogitsExportF32,
}

/// The complete, ordered cast schedule for the profile.
pub const HF_BF16_EAGER_CAST_SCHEDULE: [HfBf16EagerCastSite; 7] = [
    HfBf16EagerCastSite::EmbeddingRowStaysBf16,
    HfBf16EagerCastSite::RmsNormF32ReduceCastBack,
    HfBf16EagerCastSite::AttentionQkMatmulCastBack,
    HfBf16EagerCastSite::AttentionScaleCastBack,
    HfBf16EagerCastSite::SoftmaxF32CastBack,
    HfBf16EagerCastSite::RopeF32TableCastAtApplication,
    HfBf16EagerCastSite::LogitsExportF32,
];

/// Returns a copied embedding row without widening it.
#[must_use]
pub fn embedding_row_stays_bf16(row: &[Bf16]) -> Vec<Bf16> {
    row.to_vec()
}

/// Performs the pinned RMSNorm cast graph and returns a bf16 activation.
///
/// The variance reduction and reciprocal square root execute in f32.  The
/// normalized activation then narrows to the input dtype *before* the bf16
/// scale multiply, matching `NanbeigeRMSNorm.forward` in the pinned source.
pub fn rms_norm_f32_reduce_cast_back(
    input: &[Bf16],
    weight: &[Bf16],
    epsilon: f32,
) -> Result<Vec<Bf16>, ReferencePrimitiveError> {
    if input.is_empty() {
        return Err(ReferencePrimitiveError::EmptyActivation {
            operation: "rms_norm",
        });
    }
    if input.len() != weight.len() {
        return Err(ReferencePrimitiveError::LengthMismatch {
            operation: "rms_norm",
            expected: input.len(),
            actual: weight.len(),
        });
    }
    let mean_square = input
        .iter()
        .map(|value| {
            let widened = value.to_f32();
            widened * widened
        })
        .sum::<f32>()
        / input.len() as f32;
    let inverse_rms = (mean_square + epsilon).sqrt().recip();
    let normalized = input
        .iter()
        .map(|activation| Bf16::from_f32(activation.to_f32() * inverse_rms))
        .collect::<Vec<_>>();
    Ok(normalized
        .iter()
        .zip(weight)
        .map(|(activation, scale)| Bf16::from_f32(activation.to_f32() * scale.to_f32()))
        .collect())
}

/// Computes SiLU in f32 before the profile's bf16 activation cast.
#[must_use]
pub fn silu_f32_cast_back(input: &[Bf16]) -> Vec<Bf16> {
    let values = input
        .iter()
        .map(|value| {
            let widened = value.to_f32();
            widened / (1.0 + (-widened).exp())
        })
        .collect::<Vec<_>>();
    cast_f32_to_bf16(&values)
}

/// Computes the SwiGLU elementwise product after a bf16 SiLU activation.
pub fn swiglu_f32_cast_back(
    gate: &[Bf16],
    up: &[Bf16],
) -> Result<Vec<Bf16>, ReferencePrimitiveError> {
    if gate.len() != up.len() {
        return Err(ReferencePrimitiveError::LengthMismatch {
            operation: "swiglu",
            expected: gate.len(),
            actual: up.len(),
        });
    }
    let activated_gate = silu_f32_cast_back(gate);
    let values = activated_gate
        .iter()
        .zip(up)
        .map(|(left, right)| left.to_f32() * right.to_f32())
        .collect::<Vec<_>>();
    Ok(cast_f32_to_bf16(&values))
}

/// Adds residual activations and narrows the result back to bf16.
pub fn residual_add_f32_cast_back(
    residual: &[Bf16],
    update: &[Bf16],
) -> Result<Vec<Bf16>, ReferencePrimitiveError> {
    if residual.len() != update.len() {
        return Err(ReferencePrimitiveError::LengthMismatch {
            operation: "residual_add",
            expected: residual.len(),
            actual: update.len(),
        });
    }
    let values = residual
        .iter()
        .zip(update)
        .map(|(left, right)| left.to_f32() + right.to_f32())
        .collect::<Vec<_>>();
    Ok(cast_f32_to_bf16(&values))
}
