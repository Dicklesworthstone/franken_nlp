//! Model-specific neural-network primitives for `hf-bf16-eager`.

use super::tensor::{Bf16, cast_f32_to_bf16};
use super::layer::{NANBEIGE_HIDDEN_SIZE, NANBEIGE_INTERMEDIATE_SIZE,
    NANBEIGE_KV_PROJECTION_SIZE, NANBEIGE_Q_PROJECTION_SIZE};

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

/// Individually implemented cast sites recorded for the HF eager profile.
///
/// This enum is a code-first regression inventory, not the OQ-30 cast map.
/// It does not enumerate every per-block boundary and therefore cannot state
/// a complete temporal ordering for the model forward.  In particular, linear
/// outputs, residuals, SiLU/SwiGLU, RMSNorm scale multiplication, and further
/// attention reduction boundaries require their own modeled entries before
/// OQ-30 can become authoritative.
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

/// Non-exhaustive inventory of the cast sites presently represented in code.
///
/// Array position is not an execution-order assertion.  The full ordered
/// per-block cast graph remains an explicit OQ-30 deliverable.
pub const HF_BF16_EAGER_OBSERVED_CAST_SITES: [HfBf16EagerCastSite; 7] = [
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

/// Caller-owned RMSNorm output with the same f32 reduction and two bf16 cast
/// sites as the independent reference above. All shapes reject before a write.
pub(crate) fn rms_norm_f32_reduce_cast_back_into(
    input: &[Bf16], weight: &[Bf16], epsilon: f32, output: &mut [Bf16],
) -> Result<(), ReferencePrimitiveError> {
    if input.is_empty() {
        return Err(ReferencePrimitiveError::EmptyActivation { operation: "rms_norm" });
    }
    if input.len() != weight.len() {
        return Err(ReferencePrimitiveError::LengthMismatch {
            operation: "rms_norm", expected: input.len(), actual: weight.len(),
        });
    }
    if input.len() != output.len() {
        return Err(ReferencePrimitiveError::LengthMismatch {
            operation: "rms_norm_output", expected: input.len(), actual: output.len(),
        });
    }
    let mean_square = input.iter().map(|value| {
        let widened = value.to_f32(); widened * widened
    }).sum::<f32>() / input.len() as f32;
    let inverse_rms = (mean_square + epsilon).sqrt().recip();
    for ((destination, activation), scale) in output.iter_mut().zip(input).zip(weight) {
        let normalized = Bf16::from_f32(activation.to_f32() * inverse_rms);
        *destination = Bf16::from_f32(normalized.to_f32() * scale.to_f32());
    }
    Ok(())
}

/// Reuse the private gate rail after its projection. The SiLU activation is
/// still rounded to bf16 before multiplication by the separate up rail.
pub(crate) fn swiglu_f32_cast_back_in_place(
    gate: &mut [Bf16], up: &[Bf16],
) -> Result<(), ReferencePrimitiveError> {
    if gate.len() != up.len() {
        return Err(ReferencePrimitiveError::LengthMismatch {
            operation: "swiglu", expected: gate.len(), actual: up.len(),
        });
    }
    for (destination, up) in gate.iter_mut().zip(up) {
        let widened = destination.to_f32();
        let activated = Bf16::from_f32(widened / (1.0 + (-widened).exp()));
        *destination = Bf16::from_f32(activated.to_f32() * up.to_f32());
    }
    Ok(())
}

/// Reuse the private hidden rail after its residual input has been consumed.
pub(crate) fn residual_add_f32_cast_back_in_place(
    residual: &mut [Bf16], update: &[Bf16],
) -> Result<(), ReferencePrimitiveError> {
    if residual.len() != update.len() {
        return Err(ReferencePrimitiveError::LengthMismatch {
            operation: "residual_add", expected: residual.len(), actual: update.len(),
        });
    }
    for (destination, update) in residual.iter_mut().zip(update) {
        *destination = Bf16::from_f32(destination.to_f32() + update.to_f32());
    }
    Ok(())
}

/// Reusable single-row decoder rails, allocated only by strict-engine
/// construction after its complete context/scratch quote has been admitted.
/// Every projection overwrites its full destination before that rail is read.
/// Batch/prefill workspaces and caller-owned output logits remain separate.
pub(crate) struct DecoderWorkspace {
    pub normalized: Vec<Bf16>,
    pub query: Vec<Bf16>,
    pub key: Vec<Bf16>,
    pub value: Vec<Bf16>,
    pub update: Vec<Bf16>,
    pub gate: Vec<Bf16>,
    pub up: Vec<Bf16>,
    pub attention: Vec<f32>,
}

impl DecoderWorkspace {
    pub fn try_new(context: usize) -> Result<Self, std::collections::TryReserveError> {
        fn filled<T: Clone>(length: usize, value: T) -> Result<Vec<T>, std::collections::TryReserveError> {
            let mut values = Vec::new();
            values.try_reserve_exact(length)?;
            values.resize(length, value);
            Ok(values)
        }
        let zero = Bf16::from_bits(0);
        Ok(Self {
            normalized: filled(NANBEIGE_HIDDEN_SIZE, zero)?,
            query: filled(NANBEIGE_Q_PROJECTION_SIZE, zero)?,
            key: filled(NANBEIGE_KV_PROJECTION_SIZE, zero)?,
            value: filled(NANBEIGE_KV_PROJECTION_SIZE, zero)?,
            update: filled(NANBEIGE_HIDDEN_SIZE, zero)?,
            gate: filled(NANBEIGE_INTERMEDIATE_SIZE, zero)?,
            up: filled(NANBEIGE_INTERMEDIATE_SIZE, zero)?,
            attention: filled(context, 0.0)?,
        })
    }
}

#[cfg(test)]
mod workspace_tests {
    use super::*;

    #[test]
    fn caller_owned_rails_match_independent_cast_programs_bit_for_bit() {
        let bits = [0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x3f7f, 0x3f80,
            0x3f81, 0xbf80, 0x3c01, 0xbc01, 0x4181, 0xc181, 0x7f7f, 0xff7f,
            0x7f80, 0xff80, 0x7fc1];
        for length in [1, 3, 7, 31, 128, NANBEIGE_HIDDEN_SIZE] {
            let input: Vec<_> = (0..length).map(|i| Bf16::from_bits(bits[i % bits.len()])).collect();
            let other: Vec<_> = (0..length).map(|i| Bf16::from_bits(bits[(i * 7 + 3) % bits.len()])).collect();
            let mut normalized = vec![Bf16::from_bits(0xffff); length];
            rms_norm_f32_reduce_cast_back_into(&input, &other, RMS_NORM_EPSILON, &mut normalized).unwrap();
            assert_bits(&normalized, &rms_norm_f32_reduce_cast_back(&input, &other, RMS_NORM_EPSILON).unwrap());
            let mut product = input.clone();
            swiglu_f32_cast_back_in_place(&mut product, &other).unwrap();
            assert_bits(&product, &swiglu_f32_cast_back(&input, &other).unwrap());
            let mut residual = input.clone();
            residual_add_f32_cast_back_in_place(&mut residual, &other).unwrap();
            assert_bits(&residual, &residual_add_f32_cast_back(&input, &other).unwrap());
        }
        // Finite rows exercise nonzero norms instead of only overflow/NaN paths.
        let mut workspace = DecoderWorkspace::try_new(1).unwrap();
        let pointer = workspace.normalized.as_ptr();
        for round in 0..7 {
            let input: Vec<_> = (0..NANBEIGE_HIDDEN_SIZE).map(|i|
                Bf16::from_f32(((i * 13 + round * 3) % 127) as f32 * 0.03125 - 1.5)).collect();
            let scales: Vec<_> = (0..input.len()).map(|i|
                Bf16::from_f32(((i * 7 + round) % 17) as f32 * 0.0625)).collect();
            workspace.normalized.fill(Bf16::from_bits(0x7fc1));
            rms_norm_f32_reduce_cast_back_into(&input, &scales, RMS_NORM_EPSILON, &mut workspace.normalized).unwrap();
            assert_bits(&workspace.normalized, &rms_norm_f32_reduce_cast_back(&input, &scales, RMS_NORM_EPSILON).unwrap());
            assert_eq!(workspace.normalized.as_ptr(), pointer);
        }
    }

    #[test]
    fn shape_refusals_do_not_partially_overwrite_a_reused_rail() {
        let input = [Bf16::from_f32(1.0); 3];
        let mut output = [Bf16::from_bits(0x4242); 3];
        let before = output;
        assert!(rms_norm_f32_reduce_cast_back_into(&input, &input[..2], RMS_NORM_EPSILON, &mut output).is_err());
        assert_eq!(output, before);
        assert!(rms_norm_f32_reduce_cast_back_into(&input[..2], &input[..2], RMS_NORM_EPSILON, &mut output).is_err());
        assert_eq!(output, before);
        assert!(swiglu_f32_cast_back_in_place(&mut output, &input[..2]).is_err());
        assert_eq!(output, before);
        assert!(residual_add_f32_cast_back_in_place(&mut output, &input[..2]).is_err());
        assert_eq!(output, before);
        assert!(rms_norm_f32_reduce_cast_back_into(&[], &[], RMS_NORM_EPSILON, &mut []).is_err());
    }

    #[test]
    fn persistent_workspace_fits_both_existing_admitted_scratch_quotes() {
        for context in [1, 17, 8192] {
            let workspace = DecoderWorkspace::try_new(context).unwrap();
            let bf16_bytes: usize = [&workspace.normalized, &workspace.query, &workspace.key,
                &workspace.value, &workspace.update, &workspace.gate, &workspace.up]
                .iter().map(|rail| rail.len() * size_of::<Bf16>()).sum();
            let retained = (bf16_bytes + workspace.attention.len() * size_of::<f32>()) as u64;
            assert_eq!(retained, 71_680 + context as u64 * 4);
            let one_hidden_and_head = (NANBEIGE_HIDDEN_SIZE * size_of::<Bf16>()
                + crate::native_engine::lmhead::NANBEIGE_VOCAB_SIZE * size_of::<f32>()) as u64;
            let int8 = crate::native_engine::strict_int8::Int8MemoryRequirement::for_context(context).unwrap();
            assert!(retained + one_hidden_and_head < int8.scratch_payload_bound);
            for group in [32, 64, 128] {
                let int4 = crate::native_engine::strict_int4::Int4MemoryRequirement::for_context(context, group).unwrap();
                assert!(retained + one_hidden_and_head < int4.scratch_payload_bound);
            }
        }
    }

    fn assert_bits(actual: &[Bf16], expected: &[Bf16]) {
        assert_eq!(actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }
}
