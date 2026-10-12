//! Execution is separate from injected dispatch policy. An untrusted or
//! synthetic DetectedFeatures value must NEVER mint a target-feature handle.
use super::{DispatchError, DispatchKey, DispatchRow, KernelOperation, KernelTier, TileGeometry};
use crate::native_engine::{
    int8,
    quant_algebra::MAX_MODEL_K,
    simd::{CpuKernelError, Int8Kernel, Int8KernelKind},
};

fn native_kind(tier: KernelTier) -> Option<Int8KernelKind> {
    match tier {
        KernelTier::X3aAvx2Low7HighBit => Some(Int8KernelKind::Avx2Low7HighBit),
        KernelTier::X3bAvx2WidenedI16 => Some(Int8KernelKind::Avx2WidenedI16),
        KernelTier::X2AvxVnni => Some(Int8KernelKind::AvxVnni),
        KernelTier::X1aAvx512VnniZmm => Some(Int8KernelKind::Avx512VnniZmm),
        KernelTier::X1bAvx512VnniYmm => Some(Int8KernelKind::Avx512VnniYmm),
        KernelTier::A1Smmla => Some(Int8KernelKind::Aarch64Smmla),
        KernelTier::A2Dotprod => Some(Int8KernelKind::Aarch64Dotprod),
        KernelTier::A3Autovec => Some(Int8KernelKind::Aarch64Autovec),
        KernelTier::Scalar => None,
    }
}
/// Maximum live tile capacity. The corresponding implementation owns every
/// short M/N/K tail, including GEMV's M=1, without padding caller matrices.
pub(super) const fn tile(tier: KernelTier) -> TileGeometry {
    match tier {
        KernelTier::X3aAvx2Low7HighBit | KernelTier::X3bAvx2WidenedI16 => TileGeometry { k: 32, m: 1, n: 1 },
        KernelTier::X2AvxVnni | KernelTier::X1bAvx512VnniYmm => TileGeometry { k: 32, m: 4, n: 1 },
        KernelTier::X1aAvx512VnniZmm => TileGeometry { k: 64, m: 4, n: 1 },
        KernelTier::A1Smmla => TileGeometry { k: 8, m: 4, n: 2 },
        KernelTier::A2Dotprod | KernelTier::A3Autovec => TileGeometry { k: 16, m: 4, n: 1 },
        KernelTier::Scalar => TileGeometry::scalar(),
    }
}
pub(super) fn check_key(tier: KernelTier, key: DispatchKey) -> Result<(), DispatchError> {
    if native_kind(tier).is_some() && key.shape.k > MAX_MODEL_K {
        return Err(DispatchError::CpuKernel(CpuKernelError::ReductionTooLong));
    }
    Ok(())
}
pub(super) fn validate_tile(row: &DispatchRow) -> Result<(), DispatchError> {
    if native_kind(row.selected_tier).is_some()
        && matches!(row.key.operation, KernelOperation::Int8Gemm | KernelOperation::Int8Gemv)
        && row.tile != tile(row.selected_tier) {
        return Err(DispatchError::InvalidMeasurement {
            detail: "native INT8 measurements must name the implemented tile".to_owned(),
        });
    }
    Ok(())
}

/// Allocating convenience entry for an already-admitted finite tile. The
/// native inference path uses caller-owned QuantizedLinear output instead.
/// All operand products and K bounds are checked BEFORE output allocation.
/// A failed hardware check never substitutes scalar or publishes a selection.
pub(super) fn gemm(tier: KernelTier, activations: &[i8], m: usize, k: usize,
    weights: &[i8], n: usize) -> Result<Vec<i32>, DispatchError> {
    if tier == KernelTier::Scalar { return Ok(int8::gemm_s8s8(activations, m, k, weights, n)?); }
    let kind = native_kind(tier).ok_or(DispatchError::ForcedTierUnimplemented { requested: tier })?;
    // Check the REAL process/OS feature surface, not the dispatcher's injected
    // report. This is the only route from policy to an executable CPU handle.
    let kernel = Int8Kernel::detect(kind).map_err(DispatchError::CpuKernel)?;
    if k > MAX_MODEL_K { return Err(DispatchError::CpuKernel(CpuKernelError::ReductionTooLong)); }
    if m.checked_mul(k) != Some(activations.len()) || n.checked_mul(k) != Some(weights.len()) {
        return Err(DispatchError::CpuKernel(CpuKernelError::Shape));
    }
    let length = m.checked_mul(n).ok_or(DispatchError::CpuKernel(CpuKernelError::Shape))?;
    let mut output = Vec::new();
    output.try_reserve_exact(length).map_err(|_| DispatchError::AllocationRefused)?;
    output.resize(length, 0);
    kernel.gemm_into(activations, m, k, weights, n, &mut output).map_err(DispatchError::CpuKernel)?;
    Ok(output)
}
