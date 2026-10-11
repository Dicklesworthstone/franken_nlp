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
        _ => None,
    }
}
pub(super) const fn tile(tier: KernelTier) -> TileGeometry {
    match tier {
        KernelTier::X3aAvx2Low7HighBit | KernelTier::X3bAvx2WidenedI16 => TileGeometry { k: 32, m: 1, n: 1 },
        _ => TileGeometry::scalar(),
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
            detail: "AVX2 row-major INT8 measurements must name the implemented 1x1x32 tile".to_owned(),
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
