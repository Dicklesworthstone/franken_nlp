//! Explicit, bounded CPU INT8 kernels; capability is not a dispatch policy.
//!
//! X3a and X3b implement plan 6.5 / beads fms and 3am independently. They
//! consume borrowed row-major bytes, not a newly ratified packing format.
//! Default selection remains scalar. Construction checks the REAL host, not
//! a caller-supplied feature report; the executable handle cannot be forged
//! through Deserialize. No kernel allocates, spawns, reads configuration, or
//! changes the canonical floating-point epilogue.

use std::{error::Error, fmt, str::FromStr};
use super::{int8::{self, Int8KernelError}, quant_algebra::MAX_MODEL_K};

#[cfg(target_arch = "x86_64")]
mod avx2;
pub mod config;
#[cfg(test)]
mod tests;

/// These are implementation identifiers, not measured dispatch winners.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Int8KernelKind { Scalar, Avx2Low7HighBit, Avx2WidenedI16 }
impl Int8KernelKind {
    pub const ALL: [Self; 3] = [Self::Scalar, Self::Avx2Low7HighBit, Self::Avx2WidenedI16];
    pub const fn id(self) -> &'static str {
        match self {
            Self::Scalar => "s_scalar_i32",
            Self::Avx2Low7HighBit => "x3a_avx2_low7_high_bit",
            Self::Avx2WidenedI16 => "x3b_avx2_widened_i16",
        }
    }
}
impl fmt::Display for Int8KernelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.id()) }
}
impl FromStr for Int8KernelKind {
    type Err = CpuKernelError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL.into_iter().find(|kind| kind.id() == value)
            .ok_or(CpuKernelError::UnknownKind)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CpuKernelError {
    UnknownKind,
    UnsupportedIsa,
    LengthMismatch,
    ReductionTooLong,
    Shape,
    Scalar(Int8KernelError),
}
impl fmt::Display for CpuKernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnknownKind => "unknown explicit INT8 CPU kernel",
            Self::UnsupportedIsa => "requested INT8 CPU kernel is not supported by this host",
            Self::LengthMismatch => "INT8 CPU dot operand lengths differ",
            Self::ReductionTooLong => "INT8 CPU reduction exceeds the proved model bound",
            Self::Shape => "INT8 CPU matrix or output geometry differs",
            Self::Scalar(_) => "INT8 scalar reference refused the reduction",
        })
    }
}
impl Error for CpuKernelError {}

#[derive(Clone, Copy, Debug)]
enum Implementation {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2(avx2::Avx2Kernel),
}

/// An immutable, hardware-checked executable handle. It contains no weights,
/// mutable state, or borrowed buffers, and is safe to share across CPU workers.
#[derive(Clone, Copy, Debug)]
pub struct Int8Kernel { implementation: Implementation }
impl Default for Int8Kernel {
    fn default() -> Self { Self::scalar() }
}
impl Int8Kernel {
    pub const fn scalar() -> Self { Self { implementation: Implementation::Scalar } }
    /// Never substitute scalar when an explicit ISA request is unavailable.
    pub fn detect(kind: Int8KernelKind) -> Result<Self, CpuKernelError> {
        if kind == Int8KernelKind::Scalar { return Ok(Self::scalar()); }
        #[cfg(target_arch = "x86_64")]
        { return Ok(Self { implementation: Implementation::Avx2(avx2::Avx2Kernel::detect(kind)?) }); }
        #[cfg(not(target_arch = "x86_64"))]
        { Err(CpuKernelError::UnsupportedIsa) }
    }
    pub fn kind(self) -> Int8KernelKind {
        match self.implementation {
            Implementation::Scalar => Int8KernelKind::Scalar,
            #[cfg(target_arch = "x86_64")]
            Implementation::Avx2(kernel) => kernel.kind(),
        }
    }
    /// Full signed-byte domain, including -128 on BOTH operands. Empty dots
    /// return zero. K is bounded even for scalar, so all handles share one
    /// mathematical contract. The original unbounded scalar API is unchanged.
    pub fn dot(self, input: &[i8], weights: &[i8]) -> Result<i32, CpuKernelError> {
        if input.len() != weights.len() { return Err(CpuKernelError::LengthMismatch); }
        if input.len() > MAX_MODEL_K { return Err(CpuKernelError::ReductionTooLong); }
        match self.implementation {
            Implementation::Scalar => int8::dot_s8s8(input, weights).map_err(CpuKernelError::Scalar),
            #[cfg(target_arch = "x86_64")]
            Implementation::Avx2(kernel) => kernel.dot(input, weights),
        }
    }
    /// A(M,K) times transpose(W(N,K)) -> row-major C(M,N). All shape and
    /// reduction checks precede the first output write; no intermediate GEMV
    /// result, packed copy or float expansion is allocated. Orchestration owns
    /// tiling/cancellation: this leaf must receive an admitted finite tile.
    pub fn gemm_into(self, activations: &[i8], m: usize, k: usize, weights: &[i8],
        n: usize, output: &mut [i32]) -> Result<(), CpuKernelError> {
        if k > MAX_MODEL_K { return Err(CpuKernelError::ReductionTooLong); }
        if m.checked_mul(k) != Some(activations.len())
            || n.checked_mul(k) != Some(weights.len())
            || m.checked_mul(n) != Some(output.len()) { return Err(CpuKernelError::Shape); }
        // In particular avoid iterating an enormous zero-width M/N request.
        if output.is_empty() { return Ok(()); }
        for row in 0..m {
            let activation = &activations[row * k..(row + 1) * k];
            for column in 0..n {
                output[row * n + column] = self.dot(activation, &weights[column * k..(column + 1) * k])?;
            }
        }
        Ok(())
    }
}
