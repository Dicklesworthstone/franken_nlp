//! Explicit, bounded CPU INT8 kernels; capability is not a dispatch policy.
//!
//! CPU leaves consume borrowed row-major bytes, not a newly ratified packing
//! format. Scalar remains the default. Construction checks the REAL host, not
//! a caller-supplied feature report; executable handles cannot be forged with
//! Deserialize. No kernel allocates, spawns, reads configuration, or changes
//! the canonical floating-point epilogue. ARM A1/A2/A3 are independent bodies.

use std::{error::Error, fmt, str::FromStr};
use super::{int8::{self, Int8KernelError}, quant_algebra::MAX_MODEL_K};

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod smmla;
#[cfg(any(target_arch = "aarch64", test))]
mod autovec;
pub mod config;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod arm_tests;

/// Maximum simultaneously accumulated activation rows in a shared-weight leaf.
pub const MAX_COHORT_ROWS: usize = 4;

/// These are implementation identifiers, not measured dispatch winners.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Int8KernelKind {
    Scalar, Avx2Low7HighBit, Avx2WidenedI16, Aarch64Dotprod, Aarch64Autovec, Aarch64Smmla,
}
impl Int8KernelKind {
    pub const ALL: [Self; 6] = [Self::Scalar, Self::Avx2Low7HighBit, Self::Avx2WidenedI16,
        Self::Aarch64Dotprod, Self::Aarch64Autovec, Self::Aarch64Smmla];
    pub const fn id(self) -> &'static str {
        match self {
            Self::Scalar => "s_scalar_i32",
            Self::Avx2Low7HighBit => "x3a_avx2_low7_high_bit",
            Self::Avx2WidenedI16 => "x3b_avx2_widened_i16",
            Self::Aarch64Dotprod => "a2_dotprod",
            Self::Aarch64Autovec => "a3_autovec",
            Self::Aarch64Smmla => "a1_smmla",
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
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    Dotprod(neon::DotprodKernel),
    #[cfg(target_arch = "aarch64")]
    Autovec,
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    Smmla(smmla::SmmlaKernel),
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
    /// Never substitute another tier for an unavailable explicit request.
    /// ARM intrinsic bodies are compiled only on little-endian targets.
    pub fn detect(kind: Int8KernelKind) -> Result<Self, CpuKernelError> {
        match kind {
            Int8KernelKind::Scalar => Ok(Self::scalar()),
            #[cfg(target_arch = "x86_64")]
            Int8KernelKind::Avx2Low7HighBit | Int8KernelKind::Avx2WidenedI16 =>
                Ok(Self { implementation: Implementation::Avx2(avx2::Avx2Kernel::detect(kind)?) }),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Int8KernelKind::Aarch64Dotprod =>
                Ok(Self { implementation: Implementation::Dotprod(neon::DotprodKernel::detect()?) }),
            #[cfg(target_arch = "aarch64")]
            Int8KernelKind::Aarch64Autovec => Ok(Self { implementation: Implementation::Autovec }),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Int8KernelKind::Aarch64Smmla =>
                Ok(Self { implementation: Implementation::Smmla(smmla::SmmlaKernel::detect()?) }),
            _ => Err(CpuKernelError::UnsupportedIsa),
        }
    }
    pub fn kind(self) -> Int8KernelKind {
        match self.implementation {
            Implementation::Scalar => Int8KernelKind::Scalar,
            #[cfg(target_arch = "x86_64")]
            Implementation::Avx2(kernel) => kernel.kind(),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Implementation::Dotprod(_) => Int8KernelKind::Aarch64Dotprod,
            #[cfg(target_arch = "aarch64")]
            Implementation::Autovec => Int8KernelKind::Aarch64Autovec,
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Implementation::Smmla(_) => Int8KernelKind::Aarch64Smmla,
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
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Implementation::Dotprod(kernel) => {
                let mut output = [0]; kernel.dot_rows_into(&[input], weights, &mut output)?; Ok(output[0])
            }
            #[cfg(target_arch = "aarch64")]
            Implementation::Autovec => {
                let mut output = [0]; autovec::dot_rows_into(&[input], weights, &mut output)?; Ok(output[0])
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Implementation::Smmla(kernel) => {
                let mut output = [0]; kernel.dot_rows_into(&[input], weights, &mut output)?; Ok(output[0])
            }
        }
    }
    /// Evaluate up to four independently borrowed activation rows against ONE
    /// weight row. Vector leaves share weight blocks across the whole cohort.
    /// Rows need not be contiguous or aligned and may alias each other/weights.
    /// Check ALL lengths before any output write; K remains independently
    /// bounded for each accumulator, never one reduction across documents.
    pub fn dot_rows_into(self, inputs: &[&[i8]], weights: &[i8], output: &mut [i32])
        -> Result<(), CpuKernelError> {
        check_rows(inputs, weights, output.len())?;
        match self.implementation {
            Implementation::Scalar => {
                for (input, value) in inputs.iter().zip(output) {
                    *value = int8::dot_s8s8(input, weights).map_err(CpuKernelError::Scalar)?;
                }
                Ok(())
            }
            #[cfg(target_arch = "x86_64")]
            Implementation::Avx2(kernel) => kernel.dot_rows_into(inputs, weights, output),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Implementation::Dotprod(kernel) => kernel.dot_rows_into(inputs, weights, output),
            #[cfg(target_arch = "aarch64")]
            Implementation::Autovec => autovec::dot_rows_into(inputs, weights, output),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Implementation::Smmla(kernel) => kernel.dot_rows_into(inputs, weights, output),
        }
    }
    /// A(M,K) times transpose(W(N,K)) -> row-major C(M,N). All shape and
    /// reduction checks precede the first output write. ARM SDOT/autovec use
    /// four-row shared-weight tiles; SMMLA uses four-row/two-column tiles with
    /// register-only operand assembly. No packed matrix or scratch allocation.
    /// Orchestration owns admission and cancellation for this finite leaf.
    pub fn gemm_into(self, activations: &[i8], m: usize, k: usize, weights: &[i8],
        n: usize, output: &mut [i32]) -> Result<(), CpuKernelError> {
        check_matrix(activations.len(), m, k, weights.len(), n, output.len())?;
        if output.is_empty() { return Ok(()); }
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        if let Implementation::Smmla(kernel) = self.implementation {
            return kernel.gemm_into(activations, m, k, weights, n, output);
        }
        if matches!(self.kind(), Int8KernelKind::Aarch64Dotprod | Int8KernelKind::Aarch64Autovec) {
            return self.gemm_rows_into(activations, m, k, weights, n, output);
        }
        // Preserve the scalar and AVX2 convenience route and its reported tile.
        for row in 0..m {
            let activation = &activations[row * k..(row + 1) * k];
            for column in 0..n {
                output[row * n + column] = self.dot(activation, &weights[column * k..(column + 1) * k])?;
            }
        }
        Ok(())
    }
    fn gemm_rows_into(self, activations: &[i8], m: usize, k: usize, weights: &[i8],
        n: usize, output: &mut [i32]) -> Result<(), CpuKernelError> {
        // Called only after all matrix products, K and output length are checked.
        for column in 0..n {
            let w = &weights[column * k..(column + 1) * k];
            for base in (0..m).step_by(MAX_COHORT_ROWS) {
                let rows = (m - base).min(MAX_COHORT_ROWS);
                let mut inputs: [&[i8]; MAX_COHORT_ROWS] = [&[]; MAX_COHORT_ROWS];
                let mut sums = [0_i32; MAX_COHORT_ROWS];
                for row in 0..rows { inputs[row] = &activations[(base + row) * k..(base + row + 1) * k]; }
                self.dot_rows_into(&inputs[..rows], w, &mut sums[..rows])?;
                for row in 0..rows { output[(base + row) * n + column] = sums[row]; }
            }
        }
        Ok(())
    }
}

/// Shared safe boundary, also called by each ARM leaf before target entry.
fn check_rows(inputs: &[&[i8]], weights: &[i8], output_len: usize) -> Result<(), CpuKernelError> {
    if inputs.len() > MAX_COHORT_ROWS || inputs.len() != output_len { return Err(CpuKernelError::Shape); }
    if weights.len() > MAX_MODEL_K { return Err(CpuKernelError::ReductionTooLong); }
    if inputs.iter().any(|input| input.len() != weights.len()) { return Err(CpuKernelError::LengthMismatch); }
    Ok(())
}
fn check_matrix(input_len: usize, m: usize, k: usize, weights_len: usize, n: usize,
    output_len: usize) -> Result<(), CpuKernelError> {
    if k > MAX_MODEL_K { return Err(CpuKernelError::ReductionTooLong); }
    if m.checked_mul(k) != Some(input_len) || n.checked_mul(k) != Some(weights_len)
        || m.checked_mul(n) != Some(output_len) { return Err(CpuKernelError::Shape); }
    Ok(())
}
