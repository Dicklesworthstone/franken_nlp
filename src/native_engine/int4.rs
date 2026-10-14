//! Bounded packed-INT4 execution through the existing hardware-checked INT8 leaves.
//!
//! Weights retain the scalar API's low-nibble-first signed two's-complement
//! layout: exactly two logical weights per byte, with no row padding. Only one
//! K-wide weight row is expanded at a time, into caller-owned scratch. This is
//! an integer accumulation surface, not a new artifact recipe or a claim that
//! an INT4 model has been qualified. Group scales and floating epilogues remain
//! the caller's responsibility; groups with distinct scales must be separate
//! reductions. No allocation, spawn, implicit dispatch or full-matrix expansion.

use super::{
    quant_algebra::MAX_MODEL_K,
    simd::{CpuKernelError, Int8Kernel, Int8KernelKind, MAX_COHORT_ROWS},
};

/// An immutable executable handle. Real host detection, not a supplied feature
/// report, authorizes every non-scalar leaf. Unsupported requests never fall back.
#[derive(Clone, Copy, Debug)]
pub struct PackedInt4Kernel {
    inner: Int8Kernel,
}

impl Default for PackedInt4Kernel {
    fn default() -> Self { Self::scalar() }
}

impl PackedInt4Kernel {
    pub const fn scalar() -> Self { Self { inner: Int8Kernel::scalar() } }

    pub fn detect(kind: Int8KernelKind) -> Result<Self, CpuKernelError> {
        Ok(Self { inner: Int8Kernel::detect(kind)? })
    }

    /// Identifies the actual integer leaf, not a measured packed-INT4 winner.
    pub fn kind(self) -> Int8KernelKind { self.inner.kind() }

    /// Required i8 scratch elements, independent of M and N. Native leaves also
    /// use their existing fixed-size stack tiles. Check before allocating scratch.
    pub fn scratch_len(k: usize) -> Result<usize, CpuKernelError> {
        if k > MAX_MODEL_K { return Err(CpuKernelError::ReductionTooLong); }
        Ok(k)
    }

    /// A(M,K) times transpose(W(N,K)) -> row-major C(M,N). The flat packed
    /// matrix must contain exactly N*K nibbles, so N*K must be even. Odd K is
    /// supported when N is even: adjacent rows can share the same packed byte.
    /// All shape, packed-length, scratch and K checks precede output writes.
    /// Orchestration owns admission and cancellation around this finite leaf.
    pub fn gemm_into(self, activations: &[i8], m: usize, k: usize,
        packed_weights: &[u8], n: usize, output: &mut [i32], scratch: &mut [i8])
        -> Result<(), CpuKernelError> {
        Self::scratch_len(k)?;
        let weights = n.checked_mul(k).ok_or(CpuKernelError::Shape)?;
        if m.checked_mul(k) != Some(activations.len())
            || m.checked_mul(n) != Some(output.len())
            || weights % 2 != 0 || packed_weights.len() != weights / 2
            || scratch.len() < k {
            return Err(CpuKernelError::Shape);
        }
        if output.is_empty() { return Ok(()); }
        if k == 0 { output.fill(0); return Ok(()); }
        let scratch = &mut scratch[..k];
        // Decode each weight row once, then share it across every activation
        // cohort. Neither M nor N grows the unpacking workspace.
        for column in 0..n {
            let offset = column * k;
            for (i, weight) in scratch.iter_mut().enumerate() {
                let index = offset + i;
                let byte = packed_weights[index / 2];
                let nibble = (byte >> ((index % 2) * 4)) & 0x0f;
                *weight = ((nibble as i8) << 4) >> 4;
            }
            for base in (0..m).step_by(MAX_COHORT_ROWS) {
                let rows = (m - base).min(MAX_COHORT_ROWS);
                let mut inputs: [&[i8]; MAX_COHORT_ROWS] = [&[]; MAX_COHORT_ROWS];
                let mut sums = [0_i32; MAX_COHORT_ROWS];
                for row in 0..rows {
                    let start = (base + row) * k;
                    inputs[row] = &activations[start..start + k];
                }
                self.inner.dot_rows_into(&inputs[..rows], scratch, &mut sums[..rows])?;
                for row in 0..rows { output[(base + row) * n + column] = sums[row]; }
            }
        }
        Ok(())
    }

    /// Evaluate one logical nibble range for up to four activation rows.
    /// Group/row boundaries may share a byte, including a high-nibble start.
    /// Only the requested range is unpacked. All operand, output, range and
    /// workspace checks precede scratch or output writes. This is an integer
    /// reduction: callers must scale distinct groups before summing them.
    pub fn dot_rows_range_into(self, inputs: &[&[i8]], packed_weights: &[u8],
        nibble_offset: usize, k: usize, output: &mut [i32], scratch: &mut [i8])
        -> Result<(), CpuKernelError> {
        Self::scratch_len(k)?;
        let total = packed_weights.len().checked_mul(2).ok_or(CpuKernelError::Shape)?;
        let end = nibble_offset.checked_add(k).ok_or(CpuKernelError::Shape)?;
        if end > total || inputs.len() > MAX_COHORT_ROWS || output.len() != inputs.len()
            || scratch.len() < k { return Err(CpuKernelError::Shape); }
        if inputs.iter().any(|input| input.len() != k) { return Err(CpuKernelError::LengthMismatch); }
        if inputs.is_empty() { return Ok(()); }
        for (index, weight) in scratch[..k].iter_mut().enumerate() {
            let at = nibble_offset + index;
            let nibble = (packed_weights[at / 2] >> ((at % 2) * 4)) & 0x0f;
            *weight = ((nibble as i8) << 4) >> 4;
        }
        self.inner.dot_rows_into(inputs, &scratch[..k], output)
    }

    /// GEMV with the same exact packed layout and K-byte scratch contract.
    pub fn gemv_into(self, input: &[i8], packed_weights: &[u8], n: usize,
        output: &mut [i32], scratch: &mut [i8]) -> Result<(), CpuKernelError> {
        self.gemm_into(input, 1, input.len(), packed_weights, n, output, scratch)
    }
}

#[cfg(test)]
mod tests;
