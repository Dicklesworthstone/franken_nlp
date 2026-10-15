//! Borrowed rectangular tiles for selected vocabulary rows and model linears.
//! No contiguous scratch matrix, repacking, allocation, or new ISA body.
use super::{CpuKernelError, Int8Kernel, Int8KernelKind, MAX_COHORT_ROWS, MAX_MODEL_K};
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use super::Implementation;

/// Maximum weight rows accepted by the borrowed tile API.
pub const MAX_TILE_COLUMNS: usize = 2;
/// Fixed local integer scratch; caller output is compact, not padded to this size.
pub const MAX_TILE_VALUES: usize = MAX_COHORT_ROWS * MAX_TILE_COLUMNS;

impl Int8Kernel {
    /// Columns evaluated together by the selected native body. This is its
    /// implemented geometry, not an automatic choice or a performance claim.
    pub fn tile_columns(self) -> usize {
        match self.kind() {
            Int8KernelKind::Aarch64Smmla => MAX_TILE_COLUMNS,
            _ => 1,
        }
    }

    /// A(M,K) times transpose(W(N,K)) with separately borrowed rows. M=1..4,
    /// N=1..2; K may be zero. Output is compact row-major M*N, including N=1.
    /// All rows may be unaligned/noncontiguous, repeated, or read-only aliases.
    /// The entire tile is checked and computed before caller output is changed.
    ///
    /// SMMLA uses the existing checked two-column body with both real weight
    /// rows. Other handles execute their own shared-weight leaf per column,
    /// never scalar substitution. No inactive row is read or projected.
    pub fn dot_tile_into(self, inputs: &[&[i8]], weights: &[&[i8]], output: &mut [i32])
        -> Result<(), CpuKernelError> {
        if inputs.is_empty() || inputs.len() > MAX_COHORT_ROWS
            || weights.is_empty() || weights.len() > MAX_TILE_COLUMNS
            || inputs.len() * weights.len() != output.len() {
            return Err(CpuKernelError::Shape);
        }
        let k = weights[0].len();
        if k > MAX_MODEL_K { return Err(CpuKernelError::ReductionTooLong); }
        if inputs.iter().chain(weights.iter()).any(|row| row.len() != k) {
            return Err(CpuKernelError::LengthMismatch);
        }
        let mut result = [0_i32; MAX_TILE_VALUES];
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        if let Implementation::Smmla(kernel) = self.implementation {
            let padded = kernel.tile(inputs, weights)?;
            for row in 0..inputs.len() {
                for column in 0..weights.len() {
                    result[row * weights.len() + column] = padded[row * MAX_TILE_COLUMNS + column];
                }
            }
            output.copy_from_slice(&result[..output.len()]);
            return Ok(());
        }
        let mut column_output = [0_i32; MAX_COHORT_ROWS];
        for (column, weight) in weights.iter().enumerate() {
            self.dot_rows_into(inputs, weight, &mut column_output[..inputs.len()])?;
            for row in 0..inputs.len() {
                result[row * weights.len() + column] = column_output[row];
            }
        }
        output.copy_from_slice(&result[..output.len()]);
        Ok(())
    }
}
