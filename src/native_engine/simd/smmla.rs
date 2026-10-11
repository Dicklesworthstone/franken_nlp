//! A1 signed SMMLA with register-assembled 4x2x8 tiles over Generic rows.
//! Full matrices are never repacked, expanded or cloned. This is not the
//! separately qualified offline packing/cache-panel path or a benchmark award.
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::arch::aarch64::*;
use super::{CpuKernelError, MAX_COHORT_ROWS, check_matrix, check_rows};

const TILE_COLUMNS: usize = 2;
const TILE_VALUES: usize = MAX_COHORT_ROWS * TILE_COLUMNS;

#[derive(Clone, Copy, Debug)]
pub(super) struct SmmlaKernel { _private: () }
impl SmmlaKernel {
    pub(super) fn detect() -> Result<Self, CpuKernelError> {
        if !std::arch::is_aarch64_feature_detected!("neon")
            || !std::arch::is_aarch64_feature_detected!("i8mm") {
            return Err(CpuKernelError::UnsupportedIsa);
        }
        Ok(Self { _private: () })
    }

    pub(super) fn dot_rows_into(self, inputs: &[&[i8]], weights: &[i8], output: &mut [i32])
        -> Result<(), CpuKernelError> {
        check_rows(inputs, weights, output.len())?;
        if inputs.is_empty() { return Ok(()); }
        let values = self.tile(inputs, &[weights])?;
        for row in 0..inputs.len() { output[row] = values[row * TILE_COLUMNS]; }
        Ok(())
    }

    pub(super) fn gemm_into(self, activations: &[i8], m: usize, k: usize, weights: &[i8],
        n: usize, output: &mut [i32]) -> Result<(), CpuKernelError> {
        check_matrix(activations.len(), m, k, weights.len(), n, output.len())?;
        if output.is_empty() { return Ok(()); }
        // Keep a pair of weight rows live across the activation-row sweep.
        // Packing is only register assembly inside a tile, not a heap buffer.
        for column_base in (0..n).step_by(TILE_COLUMNS) {
            let columns = (n - column_base).min(TILE_COLUMNS);
            let mut weight_rows: [&[i8]; TILE_COLUMNS] = [&[]; TILE_COLUMNS];
            for column in 0..columns {
                weight_rows[column] = &weights[(column_base + column) * k..(column_base + column + 1) * k];
            }
            for row_base in (0..m).step_by(MAX_COHORT_ROWS) {
                let rows = (m - row_base).min(MAX_COHORT_ROWS);
                let mut inputs: [&[i8]; MAX_COHORT_ROWS] = [&[]; MAX_COHORT_ROWS];
                for row in 0..rows {
                    inputs[row] = &activations[(row_base + row) * k..(row_base + row + 1) * k];
                }
                let values = self.tile(&inputs[..rows], &weight_rows[..columns])?;
                for row in 0..rows {
                    for column in 0..columns {
                        output[(row_base + row) * n + column_base + column] = values[row * TILE_COLUMNS + column];
                    }
                }
            }
        }
        Ok(())
    }

    fn tile(self, inputs: &[&[i8]], weights: &[&[i8]]) -> Result<[i32; TILE_VALUES], CpuKernelError> {
        if inputs.is_empty() || weights.is_empty() || weights.len() > TILE_COLUMNS {
            return Err(CpuKernelError::Shape);
        }
        check_rows(inputs, weights[0], inputs.len())?;
        if weights.iter().any(|row| row.len() != weights[0].len()) { return Err(CpuKernelError::LengthMismatch); }
        // SAFETY: the private handle proves actual OS-advertised neon+i8mm.
        // Every input and weight row has the same checked K <= MAX_MODEL_K.
        // The closed match passes exactly M=1..4 input and N=1..2 weight rows.
        let result = unsafe {
            match (inputs.len(), weights.len()) {
                (1, 1) => smmla_tile::<1, 1>(inputs, weights),
                (1, 2) => smmla_tile::<1, 2>(inputs, weights),
                (2, 1) => smmla_tile::<2, 1>(inputs, weights),
                (2, 2) => smmla_tile::<2, 2>(inputs, weights),
                (3, 1) => smmla_tile::<3, 1>(inputs, weights),
                (3, 2) => smmla_tile::<3, 2>(inputs, weights),
                (4, 1) => smmla_tile::<4, 1>(inputs, weights),
                (4, 2) => smmla_tile::<4, 2>(inputs, weights),
                _ => unreachable!("checked SMMLA tile"),
            }
        };
        Ok(result)
    }
}

// SMMLA operand contract, independently of SDOT/AVX2:
//   LHS bytes [A(row0,K..K+8), A(row1,K..K+8)] are 2x8 row-major.
//   RHS bytes [W(col0,K..K+8), W(col1,K..K+8)] are 8x2 column-major.
//   Result i32 lanes are [r0c0, r0c1, r1c0, r1c1].
// Two independent accumulators cover four rows; every i32 lane is a complete
// output element, NOT a horizontal reduction or a sum across documents.
// Each update has magnitude <= 8*16384, and every full lane/partial is bounded
// by K*16384 <= 176160768. Inactive row/column halves are actual zero vectors.
#[target_feature(enable = "neon,i8mm")]
unsafe fn smmla_tile<const M: usize, const N: usize>(inputs: &[&[i8]], weights: &[&[i8]])
    -> [i32; TILE_VALUES] {
    let k = weights[0].len();
    let end = k / 8 * 8;
    // SAFETY: caller proves the ISA, M=1..4, N=1..2 and bounded equal lengths.
    // Every vld1_s8 reads exactly one contained [start,start+8) block, without
    // alignment assumptions. Missing row/column halves use zero and are NEVER
    // loaded. At most two vst1q stores write four i32 each within the local
    // eight-i32 output. Scalar tails use checked indexing on live rows only.
    unsafe {
        let zero_bytes = vdup_n_s8(0);
        let mut accumulators = [vdupq_n_s32(0); 2];
        for start in (0..end).step_by(8) {
            let w0 = vld1_s8(weights[0].as_ptr().add(start));
            let w1 = if N == 2 { vld1_s8(weights[1].as_ptr().add(start)) } else { zero_bytes };
            let rhs = vcombine_s8(w0, w1);
            for pair in 0..(M + 1) / 2 {
                let row = pair * 2;
                let a0 = vld1_s8(inputs[row].as_ptr().add(start));
                let a1 = if row + 1 < M { vld1_s8(inputs[row + 1].as_ptr().add(start)) } else { zero_bytes };
                let lhs = vcombine_s8(a0, a1);
                accumulators[pair] = vmmlaq_s32(accumulators[pair], lhs, rhs);
            }
        }
        let mut output = [0_i32; TILE_VALUES];
        for pair in 0..(M + 1) / 2 {
            vst1q_s32(output.as_mut_ptr().add(pair * 4), accumulators[pair]);
        }
        for index in end..k {
            for column in 0..N {
                let w = i32::from(weights[column][index]);
                for row in 0..M { output[row * TILE_COLUMNS + column] += i32::from(inputs[row][index]) * w; }
            }
        }
        output
    }
}
