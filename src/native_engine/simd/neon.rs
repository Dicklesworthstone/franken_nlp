//! A2 signed SDOT over borrowed rows, including shared-weight M=1..4 cohorts.
//! No offset-domain correction, saturation, speculative ISA probe or allocation.
//! This is a Generic-row candidate, not offline-panel qualification or a speed
//! claim. Runtime capability is checked independently of the dispatch registry.
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::arch::aarch64::*;
use super::{CpuKernelError, MAX_COHORT_ROWS, check_rows};

#[derive(Clone, Copy, Debug)]
pub(super) struct DotprodKernel { _private: () }
impl DotprodKernel {
    pub(super) fn detect() -> Result<Self, CpuKernelError> {
        if !std::arch::is_aarch64_feature_detected!("neon")
            || !std::arch::is_aarch64_feature_detected!("dotprod") {
            return Err(CpuKernelError::UnsupportedIsa);
        }
        Ok(Self { _private: () })
    }

    pub(super) fn dot_rows_into(self, inputs: &[&[i8]], weights: &[i8], output: &mut [i32])
        -> Result<(), CpuKernelError> {
        // Check every row before target entry or any caller-visible write.
        check_rows(inputs, weights, output.len())?;
        if inputs.is_empty() { return Ok(()); }
        // SAFETY: the unforgeable private handle proves real OS-aware neon and
        // dotprod support. check_rows proved equal lengths and K <= MAX_MODEL_K.
        // Each closed branch supplies exactly M valid slices, with M in 1..=4.
        let sums = unsafe {
            match inputs.len() {
                1 => dotprod_rows::<1>(inputs, weights),
                2 => dotprod_rows::<2>(inputs, weights),
                3 => dotprod_rows::<3>(inputs, weights),
                4 => dotprod_rows::<4>(inputs, weights),
                _ => unreachable!("checked ARM cohort"),
            }
        };
        output.copy_from_slice(&sums[..inputs.len()]);
        Ok(())
    }
}

// SDOT sums four signed-byte products into each i32 lane. The absolute bound
// for every partial, every lane and the final dot is K*16384 <= 176160768.
// Reordering exact integer additions therefore cannot change the scalar result.
// Unlike an unsigned-byte pair instruction, -128 needs no exceptional handling.
#[target_feature(enable = "neon,dotprod")]
unsafe fn dotprod_rows<const M: usize>(inputs: &[&[i8]], weights: &[i8]) -> [i32; MAX_COHORT_ROWS] {
    let end = weights.len() / 16 * 16;
    // SAFETY: the caller proved the ISA, exactly M rows and their bounded equal
    // lengths. Only complete [start,start+16) blocks are loaded; vld1q requires
    // no additional alignment. vst1q writes exactly four i32 values into a local
    // four-i32 array. Short tails use checked indexing, never an oversized load.
    unsafe {
        let mut accumulators = [vdupq_n_s32(0); M];
        for start in (0..end).step_by(16) {
            let w = vld1q_s8(weights.as_ptr().add(start));
            for row in 0..M {
                let x = vld1q_s8(inputs[row].as_ptr().add(start));
                accumulators[row] = vdotq_s32(accumulators[row], x, w);
            }
        }
        let mut output = [0_i32; MAX_COHORT_ROWS];
        for row in 0..M {
            let mut lanes = [0_i32; 4];
            vst1q_s32(lanes.as_mut_ptr(), accumulators[row]);
            // Fixed ascending lane order followed by ascending logical K tail.
            for value in lanes { output[row] += value; }
        }
        for index in end..weights.len() {
            let w = i32::from(weights[index]);
            for row in 0..M { output[row] += i32::from(inputs[row][index]) * w; }
        }
        output
    }
}
