//! A3: independent, safe Rust i32 MAC loops exposed to LLVM vectorization.
//! No intrinsics, unsafe code, float intermediates or scalar-oracle call-through.
//! Sixteen independent K lanes remove the loop-carried single-sum dependency.
//! Actual vector instructions/performance must be inspected on the pinned build;
//! the implementation name is not evidence that LLVM vectorized a given binary.

use super::{CpuKernelError, MAX_COHORT_ROWS, check_rows};

pub(super) fn dot_rows_into(inputs: &[&[i8]], weights: &[i8], output: &mut [i32])
    -> Result<(), CpuKernelError> {
    check_rows(inputs, weights, output.len())?;
    if inputs.is_empty() { return Ok(()); }
    let sums = match inputs.len() {
        1 => accumulate::<1>(inputs, weights),
        2 => accumulate::<2>(inputs, weights),
        3 => accumulate::<3>(inputs, weights),
        4 => accumulate::<4>(inputs, weights),
        _ => unreachable!("checked autovec cohort"),
    };
    output.copy_from_slice(&sums[..inputs.len()]);
    Ok(())
}

fn accumulate<const M: usize>(inputs: &[&[i8]], weights: &[i8]) -> [i32; MAX_COHORT_ROWS] {
    let end = weights.len() / 16 * 16;
    let mut lanes = [[0_i32; 16]; M];
    for start in (0..end).step_by(16) {
        let w = &weights[start..start + 16];
        for row in 0..M {
            let x = &inputs[row][start..start + 16];
            // K <= MAX_MODEL_K bounds every partial by 176160768. No wrapping
            // or saturating arithmetic is needed, even for (-128)*(-128).
            for lane in 0..16 { lanes[row][lane] += i32::from(x[lane]) * i32::from(w[lane]); }
        }
    }
    let mut output = [0_i32; MAX_COHORT_ROWS];
    for row in 0..M {
        for value in lanes[row] { output[row] += value; }
    }
    for index in end..weights.len() {
        let w = i32::from(weights[index]);
        for row in 0..M { output[row] += i32::from(inputs[row][index]) * w; }
    }
    output
}
