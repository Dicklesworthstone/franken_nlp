//! Enumerated AVX2 island. No public unsafe API or speculative ISA execution.
//!
//! The only constructor checks std's actual OS-aware AVX2 detection. Both
//! slices are rechecked before entering target-feature code. Complete 32-byte
//! blocks use unaligned loads; the remainder uses scalar indexing, never a
//! masked or oversized load. K <= 10752 bounds EVERY partial and final i32 sum.
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;
use super::{CpuKernelError, Int8KernelKind, MAX_COHORT_ROWS, MAX_MODEL_K};

// The executable island holds only its two actual implementations. Expanding
// the public cross-architecture catalog cannot create an invalid AVX2 state or
// leave a target-function match non-exhaustive on x86 builds.
#[derive(Clone, Copy, Debug)]
enum Avx2Implementation { Low7HighBit, WidenedI16 }
#[derive(Clone, Copy, Debug)]
pub(super) struct Avx2Kernel { implementation: Avx2Implementation }
impl Avx2Kernel {
    pub(super) fn detect(kind: Int8KernelKind) -> Result<Self, CpuKernelError> {
        let implementation = match kind {
            Int8KernelKind::Avx2Low7HighBit => Avx2Implementation::Low7HighBit,
            Int8KernelKind::Avx2WidenedI16 => Avx2Implementation::WidenedI16,
            _ => return Err(CpuKernelError::UnknownKind),
        };
        if !std::arch::is_x86_feature_detected!("avx2") { return Err(CpuKernelError::UnsupportedIsa); }
        Ok(Self { implementation })
    }
    pub(super) fn kind(self) -> Int8KernelKind {
        match self.implementation {
            Avx2Implementation::Low7HighBit => Int8KernelKind::Avx2Low7HighBit,
            Avx2Implementation::WidenedI16 => Int8KernelKind::Avx2WidenedI16,
        }
    }
    pub(super) fn dot_rows_into(self, inputs: &[&[i8]], weights: &[i8], output: &mut [i32])
        -> Result<(), CpuKernelError> {
        if inputs.len() > MAX_COHORT_ROWS || inputs.len() != output.len() { return Err(CpuKernelError::Shape); }
        if weights.len() > MAX_MODEL_K { return Err(CpuKernelError::ReductionTooLong); }
        if inputs.iter().any(|input| input.len() != weights.len()) { return Err(CpuKernelError::LengthMismatch); }
        if inputs.is_empty() { return Ok(()); }
        // SAFETY: this private handle proves real AVX2 support. All input rows
        // have the same bounded K as weights. The closed match instantiates only
        // M=1..4 and always with EXACTLY M valid slices; the output stays local.
        let sums = unsafe {
            match (self.implementation, inputs.len()) {
                (Avx2Implementation::Low7HighBit, 1) => low7_rows::<1>(inputs, weights),
                (Avx2Implementation::Low7HighBit, 2) => low7_rows::<2>(inputs, weights),
                (Avx2Implementation::Low7HighBit, 3) => low7_rows::<3>(inputs, weights),
                (Avx2Implementation::Low7HighBit, 4) => low7_rows::<4>(inputs, weights),
                (Avx2Implementation::WidenedI16, 1) => widened_rows::<1>(inputs, weights),
                (Avx2Implementation::WidenedI16, 2) => widened_rows::<2>(inputs, weights),
                (Avx2Implementation::WidenedI16, 3) => widened_rows::<3>(inputs, weights),
                (Avx2Implementation::WidenedI16, 4) => widened_rows::<4>(inputs, weights),
                _ => unreachable!("checked AVX2 cohort size"),
            }
        };
        output.copy_from_slice(&sums[..inputs.len()]);
        Ok(())
    }

    pub(super) fn dot(self, input: &[i8], weights: &[i8]) -> Result<i32, CpuKernelError> {
        if input.len() != weights.len() { return Err(CpuKernelError::LengthMismatch); }
        if input.len() > MAX_MODEL_K { return Err(CpuKernelError::ReductionTooLong); }
        // SAFETY: the private handle can only be constructed after real OS-aware
        // AVX2 detection. Equal slice lengths and the arithmetic bound are checked
        // above; neither target function writes to either immutable input slice.
        let value = unsafe {
            match self.implementation {
                Avx2Implementation::Low7HighBit => low7_high_bit(input, weights),
                Avx2Implementation::WidenedI16 => widened_i16(input, weights),
            }
        };
        Ok(value)
    }
}

// X3a: u = x XOR 128; dot(x,w) = dot(lo,w) + 128*dot(hi,w) - 128*sum(w).
// lo in 0..127 gives |adjacent pair| <= 32512; hi in 0..1 gives <= 256.
// Only THESE bounded operands enter maddubs: raw u never does. Each pair is
// immediately widened with madd_epi16(ones). At MAX_MODEL_K the conservative
// raw-offset-plus-correction magnitude is <= 527106048, strictly below i32::MAX.
#[target_feature(enable = "avx2")]
unsafe fn low7_high_bit(input: &[i8], weights: &[i8]) -> i32 {
    let end = input.len() / 32 * 32;
    // SAFETY: caller proved AVX2, equal lengths and K <= MAX_MODEL_K. start+32
    // never exceeds end, so loadu reads exactly 32 initialized bytes from each
    // slice. Storeu writes exactly 32 bytes into each local eight-i32 array.
    // The scalar tail uses checked indexing; no alignment or padding is assumed.
    unsafe {
        let sign = _mm256_set1_epi8(i8::MIN);
        let low_mask = _mm256_set1_epi8(127);
        let ones8 = _mm256_set1_epi8(1);
        let ones16 = _mm256_set1_epi16(1);
        let mut low = _mm256_setzero_si256();
        let mut high = _mm256_setzero_si256();
        let mut sums = _mm256_setzero_si256();
        for start in (0..end).step_by(32) {
            let x = _mm256_loadu_si256(input.as_ptr().add(start).cast());
            let w = _mm256_loadu_si256(weights.as_ptr().add(start).cast());
            let u = _mm256_xor_si256(x, sign);
            let lo = _mm256_and_si256(u, low_mask);
            // Word shift followed by a per-byte 0x01 mask extracts BOTH bytes'
            // bit 7 independently; contamination from the neighboring byte dies.
            let hi = _mm256_and_si256(_mm256_srli_epi16::<7>(u), ones8);
            low = _mm256_add_epi32(low, _mm256_madd_epi16(_mm256_maddubs_epi16(lo, w), ones16));
            high = _mm256_add_epi32(high, _mm256_madd_epi16(_mm256_maddubs_epi16(hi, w), ones16));
            sums = _mm256_add_epi32(sums, _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, w), ones16));
        }
        let mut low_lanes = [0_i32; 8];
        let mut high_lanes = [0_i32; 8];
        let mut sum_lanes = [0_i32; 8];
        _mm256_storeu_si256(low_lanes.as_mut_ptr().cast(), low);
        _mm256_storeu_si256(high_lanes.as_mut_ptr().cast(), high);
        _mm256_storeu_si256(sum_lanes.as_mut_ptr().cast(), sums);
        // Fixed ascending-lane reduction; each term covers four original bytes.
        let mut result = 0_i32;
        for lane in 0..8 {
            result += low_lanes[lane] + 128 * high_lanes[lane] - 128 * sum_lanes[lane];
        }
        for index in end..input.len() { result += i32::from(input[index]) * i32::from(weights[index]); }
        result
    }
}

// X3b is independent: sign-extend BOTH byte operands into i16, then maddwd.
// Adjacent pair <= 2*128*128 = 32768 fits i32; no offset, correction or maddubs.
// Each full dot and every partial has absolute bound K*16384 <= 176160768.
#[target_feature(enable = "avx2")]
unsafe fn widened_i16(input: &[i8], weights: &[i8]) -> i32 {
    let end = input.len() / 32 * 32;
    // SAFETY: caller proved AVX2, equal lengths and bounded K. Each iteration
    // reads [start,start+16) and [start+16,start+32), both contained in the slices.
    // Local lane arrays have exactly the 32 writable bytes required by storeu.
    // These loads and stores do not require alignment. Tail reads are indexed.
    unsafe {
        let mut lower = _mm256_setzero_si256();
        let mut upper = _mm256_setzero_si256();
        for start in (0..end).step_by(32) {
            let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(input.as_ptr().add(start).cast()));
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(weights.as_ptr().add(start).cast()));
            let x1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(input.as_ptr().add(start + 16).cast()));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(weights.as_ptr().add(start + 16).cast()));
            lower = _mm256_add_epi32(lower, _mm256_madd_epi16(x0, w0));
            upper = _mm256_add_epi32(upper, _mm256_madd_epi16(x1, w1));
        }
        let mut low_lanes = [0_i32; 8];
        let mut high_lanes = [0_i32; 8];
        _mm256_storeu_si256(low_lanes.as_mut_ptr().cast(), lower);
        _mm256_storeu_si256(high_lanes.as_mut_ptr().cast(), upper);
        let mut result = 0_i32;
        // Fixed lower-half lane order, then upper-half lane order, then tail.
        for value in low_lanes { result += value; }
        for value in high_lanes { result += value; }
        for index in end..input.len() { result += i32::from(input[index]) * i32::from(weights[index]); }
        result
    }
}


// Independent per-document accumulators, common weight bytes and correction.
// The pair/whole-dot bounds above apply to EACH row, not M times one i32 sum.
#[target_feature(enable = "avx2")]
unsafe fn low7_rows<const M: usize>(inputs: &[&[i8]], weights: &[i8]) -> [i32; MAX_COHORT_ROWS] {
    let end = weights.len() / 32 * 32;
    // SAFETY: the closed caller proves AVX2, M=1..4, inputs.len()==M and equal
    // bounded K for every row. All vector loads stay within complete 32-byte
    // blocks. Each storeu targets one local eight-i32 array. No padding or
    // alignment is assumed; tail indices are in-bounds on every original row.
    unsafe {
        let sign = _mm256_set1_epi8(i8::MIN);
        let low_mask = _mm256_set1_epi8(127);
        let ones8 = _mm256_set1_epi8(1);
        let ones16 = _mm256_set1_epi16(1);
        let zero = _mm256_setzero_si256();
        let mut low = [zero; M];
        let mut high = [zero; M];
        let mut sums = zero;
        for start in (0..end).step_by(32) {
            let w = _mm256_loadu_si256(weights.as_ptr().add(start).cast());
            sums = _mm256_add_epi32(sums, _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, w), ones16));
            for row in 0..M {
                let x = _mm256_loadu_si256(inputs[row].as_ptr().add(start).cast());
                let u = _mm256_xor_si256(x, sign);
                let lo = _mm256_and_si256(u, low_mask);
                let hi = _mm256_and_si256(_mm256_srli_epi16::<7>(u), ones8);
                low[row] = _mm256_add_epi32(low[row], _mm256_madd_epi16(_mm256_maddubs_epi16(lo, w), ones16));
                high[row] = _mm256_add_epi32(high[row], _mm256_madd_epi16(_mm256_maddubs_epi16(hi, w), ones16));
            }
        }
        let mut correction = [0_i32; 8];
        _mm256_storeu_si256(correction.as_mut_ptr().cast(), sums);
        let mut output = [0_i32; MAX_COHORT_ROWS];
        for row in 0..M {
            let mut low_lanes = [0_i32; 8];
            let mut high_lanes = [0_i32; 8];
            _mm256_storeu_si256(low_lanes.as_mut_ptr().cast(), low[row]);
            _mm256_storeu_si256(high_lanes.as_mut_ptr().cast(), high[row]);
            for lane in 0..8 {
                output[row] += low_lanes[lane] + 128 * high_lanes[lane] - 128 * correction[lane];
            }
        }
        for index in end..weights.len() {
            let w = i32::from(weights[index]);
            for row in 0..M { output[row] += i32::from(inputs[row][index]) * w; }
        }
        output
    }
}

#[target_feature(enable = "avx2")]
unsafe fn widened_rows<const M: usize>(inputs: &[&[i8]], weights: &[i8]) -> [i32; MAX_COHORT_ROWS] {
    let end = weights.len() / 32 * 32;
    // SAFETY: the caller proves real AVX2, M=1..4, exactly M input rows, and
    // equal bounded K. Both 16-byte halves of each load fit inside the complete
    // 32-byte block, including on the last row. Storeu writes only local arrays
    // of eight i32 values. Scalar tails never read beyond a row or weights.
    unsafe {
        let zero = _mm256_setzero_si256();
        let mut lower = [zero; M];
        let mut upper = [zero; M];
        for start in (0..end).step_by(32) {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(weights.as_ptr().add(start).cast()));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(weights.as_ptr().add(start + 16).cast()));
            for row in 0..M {
                let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(inputs[row].as_ptr().add(start).cast()));
                let x1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(inputs[row].as_ptr().add(start + 16).cast()));
                lower[row] = _mm256_add_epi32(lower[row], _mm256_madd_epi16(x0, w0));
                upper[row] = _mm256_add_epi32(upper[row], _mm256_madd_epi16(x1, w1));
            }
        }
        let mut output = [0_i32; MAX_COHORT_ROWS];
        for row in 0..M {
            let mut low_lanes = [0_i32; 8];
            let mut high_lanes = [0_i32; 8];
            _mm256_storeu_si256(low_lanes.as_mut_ptr().cast(), lower[row]);
            _mm256_storeu_si256(high_lanes.as_mut_ptr().cast(), upper[row]);
            for value in low_lanes { output[row] += value; }
            for value in high_lanes { output[row] += value; }
        }
        for index in end..weights.len() {
            let w = i32::from(weights[index]);
            for row in 0..M { output[row] += i32::from(inputs[row][index]) * w; }
        }
        output
    }
}

#[cfg(test)]
mod catalog_tests {
    use super::*;
    #[test]
    fn non_avx_catalog_entries_cannot_construct_an_avx_handle() {
        for kind in Int8KernelKind::ALL {
            if matches!(kind, Int8KernelKind::Avx2Low7HighBit | Int8KernelKind::Avx2WidenedI16) { continue; }
            assert!(matches!(Avx2Kernel::detect(kind), Err(CpuKernelError::UnknownKind)));
        }
    }
}
