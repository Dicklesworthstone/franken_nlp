//! Enumerated AVX2 island. No public unsafe API or speculative ISA execution.
//!
//! The only constructor checks std's actual OS-aware AVX2 detection. Both
//! slices are rechecked before entering target-feature code. Complete 32-byte
//! blocks use unaligned loads; the remainder uses scalar indexing, never a
//! masked or oversized load. K <= 10752 bounds EVERY partial and final i32 sum.
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;
use super::{CpuKernelError, Int8KernelKind, MAX_MODEL_K};

#[derive(Clone, Copy, Debug)]
pub(super) struct Avx2Kernel { kind: Int8KernelKind }
impl Avx2Kernel {
    pub(super) fn detect(kind: Int8KernelKind) -> Result<Self, CpuKernelError> {
        if !matches!(kind, Int8KernelKind::Avx2Low7HighBit | Int8KernelKind::Avx2WidenedI16) {
            return Err(CpuKernelError::UnknownKind);
        }
        if !std::arch::is_x86_feature_detected!("avx2") { return Err(CpuKernelError::UnsupportedIsa); }
        Ok(Self { kind })
    }
    pub(super) fn kind(self) -> Int8KernelKind { self.kind }
    pub(super) fn dot(self, input: &[i8], weights: &[i8]) -> Result<i32, CpuKernelError> {
        if input.len() != weights.len() { return Err(CpuKernelError::LengthMismatch); }
        if input.len() > MAX_MODEL_K { return Err(CpuKernelError::ReductionTooLong); }
        // SAFETY: the private handle can only be constructed after real OS-aware
        // AVX2 detection. Equal slice lengths and the arithmetic bound are checked
        // above; neither target function writes to either immutable input slice.
        let value = unsafe {
            match self.kind {
                Int8KernelKind::Avx2Low7HighBit => low7_high_bit(input, weights),
                Int8KernelKind::Avx2WidenedI16 => widened_i16(input, weights),
                Int8KernelKind::Scalar => unreachable!("sealed AVX2 constructor"),
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
