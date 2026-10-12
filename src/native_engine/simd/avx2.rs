//! Enumerated x86 SIMD island. No public unsafe API or speculative execution.
//!
//! Construction checks std's actual OS-aware features for the chosen body.
//! Slices are rechecked before target entry. Complete vectors use unaligned
//! loads; short tails use scalar indexing. K <= 10752 bounds every i32 sum.
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;
use super::{CpuKernelError, Int8KernelKind, MAX_COHORT_ROWS, MAX_MODEL_K};

// Private state contains only executable x86 algorithms, not the public
// cross-architecture catalog. The historical module/type name is retained.
#[derive(Clone, Copy, Debug)]
enum Avx2Implementation { Low7HighBit, WidenedI16, AvxVnni, VnniYmm, VnniZmm }
#[derive(Clone, Copy, Debug)]
pub(super) struct Avx2Kernel { implementation: Avx2Implementation }
impl Avx2Kernel {
    pub(super) fn detect(kind: Int8KernelKind) -> Result<Self, CpuKernelError> {
        let (implementation, supported) = match kind {
            Int8KernelKind::Avx2Low7HighBit => (Avx2Implementation::Low7HighBit,
                std::arch::is_x86_feature_detected!("avx2")),
            Int8KernelKind::Avx2WidenedI16 => (Avx2Implementation::WidenedI16,
                std::arch::is_x86_feature_detected!("avx2")),
            Int8KernelKind::AvxVnni => (Avx2Implementation::AvxVnni,
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("avxvnni")),
            Int8KernelKind::Avx512VnniYmm => (Avx2Implementation::VnniYmm,
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512vnni")
                    && std::arch::is_x86_feature_detected!("avx512vl")),
            Int8KernelKind::Avx512VnniZmm => (Avx2Implementation::VnniZmm,
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512vnni")),
            _ => return Err(CpuKernelError::UnknownKind),
        };
        if !supported { return Err(CpuKernelError::UnsupportedIsa); }
        Ok(Self { implementation })
    }
    pub(super) fn kind(self) -> Int8KernelKind {
        match self.implementation {
            Avx2Implementation::Low7HighBit => Int8KernelKind::Avx2Low7HighBit,
            Avx2Implementation::WidenedI16 => Int8KernelKind::Avx2WidenedI16,
            Avx2Implementation::AvxVnni => Int8KernelKind::AvxVnni,
            Avx2Implementation::VnniYmm => Int8KernelKind::Avx512VnniYmm,
            Avx2Implementation::VnniZmm => Int8KernelKind::Avx512VnniZmm,
        }
    }
    pub(super) fn dot_rows_into(self, inputs: &[&[i8]], weights: &[i8], output: &mut [i32])
        -> Result<(), CpuKernelError> {
        if inputs.len() > MAX_COHORT_ROWS || inputs.len() != output.len() { return Err(CpuKernelError::Shape); }
        if weights.len() > MAX_MODEL_K { return Err(CpuKernelError::ReductionTooLong); }
        if inputs.iter().any(|input| input.len() != weights.len()) { return Err(CpuKernelError::LengthMismatch); }
        if inputs.is_empty() { return Ok(()); }
        // SAFETY: the private handle proves every actual CPU feature required
        // by its selected body. All rows have equal bounded K. This closed
        // match instantiates M=1..4 with exactly M valid slices for every body.
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
                (Avx2Implementation::AvxVnni, 1) => avx_vnni_rows::<1>(inputs, weights),
                (Avx2Implementation::AvxVnni, 2) => avx_vnni_rows::<2>(inputs, weights),
                (Avx2Implementation::AvxVnni, 3) => avx_vnni_rows::<3>(inputs, weights),
                (Avx2Implementation::AvxVnni, 4) => avx_vnni_rows::<4>(inputs, weights),
                (Avx2Implementation::VnniYmm, 1) => evex_ymm_rows::<1>(inputs, weights),
                (Avx2Implementation::VnniYmm, 2) => evex_ymm_rows::<2>(inputs, weights),
                (Avx2Implementation::VnniYmm, 3) => evex_ymm_rows::<3>(inputs, weights),
                (Avx2Implementation::VnniYmm, 4) => evex_ymm_rows::<4>(inputs, weights),
                (Avx2Implementation::VnniZmm, 1) => evex_zmm_rows::<1>(inputs, weights),
                (Avx2Implementation::VnniZmm, 2) => evex_zmm_rows::<2>(inputs, weights),
                (Avx2Implementation::VnniZmm, 3) => evex_zmm_rows::<3>(inputs, weights),
                (Avx2Implementation::VnniZmm, 4) => evex_zmm_rows::<4>(inputs, weights),
                _ => unreachable!("checked x86 cohort size"),
            }
        };
        output.copy_from_slice(&sums[..inputs.len()]);
        Ok(())
    }

    pub(super) fn dot(self, input: &[i8], weights: &[i8]) -> Result<i32, CpuKernelError> {
        if input.len() != weights.len() { return Err(CpuKernelError::LengthMismatch); }
        if input.len() > MAX_MODEL_K { return Err(CpuKernelError::ReductionTooLong); }
        // SAFETY: the private handle proves actual features for this body;
        // equal slice lengths and bounded K are checked before entry.
        let value = unsafe {
            match self.implementation {
                Avx2Implementation::Low7HighBit => low7_high_bit(input, weights),
                Avx2Implementation::WidenedI16 => widened_i16(input, weights),
                Avx2Implementation::AvxVnni => avx_vnni_rows::<1>(&[input], weights)[0],
                Avx2Implementation::VnniYmm => evex_ymm_rows::<1>(&[input], weights)[0],
                Avx2Implementation::VnniZmm => evex_zmm_rows::<1>(&[input], weights)[0],
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

// X2: u=x XOR 0x80; dot(x,w)=dot(u,w)-128*sum(w). The non-saturating
// VEX intrinsic is _mm256_dpbusd_avx_epi32, NOT the AVX-512 EVEX spelling.
// K<=10752 bounds raw sums by 350945280, correction by 176160768, and their
// conservative combined magnitude by 527106048. All partials fit i32.
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn avx_vnni_rows<const M: usize>(inputs: &[&[i8]], weights: &[i8]) -> [i32; MAX_COHORT_ROWS] {
    let end = weights.len() / 32 * 32;
    // SAFETY: caller proves AVX2+AVX-VNNI, M=1..4 and equal bounded lengths.
    // Loads read only contained complete 32-byte blocks. Stores write exactly
    // eight i32s into local arrays. No alignment/padding; tails use indexing.
    unsafe {
        let sign = _mm256_set1_epi32(i32::from_le_bytes([0x80; 4]));
        let ones = _mm256_set1_epi32(0x0101_0101);
        let mut raw = [_mm256_setzero_si256(); M];
        let mut sums = _mm256_setzero_si256();
        for start in (0..end).step_by(32) {
            let w = _mm256_loadu_si256(weights.as_ptr().add(start).cast());
            sums = _mm256_dpbusd_avx_epi32(sums, ones, w);
            for row in 0..M {
                let x = _mm256_loadu_si256(inputs[row].as_ptr().add(start).cast());
                raw[row] = _mm256_dpbusd_avx_epi32(raw[row], _mm256_xor_si256(x, sign), w);
            }
        }
        let mut correction = [0_i32; 8];
        _mm256_storeu_si256(correction.as_mut_ptr().cast(), sums);
        let mut output = [0_i32; MAX_COHORT_ROWS];
        for row in 0..M {
            let mut lanes = [0_i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr().cast(), raw[row]);
            for lane in 0..8 { output[row] += lanes[lane] - 128 * correction[lane]; }
        }
        for index in end..weights.len() {
            let w = i32::from(weights[index]);
            for row in 0..M { output[row] += i32::from(inputs[row][index]) * w; }
        }
        output
    }
}

// Two separately target-gated EVEX bodies. They use the same exact signed
// correction and per-accumulator bounds as X2, but different vector widths.
// This macro selects intrinsics at compile time, never in the hot K loop.
macro_rules! evex_body {
    ($name:ident, $features:literal, $width:literal, $lanes:literal,
        $zero:ident, $splat:ident, $load:ident, $store:ident, $xor:ident, $dot:ident) => {
        #[target_feature(enable = $features)]
        unsafe fn $name<const M: usize>(inputs: &[&[i8]], weights: &[i8]) -> [i32; MAX_COHORT_ROWS] {
            let end = weights.len() / $width * $width;
            // SAFETY: caller proves every feature named above, M=1..4, and
            // equal bounded K. Loads cover only complete contained vectors.
            // Stores target local arrays of exactly $lanes i32s. No padding
            // or alignment is assumed. Short tails use checked signed indexing.
            unsafe {
                let sign = $splat(i32::from_le_bytes([0x80; 4]));
                let ones = $splat(0x0101_0101);
                let mut raw = [$zero(); M];
                let mut sums = $zero();
                for start in (0..end).step_by($width) {
                    let w = $load(weights.as_ptr().add(start).cast());
                    sums = $dot(sums, ones, w);
                    for row in 0..M {
                        let x = $load(inputs[row].as_ptr().add(start).cast());
                        raw[row] = $dot(raw[row], $xor(x, sign), w);
                    }
                }
                let mut correction = [0_i32; $lanes];
                $store(correction.as_mut_ptr().cast(), sums);
                let mut output = [0_i32; MAX_COHORT_ROWS];
                for row in 0..M {
                    let mut lanes = [0_i32; $lanes];
                    $store(lanes.as_mut_ptr().cast(), raw[row]);
                    for lane in 0..$lanes { output[row] += lanes[lane] - 128 * correction[lane]; }
                }
                for index in end..weights.len() {
                    let w = i32::from(weights[index]);
                    for row in 0..M { output[row] += i32::from(inputs[row][index]) * w; }
                }
                output
            }
        }
    };
}
evex_body!(evex_ymm_rows, "avx2,avx512f,avx512vnni,avx512vl", 32, 8,
    _mm256_setzero_si256, _mm256_set1_epi32, _mm256_loadu_si256,
    _mm256_storeu_si256, _mm256_xor_si256, _mm256_dpbusd_epi32);
// Dword splats, full-vector loads and XOR need AVX512F, not AVX512BW/VL.
// X1a therefore does not accidentally acquire a byte-mask or VL dependency.
evex_body!(evex_zmm_rows, "avx512f,avx512vnni", 64, 16,
    _mm512_setzero_si512, _mm512_set1_epi32, _mm512_loadu_si512,
    _mm512_storeu_si512, _mm512_xor_si512, _mm512_dpbusd_epi32);

#[cfg(test)]
mod catalog_tests {
    use super::*;
    #[test]
    fn non_x86_catalog_entries_cannot_construct_an_x86_handle() {
        for kind in Int8KernelKind::ALL {
            if matches!(kind, Int8KernelKind::Avx2Low7HighBit | Int8KernelKind::Avx2WidenedI16
                | Int8KernelKind::AvxVnni | Int8KernelKind::Avx512VnniYmm | Int8KernelKind::Avx512VnniZmm) { continue; }
            assert!(matches!(Avx2Kernel::detect(kind), Err(CpuKernelError::UnknownKind)));
        }
    }
}
