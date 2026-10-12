//! VNNI arithmetic and admission regressions. Unsupported hardware is explicit.
use franken_nlp::native_engine::simd::{CpuKernelError, Int8Kernel, Int8KernelKind};

fn kernel() -> Option<Int8Kernel> {
    match Int8Kernel::detect(Int8KernelKind::AvxVnni) {
        Ok(kernel) => Some(kernel),
        Err(CpuKernelError::UnsupportedIsa) => {
            eprintln!("VNNI tier=x2_avx_vnni RESULT=SKIPPED_UNSUPPORTED_ISA"); None
        }
        Err(error) => panic!("unexpected VNNI admission: {error}"),
    }
}
fn oracle(a: &[i8], w: &[i8]) -> i32 {
    assert_eq!(a.len(), w.len());
    i32::try_from(a.iter().zip(w).map(|(&a, &w)| i64::from(a) * i64::from(w)).sum::<i64>()).unwrap()
}
fn next(state: &mut u64) -> i8 {
    *state ^= *state << 13; *state ^= *state >> 7; *state ^= *state << 17;
    (*state >> 32) as u8 as i8
}

#[test]
fn avx_vnni_admission_matches_actual_avx2_and_vex_vnni_not_avx512() {
    let expected = {
        #[cfg(target_arch = "x86_64")]
        { std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("avxvnni") }
        #[cfg(not(target_arch = "x86_64"))]
        { false }
    };
    assert_eq!(kernel().is_some(), expected);
    assert_eq!("x2_avx_vnni".parse::<Int8KernelKind>().unwrap(), Int8KernelKind::AvxVnni);
    assert_eq!(Int8Kernel::default().kind(), Int8KernelKind::Scalar);
}

#[test]
fn zero_activations_cancel_the_offset_for_every_tail_and_weight_sum() {
    let Some(kernel) = kernel() else { return; };
    for k in (0..160).chain([3072, 6144, 10752]) {
        for value in [-128, -127, -1, 0, 1, 127] {
            let weights = vec![value; k]; let zeros = vec![0; k];
            let minimum = vec![-128; k]; let maximum = vec![127; k];
            let mut out = [999; 4];
            kernel.dot_rows_into(&[&zeros, &minimum, &maximum, &zeros], &weights, &mut out).unwrap();
            assert_eq!(out, [0, oracle(&minimum, &weights), oracle(&maximum, &weights), 0], "K={k} w={value}");
        }
    }
}

#[test]
fn vector_boundary_impulses_keep_correction_and_scalar_cleanup_separate() {
    let Some(kernel) = kernel() else { return; };
    for k in [31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129] {
        for index in 0..k {
            let mut a = vec![0; k]; a[index] = -128;
            let w: Vec<_> = (0..k).map(|i| if i % 3 == 0 { -128 } else { 127 }).collect();
            assert_eq!(kernel.dot(&a, &w).unwrap(), oracle(&a, &w), "K={k} index={index}");
        }
    }
}

#[test]
fn independently_unaligned_borrowed_cohorts_match_i64_through_all_vector_tails() {
    let Some(kernel) = kernel() else { return; };
    let mut state = 0x89af_53cd_7341_28c7_u64;
    let storage: Vec<Vec<i8>> = (0..4).map(|_| (0..10816).map(|_| next(&mut state)).collect()).collect();
    let weights: Vec<_> = (0..10816).map(|_| next(&mut state)).collect();
    for offset in 0..64 {
        for k in (0..96).chain((0..64).map(|tail| 10752 - tail)) {
            for m in 1..=4 {
                let inputs: Vec<_> = storage[..m].iter().enumerate().map(|(row, bytes)| {
                    let start = (offset + row * 13) % 64; &bytes[start..start + k]
                }).collect();
                let w = &weights[63 - offset..63 - offset + k];
                let mut out = [999; 6];
                kernel.dot_rows_into(&inputs, w, &mut out[1..1 + m]).unwrap();
                assert_eq!(out[0], 999); assert!(out[1 + m..].iter().all(|&v| v == 999));
                for row in 0..m { assert_eq!(out[1 + row], oracle(inputs[row], w)); }
            }
        }
    }
}

#[test]
fn dense_shared_weight_tiles_preserve_row_order_and_all_short_cohorts() {
    let Some(kernel) = kernel() else { return; };
    let mut state = 0x345f_853d_982a_2117;
    for m in [0_usize, 1, 2, 3, 4, 5, 7, 9] {
        for n in [0_usize, 1, 2, 3, 5] {
            for k in [0, 1, 31, 32, 33, 63, 64, 65, 3072] {
                let a: Vec<_> = (0..m * k).map(|_| next(&mut state)).collect();
                let w: Vec<_> = (0..n * k).map(|_| next(&mut state)).collect();
                let mut out = vec![999; m * n];
                kernel.gemm_into(&a, m, k, &w, n, &mut out).unwrap();
                for row in 0..m { for column in 0..n {
                    assert_eq!(out[row * n + column], oracle(&a[row * k..(row + 1) * k], &w[column * k..(column + 1) * k]));
                } }
            }
        }
    }
}

#[test]
fn final_invalid_row_and_excess_k_leave_outputs_unchanged() {
    let Some(kernel) = kernel() else { return; };
    let good = [1; 65]; let bad = [1; 64]; let mut out = [999; 4];
    assert_eq!(kernel.dot_rows_into(&[&good, &good, &good, &bad], &good, &mut out), Err(CpuKernelError::LengthMismatch));
    assert_eq!(out, [999; 4]);
    let long = vec![0; 10753];
    assert_eq!(kernel.dot_rows_into(&[&long], &long, &mut out[..1]), Err(CpuKernelError::ReductionTooLong));
    assert_eq!(out, [999; 4]);
    assert_eq!(kernel.gemm_into(&[], usize::MAX, 2, &[], 0, &mut []), Err(CpuKernelError::Shape));
    kernel.gemm_into(&[], usize::MAX, 0, &[], 0, &mut []).unwrap();
}
