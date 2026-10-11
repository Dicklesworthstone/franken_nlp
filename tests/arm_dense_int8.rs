//! Dense ARM tile definitions; ISA cases skip explicitly off the actual host.
//! These tests compare real executable handles, not injected feature reports.
use franken_nlp::native_engine::simd::{CpuKernelError, Int8Kernel, Int8KernelKind};

const ARM: [Int8KernelKind; 3] = [Int8KernelKind::Aarch64Smmla,
    Int8KernelKind::Aarch64Dotprod, Int8KernelKind::Aarch64Autovec];
fn kernels() -> Vec<Int8Kernel> {
    ARM.into_iter().filter_map(|kind| match Int8Kernel::detect(kind) {
        Ok(kernel) => Some(kernel),
        Err(CpuKernelError::UnsupportedIsa) => {
            eprintln!("ARM_DENSE tier={kind} RESULT=SKIPPED_UNSUPPORTED_ISA"); None
        }
        Err(error) => panic!("unexpected ARM construction failure: {error}"),
    }).collect()
}
fn oracle(a: &[i8], w: &[i8]) -> i32 {
    i32::try_from(a.iter().zip(w).map(|(&a, &w)| i64::from(a) * i64::from(w)).sum::<i64>()).unwrap()
}
fn pattern(len: usize, salt: usize) -> Vec<i8> {
    (0..len).map(|i| ((i * 73 + salt * 29 + 128) % 256) as u8 as i8).collect()
}

#[test]
fn dense_arm_tiles_cover_m_n_and_reduction_tails_against_independent_i64() {
    let kernels = kernels();
    for m in [0_usize, 1, 2, 3, 4, 5, 7, 9] {
        for n in [0_usize, 1, 2, 3, 5] {
            for k in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 3072, 6144, 10752] {
                let a = pattern(m * k, 1); let w = pattern(n * k, 2);
                for &kernel in &kernels {
                    let mut output = vec![123; m * n];
                    kernel.gemm_into(&a, m, k, &w, n, &mut output).unwrap();
                    for row in 0..m {
                        for column in 0..n {
                            assert_eq!(output[row * n + column], oracle(&a[row * k..(row + 1) * k],
                                &w[column * k..(column + 1) * k]), "tier={} M={m} N={n} K={k} row={row} col={column}", kernel.kind());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn smmla_layout_is_two_activation_rows_two_weight_columns_and_row_major_output() {
    let kernel = match Int8Kernel::detect(Int8KernelKind::Aarch64Smmla) {
        Ok(kernel) => kernel,
        Err(CpuKernelError::UnsupportedIsa) => {
            eprintln!("ARM_LAYOUT tier=a1_smmla RESULT=SKIPPED_UNSUPPORTED_ISA"); return;
        }
        Err(error) => panic!("unexpected SMMLA failure: {error}"),
    };
    let a = [1_i8, 2, 3, 4, 5, 6, 7, 8, -1, 2, -3, 4, -5, 6, -7, 8];
    let w = [2_i8, 2, 2, 2, 2, 2, 2, 2, -1, 0, 1, 0, -1, 0, 1, 0];
    let mut output = [123; 4];
    kernel.gemm_into(&a, 2, 8, &w, 2, &mut output).unwrap();
    assert_eq!(output, [72, 4, 8, -4]);
    // A one-column cohort must extract r0c0/r1c0, not adjacent SIMD lanes.
    let mut cohort = [123; 2];
    kernel.dot_rows_into(&[&a[..8], &a[8..]], &w[..8], &mut cohort).unwrap();
    assert_eq!(cohort, [72, 8]);
    // A single-row matrix must never read a nonexistent second activation row.
    kernel.gemm_into(&a[..8], 1, 8, &w, 2, &mut output[..2]).unwrap();
    assert_eq!(output, [72, 4, 8, -4]);
}

#[test]
fn four_by_two_extreme_tiles_bound_each_output_not_the_sum_of_all_outputs() {
    let k = 10752;
    let a: Vec<i8> = [-128, 127, -127, 0].into_iter().flat_map(|value| vec![value; k]).collect();
    let w: Vec<i8> = [-128, 127].into_iter().flat_map(|value| vec![value; k]).collect();
    let expected = [176_160_768, -174_784_512, -174_784_512, 173_419_008,
        174_784_512, -173_419_008, 0, 0];
    for kernel in kernels() {
        let mut output = [123; 8];
        kernel.gemm_into(&a, 4, k, &w, 2, &mut output).unwrap();
        assert_eq!(output, expected, "tier={}", kernel.kind());
    }
}

#[test]
fn all_arm_dense_operands_and_output_are_validated_before_writes() {
    for kernel in kernels() {
        let mut output = [123; 6];
        assert_eq!(kernel.gemm_into(&[1; 15], 2, 8, &[1; 24], 3, &mut output), Err(CpuKernelError::Shape));
        assert_eq!(output, [123; 6]);
        assert_eq!(kernel.gemm_into(&[1; 16], 2, 8, &[1; 23], 3, &mut output), Err(CpuKernelError::Shape));
        assert_eq!(output, [123; 6]);
        assert_eq!(kernel.gemm_into(&[1; 16], 2, 8, &[1; 24], 3, &mut output[..5]), Err(CpuKernelError::Shape));
        assert_eq!(output, [123; 6]);
        assert_eq!(kernel.gemm_into(&[], usize::MAX, 2, &[], 0, &mut []), Err(CpuKernelError::Shape));
        assert_eq!(kernel.gemm_into(&[], 0, 10753, &[], 0, &mut []), Err(CpuKernelError::ReductionTooLong));
        kernel.gemm_into(&[], usize::MAX, 0, &[], 0, &mut []).unwrap();
        kernel.gemm_into(&[], 2, 0, &[], 3, &mut output).unwrap();
        assert_eq!(output, [0; 6]);
    }
}

#[test]
fn dense_batching_permutation_and_unaligned_views_preserve_each_independent_result() {
    let (m, k, n) = (7, 33, 5);
    let storage_a = pattern(m * k + 7, 11); let a = &storage_a[7..];
    let storage_w = pattern(n * k + 3, 13); let w = &storage_w[3..];
    for kernel in kernels() {
        let mut whole = vec![0; m * n]; let mut singles = vec![0; m * n];
        kernel.gemm_into(a, m, k, w, n, &mut whole).unwrap();
        for row in 0..m { kernel.gemm_into(&a[row * k..(row + 1) * k], 1, k, w, n,
            &mut singles[row * n..(row + 1) * n]).unwrap(); }
        assert_eq!(whole, singles);
        let reversed: Vec<i8> = a.chunks(k).rev().flat_map(|row| row.iter().copied()).collect();
        kernel.gemm_into(&reversed, m, k, w, n, &mut singles).unwrap();
        for row in 0..m { assert_eq!(&singles[row * n..(row + 1) * n],
            &whole[(m - 1 - row) * n..(m - row) * n]); }
    }
}

#[test]
fn smmla_requires_real_i8mm_exposure_independent_of_dotprod() {
    let expected = {
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        { std::arch::is_aarch64_feature_detected!("neon") && std::arch::is_aarch64_feature_detected!("i8mm") }
        #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
        { false }
    };
    match Int8Kernel::detect(Int8KernelKind::Aarch64Smmla) {
        Ok(kernel) => { assert!(expected); assert_eq!(kernel.kind(), Int8KernelKind::Aarch64Smmla); }
        Err(CpuKernelError::UnsupportedIsa) => assert!(!expected),
        Err(error) => panic!("wrong SMMLA refusal: {error}"),
    }
}
