//! Real-host VNNI width differential definitions. No injected ISA authority.
use franken_nlp::native_engine::simd::{CpuKernelError, Int8Kernel, Int8KernelKind};

const KINDS: [Int8KernelKind; 3] = [Int8KernelKind::AvxVnni,
    Int8KernelKind::Avx512VnniYmm, Int8KernelKind::Avx512VnniZmm];
fn kernels() -> Vec<Int8Kernel> {
    KINDS.into_iter().filter_map(|kind| match Int8Kernel::detect(kind) {
        Ok(kernel) => Some(kernel),
        Err(CpuKernelError::UnsupportedIsa) => {
            eprintln!("VNNI_WIDTH tier={kind} RESULT=SKIPPED_UNSUPPORTED_ISA"); None
        }
        Err(error) => panic!("unexpected VNNI admission: {error}"),
    }).collect()
}
fn expected_support(kind: Int8KernelKind) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        let f = std::arch::is_x86_feature_detected!("avx512f");
        let vnni = std::arch::is_x86_feature_detected!("avx512vnni");
        match kind {
            Int8KernelKind::AvxVnni => avx2 && std::arch::is_x86_feature_detected!("avxvnni"),
            Int8KernelKind::Avx512VnniYmm => avx2 && f && vnni && std::arch::is_x86_feature_detected!("avx512vl"),
            Int8KernelKind::Avx512VnniZmm => f && vnni,
            _ => false,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    { let _ = kind; false }
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
fn each_width_requires_its_exact_real_host_capabilities() {
    for kind in KINDS {
        match Int8Kernel::detect(kind) {
            Ok(kernel) => { assert!(expected_support(kind)); assert_eq!(kernel.kind(), kind); }
            Err(CpuKernelError::UnsupportedIsa) => assert!(!expected_support(kind)),
            Err(error) => panic!("wrong feature refusal: {error}"),
        }
        assert_eq!(kind.id().parse::<Int8KernelKind>().unwrap(), kind);
    }
    assert_eq!(Int8Kernel::default().kind(), Int8KernelKind::Scalar);
}

#[test]
fn every_signed_byte_pair_enters_real_512_bit_vectors_and_independent_rows() {
    let kernels = kernels();
    // 32-byte exhaustive fixtures cannot exercise the ZMM vector body. Use
    // 64 bytes so all three widths actually enter their instruction loops.
    for x in -128_i16..=127 {
        let a = [x as i8; 64]; let b = [!(x as i8); 64];
        let minimum = [-128; 64]; let zero = [0; 64];
        let inputs: [&[i8]; 4] = [&a, &b, &minimum, &zero];
        for w in -128_i16..=127 {
            let weights = [w as i8; 64];
            let expected = [64 * i32::from(x) * i32::from(w),
                64 * i32::from(!(x as i8)) * i32::from(w), -8192 * i32::from(w), 0];
            for &kernel in &kernels {
                let mut out = [999; 4];
                kernel.dot_rows_into(&inputs, &weights, &mut out).unwrap();
                assert_eq!(out, expected, "tier={} x={x} w={w}", kernel.kind());
                assert_eq!(kernel.dot(&a, &weights).unwrap(), expected[0]);
            }
        }
    }
}

#[test]
fn all_64_byte_alignments_and_reduction_tails_match_independent_i64() {
    let kernels = kernels(); let mut seed = 0xd621_721e_874b_292d;
    let storage: Vec<Vec<i8>> = (0..4).map(|_| (0..10816).map(|_| next(&mut seed)).collect()).collect();
    let weights: Vec<_> = (0..10816).map(|_| next(&mut seed)).collect();
    for offset in 0..64 {
        for k in (0..160).chain((0..64).map(|tail| 10752 - tail)) {
            let w = &weights[63 - offset..63 - offset + k];
            let inputs: Vec<_> = storage.iter().enumerate().map(|(row, data)| {
                let start = (offset + row * 17) % 64; &data[start..start + k]
            }).collect();
            for m in 1..=4 {
                let expected: Vec<_> = inputs[..m].iter().map(|a| oracle(a, w)).collect();
                for &kernel in &kernels {
                    let mut out = [999; 6];
                    kernel.dot_rows_into(&inputs[..m], w, &mut out[1..1 + m]).unwrap();
                    assert_eq!(&out[1..1 + m], expected.as_slice(), "tier={} M={m} K={k} offset={offset}", kernel.kind());
                    assert_eq!(out[0], 999); assert!(out[1 + m..].iter().all(|&value| value == 999));
                }
            }
        }
    }
}

#[test]
fn full_model_extremes_and_zero_offsets_never_saturate() {
    let kernels = kernels();
    for k in [3072, 6144, 10752] {
        let a = vec![-128; k]; let b = vec![127; k]; let zeros = vec![0; k];
        for weight in [-128, -127, -1, 0, 1, 127] {
            let w = vec![weight; k]; let expected = [oracle(&a, &w), oracle(&b, &w), 0, oracle(&a, &w)];
            for &kernel in &kernels {
                let mut out = [999; 4]; kernel.dot_rows_into(&[&a, &b, &zeros, &a], &w, &mut out).unwrap();
                assert_eq!(out, expected, "tier={} K={k} w={weight}", kernel.kind());
            }
        }
    }
}

#[test]
fn dense_vnni_widths_preserve_matrix_rows_columns_and_tail_cohorts() {
    let kernels = kernels(); let mut seed = 0x9353_273a_904e_2335;
    for m in [0_usize, 1, 2, 3, 4, 5, 7, 9] {
        for n in [0_usize, 1, 2, 3, 5] {
            for k in [0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 3072] {
                let a: Vec<_> = (0..m * k).map(|_| next(&mut seed)).collect();
                let w: Vec<_> = (0..n * k).map(|_| next(&mut seed)).collect();
                for &kernel in &kernels {
                    let mut out = vec![999; m * n];
                    kernel.gemm_into(&a, m, k, &w, n, &mut out).unwrap();
                    for row in 0..m { for column in 0..n {
                        assert_eq!(out[row * n + column], oracle(&a[row * k..(row + 1) * k],
                            &w[column * k..(column + 1) * k]), "tier={} M={m} N={n} K={k}", kernel.kind());
                    } }
                }
            }
        }
    }
}

#[test]
fn invalid_cohorts_and_matrices_never_modify_the_caller_output() {
    for kernel in kernels() {
        let good = [1; 65]; let bad = [1; 64]; let mut out = [999; 4];
        assert_eq!(kernel.dot_rows_into(&[&good, &good, &good, &bad], &good, &mut out), Err(CpuKernelError::LengthMismatch));
        assert_eq!(out, [999; 4]);
        assert_eq!(kernel.dot_rows_into(&[&good; 4], &good, &mut out[..3]), Err(CpuKernelError::Shape));
        assert_eq!(out, [999; 4]);
        let mut excess = [999; 5];
        assert_eq!(kernel.dot_rows_into(&[&good; 5], &good, &mut excess), Err(CpuKernelError::Shape));
        assert_eq!(excess, [999; 5]);
        let long = vec![0; 10753];
        assert_eq!(kernel.dot(&long, &long), Err(CpuKernelError::ReductionTooLong));
        assert_eq!(kernel.gemm_into(&[1; 129], 2, 65, &[1; 130], 2, &mut out), Err(CpuKernelError::Shape));
        assert_eq!(out, [999; 4]);
        assert_eq!(kernel.gemm_into(&[], usize::MAX, 2, &[], 0, &mut []), Err(CpuKernelError::Shape));
        kernel.gemm_into(&[], usize::MAX, 0, &[], 0, &mut []).unwrap();
        kernel.dot_rows_into(&[&[], &[]], &[], &mut out[..2]).unwrap();
        assert_eq!(out, [0, 0, 999, 999]);
    }
}

#[test]
fn reused_handles_readonly_aliases_and_row_permutations_do_not_share_accumulators() {
    let data: Vec<Vec<i8>> = (0..4).map(|row| (0..129)
        .map(|index| ((index * 53 + row * 71) % 256) as u8 as i8).collect()).collect();
    let order = [3, 1, 0, 2];
    let inputs: Vec<_> = data.iter().map(Vec::as_slice).collect();
    let reordered: Vec<_> = order.iter().map(|&row| inputs[row]).collect();
    for kernel in kernels() {
        let mut original = [0; 4]; let mut permuted = [0; 4];
        kernel.dot_rows_into(&inputs, inputs[0], &mut original).unwrap();
        kernel.dot_rows_into(&reordered, inputs[0], &mut permuted).unwrap();
        for row in 0..4 { assert_eq!(permuted[row], original[order[row]]); }
        kernel.dot_rows_into(&[inputs[0]; 4], inputs[0], &mut permuted).unwrap();
        assert_eq!(permuted, [oracle(inputs[0], inputs[0]); 4]);
    }
}
