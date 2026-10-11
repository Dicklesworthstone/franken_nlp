use super::*;

fn kernels() -> Vec<Int8Kernel> {
    Int8KernelKind::ALL.into_iter().filter_map(|kind| match Int8Kernel::detect(kind) {
        Ok(kernel) => Some(kernel),
        Err(CpuKernelError::UnsupportedIsa) => {
            eprintln!("INT8_KERNEL tier={kind} RESULT=SKIPPED_UNSUPPORTED_ISA"); None
        }
        Err(error) => panic!("kernel construction failed: {error}"),
    }).collect()
}
fn oracle(input: &[i8], weights: &[i8]) -> i32 {
    let sum: i64 = input.iter().zip(weights).map(|(&x, &w)| i64::from(x) * i64::from(w)).sum();
    i32::try_from(sum).unwrap()
}
fn check(kernels: &[Int8Kernel], input: &[i8], weights: &[i8]) {
    let expected = oracle(input, weights);
    assert_eq!(int8::dot_s8s8(input, weights).unwrap(), expected);
    for &kernel in kernels {
        assert_eq!(kernel.dot(input, weights).unwrap(), expected,
            "tier={} K={}", kernel.kind(), input.len());
    }
}
fn next(state: &mut u64) -> i8 {
    *state ^= *state << 13; *state ^= *state >> 7; *state ^= *state << 17;
    (*state >> 32) as u8 as i8
}

#[test]
fn every_byte_pair_in_real_vector_lanes_matches_independent_i64() {
    let kernels = kernels();
    for x in -128_i16..=127 {
        for w in -128_i16..=127 {
            check(&kernels, &[x as i8; 32], &[w as i8; 32]);
        }
    }
}

#[test]
fn extreme_full_model_reductions_do_not_saturate_or_overflow() {
    let kernels = kernels();
    for k in int8::MODEL_KS {
        for x in [-128, -127, -1, 0, 1, 127] {
            for w in [-128, -127, -1, 0, 1, 127] {
                check(&kernels, &vec![x; k], &vec![w; k]);
            }
        }
    }
    assert_eq!(oracle(&vec![-128; MAX_MODEL_K], &vec![-128; MAX_MODEL_K]), 176_160_768);
}

#[test]
fn every_tail_and_both_slice_alignments_match_the_scalar_reference() {
    let kernels = kernels();
    let mut seed = 0x26e4_089b_592f_a1d3_u64;
    let x: Vec<_> = (0..MAX_MODEL_K + 32).map(|_| next(&mut seed)).collect();
    let w: Vec<_> = (0..MAX_MODEL_K + 32).map(|_| next(&mut seed)).collect();
    for start in 0..32 {
        // A different weight offset catches accidental alignment assumptions.
        let weight_start = 31 - start;
        for k in (0..96).chain(int8::MODEL_KS.into_iter().flat_map(|base| (0..32).map(move |tail| base - tail))) {
            check(&kernels, &x[start..start + k], &w[weight_start..weight_start + k]);
        }
    }
}

#[test]
fn alternating_signs_and_single_lane_impulses_exercise_lane_reduction() {
    let kernels = kernels();
    let x: Vec<_> = (0..MAX_MODEL_K).map(|i| if i % 2 == 0 { -128 } else { 127 }).collect();
    let w: Vec<_> = (0..MAX_MODEL_K).map(|i| if i % 3 == 0 { -128 } else { 127 }).collect();
    check(&kernels, &x, &w);
    for lane in 0..65 {
        let mut impulse = [0_i8; 65]; impulse[lane] = -128;
        check(&kernels, &impulse, &[-128; 65]);
    }
}

#[test]
fn saturation_negative_control_would_fail_the_same_fixture() {
    // Quarantined arithmetic model of RAW maddubs: saturation happens BEFORE
    // correction and cannot be repaired by widening or more frequent sums.
    let raw_pair = (2_i32 * 255 * 127).clamp(i16::MIN as i32, i16::MAX as i32);
    let broken_pair = raw_pair - 128 * (2 * 127);
    assert_ne!(broken_pair, 2 * 127 * 127);
    check(&kernels(), &[127; 64], &[127; 64]);
    check(&kernels(), &[127; 64], &[-128; 64]);
}

#[test]
fn shape_rejection_occurs_before_any_output_write() {
    for kernel in kernels() {
        assert_eq!(kernel.dot(&[1, 2], &[1]), Err(CpuKernelError::LengthMismatch));
        let long = vec![0_i8; MAX_MODEL_K + 1];
        assert_eq!(kernel.dot(&long, &long), Err(CpuKernelError::ReductionTooLong));
        let mut output = [123_i32; 6];
        assert_eq!(kernel.gemm_into(&[1; 7], 2, 4, &[1; 12], 3, &mut output), Err(CpuKernelError::Shape));
        assert_eq!(output, [123; 6]);
        assert_eq!(kernel.gemm_into(&[], usize::MAX, 2, &[], 1, &mut output), Err(CpuKernelError::Shape));
        assert_eq!(output, [123; 6]);
    }
}

#[test]
fn gemm_matches_independent_rows_for_skinny_and_tail_tiles() {
    let kernels = kernels();
    let mut seed = 0x253a_4b84_90b7_22f1;
    for m in [1, 2, 4, 7] {
        for n in [1, 3, 8] {
            for k in [0, 1, 15, 31, 32, 33, 67, 3072] {
                let x: Vec<_> = (0..m * k).map(|_| next(&mut seed)).collect();
                let w: Vec<_> = (0..n * k).map(|_| next(&mut seed)).collect();
                for &kernel in &kernels {
                    let mut output = vec![123; m * n];
                    kernel.gemm_into(&x, m, k, &w, n, &mut output).unwrap();
                    for row in 0..m {
                        for column in 0..n {
                            assert_eq!(output[row * n + column], oracle(&x[row * k..(row + 1) * k], &w[column * k..(column + 1) * k]));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn empty_shapes_terminate_and_zero_width_dots_are_zero() {
    for kernel in kernels() {
        assert_eq!(kernel.dot(&[], &[]), Ok(0));
        kernel.gemm_into(&[], usize::MAX, 0, &[], 0, &mut []).unwrap();
        let mut output = [17; 6];
        kernel.gemm_into(&[], 3, 0, &[], 2, &mut output).unwrap();
        assert_eq!(output, [0; 6]);
    }
}

#[test]
fn explicit_kind_roundtrip_and_executable_handle_is_send_sync() {
    for kind in Int8KernelKind::ALL { assert_eq!(kind.id().parse::<Int8KernelKind>().unwrap(), kind); }
    for bad in ["avx2", "auto", "fastest", "", "x3b_avx2_widened_i16 "] {
        assert_eq!(bad.parse::<Int8KernelKind>(), Err(CpuKernelError::UnknownKind));
    }
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Int8Kernel>();
    assert_eq!(Int8Kernel::default().kind(), Int8KernelKind::Scalar);
}
