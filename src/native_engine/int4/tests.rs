use super::*;
use crate::native_engine::int8;

fn signed(nibble: u8) -> i64 {
    if nibble < 8 { i64::from(nibble) } else { i64::from(nibble) - 16 }
}

fn reference(a: &[i8], m: usize, k: usize, w: &[u8], n: usize) -> Vec<i32> {
    let mut out = vec![0; m * n];
    for row in 0..m {
        for column in 0..n {
            let mut sum = 0_i64;
            for i in 0..k {
                let index = column * k + i;
                let nibble = (w[index / 2] >> ((index % 2) * 4)) & 15;
                sum += i64::from(a[row * k + i]) * signed(nibble);
            }
            out[row * n + column] = i32::try_from(sum).unwrap();
        }
    }
    out
}

#[test]
fn every_packed_byte_and_signed_activation_matches_i64() {
    let kernel = PackedInt4Kernel::scalar();
    let mut scratch = [0; 2];
    let mut output = [0; 1];
    for byte in 0..=255_u16 {
        for activation in -128..=127_i16 {
            let a = [activation as i8, !(activation as i8)];
            let w = [byte as u8];
            kernel.gemv_into(&a, &w, 1, &mut output, &mut scratch).unwrap();
            assert_eq!(output[0], reference(&a, 1, 2, &w, 1)[0]);
        }
    }
}

#[test]
fn available_native_leaves_match_i64_and_existing_scalar_across_tails() {
    for kind in Int8KernelKind::ALL {
        let kernel = match PackedInt4Kernel::detect(kind) {
            Ok(kernel) => kernel,
            Err(CpuKernelError::UnsupportedIsa) => continue,
            Err(error) => panic!("unexpected detection error: {error}"),
        };
        assert_eq!(kernel.kind(), kind);
        for k in [0, 1, 2, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 3072, 6144, 10752] {
            for m in [1, 4, 5] {
                // Odd K deliberately places alternating row starts in the
                // high nibble. The exact flat layout contains no row padding.
                let n = if k % 2 == 0 { 3 } else { 2 };
                let a: Vec<i8> = (0..m * k).map(|i| (i.wrapping_mul(97) as u8) as i8).collect();
                let w: Vec<u8> = (0..n * k / 2).map(|i| i.wrapping_mul(173) as u8).collect();
                let mut scratch = vec![0; PackedInt4Kernel::scratch_len(k).unwrap()];
                let mut output = vec![i32::MIN; m * n];
                kernel.gemm_into(&a, m, k, &w, n, &mut output, &mut scratch).unwrap();
                assert_eq!(output, reference(&a, m, k, &w, n), "{kind}, M={m}, K={k}");
                if m == 1 { assert_eq!(output, int8::gemv_int4_s8(&a, &w, n).unwrap()); }
            }
        }
    }
}

#[test]
fn worst_case_int4_reduction_is_exact_at_model_bound() {
    let k = MAX_MODEL_K;
    let a = vec![-128; k];
    let w = vec![0x88; k / 2];
    let mut scratch = vec![0; k];
    let mut output = [0];
    for kind in Int8KernelKind::ALL {
        match PackedInt4Kernel::detect(kind) {
            Ok(kernel) => {
                kernel.gemv_into(&a, &w, 1, &mut output, &mut scratch).unwrap();
                assert_eq!(output[0], (k * 1024) as i32);
            }
            Err(CpuKernelError::UnsupportedIsa) => {}
            Err(error) => panic!("unexpected detection error: {error}"),
        }
    }
}

#[test]
fn malformed_geometry_never_writes_output_or_scratch() {
    let kernel = PackedInt4Kernel::scalar();
    let mut output = [123; 2];
    let mut scratch = [99; 2];
    assert!(kernel.gemm_into(&[1], 1, 2, &[0, 0], 2, &mut output, &mut scratch).is_err());
    assert!(kernel.gemm_into(&[1, 2], 1, 2, &[0], 2, &mut output, &mut scratch).is_err());
    assert!(kernel.gemm_into(&[1, 2], 1, 2, &[0, 0, 0], 2, &mut output, &mut scratch).is_err());
    assert!(kernel.gemm_into(&[1, 2], 1, 2, &[0, 0], 2, &mut output, &mut scratch[..1]).is_err());
    assert!(kernel.gemm_into(&[1], 1, 1, &[0], 1, &mut output[..1], &mut scratch).is_err());
    assert!(kernel.gemm_into(&[], usize::MAX, 2, &[], 2, &mut output, &mut scratch).is_err());
    assert!(kernel.gemm_into(&[], 0, 0, &[], usize::MAX, &mut output, &mut scratch).is_err());
    assert!(kernel.gemm_into(&[], 0, MAX_MODEL_K + 1, &[], 0, &mut [], &mut scratch).is_err());
    assert_eq!(output, [123; 2]);
    assert_eq!(scratch, [99; 2]);
}

#[test]
fn empty_shapes_and_extra_scratch_are_well_defined() {
    let kernel = PackedInt4Kernel::scalar();
    let mut output = [7; 6];
    kernel.gemm_into(&[], 2, 0, &[], 3, &mut output, &mut []).unwrap();
    assert_eq!(output, [0; 6]);
    kernel.gemm_into(&[], 0, 2, &[0x12], 1, &mut [], &mut [0; 2]).unwrap();
    kernel.gemm_into(&[1, 2], 1, 2, &[], 0, &mut [], &mut [0; 2]).unwrap();
    let mut scratch = [77; 4];
    let mut result = [0];
    kernel.gemv_into(&[1, 2], &[0x8f], 1, &mut result, &mut scratch).unwrap();
    assert_eq!(result, [-17]);
    assert_eq!(&scratch[2..], &[77; 2]);
}

#[test]
fn unpacking_scratch_does_not_grow_with_weight_matrix() {
    let k = 64;
    let n = 166_144;
    let w = vec![0x11; n * k / 2];
    let mut output = vec![0; n];
    let mut scratch = vec![0; PackedInt4Kernel::scratch_len(k).unwrap()];
    PackedInt4Kernel::scalar().gemv_into(&vec![1; k], &w, n, &mut output, &mut scratch).unwrap();
    assert!(output.iter().all(|&sum| sum == k as i32));
    assert_eq!(scratch.len(), k);
}
