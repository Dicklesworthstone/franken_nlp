//! Rectangular tiles must preserve independent rows and selected columns.
use franken_nlp::native_engine::simd::{CpuKernelError, Int8Kernel, Int8KernelKind,
    tiles::{MAX_TILE_COLUMNS, MAX_TILE_VALUES}};

fn kernels() -> Vec<Int8Kernel> {
    Int8KernelKind::ALL.into_iter().filter_map(|kind| match Int8Kernel::detect(kind) {
        Ok(kernel) => Some(kernel),
        Err(CpuKernelError::UnsupportedIsa) => {
            eprintln!("BORROWED_TILE tier={kind} RESULT=SKIPPED_UNSUPPORTED_ISA"); None
        }
        Err(error) => panic!("unexpected kernel error: {error}"),
    }).collect()
}
fn oracle(inputs: &[&[i8]], weights: &[&[i8]]) -> Vec<i32> {
    let mut output = Vec::new();
    for input in inputs {
        for weight in weights {
            let mut sum = 0_i64;
            for index in 0..input.len() { sum += i64::from(input[index]) * i64::from(weight[index]); }
            output.push(i32::try_from(sum).unwrap());
        }
    }
    output
}
fn check(kernel: Int8Kernel, inputs: &[&[i8]], weights: &[&[i8]]) {
    let length = inputs.len() * weights.len();
    let mut output = [919; MAX_TILE_VALUES + 2];
    kernel.dot_tile_into(inputs, weights, &mut output[1..1 + length]).unwrap();
    assert_eq!(&output[1..1 + length], oracle(inputs, weights));
    assert_eq!(output[0], 919);
    assert!(output[1 + length..].iter().all(|&value| value == 919));
}

#[test]
fn borrowed_tiles_match_i64_for_all_live_shapes_alignments_and_k_tails() {
    let kernels = kernels();
    let storage: Vec<Vec<i8>> = (0..6).map(|row| (0..10752 + 64)
        .map(|i| ((i * 53 + row * 97) % 256) as u8 as i8).collect()).collect();
    for m in 1..=4 {
        for n in 1..=MAX_TILE_COLUMNS {
            for offset in 0..64 {
                for k in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 3072, 6144, 10751, 10752] {
                    let inputs: Vec<&[i8]> = (0..m).map(|row| {
                        let start = (offset + row * 7) % 64; &storage[row][start..start + k]
                    }).collect();
                    let weights: Vec<&[i8]> = (0..n).map(|column| {
                        let start = (63 - offset + column * 13) % 64; &storage[4 + column][start..start + k]
                    }).collect();
                    for &kernel in &kernels { check(kernel, &inputs, &weights); }
                }
            }
        }
    }
}

#[test]
fn one_column_output_is_compact_and_two_columns_keep_native_row_major_order() {
    let a = [1_i8, 2, 3, 4, 5, 6, 7, 8];
    let b = [-1_i8, 2, -3, 4, -5, 6, -7, 8];
    let w0 = [2_i8; 8]; let w1 = [-1_i8, 0, 1, 0, -1, 0, 1, 0];
    for kernel in kernels() {
        let mut output = [0; 4];
        kernel.dot_tile_into(&[&a, &b], &[&w0, &w1], &mut output).unwrap();
        assert_eq!(output, [72, 4, 8, -4]);
        kernel.dot_tile_into(&[&a, &b], &[&w0], &mut output[..2]).unwrap();
        assert_eq!(output, [72, 8, 8, -4]);
    }
}

#[test]
fn signed_byte_pairs_keep_independent_activation_and_weight_columns() {
    let kernels = kernels();
    for a in -128_i16..=127 {
        let x = [a as i8; 64]; let inverse = [!(a as i8); 64];
        for w in -128_i16..=127 {
            let first = [w as i8; 64]; let second = [!(w as i8); 64];
            for &kernel in &kernels { check(kernel, &[&x, &inverse], &[&first, &second]); }
        }
    }
}

#[test]
fn sparse_permuted_and_aliased_rows_do_not_require_a_contiguous_matrix() {
    let data: Vec<Vec<i8>> = (0..9).map(|row| (0..65)
        .map(|i| ((i * 29 + row * 73) % 256) as u8 as i8).collect()).collect();
    for kernel in kernels() {
        check(kernel, &[&data[8], &data[1], &data[8], &data[3]], &[&data[7], &data[0]]);
        check(kernel, &[&data[0], &data[7]], &[&data[7], &data[0]]);
        check(kernel, &[&data[4]; 4], &[&data[4]; 2]);
        check(kernel, &[&[-128; 10752]; 4], &[&[-128; 10752], &[127; 10752]]);
    }
}

#[test]
fn invalid_last_weight_and_activation_are_rejected_before_any_write() {
    let good = [1_i8; 33]; let short = [1_i8; 32]; let long = vec![0; 10753];
    for kernel in kernels() {
        let mut output = [919; 8];
        for (inputs, weights) in [
            (vec![&good[..]; 4], vec![&good[..], &short[..]]),
            (vec![&good[..], &good[..], &good[..], &short[..]], vec![&good[..]; 2]),
        ] {
            assert_eq!(kernel.dot_tile_into(&inputs, &weights, &mut output), Err(CpuKernelError::LengthMismatch));
            assert_eq!(output, [919; 8]);
        }
        assert_eq!(kernel.dot_tile_into(&[&good; 4], &[&good; 2], &mut output[..7]), Err(CpuKernelError::Shape));
        assert_eq!(kernel.dot_tile_into(&[&good; 5], &[&good], &mut output[..5]), Err(CpuKernelError::Shape));
        assert_eq!(kernel.dot_tile_into(&[&good], &[&good; 3], &mut output[..3]), Err(CpuKernelError::Shape));
        assert_eq!(kernel.dot_tile_into(&[&long], &[&long], &mut output[..1]), Err(CpuKernelError::ReductionTooLong));
        assert_eq!(kernel.dot_tile_into(&[], &[&good], &mut []), Err(CpuKernelError::Shape));
        assert_eq!(kernel.dot_tile_into(&[&good], &[], &mut []), Err(CpuKernelError::Shape));
        assert_eq!(output, [919; 8]);
    }
}

#[test]
fn native_column_geometry_does_not_change_the_selected_kernel_or_default() {
    assert_eq!(Int8Kernel::default().kind(), Int8KernelKind::Scalar);
    for kernel in kernels() {
        let expected = if kernel.kind() == Int8KernelKind::Aarch64Smmla { 2 } else { 1 };
        assert_eq!(kernel.tile_columns(), expected);
    }
}
