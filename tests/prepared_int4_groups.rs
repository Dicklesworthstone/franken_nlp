//! Reusable unpacking must preserve scalar nibble semantics and caller storage.
use franken_nlp::native_engine::{int4::PackedInt4Kernel,
    simd::{CpuKernelError, Int8KernelKind}, quant_algebra::MAX_MODEL_K};

fn kernels() -> Vec<PackedInt4Kernel> {
    Int8KernelKind::ALL.into_iter().filter_map(|kind| match PackedInt4Kernel::detect(kind) {
        Ok(kernel) => Some(kernel),
        Err(CpuKernelError::UnsupportedIsa) => {
            eprintln!("INT4_PREPARED tier={kind} RESULT=SKIPPED_UNSUPPORTED_ISA"); None
        }
        Err(error) => panic!("unexpected kernel refusal: {error}"),
    }).collect()
}
fn nibble(bytes: &[u8], index: usize) -> i64 {
    let value = i64::from((bytes[index / 2] >> (4 * (index % 2))) & 15);
    if value >= 8 { value - 16 } else { value }
}
fn oracle(input: &[i8], bytes: &[u8], offset: usize) -> i32 {
    i32::try_from(input.iter().enumerate().map(|(i, &x)| i64::from(x) * nibble(bytes, offset + i)).sum::<i64>()).unwrap()
}

#[test]
fn every_packed_byte_and_both_start_nibbles_match_independent_signed_reference() {
    for kernel in kernels() {
        for byte in 0_u16..=255 {
            for offset in [0, 1] {
                let bytes = [byte as u8; 3]; let mut scratch = [99; 8];
                let group = kernel.prepare_range(&bytes, offset, 5, &mut scratch).unwrap();
                let a = [-128, 127, 0, -1, 1]; let b = [127, -128, 1, 0, -1];
                let mut output = [123; 2];
                group.dot_rows_into(&[&a, &b], &mut output).unwrap();
                assert_eq!(output, [oracle(&a, &bytes, offset), oracle(&b, &bytes, offset)]);
                assert_eq!(group.kind(), kernel.kind()); assert_eq!(group.len(), 5);
                assert!(!group.is_empty());
                assert_eq!(&scratch[5..], &[99; 3]);
            }
        }
    }
}

#[test]
fn one_preparation_serves_many_tiles_without_borrowing_the_original_packed_source() {
    for kernel in kernels() {
        let mut bytes = [0x87, 0xf1, 0x23, 0x45]; let snapshot = bytes;
        let mut scratch = [99; 7];
        let group = kernel.prepare_range(&bytes, 1, 7, &mut scratch).unwrap();
        // The signature borrows scratch only. Reusing the packed source cannot
        // alter the already prepared immutable group or introduce stale reads.
        bytes.fill(0);
        for row in 0..64 {
            let input = [row as i8 - 32; 7]; let mut output = [123];
            group.dot_rows_into(&[&input], &mut output).unwrap();
            assert_eq!(output[0], oracle(&input, &snapshot, 1));
        }
        let next = kernel.prepare_range(&bytes, 0, 7, &mut scratch).unwrap();
        let mut output = [123]; next.dot_rows_into(&[&[-128; 7]], &mut output).unwrap();
        assert_eq!(output, [0]);
    }
}

#[test]
fn reusable_groups_preserve_all_vector_boundaries_and_maximum_reductions() {
    for kernel in kernels() {
        for k in [1_usize, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129, 3072, MAX_MODEL_K] {
            let bytes: Vec<_> = (0..k.div_ceil(2) + 2).map(|i| (i * 73 + 29) as u8).collect();
            let mut scratch = vec![99; k + 3];
            let group = kernel.prepare_range(&bytes, 1, k, &mut scratch).unwrap();
            let storage: Vec<Vec<i8>> = (0..9).map(|row| (0..k).map(|i| (i * 53 + row * 29) as u8 as i8).collect()).collect();
            for tile in storage.chunks(4) {
                let inputs: Vec<&[i8]> = tile.iter().map(Vec::as_slice).collect();
                let mut output = vec![123; tile.len()]; group.dot_rows_into(&inputs, &mut output).unwrap();
                for (value, input) in output.iter().zip(tile) { assert_eq!(*value, oracle(input, &bytes, 1)); }
            }
            assert_eq!(&scratch[k..], &[99; 3]);
        }
    }
}

#[test]
fn invalid_preparation_never_changes_scratch_including_overflow_and_empty_bounds() {
    for kernel in kernels() {
        for (offset, k, space) in [(0, 5, 8), (usize::MAX, 1, 8), (5, 0, 8), (0, 4, 3), (0, MAX_MODEL_K + 1, 8)] {
            let mut scratch = [99; 8];
            assert!(kernel.prepare_range(&[0x87; 2], offset, k, &mut scratch[..space]).is_err());
            assert_eq!(scratch, [99; 8]);
        }
    }
}

#[test]
fn malformed_execution_cannot_partially_write_and_does_not_invalidate_prepared_bytes() {
    for kernel in kernels() {
        let mut scratch = [99; 33];
        let group = kernel.prepare_range(&[0x88; 17], 0, 33, &mut scratch).unwrap();
        let good = [-128; 33]; let short = [-128; 32]; let mut output = [123; 5];
        assert_eq!(group.dot_rows_into(&[&good, &good, &good, &short], &mut output[..4]), Err(CpuKernelError::LengthMismatch));
        assert_eq!(group.dot_rows_into(&[&good; 5], &mut output), Err(CpuKernelError::Shape));
        assert_eq!(group.dot_rows_into(&[&good], &mut output[..2]), Err(CpuKernelError::Shape));
        assert_eq!(output, [123; 5]);
        group.dot_rows_into(&[&good; 4], &mut output[..4]).unwrap();
        assert_eq!(output, [33 * 1024, 33 * 1024, 33 * 1024, 33 * 1024, 123]);
    }
}

#[test]
fn empty_groups_and_empty_legacy_tiles_keep_the_no_write_contract() {
    for kernel in kernels() {
        let mut scratch = [99; 8];
        kernel.dot_rows_range_into(&[], &[0x87; 4], 0, 8, &mut [], &mut scratch).unwrap();
        assert_eq!(scratch, [99; 8]);
        let group = kernel.prepare_range(&[0x87], 2, 0, &mut scratch).unwrap();
        assert!(group.is_empty()); group.dot_rows_into(&[], &mut []).unwrap();
        let mut output = [123; 2]; group.dot_rows_into(&[&[], &[]], &mut output).unwrap();
        assert_eq!(output, [0; 2]); assert_eq!(scratch, [99; 8]);
    }
}
