//! Differential coverage for streaming scalar INT4. The explicit unpacking API
//! is retained as an independent test oracle, not used by the production path.
use franken_nlp::native_engine::{
    int8::{self, Int8KernelError},
    quant_algebra,
};

#[test]
fn streamed_gemv_matches_unpacked_reference_including_odd_rows_and_unbounded_k() {
    for k in (0_usize..70).chain([3072, 6144, 10752, 10753, 16384]) {
        for n in [0_usize, 2, 6] {
            let input: Vec<i8> = (0..k).map(|i| i.wrapping_mul(97) as u8 as i8).collect();
            let packed: Vec<u8> = (0..n * k / 2).map(|i| i.wrapping_mul(173) as u8).collect();
            let unpacked = int8::unpack_int4_to_i8(&packed);
            assert_eq!(int8::gemv_int4_s8(&input, &packed, n), int8::gemv_s8s8(&input, &unpacked, n),
                "K={k}, N={n}");
        }
    }
}

#[test]
fn every_byte_and_activation_preserves_signed_nibble_order() {
    for byte in 0..=255_u16 {
        let packed = [byte as u8];
        let weights = int8::unpack_int4_to_i8(&packed);
        for activation in -128..=127_i16 {
            let input = [activation as i8, !(activation as i8)];
            let expected = int8::dot_s8s8(&input, &weights).unwrap();
            assert_eq!(int8::gemv_int4_s8(&input, &packed, 1).unwrap(), [expected]);
            assert_eq!(int8::dot_int4_grouped_s8(&input, &packed, 1).unwrap(), expected);
            assert_eq!(int8::dot_int4_grouped_s8(&input, &packed, 2).unwrap(), expected);
        }
    }
}

#[test]
fn odd_group_boundaries_and_tails_use_the_existing_canonical_reduction() {
    for k in [0_usize, 2, 6, 32, 66, 3072, 10752, 16384] {
        let input: Vec<i8> = (0..k).map(|i| i.wrapping_mul(53) as u8 as i8).collect();
        let packed: Vec<u8> = (0..k / 2).map(|i| i.wrapping_mul(197) as u8).collect();
        let unpacked = int8::unpack_int4_to_i8(&packed);
        for width in [1, 2, 3, 7, 31, 32, 33, 10752, usize::MAX] {
            let partials: Vec<i32> = input.chunks(width).zip(unpacked.chunks(width))
                .map(|(a, w)| int8::dot_s8s8(a, w).unwrap()).collect();
            let expected = quant_algebra::sum_int4_groups_in_fixed_order(&partials)
                .map_err(Int8KernelError::CanonicalAlgebra);
            assert_eq!(int8::dot_int4_grouped_s8(&input, &packed, width), expected,
                "K={k}, group width={width}");
        }
    }
}

#[test]
fn layout_is_checked_before_group_admission_and_dimension_allocation() {
    assert_eq!(int8::gemv_int4_s8(&[1, 2], &[0], 2).unwrap_err(),
        Int8KernelError::Int4Layout { expected_weights: 4, observed_weights: 2 });
    assert_eq!(int8::gemv_int4_s8(&[1], &[0], 1).unwrap_err(),
        Int8KernelError::Int4Layout { expected_weights: 1, observed_weights: 2 });
    assert_eq!(int8::gemv_int4_s8(&[1, 2], &[], usize::MAX).unwrap_err(),
        Int8KernelError::LengthOverflow { rows: usize::MAX, columns: 2 });
    assert_eq!(int8::dot_int4_grouped_s8(&[1], &[], 0).unwrap_err(),
        Int8KernelError::Int4Layout { expected_weights: 1, observed_weights: 0 });
    assert_eq!(int8::dot_int4_grouped_s8(&[], &[], 0).unwrap_err(),
        Int8KernelError::Int4GroupWidth { group_width: 0 });
    assert_eq!(int8::dot_int4_grouped_s8(&[], &[], 1).unwrap(), 0);
    assert_eq!(int8::gemv_int4_s8(&[], &[], 3).unwrap(), [0; 3]);
}

#[test]
fn unbounded_scalar_overflow_retains_first_index_and_group_error_kind() {
    // (-128)*(-8) = 1024; the next addition after i32::MAX/1024
    // overflows. Even K keeps the exact no-padding packed representation.
    let k = 2_097_152;
    let input = vec![-128; k];
    let packed = vec![0x88; k / 2];
    let expected = Int8KernelError::AccumulatorOverflow {
        operation: "s8*s8 dot", k_index: k - 1,
    };
    assert_eq!(int8::gemv_int4_s8(&input, &packed, 1).unwrap_err(), expected);
    assert_eq!(int8::dot_int4_grouped_s8(&input, &packed, k).unwrap_err(), expected);
    // Individual groups fit; the fixed-order sum, not an individual dot,
    // owns this failure. Do not wrap, merge groups or change the error layer.
    assert!(matches!(int8::dot_int4_grouped_s8(&input, &packed, k / 2),
        Err(Int8KernelError::CanonicalAlgebra(_))));
}
