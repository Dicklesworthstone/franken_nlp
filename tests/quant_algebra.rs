use franken_nlp::native_engine::quant_algebra::{
    apply_fixed_scale_order, corrected_x86_offset_dot_i32, s8_slice_to_x86_offset_u8,
    s8_to_x86_offset_u8, signed_dot_i32, sum_int4_groups_in_fixed_order, DigestBoundRowSums,
    EpilogueScales, PhysicalSectionDigest, QuantAlgebraError, RowSumTable, ScaleOperand,
    INT4_GROUP_SUM_ORDER, MAX_MODEL_K, MAX_OFFSET_CORRECTION_K_10752,
    MAX_OFFSET_INTERMEDIATE_K_10752, MAX_S8_S8_ACCUMULATOR_K_10752,
    MAX_U8_S8_RAW_ACCUMULATOR_K_10752, ROW_SUM_TABLE_HEADER_BYTES, S8_ZERO_POINT,
    SCALE_APPLICATION_ORDER, X86_ACTIVATION_XOR_OFFSET,
};

#[test]
fn xor_offset_is_plus_128_for_every_i8_bit_pattern() {
    for value in i8::MIN..=i8::MAX {
        assert_eq!(
            s8_to_x86_offset_u8(value),
            (i16::from(value) + 128) as u8,
            "value={value}"
        );
    }
    assert_eq!(S8_ZERO_POINT, 0);
    assert_eq!(X86_ACTIVATION_XOR_OFFSET, 0x80);
    assert_eq!(
        s8_slice_to_x86_offset_u8(&[i8::MIN, -1, 0, i8::MAX]),
        vec![0, 127, 128, 255]
    );
}

#[test]
fn corrected_offset_dot_matches_signed_dot_at_full_domain_extremes_and_k10752() {
    for &(activation, weight) in &[
        (i8::MIN, i8::MIN),
        (i8::MIN, i8::MAX),
        (i8::MAX, i8::MIN),
        (i8::MAX, i8::MAX),
    ] {
        let activations = vec![activation; MAX_MODEL_K];
        let weights = vec![weight; MAX_MODEL_K];
        assert_eq!(
            corrected_x86_offset_dot_i32(&activations, &weights).unwrap(),
            signed_dot_i32(&activations, &weights).unwrap(),
            "activation={activation} weight={weight}"
        );
    }
}

#[test]
fn published_k10752_integer_bounds_are_safe_with_more_than_fourfold_headroom() {
    assert_eq!(MAX_MODEL_K as i64 * 16_384, MAX_S8_S8_ACCUMULATOR_K_10752);
    assert_eq!(
        MAX_MODEL_K as i64 * 32_640,
        MAX_U8_S8_RAW_ACCUMULATOR_K_10752
    );
    assert_eq!(MAX_MODEL_K as i64 * 16_384, MAX_OFFSET_CORRECTION_K_10752);
    assert_eq!(
        MAX_U8_S8_RAW_ACCUMULATOR_K_10752 + MAX_OFFSET_CORRECTION_K_10752,
        MAX_OFFSET_INTERMEDIATE_K_10752
    );
    assert!(MAX_OFFSET_INTERMEDIATE_K_10752 * 4 < i64::from(i32::MAX));
}

#[test]
fn digest_bound_row_sums_reject_mutated_section_and_mutated_table_before_dispatch() {
    let semantic_weights = [-128_i8, 127, 2, -3, 5, -7, 11, -13];
    let physical = semantic_weights.map(|weight| weight as u8);
    let table = RowSumTable::from_semantic_weights(&semantic_weights, 2, 4).unwrap();
    let binding = table.bind_to_physical_copy("native-int8-row-major", &physical);
    let verified = binding.verify_contiguous_s8_rows(&physical).unwrap();
    let activation = [i8::MIN, -1, 0, i8::MAX];
    assert_eq!(
        verified
            .corrected_x86_offset_dot_i32(0, &activation)
            .unwrap(),
        signed_dot_i32(&activation, &semantic_weights[..4]).unwrap()
    );

    let mut corrupt_section = physical;
    corrupt_section[0] ^= 1;
    assert!(matches!(
        binding.verify_contiguous_s8_rows(&corrupt_section),
        Err(QuantAlgebraError::SectionDigestMismatch { .. })
    ));

    let mut corrupt_table = binding.encode().unwrap();
    corrupt_table[ROW_SUM_TABLE_HEADER_BYTES] ^= 1;
    let corrupt_binding =
        DigestBoundRowSums::decode("native-int8-row-major", &corrupt_table).unwrap();
    assert!(matches!(
        corrupt_binding.verify_contiguous_s8_rows(&physical),
        Err(QuantAlgebraError::RowSumMismatch { row: 0, .. })
    ));
    assert_eq!(
        binding.physical_digest(),
        PhysicalSectionDigest::sha256(&physical)
    );
}

#[test]
fn scale_and_int4_group_orders_are_explicit_and_deterministic() {
    assert_eq!(
        SCALE_APPLICATION_ORDER,
        [
            ScaleOperand::Activation,
            ScaleOperand::Row,
            ScaleOperand::Column,
            ScaleOperand::Group,
        ]
    );
    assert_eq!(INT4_GROUP_SUM_ORDER, "increasing-logical-group-index-v1");
    let result = apply_fixed_scale_order(
        32,
        EpilogueScales {
            activation: 0.25,
            row: 2.0,
            column: 0.5,
            group: 4.0,
        },
    )
    .unwrap();
    assert_eq!(result, 32.0);
    let order_sensitive = EpilogueScales {
        activation: 1.0e20,
        row: 1.0e20,
        column: 1.0e-20,
        group: 1.0e-20,
    };
    assert!(matches!(
        apply_fixed_scale_order(1, order_sensitive),
        Err(QuantAlgebraError::NonFiniteEpilogue)
    ));
    let reversed = (((1.0_f32 * order_sensitive.group) * order_sensitive.column)
        * order_sensitive.row)
        * order_sensitive.activation;
    assert!(
        reversed.is_finite(),
        "reordering must not silently be equivalent"
    );
    assert_eq!(sum_int4_groups_in_fixed_order(&[7, -3, 9, -5]).unwrap(), 8);
}
