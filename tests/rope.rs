#![deny(unsafe_code)]

use franken_nlp::native_engine::{
    rope::{
        DEFAULT_ADMITTED_CONTEXT_CAP, NANBEIGE_HEAD_DIM, NANBEIGE_ROPE_THETA, RopeError,
        RopeProjectionVariant, RopeTablesF32,
    },
    tensor::Bf16,
};

fn assert_close(observed: f32, expected: f32, tolerance: f32) {
    assert!(
        (observed - expected).abs() <= tolerance,
        "observed={observed:.9e} expected={expected:.9e} tolerance={tolerance:.9e}"
    );
}

fn pair_norm(left: f32, right: f32) -> f32 {
    left.mul_add(left, right * right)
}

#[test]
fn nanbeige_inverse_frequency_goldens_and_admitted_cap_are_exact() {
    let tables = RopeTablesF32::nanbeige(17).expect("small admitted cap is valid");
    assert_eq!(tables.head_dim(), NANBEIGE_HEAD_DIM);
    assert_eq!(tables.position_count(), 17);
    for lane in [0, 1, 63] {
        let expected = NANBEIGE_ROPE_THETA.powf(-(2.0 * lane as f32) / 128.0);
        assert_close(tables.inverse_frequency(lane).unwrap(), expected, 1.0e-12);
    }
    for position in [0, 1, 16] {
        for lane in 0..(NANBEIGE_HEAD_DIM / 2) {
            let phase = position as f32
                * NANBEIGE_ROPE_THETA.powf(-(2.0 * lane as f32) / NANBEIGE_HEAD_DIM as f32);
            let (cosine, sine) = tables.table_value(position, lane).unwrap();
            assert_close(cosine, phase.cos(), 1.0e-7);
            assert_close(sine, phase.sin(), 1.0e-7);
        }
    }
    assert_eq!(tables.table_value(0, 0), Ok((1.0, 0.0)));
    assert!(matches!(
        tables.table_value(17, 0),
        Err(RopeError::PositionOutOfRange {
            position: 17,
            table_positions: 17,
        })
    ));
    assert!(matches!(
        tables.inverse_frequency(64),
        Err(RopeError::FrequencyLaneOutOfRange {
            lane: 64,
            half_dim: 64,
        })
    ));
    assert_eq!(DEFAULT_ADMITTED_CONTEXT_CAP, 8_192);
    assert_eq!(
        RopeTablesF32::nanbeige_default_admission()
            .expect("the default admission allocates only the default cap")
            .position_count(),
        DEFAULT_ADMITTED_CONTEXT_CAP
    );
}

#[test]
fn split_half_pairs_first_and_second_halves_not_adjacent_lanes() {
    let tables = RopeTablesF32::new(2, 4, 2.0).unwrap();
    let mut vector = [1.0_f32, 2.0, 3.0, 4.0];
    tables.apply_split_half_f32(1, &mut vector).unwrap();

    let (cosine0, sine0) = tables.table_value(1, 0).unwrap();
    let (cosine1, sine1) = tables.table_value(1, 1).unwrap();
    assert_close(vector[0], cosine0 - 3.0 * sine0, 1.0e-6);
    assert_close(vector[2], 3.0 * cosine0 + sine0, 1.0e-6);
    assert_close(vector[1], 2.0 * cosine1 - 4.0 * sine1, 1.0e-6);
    assert_close(vector[3], 4.0 * cosine1 + 2.0 * sine1, 1.0e-6);

    let adjacent_lane_zero = cosine0 - 2.0 * sine0;
    assert!(
        (vector[0] - adjacent_lane_zero).abs() > 1.0e-3,
        "the split-half result must not collapse to adjacent pairing"
    );
}

#[test]
fn position_zero_is_identity_for_f32_and_bf16_cast_profile() {
    let tables = RopeTablesF32::new(3, 4, NANBEIGE_ROPE_THETA).unwrap();
    let mut f32_vector = [1.0_f32, -2.5, 3.25, -4.0];
    let original_f32 = f32_vector;
    tables.apply_split_half_f32(0, &mut f32_vector).unwrap();
    assert_eq!(f32_vector, original_f32);

    let mut bf16_vector = original_f32.map(Bf16::from_f32);
    let original_bf16 = bf16_vector;
    tables.apply_split_half(0, &mut bf16_vector).unwrap();
    assert_eq!(bf16_vector, original_bf16);
}

#[test]
fn projection_epilogue_candidate_is_bit_equal_to_unfused_for_q_and_k() {
    let tables = RopeTablesF32::nanbeige(6).unwrap();
    let query = (0..(48 * NANBEIGE_HEAD_DIM))
        .map(|value| value as f32 * 0.03125 - 5.5)
        .collect::<Vec<_>>();
    let key = (0..(8 * NANBEIGE_HEAD_DIM))
        .map(|value| value as f32 * -0.25 + 1.0)
        .collect::<Vec<_>>();
    let mut unfused_query = query.clone();
    let mut unfused_key = key.clone();
    let mut fused_query = query;
    let mut fused_key = key;

    tables
        .apply_projected_qk_unfused(5, &mut unfused_query, &mut unfused_key)
        .unwrap();
    tables
        .apply_projected_qk_fused_epilogue(5, &mut fused_query, &mut fused_key)
        .unwrap();
    assert_eq!(fused_query, unfused_query);
    assert_eq!(fused_key, unfused_key);

    let mut default_query = (0..(48 * NANBEIGE_HEAD_DIM))
        .map(|value| value as f32 * 0.03125 - 5.5)
        .collect::<Vec<_>>();
    let mut default_key = (0..(8 * NANBEIGE_HEAD_DIM))
        .map(|value| value as f32 * -0.25 + 1.0)
        .collect::<Vec<_>>();
    assert_eq!(
        RopeProjectionVariant::default(),
        RopeProjectionVariant::Unfused
    );
    tables
        .apply_projected_qk(
            RopeProjectionVariant::default(),
            5,
            &mut default_query,
            &mut default_key,
        )
        .unwrap();
    assert_eq!(default_query, unfused_query);
    assert_eq!(default_key, unfused_key);

    let mut second_loop_query = (0..(48 * NANBEIGE_HEAD_DIM))
        .map(|value| value as f32 * 0.03125 - 5.5)
        .collect::<Vec<_>>();
    tables
        .apply_split_half_f32_all_heads(5, &mut second_loop_query)
        .unwrap();
    assert_eq!(second_loop_query, unfused_query, "both loop passes use P=5");
}

#[test]
fn split_half_rotation_preserves_every_pair_norm_and_preflight_rejects() {
    let tables = RopeTablesF32::new(5, 8, 3.0).unwrap();
    let mut vector = [-3.0_f32, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let before = (0..4)
        .map(|lane| pair_norm(vector[lane], vector[lane + 4]))
        .collect::<Vec<_>>();
    tables.apply_split_half_f32(4, &mut vector).unwrap();
    for (lane, before) in before.into_iter().enumerate() {
        assert_close(pair_norm(vector[lane], vector[lane + 4]), before, 2.0e-5);
    }
    assert!(matches!(
        tables.apply_split_half_f32(5, &mut vector),
        Err(RopeError::PositionOutOfRange { .. })
    ));
    assert!(matches!(
        RopeTablesF32::new(1, 7, 2.0),
        Err(RopeError::OddHeadDimension { head_dim: 7 })
    ));
    assert!(matches!(
        RopeTablesF32::new(0, 8, 2.0),
        Err(RopeError::ZeroAdmittedContext)
    ));
    assert!(matches!(
        RopeTablesF32::new(1, 8, 0.0),
        Err(RopeError::InvalidTheta { .. })
    ));
    eprintln!("ROPE RESULT=PASS positions=5 head_dim=8 fused_default=off");
}
