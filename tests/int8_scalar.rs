#[path = "support/int8_oracle.rs"]
mod int8_oracle;

use franken_nlp::native_engine::int8::{
    MAX_S8_S8_K_10752, MODEL_KS, dot_s8s8, gemm_s8s8, gemv_int4_s8, gemv_s8s8, is_model_shape,
    unpack_int4_to_i8,
};
use int8_oracle::{
    DomainProfile, FixtureGenerator, FixtureShape, fixture_digest, model_shape_catalog,
    verify_declared_bounds, verify_int4_fixture, verify_offset_fixture, verify_s8_fixture,
};

#[test]
fn tiny_gemm_gemv_and_int4_goldens_are_exact() {
    assert_eq!(dot_s8s8(&[1, -2, 3], &[4, 5, -6]).unwrap(), -24);
    assert_eq!(gemv_s8s8(&[1, 2], &[3, 4, -5, 6], 2).unwrap(), vec![11, 7]);
    assert_eq!(
        gemm_s8s8(&[1, 2, -1, 3], 2, 2, &[3, 4, -5, 6], 2).unwrap(),
        vec![11, 7, 9, 23]
    );
    assert_eq!(unpack_int4_to_i8(&[0x78, 0xf0]), vec![-8, 7, 0, -1]);
    assert_eq!(
        gemv_int4_s8(&[1, 2, 3, 4], &[0x78, 0xf0], 1).unwrap(),
        vec![2]
    );
}

#[test]
fn dynamic_m_gemm_rows_each_preserve_the_scalar_dot_order() {
    let m = 3;
    let k = 7;
    let n = 3;
    let activations = [
        -128, -1, 0, 1, 127, -64, 64, 127, 1, 0, -1, -128, 64, -64, 3, -5, 7, -11, 13, -17, 19,
    ];
    let weights = [
        -128, 127, -3, 5, -7, 11, -13, 17, -19, 23, -29, 31, -37, 41, 43, -47, 53, -59, 61, -67, 71,
    ];
    let actual = gemm_s8s8(&activations, m, k, &weights, n).unwrap();
    for row in 0..m {
        for column in 0..n {
            let input = &activations[row * k..(row + 1) * k];
            let weight = &weights[column * k..(column + 1) * k];
            assert_eq!(actual[row * n + column], dot_s8s8(input, weight).unwrap());
        }
    }
}

#[test]
fn every_fixed_shape_and_bound_is_catalogued_without_model_weights() {
    verify_declared_bounds().unwrap();
    for shape in model_shape_catalog() {
        assert!(
            is_model_shape(shape.k, shape.n),
            "missing model shape {shape:?}"
        );
    }
    let worst = FixtureGenerator::new(0x9a71_u64).dot_fixtures(
        FixtureShape {
            m: 1,
            n: 1,
            k: 10752,
        },
        DomainProfile::AllExtremes,
    );
    for fixture in worst {
        let report = verify_s8_fixture(&fixture).unwrap();
        if fixture.input[0] == -128 && fixture.weights[0] == -128 {
            assert_eq!(report.expected, 176_160_768);
            assert_eq!(i64::from(report.expected), MAX_S8_S8_K_10752);
        }
    }
}

#[test]
fn full_domain_property_vectors_match_i64_for_each_model_k_and_n() {
    let generator = FixtureGenerator::new(0x5eed_9eed);
    for shape in model_shape_catalog() {
        for fixture in generator.dot_fixtures(shape, DomainProfile::RandomFull { cases: 3 }) {
            verify_s8_fixture(&fixture).unwrap_or_else(|error| panic!("{error}"));
        }
    }
}

#[test]
fn exhaustive_small_and_all_tail_lengths_match_the_i64_floor() {
    let generator = FixtureGenerator::new(0x1ee7);
    for fixture in generator.dot_fixtures(
        FixtureShape { m: 1, n: 1, k: 1 },
        DomainProfile::ExhaustiveSmall,
    ) {
        verify_s8_fixture(&fixture).unwrap_or_else(|error| panic!("{error}"));
    }
    for width in [1, 8, 16, 32, 64] {
        for fixture in generator.tail_fixtures(width) {
            verify_s8_fixture(&fixture).unwrap_or_else(|error| panic!("{error}"));
        }
    }
}

#[test]
fn offset_correction_and_int4_group_cases_stay_exact() {
    let generator = FixtureGenerator::new(0x0ff5e7);
    for &k in &MODEL_KS {
        for fixture in generator.offset_extremes(k) {
            verify_offset_fixture(&fixture).unwrap_or_else(|error| panic!("{error}"));
        }
    }
    for group_size in [16, 32] {
        for fixture in generator.int4_group_fixtures(group_size) {
            verify_int4_fixture(&fixture).unwrap_or_else(|error| panic!("{error}"));
        }
    }
}

#[test]
fn fixture_generator_is_seed_deterministic_and_logs_its_adoption() {
    let shape = FixtureShape {
        m: 3,
        n: 1024,
        k: 3072,
    };
    let first =
        FixtureGenerator::new(0xdec0_de01).dot_fixtures(shape, DomainProfile::AlternatingExtremes);
    let second =
        FixtureGenerator::new(0xdec0_de01).dot_fixtures(shape, DomainProfile::AlternatingExtremes);
    let first_digests = first.iter().map(fixture_digest).collect::<Vec<_>>();
    let second_digests = second.iter().map(fixture_digest).collect::<Vec<_>>();
    assert_eq!(first_digests, second_digests);
    eprintln!(
        "INT8_SCALAR fixture_consumer=self_test seed=0xdec0de01 fixtures={} digests={:?}",
        first_digests.len(),
        first_digests
    );
    eprintln!("INT8_SCALAR RESULT=PASS mismatches=0");
}
