#![deny(unsafe_code)]

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use franken_nlp::native_engine::{
    rope::{
        DEFAULT_ADMITTED_CONTEXT_CAP, NANBEIGE_HEAD_DIM, NANBEIGE_ROPE_THETA, RopeError,
        RopeProjectionVariant, RopeTablesF32,
    },
    tensor::Bf16,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const ORACLE_FIXTURE: &str = include_str!("fixtures/rope_oracle_f32.json");
const DECODE_APPLICATION_FIXTURE: &str = include_str!("fixtures/rope_oracle_decode_bf16.json");
const PINNED_REVISION: &str = "f56ec5a9650268aa098496734743c25ea778bd2d";
const PINNED_SOURCE: &str = "docs/truth-pack/research/modeling_nanbeige.py";
const PINNED_SOURCE_SHA256: &str =
    "547737f989f5cb741c1a568acaf85f83521fa67a8f7268e19f9a37e60127c0d5";

#[derive(Debug, Deserialize)]
struct RopeOracleFixture {
    schema_version: u32,
    model_id: String,
    revision: String,
    profile: String,
    application: RopeApplicationFixture,
    producer: RopeOracleProducer,
    head_dim: usize,
    theta: u64,
    admitted_cap: usize,
    rows: Vec<RopeOracleRow>,
}

#[derive(Debug, Deserialize)]
struct RopeOracleProducer {
    source_path: String,
    source_sha256: String,
    source_lines: String,
    oracle_python: String,
    torch: String,
}

#[derive(Debug, Deserialize)]
struct RopeOracleRow {
    position: usize,
    lane: usize,
    f32_cosine_bits: String,
    f32_sine_bits: String,
    bf16_cosine_bits: String,
    bf16_sine_bits: String,
}

#[derive(Debug, Deserialize)]
struct RopeApplicationFixture {
    capture_schema_version: u32,
    cosine_bf16_hex: String,
    input_ids: Vec<u32>,
    key_head: usize,
    key_input_bf16_hex: String,
    key_rotated_bf16_hex: String,
    layer: usize,
    #[serde(rename = "loop")]
    loop_index: usize,
    modeling_source_sha256: String,
    phase: String,
    position: usize,
    profile: String,
    query_head: usize,
    query_input_bf16_hex: String,
    query_rotated_bf16_hex: String,
    prior_capture_source_closure_verification: String,
    reproduction_receipt: RopeApplicationReproductionReceipt,
    sine_bf16_hex: String,
    torch: String,
}

#[derive(Debug, Deserialize)]
struct RopeApplicationReproductionReceipt {
    capture_scope: String,
    generator_commit: String,
    generator_path: String,
    model_id: String,
    model_source_closure_verification: String,
    oracle_env_record_sha256: String,
    pinned_revision: String,
    source_manifest_sha256: String,
}

#[derive(Debug, Deserialize)]
struct RopeDecodeApplicationFixture {
    application: RopeDecodeApplication,
    receipt: RopeApplicationReproductionReceipt,
}

#[derive(Debug, Deserialize)]
struct RopeDecodeApplication {
    capture_schema_version: u32,
    cosine_bf16_hex: String,
    input_ids: Vec<u32>,
    key_head: usize,
    key_input_bf16_hex: String,
    key_rotated_bf16_hex: String,
    layer: usize,
    #[serde(rename = "loop")]
    loop_index: usize,
    modeling_source_sha256: String,
    phase: String,
    position: usize,
    prefill_input_ids: Vec<u32>,
    profile: String,
    query_head: usize,
    query_input_bf16_hex: String,
    query_rotated_bf16_hex: String,
    sine_bf16_hex: String,
    torch: String,
}

fn assert_close(observed: f32, expected: f32, tolerance: f32) {
    assert!(
        (observed - expected).abs() <= tolerance,
        "observed={observed:.9e} expected={expected:.9e} tolerance={tolerance:.9e}"
    );
}

fn pair_norm(left: f32, right: f32) -> f32 {
    left.mul_add(left, right * right)
}

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn sha256_path(path: &Path) -> String {
    format!(
        "{:x}",
        Sha256::digest(fs::read(path).unwrap_or_else(|error| {
            panic!("read {} for fixture provenance: {error}", path.display())
        }))
    )
}

fn parse_u32_bits(bits: &str, field: &str) -> u32 {
    assert_eq!(bits.len(), 8, "oracle {field} must have eight hex digits");
    u32::from_str_radix(bits, 16)
        .unwrap_or_else(|error| panic!("parse oracle {field}={bits:?} as u32: {error}"))
}

fn parse_u16_bits(bits: &str, field: &str) -> u16 {
    assert_eq!(bits.len(), 4, "oracle {field} must have four hex digits");
    u16::from_str_radix(bits, 16)
        .unwrap_or_else(|error| panic!("parse oracle {field}={bits:?} as u16: {error}"))
}

fn parse_bf16_vector(bits: &str, field: &str, expected_elements: usize) -> Vec<Bf16> {
    assert_eq!(
        bits.len(),
        expected_elements * 4,
        "oracle {field} must encode exactly {expected_elements} bf16 values"
    );
    bits.as_bytes()
        .chunks_exact(4)
        .enumerate()
        .map(|(index, chunk)| {
            let chunk = std::str::from_utf8(chunk).unwrap_or_else(|error| {
                panic!("oracle {field} chunk={index} is not UTF-8: {error}")
            });
            let value = u16::from_str_radix(chunk, 16).unwrap_or_else(|error| {
                panic!("parse oracle {field} chunk={index} value={chunk:?} as bf16: {error}")
            });
            Bf16::from_bits(value)
        })
        .collect()
}

fn real_shape_values(elements: usize, multiplier: f32, offset: f32) -> Vec<f32> {
    (0..elements)
        .map(|index| index as f32 * multiplier + offset)
        .collect()
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
fn pinned_oracle_rope_fixture_covers_boundary_positions_and_casts() {
    let fixture: RopeOracleFixture =
        serde_json::from_str(ORACLE_FIXTURE).expect("the pinned RoPE oracle fixture is valid JSON");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.model_id, "Nanbeige/Nanbeige4.2-3B");
    assert_eq!(fixture.revision, PINNED_REVISION);
    assert_eq!(fixture.profile, "hf-bf16-eager");
    assert_eq!(fixture.producer.source_path, PINNED_SOURCE);
    assert_eq!(fixture.producer.source_sha256, PINNED_SOURCE_SHA256);
    assert_eq!(
        sha256_path(&repo_path(PINNED_SOURCE)),
        fixture.producer.source_sha256,
        "the fixture must remain bound to the archived pinned oracle source"
    );
    assert_eq!(fixture.producer.source_lines, "939-967");
    assert_eq!(fixture.producer.oracle_python, "3.11.14");
    assert_eq!(fixture.producer.torch, "2.6.0");
    assert_eq!(fixture.head_dim, NANBEIGE_HEAD_DIM);
    assert_eq!(fixture.theta, NANBEIGE_ROPE_THETA as u64);
    assert!(fixture.admitted_cap > 1);

    let tables = RopeTablesF32::nanbeige(fixture.admitted_cap)
        .expect("the oracle fixture's admitted cap is valid");
    let positions = fixture
        .rows
        .iter()
        .map(|row| row.position)
        .collect::<BTreeSet<_>>();
    assert_eq!(positions, BTreeSet::from([0, 1, fixture.admitted_cap - 1]));
    for row in fixture.rows {
        assert!(row.lane < NANBEIGE_HEAD_DIM / 2);
        let (cosine, sine) = tables.table_value(row.position, row.lane).unwrap();
        let oracle_cosine = f32::from_bits(parse_u32_bits(&row.f32_cosine_bits, "f32 cosine"));
        let oracle_sine = f32::from_bits(parse_u32_bits(&row.f32_sine_bits, "f32 sine"));
        assert_close(cosine, oracle_cosine, 2.0e-6);
        assert_close(sine, oracle_sine, 2.0e-6);
        assert_eq!(
            Bf16::from_f32(cosine).to_bits(),
            parse_u16_bits(&row.bf16_cosine_bits, "bf16 cosine"),
            "hf-bf16-eager cosine cast must match oracle position={} lane={}",
            row.position,
            row.lane
        );
        assert_eq!(
            Bf16::from_f32(sine).to_bits(),
            parse_u16_bits(&row.bf16_sine_bits, "bf16 sine"),
            "hf-bf16-eager sine cast must match oracle position={} lane={}",
            row.position,
            row.lane
        );
    }
}

#[test]
fn hf_bf16_eager_rope_application_matches_captured_qk_head() {
    let fixture: RopeOracleFixture =
        serde_json::from_str(ORACLE_FIXTURE).expect("the pinned RoPE oracle fixture is valid JSON");
    let application = fixture.application;
    assert_eq!(application.capture_schema_version, 2);
    assert_eq!(application.profile, "hf-bf16-eager");
    assert_eq!(application.phase, "prefill");
    assert_eq!(application.layer, 0);
    assert_eq!(application.loop_index, 0);
    assert_eq!(application.position, 1);
    assert_eq!(application.query_head, 0);
    assert_eq!(application.key_head, 0);
    assert_eq!(application.input_ids, vec![1, 2]);
    assert_eq!(application.modeling_source_sha256, PINNED_SOURCE_SHA256);
    assert_eq!(application.torch, "2.6.0");
    assert_eq!(
        application.prior_capture_source_closure_verification, "not_mechanically_verified",
        "the historical capture must not be retroactively represented as a mechanically verified model closure"
    );
    let receipt = application.reproduction_receipt;
    assert_eq!(
        receipt.capture_scope,
        "loop0/layer0/prefill/position1/query-head0/key-head0"
    );
    assert_eq!(receipt.generator_path, "scripts/gen_reference_fixtures.py");
    assert_eq!(
        receipt.generator_commit,
        "80e15fc0e1f7c8303562724891b8e800522be2e0"
    );
    assert_eq!(receipt.model_id, "Nanbeige/Nanbeige4.2-3B");
    assert_eq!(receipt.pinned_revision, PINNED_REVISION);
    assert_eq!(
        receipt.model_source_closure_verification,
        "passed-before-model-load"
    );
    assert_eq!(
        receipt.source_manifest_sha256,
        sha256_path(&repo_path("docs/truth-pack/nanbeige4.2-3b.source.json")),
        "the fresh capture receipt must stay bound to the source manifest that names both weight shards"
    );
    assert_eq!(receipt.oracle_env_record_sha256.len(), 64);
    assert!(
        receipt
            .oracle_env_record_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    );

    let mut query = parse_bf16_vector(
        &application.query_input_bf16_hex,
        "application query input",
        NANBEIGE_HEAD_DIM,
    );
    let mut key = parse_bf16_vector(
        &application.key_input_bf16_hex,
        "application key input",
        NANBEIGE_HEAD_DIM,
    );
    let expected_query = parse_bf16_vector(
        &application.query_rotated_bf16_hex,
        "application rotated query",
        NANBEIGE_HEAD_DIM,
    );
    let expected_key = parse_bf16_vector(
        &application.key_rotated_bf16_hex,
        "application rotated key",
        NANBEIGE_HEAD_DIM,
    );
    let captured_cosine = parse_bf16_vector(
        &application.cosine_bf16_hex,
        "application cosine row",
        NANBEIGE_HEAD_DIM,
    );
    let captured_sine = parse_bf16_vector(
        &application.sine_bf16_hex,
        "application sine row",
        NANBEIGE_HEAD_DIM,
    );
    let tables = RopeTablesF32::nanbeige(application.position + 1)
        .expect("application capture position is an admitted table row");
    for lane in 0..NANBEIGE_HEAD_DIM {
        let table_lane = lane % (NANBEIGE_HEAD_DIM / 2);
        let (cosine, sine) = tables
            .table_value(application.position, table_lane)
            .unwrap();
        assert_eq!(
            Bf16::from_f32(cosine),
            captured_cosine[table_lane],
            "captured hf-bf16-eager cosine lane={lane}"
        );
        assert_eq!(
            Bf16::from_f32(sine),
            captured_sine[table_lane],
            "captured hf-bf16-eager sine lane={lane}"
        );
    }
    tables
        .apply_split_half(application.position, &mut query)
        .unwrap();
    tables
        .apply_split_half(application.position, &mut key)
        .unwrap();
    assert_eq!(query, expected_query, "captured Q RoPE application bits");
    assert_eq!(key, expected_key, "captured K RoPE application bits");
    eprintln!(
        "ROPE_APPLICATION_CAPTURE RESULT=PASS profile=hf-bf16-eager historical_capture=not_mechanically_verified reproduction_receipt=source-closure-verified position=1 q_head=0 k_head=0"
    );
}

#[test]
fn hf_bf16_eager_rope_decode_append_matches_captured_qk_head() {
    let fixture: RopeDecodeApplicationFixture = serde_json::from_str(DECODE_APPLICATION_FIXTURE)
        .expect("the pinned RoPE decode application fixture is valid JSON");
    let application = fixture.application;
    assert_eq!(application.capture_schema_version, 2);
    assert_eq!(application.profile, "hf-bf16-eager");
    assert_eq!(application.phase, "decode-append");
    assert_eq!(application.layer, 0);
    assert_eq!(application.loop_index, 0);
    assert_eq!(application.position, 2);
    assert_eq!(application.prefill_input_ids, vec![1, 2]);
    assert_eq!(application.input_ids, vec![13]);
    assert_eq!(application.query_head, 0);
    assert_eq!(application.key_head, 0);
    assert_eq!(application.modeling_source_sha256, PINNED_SOURCE_SHA256);
    assert_eq!(application.torch, "2.6.0");
    assert_eq!(
        fixture.receipt.capture_scope,
        "loop0/layer0/decode-append/position2/query-head0/key-head0"
    );
    assert_eq!(
        fixture.receipt.generator_path,
        "scripts/gen_reference_fixtures.py"
    );
    assert_eq!(
        fixture.receipt.generator_commit,
        "39e4dd81e7c22268221d83d034ae54701afe8743"
    );
    assert_eq!(fixture.receipt.model_id, "Nanbeige/Nanbeige4.2-3B");
    assert_eq!(fixture.receipt.pinned_revision, PINNED_REVISION);
    assert_eq!(
        fixture.receipt.model_source_closure_verification,
        "passed-before-model-load"
    );
    assert_eq!(
        fixture.receipt.source_manifest_sha256,
        sha256_path(&repo_path("docs/truth-pack/nanbeige4.2-3b.source.json")),
        "the decode capture receipt must stay bound to the source manifest that names both weight shards"
    );
    assert_eq!(fixture.receipt.oracle_env_record_sha256.len(), 64);
    assert!(
        fixture
            .receipt
            .oracle_env_record_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    );

    let mut query = parse_bf16_vector(
        &application.query_input_bf16_hex,
        "decode application query input",
        NANBEIGE_HEAD_DIM,
    );
    let mut key = parse_bf16_vector(
        &application.key_input_bf16_hex,
        "decode application key input",
        NANBEIGE_HEAD_DIM,
    );
    let expected_query = parse_bf16_vector(
        &application.query_rotated_bf16_hex,
        "decode application rotated query",
        NANBEIGE_HEAD_DIM,
    );
    let expected_key = parse_bf16_vector(
        &application.key_rotated_bf16_hex,
        "decode application rotated key",
        NANBEIGE_HEAD_DIM,
    );
    let captured_cosine = parse_bf16_vector(
        &application.cosine_bf16_hex,
        "decode application cosine row",
        NANBEIGE_HEAD_DIM,
    );
    let captured_sine = parse_bf16_vector(
        &application.sine_bf16_hex,
        "decode application sine row",
        NANBEIGE_HEAD_DIM,
    );
    let tables = RopeTablesF32::nanbeige(application.position + 1)
        .expect("decode application capture position is an admitted table row");
    for lane in 0..NANBEIGE_HEAD_DIM {
        let table_lane = lane % (NANBEIGE_HEAD_DIM / 2);
        let (cosine, sine) = tables
            .table_value(application.position, table_lane)
            .unwrap();
        assert_eq!(
            Bf16::from_f32(cosine),
            captured_cosine[lane],
            "captured hf-bf16-eager decode cosine lane={lane}"
        );
        assert_eq!(
            Bf16::from_f32(sine),
            captured_sine[lane],
            "captured hf-bf16-eager decode sine lane={lane}"
        );
    }
    tables
        .apply_split_half(application.position, &mut query)
        .unwrap();
    tables
        .apply_split_half(application.position, &mut key)
        .unwrap();
    assert_eq!(
        query, expected_query,
        "captured decode Q RoPE application bits"
    );
    assert_eq!(key, expected_key, "captured decode K RoPE application bits");
    eprintln!(
        "ROPE_DECODE_APPLICATION_CAPTURE RESULT=PASS profile=hf-bf16-eager position=2 q_head=0 k_head=0"
    );
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
fn real_shape_prefill_and_decode_qk_projection_boundaries_are_exact() {
    const QUERY_ELEMENTS: usize = 48 * NANBEIGE_HEAD_DIM;
    const KEY_ELEMENTS: usize = 8 * NANBEIGE_HEAD_DIM;
    let tables = RopeTablesF32::nanbeige(18).expect("the admitted real-shape cap is valid");

    for (phase, position) in [
        ("prefill", 0_usize),
        ("prefill", 1),
        ("prefill", 2),
        ("decode", 17),
    ] {
        let query = real_shape_values(QUERY_ELEMENTS, 0.000_976_562_5, -3.0);
        let key = real_shape_values(KEY_ELEMENTS, -0.001_953_125, 1.5);
        let mut unfused_query = query.clone();
        let mut unfused_key = key.clone();
        let mut fused_query = query.clone();
        let mut fused_key = key.clone();
        let mut loop_two_query = query.clone();
        let mut loop_two_key = key.clone();

        tables
            .apply_projected_qk_unfused(position, &mut unfused_query, &mut unfused_key)
            .unwrap();
        tables
            .apply_projected_qk_fused_epilogue(position, &mut fused_query, &mut fused_key)
            .unwrap();
        tables
            .apply_projected_qk(
                RopeProjectionVariant::default(),
                position,
                &mut loop_two_query,
                &mut loop_two_key,
            )
            .unwrap();

        assert_eq!(
            fused_query, unfused_query,
            "{phase} Q fused/unfused position={position}"
        );
        assert_eq!(
            fused_key, unfused_key,
            "{phase} K fused/unfused position={position}"
        );
        assert_eq!(
            loop_two_query, unfused_query,
            "both logical loop passes must use the same RoPE row at {phase} position={position}"
        );
        assert_eq!(
            loop_two_key, unfused_key,
            "both logical loop passes must use the same RoPE row at {phase} position={position}"
        );
    }
    eprintln!(
        "ROPE RESULT=PASS oracle_fixture=pinned-source positions=0,1,17 q_elements={QUERY_ELEMENTS} k_elements={KEY_ELEMENTS}"
    );
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
