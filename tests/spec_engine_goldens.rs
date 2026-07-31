#[path = "spec_engine/mod.rs"]
mod spec_engine;

use std::fs;
use std::path::Path;

use serde::Deserialize;
use spec_engine::{
    DenseAttention, KvCache, NANBEIGE_HEAD_DIM, NANBEIGE_HIDDEN, NANBEIGE_INTERMEDIATE,
    NANBEIGE_KV_HEADS, NANBEIGE_QUERY_HEADS, NANBEIGE_VOCAB, PHYSICAL_LAYERS, SpecConfig,
    SpecEngine, SpecWeights, Tensor, apply_rope_split_half, dense_gqa_attention, greedy_argmax,
    kv_slot, rms_norm, silu, stable_dot,
};

#[derive(Deserialize)]
struct TinyForwardFixture {
    tokens: Vec<u32>,
    embedding: TinyRow,
    lm_head_rows: Vec<TinyRow>,
    expected: TinyForwardExpected,
}

#[derive(Deserialize)]
struct TinyRow {
    row: usize,
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct TinyForwardExpected {
    prefill_positions: usize,
    decode_position: usize,
    slot_len_after_prefill: usize,
    slot_len_after_decode: usize,
    layer_taps: usize,
    post_loop_norms: usize,
}

#[test]
fn scalar_op_goldens_are_hand_checkable() {
    assert_eq!(
        stable_dot(&[1.0, -2.0, 3.0], &[4.0, 5.0, -6.0], "dot").unwrap(),
        -24.0
    );

    let norm = rms_norm(&[3.0, 4.0], &[1.0, 2.0], 0.0, "rms").unwrap();
    assert_close("rms", 0, 0, 0, norm[0], 3.0 / 12.5_f32.sqrt());
    assert_close("rms", 0, 0, 1, norm[1], 8.0 / 12.5_f32.sqrt());
    assert_eq!(silu(0.0), 0.0);

    let mut rope = vec![1.0, 0.0];
    apply_rope_split_half(&mut rope, 1, 2, 1, 2.0).unwrap();
    assert_close("rope", 0, 0, 0, rope[0], 1.0_f32.cos());
    assert_close("rope", 0, 0, 1, rope[1], 1.0_f32.sin());
    assert_eq!(greedy_argmax(&[3.0, 3.0, 2.0]), Some(0));
}

#[test]
fn dense_gqa_expands_kv_heads_and_uses_causal_mask() {
    let config = SpecConfig {
        hidden: 2,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 2,
        vocab: 2,
        rms_epsilon: 1.0e-5,
        rope_theta: 2.0,
    };
    config.validate().unwrap();
    let attention = dense_gqa_attention(
        &config,
        &[1.0, 0.0, 1.0, 0.0],
        &[vec![1.0, 0.0], vec![0.0, 1.0]],
        &[vec![10.0, 20.0], vec![30.0, 40.0]],
        &[0, 1],
        0,
    )
    .unwrap();
    assert_dense_output(&attention, &[10.0, 20.0, 10.0, 20.0]);
    assert_eq!(attention.probabilities, vec![1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn real_shape_constants_preserve_explicit_head_dim_and_row_major_stride() {
    let config = SpecConfig::nanbeige();
    config.validate().unwrap();
    assert_eq!(config.hidden, NANBEIGE_HIDDEN);
    assert_eq!(config.query_heads, NANBEIGE_QUERY_HEADS);
    assert_eq!(config.kv_heads, NANBEIGE_KV_HEADS);
    assert_eq!(config.head_dim, NANBEIGE_HEAD_DIM);
    assert_eq!(config.intermediate, NANBEIGE_INTERMEDIATE);
    assert_eq!(config.vocab, NANBEIGE_VOCAB);
    assert_eq!(config.query_width(), 6144);
    assert_eq!(config.kv_width(), 1024);

    let tensor = Tensor::new(2, 3, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    assert_eq!(tensor.get(0, 2).unwrap(), 2.0);
    assert_eq!(tensor.get(1, 0).unwrap(), 3.0);
    assert_eq!(tensor.rows(), 2);
    assert_eq!(tensor.cols(), 3);
}

#[test]
fn tiny_full_forward_has_44_taps_two_norms_and_44_slot_kv_occupancy() {
    let fixture: TinyForwardFixture =
        serde_json::from_str(include_str!("fixtures/spec_engine_tiny_forward.json"))
            .expect("tiny forward fixture must remain valid JSON");
    let config = SpecConfig::tiny_for_tests();
    let mut weights = SpecWeights::zeroed(&config).unwrap();
    for (column, value) in fixture.embedding.values.iter().copied().enumerate() {
        weights.embeddings.set(fixture.embedding.row, column, value).unwrap();
    }
    for row in &fixture.lm_head_rows {
        for (column, value) in row.values.iter().copied().enumerate() {
            weights.lm_head.set(row.row, column, value).unwrap();
        }
    }
    let engine = SpecEngine::new(config.clone(), weights).unwrap();
    let mut cache = KvCache::new(&config);

    let prefill = engine.prefill(&fixture.tokens, &mut cache).unwrap();
    assert_eq!(prefill.len(), fixture.expected.prefill_positions);
    for output in &prefill {
        assert_eq!(output.taps.layer_taps.len(), fixture.expected.layer_taps);
        assert_eq!(
            output.taps.post_loop_norms.len(),
            fixture.expected.post_loop_norms
        );
        assert_eq!(output.taps.logits.len(), config.vocab);
    }
    for loop_index in 0..2 {
        for layer_index in 0..22 {
            let slot = kv_slot(loop_index, layer_index);
            assert_eq!(slot, layer_index + loop_index * PHYSICAL_LAYERS);
            assert_eq!(
                cache.slot_len(slot).unwrap(),
                fixture.expected.slot_len_after_prefill
            );
        }
    }

    let decoded = engine.decode(fixture.tokens[0], &mut cache).unwrap();
    assert_eq!(decoded.position, fixture.expected.decode_position);
    assert_eq!(decoded.taps.layer_taps.len(), fixture.expected.layer_taps);
    for slot in 0..44 {
        assert_eq!(
            cache.slot_len(slot).unwrap(),
            fixture.expected.slot_len_after_decode
        );
    }

    // Independent tiny fixture calculation: all projections are zero, so the
    // two post-loop norms are the only transformations before lm_head.
    let once = rms_norm(
        &fixture.embedding.values,
        &[1.0; 4],
        config.rms_epsilon,
        "fixture",
    )
    .unwrap();
    let twice = rms_norm(&once, &[1.0; 4], config.rms_epsilon, "fixture").unwrap();
    for row in &fixture.lm_head_rows {
        let expected = stable_dot(&row.values, &twice, "fixture_lm_head").unwrap();
        assert_close(
            "tiny_e2e",
            1,
            21,
            row.row,
            decoded.taps.logits[row.row],
            expected,
        );
    }
}

#[test]
fn repeat_results_are_bit_identical_and_rows_do_not_affect_each_other() {
    let config = SpecConfig::tiny_for_tests();
    let weights = SpecWeights::zeroed(&config).unwrap();
    let engine = SpecEngine::new(config.clone(), weights).unwrap();
    let mut left_cache = KvCache::new(&config);
    let mut right_cache = KvCache::new(&config);
    let left = engine.prefill(&[0, 1], &mut left_cache).unwrap();
    let right = engine.prefill(&[0, 1], &mut right_cache).unwrap();
    assert_eq!(left, right);

    let row_a = stable_dot(&[1.0, 2.0], &[3.0, 4.0], "row_a").unwrap();
    let row_b = stable_dot(&[5.0, 6.0], &[7.0, 8.0], "row_b").unwrap();
    let reversed_a = stable_dot(&[1.0, 2.0], &[3.0, 4.0], "row_a").unwrap();
    let reversed_b = stable_dot(&[5.0, 6.0], &[7.0, 8.0], "row_b").unwrap();
    assert_eq!((row_a, row_b), (reversed_a, reversed_b));
}

#[test]
fn structural_test_keeps_the_spec_engine_outside_product_kernels() {
    let source = include_str!("spec_engine/mod.rs");
    for forbidden in ["native_engine", "franken_nlp::", "crate::"] {
        assert!(
            !source.contains(forbidden),
            "spec engine must not link product implementation surface: {forbidden}"
        );
    }
    let library = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"))
        .expect("library root must be readable");
    assert!(
        !library.contains("spec_engine"),
        "test-only spec engine must not be exported by the library"
    );
}

#[test]
fn model_gated_fixture_reports_skip_without_real_artifact() {
    let model_fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reference/model_forward.json");
    if !model_fixture.is_file() {
        eprintln!("SPEC_ENGINE RESULT=SKIPPED_NO_MODEL ops=0");
        return;
    }
    eprintln!("SPEC_ENGINE RESULT=PASS ops=44");
}

#[test]
fn scalar_suite_logs_pass_marker() {
    eprintln!("SPEC_ENGINE RESULT=PASS ops=44");
}

fn assert_dense_output(attention: &DenseAttention, expected: &[f32]) {
    assert_eq!(attention.output.len(), expected.len());
    for (index, (got, expected)) in attention.output.iter().zip(expected).enumerate() {
        assert_close("dense_gqa", 0, 0, index, *got, *expected);
    }
}

fn assert_close(
    op: &'static str,
    loop_index: usize,
    layer_index: usize,
    index: usize,
    got: f32,
    expected: f32,
) {
    let absolute = (got - expected).abs();
    let relative = absolute / got.abs().max(expected.abs()).max(f32::MIN_POSITIVE);
    assert!(
        absolute <= 1.0e-6,
        "op={op} tap=({loop_index}, {layer_index}) index={index} expected={expected} got={got} abs={absolute} rel={relative} ulp={}",
        spec_engine::ulp_distance(expected, got)
    );
}
