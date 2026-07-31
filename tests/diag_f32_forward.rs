#![deny(unsafe_code)]

use franken_nlp::native_engine::{
    diagnostic_f32::{
        DIAGNOSTIC_F32_PROFILE, DiagnosticF32Config, DiagnosticF32Engine,
        DiagnosticF32LayerWeights, DiagnosticF32Matrix, DiagnosticF32RopeTables,
        DiagnosticF32Weights, TokenFlip, diagnostic_f32_metrics, first_structural_divergence,
        softmax_f32, unlisted_token_flips,
    },
    kv::{KV_SLOT_COUNT, LOOP_COUNT, PHYSICAL_LAYER_COUNT, slot_for},
    tensor::Bf16,
};

fn matrix(rows: usize, columns: usize, values: Vec<f32>) -> DiagnosticF32Matrix {
    DiagnosticF32Matrix::new(rows, columns, values).expect("fixture matrix shape is valid")
}

fn identity(rows: usize, columns: usize) -> DiagnosticF32Matrix {
    let mut values = vec![0.0; rows * columns];
    for row in 0..rows.min(columns) {
        values[row * columns + row] = 1.0;
    }
    matrix(rows, columns, values)
}

fn miniature_config() -> DiagnosticF32Config {
    DiagnosticF32Config {
        hidden_size: 4,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate_size: 4,
        vocab_size: 3,
        rms_epsilon: 1.0e-5,
        rope_theta: 70_000_000.0,
    }
}

fn miniature_weights(config: &DiagnosticF32Config) -> DiagnosticF32Weights {
    let layer = || DiagnosticF32LayerWeights {
        input_norm: vec![1.0; config.hidden_size],
        q_proj: identity(config.query_width(), config.hidden_size),
        k_proj: identity(config.kv_width(), config.hidden_size),
        v_proj: identity(config.kv_width(), config.hidden_size),
        o_proj: identity(config.hidden_size, config.query_width()),
        post_attention_norm: vec![1.0; config.hidden_size],
        gate_proj: identity(config.intermediate_size, config.hidden_size),
        up_proj: identity(config.intermediate_size, config.hidden_size),
        down_proj: identity(config.hidden_size, config.intermediate_size),
    };
    DiagnosticF32Weights {
        embeddings: matrix(
            config.vocab_size,
            config.hidden_size,
            vec![
                0.0, 0.0, 0.0, 0.0, // token 0
                1.0, 0.5, -0.25, 0.75, // token 1
                -0.5, 0.25, 1.0, 0.125, // token 2
            ],
        ),
        layers: std::array::from_fn(|_| layer()),
        final_norm: vec![1.0; config.hidden_size],
        lm_head: matrix(
            config.vocab_size,
            config.hidden_size,
            vec![
                1.0, 0.0, 0.0, 0.0, // row 0
                0.0, 1.0, 0.0, 0.0, // row 1
                0.0, 0.0, 1.0, 1.0, // row 2
            ],
        ),
    }
}

fn miniature_engine(max_positions: usize) -> DiagnosticF32Engine {
    let config = miniature_config();
    let weights = miniature_weights(&config);
    DiagnosticF32Engine::new(config, weights, max_positions).expect("fixture engine validates")
}

#[test]
fn diagnostic_f32_is_single_widen_structural_bisect_oracle() {
    let widened = DiagnosticF32Matrix::widen_from_bf16(
        1,
        2,
        &[Bf16::from_bits(0x3f81), Bf16::from_bits(0xbf82)],
    )
    .expect("one-time bf16 widening has an exact source shape");
    assert_eq!(
        widened.row(0).expect("widened row"),
        &[f32::from_bits(0x3f81_0000), f32::from_bits(0xbf82_0000)]
    );

    let probabilities = softmax_f32(&[0.0, 1.0]).expect("nonempty f32 softmax");
    assert!((probabilities[0] - 0.268_941_43).abs() < 1.0e-6);
    assert!((probabilities[1] - 0.731_058_6).abs() < 1.0e-6);
    let tables = DiagnosticF32RopeTables::new(17, 4, 70_000_000.0)
        .expect("f32 RoPE table dimensions are valid");
    let mut rope_vector = vec![1.0, 2.0, 3.0, 4.0];
    tables
        .apply_all_heads(0, &mut rope_vector)
        .expect("position zero is in table");
    assert_eq!(rope_vector, vec![1.0, 2.0, 3.0, 4.0]);

    let mut left = miniature_engine(2);
    let mut right = miniature_engine(2);
    assert_eq!(left.profile(), DIAGNOSTIC_F32_PROFILE);
    let first = left.decode(1).expect("first diagnostic f32 decode");
    assert_eq!(
        first,
        right.decode(1).expect("identical engine is deterministic"),
        "diagnostic-f32 must be bit-identical for a fixed single-threaded input"
    );
    assert_eq!(first.position, 0);
    assert_eq!(first.taps.layer_taps.len(), KV_SLOT_COUNT);
    assert_eq!(first.taps.post_loop_norms.len(), LOOP_COUNT);
    assert_eq!(first.taps.logits.len(), 3);
    for (index, tap) in first.taps.layer_taps.iter().enumerate() {
        let loop_index = index / PHYSICAL_LAYER_COUNT;
        let layer_index = index % PHYSICAL_LAYER_COUNT;
        assert_eq!(tap.loop_index, loop_index);
        assert_eq!(tap.layer_index, layer_index);
        assert_eq!(Some(tap.kv_slot), slot_for(loop_index, layer_index));
        assert!(tap.input.iter().all(|value| value.is_finite()));
        assert!(tap.output.iter().all(|value| value.is_finite()));
    }
    for slot in 0..KV_SLOT_COUNT {
        assert_eq!(left.kv_cache().len_for_slot(slot), Ok(1));
    }
    assert_eq!(
        left.decode(2)
            .expect("second diagnostic f32 decode")
            .position,
        1,
        "f32 K/V cache retains the first f32 position"
    );
    for slot in 0..KV_SLOT_COUNT {
        assert_eq!(left.kv_cache().len_for_slot(slot), Ok(2));
    }

    let metrics =
        diagnostic_f32_metrics(&[1.0, -2.0], &[1.0, -2.25]).expect("same-width f32 metric vectors");
    assert_eq!(metrics.max_abs, 0.25);
    assert!(metrics.max_rel > 0.12);
    assert!(metrics.max_ulp > 0);
    assert!(metrics.cosine > 0.99);

    let mut mis_slotted_cache_tap = first.clone();
    mis_slotted_cache_tap.taps.layer_taps[7].output[0] += 1.0;
    assert_eq!(
        first_structural_divergence(&first.taps, &mis_slotted_cache_tap.taps, 0.0),
        Some(
            franken_nlp::native_engine::diagnostic_f32::FirstStructuralDivergence {
                loop_index: 0,
                layer_index: Some(7),
                point: "post_mlp",
            }
        ),
        "a seeded wrong-K/V structural result localizes to its layer tap"
    );
    let mut dropped_boundary_norm = first.clone();
    dropped_boundary_norm.taps.post_loop_norms[0][0] += 1.0;
    assert_eq!(
        first_structural_divergence(&first.taps, &dropped_boundary_norm.taps, 0.0),
        Some(
            franken_nlp::native_engine::diagnostic_f32::FirstStructuralDivergence {
                loop_index: 0,
                layer_index: None,
                point: "post_loop_norm",
            }
        ),
        "a dropped boundary norm localizes to the loop boundary"
    );
    let mut rope_half_swap = first.clone();
    rope_half_swap.taps.layer_taps[3].input[0] += 1.0;
    assert_eq!(
        first_structural_divergence(&first.taps, &rope_half_swap.taps, 0.0),
        Some(
            franken_nlp::native_engine::diagnostic_f32::FirstStructuralDivergence {
                loop_index: 0,
                layer_index: Some(3),
                point: "pre_attention",
            }
        ),
        "a seeded RoPE structural result localizes before attention"
    );

    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/token_flips/diagnostic_f32_v1.json"))
            .expect("token flip fixture is valid JSON");
    assert_eq!(fixture["profile"], DIAGNOSTIC_F32_PROFILE);
    let known_flip = TokenFlip {
        prompt_id: "tiny-prompt-a".to_owned(),
        position: 2,
        bf16_token: 11,
        f32_token: 13,
        logit_gap: 0.125,
    };
    assert!(
        unlisted_token_flips(
            std::slice::from_ref(&known_flip),
            std::slice::from_ref(&known_flip)
        )
        .is_empty()
    );
    let unexpected = TokenFlip {
        f32_token: 14,
        ..known_flip.clone()
    };
    assert_eq!(
        unlisted_token_flips(&[known_flip], &[unexpected]).len(),
        1,
        "an unlisted bf16/f32 token flip remains a regression"
    );

    eprintln!(
        "DIAG_F32 RESULT=PASS taps={} flips_expected=0 flips_observed=0 model_fixture=SKIPPED_NO_MODEL",
        KV_SLOT_COUNT
    );
}
