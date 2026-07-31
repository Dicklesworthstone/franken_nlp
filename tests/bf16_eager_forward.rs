#![deny(unsafe_code)]

use franken_nlp::native_engine::{
    attention::{
        ATTENTION_SCALE, KV_HEAD_COUNT, QUERY_HEAD_COUNT, QUERY_HEADS_PER_KV_HEAD,
        eager_attention_head, kv_head_for_query, softmax_f32_cast_back,
    },
    hf_bf16_eager::{
        HF_BF16_EAGER_PROFILE, HfBf16EagerEngine, HfBf16EagerError, HfBf16EagerWeights,
    },
    layer::{
        HfBf16EagerLayerWeights, NANBEIGE_HIDDEN_SIZE, NANBEIGE_INTERMEDIATE_SIZE,
        NANBEIGE_KV_PROJECTION_SIZE, NANBEIGE_Q_PROJECTION_SIZE,
    },
    lmhead::{NANBEIGE_F32_LOGIT_BYTES, NANBEIGE_VOCAB_SIZE, export_logits_f32, greedy_argmax},
    nn::{
        HF_BF16_EAGER_CAST_SCHEDULE, HfBf16EagerCastSite, RMS_NORM_EPSILON,
        embedding_row_stays_bf16, residual_add_f32_cast_back, rms_norm_f32_reduce_cast_back,
        swiglu_f32_cast_back,
    },
    rope::{NANBEIGE_HEAD_DIM, NANBEIGE_ROPE_THETA, RopeTablesF32},
    tensor::Bf16,
    weights::Bf16Matrix,
};

fn bf16s(values: &[f32]) -> Vec<Bf16> {
    values.iter().copied().map(Bf16::from_f32).collect()
}

fn assert_close(observed: f32, expected: f32, tolerance: f32) {
    assert!(
        (observed - expected).abs() <= tolerance,
        "observed={observed} expected={expected} tolerance={tolerance}"
    );
}

#[test]
fn bf16_eager_reference_primitives_and_cast_schedule() {
    let tie_to_even = Bf16::from_f32(f32::from_bits(0x3f80_8000));
    assert_eq!(tie_to_even.to_bits(), 0x3f80);

    let embedding = [Bf16::from_bits(0x3f81), Bf16::from_bits(0xbf82)];
    let preserved = embedding_row_stays_bf16(&embedding);
    assert_eq!(
        preserved, embedding,
        "embedding rows must not widen before use"
    );

    let norm_input = bf16s(&[3.0, 4.0]);
    let norm_weight = bf16s(&[1.0, 1.0]);
    let normalized = rms_norm_f32_reduce_cast_back(&norm_input, &norm_weight, RMS_NORM_EPSILON)
        .expect("hand-sized RMSNorm has matching shape");
    assert_close(normalized[0].to_f32(), 0.848, 0.01);
    assert_close(normalized[1].to_f32(), 1.133, 0.01);

    let probabilities = softmax_f32_cast_back(&[0.0, 1.0]).expect("nonempty softmax");
    assert_close(probabilities[0].to_f32(), 0.2689, 0.01);
    assert_close(probabilities[1].to_f32(), 0.7311, 0.01);

    let rope =
        RopeTablesF32::new(17, 4, NANBEIGE_ROPE_THETA).expect("even miniature head dimension");
    for position in [0, 1, 16] {
        let mut vector = bf16s(&[1.0, 2.0, 3.0, 4.0]);
        rope.apply_split_half(position, &mut vector)
            .expect("precomputed table position");
        if position == 0 {
            assert_eq!(vector, bf16s(&[1.0, 2.0, 3.0, 4.0]));
        }
    }
    assert_eq!(NANBEIGE_HEAD_DIM, 128);

    assert_eq!(QUERY_HEAD_COUNT, 48);
    assert_eq!(KV_HEAD_COUNT, 8);
    assert_eq!(QUERY_HEADS_PER_KV_HEAD, 6);
    assert_eq!(kv_head_for_query(0), Ok(0));
    assert_eq!(kv_head_for_query(5), Ok(0));
    assert_eq!(kv_head_for_query(6), Ok(1));
    assert_eq!(kv_head_for_query(47), Ok(7));
    assert_close(ATTENTION_SCALE, 1.0 / 128.0_f32.sqrt(), 1.0e-8);

    let query = vec![Bf16::from_f32(1.0); NANBEIGE_HEAD_DIM];
    let mut keys = vec![Bf16::from_f32(0.0); 2 * NANBEIGE_HEAD_DIM];
    keys[NANBEIGE_HEAD_DIM..].fill(Bf16::from_f32(1.0));
    let values = [
        vec![Bf16::from_f32(1.0); NANBEIGE_HEAD_DIM],
        vec![Bf16::from_f32(3.0); NANBEIGE_HEAD_DIM],
    ]
    .concat();
    let attended = eager_attention_head(&query, &keys, &values, 2).expect("valid causal prefix");
    assert!(
        attended[0].to_f32() > 2.9,
        "scaled eager softmax selects the matching key"
    );

    let swiglu = swiglu_f32_cast_back(&bf16s(&[0.0, 1.0]), &bf16s(&[2.0, 3.0]))
        .expect("matching gate/up vectors");
    assert_close(swiglu[0].to_f32(), 0.0, 0.01);
    assert!(swiglu[1].to_f32() > 2.1);
    let residual = residual_add_f32_cast_back(&bf16s(&[1.0, 2.0]), &bf16s(&[3.0, 4.0]))
        .expect("matching residual vectors");
    assert_eq!(residual, bf16s(&[4.0, 6.0]));

    let lm_head = Bf16Matrix::new(3, 2, bf16s(&[1.0, 0.0, 0.0, 2.0, -1.0, -1.0]))
        .expect("tiny untied lm head");
    assert_eq!(
        lm_head.row(1).expect("second row exists"),
        &bf16s(&[0.0, 2.0])
    );
    let logits = export_logits_f32(&bf16s(&[2.0, 3.0]), &lm_head).expect("matching hidden width");
    assert_eq!(logits, vec![2.0, 6.0, -5.0]);
    assert_eq!(greedy_argmax(&logits), Some(1));
    assert_eq!(
        greedy_argmax(&[2.0, 2.0]),
        Some(0),
        "ties preserve first token id"
    );
    assert_eq!(NANBEIGE_F32_LOGIT_BYTES, NANBEIGE_VOCAB_SIZE * 4);

    assert_eq!(NANBEIGE_HIDDEN_SIZE, 3_072);
    assert_eq!(NANBEIGE_Q_PROJECTION_SIZE, 6_144);
    assert_eq!(NANBEIGE_KV_PROJECTION_SIZE, 1_024);
    assert_eq!(NANBEIGE_INTERMEDIATE_SIZE, 10_752);
    assert_eq!(
        HF_BF16_EAGER_CAST_SCHEDULE,
        [
            HfBf16EagerCastSite::EmbeddingRowStaysBf16,
            HfBf16EagerCastSite::RmsNormF32ReduceCastBack,
            HfBf16EagerCastSite::SoftmaxF32CastBack,
            HfBf16EagerCastSite::RopeF32TableCastAtApplication,
            HfBf16EagerCastSite::LogitsExportF32,
        ]
    );

    eprintln!("BF16_EAGER RESULT=PASS taps=0 model_fixture=SKIPPED_NO_MODEL");
}

#[test]
fn bf16_eager_engine_refuses_a_non_nanbeige_model_before_cache_allocation() {
    let scalar = Bf16Matrix::new(1, 1, bf16s(&[1.0])).expect("one scalar");
    let tiny_layer = HfBf16EagerLayerWeights {
        input_norm: bf16s(&[1.0]),
        q_proj: scalar.clone(),
        k_proj: scalar.clone(),
        v_proj: scalar.clone(),
        o_proj: scalar.clone(),
        post_attention_norm: bf16s(&[1.0]),
        gate_proj: scalar.clone(),
        up_proj: scalar.clone(),
        down_proj: scalar.clone(),
    };
    let error = HfBf16EagerEngine::new(
        HfBf16EagerWeights {
            embeddings: scalar.clone(),
            layers: std::array::from_fn(|_| tiny_layer.clone()),
            final_norm: bf16s(&[1.0]),
            lm_head: scalar,
        },
        1,
    )
    .expect_err("a tiny fixture cannot impersonate the single-model wedge");
    assert!(matches!(
        error,
        HfBf16EagerError::ModelMatrixShape {
            name: "embeddings",
            expected_rows: NANBEIGE_VOCAB_SIZE,
            expected_columns: NANBEIGE_HIDDEN_SIZE,
            ..
        }
    ));
    assert_eq!(HF_BF16_EAGER_PROFILE, "hf-bf16-eager");
}
