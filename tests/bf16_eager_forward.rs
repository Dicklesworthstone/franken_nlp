#![deny(unsafe_code)]

use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
};

use franken_nlp::native_engine::{
    attention::{
        ATTENTION_SCALE, KV_HEAD_COUNT, QUERY_HEAD_COUNT, QUERY_HEADS_PER_KV_HEAD,
        eager_attention_head, kv_head_for_query, softmax_f32_cast_back,
    },
    hf_bf16_eager::{
        HF_BF16_EAGER_PROFILE, HfBf16EagerEngine, HfBf16EagerError, HfBf16EagerSourceError,
        HfBf16EagerWeights,
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
use serde::Deserialize;
use sha2::{Digest, Sha256};

const HF_BF16_EAGER_TRACE_PROMPT: &str = "tests/fixtures/reference/hf-bf16-eager/prompt-000";
const PINNED_MODEL_SOURCE_ENV: &str = "FNLP_PINNED_MODEL_SOURCE";
const NANBEIGE_PHYSICAL_LAYER_COUNT: usize = 22;
const NANBEIGE_LOGICAL_LAYER_COUNT: usize = 44;

#[derive(Debug, Deserialize)]
struct HfBf16EagerTrace {
    profile: String,
    greedy_tokens: Vec<u32>,
    logits: HfBf16EagerTraceTensor,
    append: HfBf16EagerTracePhase,
    prefill: HfBf16EagerTracePhase,
}

#[derive(Debug, Deserialize)]
struct HfBf16EagerTracePhase {
    phase: String,
    records: Vec<HfBf16EagerTraceTensor>,
}

#[derive(Debug, Deserialize)]
struct HfBf16EagerTraceTensor {
    tap_name: String,
    relative_path: PathBuf,
    sha256: String,
    byte_length: usize,
    dtype: String,
    element_size: usize,
    shape: Vec<usize>,
    #[serde(rename = "loop")]
    loop_index: Option<usize>,
    layer: Option<usize>,
}

fn bf16s(values: &[f32]) -> Vec<Bf16> {
    values.iter().copied().map(Bf16::from_f32).collect()
}

fn assert_close(observed: f32, expected: f32, tolerance: f32) {
    assert!(
        (observed - expected).abs() <= tolerance,
        "observed={observed} expected={expected} tolerance={tolerance}"
    );
}

fn armed_pinned_model_source() -> Option<PathBuf> {
    let Some(source) = env::var_os(PINNED_MODEL_SOURCE_ENV).map(PathBuf::from) else {
        eprintln!(
            "BF16_EAGER RESULT=SKIPPED_NO_MODEL taps=0 norms=0 reason={PINNED_MODEL_SOURCE_ENV}-unset"
        );
        return None;
    };
    if !source.is_dir() {
        eprintln!(
            "BF16_EAGER RESULT=SKIPPED_NO_MODEL taps=0 norms=0 reason={PINNED_MODEL_SOURCE_ENV}-not-directory"
        );
        return None;
    }
    Some(source)
}

fn trace_tensor_element_count(tensor: &HfBf16EagerTraceTensor) -> usize {
    tensor
        .shape
        .iter()
        .copied()
        .try_fold(1_usize, |count, dimension| count.checked_mul(dimension))
        .expect("oracle trace tensor shape must not overflow usize")
}

fn assert_trace_tensor_integrity(root: &Path, tensor: &HfBf16EagerTraceTensor) -> Vec<u8> {
    assert!(
        !tensor.relative_path.is_absolute(),
        "oracle trace tensor path must remain relative: {}",
        tensor.relative_path.display()
    );
    assert_eq!(
        tensor.byte_length,
        trace_tensor_element_count(tensor)
            .checked_mul(tensor.element_size)
            .expect("oracle trace byte length must not overflow usize"),
        "oracle trace byte length must match declared shape for {}",
        tensor.tap_name,
    );
    let bytes = fs::read(root.join(&tensor.relative_path)).unwrap_or_else(|error| {
        panic!(
            "read oracle trace tensor {}: {error}",
            tensor.relative_path.display()
        )
    });
    assert_eq!(
        bytes.len(),
        tensor.byte_length,
        "oracle trace tensor length must match record for {}",
        tensor.tap_name,
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        tensor.sha256,
        "oracle trace tensor digest must match record for {}",
        tensor.tap_name,
    );
    bytes
}

fn assert_trace_tensor_continuity(
    observed: &HfBf16EagerTraceTensor,
    predecessor: &HfBf16EagerTraceTensor,
    description: &str,
) {
    assert_eq!(
        observed.sha256, predecessor.sha256,
        "{description} must preserve the preceding bf16 tensor bytes"
    );
    assert_eq!(
        observed.shape, predecessor.shape,
        "{description} must preserve the preceding bf16 tensor shape"
    );
    assert_eq!(
        observed.byte_length, predecessor.byte_length,
        "{description} must preserve the preceding bf16 tensor byte length"
    );
}

fn decode_native_f32(bytes: &[u8]) -> f32 {
    f32::from_ne_bytes(
        bytes
            .try_into()
            .expect("every decoded logit must occupy exactly four bytes"),
    )
}

fn assert_hf_bf16_eager_trace_phase(
    root: &Path,
    phase: &HfBf16EagerTracePhase,
    expected_sequence_length: usize,
) {
    let post_embed = phase
        .records
        .iter()
        .find(|record| record.tap_name == "post_embed")
        .expect("every trace phase must include its post-embedding tensor");
    let pre_lm_head = phase
        .records
        .iter()
        .find(|record| record.tap_name == "pre_lm_head")
        .expect("every trace phase must include its pre-lm-head tensor");
    let mut pre_layers = BTreeMap::new();
    let mut post_layers = BTreeMap::new();
    let mut post_loop_norms = BTreeMap::new();

    for record in &phase.records {
        match record.tap_name.as_str() {
            "pre_layer" => {
                let key = (
                    record
                        .loop_index
                        .expect("pre-layer record must name a loop"),
                    record.layer.expect("pre-layer record must name a layer"),
                );
                assert!(
                    pre_layers.insert(key, record).is_none(),
                    "trace phase must not duplicate pre-layer tap {key:?}"
                );
            }
            "post_layer" => {
                let key = (
                    record
                        .loop_index
                        .expect("post-layer record must name a loop"),
                    record.layer.expect("post-layer record must name a layer"),
                );
                assert!(
                    post_layers.insert(key, record).is_none(),
                    "trace phase must not duplicate post-layer tap {key:?}"
                );
            }
            "post_loop_norm" => {
                let loop_index = record
                    .loop_index
                    .expect("post-loop-norm record must name a loop");
                assert!(
                    post_loop_norms.insert(loop_index, record).is_none(),
                    "trace phase must not duplicate post-loop norm {loop_index}"
                );
            }
            _ => {}
        }
    }

    assert_eq!(
        pre_layers.len(),
        NANBEIGE_LOGICAL_LAYER_COUNT,
        "{} trace must retain each logical pre-layer state",
        phase.phase,
    );
    assert_eq!(
        post_layers.len(),
        NANBEIGE_LOGICAL_LAYER_COUNT,
        "{} trace must retain every one of the 44 logical post-layer states",
        phase.phase,
    );
    assert_eq!(
        post_loop_norms.len(),
        2,
        "{} trace must retain both post-loop norm states",
        phase.phase,
    );

    assert_eq!(post_embed.dtype, "bfloat16");
    assert_eq!(pre_lm_head.dtype, "bfloat16");
    assert_trace_tensor_integrity(root, post_embed);

    for loop_index in 0..2 {
        let post_loop_norm = post_loop_norms.get(&loop_index).unwrap_or_else(|| {
            panic!(
                "{} trace is missing post-loop norm {loop_index}",
                phase.phase
            )
        });
        assert_eq!(post_loop_norm.dtype, "bfloat16");
        assert_eq!(
            post_loop_norm.shape,
            vec![1, expected_sequence_length, NANBEIGE_HIDDEN_SIZE]
        );
        assert_trace_tensor_integrity(root, post_loop_norm);

        for layer in 0..NANBEIGE_PHYSICAL_LAYER_COUNT {
            let key = (loop_index, layer);
            let pre_layer = pre_layers.get(&key).unwrap_or_else(|| {
                panic!("{} trace is missing pre-layer tap {key:?}", phase.phase)
            });
            let post_layer = post_layers.get(&key).unwrap_or_else(|| {
                panic!("{} trace is missing post-layer tap {key:?}", phase.phase)
            });
            for tensor in [*pre_layer, *post_layer] {
                assert_eq!(tensor.dtype, "bfloat16");
                assert_eq!(
                    tensor.shape,
                    vec![1, expected_sequence_length, NANBEIGE_HIDDEN_SIZE],
                    "{} trace layer tap {key:?} has an unexpected shape",
                    phase.phase,
                );
                assert_trace_tensor_integrity(root, tensor);
            }
            let predecessor = if layer == 0 && loop_index == 0 {
                post_embed
            } else if layer == 0 {
                post_loop_norms
                    .get(&(loop_index - 1))
                    .expect("second loop must follow the first post-loop norm")
            } else {
                post_layers
                    .get(&(loop_index, layer - 1))
                    .expect("each layer after zero must follow its preceding layer")
            };
            assert_trace_tensor_continuity(
                pre_layer,
                predecessor,
                &format!("{} trace pre-layer {key:?}", phase.phase),
            );
        }
    }

    assert_trace_tensor_integrity(root, pre_lm_head);
    assert_trace_tensor_continuity(
        pre_lm_head,
        post_loop_norms
            .get(&1)
            .expect("second post-loop norm must exist"),
        &format!("{} trace pre-lm-head", phase.phase),
    );
}

#[test]
fn hf_bf16_eager_fixture_binds_the_44_layer_l2_ladder_and_greedy_seed() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(HF_BF16_EAGER_TRACE_PROMPT);
    let trace: HfBf16EagerTrace = serde_json::from_slice(
        &fs::read(root.join("trace.json")).expect("read hf-bf16 eager oracle trace"),
    )
    .expect("parse hf-bf16 eager oracle trace");
    assert_eq!(trace.profile, HF_BF16_EAGER_PROFILE);

    assert_hf_bf16_eager_trace_phase(&root, &trace.append, 1);
    assert_hf_bf16_eager_trace_phase(&root, &trace.prefill, 9);

    assert_eq!(trace.logits.tap_name, "logits");
    assert_eq!(trace.logits.dtype, "float32");
    assert_eq!(trace.logits.shape, vec![1, 9, NANBEIGE_VOCAB_SIZE]);
    let logits = assert_trace_tensor_integrity(&root, &trace.logits);
    let final_row_start = logits
        .len()
        .checked_sub(NANBEIGE_F32_LOGIT_BYTES)
        .expect("prefill logits must contain one full vocabulary row");
    let greedy_seed = logits[final_row_start..]
        .chunks_exact(std::mem::size_of::<f32>())
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            decode_native_f32(left).total_cmp(&decode_native_f32(right))
        })
        .expect("prefill logits must not be empty")
        .0 as u32;
    assert_eq!(
        trace.greedy_tokens.first().copied(),
        Some(greedy_seed),
        "the frozen greedy stream must begin with the prefill final-row argmax"
    );

    eprintln!(
        "BF16_EAGER_FIXTURE_CONTRACT RESULT=PASS logical_layers=44 post_loop_norms=2 greedy_seed={greedy_seed}"
    );
}

/// Exercises the actual source-backed semantic engine once the release or
/// parity runner arms this model-gated check.  Fixture-integrity coverage above
/// proves the frozen oracle inventory; `zre` owns the separate per-tensor L2
/// metric comparison against those bytes.
#[test]
fn bf16_eager_armed_source_runs_all_44_layers_and_both_boundary_norms() {
    let Some(source) = armed_pinned_model_source() else {
        return;
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(HF_BF16_EAGER_TRACE_PROMPT);
    let trace: HfBf16EagerTrace = serde_json::from_slice(
        &fs::read(root.join("trace.json")).expect("read hf-bf16 eager oracle trace"),
    )
    .expect("parse hf-bf16 eager oracle trace");
    let token_id = *trace
        .greedy_tokens
        .first()
        .expect("frozen eager trace must retain a greedy seed");

    let weights = HfBf16EagerWeights::from_pinned_source(&source)
        .expect("armed source closure must satisfy the pinned bf16 tensor census");
    let mut engine = HfBf16EagerEngine::new(weights, 1)
        .expect("one-position eager engine must admit all 44 bf16 K/V slots");
    let forward = engine
        .decode(token_id)
        .expect("armed eager source must execute the frozen greedy seed");

    assert_eq!(forward.position, 0);
    assert_eq!(forward.layer_outputs.len(), NANBEIGE_LOGICAL_LAYER_COUNT);
    for (logical_slot, output) in forward.layer_outputs.iter().enumerate() {
        let loop_index = logical_slot / NANBEIGE_PHYSICAL_LAYER_COUNT;
        let layer_index = logical_slot % NANBEIGE_PHYSICAL_LAYER_COUNT;
        assert_eq!(output.loop_index, loop_index);
        assert_eq!(output.layer_index, layer_index);
        assert_eq!(output.kv_slot, logical_slot);
        assert_eq!(output.hidden.len(), NANBEIGE_HIDDEN_SIZE);
    }
    for norm in &forward.post_loop_norms {
        assert_eq!(norm.len(), NANBEIGE_HIDDEN_SIZE);
    }
    assert_eq!(forward.logits.len(), NANBEIGE_VOCAB_SIZE);
    assert_eq!(greedy_argmax(&forward.logits), Some(forward.greedy_token));
    for slot in 0..NANBEIGE_LOGICAL_LAYER_COUNT {
        assert_eq!(engine.kv_cache().len_for_slot(slot), Ok(1));
    }

    eprintln!(
        "BF16_EAGER RESULT=PASS taps={} norms={} source=armed greedy_token={}",
        forward.layer_outputs.len(),
        forward.post_loop_norms.len(),
        forward.greedy_token,
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

    let cast_site_input = bf16s(&[3.4375, 2.578125, -0.79296875, -2.40625]);
    let cast_site_weight = bf16s(&[0.06787109375, -0.5703125, 1.703125, -1.1796875]);
    let cast_site_output =
        rms_norm_f32_reduce_cast_back(&cast_site_input, &cast_site_weight, RMS_NORM_EPSILON)
            .expect("cast-site regression vector has matching shape");
    assert_eq!(
        cast_site_output,
        bf16s(&[0.09326171875, -0.58984375, -0.54296875, 1.140625]),
        "RMSNorm must narrow the normalized activation before the bf16 scale multiply"
    );

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

#[test]
fn bf16_eager_source_loader_refuses_an_unverified_source_root() {
    let missing_source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/reference/absent-pinned-source-closure");
    let error = HfBf16EagerWeights::from_pinned_source(missing_source)
        .expect_err("the source loader must authenticate the complete pinned closure first");
    assert!(matches!(
        error,
        HfBf16EagerSourceError::SourceClosure { .. }
    ));
}
