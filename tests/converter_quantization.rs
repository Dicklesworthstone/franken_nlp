//! Cross-module Generic-panel contract between converter routing and the
//! portable stage quantizer. These checks stay source-free and do not execute
//! `fnlp convert`; the real closure conversion remains controller-owned.

use franken_nlp::artifact::converter::{remap_tensor_name, StorageStage};
use franken_nlp::artifact::quantize::encode_generic_panel;

const ONE_ROW_BF16: [u8; 8] = [0x80, 0x3f, 0x80, 0xbf, 0x00, 0x3f, 0x00, 0xbf];
const ONE_ROW_F32: [f32; 4] = [1.0, -1.0, 0.5, -0.5];

#[test]
fn every_int8_route_emits_the_same_portable_generic_panel_contract() {
    let scale_bytes = (1.0_f32 / 127.0).to_bits().to_le_bytes();
    let expected_data = [127_u8, 129, 64, 192];
    let expected_row_sum = 0_i32.to_le_bytes();
    let routes = [
        (
            "model.layers.0.mlp.gate_proj.weight",
            StorageStage::Int8Stage2A,
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            StorageStage::Int8Stage2B,
        ),
        ("lm_head.weight", StorageStage::Int8Stage2C),
    ];

    for (source_name, expected_stage) in routes {
        let route = remap_tensor_name(source_name).expect("named converter route");
        assert_eq!(route.stage, expected_stage, "route={source_name}");

        let panel = encode_generic_panel(route.stage, &ONE_ROW_BF16, &ONE_ROW_F32, 1, 4)
            .expect("validated finite panel");
        assert_eq!(panel.data, expected_data, "route={source_name}");
        assert_eq!(panel.scales, scale_bytes, "route={source_name}");
        assert_eq!(panel.row_sums, expected_row_sum, "route={source_name}");
    }
}

#[test]
fn bf16_route_preserves_source_bytes_without_quantization_metadata() {
    let route =
        remap_tensor_name("model.layers.0.input_layernorm.weight").expect("named converter route");
    assert_eq!(route.stage, StorageStage::Bf16Verbatim);

    let panel = encode_generic_panel(route.stage, &ONE_ROW_BF16, &ONE_ROW_F32, 1, 4)
        .expect("validated BF16 panel");
    assert_eq!(panel.data, ONE_ROW_BF16);
    assert!(panel.scales.is_empty());
    assert!(panel.row_sums.is_empty());
}
