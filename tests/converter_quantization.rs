//! Cross-module Generic-panel contract between converter routing and the
//! portable stage quantizer. These checks stay source-free and do not execute
//! `fnlp convert`; the real closure conversion remains controller-owned.

use franken_nlp::artifact::converter::{
    remap_tensor_name, stream_routed_bf16_panels, transform_routed_panel, ConverterError,
    GenericPanel, PanelPlan, StorageStage,
};
use franken_nlp::artifact::quantize::encode_generic_panel;
use franken_nlp::artifact::safetensors::{RowPanel, SafetensorDtype, TensorCensusEntry};

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

#[test]
fn routed_stream_applies_each_storage_stage_without_retaining_the_tensor() {
    let source_names = [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
    ];
    let census = source_names
        .into_iter()
        .map(|name| TensorCensusEntry {
            name: name.to_owned(),
            dtype: SafetensorDtype::Bf16,
            shape: vec![1, 4],
            len: 8,
        })
        .collect::<Vec<_>>();
    let routes = census
        .iter()
        .map(|entry| remap_tensor_name(&entry.name).expect("named converter route"))
        .collect::<Vec<_>>();
    let plans = census
        .iter()
        .map(|entry| PanelPlan::for_tensor(entry, 8).expect("one bounded row"))
        .collect::<Vec<_>>();

    let mut produced = Vec::new();
    let report = stream_routed_bf16_panels(
        &census,
        &routes,
        &plans,
        |_, _| Ok(ONE_ROW_BF16.to_vec()),
        |entry, route, panel, source_bf16, decoded_f32| {
            let generic = transform_routed_panel(entry, route, panel, source_bf16, decoded_f32)
                .expect("canonical route transforms one bounded panel");
            produced.push((entry.name.clone(), generic));
            Ok(())
        },
    )
    .expect("every panel is consumed in source order");

    assert_eq!(report.tensors, 4);
    assert_eq!(report.panels, 4);
    assert_eq!(report.source_bytes, 32);
    assert_eq!(report.f32_work_bytes, 64);
    assert_eq!(produced.len(), 4);
    assert!(matches!(
        &produced[0].1,
        GenericPanel::Bf16Verbatim { bytes } if bytes == &ONE_ROW_BF16
    ));

    let scale_bytes = (1.0_f32 / 127.0).to_bits().to_le_bytes();
    for (source_name, generic) in produced.iter().skip(1) {
        assert!(
            matches!(
                generic,
                GenericPanel::Int8 {
                    values,
                    scales_le,
                    row_sums_le,
                } if values == &vec![127, -127, 64, -64]
                    && scales_le == &scale_bytes
                    && row_sums_le == &0_i32.to_le_bytes()
            ),
            "route={source_name}"
        );
    }
}

#[test]
fn routed_transform_refuses_a_substituted_f32_work_panel() {
    let entry = TensorCensusEntry {
        name: "model.layers.0.mlp.gate_proj.weight".to_owned(),
        dtype: SafetensorDtype::Bf16,
        shape: vec![1, 4],
        len: 8,
    };
    let route = remap_tensor_name(&entry.name).expect("named converter route");
    let error = transform_routed_panel(
        &entry,
        &route,
        RowPanel::Rows {
            start_row: 0,
            row_count: 1,
        },
        &ONE_ROW_BF16,
        &[1.0, -1.0, 0.75, -0.5],
    )
    .expect_err("substituted f32 work must not enter generic quantization");

    assert!(matches!(
        error,
        ConverterError::Quantize(
            franken_nlp::artifact::quantize::QuantizeError::DecodedBf16Mismatch { element: 2, .. }
        )
    ));
}
