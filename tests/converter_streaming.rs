//! Public routed-panel transform contracts.
//!
//! These source-free checks bind the converter's canonical route table to the
//! bounded Generic bytes that the future streaming envelope will receive. They
//! deliberately do not execute `fnlp convert`; real closure conversion remains
//! controller-owned.

use franken_nlp::artifact::converter::{
    ConverterError, GenericPanel, StorageStage, remap_tensor_name, transform_routed_panel,
};
use franken_nlp::artifact::quantize::QuantizeError;
use franken_nlp::artifact::safetensors::{RowPanel, SafetensorDtype, TensorCensusEntry};

const ONE_ROW_BF16: [u8; 8] = [0x80, 0x3f, 0x80, 0xbf, 0x00, 0x3f, 0x00, 0xbf];
const ONE_ROW_F32: [f32; 4] = [1.0, -1.0, 0.5, -0.5];

fn one_row_entry(name: &str) -> TensorCensusEntry {
    TensorCensusEntry {
        name: name.to_owned(),
        dtype: SafetensorDtype::Bf16,
        shape: vec![1, 4],
        len: 8,
    }
}

#[test]
fn routed_int8_panels_have_one_canonical_generic_byte_spelling() {
    let expected_scale = (1.0_f32 / 127.0).to_bits().to_le_bytes().to_vec();
    let expected_row_sum = 0_i32.to_le_bytes().to_vec();
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

    for (name, expected_stage) in routes {
        let entry = one_row_entry(name);
        let route = remap_tensor_name(name).expect("named converter route");
        assert_eq!(route.stage, expected_stage, "route={name}");

        let GenericPanel::Int8 {
            values,
            scales_le,
            row_sums_le,
        } = transform_routed_panel(
            &entry,
            &route,
            RowPanel::Rows {
                start_row: 0,
                row_count: 1,
            },
            &ONE_ROW_BF16,
            &ONE_ROW_F32,
        )
        .expect("finite verified panel")
        else {
            panic!("int8 route must emit int8 Generic bytes: route={name}");
        };

        assert_eq!(values, vec![127, -127, 64, -64], "route={name}");
        assert_eq!(scales_le, expected_scale, "route={name}");
        assert_eq!(row_sums_le, expected_row_sum, "route={name}");
    }
}

#[test]
fn routed_bf16_panel_preserves_the_verified_source_bytes_exactly() {
    let entry = one_row_entry("model.embed_tokens.weight");
    let route = remap_tensor_name(&entry.name).expect("named converter route");

    assert_eq!(route.stage, StorageStage::Bf16Verbatim);
    assert_eq!(
        transform_routed_panel(
            &entry,
            &route,
            RowPanel::Rows {
                start_row: 0,
                row_count: 1,
            },
            &ONE_ROW_BF16,
            &ONE_ROW_F32,
        )
        .expect("verified BF16 route"),
        GenericPanel::Bf16Verbatim {
            bytes: ONE_ROW_BF16.to_vec(),
        }
    );
}

#[test]
fn routed_int8_panel_preserves_the_typed_nonfinite_refusal() {
    let entry = one_row_entry("lm_head.weight");
    let route = remap_tensor_name(&entry.name).expect("named converter route");
    let mut nonfinite = ONE_ROW_F32;
    nonfinite[2] = f32::NAN;

    assert_eq!(
        transform_routed_panel(
            &entry,
            &route,
            RowPanel::Rows {
                start_row: 0,
                row_count: 1,
            },
            &ONE_ROW_BF16,
            &nonfinite,
        ),
        Err(ConverterError::Quantize(QuantizeError::NonFinite {
            row: 0,
            column: 2,
            bits: f32::NAN.to_bits(),
        }))
    );
}
