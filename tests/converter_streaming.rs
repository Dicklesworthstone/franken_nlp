//! Public routed-panel transform contracts.
//!
//! These source-free checks bind the converter's canonical route table to the
//! bounded Generic bytes that the future streaming envelope will receive. They
//! deliberately do not execute `fnlp convert`; real closure conversion remains
//! controller-owned.

use franken_nlp::artifact::converter::{
    ConversionPreflight, ConverterError, OutputRange, OutputRangePlan, PeakRssFormula,
    StorageStage, remap_tensor_name, transform_routed_panel,
};
use franken_nlp::artifact::quantize::{GenericPanelBytes, QuantizeError};
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

        let panel = transform_routed_panel(
            &entry,
            &route,
            RowPanel::Rows {
                start_row: 0,
                row_count: 1,
            },
            &ONE_ROW_BF16,
            &ONE_ROW_F32,
        )
        .expect("finite verified panel");

        assert_eq!(panel.data, vec![127_u8, 129, 64, 192], "route={name}");
        assert_eq!(panel.scales, expected_scale, "route={name}");
        assert_eq!(panel.row_sums, expected_row_sum, "route={name}");
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
        GenericPanelBytes {
            data: ONE_ROW_BF16.to_vec(),
            scales: Vec::new(),
            row_sums: Vec::new(),
        }
    );
}

#[test]
fn routed_int8_panel_preserves_the_typed_nonfinite_refusal() {
    let entry = one_row_entry("lm_head.weight");
    let route = remap_tensor_name(&entry.name).expect("named converter route");
    let mut source_with_nan = ONE_ROW_BF16;
    source_with_nan[4..6].copy_from_slice(&0x7fc0_u16.to_le_bytes());
    let mut nonfinite = ONE_ROW_F32;
    nonfinite[2] = f32::from_bits(0x7fc0_0000);

    assert_eq!(
        transform_routed_panel(
            &entry,
            &route,
            RowPanel::Rows {
                start_row: 0,
                row_count: 1,
            },
            &source_with_nan,
            &nonfinite,
        ),
        Err(ConverterError::Quantize(QuantizeError::NonFinite {
            row: 0,
            column: 2,
            bits: 0x7fc0_0000,
        }))
    );
}

#[test]
fn routed_int8_panel_refuses_a_tampered_decoded_element_before_quantization() {
    let entry = one_row_entry("lm_head.weight");
    let route = remap_tensor_name(&entry.name).expect("named converter route");
    let mut tampered = ONE_ROW_F32;
    tampered[2] = f32::from_bits(0x7fc0_0000);

    assert_eq!(
        transform_routed_panel(
            &entry,
            &route,
            RowPanel::Rows {
                start_row: 0,
                row_count: 1,
            },
            &ONE_ROW_BF16,
            &tampered,
        ),
        Err(ConverterError::Quantize(QuantizeError::DecodedBf16Mismatch {
            element: 2,
            expected_bits: 0x3f00_0000,
            observed_bits: 0x7fc0_0000,
        }))
    );
}

#[test]
fn routed_transform_refuses_a_whole_tensor_request_before_quantization() {
    let entry = one_row_entry("lm_head.weight");
    let route = remap_tensor_name(&entry.name).expect("named converter route");

    assert_eq!(
        transform_routed_panel(
            &entry,
            &route,
            RowPanel::WholeTensor,
            &ONE_ROW_BF16,
            &ONE_ROW_F32,
        ),
        Err(ConverterError::PipelinePlanAlignment {
            tensor: "lm_head.weight".to_owned(),
            detail: "routed transform requires a complete-row panel, not a whole-tensor panel"
                .to_owned(),
        })
    );
}

#[test]
fn output_range_plan_refuses_duplicate_names_and_arithmetic_overflow() {
    assert_eq!(
        OutputRangePlan::contiguous(&[("data".to_owned(), 4), ("data".to_owned(), 2)]),
        Err(ConverterError::DuplicateOutputRange {
            name: "data".to_owned(),
        })
    );
    assert_eq!(
        OutputRangePlan::contiguous(&[("data".to_owned(), u64::MAX), ("scales".to_owned(), 1)]),
        Err(ConverterError::Arithmetic {
            invariant: "output range end",
        })
    );
}

#[test]
fn output_range_plan_rejects_noncontiguous_or_incomplete_directories() {
    let gap = OutputRangePlan {
        ranges: vec![OutputRange {
            name: "data".to_owned(),
            offset: 4,
            len: 1,
        }],
        file_len: 5,
    };
    assert_eq!(
        gap.validate(),
        Err(ConverterError::OutputRangeLayout {
            name: "data".to_owned(),
            expected_offset: 0,
            actual_offset: 4,
        })
    );

    let incomplete = OutputRangePlan {
        ranges: vec![OutputRange {
            name: "data".to_owned(),
            offset: 0,
            len: 4,
        }],
        file_len: 5,
    };
    assert_eq!(
        incomplete.validate(),
        Err(ConverterError::OutputFileLength {
            expected: 4,
            actual: 5,
        })
    );
}

#[test]
fn conversion_preflight_keeps_the_machine_footprint_block_stable() {
    let preflight = ConversionPreflight {
        closure_bytes_to_read: 10,
        staged_output_bytes: 11,
        peak_rss: PeakRssFormula {
            largest_source_panel_bytes: 1,
            largest_f32_panel_bytes: 2,
            quant_packing_scratch_bytes: 3,
            output_buffer_bytes: 4,
            parser_metadata_bytes: 5,
            margin_bytes: 6,
        },
        final_disk_bytes: 12,
    };

    assert_eq!(
        preflight.stderr_block().expect("checked footprint formula"),
        "CONVERT PREFLIGHT closure-bytes=10 staged-output-bytes=11 peak-rss=largest-source-panel=1 + largest-f32-panel=2 + quant-packing-scratch=3 + output-buffer=4 + parser-metadata=5 + margin=6 = 21 bytes final-disk-bytes=12"
    );
}
