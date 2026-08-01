//! Always-on converter-core contracts; real source closure conversion remains
//! model-gated in `scripts/e2e_convert_roundtrip.sh`.

use franken_nlp::artifact::converter::{
    BF16_VERBATIM_V1, CONVERSION_RECEIPT_SCHEMA, ConversionReceipt, ConversionSourceManifest,
    ConvertArch, ConvertRequest, ConverterError, DEFAULT_PANEL_BYTES, GENERIC_PACKING_V1,
    OutputRange, OutputRangePlan, PINNED_LOGICAL_PAYLOAD_BYTES, PORTABLE_QUANT_V1, StorageStage,
    expected_nanbeige42_census, plan_generic_payload, prepare_convert_request, remap_tensor_name,
    validate_nanbeige42_census, validate_pinned_logical_payload_bytes,
};
use franken_nlp::artifact::safetensors::TensorCensusEntry;

#[test]
fn pinned_source_manifest_is_closed_and_has_the_required_accounting() {
    let manifest = ConversionSourceManifest::parse(include_str!(
        "../docs/truth-pack/nanbeige4.2-3b.source.json"
    ))
    .expect("committed canonical source manifest");

    assert_eq!(manifest.files.len(), 10);
    assert_eq!(manifest.closure_total_bytes, 8_360_887_509);
    assert_eq!(manifest.logical_safetensors_payload_bytes, 8_339_601_408);
}

#[test]
fn exact_generated_census_is_complete_and_routes_every_tensor() {
    let expected = expected_nanbeige42_census();
    assert_eq!(expected.len(), 201);

    for tensor in expected {
        let route = remap_tensor_name(&tensor.name).expect("complete converter remap");
        if tensor.name.ends_with("input_layernorm.weight") {
            assert_eq!(route.stage, StorageStage::Bf16Verbatim);
        }
    }
}

#[test]
fn generic_payload_plan_precomputes_exact_int8_and_bf16_section_ranges() {
    let census = vec![
        TensorCensusEntry {
            name: "model.embed_tokens.weight".to_owned(),
            dtype: franken_nlp::artifact::safetensors::SafetensorDtype::Bf16,
            shape: vec![3, 2],
            len: 12,
        },
        TensorCensusEntry {
            name: "lm_head.weight".to_owned(),
            dtype: franken_nlp::artifact::safetensors::SafetensorDtype::Bf16,
            shape: vec![4, 2],
            len: 16,
        },
    ];
    let routes = census
        .iter()
        .map(|entry| remap_tensor_name(&entry.name).expect("named converter route"))
        .collect::<Vec<_>>();

    let plan = plan_generic_payload(&census, &routes).expect("complete Generic layout");

    assert_eq!(plan.payload_bytes, 20);
    assert_eq!(plan.scale_bytes, 16);
    assert_eq!(plan.row_sum_bytes, 16);
    assert_eq!(plan.tensors[0].internal_name, "embed");
    assert_eq!(plan.tensors[0].quantization, BF16_VERBATIM_V1);
    assert_eq!(plan.tensors[0].data.offset, 0);
    assert_eq!(plan.tensors[0].data.len, 12);
    assert_eq!(plan.tensors[0].scale.len, 0);
    assert_eq!(plan.tensors[0].row_sum.len, 0);
    assert_eq!(plan.tensors[1].internal_name, "lm_head");
    assert_eq!(plan.tensors[1].quantization, PORTABLE_QUANT_V1);
    assert_eq!(plan.tensors[1].data.offset, 12);
    assert_eq!(plan.tensors[1].data.len, 8);
    assert_eq!(plan.tensors[1].scale.offset, 0);
    assert_eq!(plan.tensors[1].scale.len, 16);
    assert_eq!(plan.tensors[1].row_sum.offset, 0);
    assert_eq!(plan.tensors[1].row_sum.len, 16);
}

#[test]
fn census_validation_reports_a_clean_round_trip() {
    let actual: Vec<_> = expected_nanbeige42_census()
        .into_iter()
        .map(|tensor| TensorCensusEntry {
            name: tensor.name,
            dtype: tensor.dtype,
            shape: tensor.shape,
            len: tensor.len,
        })
        .collect();

    assert!(
        validate_nanbeige42_census(&actual)
            .expect("matching census")
            .is_match()
    );
    assert_eq!(
        validate_pinned_logical_payload_bytes(&actual).expect("pinned payload total"),
        PINNED_LOGICAL_PAYLOAD_BYTES
    );
}

#[test]
fn payload_total_refuses_a_census_byte_drift() {
    let mut actual: Vec<_> = expected_nanbeige42_census()
        .into_iter()
        .map(|tensor| TensorCensusEntry {
            name: tensor.name,
            dtype: tensor.dtype,
            shape: tensor.shape,
            len: tensor.len,
        })
        .collect();
    actual[0].len -= 2;

    assert!(matches!(
        validate_pinned_logical_payload_bytes(&actual),
        Err(ConverterError::CensusPayloadBytes { expected, actual })
            if expected == PINNED_LOGICAL_PAYLOAD_BYTES
                && actual == PINNED_LOGICAL_PAYLOAD_BYTES - 2
    ));
}

#[test]
fn oq1_extra_is_a_design_assumption_abort_not_a_tolerated_extra() {
    let mut actual: Vec<_> = expected_nanbeige42_census()
        .into_iter()
        .map(|tensor| TensorCensusEntry {
            name: tensor.name,
            dtype: tensor.dtype,
            shape: tensor.shape,
            len: tensor.len,
        })
        .collect();
    actual.push(TensorCensusEntry {
        name: "model.layers.0.mHC.weight".to_owned(),
        dtype: franken_nlp::artifact::safetensors::SafetensorDtype::Bf16,
        shape: vec![1],
        len: 2,
    });

    let error = validate_nanbeige42_census(&actual).expect_err("OQ-1 must abort");
    assert!(error.to_string().contains("OQ-1"));
}

#[test]
fn receipt_requires_every_identity_and_serializes_canonically() {
    let receipt = ConversionReceipt {
        receipt_schema: CONVERSION_RECEIPT_SCHEMA.to_owned(),
        source_root_sha256: "a".repeat(64),
        census_sha256: "b".repeat(64),
        converter_commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
        recipe_id: "nanbeige42-int8-v1".to_owned(),
        rounding_id: "portable-quant-v1".to_owned(),
        packing_id: "generic-v1".to_owned(),
        measured_peak_rss_bytes: 10,
        measured_scratch_bytes: 4,
        peak_rss_cap_bytes: 10,
        final_disk_bytes: 20,
        measured_disk_bytes: 20,
        output_len: 20,
        output_sha256: "c".repeat(64),
        license_bundle_sha256: "d".repeat(64),
    };

    let json = receipt.canonical_json().expect("complete receipt");
    assert!(json.contains("\"recipe_id\":\"nanbeige42-int8-v1\""));
    assert_eq!(
        ConversionReceipt::parse_canonical_json(&json).expect("canonical receipt parses"),
        receipt
    );
    assert_eq!(
        ConversionReceipt::parse_canonical_json(&format!("{json}\n")),
        Err(ConverterError::ReceiptNonCanonical)
    );
    let mut unknown_field =
        serde_json::from_str::<serde_json::Value>(&json).expect("canonical receipt is JSON");
    unknown_field
        .as_object_mut()
        .expect("receipt root is an object")
        .insert("unexpected".to_owned(), serde_json::Value::Null);
    let unknown_field = franken_nlp::canonjson::canonical_string(&unknown_field)
        .expect("hostile unknown-field fixture is canonical JSON");
    assert!(matches!(
        ConversionReceipt::parse_canonical_json(&unknown_field),
        Err(ConverterError::ReceiptParse { .. })
    ));

    let undersized_disk = ConversionReceipt {
        output_len: 21,
        measured_disk_bytes: 21,
        ..receipt.clone()
    };
    assert_eq!(
        undersized_disk.canonical_json(),
        Err(ConverterError::ReceiptField {
            field: "final_disk_bytes",
            detail: "must cover measured_disk_bytes".to_owned(),
        })
    );

    let excessive_scratch = ConversionReceipt {
        measured_scratch_bytes: 11,
        ..receipt.clone()
    };
    assert_eq!(
        excessive_scratch.canonical_json(),
        Err(ConverterError::ReceiptField {
            field: "measured_scratch_bytes",
            detail: "must not exceed measured_peak_rss_bytes".to_owned(),
        })
    );

    let unpinned_recipe = ConversionReceipt {
        recipe_id: "other-recipe-v1".to_owned(),
        ..receipt.clone()
    };
    assert_eq!(
        unpinned_recipe.canonical_json(),
        Err(ConverterError::ReceiptField {
            field: "recipe_id",
            detail:
                "expected pinned identifier \"nanbeige42-int8-v1\", observed \"other-recipe-v1\""
                    .to_owned(),
        })
    );
    let unpinned_rounding = ConversionReceipt {
        rounding_id: "other-rounding-v1".to_owned(),
        ..receipt.clone()
    };
    assert_eq!(
        unpinned_rounding.canonical_json(),
        Err(ConverterError::ReceiptField {
            field: "rounding_id",
            detail:
                "expected pinned identifier \"portable-quant-v1\", observed \"other-rounding-v1\""
                    .to_owned(),
        })
    );
    let unpinned_packing = ConversionReceipt {
        packing_id: "other-packing-v1".to_owned(),
        ..receipt.clone()
    };
    assert_eq!(
        unpinned_packing.canonical_json(),
        Err(ConverterError::ReceiptField {
            field: "packing_id",
            detail: format!(
                "expected pinned identifier {GENERIC_PACKING_V1:?}, observed \"other-packing-v1\""
            ),
        })
    );

    let malformed_commit = ConversionReceipt {
        converter_commit: "unbound-converter".to_owned(),
        ..receipt
    };
    assert_eq!(
        malformed_commit.canonical_json(),
        Err(ConverterError::ReceiptField {
            field: "converter_commit",
            detail: "must be a lowercase 40-character Git commit".to_owned(),
        })
    );
}

#[test]
fn receipt_parser_rejects_duplicate_missing_and_wrongly_typed_schema_fields() {
    let receipt = ConversionReceipt {
        receipt_schema: CONVERSION_RECEIPT_SCHEMA.to_owned(),
        source_root_sha256: "a".repeat(64),
        census_sha256: "b".repeat(64),
        converter_commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
        recipe_id: "nanbeige42-int8-v1".to_owned(),
        rounding_id: "portable-quant-v1".to_owned(),
        packing_id: "generic-v1".to_owned(),
        measured_peak_rss_bytes: 10,
        measured_scratch_bytes: 4,
        peak_rss_cap_bytes: 10,
        final_disk_bytes: 20,
        measured_disk_bytes: 20,
        output_len: 20,
        output_sha256: "c".repeat(64),
        license_bundle_sha256: "d".repeat(64),
    };
    let canonical = receipt
        .canonical_json()
        .expect("complete synthetic receipt serializes");

    let duplicate_receipt_schema = canonical.replacen(
        "\"receipt_schema\":\"fnlp-conversion-receipt-v1\"",
        "\"receipt_schema\":\"fnlp-conversion-receipt-v1\",\"receipt_schema\":\"fnlp-conversion-receipt-v1\"",
        1,
    );
    assert!(matches!(
        ConversionReceipt::parse_canonical_json(&duplicate_receipt_schema),
        Err(ConverterError::ReceiptJson(_))
    ));

    let mut missing_output_sha256 =
        serde_json::from_str::<serde_json::Value>(&canonical).expect("canonical receipt is JSON");
    missing_output_sha256
        .as_object_mut()
        .expect("receipt root is an object")
        .remove("output_sha256");
    let missing_output_sha256 = franken_nlp::canonjson::canonical_string(&missing_output_sha256)
        .expect("missing-field hostile receipt stays canonical JSON");
    assert!(matches!(
        ConversionReceipt::parse_canonical_json(&missing_output_sha256),
        Err(ConverterError::ReceiptParse { .. })
    ));

    let mut unpinned_recipe =
        serde_json::from_str::<serde_json::Value>(&canonical).expect("canonical receipt is JSON");
    unpinned_recipe
        .as_object_mut()
        .expect("receipt root is an object")
        .insert(
            "recipe_id".to_owned(),
            serde_json::Value::from("unbound-recipe-v1"),
        );
    let unpinned_recipe = franken_nlp::canonjson::canonical_string(&unpinned_recipe)
        .expect("unpinned recipe hostile receipt stays canonical JSON");
    assert!(matches!(
        ConversionReceipt::parse_canonical_json(&unpinned_recipe),
        Err(ConverterError::ReceiptField {
            field: "recipe_id",
            ..
        })
    ));

    let mut numeric_source_root =
        serde_json::from_str::<serde_json::Value>(&canonical).expect("canonical receipt is JSON");
    numeric_source_root
        .as_object_mut()
        .expect("receipt root is an object")
        .insert(
            "source_root_sha256".to_owned(),
            serde_json::Value::from(7_u64),
        );
    let numeric_source_root = franken_nlp::canonjson::canonical_string(&numeric_source_root)
        .expect("wrong-type hostile receipt stays canonical JSON");
    assert!(matches!(
        ConversionReceipt::parse_canonical_json(&numeric_source_root),
        Err(ConverterError::ReceiptParse { .. })
    ));
}

#[test]
fn cli_contract_admits_only_recipe_and_generic_arch() {
    let request = ConvertRequest {
        source_dir: "source".into(),
        source_manifest: "manifest.json".into(),
        recipe_id: "nanbeige42-int8-v1".to_owned(),
        arch: ConvertArch::parse("generic").expect("generic arch"),
        output: "artifact.fnlpq".into(),
        yes: true,
        strict_source_dir: false,
        robot: false,
    };
    request.validate().expect("reference invocation contract");
    assert!(ConvertArch::parse("x86-avx2").is_err());
}

#[test]
fn conversion_admission_rejects_the_recipe_before_source_access() {
    let request = ConvertRequest {
        source_dir: "/unreachable-source".into(),
        source_manifest: "/unreachable-manifest.json".into(),
        recipe_id: "unapproved-recipe".to_owned(),
        arch: ConvertArch::Generic,
        output: "artifact.fnlpq".into(),
        yes: false,
        strict_source_dir: false,
        robot: false,
    };

    assert!(matches!(
        prepare_convert_request(&request, DEFAULT_PANEL_BYTES),
        Err(ConverterError::InvalidConvertArgument {
            argument: "--recipe",
            ..
        })
    ));
}

#[test]
fn externally_supplied_output_range_plans_refuse_duplicate_names() {
    let plan = OutputRangePlan {
        ranges: vec![
            OutputRange {
                name: "data".to_owned(),
                offset: 0,
                len: 4,
            },
            OutputRange {
                name: "data".to_owned(),
                offset: 4,
                len: 2,
            },
        ],
        file_len: 6,
    };

    assert_eq!(
        plan.validate(),
        Err(ConverterError::DuplicateOutputRange {
            name: "data".to_owned(),
        })
    );
}
