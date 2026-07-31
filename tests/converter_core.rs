//! Always-on converter-core contracts; real source closure conversion remains
//! model-gated in `scripts/e2e_convert_roundtrip.sh`.

use franken_nlp::artifact::converter::{
    expected_nanbeige42_census, remap_tensor_name, validate_nanbeige42_census, ConversionReceipt,
    ConversionSourceManifest, ConvertArch, ConvertRequest, StorageStage,
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

    assert!(validate_nanbeige42_census(&actual)
        .expect("matching census")
        .is_match());
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
        source_root_sha256: "a".repeat(64),
        census_sha256: "b".repeat(64),
        converter_commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
        recipe_id: "nanbeige42-int8-v1".to_owned(),
        rounding_id: "portable-quant-v1".to_owned(),
        packing_id: "generic-v1".to_owned(),
        measured_peak_rss_bytes: 10,
        peak_rss_cap_bytes: 10,
        final_disk_bytes: 20,
        output_len: 20,
        output_sha256: "c".repeat(64),
        license_bundle_sha256: "d".repeat(64),
    };

    let json = receipt.canonical_json().expect("complete receipt");
    assert!(json.contains("\"recipe_id\":\"nanbeige42-int8-v1\""));
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
