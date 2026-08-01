use franken_nlp::artifact::format::{
    framed_sha256, logical_model_sha256, logical_tensor_sha256, write, ArchTarget, CanonicalDtype,
    FnlpqWriterInput, PackingSetInput, SectionKind, SectionPayload, SectionRange, TensorInput,
};
use franken_nlp::artifact::packing::{
    derive_native_packing, require_native_packing, verify_derived_packing, NativeCacheAddress,
    NativePackingTarget, PackingError, TILE_TABLE_VERSION_V1,
};
use franken_nlp::artifact::reader::FnlpqArtifact;
use sha2::{Digest, Sha256};

#[test]
fn deterministic_synthetic_matrix_derives_and_recovers_all_five_targets() {
    for seed in 1..=16 {
        let generic_bytes = generic_root(seed);
        let generic = FnlpqArtifact::from_bytes(generic_bytes.clone())
            .expect("synthetic Generic root must validate");
        for target in NativePackingTarget::ALL {
            let first = derive_native_packing(generic_bytes.clone(), target, TILE_TABLE_VERSION_V1)
                .expect("every closed target derives from each synthetic Generic root");
            let second =
                derive_native_packing(generic_bytes.clone(), target, TILE_TABLE_VERSION_V1)
                    .expect("repeated deterministic derivation succeeds");
            assert_eq!(first, second, "seed={seed} target={target:?}");
            assert_ne!(first.bytes, generic_bytes, "derived bytes must be physical");
            assert_eq!(
                first.logical_model_sha256,
                generic.logical_model_sha256(),
                "seed={seed} target={target:?}"
            );
            assert!(
                first.footprint.peak_derivation_bytes
                    >= first.footprint.steady_state_retained_bytes,
                "reported materialized peak must cover retained Generic plus cache bytes"
            );
            let derived = FnlpqArtifact::from_bytes(first.bytes.clone())
                .expect("derived native envelope must validate");
            assert_ne!(
                derived.packing_set_sha256(),
                generic.packing_set_sha256(),
                "native packing set must have a distinct physical identity"
            );
            verify_derived_packing(&generic, &derived, target, TILE_TABLE_VERSION_V1)
                .expect("derived payload must exactly reconstruct Generic logical tensors");
            assert_eq!(
                generic_bytes,
                generic_root(seed),
                "derive must not mutate the Generic root bytes"
            );
        }
    }
}

#[test]
fn tile_table_version_is_part_of_the_cache_identity() {
    let generic_bytes = generic_root(17);
    let v1 = derive_native_packing(
        generic_bytes.clone(),
        NativePackingTarget::X86Avx2,
        TILE_TABLE_VERSION_V1,
    )
    .expect("v1 native derivation");
    let v2 = derive_native_packing(generic_bytes, NativePackingTarget::X86Avx2, "tile-table-v2")
        .expect("new tile table has a separate deterministic cache identity");

    assert_ne!(v1.address.packing_id, v2.address.packing_id);
    assert_ne!(v1.address.content_address, v2.address.content_address);
    assert_ne!(v1.bytes, v2.bytes);
    assert_eq!(v1.logical_model_sha256, v2.logical_model_sha256);
}

#[test]
fn cache_address_is_bound_to_a_closed_target_table_pair() {
    let root = "ab".repeat(32);
    let avx2 = NativeCacheAddress::for_target(
        root.clone(),
        NativePackingTarget::X86Avx2,
        TILE_TABLE_VERSION_V1.to_owned(),
    )
    .expect("closed AVX2 target/table pair forms an address");
    let vnni = NativeCacheAddress::for_target(
        root,
        NativePackingTarget::X86Vnni256,
        TILE_TABLE_VERSION_V1.to_owned(),
    )
    .expect("closed VNNI target/table pair forms an address");

    assert_eq!(avx2.packing_id, "x86-avx2-tile-table-v1");
    assert_ne!(avx2.packing_id, vnni.packing_id);
    assert_ne!(avx2.content_address, vnni.content_address);
    assert_eq!(
        avx2.cache_path("/owner-model-root"),
        std::path::PathBuf::from("/owner-model-root")
            .join("native")
            .join(&avx2.content_address)
            .join("x86-avx2-tile-table-v1.fnlpq")
    );
}

#[test]
fn cache_key_raw_sha_and_fnlpq_file_identity_remain_distinct() {
    let generic_bytes = generic_root(23);
    let derived = derive_native_packing(
        generic_bytes.clone(),
        NativePackingTarget::X86Avx2,
        TILE_TABLE_VERSION_V1,
    )
    .expect("synthetic native cache");

    assert_eq!(
        derived.address.whole_artifact_sha256,
        hex(&Sha256::digest(&generic_bytes)),
        "the content-address triple records raw Generic root SHA-256"
    );
    assert_eq!(
        derived.fnlpq_file_sha256,
        hex(&framed_sha256("fnlpq-file-v1", &[&derived.bytes])
            .expect("derived bytes admit the frozen domain frame")),
        "the .fnlpq physical identity uses the format's domain frame"
    );
    assert_ne!(
        derived.address.whole_artifact_sha256, derived.fnlpq_file_sha256,
        "raw Generic-root addressing and framed derived-file identity are distinct fields"
    );
}

#[test]
fn corrupted_native_cache_is_rejected_by_the_checked_reader() {
    let generic_bytes = generic_root(29);
    let derived = derive_native_packing(
        generic_bytes,
        NativePackingTarget::Aarch64I8mm,
        TILE_TABLE_VERSION_V1,
    )
    .expect("synthetic native cache");
    let checked =
        FnlpqArtifact::from_bytes(derived.bytes.clone()).expect("fresh native cache must validate");
    let native = checked
        .sections()
        .iter()
        .find(|section| section.kind == SectionKind::NativePackingPayload)
        .expect("derived cache has one native payload");
    let corrupt_offset = usize::try_from(native.file_offset).expect("fixture offset fits usize");
    let mut corrupted = derived.bytes;
    corrupted[corrupt_offset] ^= 0x01;

    assert!(
        FnlpqArtifact::from_bytes(corrupted).is_err(),
        "stored-section digest must reject a corrupt native cache before dispatch"
    );
}

#[test]
fn dispatch_refuses_cross_arch_fallback_and_names_the_derive_command() {
    let generic_bytes = generic_root(31);
    let generic =
        FnlpqArtifact::from_bytes(generic_bytes).expect("synthetic Generic root must validate");

    assert!(matches!(
        require_native_packing(
            &generic,
            NativePackingTarget::X86Vnni512,
            "/models/generic.fnlpq"
        ),
        Err(PackingError::MissingDerivation { command })
            if command == "fnlp models derive --generic /models/generic.fnlpq --arch x86-vnni512"
    ));
}

#[test]
fn derived_artifact_is_never_admitted_as_a_new_generic_root() {
    let generic = generic_root(37);
    let derived =
        derive_native_packing(generic, NativePackingTarget::X86Avx2, TILE_TABLE_VERSION_V1)
            .expect("synthetic native cache");

    assert!(matches!(
        derive_native_packing(
            derived.bytes,
            NativePackingTarget::X86Vnni256,
            TILE_TABLE_VERSION_V1,
        ),
        Err(PackingError::GenericRoot { detail })
            if detail == "root already contains a native packing payload"
    ));
}

#[test]
fn verification_refuses_a_different_native_target() {
    let generic_bytes = generic_root(41);
    let generic = FnlpqArtifact::from_bytes(generic_bytes.clone())
        .expect("synthetic Generic root must validate");
    let derived = derive_native_packing(
        generic_bytes,
        NativePackingTarget::X86Avx2,
        TILE_TABLE_VERSION_V1,
    )
    .expect("synthetic AVX2 native cache");
    let checked = FnlpqArtifact::from_bytes(derived.bytes)
        .expect("synthetic AVX2 native cache must validate");

    assert!(matches!(
        verify_derived_packing(
            &generic,
            &checked,
            NativePackingTarget::Aarch64Sdot,
            TILE_TABLE_VERSION_V1,
        ),
        Err(PackingError::NativePayload { detail })
            if detail.contains("does not match target aarch64-sdot")
    ));
}

#[test]
fn closed_cli_target_spellings_do_not_accept_envelope_aliases() {
    for target in NativePackingTarget::ALL {
        assert_eq!(
            NativePackingTarget::parse(target.cli_name()).expect("closed spelling parses"),
            target
        );
    }
    assert!(matches!(
        NativePackingTarget::parse("x86-vnni-256"),
        Err(PackingError::UnsupportedTarget { .. })
    ));
}

fn generic_root(seed: u64) -> Vec<u8> {
    let embed_payload = pseudo_random_bytes(seed, 4);
    let layer_payload = pseudo_random_bytes(seed.wrapping_add(1), 4);
    // The checked reader validates every scale as finite and strictly
    // positive. Keep tensor payload/row-sum bytes synthetic while giving each
    // one-group fixture a canonical valid f32 scale.
    let embed_scale = 0.5_f32.to_le_bytes();
    let layer_scale = 0.25_f32.to_le_bytes();
    let embed_row_sum = pseudo_random_bytes(seed.wrapping_add(2), 4);
    let layer_row_sum = pseudo_random_bytes(seed.wrapping_add(3), 4);
    let embed_tensor_sha256 = logical_tensor_sha256(
        "model.embed_tokens.weight",
        "bf16",
        &[2],
        "bf16-verbatim-v1",
        &embed_payload,
        &embed_scale,
        &embed_row_sum,
    )
    .expect("synthetic embedding logical tensor identity");
    let layer_tensor_sha256 = logical_tensor_sha256(
        "model.layers.0.mlp.down_proj.weight",
        "bf16",
        &[2],
        "bf16-verbatim-v1",
        &layer_payload,
        &layer_scale,
        &layer_row_sum,
    )
    .expect("synthetic layer logical tensor identity");
    let logical_model_sha256 = logical_model_sha256(
        &[embed_tensor_sha256, layer_tensor_sha256],
        &[
            ("model_config", br#"{"hidden_size":2}"#.as_slice()),
            ("tokenizer_model", &[0x50, 0x4b, 0x03, 0x04]),
            ("tokenizer_config", br#"{"bos_token":"<s>"}"#.as_slice()),
            ("chat_template", b"{% set x = 1 %}"),
        ],
    )
    .expect("synthetic logical model identity");
    let payload = [embed_payload.as_slice(), layer_payload.as_slice()].concat();
    let scales = [embed_scale.as_slice(), layer_scale.as_slice()].concat();
    let row_sums = [embed_row_sum.as_slice(), layer_row_sum.as_slice()].concat();
    write(&FnlpqWriterInput {
        model_id: "FnlpqNativePackingFixture".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "native-packing-fixture-v1".to_owned(),
        source_root_sha256: format!("{seed:064x}"),
        logical_model_sha256: hex(&logical_model_sha256),
        sections: vec![
            SectionPayload::new(
                "generic-payload",
                SectionKind::GenericTensorPayload,
                payload,
                64,
            ),
            SectionPayload::new(
                "generic-scales",
                SectionKind::GenericTensorScales,
                scales,
                8,
            ),
            SectionPayload::new(
                "generic-row-sums",
                SectionKind::GenericTensorRowSums,
                row_sums,
                8,
            ),
            SectionPayload::new(
                "tokenizer-model",
                SectionKind::TokenizerModel,
                vec![0x50, 0x4b, 0x03, 0x04],
                32,
            ),
            SectionPayload::new(
                "model-config",
                SectionKind::ModelConfig,
                b"{\"hidden_size\":2}".to_vec(),
                8,
            ),
            SectionPayload::new(
                "tokenizer-config",
                SectionKind::TokenizerConfig,
                b"{\"bos_token\":\"<s>\"}".to_vec(),
                8,
            ),
            SectionPayload::new(
                "chat-template",
                SectionKind::ChatTemplate,
                b"{% set x = 1 %}".to_vec(),
                8,
            ),
            SectionPayload::new(
                "license-bundle",
                SectionKind::LicenseBundle,
                b"Apache-2.0\nModel origin: Nanbeige/Nanbeige4.2-3B\n".to_vec(),
                16,
            ),
        ],
        tensors: vec![
            TensorInput {
                name: "model.embed_tokens.weight".to_owned(),
                canonical_dtype: CanonicalDtype::Bf16,
                shape: vec![2],
                canonical_logical_sha256: hex(&embed_tensor_sha256),
                quantization: "bf16-verbatim-v1".to_owned(),
                data: SectionRange::new("generic-payload", 0, 4),
                scale: SectionRange::new("generic-scales", 0, 4),
                row_sum: SectionRange::new("generic-row-sums", 0, 4),
            },
            TensorInput {
                name: "model.layers.0.mlp.down_proj.weight".to_owned(),
                canonical_dtype: CanonicalDtype::Bf16,
                shape: vec![2],
                canonical_logical_sha256: hex(&layer_tensor_sha256),
                quantization: "bf16-verbatim-v1".to_owned(),
                data: SectionRange::new("generic-payload", 4, 4),
                scale: SectionRange::new("generic-scales", 4, 4),
                row_sum: SectionRange::new("generic-row-sums", 4, 4),
            },
        ],
        packing_sets: vec![PackingSetInput {
            id: "generic".to_owned(),
            target: ArchTarget::Generic,
            section_names: vec![
                "generic-payload".to_owned(),
                "generic-scales".to_owned(),
                "generic-row-sums".to_owned(),
            ],
        }],
    })
    .expect("synthetic Generic root writes")
    .bytes
}

fn pseudo_random_bytes(mut state: u64, len: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        bytes.push((state >> 56) as u8);
    }
    bytes
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut rendered = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        rendered.push(char::from(HEX[usize::from(byte >> 4)]));
        rendered.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    rendered
}
