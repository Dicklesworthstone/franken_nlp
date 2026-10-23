use std::collections::BTreeSet;
use std::path::PathBuf;

use franken_nlp::artifact::format::{
    ArchTarget, CanonicalDtype, FnlpqWriterInput, PackingSetInput, SectionKind, SectionPayload,
    SectionRange, TensorInput, framed_sha256, logical_model_sha256, logical_tensor_sha256, write,
};
use franken_nlp::artifact::reader::{FnlpqArtifact, FnlpqReadError};

const HOSTILE_CASES: &str = include_str!("fixtures/fnlpq/hostile_cases.json");

const CORPUS_IDS: &[&str] = &[
    "FNLPG-H001-TRUNCATED-PRELUDE",
    "FNLPG-H002-HEADER-LEN-OVERFLOW",
    "FNLPG-H003-HEADER-OVER-CAP",
    "FNLPG-H004-WRONG-FILE-LEN",
    "FNLPG-H005-UNKNOWN-REQUIRED-FLAG",
    "FNLPG-H006-HEADER-SHA-MISMATCH",
    "FNLPG-H007-DUPLICATE-JSON-KEY",
    "FNLPG-H008-UNICODE-ESCAPE",
    "FNLPG-H009-SECTION-TABLE-OVERLAP",
    "FNLPG-H010-DUPLICATE-REQUIRED-SECTION-KIND",
    "FNLPG-H011-UNKNOWN-SECTION-KIND",
    "FNLPG-H012-NONZERO-ALIGNMENT-GAP",
    "FNLPG-H013-RANGE-INTO-HEADER",
    "FNLPG-H014-DUPLICATE-TENSOR-NAME",
    "FNLPG-H015-NAN-SCALE",
    "FNLPG-H016-PACKING-ARCH-MISMATCH",
    "FNLPG-H017-DUPLICATE-SECTION-NAME",
    "FNLPG-H018-INFINITE-SCALE",
];

#[test]
fn owned_reader_accepts_writer_golden_and_refuses_wrong_target() {
    let artifact = FnlpqArtifact::from_bytes(golden_bytes()).expect("writer golden must parse");
    assert_eq!(artifact.model_id(), "FnlpqGeneratedFuzz");
    assert_eq!(
        artifact.revision(),
        "f56ec5a9650268aa098496734743c25ea778bd2d"
    );
    assert_eq!(artifact.sections().len(), 8);
    assert_eq!(artifact.tensors().len(), 1);
    assert_eq!(
        artifact.select_packing(ArchTarget::Generic).unwrap().id,
        "generic"
    );
    assert!(matches!(
        artifact.select_packing(ArchTarget::Aarch64Sdot),
        Err(FnlpqReadError::MissingPackingDerivation { .. })
    ));
}

#[test]
fn corpus_index_has_a_dedicated_fixture_for_every_oq31_hostile_id() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/fnlpq");
    let named: BTreeSet<_> = std::fs::read_dir(&root)
        .expect("committed fnlpq hostile corpus directory")
        .map(|entry| entry.expect("corpus entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .map(|path| std::fs::read_to_string(path).expect("corpus fixture text"))
        .collect();
    assert_eq!(CORPUS_IDS.len(), 18);
    for id in CORPUS_IDS {
        assert!(HOSTILE_CASES.contains(id), "OQ-31 index omitted {id}");
        assert!(
            named.iter().any(|fixture| fixture.contains(id)),
            "missing dedicated hostile corpus fixture {id}"
        );
    }
}

#[test]
fn fixed_preheader_directory_and_scale_mutations_reject_without_panicking() {
    let valid = golden_bytes();
    let directory_start = directory_start(&valid);
    let directory_end = directory_start + u64_at(&valid, 24) as usize * 80;
    let first_section_offset = u64_at(&valid, directory_start + 16) as usize;

    let mut cases = Vec::new();
    cases.push(("FNLPG-H001-TRUNCATED-PRELUDE", valid[..79].to_vec()));

    let mut header_overflow = valid.clone();
    header_overflow[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    cases.push(("FNLPG-H002-HEADER-LEN-OVERFLOW", header_overflow));

    let mut header_cap = valid.clone();
    header_cap[16..24].copy_from_slice(&(1_048_577_u64).to_le_bytes());
    cases.push(("FNLPG-H003-HEADER-OVER-CAP", header_cap));

    let mut wrong_file_len = valid.clone();
    let mismatched_len = wrong_file_len.len() as u64 + 1;
    wrong_file_len[40..48].copy_from_slice(&mismatched_len.to_le_bytes());
    cases.push(("FNLPG-H004-WRONG-FILE-LEN", wrong_file_len));

    let mut unknown_flags = valid.clone();
    unknown_flags[12..16].copy_from_slice(&1_u32.to_le_bytes());
    cases.push(("FNLPG-H005-UNKNOWN-REQUIRED-FLAG", unknown_flags));

    let mut header_flip = valid.clone();
    header_flip[80] ^= 1;
    cases.push(("FNLPG-H006-HEADER-SHA-MISMATCH", header_flip));

    let mut overlap = valid.clone();
    overlap[directory_start + 80 + 16..directory_start + 80 + 24]
        .copy_from_slice(&(first_section_offset as u64).to_le_bytes());
    cases.push(("FNLPG-H009-SECTION-TABLE-OVERLAP", overlap));

    let mut unknown_kind = valid.clone();
    unknown_kind[directory_start..directory_start + 4].copy_from_slice(&99_u32.to_le_bytes());
    cases.push(("FNLPG-H011-UNKNOWN-SECTION-KIND", unknown_kind));

    let mut gap_fraud = valid.clone();
    assert!(
        first_section_offset > directory_end,
        "golden carries an alignment gap"
    );
    gap_fraud[directory_end] = 1;
    cases.push(("FNLPG-H012-NONZERO-ALIGNMENT-GAP", gap_fraud));

    let mut into_header = valid.clone();
    into_header[directory_start + 16..directory_start + 24].copy_from_slice(&80_u64.to_le_bytes());
    cases.push(("FNLPG-H013-RANGE-INTO-HEADER", into_header));

    let mut nan_scale = valid.clone();
    let scale_directory = directory_start + 80;
    let scale_offset = u64_at(&nan_scale, scale_directory + 16) as usize;
    let scale_len = u64_at(&nan_scale, scale_directory + 24) as usize;
    nan_scale[scale_offset..scale_offset + 4].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
    refresh_section_digest(
        &mut nan_scale,
        scale_directory,
        b"generic-scales",
        scale_offset,
        scale_len,
    );
    cases.push(("FNLPG-H015-NAN-SCALE", nan_scale));

    let mut inf_scale = valid.clone();
    inf_scale[scale_offset..scale_offset + 4]
        .copy_from_slice(&f32::INFINITY.to_bits().to_le_bytes());
    refresh_section_digest(
        &mut inf_scale,
        scale_directory,
        b"generic-scales",
        scale_offset,
        scale_len,
    );
    cases.push(("FNLPG-H018-INFINITE-SCALE", inf_scale));

    for (id, mutation) in &cases {
        let error = FnlpqArtifact::from_bytes(mutation.clone()).expect_err(id);
        eprintln!("FNLPG case={id} RESULT=REJECT detail={error}");
    }
    eprintln!(
        "FUZZ_SUMMARY cases={} rejects={} zero_panics=true",
        cases.len(),
        cases.len()
    );
}

#[test]
fn sampled_single_bit_mutations_are_all_typed_rejections() {
    let valid = golden_bytes();
    let mut cases = 0_usize;
    for offset in (0..valid.len()).step_by(97) {
        let mut mutation = valid.clone();
        mutation[offset] ^= 0x01;
        assert!(
            FnlpqArtifact::from_bytes(mutation).is_err(),
            "bit mutation at offset {offset} unexpectedly parsed"
        );
        cases += 1;
    }
    eprintln!("FUZZ_SUMMARY cases={cases} rejects={cases} zero_panics=true");
}

#[test]
fn generated_artifacts_round_trip_and_refuse_each_stored_payload_bit_flip() {
    const SEEDS: [u64; 12] = [
        0x0000_0000_0000_0001,
        0x0000_0000_0000_0002,
        0x0000_0000_0000_0003,
        0x0123_4567_89ab_cdef,
        0x5a5a_5a5a_5a5a_5a5a,
        0xa5a5_a5a5_a5a5_a5a5,
        0xcafe_babe_dead_beef,
        0xd1ce_f00d_1234_5678,
        0xdead_beef_c001_d00d,
        0xfeed_face_2468_1357,
        0xffff_ffff_ffff_fffe,
        0xffff_ffff_ffff_ffff,
    ];

    let mut valid_artifacts = 0_usize;
    let mut rejected_payload_flips = 0_usize;
    for seed in SEEDS {
        let valid = generated_valid_artifact(seed);
        let artifact = FnlpqArtifact::from_bytes(valid.clone())
            .expect("writer-generated artifact must load before mutation");
        assert_eq!(
            artifact
                .reserialize()
                .expect("checked generated artifact reserializes"),
            valid,
            "generated fixture seed={seed:#018x} must retain canonical bytes"
        );
        valid_artifacts += 1;

        let directory = directory_start(&valid);
        let section_count = u64_at(&valid, 24) as usize;
        for ordinal in 0..section_count {
            let entry = directory + ordinal * 80;
            let offset = u64_at(&valid, entry + 16) as usize;
            let len = u64_at(&valid, entry + 24) as usize;
            for payload_offset in offset..offset + len {
                let mut mutation = valid.clone();
                mutation[payload_offset] ^= 0x01;
                assert!(
                    FnlpqArtifact::from_bytes(mutation).is_err(),
                    "stored payload flip accepted seed={seed:#018x} section={ordinal} offset={payload_offset}"
                );
                rejected_payload_flips += 1;
            }
        }
    }
    eprintln!(
        "FUZZ_SUMMARY generated_artifacts={valid_artifacts} stored_payload_flips={rejected_payload_flips} verdict=REFUSED"
    );
}

fn generated_valid_artifact(seed: u64) -> Vec<u8> {
    let element_count = usize::try_from(seed % 31 + 1).expect("bounded generated shape");
    let mut state = seed;
    let mut next_byte = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 56) as u8
    };
    let payload = (0..element_count * 2)
        .map(|_| next_byte())
        .collect::<Vec<_>>();
    let scales = [0.5_f32.to_le_bytes(), 1.0_f32.to_le_bytes()].concat();
    let row_sums = [0_i32.to_le_bytes(), 1_i32.to_le_bytes()].concat();
    let shape = vec![u32::try_from(element_count).expect("bounded generated shape")];
    let tensor_digest = logical_tensor_sha256(
        "generated.weight",
        "bf16",
        &shape,
        "bf16-verbatim-v1",
        &payload,
        &scales,
        &row_sums,
    )
    .expect("generated logical tensor digest");
    let model_config = b"{\"hidden_size\":1}".to_vec();
    let tokenizer_model = vec![0x0a, 0x01, 0x41];
    let tokenizer_config = b"{\"bos_token\":\"<s>\"}".to_vec();
    let chat_template = b"{{ generated }}".to_vec();
    let logical_model = logical_model_sha256(
        "FnlpqGeneratedFuzz",
        "f56ec5a9650268aa098496734743c25ea778bd2d",
        &[tensor_digest],
        &[
            ("model_config", model_config.as_slice()),
            ("tokenizer_model", tokenizer_model.as_slice()),
            ("tokenizer_config", tokenizer_config.as_slice()),
            ("chat_template", chat_template.as_slice()),
        ],
    )
    .expect("generated logical model digest");
    write(&FnlpqWriterInput {
        model_id: "FnlpqGeneratedFuzz".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "generated-fuzz-v1".to_owned(),
        source_root_sha256: hex(&framed_sha256(
            "fnlpq-source-root-v1",
            &[b"generated fuzz source"],
        )
        .expect("generated source root digest")),
        logical_model_sha256: hex(&logical_model),
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
                tokenizer_model,
                8,
            ),
            SectionPayload::new("model-config", SectionKind::ModelConfig, model_config, 8),
            SectionPayload::new(
                "tokenizer-config",
                SectionKind::TokenizerConfig,
                tokenizer_config,
                8,
            ),
            SectionPayload::new("chat-template", SectionKind::ChatTemplate, chat_template, 8),
            SectionPayload::new(
                "license-bundle",
                SectionKind::LicenseBundle,
                b"Apache-2.0\n",
                8,
            ),
        ],
        tensors: vec![TensorInput {
            name: "generated.weight".to_owned(),
            canonical_dtype: CanonicalDtype::Bf16,
            shape,
            canonical_logical_sha256: hex(&tensor_digest),
            quantization: "bf16-verbatim-v1".to_owned(),
            data: SectionRange::new("generic-payload", 0, (element_count * 2) as u64),
            scale: SectionRange::new("generic-scales", 0, 8),
            row_sum: SectionRange::new("generic-row-sums", 0, 8),
        }],
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
    .expect("generated fixture writes")
    .bytes
}

fn golden_bytes() -> Vec<u8> {
    generated_valid_artifact(0x6d2f_5a83_1b94_c0de)
}

fn directory_start(bytes: &[u8]) -> usize {
    80 + u64_at(bytes, 16) as usize
}

fn u64_at(bytes: &[u8], start: usize) -> u64 {
    u64::from_le_bytes(
        bytes[start..start + 8]
            .try_into()
            .expect("fixed fixture range"),
    )
}

fn refresh_section_digest(
    bytes: &mut [u8],
    directory_offset: usize,
    name: &[u8],
    payload_offset: usize,
    payload_len: usize,
) {
    let digest = framed_sha256(
        "fnlpq-section-v1",
        &[name, &bytes[payload_offset..payload_offset + payload_len]],
    )
    .expect("fixed digest tag");
    bytes[directory_offset + 48..directory_offset + 80].copy_from_slice(&digest);
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

/// Exercise the production file-backed open path, using only a tiny fixture
/// owned by this test. No full-model or platform activation claim is made.
fn with_range_artifact<T>(
    bytes: &[u8],
    check: impl FnOnce(Result<franken_nlp::artifact::reader::FnlpqRangeReader, FnlpqReadError>) -> T,
) -> T {
    use std::{fs::OpenOptions, io::Write, sync::atomic::{AtomicU64, Ordering}};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture(PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
    }
    let path = std::env::temp_dir().join(format!(
        "fnlpq-semantic-range-{}-{}.fnlpq", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed),
    ));
    let mut file = OpenOptions::new().write(true).create_new(true).open(&path).unwrap();
    let fixture = Fixture(path); // Cleanup authority follows successful create_new only.
    file.write_all(bytes).unwrap();
    drop(file);
    check(franken_nlp::artifact::reader::FnlpqRangeReader::open(&fixture.0))
}

fn fixture_header(bytes: &[u8]) -> serde_json::Value {
    serde_json::from_slice(&bytes[80..directory_start(bytes)]).unwrap()
}
fn replace_fixture_header(bytes: &mut [u8], header: &serde_json::Value) {
    use sha2::{Digest, Sha256};
    let end = directory_start(bytes);
    let encoded = franken_nlp::canonjson::canonical_bytes(header).unwrap();
    assert_eq!(encoded.len(), end - 80, "digest-only mutations preserve directory offsets");
    bytes[80..end].copy_from_slice(&encoded);
    bytes[48..80].copy_from_slice(&Sha256::digest(&encoded));
}
fn fixture_section(bytes: &[u8], ordinal: usize) -> &[u8] {
    let entry = directory_start(bytes) + ordinal * 80;
    let start = u64_at(bytes, entry + 16) as usize;
    let len = u64_at(bytes, entry + 24) as usize;
    &bytes[start..start + len]
}

/// Update every physical consistency link after a payload mutation. Logical
/// tensor/model/license identities deliberately stay unchanged unless the
/// caller explicitly rebinds them. A stored-hash rejection cannot mask the
/// semantic verification under test.
fn refresh_fixture_physical_identities(bytes: &mut [u8]) -> serde_json::Value {
    let mut header = fixture_header(bytes);
    let directory = directory_start(bytes);
    let sections = header["sections"].as_array().unwrap().clone();
    let mut stored = Vec::new();
    for (ordinal, section) in sections.iter().enumerate() {
        let digest = framed_sha256("fnlpq-section-v1", &[
            section["name"].as_str().unwrap().as_bytes(), fixture_section(bytes, ordinal),
        ]).unwrap();
        bytes[directory + ordinal * 80 + 48..directory + ordinal * 80 + 80].copy_from_slice(&digest);
        stored.push(hex(&digest));
    }
    for source in header["materialized_sources"].as_array_mut().unwrap() {
        source["sha256"] = stored[source["section_ordinal"].as_u64().unwrap() as usize].clone().into();
    }
    for set in header["packing_sets"].as_array_mut().unwrap() {
        for representation in set["representations"].as_array_mut().unwrap() {
            representation["sha256"] = stored[representation["section_ordinal"].as_u64().unwrap() as usize].clone().into();
        }
    }
    let mut sets = Vec::new();
    for set in header["packing_sets"].as_array().unwrap() {
        let mut representations = Vec::new();
        for representation in set["representations"].as_array().unwrap() {
            let ordinal = representation["section_ordinal"].as_u64().unwrap() as usize;
            representations.push(serde_json::json!({
                "byte_cost": representation["byte_cost"],
                "section_name": sections[ordinal]["name"],
                "stored_sha256": representation["sha256"],
            }));
        }
        representations.sort_by(|a, b| a["section_name"].as_str().unwrap().as_bytes()
            .cmp(b["section_name"].as_str().unwrap().as_bytes()));
        sets.push(serde_json::json!({
            "id": set["id"], "representations": representations, "target": set["target"],
        }));
    }
    let canonical = franken_nlp::canonjson::canonical_bytes(&sets).unwrap();
    header["packing_set_sha256"] = hex(&framed_sha256("fnlpq-packing-set-v1", &[
        header["recipe_id"].as_str().unwrap().as_bytes(), &canonical,
    ]).unwrap()).into();
    header
}

#[test]
fn range_open_matches_owned_semantic_identity_and_packing_selection() {
    for seed in [1, 2, 0x6d2f_5a83_1b94_c0de] {
        let bytes = generated_valid_artifact(seed);
        let owned = FnlpqArtifact::from_bytes(bytes.clone()).unwrap();
        with_range_artifact(&bytes, |result| {
            let range = result.unwrap_or_else(|error| panic!("valid range fixture refused: {error}"));
            assert_eq!(range.model_id(), owned.model_id());
            assert_eq!(range.revision(), owned.revision());
            assert_eq!(range.recipe_id(), owned.recipe_id());
            assert_eq!(range.source_root_sha256(), owned.source_root_sha256());
            assert_eq!(range.logical_model_sha256(), owned.logical_model_sha256());
            assert_eq!(range.packing_set_sha256(), owned.packing_set_sha256());
            assert_eq!(range.license_bundle_sha256(), owned.license_bundle_sha256());
            assert_eq!(range.tensors(), owned.tensors());
            assert_eq!(range.sections(), owned.sections());
            assert_eq!(range.select_packing(ArchTarget::Generic).unwrap(), owned.select_packing(ArchTarget::Generic).unwrap());
            assert!(matches!(range.select_packing(ArchTarget::Aarch64Sdot),
                Err(FnlpqReadError::MissingPackingDerivation { .. })));
        });
    }
}

#[test]
fn range_open_rejects_forged_semantic_digests_even_with_valid_header_and_stored_hashes() {
    for field in ["license_bundle_sha256", "packing_set_sha256", "logical_model_sha256", "canonical_logical_sha256"] {
        let mut bytes = golden_bytes();
        let mut header = fixture_header(&bytes);
        if field == "canonical_logical_sha256" { header["tensors"][0][field] = "0".repeat(64).into(); }
        else { header[field] = "0".repeat(64).into(); }
        replace_fixture_header(&mut bytes, &header);
        let owned_error = FnlpqArtifact::from_bytes(bytes.clone()).unwrap_err();
        assert!(owned_error.to_string().contains(field), "{owned_error}");
        with_range_artifact(&bytes, |result| {
            let error = result.err().expect("forged semantic identity must fail file-backed open");
            assert!(error.to_string().contains(field), "{error}");
        });
    }
}

#[test]
fn physical_rehash_cannot_hide_changed_tensor_semantic_source_or_license_bytes() {
    let valid = golden_bytes();
    let original = FnlpqArtifact::from_bytes(valid.clone()).unwrap();
    for (kind, expected) in [
        (SectionKind::GenericTensorPayload, "canonical_logical_sha256"),
        (SectionKind::GenericTensorRowSums, "canonical_logical_sha256"),
        (SectionKind::TokenizerModel, "logical_model_sha256"),
        (SectionKind::ModelConfig, "logical_model_sha256"),
        (SectionKind::TokenizerConfig, "logical_model_sha256"),
        (SectionKind::ChatTemplate, "logical_model_sha256"),
        (SectionKind::LicenseBundle, "license_bundle_sha256"),
    ] {
        let mut bytes = valid.clone();
        let section = original.sections().iter().find(|section| section.kind == kind).unwrap();
        bytes[section.file_offset as usize] ^= 1;
        let header = refresh_fixture_physical_identities(&mut bytes);
        replace_fixture_header(&mut bytes, &header);
        let owned_error = FnlpqArtifact::from_bytes(bytes.clone()).unwrap_err();
        assert!(owned_error.to_string().contains(expected), "{owned_error}");
        with_range_artifact(&bytes, |result| {
            let error = result.err().expect("physical identities cannot replace semantic identities");
            assert!(error.to_string().contains(expected), "{error}");
        });
    }
}

#[test]
fn range_open_rejects_invalid_scales_even_when_all_declared_digests_match_the_bytes() {
    let valid = golden_bytes();
    let original = FnlpqArtifact::from_bytes(valid.clone()).unwrap();
    let scale = original.sections().iter().find(|section| section.kind == SectionKind::GenericTensorScales).unwrap();
    for invalid in [0.0_f32, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bytes = valid.clone();
        let offset = scale.file_offset as usize;
        bytes[offset..offset + 4].copy_from_slice(&invalid.to_le_bytes());
        let mut header = refresh_fixture_physical_identities(&mut bytes);
        let mut tensor_digests = Vec::new();
        for (index, tensor) in original.tensors().iter().enumerate() {
            let mapping = |range: &franken_nlp::artifact::reader::CheckedMapping| {
                let section = fixture_section(&bytes, range.section_ordinal as usize);
                &section[range.offset as usize..(range.offset + range.len) as usize]
            };
            let digest = logical_tensor_sha256(&tensor.name, &tensor.canonical_dtype, &tensor.shape,
                &tensor.quantization, mapping(&tensor.data), mapping(&tensor.scale), mapping(&tensor.row_sum)).unwrap();
            header["tensors"][index]["canonical_logical_sha256"] = hex(&digest).into();
            tensor_digests.push(digest);
        }
        let sources: Vec<_> = header["materialized_sources"].as_array().unwrap().iter().map(|source| (
            source["name"].as_str().unwrap(),
            fixture_section(&bytes, source["section_ordinal"].as_u64().unwrap() as usize),
        )).collect();
        let model_digest = logical_model_sha256(original.model_id(), original.revision(), &tensor_digests, &sources).unwrap();
        header["logical_model_sha256"] = hex(&model_digest).into();
        replace_fixture_header(&mut bytes, &header);
        let owned_error = FnlpqArtifact::from_bytes(bytes.clone()).unwrap_err();
        assert!(owned_error.to_string().contains("scale encoding"), "{owned_error}");
        with_range_artifact(&bytes, |result| {
            let error = result.err().expect("matching digests cannot make an invalid scale executable");
            assert!(error.to_string().contains("scale encoding"), "{error}");
        });
    }
}
