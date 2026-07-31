use std::io::Write;

use franken_nlp::artifact::format::{
    ArchTarget, CanonicalDtype, FnlpqWriteError, FnlpqWriterInput, PackingSetInput, SectionKind,
    SectionPayload, SectionRange, decode_prelude, encode_f32_scales, framed_sha256,
    framed_sha256_hex, logical_model_sha256 as compute_logical_model_sha256, logical_tensor_sha256,
    streaming_input_from_materialized, write, write_streaming,
};
use franken_nlp::artifact::format::LogicalTensorStreamingHasher;
use franken_nlp::artifact::format::StreamingSectionHasher;
use franken_nlp::artifact::reader::FnlpqArtifact;
use franken_nlp::canonjson;
use serde::Serialize;
use sha2::{Digest, Sha256};

const GOLDEN_BYTES_HEX: &str = include_str!("fixtures/fnlpq/golden/tiny_v1.hex");
const GOLDEN_HEADER: &str = include_str!("fixtures/fnlpq/golden/tiny_v1.header.json");
const FIELD_INVENTORY: &str = include_str!("fixtures/fnlpq/field_inventory.json");

fn tiny_input() -> FnlpqWriterInput {
    let scales = encode_f32_scales(&[0.5]).expect("finite synthetic scale");
    let payload = vec![0x80, 0x3f, 0x00, 0x40];
    let row_sums = 0_i32.to_le_bytes().to_vec();
    let tensor_sha256 = logical_tensor_sha256(
        "model.embed_tokens.weight",
        "bf16",
        &[2],
        "bf16-verbatim-v1",
        &payload,
        &scales,
        &row_sums,
    )
    .expect("valid tiny logical tensor identity");
    let logical_model_sha256 = compute_logical_model_sha256(
        &[tensor_sha256],
        &[
            ("model_config", br#"{"hidden_size":2}"#.as_slice()),
            ("tokenizer_model", &[0x50, 0x4b, 0x03, 0x04]),
            ("tokenizer_config", br#"{"bos_token":"<s>"}"#.as_slice()),
            ("chat_template", b"{% set x = 1 %}"),
        ],
    )
    .expect("valid tiny logical-model identity");
    FnlpqWriterInput {
        model_id: "FnlpqTinyGolden".to_owned(),
        revision: "f56ec5a9650268aa098496734743c25ea778bd2d".to_owned(),
        recipe_id: "tiny-golden-v1".to_owned(),
        source_root_sha256: framed_sha256_hex("fnlpq-source-root-v1", &[b"tiny source manifest"])
            .expect("valid source identity"),
        logical_model_sha256: hex(&logical_model_sha256),
        // Deliberately unordered: the writer establishes the canonical v1
        // directory sequence by kind and native section name.
        sections: vec![
            SectionPayload::new(
                "license-bundle",
                SectionKind::LicenseBundle,
                b"Apache-2.0\nModel origin: Nanbeige/Nanbeige4.2-3B\n".to_vec(),
                16,
            ),
            SectionPayload::new(
                "tokenizer-model",
                SectionKind::TokenizerModel,
                vec![0x50, 0x4b, 0x03, 0x04],
                32,
            ),
            SectionPayload::new(
                "native-avx2",
                SectionKind::NativePackingPayload,
                vec![0xa5, 0x5a, 0x01],
                128,
            ),
            SectionPayload::new(
                "generic-row-sums",
                SectionKind::GenericTensorRowSums,
                row_sums.clone(),
                8,
            ),
            SectionPayload::new(
                "model-config",
                SectionKind::ModelConfig,
                b"{\"hidden_size\":2}".to_vec(),
                8,
            ),
            SectionPayload::new(
                "generic-payload",
                SectionKind::GenericTensorPayload,
                payload.clone(),
                64,
            ),
            SectionPayload::new(
                "tokenizer-config",
                SectionKind::TokenizerConfig,
                b"{\"bos_token\":\"<s>\"}".to_vec(),
                8,
            ),
            SectionPayload::new(
                "generic-scales",
                SectionKind::GenericTensorScales,
                scales,
                8,
            ),
            SectionPayload::new(
                "chat-template",
                SectionKind::ChatTemplate,
                b"{% set x = 1 %}".to_vec(),
                8,
            ),
        ],
        tensors: vec![franken_nlp::artifact::format::TensorInput {
            name: "model.embed_tokens.weight".to_owned(),
            canonical_dtype: CanonicalDtype::Bf16,
            shape: vec![2],
            canonical_logical_sha256: hex(&tensor_sha256),
            quantization: "bf16-verbatim-v1".to_owned(),
            data: SectionRange::new("generic-payload", 0, 4),
            scale: SectionRange::new("generic-scales", 0, 4),
            row_sum: SectionRange::new("generic-row-sums", 0, 4),
        }],
        packing_sets: vec![
            PackingSetInput {
                id: "avx2-native".to_owned(),
                target: ArchTarget::X86Avx2,
                section_names: vec!["native-avx2".to_owned()],
            },
            PackingSetInput {
                id: "generic".to_owned(),
                target: ArchTarget::Generic,
                section_names: vec![
                    "generic-row-sums".to_owned(),
                    "generic-payload".to_owned(),
                    "generic-scales".to_owned(),
                ],
            },
        ],
    }
}

#[test]
fn incremental_logical_tensor_hasher_matches_the_framed_v1_identity() {
    let data = [0x80, 0x3f, 0x00, 0x40];
    let scales = [0x00, 0x00, 0x00, 0x3f];
    let row_sums = [0_i32.to_le_bytes(), 1_i32.to_le_bytes()].concat();
    let expected = framed_sha256(
        "fnlpq-logical-tensor-v1",
        &[
            b"model.layers.0.self_attn.q_proj.weight",
            b"i8",
            &[1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
            b"portable-quant-v1",
            &data,
            &scales,
            &row_sums,
        ],
    )
    .expect("bounded framed identity");

    let mut incremental = LogicalTensorStreamingHasher::new(
        "model.layers.0.self_attn.q_proj.weight",
        "i8",
        &[2],
        "portable-quant-v1",
        data.len() as u64,
        scales.len() as u64,
        row_sums.len() as u64,
    )
    .expect("bounded logical tensor declaration");
    incremental
        .write_data(&data[..1])
        .expect("first payload panel");
    incremental
        .write_data(&data[1..])
        .expect("second payload panel");
    incremental
        .write_scale(&scales)
        .expect("complete scale sidecar");
    incremental
        .write_row_sum(&row_sums)
        .expect("complete row-sum sidecar");

    assert_eq!(incremental.finish().expect("complete first pass"), expected);
}

#[test]
fn incremental_logical_tensor_hasher_rejects_sidecars_before_payload_completion() {
    let mut incremental = LogicalTensorStreamingHasher::new(
        "model.embed_tokens.weight",
        "bf16",
        &[2],
        "bf16-verbatim-v1",
        4,
        0,
        0,
    )
    .expect("bounded logical tensor declaration");

    assert!(matches!(
        incremental.write_scale(&[]),
        Err(FnlpqWriteError::LogicalTensorStream { field: "data", .. })
    ));
}

#[test]
fn incremental_section_hasher_matches_the_framed_v1_identity() {
    let bytes = b"a bounded generic payload";
    let expected = framed_sha256("fnlpq-section-v1", &[b"generic-payload", bytes])
        .expect("bounded framed section identity");
    let mut incremental = StreamingSectionHasher::new("generic-payload", bytes.len() as u64)
        .expect("bounded section declaration");
    incremental
        .write(&bytes[..7])
        .expect("first payload chunk");
    incremental
        .write(&bytes[7..])
        .expect("second payload chunk");

    assert_eq!(incremental.finish().expect("complete first pass"), expected);
}

#[test]
fn incremental_section_hasher_rejects_overflow_before_digesting() {
    let mut incremental = StreamingSectionHasher::new("generic-payload", 2)
        .expect("bounded section declaration");

    assert!(matches!(
        incremental.write(b"abc"),
        Err(FnlpqWriteError::StoredIdentity { .. })
    ));
}

#[test]
fn tiny_v1_matches_committed_byte_and_header_goldens() {
    let written = write(&tiny_input()).expect("tiny synthetic artifact writes");
    let expected = decode_hex(GOLDEN_BYTES_HEX);
    assert_eq!(written.bytes, expected, "canonical writer bytes changed");
    assert_eq!(
        std::str::from_utf8(&written.header_bytes).expect("canonical header UTF-8"),
        GOLDEN_HEADER.trim_end(),
        "canonical header changed"
    );

    let prelude = decode_prelude(&written.bytes).expect("complete prelude");
    assert_eq!(prelude.magic, *b"FNLPQ\0\0\x01");
    assert_eq!(prelude.format_version, 1);
    assert_eq!(prelude.required_flags, 0);
    assert_eq!(prelude.header_len as usize, written.header_bytes.len());
    assert_eq!(prelude.section_count as usize, written.sections.len());
    assert_eq!(prelude.tensor_count, 1);
    assert_eq!(prelude.file_len as usize, written.bytes.len());
    let expected_header_sha256: [u8; 32] = Sha256::digest(&written.header_bytes).into();
    assert_eq!(
        prelude.header_sha256, expected_header_sha256,
        "prelude header hash must cover exactly header_len bytes"
    );

    assert_eq!(
        canonjson::canonicalize_str(
            std::str::from_utf8(&written.header_bytes).expect("header is UTF-8"),
            canonjson::ParseLimits::default(),
        )
        .expect("header reparses with duplicate-key rejection"),
        written.header_bytes
    );

    let directory_end = 80 + written.header_bytes.len() + written.sections.len() * 80;
    let mut expected_gap_start = directory_end;
    for section in &written.sections {
        let offset = section.file_offset as usize;
        assert!(
            written.bytes[expected_gap_start..offset]
                .iter()
                .all(|byte| *byte == 0),
            "only zero alignment padding is permitted before {}",
            section.name
        );
        assert_eq!(offset as u64 % section.alignment, 0);
        let stored = &written.bytes[offset..offset + section.stored_len as usize];
        assert_eq!(
            framed_sha256("fnlpq-section-v1", &[section.name.as_bytes(), stored])
                .expect("valid digest frame"),
            section.stored_sha256,
            "section {} digest mismatch",
            section.name
        );
        expected_gap_start = offset + section.stored_len as usize;
    }
    assert_eq!(
        expected_gap_start,
        written.bytes.len(),
        "no trailing padding"
    );
    let artifact = FnlpqArtifact::from_bytes(written.bytes.clone())
        .expect("canonical writer bytes load through the checked reader");
    assert_eq!(
        artifact
            .reserialize()
            .expect("checked artifact reserializes"),
        written.bytes,
        "load then re-serialize must preserve canonical bytes"
    );
}

#[test]
fn streaming_writer_matches_the_materialized_v1_oracle_byte_for_byte() {
    let input = tiny_input();
    let materialized = write(&input).expect("tiny in-memory oracle writes");
    let streaming = streaming_input_from_materialized(&input)
        .expect("tiny oracle input supplies verified streaming metadata");
    let mut streamed_bytes = Vec::new();
    let streamed = write_streaming(&streaming, &mut streamed_bytes, |section, sink| {
        let source = input
            .sections
            .iter()
            .find(|candidate| candidate.name == section.name)
            .expect("streaming plan section originates from tiny input");
        sink.write_all(&source.bytes)
            .map_err(|error| FnlpqWriteError::Io {
                operation: "write synthetic streaming section",
                detail: error.to_string(),
            })
    })
    .expect("all tiny planned sections stream with their first-pass digests");

    assert_eq!(streamed_bytes, materialized.bytes);
    assert_eq!(streamed.header_bytes, materialized.header_bytes);
    assert_eq!(streamed.header_sha256, materialized.header_sha256);
    assert_eq!(streamed.sections, materialized.sections);
    assert_eq!(streamed.file_len as usize, materialized.bytes.len());
    assert_eq!(streamed.fnlpq_file_sha256, materialized.fnlpq_file_sha256);
    assert_eq!(streamed.packing_set_sha256, materialized.packing_set_sha256);
    assert_eq!(streamed.license_bundle_sha256, materialized.license_bundle_sha256);
}

#[test]
fn streaming_writer_rejects_a_short_second_pass_section() {
    let input = tiny_input();
    let streaming = streaming_input_from_materialized(&input)
        .expect("tiny oracle input supplies verified streaming metadata");
    let expected_section = streaming
        .sections
        .iter()
        .find(|section| section.kind == SectionKind::GenericTensorPayload)
        .expect("tiny fixture has a generic tensor payload")
        .name
        .clone();
    let mut streamed_bytes = Vec::new();
    let error = write_streaming(&streaming, &mut streamed_bytes, |section, sink| {
        let source = input
            .sections
            .iter()
            .find(|candidate| candidate.name == section.name)
            .expect("streaming plan section originates from tiny input");
        let bytes = if section.name == expected_section {
            let short_len = source.bytes.len().checked_sub(1).ok_or_else(|| {
                FnlpqWriteError::Missing {
                    field: "synthetic generic tensor payload byte",
                    value: section.name.clone(),
                }
            })?;
            source.bytes.get(..short_len).ok_or_else(|| FnlpqWriteError::Missing {
                field: "synthetic short streaming section range",
                value: section.name.clone(),
            })?
        } else {
            &source.bytes
        };
        sink.write_all(bytes).map_err(|error| FnlpqWriteError::Io {
            operation: "write short synthetic streaming section",
            detail: error.to_string(),
        })
    })
    .expect_err("a short second pass must not mint an artifact identity");

    assert!(matches!(
        error,
        FnlpqWriteError::StoredIdentity {
            section,
            actual,
            ..
        } if section == expected_section && actual == "underflow-1-bytes"
    ));
}

#[test]
fn streaming_writer_rejects_a_tampered_second_pass_section() {
    let input = tiny_input();
    let streaming = streaming_input_from_materialized(&input)
        .expect("tiny oracle input supplies verified streaming metadata");
    let expected_section = streaming
        .sections
        .iter()
        .find(|section| section.kind == SectionKind::GenericTensorPayload)
        .expect("tiny fixture has a generic tensor payload")
        .name
        .clone();
    let mut streamed_bytes = Vec::new();
    let error = write_streaming(&streaming, &mut streamed_bytes, |section, sink| {
        let source = input
            .sections
            .iter()
            .find(|candidate| candidate.name == section.name)
            .expect("streaming plan section originates from tiny input");
        (if section.name == expected_section {
            let mut tampered = source.bytes.clone();
            let first = tampered.first_mut().ok_or_else(|| FnlpqWriteError::Missing {
                field: "synthetic generic tensor payload byte",
                value: section.name.clone(),
            })?;
            *first ^= 1;
            sink.write_all(&tampered)
        } else {
            sink.write_all(&source.bytes)
        })
        .map_err(|error| FnlpqWriteError::Io {
            operation: "write tampered synthetic streaming section",
            detail: error.to_string(),
        })
    })
    .expect_err("a tampered second pass must not mint an artifact identity");

    assert!(matches!(
        error,
        FnlpqWriteError::StoredIdentity { section, .. } if section == expected_section
    ));
}

#[test]
fn streaming_writer_binds_the_license_claim_to_emitted_bytes() {
    let input = tiny_input();
    let mut streaming = streaming_input_from_materialized(&input)
        .expect("tiny oracle input supplies verified streaming metadata");
    let expected_section = streaming
        .sections
        .iter()
        .find(|section| section.kind == SectionKind::LicenseBundle)
        .expect("tiny fixture has a license bundle")
        .name
        .clone();
    streaming.license_bundle_sha256 = [0xff; 32];
    let mut streamed_bytes = Vec::new();
    let error = write_streaming(&streaming, &mut streamed_bytes, |section, sink| {
        let source = input
            .sections
            .iter()
            .find(|candidate| candidate.name == section.name)
            .expect("streaming plan section originates from tiny input");
        sink.write_all(&source.bytes)
            .map_err(|error| FnlpqWriteError::Io {
                operation: "write synthetic streaming section",
                detail: error.to_string(),
            })
    })
    .expect_err("the license header claim must match emitted license bytes");

    assert!(matches!(
        error,
        FnlpqWriteError::StoredIdentity { section, .. } if section == expected_section
    ));
}

#[test]
fn field_inventory_ids_are_all_covered_by_the_writer_golden() {
    const IDS: &[&str] = &[
        "prelude.magic",
        "prelude.format_version",
        "prelude.required_flags",
        "prelude.header_len",
        "prelude.section_count",
        "prelude.tensor_count",
        "prelude.file_len",
        "prelude.header_sha256",
        "directory.kind",
        "directory.flags",
        "directory.name_index",
        "directory.file_offset",
        "directory.stored_len",
        "directory.logical_len",
        "directory.alignment",
        "directory.stored_sha256",
        "header.header_schema",
        "header.model.model_id",
        "header.model.revision",
        "header.recipe_id",
        "header.source_root_sha256",
        "header.logical_model_sha256",
        "header.packing_set_sha256",
        "header.license_bundle_sha256",
        "header.limits_profile",
        "header.materialized_sources",
        "header.materialized_sources.entry.section_ordinal",
        "header.materialized_sources.entry.sha256",
        "header.sections",
        "header.sections.ordinal",
        "header.sections.name",
        "header.sections.kind",
        "header.sections.required",
        "header.tensors",
        "header.tensors.name",
        "header.tensors.canonical_dtype",
        "header.tensors.shape",
        "header.tensors.canonical_logical_sha256",
        "header.tensors.logical_bytes",
        "header.tensors.generic.quantization",
        "header.tensors.generic.mapping",
        "header.packing_sets",
        "header.packing_sets.id",
        "header.packing_sets.target",
        "header.packing_sets.representations",
    ];
    for id in IDS {
        assert!(
            FIELD_INVENTORY.contains(id),
            "stable OQ-31 inventory ID {id} disappeared"
        );
    }
    let written = write(&tiny_input()).expect("golden writer input succeeds");
    assert_eq!(
        written.sections.len(),
        9,
        "all v1 section kinds are represented"
    );
    assert!(
        written
            .sections
            .iter()
            .any(|section| section.kind == SectionKind::NativePackingPayload)
    );
}

#[test]
fn canonical_input_order_and_digest_taxonomy_are_stable() {
    let input = tiny_input();
    let first = write(&input).expect("first canonical write");
    let second = write(&input).expect("second canonical write");
    assert_eq!(
        first.bytes, second.bytes,
        "identical input must be byte stable"
    );

    let mut repacked = input.clone();
    repacked
        .sections
        .iter_mut()
        .find(|section| section.kind == SectionKind::NativePackingPayload)
        .expect("tiny input has native packing")
        .bytes
        .push(0x02);
    let repacked = write(&repacked).expect("native repack writes");
    assert_eq!(
        input.logical_model_sha256,
        tiny_input().logical_model_sha256
    );
    assert_ne!(first.packing_set_sha256, repacked.packing_set_sha256);
    assert_ne!(first.fnlpq_file_sha256, repacked.fnlpq_file_sha256);
    assert_eq!(first.license_bundle_sha256, repacked.license_bundle_sha256);

    let mut notice_only = input;
    notice_only
        .sections
        .iter_mut()
        .find(|section| section.kind == SectionKind::LicenseBundle)
        .expect("tiny input has license bundle")
        .bytes
        .extend_from_slice(b"Modified by franken_nlp.\n");
    let notice_only = write(&notice_only).expect("notice correction writes");
    assert_eq!(first.packing_set_sha256, notice_only.packing_set_sha256);
    assert_ne!(
        first.license_bundle_sha256,
        notice_only.license_bundle_sha256
    );
    assert_ne!(first.fnlpq_file_sha256, notice_only.fnlpq_file_sha256);
}

#[test]
fn invalid_names_duplicate_mappings_and_nonfinite_scales_reject() {
    assert_eq!(
        encode_f32_scales(&[f32::NAN]),
        Err(FnlpqWriteError::NonFiniteScale { index: 0 })
    );
    assert_eq!(
        encode_f32_scales(&[f32::INFINITY]),
        Err(FnlpqWriteError::NonFiniteScale { index: 0 })
    );

    let mut duplicate = tiny_input();
    duplicate.sections.push(SectionPayload::new(
        "generic-payload",
        SectionKind::NativePackingPayload,
        vec![0],
        1,
    ));
    assert!(matches!(
        write(&duplicate),
        Err(FnlpqWriteError::Duplicate {
            field: "section name",
            ..
        })
    ));

    let mut malformed_name = tiny_input();
    malformed_name.tensors[0].name = "model/illegal".to_owned();
    assert!(matches!(
        write(&malformed_name),
        Err(FnlpqWriteError::InvalidAuthority {
            field: "tensor.name",
            ..
        })
    ));

    let mut overlap = tiny_input();
    let mut alias = overlap.tensors[0].clone();
    alias.name = "model.norm.weight".to_owned();
    overlap.tensors.push(alias);
    assert!(matches!(
        write(&overlap),
        Err(FnlpqWriteError::Tensor { .. })
    ));
}

#[derive(Serialize)]
struct EscapeProbe<'a> {
    z: u64,
    escape: &'a str,
    a: u64,
}

#[test]
fn canonical_escape_and_key_order_are_pinned_without_permitting_escaped_authorities() {
    let emitted = canonjson::canonical_string(&EscapeProbe {
        z: 2,
        escape: "quote\"slash\\",
        a: 1,
    })
    .expect("finite typed JSON");
    assert_eq!(emitted, r#"{"a":1,"escape":"quote\"slash\\","z":2}"#);
}

fn decode_hex(input: &str) -> Vec<u8> {
    let digits: Vec<_> = input
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect();
    assert_eq!(digits.len() % 2, 0, "golden hex needs complete bytes");
    digits
        .chunks_exact(2)
        .map(|pair| (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]))
        .collect()
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => panic!("invalid golden hex digit {byte:?}"),
    }
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
