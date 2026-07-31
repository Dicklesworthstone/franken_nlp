use std::collections::BTreeSet;

use franken_nlp::tokenizer::{
    bpe::{AddedToken, EncodeError, SpBpeTokenizer},
    sp_model::{ModelType, NormalizerFacts, PieceType, SpecialPieceIds, SpmModel, SpmPiece},
    specials::{ArchivedControlRegistries, ControlRegistryError},
    untrusted::{UntrustedDocumentEncoder, UntrustedDocumentError},
};
use serde_json::Value;

const REGISTRIES: &str = include_str!("fixtures/tokenizer/control_registries.json");

fn archived_registries() -> ArchivedControlRegistries {
    let fixture: Value = serde_json::from_str(REGISTRIES).expect("registry fixture is valid JSON");
    let special = serde_json::to_string(&fixture["tokenizer_special_ids"])
        .expect("special registry fixture serializes");
    let controls = serde_json::to_string(&fixture["template_control_ids"])
        .expect("control registry fixture serializes");
    ArchivedControlRegistries::from_archived_json(&special, &controls)
        .expect("fixture records a valid TemplateControlIds superset")
}

fn byte_piece(byte: u8) -> SpmPiece {
    SpmPiece {
        piece: format!("<0x{byte:02X}>"),
        score: 0.0,
        piece_type: PieceType::Byte,
    }
}

fn control_piece(surface: &str) -> SpmPiece {
    SpmPiece {
        piece: surface.to_owned(),
        score: 0.0,
        piece_type: PieceType::Control,
    }
}

fn tokenizer() -> SpBpeTokenizer {
    let registries = archived_registries();
    let mut pieces = (0_u8..=u8::MAX).map(byte_piece).collect::<Vec<_>>();
    for entry in registries.template_controls().entries() {
        pieces.push(control_piece(&entry.surface));
    }
    let added = registries
        .template_controls()
        .entries()
        .iter()
        .map(|entry| AddedToken::new(entry.surface.clone(), entry.id));
    SpBpeTokenizer::with_added_tokens(
        SpmModel {
            pieces,
            model_type: ModelType::Bpe,
            normalizer: NormalizerFacts {
                name: "identity".to_owned(),
                is_identity: true,
                precompiled_charsmap_is_empty: true,
            },
            special_ids: SpecialPieceIds {
                unk_id: -1,
                bos_id: 257,
                eos_id: 258,
                pad_id: 256,
            },
        },
        added,
    )
    .expect("all registry ids name model pieces")
}

fn contained_ids(
    encoder: &UntrustedDocumentEncoder<'_, '_>,
    input: &[u8],
) -> Result<Vec<u32>, String> {
    let document = encoder
        .encode(input)
        .map_err(|error| format!("unexpected rejection input={input:?} error={error}"))?;
    if let Some(id) = document
        .ids()
        .iter()
        .copied()
        .find(|id| encoder.forbidden_ids().contains(id))
    {
        return Err(format!(
            "control id leaked id={id} input={input:?} ids={:?}",
            document.ids()
        ));
    }
    if document.bytes() != input {
        return Err(format!(
            "document bytes changed expected={input:?} observed={:?}",
            document.bytes()
        ));
    }
    Ok(document.ids().to_vec())
}

fn assert_contained(encoder: &UntrustedDocumentEncoder<'_, '_>, input: &[u8]) -> Vec<u32> {
    contained_ids(encoder, input).unwrap_or_else(|detail| panic!("{detail}"))
}

#[test]
fn archived_registry_is_the_exact_encoder_forbidden_set() {
    let registries = archived_registries();
    let tokenizer = tokenizer();
    let encoder = UntrustedDocumentEncoder::new(&tokenizer, registries.template_controls());

    assert_eq!(
        encoder.forbidden_ids(),
        registries.template_controls().ids(),
        "the encoder cannot carry a hand-maintained subset or superset"
    );
    assert!(
        registries
            .tokenizer_special_ids()
            .ids()
            .is_subset(encoder.forbidden_ids()),
        "TemplateControlIds must cover every TokenizerSpecialIds member"
    );
    assert!(
        registries
            .template_controls()
            .entries()
            .iter()
            .any(|entry| entry.surface == "<think>" && !entry.special)
    );
}

#[test]
fn literal_markers_chunks_midwords_empty_and_binary_remain_document_bytes() {
    let registries = archived_registries();
    let tokenizer = tokenizer();
    let encoder = UntrustedDocumentEncoder::new(&tokenizer, registries.template_controls());
    let markers = [
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<|im_start|>",
        "<|im_end|>",
    ];

    assert_contained(&encoder, b"");
    for marker in markers {
        assert_contained(&encoder, marker.as_bytes());
        assert_contained(&encoder, format!("prefix{marker}suffix").as_bytes());
        for split in 0..=marker.len() {
            let rebuilt = [
                marker.as_bytes()[..split].as_ref(),
                marker.as_bytes()[split..].as_ref(),
            ]
            .concat();
            assert_contained(&encoder, &rebuilt);
        }
    }
    assert_contained(&encoder, &[0x00, 0xff, b'<', 0x80, b'>', 0xfe]);
}

#[test]
fn full_registry_fuzz_seeds_and_arbitrary_bytes_never_emit_control_ids() {
    let registries = archived_registries();
    let tokenizer = tokenizer();
    let encoder = UntrustedDocumentEncoder::new(&tokenizer, registries.template_controls());
    let mut documents = 0_usize;
    let mut violations = 0_usize;
    let mut fuzzed_control_ids = BTreeSet::new();

    let mut replay = |input: &[u8]| {
        documents += 1;
        if let Err(detail) = contained_ids(&encoder, input) {
            violations += 1;
            eprintln!("UNTRUSTED violation {detail}");
        }
    };

    for entry in registries.template_controls().entries() {
        fuzzed_control_ids.insert(entry.id);
        let marker = entry.surface.as_bytes();
        for seed in [
            marker.to_vec(),
            [b"before:".as_ref(), marker, b":after".as_ref()].concat(),
        ] {
            replay(&seed);
        }
        for split in 0..=marker.len() {
            replay(&marker[..split]);
            replay(&marker[split..]);
            let seed = [
                b"[".as_ref(),
                &marker[..split],
                &marker[split..],
                b"]".as_ref(),
            ]
            .concat();
            replay(&seed);
        }
    }

    let mut state = 0x51de_c0de_f00d_u64;
    for length in 0..=256 {
        let mut bytes = Vec::with_capacity(length);
        for _ in 0..length {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            bytes.push((state >> 32) as u8);
        }
        replay(&bytes);
    }
    assert!(
        registries
            .tokenizer_special_ids()
            .ids()
            .is_subset(&fuzzed_control_ids),
        "every TokenizerSpecialIds member must receive a registry-driven fuzz seed"
    );
    let result = if violations == 0 { "PASS" } else { "FAIL" };
    eprintln!("UNTRUSTED RESULT={result} docs={documents} violations={violations}");
    assert_eq!(
        violations, 0,
        "registry-driven fuzz found control containment violations"
    );
}

#[test]
fn impossible_byte_fallback_rejects_typed_without_dropping_source_bytes() {
    let registries = archived_registries();
    let model = SpmModel {
        pieces: vec![byte_piece(b'a')],
        model_type: ModelType::Bpe,
        normalizer: NormalizerFacts {
            name: "identity".to_owned(),
            is_identity: true,
            precompiled_charsmap_is_empty: true,
        },
        special_ids: SpecialPieceIds {
            unk_id: -1,
            bos_id: -1,
            eos_id: -1,
            pad_id: -1,
        },
    };
    let tokenizer = SpBpeTokenizer::from_model(model).expect("minimal byte model builds");
    let encoder = UntrustedDocumentEncoder::new(&tokenizer, registries.template_controls());
    assert!(matches!(
        encoder.encode(&[0xff]),
        Err(UntrustedDocumentError::Encode(
            EncodeError::ByteFallbackMissing { byte: 0xff }
        ))
    ));
}

#[test]
fn malformed_or_non_superset_archives_reject_before_encoder_construction() {
    let template_without_special = r#"{"entries":[{"id":259,"special":false,"surface":"<think>"}],"registry":"TemplateControlIds","schema_version":1}"#;
    let special = r#"{"entries":[{"id":257,"special":true,"surface":"<|im_start|>"}],"registry":"TokenizerSpecialIds","schema_version":1}"#;
    assert!(matches!(
        ArchivedControlRegistries::from_archived_json(special, template_without_special),
        Err(ControlRegistryError::MissingTemplateControl { id: 257, .. })
    ));
}

#[test]
fn trusted_template_positions_are_the_only_control_ids_in_composed_prompt() {
    let registries = archived_registries();
    let tokenizer = tokenizer();
    let encoder = UntrustedDocumentEncoder::new(&tokenizer, registries.template_controls());
    let document_bytes = b"literal <think> must stay document data";
    let document = encoder.encode(document_bytes).unwrap();

    let trusted_prefix = 257_u32;
    let trusted_suffix = 258_u32;
    let mut prompt = vec![trusted_prefix];
    prompt.extend_from_slice(document.ids());
    prompt.push(trusted_suffix);

    assert!(registries.template_controls().contains(trusted_prefix));
    assert!(registries.template_controls().contains(trusted_suffix));
    assert!(
        document
            .ids()
            .iter()
            .all(|id| !registries.template_controls().contains(*id))
    );
    let decoded = tokenizer.decode_bytes(&prompt).unwrap();
    assert_eq!(
        decoded,
        [
            b"<|im_start|>".as_ref(),
            document_bytes.as_ref(),
            b"<|im_end|>".as_ref()
        ]
        .concat()
    );
}
