use std::collections::BTreeSet;

use franken_nlp::tokenizer::{
    bpe::{AddedToken, EncodeError, SpBpeTokenizer},
    sp_model::{ModelType, NormalizerFacts, PieceType, SpecialPieceIds, SpmModel, SpmPiece},
    specials::{ArchivedControlRegistries, ControlRegistryError},
    untrusted::{SafeBpeLimits, SafeBpeWork, UntrustedDocumentEncoder, UntrustedDocumentError,
        UntrustedEncoding},
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

fn compact_limits() -> SafeBpeLimits {
    SafeBpeLimits {
        max_input_bytes: 4096,
        max_output_tokens: 4096,
        max_pair_checks: 1_000_000,
        max_pair_bytes: 8_000_000,
    }
}

fn compact_tokenizer(extra: Option<AddedToken>) -> (SpBpeTokenizer, u32) {
    let registries = archived_registries();
    let mut pieces = (0_u8..=u8::MAX).map(byte_piece).collect::<Vec<_>>();
    for entry in registries.template_controls().entries() {
        pieces.push(control_piece(&entry.surface));
    }
    let hello_id = pieces.len() as u32 + 8;
    for (surface, score) in [
        ("▁", 0.0), ("h", 0.0), ("e", 0.0), ("l", 0.0), ("o", 0.0),
        ("he", 100.0), ("hel", 90.0), ("hell", 80.0), ("hello", 70.0),
        ("▁h", 60.0), ("▁he", 50.0), ("▁hel", 40.0), ("▁hell", 30.0), ("▁hello", 20.0),
    ] {
        pieces.push(SpmPiece { piece: surface.to_owned(), score, piece_type: PieceType::Normal });
    }
    let added = registries.template_controls().entries().iter()
        .map(|entry| AddedToken::new(entry.surface.clone(), entry.id)).chain(extra);
    let tokenizer = SpBpeTokenizer::with_added_tokens(SpmModel {
        pieces, model_type: ModelType::Bpe,
        normalizer: NormalizerFacts { name: "identity".to_owned(), is_identity: true,
            precompiled_charsmap_is_empty: true },
        special_ids: SpecialPieceIds { unk_id: -1, bos_id: 257, eos_id: 258, pad_id: 256 },
    }, added).unwrap();
    (tokenizer, hello_id)
}

#[test]
fn explicit_safe_bpe_compacts_ordinary_pieces_without_changing_legacy_encoding() {
    let registries = archived_registries();
    let (tokenizer, hello_id) = compact_tokenizer(None);
    let encoder = UntrustedDocumentEncoder::new(&tokenizer, registries.template_controls());
    let input = b"xhello hello";
    let mut work = SafeBpeWork::default();
    let compact = encoder.encode_with_policy(input, &UntrustedEncoding::SafeBpeV1(compact_limits()),
        &mut work, &mut || true).unwrap();
    assert_eq!(compact.ids(), &[u32::from(b'x'), hello_id, hello_id + 5]);
    assert_eq!(compact.bytes(), input);
    assert_eq!(tokenizer.decode_bytes(compact.ids()).unwrap(), input);
    assert_eq!(work.input_bytes(), input.len());
    assert_eq!(work.output_tokens(), 3);
    assert!(work.pair_checks() > 0 && work.pair_bytes() >= work.pair_checks());
    let default = encoder.encode_with_policy(input, &UntrustedEncoding::default(),
        &mut SafeBpeWork::default(), &mut || true).unwrap();
    assert_eq!(default, encoder.encode(input).unwrap());
    assert_eq!(default.ids().len(), input.len());
}

#[test]
fn safe_bpe_preserves_whitespace_literal_markers_unicode_and_invalid_bytes_in_prompts() {
    let registries = archived_registries();
    let (tokenizer, _) = compact_tokenizer(None);
    let encoder = UntrustedDocumentEncoder::new(&tokenizer, registries.template_controls());
    for input in [
        b"".as_slice(), b"hello", b"  hello  hello\t\r\n", b"<think>hello</think><|im_start|>",
        "▁hello▁ hello".as_bytes(), "éhello 🦀hello 你好".as_bytes(),
        &[0x00, 0xff, 0x80, b'<', b'>', 0xfe],
    ] {
        let document = encoder.encode_with_policy(input, &UntrustedEncoding::SafeBpeV1(compact_limits()),
            &mut SafeBpeWork::default(), &mut || true).unwrap();
        assert_eq!(tokenizer.decode_bytes(document.ids()).unwrap(), input);
        assert!(document.ids().iter().all(|&id| !registries.template_controls().contains(id)));
        let mut prompt = vec![257];
        prompt.extend_from_slice(document.ids());
        prompt.push(258);
        assert_eq!(tokenizer.decode_bytes(&prompt).unwrap(),
            [b"<|im_start|>".as_slice(), input, b"<|im_end|>"].concat());
    }
    // A missing ordinary whitespace piece must fall back to ASCII source
    // spaces, never the UTF-8 bytes of the tokenizer's normalized marker.
    let byte_only = self::tokenizer();
    let encoder = UntrustedDocumentEncoder::new(&byte_only, registries.template_controls());
    let document = encoder.encode_with_policy(b"a  b", &UntrustedEncoding::SafeBpeV1(compact_limits()),
        &mut SafeBpeWork::default(), &mut || true).unwrap();
    assert_eq!(document.ids(), &[u32::from(b'a'), 32, 32, u32::from(b'b')]);
}

#[test]
fn safe_bpe_excludes_normal_control_ids_and_every_injected_decode_override() {
    let (tokenizer, hello_id) = compact_tokenizer(None);
    let mut fixture: Value = serde_json::from_str(REGISTRIES).unwrap();
    fixture["template_control_ids"]["entries"].as_array_mut().unwrap().push(
        serde_json::json!({"id":hello_id, "special":false, "surface":"hello"}));
    let controls = ArchivedControlRegistries::from_archived_json(
        &fixture["tokenizer_special_ids"].to_string(), &fixture["template_control_ids"].to_string()).unwrap();
    let encoder = UntrustedDocumentEncoder::new(&tokenizer, controls.template_controls());
    let document = encoder.encode_with_policy(b"xhello", &UntrustedEncoding::SafeBpeV1(compact_limits()),
        &mut SafeBpeWork::default(), &mut || true).unwrap();
    assert!(!document.ids().contains(&hello_id));
    assert_eq!(tokenizer.decode_bytes(document.ids()).unwrap(), b"xhello");

    let registries = archived_registries();
    let (overridden, _) = compact_tokenizer(Some(AddedToken::new("DIFFERENT", hello_id)));
    let encoder = UntrustedDocumentEncoder::new(&overridden, registries.template_controls());
    let document = encoder.encode_with_policy(b"xhello", &UntrustedEncoding::SafeBpeV1(compact_limits()),
        &mut SafeBpeWork::default(), &mut || true).unwrap();
    assert!(!document.ids().contains(&hello_id));
    assert_eq!(overridden.decode_bytes(document.ids()).unwrap(), b"xhello");

    let (overridden_byte, _) = compact_tokenizer(Some(AddedToken::new("DIFFERENT", u32::from(b'x'))));
    let encoder = UntrustedDocumentEncoder::new(&overridden_byte, registries.template_controls());
    assert!(matches!(encoder.encode_with_policy(b"xhello", &UntrustedEncoding::SafeBpeV1(compact_limits()),
        &mut SafeBpeWork::default(), &mut || true), Err(UntrustedDocumentError::UnsafeTokenId { id }) if id == u32::from(b'x')));
}

#[test]
fn safe_bpe_accounts_all_segments_and_retains_work_after_refusal() {
    let registries = archived_registries();
    let (tokenizer, _) = compact_tokenizer(None);
    let encoder = UntrustedDocumentEncoder::new(&tokenizer, registries.template_controls());
    let mut limits = compact_limits();
    limits.max_input_bytes = 3;
    limits.max_output_tokens = 3;
    let policy = UntrustedEncoding::SafeBpeV1(limits);
    let mut work = SafeBpeWork::default();
    encoder.encode_with_policy(b"a", &policy, &mut work, &mut || true).unwrap();
    encoder.encode_with_policy(b"bc", &policy, &mut work, &mut || true).unwrap();
    let before = work;
    assert!(matches!(encoder.encode_with_policy(b"d", &policy, &mut work, &mut || true),
        Err(UntrustedDocumentError::EncodingBudget { resource: "input bytes" })));
    assert_eq!(work, before);
    assert_eq!(work.input_bytes(), 3);
    assert_eq!(work.output_tokens(), 3);

    let mut limits = compact_limits();
    limits.max_output_tokens = 1;
    let mut work = SafeBpeWork::default();
    assert!(matches!(encoder.encode_with_policy(b"abc", &UntrustedEncoding::SafeBpeV1(limits),
        &mut work, &mut || true), Err(UntrustedDocumentError::EncodingBudget { resource: "output tokens" })));
    assert_eq!(work.input_bytes(), 3);
    assert_eq!(work.output_tokens(), 1);
}

#[test]
fn safe_bpe_bounds_quadratic_search_by_pair_count_and_candidate_bytes() {
    let registries = archived_registries();
    let (tokenizer, _) = compact_tokenizer(None);
    let encoder = UntrustedDocumentEncoder::new(&tokenizer, registries.template_controls());
    let mut limits = compact_limits();
    limits.max_pair_checks = 1;
    let mut work = SafeBpeWork::default();
    assert!(matches!(encoder.encode_with_policy(b"xhello", &UntrustedEncoding::SafeBpeV1(limits),
        &mut work, &mut || true), Err(UntrustedDocumentError::EncodingBudget { resource: "pair checks" })));
    assert_eq!(work.pair_checks(), 1);
    assert_eq!(work.output_tokens(), 1);
    let mut limits = compact_limits();
    limits.max_pair_bytes = 1;
    assert!(matches!(encoder.encode_with_policy(b"xhello", &UntrustedEncoding::SafeBpeV1(limits),
        &mut SafeBpeWork::default(), &mut || true), Err(UntrustedDocumentError::EncodingBudget { resource: "pair bytes" })));
}

#[test]
fn safe_bpe_cancellation_during_merge_and_after_encoding_returns_no_document() {
    let registries = archived_registries();
    let (tokenizer, _) = compact_tokenizer(None);
    let encoder = UntrustedDocumentEncoder::new(&tokenizer, registries.template_controls());
    let policy = UntrustedEncoding::SafeBpeV1(compact_limits());
    let mut calls = 0;
    let mut work = SafeBpeWork::default();
    assert!(matches!(encoder.encode_with_policy(b"xhello", &policy, &mut work, &mut || {
        calls += 1; calls < 5
    }), Err(UntrustedDocumentError::Cancelled)));
    assert!(work.input_bytes() > 0 && work.pair_checks() > 0);
    let mut total = 0;
    encoder.encode_with_policy(b"xhello", &policy, &mut SafeBpeWork::default(), &mut || {
        total += 1; true
    }).unwrap();
    let mut calls = 0;
    assert!(matches!(encoder.encode_with_policy(b"xhello", &policy, &mut SafeBpeWork::default(), &mut || {
        calls += 1; calls < total
    }), Err(UntrustedDocumentError::Cancelled)));
}

#[test]
fn safe_bpe_policy_has_explicit_identity_and_finite_preallocation_bounds() {
    let policy = UntrustedEncoding::SafeBpeV1(compact_limits());
    assert_ne!(policy.version(), UntrustedEncoding::default().version());
    let json = serde_json::to_string(&policy).unwrap();
    assert_eq!(serde_json::from_str::<UntrustedEncoding>(&json).unwrap(), policy);
    let requirements = compact_limits().requirements().unwrap();
    assert_eq!(requirements.workspace_bytes, 256 * 4096 + 16 * 4096 + 4096);
    let mut invalid = compact_limits();
    invalid.max_input_bytes = SafeBpeLimits::MAX_INPUT_BYTES + 1;
    assert!(invalid.requirements().is_err());
    invalid = compact_limits(); invalid.max_pair_checks = 0;
    assert!(UntrustedEncoding::SafeBpeV1(invalid).validate().is_err());
    assert!(serde_json::from_str::<UntrustedEncoding>(r#"{"kind":"byte_fallback","unexpected":1}"#).is_err());
}

#[test]
fn pinned_safe_bpe_uses_real_ordinary_vocabulary_and_preserves_registry_markers() {
    use franken_nlp::tokenizer::{embedded::EmbeddedTokenizer, pinned_controls};
    let tokenizer = EmbeddedTokenizer::pinned().unwrap();
    let registries = pinned_controls::pinned().unwrap();
    let encoder = UntrustedDocumentEncoder::new(tokenizer.tokenizer(), registries.template_controls());
    let input = b"The quick brown fox jumps over the lazy dog. Literal <think> and <|im_start|> remain document data.";
    let document = encoder.encode_with_policy(input, &UntrustedEncoding::SafeBpeV1(compact_limits()),
        &mut SafeBpeWork::default(), &mut || true).unwrap();
    assert!(document.ids().len() < input.len());
    assert_eq!(tokenizer.tokenizer().decode_bytes(document.ids()).unwrap(), input);
    assert!(document.ids().iter().all(|&id| !registries.template_controls().contains(id)));
}
