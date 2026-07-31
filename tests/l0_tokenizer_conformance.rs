//! Hermetic L0 tokenizer semantics plus the pinned slow-reference corpus.
//!
//! The synthetic rows lock local BPE invariants, while the exact Apache-2.0
//! SentencePiece model and frozen slow-reference IDs exercise the one-model
//! vocabulary without making a model-weight artifact a test prerequisite.

use std::collections::BTreeMap;

use franken_nlp::native_engine::lmhead::NANBEIGE_VOCAB_SIZE;
use franken_nlp::tokenizer::{
    bpe::{AddedToken, DecodeBytesError, DecodeTextError, EncodeOptions, SpBpeTokenizer},
    sp_model::{parse_spm_model, NormalizerFacts, PieceType, SpecialPieceIds, SpmModel, SpmPiece},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const PINNED_TOKENIZER_MODEL: &[u8] = include_bytes!("fixtures/reference/tokenizer.model");
const PINNED_TOKENIZER_SHA256: &str =
    "fb41d04798b714520a9b075727b0226538b7330254299062742c50ec8374bc36";
const PINNED_ADDED_TOKENS: &[u8] = include_bytes!("fixtures/reference/added_tokens.json");
const PINNED_ADDED_TOKENS_SHA256: &str =
    "9e3b127a27647df2c353cc1e5500826f7cdbe8bd15a458e368bba8422e9719cf";
const PINNED_MODEL_CONFIG: &[u8] = include_bytes!("fixtures/reference/config.json");
const PINNED_MODEL_CONFIG_SHA256: &str =
    "f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19";
const PINNED_TOKENIZER_CONFIG: &[u8] = include_bytes!("fixtures/reference/tokenizer_config.json");
const PINNED_TOKENIZER_CONFIG_SHA256: &str =
    "3edfa64a0826a77e9412b9008f1febf3fe906a68fd616b6de4cd15897a8c8518";
const REFERENCE_AUXILIARY: &str = include_str!("fixtures/reference/auxiliary.json");
const REFERENCE_INPUTS: &str = include_str!("fixtures/reference_inputs.json");

const SP_PIECE_COUNT: usize = 166_100;
const ADDED_TOKEN_COUNT: usize = 7;
const REAL_VOCAB_SIZE: usize = SP_PIECE_COUNT + ADDED_TOKEN_COUNT;
const PADDED_EMBEDDING_ROWS: usize = 37;

#[derive(Deserialize)]
struct ReferenceInputs {
    tokenizer_cases: Vec<ReferenceInputCase>,
}

#[derive(Deserialize)]
struct ReferenceInputCase {
    id: String,
    text: String,
}

#[derive(Deserialize)]
struct AuxiliaryFixtures {
    tokenizer_cases: Vec<SlowTokenizerCase>,
}

#[derive(Deserialize)]
struct SlowTokenizerCase {
    id: String,
    input_sha256: String,
    token_ids: Vec<u32>,
    token_ids_sha256: String,
}

#[derive(Deserialize)]
struct PinnedModelConfig {
    vocab_size: usize,
}

#[derive(Deserialize)]
struct PinnedTokenizerConfig {
    added_tokens_decoder: BTreeMap<String, PinnedAddedTokenMetadata>,
}

#[derive(Deserialize)]
struct PinnedAddedTokenMetadata {
    content: String,
    special: bool,
}

fn piece(surface: &str, score: f32, piece_type: PieceType) -> SpmPiece {
    SpmPiece {
        piece: surface.to_owned(),
        score,
        piece_type,
    }
}

fn toy_model() -> SpmModel {
    SpmModel {
        pieces: vec![
            piece("<unk>", 0.0, PieceType::Unknown),
            piece("<s>", 0.0, PieceType::Control),
            piece("</s>", 0.0, PieceType::Control),
            piece("▁", -10.0, PieceType::Normal),
            piece("a", -10.0, PieceType::Normal),
            piece("b", -10.0, PieceType::Normal),
            piece("▁a", 5.0, PieceType::Normal),
            piece("ab", 6.0, PieceType::Normal),
            piece("▁ab", 20.0, PieceType::Normal),
            piece("<0xFF>", 0.0, PieceType::Byte),
            piece("<user>", 0.0, PieceType::UserDefined),
        ],
        model_type: franken_nlp::tokenizer::sp_model::ModelType::Bpe,
        normalizer: NormalizerFacts {
            name: "identity".to_owned(),
            is_identity: true,
            precompiled_charsmap_is_empty: true,
        },
        special_ids: SpecialPieceIds {
            unk_id: 0,
            bos_id: 1,
            eos_id: 2,
            pad_id: -1,
        },
    }
}

fn tokenizer() -> SpBpeTokenizer {
    SpBpeTokenizer::from_model(toy_model()).expect("toy SPM model is internally consistent")
}

fn digest_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_id_digest(ids: &[u32]) -> String {
    let mut encoded = serde_json::to_string_pretty(ids).expect("token IDs serialize to JSON");
    encoded.push('\n');
    digest_hex(encoded.as_bytes())
}

#[test]
fn l0_synthetic_merge_rows_log_exact_digests() {
    let tokenizer = tokenizer();
    let rows: [(&str, Vec<u32>); 2] = [("ab", vec![1, 8]), (" a", vec![1, 3, 6])];
    for (input, expected_ids) in rows {
        let got_ids = tokenizer.encode_ids(input).expect("toy input encodes");
        let expected = expected_ids
            .iter()
            .flat_map(|id| (*id).to_le_bytes())
            .collect::<Vec<_>>();
        let got = got_ids
            .iter()
            .flat_map(|id| (*id).to_le_bytes())
            .collect::<Vec<_>>();
        eprintln!(
            "L0 corpus=synthetic RESULT={} lines=1 input_sha256={} expected_ids_sha256={} got_ids_sha256={}",
            if got_ids == expected_ids {
                "PASS"
            } else {
                "FAIL"
            },
            digest_hex(input.as_bytes()),
            digest_hex(&expected),
            digest_hex(&got),
        );
        assert_eq!(
            got_ids, expected_ids,
            "L0 first_diverging_index is logged by the pinned-corpus harness"
        );
    }
    eprintln!("L0_TOKENIZER RESULT=PASS scope=synthetic mismatches=0");
}

#[test]
fn merge_score_order_selects_the_highest_scored_adjacent_pair() {
    let tokenizer = tokenizer();
    // `ab` (score 6) beats `▁a` (score 5), then `▁ab` (score 20)
    // becomes available and wins the second merge round.
    assert_eq!(tokenizer.encode_ids("ab").unwrap(), vec![1, 8]);
}

#[test]
fn bos_eos_and_empty_input_are_explicit() {
    let tokenizer = tokenizer();
    assert_eq!(tokenizer.encode_ids("").unwrap(), vec![1]);
    assert_eq!(
        tokenizer
            .encode_ids_with_options(
                "",
                EncodeOptions {
                    add_bos: false,
                    add_eos: true,
                },
            )
            .unwrap(),
        vec![2]
    );
}

#[test]
fn byte_fallback_is_lossless_and_text_decode_requires_an_explicit_opt_in() {
    let tokenizer = tokenizer();
    let ids = tokenizer
        .encode_bytes(&[0xff], EncodeOptions::default())
        .expect("byte piece exists");
    assert_eq!(ids, vec![1, 9]);
    assert_eq!(tokenizer.decode_bytes(&[9]).unwrap(), vec![0xff]);
    assert!(matches!(
        tokenizer.decode_text(&[9]),
        Err(DecodeTextError::InvalidUtf8 { .. })
    ));
    let lossy = tokenizer.decode_text_lossy(&[9]).unwrap();
    assert!(lossy.had_invalid_utf8);
    assert_eq!(lossy.text, "�");
}

#[test]
fn added_tokens_take_longest_surface_precedence_before_bpe() {
    let tokenizer = SpBpeTokenizer::with_added_tokens(
        toy_model(),
        [AddedToken::new("<x>", 0), AddedToken::new("<x:y>", 2)],
    )
    .expect("registered ids exist in toy SPM table");
    assert_eq!(tokenizer.encode_ids("<x:y>").unwrap(), vec![1, 2]);
    assert_eq!(tokenizer.decode_bytes(&[2]).unwrap(), b"<x:y>");
}

#[test]
fn sentencepiece_dummy_prefix_round_trips_leading_whitespace() {
    let tokenizer = tokenizer();
    let ids = tokenizer.encode_ids(" a").unwrap();
    assert_eq!(ids, vec![1, 3, 6]);
    assert_eq!(tokenizer.decode_text(&ids).unwrap(), " a");
}

#[test]
fn unknown_token_ids_fail_closed_and_arbitrary_inputs_never_panic() {
    let tokenizer = tokenizer();
    assert!(matches!(
        tokenizer.decode_bytes(&[99]),
        Err(DecodeBytesError::UnknownTokenId { id: 99, .. })
    ));

    let alphabet = ['a', 'b', ' ', 'λ'];
    let mut state = 0x5eed_cafe_u64;
    for length in 0..64 {
        let mut input = String::new();
        for _ in 0..length {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            input.push(alphabet[(state as usize) % alphabet.len()]);
        }
        let _ = tokenizer.encode_ids(&input);
        let _ = tokenizer.encode_bytes(input.as_bytes(), EncodeOptions::default());
    }
}

#[test]
fn pinned_slow_reference_vocabulary_is_token_id_exact() {
    assert_eq!(
        digest_hex(PINNED_TOKENIZER_MODEL),
        PINNED_TOKENIZER_SHA256,
        "the real-vocabulary fixture must remain the pinned tokenizer.model bytes"
    );
    assert_eq!(
        digest_hex(PINNED_ADDED_TOKENS),
        PINNED_ADDED_TOKENS_SHA256,
        "the real-vocabulary fixture must remain the pinned added_tokens.json bytes"
    );
    assert_eq!(
        digest_hex(PINNED_MODEL_CONFIG),
        PINNED_MODEL_CONFIG_SHA256,
        "the real-vocabulary fixture must remain the pinned config.json bytes"
    );
    assert_eq!(
        digest_hex(PINNED_TOKENIZER_CONFIG),
        PINNED_TOKENIZER_CONFIG_SHA256,
        "the real-vocabulary fixture must remain the pinned tokenizer_config.json bytes"
    );

    let model = parse_spm_model(PINNED_TOKENIZER_MODEL)
        .expect("the hash-checked pinned tokenizer fixture must parse");
    assert_eq!(
        model.pieces.len(),
        SP_PIECE_COUNT,
        "the SentencePiece file owns exactly the base-ID range"
    );

    let added_by_surface: BTreeMap<String, u32> = serde_json::from_slice(PINNED_ADDED_TOKENS)
        .expect("the hash-checked pinned added-token registry is valid JSON");
    assert_eq!(
        added_by_surface.len(),
        ADDED_TOKEN_COUNT,
        "added_tokens.json owns precisely the seven post-SP real tokens"
    );
    let added_by_id = added_by_surface
        .iter()
        .map(|(surface, id)| (*id, surface.as_str()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        added_by_id.keys().copied().collect::<Vec<_>>(),
        (SP_PIECE_COUNT as u32..REAL_VOCAB_SIZE as u32).collect::<Vec<_>>(),
        "added-token IDs must be the complete contiguous real-ID tail"
    );

    let tokenizer_config: PinnedTokenizerConfig = serde_json::from_slice(PINNED_TOKENIZER_CONFIG)
        .expect("the hash-checked pinned tokenizer config is valid JSON");
    let configured_added_by_id = tokenizer_config
        .added_tokens_decoder
        .into_iter()
        .filter_map(|(id, metadata)| {
            let id = id
                .parse::<u32>()
                .expect("tokenizer_config added-token keys are unsigned IDs");
            (id >= SP_PIECE_COUNT as u32).then_some((id, metadata))
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        configured_added_by_id.len(),
        ADDED_TOKEN_COUNT,
        "tokenizer_config must agree that exactly seven IDs extend SentencePiece"
    );
    for (&id, surface) in &added_by_id {
        let metadata = configured_added_by_id
            .get(&id)
            .expect("every added_tokens.json ID appears in tokenizer_config");
        assert_eq!(
            metadata.content, surface,
            "tokenizer metadata surface drifted at added token id={id}"
        );
    }
    for id in 166_100..=166_102 {
        assert!(
            configured_added_by_id[&id].special,
            "pinned tokenizer metadata must retain special=true at id={id}"
        );
    }
    for id in 166_103..=166_106 {
        assert!(
            !configured_added_by_id[&id].special,
            "thinking and tool markers remain template controls even though their tokenizer metadata is special=false at id={id}"
        );
    }

    let model_config: PinnedModelConfig = serde_json::from_slice(PINNED_MODEL_CONFIG)
        .expect("the hash-checked pinned model config is valid JSON");
    assert_eq!(
        model_config.vocab_size, 166_144,
        "config.json declares the embedding/lm-head row width"
    );
    assert_eq!(
        NANBEIGE_VOCAB_SIZE, model_config.vocab_size,
        "the native embedding/lm-head width must match pinned config.json"
    );
    assert_eq!(REAL_VOCAB_SIZE, 166_107, "real token vocabulary size");
    assert_eq!(
        model_config.vocab_size - REAL_VOCAB_SIZE,
        PADDED_EMBEDDING_ROWS,
        "the configured width has only unassigned alignment padding after real tokens"
    );
    assert_eq!(
        model_config.vocab_size % 128,
        0,
        "embedding width is 128-aligned"
    );

    let tokenizer = SpBpeTokenizer::with_added_tokens(
        model,
        added_by_surface
            .iter()
            .map(|(surface, id)| AddedToken::new(surface.clone(), *id)),
    )
    .expect("the pinned SentencePiece and added-token registries must build a tokenizer");
    assert_eq!(
        tokenizer.piece_count(),
        SP_PIECE_COUNT,
        "SentencePiece and added-token ID layers stay distinct in the tokenizer"
    );
    for (&id, surface) in &added_by_id {
        let encoded = tokenizer
            .encode_ids_with_options(
                surface,
                EncodeOptions {
                    add_bos: false,
                    add_eos: false,
                },
            )
            .expect("every pinned added-token surface must encode");
        assert_eq!(
            encoded,
            vec![id],
            "added-token surface must retain its exact real-vocabulary id"
        );
        assert_eq!(
            tokenizer.decode_bytes(&[id]).expect("added id must decode"),
            surface.as_bytes(),
            "added-token id must retain its exact surface"
        );
    }
    for id in REAL_VOCAB_SIZE as u32..model_config.vocab_size as u32 {
        assert_eq!(
            tokenizer.decode_bytes(&[id]),
            Err(DecodeBytesError::UnknownTokenId {
                id,
                piece_count: SP_PIECE_COUNT,
            }),
            "alignment-padding row id={id} must never decode as a token"
        );
    }
    let inputs: ReferenceInputs =
        serde_json::from_str(REFERENCE_INPUTS).expect("reference input corpus JSON is valid");
    let fixtures: AuxiliaryFixtures =
        serde_json::from_str(REFERENCE_AUXILIARY).expect("slow tokenizer fixture JSON is valid");

    let mut mismatches = 0_usize;
    for fixture in fixtures.tokenizer_cases {
        let input = inputs
            .tokenizer_cases
            .iter()
            .find(|candidate| candidate.id == fixture.id)
            .expect("every slow-reference tokenizer row has a repository-authored input");
        assert_eq!(
            digest_hex(input.text.as_bytes()),
            fixture.input_sha256,
            "fixture input digest drifted for case={}",
            fixture.id
        );
        assert_eq!(
            canonical_id_digest(&fixture.token_ids),
            fixture.token_ids_sha256,
            "fixture expected-ID digest drifted for case={}",
            fixture.id
        );

        let got = tokenizer
            .encode_ids_with_options(
                &input.text,
                EncodeOptions {
                    add_bos: false,
                    add_eos: false,
                },
            )
            .expect("pinned slow-reference text must encode");
        assert!(
            got.iter().all(|id| *id < REAL_VOCAB_SIZE as u32),
            "encoder emitted an unassigned alignment-padding id for case={}",
            fixture.id
        );
        if got != fixture.token_ids {
            mismatches += 1;
            let index = got
                .iter()
                .zip(&fixture.token_ids)
                .position(|(actual, expected)| actual != expected)
                .unwrap_or_else(|| got.len().min(fixture.token_ids.len()));
            let start = index.saturating_sub(8);
            let expected_end = fixture.token_ids.len().min(index.saturating_add(9));
            let got_end = got.len().min(index.saturating_add(9));
            eprintln!(
                "L0 corpus=slow-reference case={} RESULT=FAIL input_sha256={} expected_ids_sha256={} got_ids_sha256={} first_diverging_index={} expected_context={:?} got_context={:?}",
                fixture.id,
                fixture.input_sha256,
                fixture.token_ids_sha256,
                canonical_id_digest(&got),
                index,
                &fixture.token_ids[start..expected_end],
                &got[start..got_end],
            );
            continue;
        }
        eprintln!(
            "L0 corpus=slow-reference case={} RESULT=PASS input_sha256={} expected_ids_sha256={} got_ids_sha256={}",
            fixture.id,
            fixture.input_sha256,
            fixture.token_ids_sha256,
            canonical_id_digest(&got),
        );
    }
    eprintln!(
        "L0_TOKENIZER RESULT={} scope=slow-reference mismatches={mismatches}",
        if mismatches == 0 { "PASS" } else { "FAIL" }
    );
    assert_eq!(
        mismatches, 0,
        "slow-reference corpus must be token-id exact"
    );
}
