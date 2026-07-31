//! Hermetic L0 tokenizer semantics before the pinned oracle corpus is materialized.
//!
//! The committed Phase -1 corpus replaces the synthetic rows below for the
//! actual slow-reference equality gate. These rows lock the in-repo semantic
//! authority: score-ordered merges, byte fallback, injected-token precedence,
//! BOS/EOS options, and strict-vs-lossy decode behavior.

use franken_nlp::tokenizer::{
    bpe::{AddedToken, DecodeBytesError, DecodeTextError, EncodeOptions, SpBpeTokenizer},
    sp_model::{NormalizerFacts, PieceType, SpecialPieceIds, SpmModel, SpmPiece},
};
use sha2::{Digest, Sha256};

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

#[test]
fn l0_synthetic_merge_rows_log_exact_digests() {
    let tokenizer = tokenizer();
    let rows = [("ab", vec![1, 8]), (" a", vec![1, 3, 8])];
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
