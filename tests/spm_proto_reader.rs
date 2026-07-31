//! Hermetic conformance cases for the bounded SentencePiece protobuf reader.
//!
//! The real pinned `tokenizer.model` golden is added with the verified tokenizer
//! asset; these tests intentionally exercise the same accepted ModelProto shape
//! without a model-weight dependency.

use franken_nlp::tokenizer::sp_model::{
    MAX_PIECE_STRING_BYTES, ModelType, PieceType, SpmError, SpmErrorKind, parse_spm_model,
};

fn varint(mut value: u64) -> Vec<u8> {
    let mut output = Vec::new();
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        output.push(byte);
        if value == 0 {
            return output;
        }
    }
}

fn key(field: u32, wire_type: u8) -> Vec<u8> {
    varint((u64::from(field) << 3) | u64::from(wire_type))
}

fn add_varint(output: &mut Vec<u8>, field: u32, value: u64) {
    output.extend(key(field, 0));
    output.extend(varint(value));
}

fn add_fixed32(output: &mut Vec<u8>, field: u32, value: u32) {
    output.extend(key(field, 5));
    output.extend(value.to_le_bytes());
}

fn add_bytes(output: &mut Vec<u8>, field: u32, value: &[u8]) {
    output.extend(key(field, 2));
    output.extend(varint(value.len() as u64));
    output.extend(value);
}

fn piece(text: &[u8], score_bits: u32, piece_type: u64) -> Vec<u8> {
    let mut output = Vec::new();
    add_bytes(&mut output, 1, text);
    add_fixed32(&mut output, 2, score_bits);
    add_varint(&mut output, 3, piece_type);
    output
}

fn trainer(model_type: u64) -> Vec<u8> {
    let mut output = Vec::new();
    add_varint(&mut output, 3, model_type);
    add_varint(&mut output, 40, 0);
    add_varint(&mut output, 41, 1);
    add_varint(&mut output, 42, 2);
    add_varint(&mut output, 43, u64::MAX);
    output
}

fn identity_normalizer() -> Vec<u8> {
    let mut output = Vec::new();
    add_bytes(&mut output, 1, b"identity");
    output
}

fn model_with_piece(piece_payload: Vec<u8>) -> Vec<u8> {
    let mut output = Vec::new();
    add_bytes(&mut output, 1, &piece_payload);
    add_bytes(&mut output, 2, &trainer(2));
    add_bytes(&mut output, 3, &identity_normalizer());
    output
}

fn valid_model() -> Vec<u8> {
    let mut output = model_with_piece(piece(b"<unk>", (-4.25_f32).to_bits(), 2));
    add_bytes(&mut output, 1, &piece(b"a", 1.5_f32.to_bits(), 1));
    output
}

fn assert_context(error: &SpmError) {
    let display = error.to_string();
    assert!(display.contains("offset="), "missing offset: {display}");
    assert!(display.contains("field=") || error.field_number.is_none());
}

#[test]
fn parses_the_bpe_identity_model_subset_exactly() {
    let model = parse_spm_model(&valid_model()).expect("minimal BPE ModelProto must parse");
    assert_eq!(model.model_type, ModelType::Bpe);
    assert!(model.normalizer.is_identity);
    assert_eq!(model.normalizer.name, "identity");
    assert_eq!(model.special_ids.unk_id, 0);
    assert_eq!(model.special_ids.bos_id, 1);
    assert_eq!(model.special_ids.eos_id, 2);
    assert_eq!(model.special_ids.pad_id, -1);
    assert_eq!(model.pieces.len(), 2);
    assert_eq!(model.pieces[0].piece, "<unk>");
    assert_eq!(model.pieces[0].score.to_bits(), (-4.25_f32).to_bits());
    assert_eq!(model.pieces[0].piece_type, PieceType::Unknown);
    assert_eq!(model.pieces[1].piece, "a");
    assert_eq!(model.pieces[1].score.to_bits(), 1.5_f32.to_bits());
    assert_eq!(model.pieces[1].piece_type, PieceType::Normal);
}

#[test]
fn skips_all_non_group_unknown_wire_types() {
    let mut input = valid_model();
    add_varint(&mut input, 99, 7);
    input.extend(key(100, 1));
    input.extend([0; 8]);
    add_bytes(&mut input, 101, b"ignored");
    add_fixed32(&mut input, 102, 0xfeed_beef);
    assert!(parse_spm_model(&input).is_ok());
}

#[test]
fn rejects_unterminated_and_overlong_varints() {
    let unterminated = parse_spm_model(&[0x80]).expect_err("truncated varint must reject");
    assert!(matches!(
        unterminated.kind,
        SpmErrorKind::VarintUnterminated
    ));
    assert_context(&unterminated);

    let overlong = parse_spm_model(&[0x80; 11]).expect_err("eleven-byte varint must reject");
    assert!(matches!(
        overlong.kind,
        SpmErrorKind::VarintTooLong { .. } | SpmErrorKind::VarintOverflow
    ));
    assert_context(&overlong);
}

#[test]
fn rejects_overflow_and_truncated_lengths() {
    let mut overflow = key(1, 2);
    overflow.extend([0xff; 9]);
    overflow.push(0x01);
    let error = parse_spm_model(&overflow).expect_err("overflow length must reject");
    assert!(matches!(error.kind, SpmErrorKind::LengthOverflow));
    assert_context(&error);

    let error = parse_spm_model(&[0x0a, 0x02, 0x0a])
        .expect_err("nested message beyond its length boundary must reject");
    assert!(matches!(error.kind, SpmErrorKind::Truncated { .. }));
    assert_context(&error);
}

#[test]
fn rejects_trailing_garbage_after_a_complete_model() {
    let mut input = valid_model();
    // A zero protobuf key cannot begin another field, so it is malformed
    // trailing data rather than an unknown field that the closed reader may
    // safely skip.
    input.push(0);
    let error = parse_spm_model(&input).expect_err("trailing garbage must reject");
    assert!(matches!(error.kind, SpmErrorKind::InvalidFieldNumber));
    assert_context(&error);
}

#[test]
fn rejects_conflicting_singular_and_invalid_utf8_piece_fields() {
    let mut duplicate_piece = piece(b"a", 0.0_f32.to_bits(), 1);
    add_bytes(&mut duplicate_piece, 1, b"b");
    let error = parse_spm_model(&model_with_piece(duplicate_piece))
        .expect_err("different duplicate piece text must reject");
    assert!(matches!(
        error.kind,
        SpmErrorKind::DuplicateSingular {
            field_name: "piece.piece"
        }
    ));
    assert_context(&error);

    let error = parse_spm_model(&model_with_piece(piece(&[0xff], 0, 1)))
        .expect_err("invalid piece UTF-8 must reject");
    assert!(matches!(
        error.kind,
        SpmErrorKind::InvalidUtf8 {
            field_name: "piece.piece"
        }
    ));
    assert_context(&error);
}

#[test]
fn rejects_group_wire_type_and_piece_string_bomb() {
    let error = parse_spm_model(&[0x0b]).expect_err("groups are not a supported wire type");
    assert!(matches!(error.kind, SpmErrorKind::GroupWireTypeUnsupported));
    assert_context(&error);

    let long_piece = vec![b'x'; MAX_PIECE_STRING_BYTES + 1];
    let error = parse_spm_model(&model_with_piece(piece(&long_piece, 0, 1)))
        .expect_err("piece string cap must reject");
    assert!(matches!(
        error.kind,
        SpmErrorKind::LimitExceeded {
            limit_name: "MAX_PIECE_STRING_BYTES",
            ..
        }
    ));
    assert_context(&error);
}

#[test]
fn rejects_non_bpe_and_non_identity_assertions() {
    let mut non_bpe = Vec::new();
    add_bytes(&mut non_bpe, 1, &piece(b"a", 0, 1));
    add_bytes(&mut non_bpe, 2, &trainer(1));
    add_bytes(&mut non_bpe, 3, &identity_normalizer());
    assert!(matches!(
        parse_spm_model(&non_bpe)
            .expect_err("unigram must reject")
            .kind,
        SpmErrorKind::UnsupportedModelType { value: 1 }
    ));

    let mut non_identity = Vec::new();
    add_bytes(&mut non_identity, 1, &piece(b"a", 0, 1));
    add_bytes(&mut non_identity, 2, &trainer(2));
    let mut normalizer = Vec::new();
    add_bytes(&mut normalizer, 1, b"nmt_nfkc");
    add_bytes(&mut non_identity, 3, &normalizer);
    assert!(matches!(
        parse_spm_model(&non_identity)
            .expect_err("non-identity normalizer must reject")
            .kind,
        SpmErrorKind::NonIdentityNormalizer { .. }
    ));
}
