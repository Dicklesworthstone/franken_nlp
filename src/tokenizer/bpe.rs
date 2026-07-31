//! Exactness-oriented SentencePiece BPE operations for the pinned model shape.
//!
//! This is deliberately the simple semantic implementation used as the L0
//! authority: it applies the SentencePiece dummy-prefix/whitespace marker,
//! performs score-ordered adjacent-pair merges, uses BYTE pieces for otherwise
//! unrepresentable bytes, and keeps byte decoding separate from UTF-8 decoding.
//! Future lookup-table acceleration must differential-test against this module.

use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    str::{Utf8Error, from_utf8},
};

use super::sp_model::{PieceType, SpecialPieceIds, SpmModel};

/// SentencePiece's visible replacement for an escaped input whitespace byte.
pub const WHITESPACE_MARKER: char = '\u{2581}';

/// Controls explicit BOS/EOS insertion for one encoding operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeOptions {
    /// Insert the trainer-declared BOS id before user input.
    pub add_bos: bool,
    /// Insert the trainer-declared EOS id after user input.
    pub add_eos: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            // Pinned tokenizer_config.json facts, promoted through the fixture
            // corpus: add_bos=true and add_eos=false.
            add_bos: true,
            add_eos: false,
        }
    }
}

/// An added or user-defined token that must be injected before BPE segmentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedToken {
    pub content: String,
    pub id: u32,
}

impl AddedToken {
    #[must_use]
    pub fn new(content: impl Into<String>, id: u32) -> Self {
        Self {
            content: content.into(),
            id,
        }
    }
}

/// Successful explicitly lossy text decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LossyText {
    pub text: String,
    pub had_invalid_utf8: bool,
}

/// Construction failures for a model whose BPE surface is internally inconsistent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BpeBuildError {
    NonIdentityNormalizer,
    PieceCountOutOfRange { piece_count: usize },
    DuplicateNormalPiece { surface: String },
    DuplicateBytePiece { byte: u8 },
    MalformedBytePiece { surface: String },
    EmptyAddedToken,
    ConflictingAddedToken { surface: String },
    ConflictingAddedTokenId { id: u32 },
}

impl fmt::Display for BpeBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonIdentityNormalizer => {
                write!(formatter, "SP_BPE build error=non-identity-normalizer")
            }
            Self::PieceCountOutOfRange { piece_count } => write!(
                formatter,
                "SP_BPE build error=piece-count-out-of-range piece_count={piece_count}"
            ),
            Self::DuplicateNormalPiece { surface } => {
                write!(
                    formatter,
                    "SP_BPE build error=duplicate-normal-piece surface={surface:?}"
                )
            }
            Self::DuplicateBytePiece { byte } => {
                write!(
                    formatter,
                    "SP_BPE build error=duplicate-byte-piece byte=0x{byte:02X}"
                )
            }
            Self::MalformedBytePiece { surface } => {
                write!(
                    formatter,
                    "SP_BPE build error=malformed-byte-piece surface={surface:?}"
                )
            }
            Self::EmptyAddedToken => write!(formatter, "SP_BPE build error=empty-added-token"),
            Self::ConflictingAddedToken { surface } => write!(
                formatter,
                "SP_BPE build error=conflicting-added-token surface={surface:?}"
            ),
            Self::ConflictingAddedTokenId { id } => {
                write!(
                    formatter,
                    "SP_BPE build error=conflicting-added-token-id id={id}"
                )
            }
        }
    }
}

impl Error for BpeBuildError {}

/// Input-side refusal. Bytes never silently disappear when no BYTE piece exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    ByteFallbackMissing { byte: u8 },
    ConfiguredSpecialIdOutOfRange { field: &'static str, id: i32 },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ByteFallbackMissing { byte } => {
                write!(
                    formatter,
                    "SP_BPE encode error=byte-fallback-missing byte=0x{byte:02X}"
                )
            }
            Self::ConfiguredSpecialIdOutOfRange { field, id } => write!(
                formatter,
                "SP_BPE encode error=configured-special-id-out-of-range field={field} id={id}"
            ),
        }
    }
}

impl Error for EncodeError {}

/// Byte decode remains fallible because arbitrary token ids must fail closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeBytesError {
    UnknownTokenId { id: u32, piece_count: usize },
}

impl fmt::Display for DecodeBytesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTokenId { id, piece_count } => write!(
                formatter,
                "SP_BPE decode error=unknown-token-id id={id} piece_count={piece_count}"
            ),
        }
    }
}

impl Error for DecodeBytesError {}

/// Strict text decoding errors; callers must explicitly opt into lossy output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeTextError {
    Bytes(DecodeBytesError),
    InvalidUtf8 {
        valid_up_to: usize,
        error_len: Option<usize>,
    },
}

impl fmt::Display for DecodeTextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytes(error) => error.fmt(formatter),
            Self::InvalidUtf8 {
                valid_up_to,
                error_len,
            } => write!(
                formatter,
                "SP_BPE decode_text error=invalid-utf8 valid_up_to={valid_up_to} error_len={error_len:?}"
            ),
        }
    }
}

impl Error for DecodeTextError {}

#[derive(Debug, Clone)]
struct ScoredPiece {
    id: u32,
    score: f32,
}

#[derive(Debug, Clone)]
enum DecoderPiece {
    Text {
        content: String,
        piece_type: PieceType,
    },
    Byte(u8),
}

#[derive(Debug, Clone)]
struct Symbol {
    text: String,
    next: Option<usize>,
    alive: bool,
}

/// Dependency-free SentencePiece BPE encoder/decoder.
///
/// `from_model` accepts the SPM-piece surface alone. Use
/// [`Self::with_added_tokens`] once the pinned added-token registry is loaded;
/// those surfaces then take precedence over ordinary BPE segmentation.
#[derive(Debug, Clone)]
pub struct SpBpeTokenizer {
    pieces: Vec<DecoderPiece>,
    normal_pieces: BTreeMap<String, ScoredPiece>,
    byte_pieces: [Option<u32>; 256],
    injected_by_text: Vec<AddedToken>,
    injected_by_id: BTreeMap<u32, String>,
    special_ids: SpecialPieceIds,
}

impl SpBpeTokenizer {
    /// Builds the ordinary BPE surface without externally supplied added tokens.
    pub fn from_model(model: SpmModel) -> Result<Self, BpeBuildError> {
        Self::with_added_tokens(model, std::iter::empty::<AddedToken>())
    }

    /// Builds the BPE surface and registers exact-precedence added tokens.
    pub fn with_added_tokens(
        model: SpmModel,
        added_tokens: impl IntoIterator<Item = AddedToken>,
    ) -> Result<Self, BpeBuildError> {
        if !model.normalizer.is_identity || !model.normalizer.precompiled_charsmap_is_empty {
            return Err(BpeBuildError::NonIdentityNormalizer);
        }

        let piece_count = model.pieces.len();
        let mut pieces = Vec::new();
        let mut normal_pieces = BTreeMap::new();
        let mut byte_pieces = [None; 256];
        let mut injected = BTreeMap::new();
        let mut injected_by_id = BTreeMap::new();

        for (index, piece) in model.pieces.into_iter().enumerate() {
            let id = u32::try_from(index)
                .map_err(|_| BpeBuildError::PieceCountOutOfRange { piece_count })?;
            match piece.piece_type {
                PieceType::Normal => {
                    if normal_pieces
                        .insert(
                            piece.piece.clone(),
                            ScoredPiece {
                                id,
                                score: piece.score,
                            },
                        )
                        .is_some()
                    {
                        return Err(BpeBuildError::DuplicateNormalPiece {
                            surface: piece.piece,
                        });
                    }
                    pieces.push(DecoderPiece::Text {
                        content: piece.piece,
                        piece_type: PieceType::Normal,
                    });
                }
                PieceType::Byte => {
                    let byte = parse_byte_piece(&piece.piece).ok_or_else(|| {
                        BpeBuildError::MalformedBytePiece {
                            surface: piece.piece.clone(),
                        }
                    })?;
                    if byte_pieces[usize::from(byte)].replace(id).is_some() {
                        return Err(BpeBuildError::DuplicateBytePiece { byte });
                    }
                    pieces.push(DecoderPiece::Byte(byte));
                }
                PieceType::UserDefined => {
                    insert_injected(
                        &mut injected,
                        &mut injected_by_id,
                        AddedToken::new(piece.piece.clone(), id),
                    )?;
                    pieces.push(DecoderPiece::Text {
                        content: piece.piece,
                        piece_type: PieceType::UserDefined,
                    });
                }
                piece_type => pieces.push(DecoderPiece::Text {
                    content: piece.piece,
                    piece_type,
                }),
            }
        }

        for token in added_tokens {
            if token.content.is_empty() {
                return Err(BpeBuildError::EmptyAddedToken);
            }
            // The tokenizer's explicit registry can occupy ids immediately
            // after the SentencePiece table.  They are not synthetic SPM
            // pieces: `injected_by_id` owns their decode surface and
            // `injected_by_text` owns their exact-precedence encode surface.
            insert_injected(&mut injected, &mut injected_by_id, token)?;
        }

        let mut injected_by_text: Vec<_> = injected
            .into_iter()
            .map(|(content, id)| AddedToken { content, id })
            .collect();
        // Match at the first source offset; longest surface then lowest id
        // resolves equal-offset overlaps deterministically.
        injected_by_text.sort_by(|left, right| {
            right
                .content
                .len()
                .cmp(&left.content.len())
                .then_with(|| left.id.cmp(&right.id))
        });

        Ok(Self {
            pieces,
            normal_pieces,
            byte_pieces,
            injected_by_text,
            injected_by_id,
            special_ids: model.special_ids,
        })
    }

    #[must_use]
    pub fn piece_count(&self) -> usize {
        self.pieces.len()
    }

    /// Encodes text with the pinned default `add_bos=true`, `add_eos=false`.
    pub fn encode_ids(&self, input: &str) -> Result<Vec<u32>, EncodeError> {
        self.encode_ids_with_options(input, EncodeOptions::default())
    }

    /// Encodes text with caller-controlled special-id insertion.
    pub fn encode_ids_with_options(
        &self,
        input: &str,
        options: EncodeOptions,
    ) -> Result<Vec<u32>, EncodeError> {
        let mut ids = self.prefix_ids(options)?;
        let mut cursor = 0;
        // SentencePiece prepends its dummy whitespace marker once for the
        // complete input. An added token is injected into that stream; it does
        // not start a fresh SentencePiece encoding region after itself.
        let mut prepend_dummy_prefix = true;
        while cursor < input.len() {
            let Some((offset, token)) = self.next_injected_token(&input[cursor..]) else {
                self.encode_normal_text(&input[cursor..], &mut ids, prepend_dummy_prefix)?;
                break;
            };
            let start = cursor + offset;
            self.encode_normal_text(&input[cursor..start], &mut ids, prepend_dummy_prefix)?;
            ids.push(token.id);
            cursor = start + token.content.len();
            prepend_dummy_prefix = false;
        }
        self.suffix_ids(&mut ids, options)?;
        Ok(ids)
    }

    /// Encodes arbitrary bytes. Valid UTF-8 follows ordinary BPE; invalid UTF-8
    /// routes byte-for-byte through the BYTE-piece table.
    pub fn encode_bytes(
        &self,
        input: &[u8],
        options: EncodeOptions,
    ) -> Result<Vec<u32>, EncodeError> {
        match from_utf8(input) {
            Ok(text) => self.encode_ids_with_options(text, options),
            Err(_) => {
                let mut ids = self.prefix_ids(options)?;
                self.encode_byte_fallback(input, &mut ids)?;
                self.suffix_ids(&mut ids, options)?;
                Ok(ids)
            }
        }
    }

    /// Encode every source byte through the BYTE-piece table only.
    ///
    /// This deliberately bypasses all added-token and ordinary-BPE matching.
    /// It is the fail-closed primitive used by the untrusted-document boundary:
    /// callers can prove that marker-looking source text did not become an
    /// injected template-control id, then verify exact byte decoding.
    pub fn encode_byte_fallback_only(&self, input: &[u8]) -> Result<Vec<u32>, EncodeError> {
        let mut ids = Vec::with_capacity(input.len());
        self.encode_byte_fallback(input, &mut ids)?;
        Ok(ids)
    }

    /// Reassembles the exact decoded byte stream. Invalid token ids fail closed.
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, DecodeBytesError> {
        let mut output = Vec::new();
        let mut initial_marker_consumed = false;
        for &id in ids {
            if let Some(surface) = self.injected_by_id.get(&id) {
                output.extend_from_slice(surface.as_bytes());
                initial_marker_consumed = true;
                continue;
            }
            let piece = self.piece_for_id(id)?;
            match piece {
                DecoderPiece::Byte(byte) => {
                    output.push(*byte);
                    initial_marker_consumed = true;
                }
                DecoderPiece::Text {
                    content: _,
                    piece_type: PieceType::Control,
                } => {
                    // SentencePiece control pieces are not rendered by ordinary
                    // decode; registered added-token surfaces are handled above.
                }
                DecoderPiece::Text { content, .. } => {
                    append_sentencepiece_text(&mut output, &mut initial_marker_consumed, content);
                }
            }
        }
        Ok(output)
    }

    /// Decodes only valid UTF-8. No U+FFFD replacement is ever implicit.
    pub fn decode_text(&self, ids: &[u32]) -> Result<String, DecodeTextError> {
        let bytes = self.decode_bytes(ids).map_err(DecodeTextError::Bytes)?;
        String::from_utf8(bytes).map_err(decode_utf8_error)
    }

    /// Explicitly opts into lossy UTF-8 conversion and labels the result.
    pub fn decode_text_lossy(&self, ids: &[u32]) -> Result<LossyText, DecodeBytesError> {
        let bytes = self.decode_bytes(ids)?;
        Ok(match String::from_utf8(bytes) {
            Ok(text) => LossyText {
                text,
                had_invalid_utf8: false,
            },
            Err(error) => LossyText {
                text: String::from_utf8_lossy(error.as_bytes()).into_owned(),
                had_invalid_utf8: true,
            },
        })
    }

    fn prefix_ids(&self, options: EncodeOptions) -> Result<Vec<u32>, EncodeError> {
        let mut ids = Vec::new();
        if options.add_bos {
            ids.push(self.configured_id("bos_id", self.special_ids.bos_id)?);
        }
        Ok(ids)
    }

    fn suffix_ids(&self, ids: &mut Vec<u32>, options: EncodeOptions) -> Result<(), EncodeError> {
        if options.add_eos {
            ids.push(self.configured_id("eos_id", self.special_ids.eos_id)?);
        }
        Ok(())
    }

    fn configured_id(&self, field: &'static str, id: i32) -> Result<u32, EncodeError> {
        let configured_id = id;
        let id = u32::try_from(id)
            .map_err(|_| EncodeError::ConfiguredSpecialIdOutOfRange { field, id })?;
        if usize::try_from(id)
            .ok()
            .is_none_or(|index| index >= self.pieces.len())
        {
            return Err(EncodeError::ConfiguredSpecialIdOutOfRange {
                field,
                id: configured_id,
            });
        }
        Ok(id)
    }

    fn next_injected_token<'a>(&'a self, input: &str) -> Option<(usize, &'a AddedToken)> {
        self.injected_by_text
            .iter()
            .filter_map(|token| input.find(&token.content).map(|offset| (offset, token)))
            .min_by(|(left_offset, left), (right_offset, right)| {
                left_offset
                    .cmp(right_offset)
                    .then_with(|| right.content.len().cmp(&left.content.len()))
                    .then_with(|| left.id.cmp(&right.id))
            })
    }

    fn encode_normal_text(
        &self,
        input: &str,
        ids: &mut Vec<u32>,
        prepend_dummy_prefix: bool,
    ) -> Result<(), EncodeError> {
        if input.is_empty() {
            return Ok(());
        }
        let normalized = sentencepiece_whitespace(input, prepend_dummy_prefix);
        let mut symbols: Vec<Symbol> = normalized
            .chars()
            .map(|character| Symbol {
                text: character.to_string(),
                next: None,
                alive: true,
            })
            .collect();
        for index in 0..symbols.len().saturating_sub(1) {
            symbols[index].next = Some(index + 1);
        }
        self.merge_symbols(&mut symbols);

        let mut current = Some(0);
        while let Some(index) = current {
            let symbol = &symbols[index];
            if symbol.alive {
                if let Some(piece) = self.normal_pieces.get(&symbol.text) {
                    ids.push(piece.id);
                } else {
                    self.encode_byte_fallback(symbol.text.as_bytes(), ids)?;
                }
            }
            current = symbol.next;
        }
        Ok(())
    }

    fn merge_symbols(&self, symbols: &mut [Symbol]) {
        loop {
            let mut best: Option<(usize, usize, f32, String)> = None;
            let mut left = Some(0);
            while let Some(left_index) = left {
                let Some(right_index) = symbols[left_index].next else {
                    break;
                };
                let joined = format!("{}{}", symbols[left_index].text, symbols[right_index].text);
                if let Some(piece) = self.normal_pieces.get(&joined) {
                    let should_replace = best
                        .as_ref()
                        .is_none_or(|(_, _, score, _)| piece.score.total_cmp(score).is_gt());
                    if should_replace {
                        best = Some((left_index, right_index, piece.score, joined));
                    }
                }
                left = symbols[left_index].next;
            }

            let Some((left_index, right_index, _, joined)) = best else {
                return;
            };
            symbols[left_index].text = joined;
            symbols[left_index].next = symbols[right_index].next;
            symbols[right_index].alive = false;
            symbols[right_index].next = None;
        }
    }

    fn encode_byte_fallback(&self, bytes: &[u8], ids: &mut Vec<u32>) -> Result<(), EncodeError> {
        for &byte in bytes {
            let id = self.byte_pieces[usize::from(byte)]
                .ok_or(EncodeError::ByteFallbackMissing { byte })?;
            ids.push(id);
        }
        Ok(())
    }

    fn piece_for_id(&self, id: u32) -> Result<&DecoderPiece, DecodeBytesError> {
        usize::try_from(id)
            .ok()
            .and_then(|index| self.pieces.get(index))
            .ok_or(DecodeBytesError::UnknownTokenId {
                id,
                piece_count: self.pieces.len(),
            })
    }
}

fn insert_injected(
    injected: &mut BTreeMap<String, u32>,
    injected_by_id: &mut BTreeMap<u32, String>,
    token: AddedToken,
) -> Result<(), BpeBuildError> {
    if token.content.is_empty() {
        return Err(BpeBuildError::EmptyAddedToken);
    }
    if let Some(existing) = injected.get(&token.content) {
        if *existing != token.id {
            return Err(BpeBuildError::ConflictingAddedToken {
                surface: token.content,
            });
        }
        return Ok(());
    }
    if let Some(existing) = injected_by_id.get(&token.id) {
        if existing != &token.content {
            return Err(BpeBuildError::ConflictingAddedTokenId { id: token.id });
        }
        return Ok(());
    }
    injected_by_id.insert(token.id, token.content.clone());
    injected.insert(token.content, token.id);
    Ok(())
}

fn parse_byte_piece(surface: &str) -> Option<u8> {
    let bytes = surface.as_bytes();
    if bytes.len() != 6 || &bytes[..3] != b"<0x" || bytes[5] != b'>' {
        return None;
    }
    Some((hex_value(bytes[3])? << 4) | hex_value(bytes[4])?)
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn sentencepiece_whitespace(input: &str, prepend_dummy_prefix: bool) -> String {
    let mut normalized = String::new();
    if prepend_dummy_prefix {
        normalized.push(WHITESPACE_MARKER);
    }
    for character in input.chars() {
        if character == ' ' {
            normalized.push(WHITESPACE_MARKER);
        } else {
            normalized.push(character);
        }
    }
    normalized
}

fn append_sentencepiece_text(
    output: &mut Vec<u8>,
    initial_marker_consumed: &mut bool,
    piece: &str,
) {
    for character in piece.chars() {
        if character == WHITESPACE_MARKER {
            if *initial_marker_consumed {
                output.push(b' ');
            } else {
                // SentencePiece's dummy prefix is precisely the first marker
                // in the complete decoded stream. A second leading marker is
                // real source whitespace and therefore emits an ASCII space.
                *initial_marker_consumed = true;
            }
        } else {
            let mut buffer = [0_u8; 4];
            output.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
            *initial_marker_consumed = true;
        }
    }
}

fn decode_utf8_error(error: std::string::FromUtf8Error) -> DecodeTextError {
    let utf8_error: Utf8Error = error.utf8_error();
    DecodeTextError::InvalidUtf8 {
        valid_up_to: utf8_error.valid_up_to(),
        error_len: utf8_error.error_len(),
    }
}
