//! Dependency-free, fail-closed reader for the SentencePiece `ModelProto` subset.
//!
//! Accepted fields are deliberately closed: top-level `pieces`, `trainer_spec`,
//! and `normalizer_spec`; each piece's `piece`, `score`, and `type`; and the
//! trainer's `model_type`, `unk_id`, `bos_id`, `eos_id`, and `pad_id`. Unknown
//! fields are skipped only for protobuf's four non-group wire types. Groups,
//! malformed lengths, conflicting singular fields, and data beyond a nested
//! message boundary are errors.
//!
//! The limits below are part of the parser contract and fuzz target: no varint
//! exceeds ten bytes, messages never exceed [`MAX_MESSAGE_NESTING`], and no
//! untrusted length can reserve memory before its checked allocation charge.

use std::{error::Error, fmt, mem::size_of};

/// A protobuf varint cannot be longer than ten bytes for a `u64` value.
pub const MAX_VARINT_BYTES: usize = 10;
/// The supported ModelProto structure needs only two nesting levels; retain a
/// small explicit ceiling for future accepted nested messages.
pub const MAX_MESSAGE_NESTING: usize = 16;
/// The pinned tokenizer has 166,144 pieces. This leaves substantial headroom.
pub const MAX_PIECE_COUNT: usize = 2_000_000;
/// A tokenizer piece is normally tiny; this prevents hostile string prefixes.
pub const MAX_PIECE_STRING_BYTES: usize = 16 * 1024;
/// The pinned model is approximately 2.8 MiB, not a general-purpose container.
pub const MAX_TOTAL_INPUT_BYTES: usize = 128 * 1024 * 1024;
/// Conservative accounting for piece structs and owned strings, charged before
/// allocating each owned value.
pub const MAX_TOTAL_ALLOCATION_BYTES: usize = 256 * 1024 * 1024;

const MODEL_FIELD_PIECES: u32 = 1;
const MODEL_FIELD_TRAINER_SPEC: u32 = 2;
const MODEL_FIELD_NORMALIZER_SPEC: u32 = 3;

const PIECE_FIELD_TEXT: u32 = 1;
const PIECE_FIELD_SCORE: u32 = 2;
const PIECE_FIELD_TYPE: u32 = 3;

const TRAINER_FIELD_MODEL_TYPE: u32 = 3;
const TRAINER_FIELD_UNK_ID: u32 = 40;
const TRAINER_FIELD_BOS_ID: u32 = 41;
const TRAINER_FIELD_EOS_ID: u32 = 42;
const TRAINER_FIELD_PAD_ID: u32 = 43;

const NORMALIZER_FIELD_NAME: u32 = 1;

const WIRE_VARINT: u8 = 0;
const WIRE_FIXED64: u8 = 1;
const WIRE_LENGTH_DELIMITED: u8 = 2;
const WIRE_START_GROUP: u8 = 3;
const WIRE_END_GROUP: u8 = 4;
const WIRE_FIXED32: u8 = 5;

/// The SentencePiece model kind accepted by this one-model tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// SentencePiece `BPE = 2`.
    Bpe,
}

/// SentencePiece's closed piece-type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl PieceType {
    fn from_proto(value: u64, context: Context) -> Result<Self, SpmError> {
        match value {
            1 => Ok(Self::Normal),
            2 => Ok(Self::Unknown),
            3 => Ok(Self::Control),
            4 => Ok(Self::UserDefined),
            5 => Ok(Self::Unused),
            6 => Ok(Self::Byte),
            _ => Err(context.error(SpmErrorKind::InvalidEnum {
                enum_name: "SentencePiece.Type",
                value,
            })),
        }
    }
}

/// A single SentencePiece entry. `score` preserves protobuf's f32 bit pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct SpmPiece {
    pub piece: String,
    pub score: f32,
    pub piece_type: PieceType,
}

/// The trainer-owned ids used by downstream special-token validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialPieceIds {
    pub unk_id: i32,
    pub bos_id: i32,
    pub eos_id: i32,
    pub pad_id: i32,
}

/// Facts required to prove that the tokenizer uses the identity normalizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizerFacts {
    pub name: String,
    pub is_identity: bool,
}

/// The bounded semantic projection of a SentencePiece ModelProto.
#[derive(Debug, Clone, PartialEq)]
pub struct SpmModel {
    pub pieces: Vec<SpmPiece>,
    pub model_type: ModelType,
    pub normalizer: NormalizerFacts,
    pub special_ids: SpecialPieceIds,
}

/// A machine-actionable parse failure with exact protobuf context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpmError {
    pub offset: usize,
    pub field_number: Option<u32>,
    pub wire_type: Option<u8>,
    pub kind: SpmErrorKind,
}

/// Stable, typed rejection categories for the constrained protobuf surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpmErrorKind {
    InputTooLarge {
        limit: usize,
        actual: usize,
    },
    VarintUnterminated,
    VarintTooLong {
        limit: usize,
    },
    VarintOverflow,
    LengthOverflow,
    Truncated {
        needed: usize,
        remaining: usize,
    },
    InvalidFieldNumber,
    InvalidWireType {
        wire_type: u8,
    },
    GroupWireTypeUnsupported,
    LimitExceeded {
        limit_name: &'static str,
        limit: usize,
    },
    DuplicateSingular {
        field_name: &'static str,
    },
    InvalidUtf8 {
        field_name: &'static str,
    },
    InvalidEnum {
        enum_name: &'static str,
        value: u64,
    },
    InvalidInt32 {
        field_name: &'static str,
        value: u64,
    },
    MissingRequiredField {
        field_name: &'static str,
    },
    UnsupportedModelType {
        value: i32,
    },
    NonIdentityNormalizer {
        name: String,
    },
}

impl fmt::Display for SpmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "SPM_PROTO offset={}", self.offset)?;
        if let Some(field_number) = self.field_number {
            write!(formatter, " field={field_number}")?;
        }
        if let Some(wire_type) = self.wire_type {
            write!(formatter, " wire={wire_type}")?;
        }
        match &self.kind {
            SpmErrorKind::InputTooLarge { limit, actual } => {
                write!(
                    formatter,
                    " limit=MAX_TOTAL_INPUT_BYTES actual={actual} max={limit}"
                )
            }
            SpmErrorKind::VarintUnterminated => write!(formatter, " error=unterminated-varint"),
            SpmErrorKind::VarintTooLong { limit } => {
                write!(formatter, " limit=MAX_VARINT_BYTES max={limit}")
            }
            SpmErrorKind::VarintOverflow => write!(formatter, " error=varint-overflow"),
            SpmErrorKind::LengthOverflow => write!(formatter, " error=length-overflow"),
            SpmErrorKind::Truncated { needed, remaining } => {
                write!(
                    formatter,
                    " error=truncated needed={needed} remaining={remaining}"
                )
            }
            SpmErrorKind::InvalidFieldNumber => write!(formatter, " error=invalid-field-number"),
            SpmErrorKind::InvalidWireType { wire_type } => {
                write!(formatter, " error=invalid-wire-type value={wire_type}")
            }
            SpmErrorKind::GroupWireTypeUnsupported => {
                write!(formatter, " error=group-wire-type-unsupported")
            }
            SpmErrorKind::LimitExceeded { limit_name, limit } => {
                write!(formatter, " limit={limit_name} max={limit}")
            }
            SpmErrorKind::DuplicateSingular { field_name } => {
                write!(
                    formatter,
                    " error=conflicting-duplicate field_name={field_name}"
                )
            }
            SpmErrorKind::InvalidUtf8 { field_name } => {
                write!(formatter, " error=invalid-utf8 field_name={field_name}")
            }
            SpmErrorKind::InvalidEnum { enum_name, value } => {
                write!(
                    formatter,
                    " error=invalid-enum enum={enum_name} value={value}"
                )
            }
            SpmErrorKind::InvalidInt32 { field_name, value } => {
                write!(
                    formatter,
                    " error=invalid-int32 field_name={field_name} value={value}"
                )
            }
            SpmErrorKind::MissingRequiredField { field_name } => {
                write!(formatter, " error=missing-required field_name={field_name}")
            }
            SpmErrorKind::UnsupportedModelType { value } => {
                write!(formatter, " error=unsupported-model-type value={value}")
            }
            SpmErrorKind::NonIdentityNormalizer { name } => {
                write!(formatter, " error=non-identity-normalizer name={name:?}")
            }
        }
    }
}

impl Error for SpmError {}

/// Parses the exact SentencePiece protobuf subset accepted by FrankenNLP.
///
/// The parser is total for byte strings: hostile input returns [`SpmError`]
/// before allocation or indexing can exceed the declared limits.
pub fn parse_spm_model(input: &[u8]) -> Result<SpmModel, SpmError> {
    ensure_input_length(input.len())?;

    let mut allocation = 0_usize;
    let mut reader = Reader::new(input, 0);
    let mut pieces = Vec::new();
    let mut trainer_payload: Option<(&[u8], usize)> = None;
    let mut normalizer_payload: Option<(&[u8], usize)> = None;

    while !reader.is_finished() {
        let context = reader.read_key()?;
        match context.field_number {
            MODEL_FIELD_PIECES => {
                reader.require_wire(context, WIRE_LENGTH_DELIMITED)?;
                let (payload, payload_offset) =
                    reader.read_length_delimited_with_offset(context)?;
                ensure_piece_room(pieces.len(), context)?;
                let piece = parse_piece(payload, payload_offset, 1, &mut allocation)?;
                charge_allocation(&mut allocation, size_of::<SpmPiece>(), context)?;
                pieces.push(piece);
            }
            MODEL_FIELD_TRAINER_SPEC => {
                reader.require_wire(context, WIRE_LENGTH_DELIMITED)?;
                let (payload, payload_offset) =
                    reader.read_length_delimited_with_offset(context)?;
                set_payload_once(
                    &mut trainer_payload,
                    (payload, payload_offset),
                    "trainer_spec",
                    context,
                )?;
            }
            MODEL_FIELD_NORMALIZER_SPEC => {
                reader.require_wire(context, WIRE_LENGTH_DELIMITED)?;
                let (payload, payload_offset) =
                    reader.read_length_delimited_with_offset(context)?;
                set_payload_once(
                    &mut normalizer_payload,
                    (payload, payload_offset),
                    "normalizer_spec",
                    context,
                )?;
            }
            _ => reader.skip_field(context)?,
        }
    }

    let trainer_payload = trainer_payload.ok_or_else(|| missing("trainer_spec", input.len()))?;
    let normalizer_payload =
        normalizer_payload.ok_or_else(|| missing("normalizer_spec", input.len()))?;
    let (model_type, special_ids) = parse_trainer(trainer_payload.0, trainer_payload.1, 1)?;
    let normalizer = parse_normalizer(
        normalizer_payload.0,
        normalizer_payload.1,
        1,
        &mut allocation,
    )?;

    Ok(SpmModel {
        pieces,
        model_type,
        normalizer,
        special_ids,
    })
}

fn parse_piece(
    input: &[u8],
    base_offset: usize,
    depth: usize,
    allocation: &mut usize,
) -> Result<SpmPiece, SpmError> {
    ensure_depth(base_offset, depth)?;
    let mut reader = Reader::new(input, base_offset);
    let mut piece: Option<String> = None;
    let mut score_bits: Option<u32> = None;
    let mut piece_type: Option<PieceType> = None;

    while !reader.is_finished() {
        let context = reader.read_key()?;
        match context.field_number {
            PIECE_FIELD_TEXT => {
                reader.require_wire(context, WIRE_LENGTH_DELIMITED)?;
                let value = reader.read_bounded_string(
                    context,
                    "piece.piece",
                    MAX_PIECE_STRING_BYTES,
                    allocation,
                )?;
                set_once(&mut piece, value, "piece.piece", context)?;
            }
            PIECE_FIELD_SCORE => {
                reader.require_wire(context, WIRE_FIXED32)?;
                let value = reader.read_fixed32(context)?;
                set_once(&mut score_bits, value, "piece.score", context)?;
            }
            PIECE_FIELD_TYPE => {
                reader.require_wire(context, WIRE_VARINT)?;
                let value = PieceType::from_proto(reader.read_varint(context)?, context)?;
                set_once(&mut piece_type, value, "piece.type", context)?;
            }
            _ => reader.skip_field(context)?,
        }
    }

    let piece = piece.ok_or_else(|| missing("piece.piece", base_offset + input.len()))?;
    Ok(SpmPiece {
        piece,
        score: f32::from_bits(score_bits.unwrap_or(0)),
        piece_type: piece_type.unwrap_or(PieceType::Normal),
    })
}

fn parse_trainer(
    input: &[u8],
    base_offset: usize,
    depth: usize,
) -> Result<(ModelType, SpecialPieceIds), SpmError> {
    ensure_depth(base_offset, depth)?;
    let mut reader = Reader::new(input, base_offset);
    let mut model_type: Option<i32> = None;
    let mut unk_id: Option<i32> = None;
    let mut bos_id: Option<i32> = None;
    let mut eos_id: Option<i32> = None;
    let mut pad_id: Option<i32> = None;
    let mut model_type_context: Option<Context> = None;

    while !reader.is_finished() {
        let context = reader.read_key()?;
        match context.field_number {
            TRAINER_FIELD_MODEL_TYPE => {
                reader.require_wire(context, WIRE_VARINT)?;
                let value = parse_i32(
                    reader.read_varint(context)?,
                    "trainer_spec.model_type",
                    context,
                )?;
                set_once(&mut model_type, value, "trainer_spec.model_type", context)?;
                model_type_context = Some(context);
            }
            TRAINER_FIELD_UNK_ID => {
                reader.require_wire(context, WIRE_VARINT)?;
                let value =
                    parse_i32(reader.read_varint(context)?, "trainer_spec.unk_id", context)?;
                set_once(&mut unk_id, value, "trainer_spec.unk_id", context)?;
            }
            TRAINER_FIELD_BOS_ID => {
                reader.require_wire(context, WIRE_VARINT)?;
                let value =
                    parse_i32(reader.read_varint(context)?, "trainer_spec.bos_id", context)?;
                set_once(&mut bos_id, value, "trainer_spec.bos_id", context)?;
            }
            TRAINER_FIELD_EOS_ID => {
                reader.require_wire(context, WIRE_VARINT)?;
                let value =
                    parse_i32(reader.read_varint(context)?, "trainer_spec.eos_id", context)?;
                set_once(&mut eos_id, value, "trainer_spec.eos_id", context)?;
            }
            TRAINER_FIELD_PAD_ID => {
                reader.require_wire(context, WIRE_VARINT)?;
                let value =
                    parse_i32(reader.read_varint(context)?, "trainer_spec.pad_id", context)?;
                set_once(&mut pad_id, value, "trainer_spec.pad_id", context)?;
            }
            _ => reader.skip_field(context)?,
        }
    }

    let model_type =
        model_type.ok_or_else(|| missing("trainer_spec.model_type", base_offset + input.len()))?;
    if model_type != 2 {
        let context = model_type_context
            .ok_or_else(|| missing("trainer_spec.model_type", base_offset + input.len()))?;
        return Err(context.error(SpmErrorKind::UnsupportedModelType { value: model_type }));
    }
    Ok((
        ModelType::Bpe,
        SpecialPieceIds {
            unk_id: unk_id.unwrap_or(0),
            bos_id: bos_id.unwrap_or(1),
            eos_id: eos_id.unwrap_or(2),
            pad_id: pad_id.unwrap_or(-1),
        },
    ))
}

fn parse_normalizer(
    input: &[u8],
    base_offset: usize,
    depth: usize,
    allocation: &mut usize,
) -> Result<NormalizerFacts, SpmError> {
    ensure_depth(base_offset, depth)?;
    let mut reader = Reader::new(input, base_offset);
    let mut name: Option<String> = None;
    let mut name_context: Option<Context> = None;
    while !reader.is_finished() {
        let context = reader.read_key()?;
        match context.field_number {
            NORMALIZER_FIELD_NAME => {
                reader.require_wire(context, WIRE_LENGTH_DELIMITED)?;
                let value = reader.read_bounded_string(
                    context,
                    "normalizer_spec.name",
                    MAX_PIECE_STRING_BYTES,
                    allocation,
                )?;
                set_once(&mut name, value, "normalizer_spec.name", context)?;
                name_context = Some(context);
            }
            _ => reader.skip_field(context)?,
        }
    }
    let name = name.ok_or_else(|| missing("normalizer_spec.name", base_offset + input.len()))?;
    if name != "identity" {
        let context = name_context
            .ok_or_else(|| missing("normalizer_spec.name", base_offset + input.len()))?;
        return Err(context.error(SpmErrorKind::NonIdentityNormalizer { name }));
    }
    Ok(NormalizerFacts {
        name,
        is_identity: true,
    })
}

fn parse_i32(value: u64, field_name: &'static str, context: Context) -> Result<i32, SpmError> {
    let signed = value as i64;
    i32::try_from(signed)
        .map_err(|_| context.error(SpmErrorKind::InvalidInt32 { field_name, value }))
}

fn ensure_depth(offset: usize, depth: usize) -> Result<(), SpmError> {
    if depth > MAX_MESSAGE_NESTING {
        return Err(SpmError {
            offset,
            field_number: None,
            wire_type: None,
            kind: SpmErrorKind::LimitExceeded {
                limit_name: "MAX_MESSAGE_NESTING",
                limit: MAX_MESSAGE_NESTING,
            },
        });
    }
    Ok(())
}

fn ensure_input_length(input_length: usize) -> Result<(), SpmError> {
    if input_length > MAX_TOTAL_INPUT_BYTES {
        return Err(SpmError {
            offset: 0,
            field_number: None,
            wire_type: None,
            kind: SpmErrorKind::InputTooLarge {
                limit: MAX_TOTAL_INPUT_BYTES,
                actual: input_length,
            },
        });
    }
    Ok(())
}

fn ensure_piece_room(piece_count: usize, context: Context) -> Result<(), SpmError> {
    if piece_count >= MAX_PIECE_COUNT {
        return Err(context.error(SpmErrorKind::LimitExceeded {
            limit_name: "MAX_PIECE_COUNT",
            limit: MAX_PIECE_COUNT,
        }));
    }
    Ok(())
}

fn charge_allocation(
    allocation: &mut usize,
    amount: usize,
    context: Context,
) -> Result<(), SpmError> {
    let next = allocation.checked_add(amount).ok_or_else(|| {
        context.error(SpmErrorKind::LimitExceeded {
            limit_name: "MAX_TOTAL_ALLOCATION_BYTES",
            limit: MAX_TOTAL_ALLOCATION_BYTES,
        })
    })?;
    if next > MAX_TOTAL_ALLOCATION_BYTES {
        return Err(context.error(SpmErrorKind::LimitExceeded {
            limit_name: "MAX_TOTAL_ALLOCATION_BYTES",
            limit: MAX_TOTAL_ALLOCATION_BYTES,
        }));
    }
    *allocation = next;
    Ok(())
}

fn set_once<T: PartialEq>(
    slot: &mut Option<T>,
    value: T,
    field_name: &'static str,
    context: Context,
) -> Result<(), SpmError> {
    if let Some(existing) = slot {
        if *existing != value {
            return Err(context.error(SpmErrorKind::DuplicateSingular { field_name }));
        }
        return Ok(());
    }
    *slot = Some(value);
    Ok(())
}

fn set_payload_once<'a>(
    slot: &mut Option<(&'a [u8], usize)>,
    value: (&'a [u8], usize),
    field_name: &'static str,
    context: Context,
) -> Result<(), SpmError> {
    if let Some(existing) = slot {
        if existing.0 != value.0 {
            return Err(context.error(SpmErrorKind::DuplicateSingular { field_name }));
        }
        return Ok(());
    }
    *slot = Some(value);
    Ok(())
}

fn missing(field_name: &'static str, offset: usize) -> SpmError {
    SpmError {
        offset,
        field_number: None,
        wire_type: None,
        kind: SpmErrorKind::MissingRequiredField { field_name },
    }
}

#[derive(Clone, Copy)]
struct Context {
    offset: usize,
    field_number: u32,
    wire_type: u8,
}

impl Context {
    fn error(self, kind: SpmErrorKind) -> SpmError {
        SpmError {
            offset: self.offset,
            field_number: Some(self.field_number),
            wire_type: Some(self.wire_type),
            kind,
        }
    }
}

struct Reader<'a> {
    input: &'a [u8],
    position: usize,
    base_offset: usize,
}

impl<'a> Reader<'a> {
    fn new(input: &'a [u8], base_offset: usize) -> Self {
        Self {
            input,
            position: 0,
            base_offset,
        }
    }

    fn is_finished(&self) -> bool {
        self.position == self.input.len()
    }

    fn read_key(&mut self) -> Result<Context, SpmError> {
        let offset = self.absolute_offset();
        let key = self.read_raw_varint(None)?;
        let field_number = u32::try_from(key >> 3).map_err(|_| SpmError {
            offset,
            field_number: None,
            wire_type: None,
            kind: SpmErrorKind::InvalidFieldNumber,
        })?;
        let wire_type = (key & 0b111) as u8;
        if field_number == 0 {
            return Err(SpmError {
                offset,
                field_number: None,
                wire_type: Some(wire_type),
                kind: SpmErrorKind::InvalidFieldNumber,
            });
        }
        if wire_type > WIRE_FIXED32 {
            return Err(SpmError {
                offset,
                field_number: Some(field_number),
                wire_type: Some(wire_type),
                kind: SpmErrorKind::InvalidWireType { wire_type },
            });
        }
        Ok(Context {
            offset,
            field_number,
            wire_type,
        })
    }

    fn require_wire(&self, context: Context, expected: u8) -> Result<(), SpmError> {
        if context.wire_type == expected {
            Ok(())
        } else if matches!(context.wire_type, WIRE_START_GROUP | WIRE_END_GROUP) {
            Err(context.error(SpmErrorKind::GroupWireTypeUnsupported))
        } else {
            Err(context.error(SpmErrorKind::InvalidWireType {
                wire_type: context.wire_type,
            }))
        }
    }

    fn read_varint(&mut self, context: Context) -> Result<u64, SpmError> {
        self.read_raw_varint(Some(context))
    }

    fn read_raw_varint(&mut self, context: Option<Context>) -> Result<u64, SpmError> {
        let start = self.absolute_offset();
        let mut value = 0_u64;
        for shift_index in 0..MAX_VARINT_BYTES {
            let byte = self.read_byte(context)?;
            if shift_index == MAX_VARINT_BYTES - 1 && byte > 1 {
                return Err(self.error_at(context, start, SpmErrorKind::VarintOverflow));
            }
            value |= u64::from(byte & 0x7f) << (shift_index * 7);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(self.error_at(
            context,
            start,
            SpmErrorKind::VarintTooLong {
                limit: MAX_VARINT_BYTES,
            },
        ))
    }

    fn read_length_delimited(&mut self, context: Context) -> Result<&'a [u8], SpmError> {
        self.read_length_delimited_with_offset(context)
            .map(|(bytes, _)| bytes)
    }

    fn read_length_delimited_with_offset(
        &mut self,
        context: Context,
    ) -> Result<(&'a [u8], usize), SpmError> {
        let length_offset = self.absolute_offset();
        let length = self.read_varint(context)?;
        let length =
            usize::try_from(length).map_err(|_| context.error(SpmErrorKind::LengthOverflow))?;
        let end = self.position.checked_add(length).ok_or_else(|| {
            self.error_at(Some(context), length_offset, SpmErrorKind::LengthOverflow)
        })?;
        if end > self.input.len() {
            return Err(context.error(SpmErrorKind::Truncated {
                needed: length,
                remaining: self.input.len().saturating_sub(self.position),
            }));
        }
        let payload_offset = self.absolute_offset();
        let result = &self.input[self.position..end];
        self.position = end;
        Ok((result, payload_offset))
    }

    fn read_bounded_string(
        &mut self,
        context: Context,
        field_name: &'static str,
        max_length: usize,
        allocation: &mut usize,
    ) -> Result<String, SpmError> {
        let bytes = self.read_length_delimited(context)?;
        if bytes.len() > max_length {
            return Err(context.error(SpmErrorKind::LimitExceeded {
                limit_name: "MAX_PIECE_STRING_BYTES",
                limit: max_length,
            }));
        }
        charge_allocation(allocation, bytes.len(), context)?;
        let value = std::str::from_utf8(bytes)
            .map_err(|_| context.error(SpmErrorKind::InvalidUtf8 { field_name }))?;
        Ok(value.to_owned())
    }

    fn read_fixed32(&mut self, context: Context) -> Result<u32, SpmError> {
        let bytes = self.read_exact(4, context)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn skip_field(&mut self, context: Context) -> Result<(), SpmError> {
        match context.wire_type {
            WIRE_VARINT => self.read_varint(context).map(|_| ()),
            WIRE_FIXED64 => self.read_exact(8, context).map(|_| ()),
            WIRE_LENGTH_DELIMITED => self.read_length_delimited(context).map(|_| ()),
            WIRE_FIXED32 => self.read_exact(4, context).map(|_| ()),
            WIRE_START_GROUP | WIRE_END_GROUP => {
                Err(context.error(SpmErrorKind::GroupWireTypeUnsupported))
            }
            wire_type => Err(context.error(SpmErrorKind::InvalidWireType { wire_type })),
        }
    }

    fn read_byte(&mut self, context: Option<Context>) -> Result<u8, SpmError> {
        if let Some(byte) = self.input.get(self.position).copied() {
            self.position += 1;
            Ok(byte)
        } else {
            Err(self.error_at(
                context,
                self.absolute_offset(),
                SpmErrorKind::VarintUnterminated,
            ))
        }
    }

    fn read_exact(&mut self, length: usize, context: Context) -> Result<&'a [u8], SpmError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| context.error(SpmErrorKind::LengthOverflow))?;
        if end > self.input.len() {
            return Err(context.error(SpmErrorKind::Truncated {
                needed: length,
                remaining: self.input.len().saturating_sub(self.position),
            }));
        }
        let result = &self.input[self.position..end];
        self.position = end;
        Ok(result)
    }

    fn absolute_offset(&self) -> usize {
        self.base_offset.saturating_add(self.position)
    }

    fn error_at(&self, context: Option<Context>, offset: usize, kind: SpmErrorKind) -> SpmError {
        match context {
            Some(context) => context.error(kind),
            None => SpmError {
                offset,
                field_number: None,
                wire_type: None,
                kind,
            },
        }
    }
}

#[cfg(test)]
mod limit_tests {
    use super::*;

    fn context() -> Context {
        Context {
            offset: 11,
            field_number: 7,
            wire_type: WIRE_LENGTH_DELIMITED,
        }
    }

    #[test]
    fn rejects_declared_limit_boundaries_without_constructing_hostile_inputs() {
        assert!(matches!(
            ensure_depth(0, MAX_MESSAGE_NESTING + 1),
            Err(SpmError {
                kind: SpmErrorKind::LimitExceeded {
                    limit_name: "MAX_MESSAGE_NESTING",
                    ..
                },
                ..
            })
        ));
        assert!(matches!(
            ensure_input_length(MAX_TOTAL_INPUT_BYTES + 1),
            Err(SpmError {
                kind: SpmErrorKind::InputTooLarge { .. },
                ..
            })
        ));
        assert!(matches!(
            ensure_piece_room(MAX_PIECE_COUNT, context()),
            Err(SpmError {
                kind: SpmErrorKind::LimitExceeded {
                    limit_name: "MAX_PIECE_COUNT",
                    ..
                },
                ..
            })
        ));
        let mut allocation = MAX_TOTAL_ALLOCATION_BYTES;
        assert!(matches!(
            charge_allocation(&mut allocation, 1, context()),
            Err(SpmError {
                kind: SpmErrorKind::LimitExceeded {
                    limit_name: "MAX_TOTAL_ALLOCATION_BYTES",
                    ..
                },
                ..
            })
        ));
    }
}
