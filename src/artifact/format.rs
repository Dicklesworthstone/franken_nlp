//! Canonical `.fnlpq` envelope v1 writing.
//!
//! The writer deliberately takes typed input only.  It never accepts a
//! `serde_json::Value` or a caller-provided header: the prelude, directory,
//! header order, padding, and domain-framed digests are all constructed here.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::canonjson;

/// The only v1 file signature.
pub const MAGIC: [u8; 8] = *b"FNLPQ\0\0\x01";
/// The v1 envelope version encoded in the prelude.
pub const FORMAT_VERSION: u32 = 1;
/// Bytes in the fixed prelude.
pub const PRELUDE_BYTES: usize = 80;
/// Bytes in one fixed section-directory entry.
pub const SECTION_DIRECTORY_ENTRY_BYTES: usize = 80;
/// The generic representation that retains canonical BF16 tensor bytes
/// without quantization.
pub const BF16_VERBATIM_V1: &str = "bf16-verbatim-v1";
/// Maximum accepted canonical-header size in bytes.
pub const MAX_HEADER_BYTES: u64 = 1_048_576;
/// Maximum v1 directory entries and logical tensor declarations.
pub const MAX_ENTRIES: usize = 4_096;
/// Maximum v1 file size (64 GiB).
pub const MAX_FILE_BYTES: u64 = 68_719_476_736;
/// Largest accepted v1 section alignment.
pub const MAX_ALIGNMENT: u64 = 4_096;

const REQUIRED_SECTION_KINDS: [SectionKind; 8] = [
    SectionKind::GenericTensorPayload,
    SectionKind::GenericTensorScales,
    SectionKind::GenericTensorRowSums,
    SectionKind::TokenizerModel,
    SectionKind::ModelConfig,
    SectionKind::TokenizerConfig,
    SectionKind::ChatTemplate,
    SectionKind::LicenseBundle,
];

const MATERIALIZED_SOURCE_KINDS: [(&str, SectionKind); 4] = [
    ("model_config", SectionKind::ModelConfig),
    ("tokenizer_model", SectionKind::TokenizerModel),
    ("tokenizer_config", SectionKind::TokenizerConfig),
    ("chat_template", SectionKind::ChatTemplate),
];

/// The fixed v1 section-kind namespace.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SectionKind {
    /// Canonical generic tensor representation bytes.
    GenericTensorPayload = 1,
    /// Per-group scales for the generic representation.
    GenericTensorScales = 2,
    /// Per-row correction metadata for the generic representation.
    GenericTensorRowSums = 3,
    /// The exact upstream `tokenizer.model` bytes.
    TokenizerModel = 4,
    /// Exact materialized model configuration bytes.
    ModelConfig = 5,
    /// Exact materialized tokenizer configuration bytes.
    TokenizerConfig = 6,
    /// Exact materialized chat-template bytes.
    ChatTemplate = 7,
    /// Apache-2.0 text, attribution, and modification notice bundle.
    LicenseBundle = 8,
    /// A target-specific packing payload declared by a packing set.
    NativePackingPayload = 9,
}

impl SectionKind {
    /// Directory integer for this kind.
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// Closed-schema header enum spelling.
    pub const fn header_name(self) -> &'static str {
        match self {
            Self::GenericTensorPayload => "GENERIC_TENSOR_PAYLOAD",
            Self::GenericTensorScales => "GENERIC_TENSOR_SCALES",
            Self::GenericTensorRowSums => "GENERIC_TENSOR_ROW_SUMS",
            Self::TokenizerModel => "TOKENIZER_MODEL",
            Self::ModelConfig => "MODEL_CONFIG",
            Self::TokenizerConfig => "TOKENIZER_CONFIG",
            Self::ChatTemplate => "CHAT_TEMPLATE",
            Self::LicenseBundle => "LICENSE_BUNDLE",
            Self::NativePackingPayload => "NATIVE_PACKING_PAYLOAD",
        }
    }

    const fn required_singleton(self) -> bool {
        !matches!(self, Self::NativePackingPayload)
    }

    const fn section_cap(self) -> u64 {
        match self {
            Self::TokenizerModel => 64 * 1024 * 1024,
            Self::ModelConfig | Self::TokenizerConfig => 16 * 1024 * 1024,
            Self::ChatTemplate => 8 * 1024 * 1024,
            Self::LicenseBundle => 1024 * 1024,
            _ => MAX_FILE_BYTES,
        }
    }
}

/// Dispatch target names allowed by the v1 packing-set schema.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ArchTarget {
    /// Portable representation.
    Generic,
    /// AArch64 SDOT representation.
    Aarch64Sdot,
    /// AArch64 I8MM representation.
    Aarch64I8mm,
    /// x86 VNNI with 256-bit vectors.
    X86Vnni256,
    /// x86 VNNI with 512-bit vectors.
    X86Vnni512,
    /// x86 AVX2 representation.
    X86Avx2,
}

impl ArchTarget {
    /// Closed-schema target spelling.
    pub const fn header_name(self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::Aarch64Sdot => "aarch64-sdot",
            Self::Aarch64I8mm => "aarch64-i8mm",
            Self::X86Vnni256 => "x86-vnni-256",
            Self::X86Vnni512 => "x86-vnni-512",
            Self::X86Avx2 => "x86-avx2",
        }
    }
}

/// Canonical logical tensor element type.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CanonicalDtype {
    /// Verbatim bfloat16 bytes.  This is required for high-precision tensors.
    Bf16,
    /// IEEE-754 binary32 logical tensor bytes.
    F32,
    /// Signed 8-bit logical tensor bytes.
    I8,
}

impl CanonicalDtype {
    /// Closed-schema dtype spelling.
    pub const fn header_name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::I8 => "i8",
        }
    }

    /// Exact byte width of one canonical logical element in v1.
    pub const fn logical_bytes_per_element(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
            Self::I8 => 1,
        }
    }
}

/// A named stored section supplied to the canonical writer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SectionPayload {
    /// Printable-ASCII authority name, unique in the artifact.
    pub name: String,
    /// Closed v1 section kind.
    pub kind: SectionKind,
    /// Exact stored bytes.
    pub bytes: Vec<u8>,
    /// Required power-of-two placement alignment.
    pub alignment: u64,
}

impl SectionPayload {
    /// Construct a stored section.
    pub fn new(
        name: impl Into<String>,
        kind: SectionKind,
        bytes: impl Into<Vec<u8>>,
        alignment: u64,
    ) -> Self {
        Self {
            name: name.into(),
            kind,
            bytes: bytes.into(),
            alignment,
        }
    }
}

/// A byte range within one named section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SectionRange {
    /// Name of the target section.
    pub section_name: String,
    /// Offset from the start of that stored section.
    pub offset: u64,
    /// Byte length inside that stored section.
    pub len: u64,
}

impl SectionRange {
    /// Construct a checked-at-write-time section range.
    pub fn new(section_name: impl Into<String>, offset: u64, len: u64) -> Self {
        Self {
            section_name: section_name.into(),
            offset,
            len,
        }
    }
}

/// One logical tensor and its generic-representation mappings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorInput {
    /// Sorted, dotted printable-ASCII tensor name after canonicalization.
    pub name: String,
    /// Logical source dtype; high-precision tensors remain BF16 verbatim.
    pub canonical_dtype: CanonicalDtype,
    /// Rank 1 through 8, with every dimension nonzero.
    pub shape: Vec<u32>,
    /// Domain-framed logical identity supplied by the converter.
    pub canonical_logical_sha256: String,
    /// Fixed generic-representation quantization recipe identifier.
    pub quantization: String,
    /// Mapping into `GENERIC_TENSOR_PAYLOAD`.
    pub data: SectionRange,
    /// Mapping into `GENERIC_TENSOR_SCALES`.
    pub scale: SectionRange,
    /// Mapping into `GENERIC_TENSOR_ROW_SUMS`.
    pub row_sum: SectionRange,
}

/// A packing set derived from complete stored sections.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackingSetInput {
    /// Unique authority ID.  `generic` is mandatory.
    pub id: String,
    /// Target that may consume this packing set.
    pub target: ArchTarget,
    /// Names of directory sections retained by this representation.
    pub section_names: Vec<String>,
}

/// Typed writer input.  Identity values not reconstructible from packing bytes
/// are supplied by the converter's earlier canonical-source stages.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FnlpqWriterInput {
    /// Model source identifier.
    pub model_id: String,
    /// Full 40-hex immutable source revision.
    pub revision: String,
    /// Immutable converter recipe ID.
    pub recipe_id: String,
    /// `fnlpq-source-root-v1` identity generated by source acquisition.
    pub source_root_sha256: String,
    /// `fnlpq-logical-model-v1` identity generated from canonical tensors.
    pub logical_model_sha256: String,
    /// Every stored payload section.
    pub sections: Vec<SectionPayload>,
    /// Canonical logical tensor declarations.
    pub tensors: Vec<TensorInput>,
    /// Generic plus optional native packing representations.
    pub packing_sets: Vec<PackingSetInput>,
}

/// A fully serialized canonical artifact and its computed identities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WrittenFnlpq {
    /// Exact bytes written to the artifact file.
    pub bytes: Vec<u8>,
    /// Exact canonical JSON header bytes covered by the prelude hash.
    pub header_bytes: Vec<u8>,
    /// SHA-256 of `header_bytes` as stored in the prelude.
    pub header_sha256: [u8; 32],
    /// Ordered directory records for diagnostics and receipts.
    pub sections: Vec<WrittenSection>,
    /// `D("fnlpq-file-v1", exact_file_bytes)`.
    pub fnlpq_file_sha256: [u8; 32],
    /// `D("fnlpq-packing-set-v1", ...)` represented in the header.
    pub packing_set_sha256: [u8; 32],
    /// `D("fnlpq-license-bundle-v1", license_bytes)` represented in the header.
    pub license_bundle_sha256: [u8; 32],
}

/// One resolved directory record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WrittenSection {
    /// Header/directory ordinal.
    pub ordinal: u64,
    /// Unique section authority name.
    pub name: String,
    /// Closed v1 kind.
    pub kind: SectionKind,
    /// Absolute file offset.
    pub file_offset: u64,
    /// Stored byte length.
    pub stored_len: u64,
    /// Required power-of-two alignment.
    pub alignment: u64,
    /// Domain-framed stored payload digest.
    pub stored_sha256: [u8; 32],
}

/// The fixed prelude parsed only as far as the writer tests need to prove its
/// byte layout.  The checked reader owns hostile-input validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Prelude {
    /// Exact v1 magic bytes.
    pub magic: [u8; 8],
    /// Format version.
    pub format_version: u32,
    /// Required v1 flags.
    pub required_flags: u32,
    /// Canonical header length.
    pub header_len: u64,
    /// Directory entry count.
    pub section_count: u64,
    /// Canonical logical tensor count.
    pub tensor_count: u64,
    /// Exact artifact file length.
    pub file_len: u64,
    /// SHA-256 of the exact header bytes.
    pub header_sha256: [u8; 32],
}

/// Typed failures emitted before the writer creates any ambiguous bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FnlpqWriteError {
    /// An authority-bearing identifier did not meet its precise grammar.
    InvalidAuthority { field: &'static str, value: String },
    /// A fixed-width digest was not lowercase hex.
    InvalidDigest { field: &'static str, value: String },
    /// A precondition requiring distinct names or kinds failed.
    Duplicate { field: &'static str, value: String },
    /// A required format constituent was not supplied.
    Missing { field: &'static str, value: String },
    /// A supplied collection or byte range exceeded a v1 cap.
    Limit {
        field: &'static str,
        observed: u64,
        cap: u64,
    },
    /// Alignment did not meet v1's power-of-two contract.
    InvalidAlignment { section: String, alignment: u64 },
    /// Checked arithmetic would exceed the format's address space.
    Arithmetic { field: &'static str },
    /// A tensor declaration violates the bounded logical schema.
    Tensor { tensor: String, reason: String },
    /// A tensor mapping does not point at its required generic section.
    Mapping {
        tensor: String,
        mapping: &'static str,
        reason: String,
    },
    /// A declared logical identity did not match the canonical bytes being
    /// written.  The writer never serializes a caller-supplied stale claim.
    LogicalIdentity {
        record: String,
        expected: String,
        actual: String,
    },
    /// A packing set is incomplete or does not name valid stored sections.
    Packing { set: String, reason: String },
    /// JSON serialization of a typed closed schema failed.
    CanonicalJson(String),
    /// A scale cannot be represented in the fixed IEEE binary section.
    NonFiniteScale { index: usize },
    /// Fewer than 80 bytes were available for the fixed prelude.
    TruncatedPrelude { observed: usize },
}

impl fmt::Display for FnlpqWriteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAuthority { field, value } => {
                write!(formatter, "invalid {field} authority identifier: {value:?}")
            }
            Self::InvalidDigest { field, value } => {
                write!(
                    formatter,
                    "invalid {field} lowercase SHA-256 digest: {value:?}"
                )
            }
            Self::Duplicate { field, value } => write!(formatter, "duplicate {field}: {value}"),
            Self::Missing { field, value } => write!(formatter, "missing {field}: {value}"),
            Self::Limit {
                field,
                observed,
                cap,
            } => write!(formatter, "{field} exceeds v1 cap {cap}: {observed}"),
            Self::InvalidAlignment { section, alignment } => write!(
                formatter,
                "section {section} has invalid v1 alignment {alignment}"
            ),
            Self::Arithmetic { field } => {
                write!(formatter, "checked arithmetic overflow for {field}")
            }
            Self::Tensor { tensor, reason } => write!(formatter, "tensor {tensor}: {reason}"),
            Self::Mapping {
                tensor,
                mapping,
                reason,
            } => write!(formatter, "tensor {tensor} {mapping} mapping: {reason}"),
            Self::LogicalIdentity {
                record,
                expected,
                actual,
            } => write!(
                formatter,
                "logical identity mismatch for {record}: expected={expected} actual={actual}"
            ),
            Self::Packing { set, reason } => write!(formatter, "packing set {set}: {reason}"),
            Self::CanonicalJson(reason) => write!(formatter, "canonical header JSON: {reason}"),
            Self::NonFiniteScale { index } => {
                write!(formatter, "non-finite scale at index {index}")
            }
            Self::TruncatedPrelude { observed } => {
                write!(
                    formatter,
                    "truncated fnlpq prelude: expected 80 bytes, found {observed}"
                )
            }
        }
    }
}

impl Error for FnlpqWriteError {}

/// Encode finite f32 scales in the only accepted v1 binary representation.
///
/// Scale values never enter canonical JSON.  Rejecting NaN and infinity here
/// gives converters a precise error before section bytes are accepted.
pub fn encode_f32_scales(scales: &[f32]) -> Result<Vec<u8>, FnlpqWriteError> {
    let mut encoded = Vec::with_capacity(scales.len().saturating_mul(4));
    for (index, scale) in scales.iter().copied().enumerate() {
        if !scale.is_finite() {
            return Err(FnlpqWriteError::NonFiniteScale { index });
        }
        encoded.extend_from_slice(&scale.to_bits().to_le_bytes());
    }
    Ok(encoded)
}

/// Compute a v1 domain-framed SHA-256 identity.
pub fn framed_sha256(tag: &str, fields: &[&[u8]]) -> Result<[u8; 32], FnlpqWriteError> {
    if !is_printable_ascii(tag) {
        return Err(FnlpqWriteError::InvalidAuthority {
            field: "digest tag",
            value: tag.to_owned(),
        });
    }
    let field_count = u64::try_from(fields.len()).map_err(|_| FnlpqWriteError::Arithmetic {
        field: "digest field count",
    })?;
    let mut hasher = Sha256::new();
    hasher.update(tag.as_bytes());
    hasher.update([0]);
    hasher.update(field_count.to_le_bytes());
    for field in fields {
        let len = u64::try_from(field.len()).map_err(|_| FnlpqWriteError::Arithmetic {
            field: "digest field length",
        })?;
        hasher.update(len.to_le_bytes());
        hasher.update(field);
    }
    Ok(hasher.finalize().into())
}

/// Return the lowercase-hex spelling of a domain-framed identity.
pub fn framed_sha256_hex(tag: &str, fields: &[&[u8]]) -> Result<String, FnlpqWriteError> {
    Ok(hex_lower(&framed_sha256(tag, fields)?))
}

/// Compute the canonical identity of one logical tensor record.
///
/// The record binds the name, source dtype, shape, generic quantization
/// recipe, and every generic byte range.  It is deliberately independent of
/// section placement and native packing bytes.
pub fn logical_tensor_sha256(
    name: &str,
    canonical_dtype: &str,
    shape: &[u32],
    quantization: &str,
    data: &[u8],
    scale: &[u8],
    row_sum: &[u8],
) -> Result<[u8; 32], FnlpqWriteError> {
    let mut encoded_shape = Vec::with_capacity(8 + shape.len().saturating_mul(4));
    encoded_shape.extend_from_slice(
        &u64::try_from(shape.len())
            .map_err(|_| FnlpqWriteError::Arithmetic {
                field: "logical tensor shape rank",
            })?
            .to_le_bytes(),
    );
    for dimension in shape {
        encoded_shape.extend_from_slice(&dimension.to_le_bytes());
    }
    framed_sha256(
        "fnlpq-logical-tensor-v1",
        &[
            name.as_bytes(),
            canonical_dtype.as_bytes(),
            &encoded_shape,
            quantization.as_bytes(),
            data,
            scale,
            row_sum,
        ],
    )
}

/// Compute the domain-separated identity of one materialized semantic source.
pub fn materialized_source_sha256(
    source_name: &str,
    bytes: &[u8],
) -> Result<[u8; 32], FnlpqWriteError> {
    framed_sha256(
        "fnlpq-materialized-source-v1",
        &[source_name.as_bytes(), bytes],
    )
}

/// Compute the canonical model identity over sorted tensor records and the
/// four materialized configuration/tokenizer/template identities.
pub fn logical_model_sha256(
    tensor_digests: &[[u8; 32]],
    materialized_sources: &[(&str, &[u8])],
) -> Result<[u8; 32], FnlpqWriteError> {
    let source_digests = materialized_sources
        .iter()
        .map(|(name, bytes)| materialized_source_sha256(name, bytes))
        .collect::<Result<Vec<_>, _>>()?;
    let mut fields = Vec::with_capacity(2 + tensor_digests.len() + source_digests.len());
    fields.push(b"tensors".as_slice());
    fields.extend(tensor_digests.iter().map(|digest| digest.as_slice()));
    fields.push(b"materialized_sources".as_slice());
    fields.extend(source_digests.iter().map(|digest| digest.as_slice()));
    framed_sha256("fnlpq-logical-model-v1", &fields)
}

/// Serialize a typed logical model into canonical v1 `.fnlpq` bytes.
pub fn write(input: &FnlpqWriterInput) -> Result<WrittenFnlpq, FnlpqWriteError> {
    validate_authority("model_id", &input.model_id)?;
    validate_revision(&input.revision)?;
    validate_authority("recipe_id", &input.recipe_id)?;
    validate_digest("source_root_sha256", &input.source_root_sha256)?;
    validate_digest("logical_model_sha256", &input.logical_model_sha256)?;

    let sections = canonical_sections(&input.sections)?;
    let section_by_name: BTreeMap<_, _> = sections
        .iter()
        .enumerate()
        .map(|(ordinal, section)| (section.name.as_str(), (ordinal, section)))
        .collect();
    let sections_by_kind: BTreeMap<_, _> = sections
        .iter()
        .enumerate()
        .map(|(ordinal, section)| (section.kind, (ordinal, section)))
        .collect();

    let tensors = canonical_tensors(&input.tensors, &section_by_name, &sections_by_kind)?;
    let packing_sets =
        canonical_packing_sets(&input.packing_sets, &section_by_name, &sections_by_kind)?;
    let logical_tensor_digests = tensors
        .iter()
        .map(|tensor| {
            let data = input_range_bytes(&sections, &tensor.data)?;
            let scale = input_range_bytes(&sections, &tensor.scale)?;
            let row_sum = input_range_bytes(&sections, &tensor.row_sum)?;
            let observed = logical_tensor_sha256(
                &tensor.name,
                tensor.canonical_dtype.header_name(),
                &tensor.shape,
                &tensor.quantization,
                data,
                scale,
                row_sum,
            )?;
            let actual = hex_lower(&observed);
            if actual != tensor.canonical_logical_sha256 {
                return Err(FnlpqWriteError::LogicalIdentity {
                    record: tensor.name.clone(),
                    expected: tensor.canonical_logical_sha256.clone(),
                    actual,
                });
            }
            Ok(observed)
        })
        .collect::<Result<Vec<_>, FnlpqWriteError>>()?;
    let materialized_sources = MATERIALIZED_SOURCE_KINDS
        .iter()
        .map(|(name, kind)| {
            let section = sections_by_kind
                .get(kind)
                .expect("required materialized source section was validated")
                .1;
            (*name, section.bytes.as_slice())
        })
        .collect::<Vec<_>>();
    let observed_logical_model =
        logical_model_sha256(&logical_tensor_digests, &materialized_sources)?;
    let actual_logical_model = hex_lower(&observed_logical_model);
    if actual_logical_model != input.logical_model_sha256 {
        return Err(FnlpqWriteError::LogicalIdentity {
            record: "logical_model_sha256".to_owned(),
            expected: input.logical_model_sha256.clone(),
            actual: actual_logical_model,
        });
    }
    let license_section = sections_by_kind
        .get(&SectionKind::LicenseBundle)
        .expect("validated required singleton section")
        .1;
    let license_bundle_sha256 =
        framed_sha256("fnlpq-license-bundle-v1", &[&license_section.bytes])?;
    let section_hashes: Vec<[u8; 32]> = sections
        .iter()
        .map(|section| {
            framed_sha256(
                "fnlpq-section-v1",
                &[section.name.as_bytes(), &section.bytes],
            )
        })
        .collect::<Result<_, _>>()?;
    let packing_set_sha256 =
        packing_set_digest(&input.recipe_id, &packing_sets, &sections, &section_hashes)?;
    let header = build_header(
        input,
        &sections,
        &section_hashes,
        &tensors,
        &packing_sets,
        &packing_set_sha256,
        &license_bundle_sha256,
    )?;
    let header_bytes = canonjson::canonical_bytes(&header)
        .map_err(|error| FnlpqWriteError::CanonicalJson(error.to_string()))?;
    let header_len =
        u64::try_from(header_bytes.len()).map_err(|_| FnlpqWriteError::Arithmetic {
            field: "header length",
        })?;
    if header_len > MAX_HEADER_BYTES {
        return Err(FnlpqWriteError::Limit {
            field: "header_len",
            observed: header_len,
            cap: MAX_HEADER_BYTES,
        });
    }
    let header_sha256: [u8; 32] = Sha256::digest(&header_bytes).into();
    let directory_bytes = u64::try_from(sections.len())
        .map_err(|_| FnlpqWriteError::Arithmetic {
            field: "section count",
        })?
        .checked_mul(SECTION_DIRECTORY_ENTRY_BYTES as u64)
        .ok_or(FnlpqWriteError::Arithmetic {
            field: "directory length",
        })?;
    let directory_end = (PRELUDE_BYTES as u64)
        .checked_add(header_len)
        .and_then(|value| value.checked_add(directory_bytes))
        .ok_or(FnlpqWriteError::Arithmetic {
            field: "directory end",
        })?;

    let mut cursor = directory_end;
    let mut written_sections = Vec::with_capacity(sections.len());
    for (ordinal, (section, stored_sha256)) in sections.iter().zip(section_hashes).enumerate() {
        cursor = align_up(cursor, section.alignment)?;
        let stored_len =
            u64::try_from(section.bytes.len()).map_err(|_| FnlpqWriteError::Arithmetic {
                field: "section stored length",
            })?;
        let end = cursor
            .checked_add(stored_len)
            .ok_or(FnlpqWriteError::Arithmetic {
                field: "section end",
            })?;
        if end > MAX_FILE_BYTES {
            return Err(FnlpqWriteError::Limit {
                field: "file_len",
                observed: end,
                cap: MAX_FILE_BYTES,
            });
        }
        written_sections.push(WrittenSection {
            ordinal: ordinal as u64,
            name: section.name.clone(),
            kind: section.kind,
            file_offset: cursor,
            stored_len,
            alignment: section.alignment,
            stored_sha256,
        });
        cursor = end;
    }

    let file_len = cursor;
    let file_len_usize = usize::try_from(file_len).map_err(|_| FnlpqWriteError::Limit {
        field: "file_len on this host",
        observed: file_len,
        cap: usize::MAX as u64,
    })?;
    let mut bytes = vec![0; file_len_usize];
    write_prelude(
        &mut bytes[..PRELUDE_BYTES],
        Prelude {
            magic: MAGIC,
            format_version: FORMAT_VERSION,
            required_flags: 0,
            header_len,
            section_count: sections.len() as u64,
            tensor_count: tensors.len() as u64,
            file_len,
            header_sha256,
        },
    );
    let header_start = PRELUDE_BYTES;
    let header_end =
        header_start
            .checked_add(header_bytes.len())
            .ok_or(FnlpqWriteError::Arithmetic {
                field: "header end",
            })?;
    bytes[header_start..header_end].copy_from_slice(&header_bytes);
    let directory_start = header_end;
    for section in &written_sections {
        let start = directory_start
            .checked_add(
                (section.ordinal as usize)
                    .checked_mul(SECTION_DIRECTORY_ENTRY_BYTES)
                    .ok_or(FnlpqWriteError::Arithmetic {
                        field: "directory entry offset",
                    })?,
            )
            .ok_or(FnlpqWriteError::Arithmetic {
                field: "directory entry end",
            })?;
        write_directory_entry(
            &mut bytes[start..start + SECTION_DIRECTORY_ENTRY_BYTES],
            section,
        );
    }
    for (section, written) in sections.iter().zip(&written_sections) {
        let start = usize::try_from(written.file_offset).map_err(|_| FnlpqWriteError::Limit {
            field: "section file offset on this host",
            observed: written.file_offset,
            cap: usize::MAX as u64,
        })?;
        let end = start
            .checked_add(section.bytes.len())
            .ok_or(FnlpqWriteError::Arithmetic {
                field: "section copy end",
            })?;
        bytes[start..end].copy_from_slice(&section.bytes);
    }

    let fnlpq_file_sha256 = framed_sha256("fnlpq-file-v1", &[&bytes])?;
    Ok(WrittenFnlpq {
        bytes,
        header_bytes,
        header_sha256,
        sections: written_sections,
        fnlpq_file_sha256,
        packing_set_sha256,
        license_bundle_sha256,
    })
}

/// Decode a fixed prelude for writer-side byte-layout assertions.
pub fn decode_prelude(bytes: &[u8]) -> Result<Prelude, FnlpqWriteError> {
    if bytes.len() < PRELUDE_BYTES {
        return Err(FnlpqWriteError::TruncatedPrelude {
            observed: bytes.len(),
        });
    }
    let mut magic = [0; 8];
    magic.copy_from_slice(&bytes[..8]);
    let mut header_sha256 = [0; 32];
    header_sha256.copy_from_slice(&bytes[48..80]);
    Ok(Prelude {
        magic,
        format_version: u32::from_le_bytes(bytes[8..12].try_into().expect("fixed range")),
        required_flags: u32::from_le_bytes(bytes[12..16].try_into().expect("fixed range")),
        header_len: u64::from_le_bytes(bytes[16..24].try_into().expect("fixed range")),
        section_count: u64::from_le_bytes(bytes[24..32].try_into().expect("fixed range")),
        tensor_count: u64::from_le_bytes(bytes[32..40].try_into().expect("fixed range")),
        file_len: u64::from_le_bytes(bytes[40..48].try_into().expect("fixed range")),
        header_sha256,
    })
}

fn canonical_sections(input: &[SectionPayload]) -> Result<Vec<SectionPayload>, FnlpqWriteError> {
    if input.len() > MAX_ENTRIES {
        return Err(FnlpqWriteError::Limit {
            field: "section_count",
            observed: input.len() as u64,
            cap: MAX_ENTRIES as u64,
        });
    }
    let mut names = BTreeSet::new();
    let mut kind_counts = BTreeMap::new();
    for section in input {
        validate_authority("section.name", &section.name)?;
        if !names.insert(section.name.as_str()) {
            return Err(FnlpqWriteError::Duplicate {
                field: "section name",
                value: section.name.clone(),
            });
        }
        if !is_valid_alignment(section.alignment) {
            return Err(FnlpqWriteError::InvalidAlignment {
                section: section.name.clone(),
                alignment: section.alignment,
            });
        }
        let stored_len =
            u64::try_from(section.bytes.len()).map_err(|_| FnlpqWriteError::Arithmetic {
                field: "section stored length",
            })?;
        if stored_len > section.kind.section_cap() {
            return Err(FnlpqWriteError::Limit {
                field: "section stored length",
                observed: stored_len,
                cap: section.kind.section_cap(),
            });
        }
        *kind_counts.entry(section.kind).or_insert(0_usize) += 1;
    }
    for kind in REQUIRED_SECTION_KINDS {
        match kind_counts.get(&kind).copied().unwrap_or_default() {
            0 => {
                return Err(FnlpqWriteError::Missing {
                    field: "required section kind",
                    value: kind.header_name().to_owned(),
                });
            }
            1 => {}
            _ => {
                return Err(FnlpqWriteError::Duplicate {
                    field: "required section kind",
                    value: kind.header_name().to_owned(),
                });
            }
        }
    }
    let mut sections = input.to_vec();
    sections.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.name.as_bytes().cmp(right.name.as_bytes()))
    });
    Ok(sections)
}

fn canonical_tensors(
    input: &[TensorInput],
    sections_by_name: &BTreeMap<&str, (usize, &SectionPayload)>,
    sections_by_kind: &BTreeMap<SectionKind, (usize, &SectionPayload)>,
) -> Result<Vec<TensorInput>, FnlpqWriteError> {
    if input.is_empty() || input.len() > MAX_ENTRIES {
        return Err(FnlpqWriteError::Limit {
            field: "tensor_count",
            observed: input.len() as u64,
            cap: MAX_ENTRIES as u64,
        });
    }
    let mut names = BTreeSet::new();
    let mut ranges: BTreeMap<&str, Vec<(&str, u64, u64)>> = BTreeMap::new();
    for tensor in input {
        validate_authority("tensor.name", &tensor.name)?;
        if !names.insert(tensor.name.as_str()) {
            return Err(FnlpqWriteError::Duplicate {
                field: "tensor name",
                value: tensor.name.clone(),
            });
        }
        validate_digest(
            "tensor.canonical_logical_sha256",
            &tensor.canonical_logical_sha256,
        )?;
        validate_authority("tensor.quantization", &tensor.quantization)?;
        if tensor.shape.is_empty() || tensor.shape.len() > 8 {
            return Err(FnlpqWriteError::Tensor {
                tensor: tensor.name.clone(),
                reason: "shape rank must be in 1..=8".to_owned(),
            });
        }
        let logical_bytes = logical_byte_len(&tensor.name, &tensor.shape, tensor.canonical_dtype)?;
        if tensor.canonical_dtype == CanonicalDtype::Bf16 && tensor.quantization == BF16_VERBATIM_V1
        {
            if tensor.data.len != logical_bytes {
                return Err(FnlpqWriteError::Mapping {
                    tensor: tensor.name.clone(),
                    mapping: "data",
                    reason: format!(
                        "bf16-verbatim-v1 requires {logical_bytes} bytes for shape {:?}, observed {}",
                        tensor.shape, tensor.data.len
                    ),
                });
            }
        }
        let expected = [
            ("data", &tensor.data, SectionKind::GenericTensorPayload),
            ("scale", &tensor.scale, SectionKind::GenericTensorScales),
            (
                "row_sum",
                &tensor.row_sum,
                SectionKind::GenericTensorRowSums,
            ),
        ];
        for (mapping_name, mapping, kind) in expected {
            let (_, target) = sections_by_name
                .get(mapping.section_name.as_str())
                .ok_or_else(|| FnlpqWriteError::Mapping {
                    tensor: tensor.name.clone(),
                    mapping: mapping_name,
                    reason: format!("section {:?} is absent", mapping.section_name),
                })?;
            if target.kind != kind {
                return Err(FnlpqWriteError::Mapping {
                    tensor: tensor.name.clone(),
                    mapping: mapping_name,
                    reason: format!("must target {}", kind.header_name()),
                });
            }
            let end = mapping.offset.checked_add(mapping.len).ok_or_else(|| {
                FnlpqWriteError::Mapping {
                    tensor: tensor.name.clone(),
                    mapping: mapping_name,
                    reason: "offset plus length overflows u64".to_owned(),
                }
            })?;
            if end > target.bytes.len() as u64 {
                return Err(FnlpqWriteError::Mapping {
                    tensor: tensor.name.clone(),
                    mapping: mapping_name,
                    reason: format!(
                        "range [{}, {end}) exceeds section length {}",
                        mapping.offset,
                        target.bytes.len()
                    ),
                });
            }
            if mapping.len > 0 {
                ranges
                    .entry(mapping.section_name.as_str())
                    .or_default()
                    .push((tensor.name.as_str(), mapping.offset, end));
            }
        }
    }
    for (section_name, mut claimed) in ranges {
        claimed.sort_by_key(|(_, start, _)| *start);
        for pair in claimed.windows(2) {
            if pair[0].2 > pair[1].1 {
                return Err(FnlpqWriteError::Tensor {
                    tensor: pair[1].0.to_owned(),
                    reason: format!(
                        "mapping overlaps tensor {} in section {section_name} at [{}, {}) and [{}, {})",
                        pair[0].0, pair[0].1, pair[0].2, pair[1].1, pair[1].2
                    ),
                });
            }
        }
    }
    if sections_by_kind.len() < REQUIRED_SECTION_KINDS.len() {
        return Err(FnlpqWriteError::Missing {
            field: "required generic sections",
            value: "writer section index".to_owned(),
        });
    }
    let mut tensors = input.to_vec();
    tensors.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
    Ok(tensors)
}

fn canonical_packing_sets(
    input: &[PackingSetInput],
    sections_by_name: &BTreeMap<&str, (usize, &SectionPayload)>,
    sections_by_kind: &BTreeMap<SectionKind, (usize, &SectionPayload)>,
) -> Result<Vec<PackingSetInput>, FnlpqWriteError> {
    if input.is_empty() || input.len() > 16 {
        return Err(FnlpqWriteError::Limit {
            field: "packing_sets",
            observed: input.len() as u64,
            cap: 16,
        });
    }
    let mut ids = BTreeSet::new();
    for set in input {
        validate_authority("packing_set.id", &set.id)?;
        if !ids.insert(set.id.as_str()) {
            return Err(FnlpqWriteError::Duplicate {
                field: "packing set id",
                value: set.id.clone(),
            });
        }
        if set.section_names.is_empty() || set.section_names.len() > 16 {
            return Err(FnlpqWriteError::Packing {
                set: set.id.clone(),
                reason: "representation count must be in 1..=16".to_owned(),
            });
        }
        let mut names = BTreeSet::new();
        for name in &set.section_names {
            validate_authority("packing representation section", name)?;
            if !names.insert(name.as_str()) {
                return Err(FnlpqWriteError::Packing {
                    set: set.id.clone(),
                    reason: format!("duplicate representation section {name}"),
                });
            }
            if !sections_by_name.contains_key(name.as_str()) {
                return Err(FnlpqWriteError::Packing {
                    set: set.id.clone(),
                    reason: format!("unknown representation section {name}"),
                });
            }
        }
    }
    let generic = input
        .iter()
        .find(|set| set.id == "generic")
        .ok_or_else(|| FnlpqWriteError::Missing {
            field: "packing set",
            value: "generic".to_owned(),
        })?;
    if generic.target != ArchTarget::Generic {
        return Err(FnlpqWriteError::Packing {
            set: generic.id.clone(),
            reason: "generic set must declare the generic target".to_owned(),
        });
    }
    let required_generic: BTreeSet<_> = [
        SectionKind::GenericTensorPayload,
        SectionKind::GenericTensorScales,
        SectionKind::GenericTensorRowSums,
    ]
    .into_iter()
    .map(|kind| {
        sections_by_kind
            .get(&kind)
            .expect("required section checked before packing sets")
            .1
            .name
            .as_str()
    })
    .collect();
    let supplied_generic: BTreeSet<_> = generic.section_names.iter().map(String::as_str).collect();
    if supplied_generic != required_generic {
        return Err(FnlpqWriteError::Packing {
            set: generic.id.clone(),
            reason: "generic set must bind exactly payload, scale, and row-sum sections".to_owned(),
        });
    }
    for set in input {
        if set.id != "generic" {
            for name in &set.section_names {
                let kind = sections_by_name
                    .get(name.as_str())
                    .expect("named section checked above")
                    .1
                    .kind;
                if kind != SectionKind::NativePackingPayload {
                    return Err(FnlpqWriteError::Packing {
                        set: set.id.clone(),
                        reason: format!("native packing set references non-native section {name}"),
                    });
                }
            }
        }
    }
    let mut sets = input.to_vec();
    for set in &mut sets {
        set.section_names
            .sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    }
    sets.sort_by(|left, right| left.id.as_bytes().cmp(right.id.as_bytes()));
    Ok(sets)
}

fn build_header(
    input: &FnlpqWriterInput,
    sections: &[SectionPayload],
    section_hashes: &[[u8; 32]],
    tensors: &[TensorInput],
    packing_sets: &[PackingSetInput],
    packing_set_sha256: &[u8; 32],
    license_bundle_sha256: &[u8; 32],
) -> Result<Header, FnlpqWriteError> {
    let ordinals: BTreeMap<_, _> = sections
        .iter()
        .enumerate()
        .map(|(ordinal, section)| (section.name.as_str(), ordinal as u64))
        .collect();
    let section_hashes_by_name: BTreeMap<_, _> = sections
        .iter()
        .zip(section_hashes)
        .map(|(section, digest)| (section.name.as_str(), hex_lower(digest)))
        .collect();
    let materialized_sources = MATERIALIZED_SOURCE_KINDS
        .into_iter()
        .map(|(name, kind)| {
            let section = sections
                .iter()
                .find(|section| section.kind == kind)
                .expect("validated required source section");
            HeaderMaterializedSource {
                name: name.to_owned(),
                section_ordinal: *ordinals
                    .get(section.name.as_str())
                    .expect("section ordinal exists"),
                sha256: section_hashes_by_name
                    .get(section.name.as_str())
                    .expect("section digest exists")
                    .clone(),
            }
        })
        .collect();
    let sections_header = sections
        .iter()
        .enumerate()
        .map(|(ordinal, section)| HeaderSection {
            kind: section.kind.header_name(),
            name: section.name.clone(),
            ordinal: ordinal as u64,
            required: section.kind.required_singleton(),
        })
        .collect();
    let tensors_header = tensors
        .iter()
        .map(|tensor| {
            Ok(HeaderTensor {
                canonical_dtype: tensor.canonical_dtype.header_name(),
                canonical_logical_sha256: tensor.canonical_logical_sha256.clone(),
                generic: HeaderGenericMapping {
                    data: header_mapping(&tensor.data, &ordinals)?,
                    quantization: tensor.quantization.clone(),
                    row_sum: header_mapping(&tensor.row_sum, &ordinals)?,
                    scale: header_mapping(&tensor.scale, &ordinals)?,
                },
                logical_bytes: logical_byte_len(
                    &tensor.name,
                    &tensor.shape,
                    tensor.canonical_dtype,
                )?,
                name: tensor.name.clone(),
                shape: tensor.shape.clone(),
            })
        })
        .collect::<Result<_, _>>()?;
    let packing_sets_header = packing_sets
        .iter()
        .map(|set| {
            let mut representations = set
                .section_names
                .iter()
                .map(|name| {
                    let section = sections
                        .iter()
                        .find(|section| section.name == *name)
                        .expect("packing section was validated");
                    HeaderPackingRepresentation {
                        byte_cost: section.bytes.len() as u64,
                        section_ordinal: *ordinals
                            .get(name.as_str())
                            .expect("packing section ordinal exists"),
                        sha256: section_hashes_by_name
                            .get(name.as_str())
                            .expect("packing section hash exists")
                            .clone(),
                    }
                })
                .collect::<Vec<_>>();
            representations.sort_by_key(|representation| representation.section_ordinal);
            HeaderPackingSet {
                id: set.id.clone(),
                representations,
                target: set.target.header_name(),
            }
        })
        .collect();
    Ok(Header {
        header_schema: "fnlpq-header-v1",
        license_bundle_sha256: hex_lower(license_bundle_sha256),
        limits_profile: "fnlpq-limits-v1",
        logical_model_sha256: input.logical_model_sha256.clone(),
        materialized_sources,
        model: HeaderModel {
            model_id: input.model_id.clone(),
            revision: input.revision.clone(),
        },
        packing_set_sha256: hex_lower(packing_set_sha256),
        packing_sets: packing_sets_header,
        recipe_id: input.recipe_id.clone(),
        sections: sections_header,
        source_root_sha256: input.source_root_sha256.clone(),
        tensors: tensors_header,
    })
}

fn input_range_bytes<'a>(
    sections: &'a [SectionPayload],
    range: &SectionRange,
) -> Result<&'a [u8], FnlpqWriteError> {
    let section = sections
        .iter()
        .find(|section| section.name == range.section_name)
        .ok_or_else(|| FnlpqWriteError::Missing {
            field: "logical tensor mapping section",
            value: range.section_name.clone(),
        })?;
    let start = usize::try_from(range.offset).map_err(|_| FnlpqWriteError::Arithmetic {
        field: "logical tensor mapping offset",
    })?;
    let len = usize::try_from(range.len).map_err(|_| FnlpqWriteError::Arithmetic {
        field: "logical tensor mapping length",
    })?;
    let end = start.checked_add(len).ok_or(FnlpqWriteError::Arithmetic {
        field: "logical tensor mapping end",
    })?;
    section
        .bytes
        .get(start..end)
        .ok_or_else(|| FnlpqWriteError::Mapping {
            tensor: "logical identity".to_owned(),
            mapping: "range",
            reason: format!("range [{start}, {end}) exceeds section {}", section.name),
        })
}

fn logical_byte_len(
    tensor_name: &str,
    shape: &[u32],
    canonical_dtype: CanonicalDtype,
) -> Result<u64, FnlpqWriteError> {
    let mut element_count = 1_u64;
    for dimension in shape {
        if *dimension == 0 {
            return Err(FnlpqWriteError::Tensor {
                tensor: tensor_name.to_owned(),
                reason: "shape dimensions must be nonzero".to_owned(),
            });
        }
        element_count = element_count
            .checked_mul(u64::from(*dimension))
            .ok_or_else(|| FnlpqWriteError::Tensor {
                tensor: tensor_name.to_owned(),
                reason: "shape element product overflows u64".to_owned(),
            })?;
    }
    element_count
        .checked_mul(canonical_dtype.logical_bytes_per_element())
        .ok_or_else(|| FnlpqWriteError::Tensor {
            tensor: tensor_name.to_owned(),
            reason: "logical byte length overflows u64".to_owned(),
        })
}

fn header_mapping(
    range: &SectionRange,
    ordinals: &BTreeMap<&str, u64>,
) -> Result<HeaderMapping, FnlpqWriteError> {
    Ok(HeaderMapping {
        length: range.len,
        offset: range.offset,
        section_ordinal: *ordinals.get(range.section_name.as_str()).ok_or_else(|| {
            FnlpqWriteError::Missing {
                field: "header mapping section",
                value: range.section_name.clone(),
            }
        })?,
    })
}

fn packing_set_digest(
    recipe_id: &str,
    packing_sets: &[PackingSetInput],
    sections: &[SectionPayload],
    section_hashes: &[[u8; 32]],
) -> Result<[u8; 32], FnlpqWriteError> {
    let section_index: BTreeMap<_, _> = sections
        .iter()
        .enumerate()
        .map(|(ordinal, section)| (section.name.as_str(), ordinal))
        .collect();
    let digest_view: Vec<_> = packing_sets
        .iter()
        .map(|set| {
            let mut representations: Vec<_> = set
                .section_names
                .iter()
                .map(|name| {
                    let ordinal = *section_index
                        .get(name.as_str())
                        .expect("packing sections were validated before digest construction");
                    PackingDigestRepresentation {
                        byte_cost: sections[ordinal].bytes.len() as u64,
                        section_name: name,
                        stored_sha256: hex_lower(&section_hashes[ordinal]),
                    }
                })
                .collect();
            representations.sort_by(|left, right| {
                left.section_name
                    .as_bytes()
                    .cmp(right.section_name.as_bytes())
            });
            PackingDigestSet {
                id: &set.id,
                representations,
                target: set.target.header_name(),
            }
        })
        .collect();
    let canonical = canonjson::canonical_bytes(&digest_view)
        .map_err(|error| FnlpqWriteError::CanonicalJson(error.to_string()))?;
    framed_sha256("fnlpq-packing-set-v1", &[recipe_id.as_bytes(), &canonical])
}

fn align_up(value: u64, alignment: u64) -> Result<u64, FnlpqWriteError> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or(FnlpqWriteError::Arithmetic {
                field: "section alignment",
            })
    }
}

fn write_prelude(bytes: &mut [u8], prelude: Prelude) {
    bytes[..8].copy_from_slice(&prelude.magic);
    bytes[8..12].copy_from_slice(&prelude.format_version.to_le_bytes());
    bytes[12..16].copy_from_slice(&prelude.required_flags.to_le_bytes());
    bytes[16..24].copy_from_slice(&prelude.header_len.to_le_bytes());
    bytes[24..32].copy_from_slice(&prelude.section_count.to_le_bytes());
    bytes[32..40].copy_from_slice(&prelude.tensor_count.to_le_bytes());
    bytes[40..48].copy_from_slice(&prelude.file_len.to_le_bytes());
    bytes[48..80].copy_from_slice(&prelude.header_sha256);
}

fn write_directory_entry(bytes: &mut [u8], section: &WrittenSection) {
    bytes[..4].copy_from_slice(&section.kind.code().to_le_bytes());
    bytes[4..8].copy_from_slice(&0_u32.to_le_bytes());
    bytes[8..16].copy_from_slice(&section.ordinal.to_le_bytes());
    bytes[16..24].copy_from_slice(&section.file_offset.to_le_bytes());
    bytes[24..32].copy_from_slice(&section.stored_len.to_le_bytes());
    bytes[32..40].copy_from_slice(&section.stored_len.to_le_bytes());
    bytes[40..48].copy_from_slice(&section.alignment.to_le_bytes());
    bytes[48..80].copy_from_slice(&section.stored_sha256);
}

fn validate_authority(field: &'static str, value: &str) -> Result<(), FnlpqWriteError> {
    if value.len() > 128 || !is_authority_ascii(value) {
        return Err(FnlpqWriteError::InvalidAuthority {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_revision(value: &str) -> Result<(), FnlpqWriteError> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(FnlpqWriteError::InvalidAuthority {
            field: "revision",
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_digest(field: &'static str, value: &str) -> Result<(), FnlpqWriteError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(FnlpqWriteError::InvalidDigest {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn is_authority_ascii(value: &str) -> bool {
    let mut bytes = value.bytes();
    match bytes.next() {
        Some(first) if first.is_ascii_alphanumeric() => {}
        _ => return false,
    }
    bytes.all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'+' | b'-')
    })
}

fn is_printable_ascii(value: &str) -> bool {
    value.bytes().all(|byte| (b' '..=b'~').contains(&byte))
}

fn is_valid_alignment(alignment: u64) -> bool {
    alignment != 0 && alignment <= MAX_ALIGNMENT && alignment.is_power_of_two()
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[derive(Serialize)]
struct Header {
    header_schema: &'static str,
    license_bundle_sha256: String,
    limits_profile: &'static str,
    logical_model_sha256: String,
    materialized_sources: Vec<HeaderMaterializedSource>,
    model: HeaderModel,
    packing_set_sha256: String,
    packing_sets: Vec<HeaderPackingSet>,
    recipe_id: String,
    sections: Vec<HeaderSection>,
    source_root_sha256: String,
    tensors: Vec<HeaderTensor>,
}

#[derive(Serialize)]
struct HeaderModel {
    model_id: String,
    revision: String,
}

#[derive(Serialize)]
struct HeaderMaterializedSource {
    name: String,
    section_ordinal: u64,
    sha256: String,
}

#[derive(Serialize)]
struct HeaderSection {
    kind: &'static str,
    name: String,
    ordinal: u64,
    required: bool,
}

#[derive(Serialize)]
struct HeaderTensor {
    canonical_dtype: &'static str,
    canonical_logical_sha256: String,
    generic: HeaderGenericMapping,
    logical_bytes: u64,
    name: String,
    shape: Vec<u32>,
}

#[derive(Serialize)]
struct HeaderGenericMapping {
    data: HeaderMapping,
    quantization: String,
    row_sum: HeaderMapping,
    scale: HeaderMapping,
}

#[derive(Serialize)]
struct HeaderMapping {
    length: u64,
    offset: u64,
    section_ordinal: u64,
}

#[derive(Serialize)]
struct HeaderPackingSet {
    id: String,
    representations: Vec<HeaderPackingRepresentation>,
    target: &'static str,
}

#[derive(Serialize)]
struct HeaderPackingRepresentation {
    byte_cost: u64,
    section_ordinal: u64,
    sha256: String,
}

#[derive(Serialize)]
struct PackingDigestSet<'a> {
    id: &'a str,
    representations: Vec<PackingDigestRepresentation<'a>>,
    target: &'static str,
}

#[derive(Serialize)]
struct PackingDigestRepresentation<'a> {
    byte_cost: u64,
    section_name: &'a str,
    stored_sha256: String,
}
