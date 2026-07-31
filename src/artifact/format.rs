//! Canonical `.fnlpq` envelope v1 writing.
//!
//! The writer deliberately takes typed input only.  It never accepts a
//! `serde_json::Value` or a caller-provided header: the prelude, directory,
//! header order, padding, and domain-framed digests are all constructed here.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::io::{self, Write};

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

/// One section's first-pass evidence for the bounded streaming writer.
///
/// The converter computes this record before creating the staging file.  The
/// second pass must stream exactly `stored_len` bytes whose domain-framed
/// digest equals `stored_sha256`; the writer checks both conditions while
/// emitting the canonical envelope.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingSection {
    /// Printable-ASCII authority name, unique in the artifact.
    pub name: String,
    /// Closed v1 section kind.
    pub kind: SectionKind,
    /// Exact byte count computed during the planning pass.
    pub stored_len: u64,
    /// Required power-of-two placement alignment.
    pub alignment: u64,
    /// `D("fnlpq-section-v1", name, stored bytes)` from the planning pass.
    pub stored_sha256: [u8; 32],
}

/// Metadata required to produce a canonical v1 envelope without retaining its
/// payload sections in memory.
///
/// This is the second-pass input: the caller has already validated every
/// logical tensor identity and materialized-source identity while creating the
/// section records.  [`write_streaming`] verifies the stored section bytes as
/// they flow into the output and never creates a whole-file allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FnlpqStreamingInput {
    /// Model source identifier.
    pub model_id: String,
    /// Full 40-hex immutable source revision.
    pub revision: String,
    /// Immutable converter recipe ID.
    pub recipe_id: String,
    /// `fnlpq-source-root-v1` identity generated by source acquisition.
    pub source_root_sha256: String,
    /// `fnlpq-logical-model-v1` identity generated during the first pass.
    pub logical_model_sha256: String,
    /// Every planned stored section, without its payload bytes.
    pub sections: Vec<StreamingSection>,
    /// Canonical logical tensor declarations validated during the first pass.
    pub tensors: Vec<TensorInput>,
    /// Generic plus optional native packing representations.
    pub packing_sets: Vec<PackingSetInput>,
    /// `D("fnlpq-license-bundle-v1", license bytes)` from the first pass.
    pub license_bundle_sha256: [u8; 32],
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

/// Canonical v1 identities from a streaming write, excluding whole-file bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamedFnlpq {
    /// Exact canonical JSON header bytes covered by the prelude hash.
    pub header_bytes: Vec<u8>,
    /// SHA-256 of `header_bytes` as stored in the prelude.
    pub header_sha256: [u8; 32],
    /// Ordered directory records for diagnostics and receipts.
    pub sections: Vec<WrittenSection>,
    /// Exact final artifact byte length.
    pub file_len: u64,
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
    /// A bounded first-pass logical-tensor digest received its data, scale, or
    /// row-sum bytes out of order or at a length different from its plan.
    LogicalTensorStream { field: &'static str, detail: String },
    /// A packing set is incomplete or does not name valid stored sections.
    Packing { set: String, reason: String },
    /// JSON serialization of a typed closed schema failed.
    CanonicalJson(String),
    /// A scale cannot be represented in the fixed IEEE binary section.
    NonFiniteScale { index: usize },
    /// Streaming output or a section callback could not write exact bytes.
    Io {
        operation: &'static str,
        detail: String,
    },
    /// A second-pass section differed from its first-pass evidence.
    StoredIdentity {
        section: String,
        expected: String,
        actual: String,
    },
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
            Self::LogicalTensorStream { field, detail } => {
                write!(formatter, "logical tensor stream {field}: {detail}")
            }
            Self::Packing { set, reason } => write!(formatter, "packing set {set}: {reason}"),
            Self::CanonicalJson(reason) => write!(formatter, "canonical header JSON: {reason}"),
            Self::NonFiniteScale { index } => {
                write!(formatter, "non-finite scale at index {index}")
            }
            Self::Io { operation, detail } => write!(formatter, "{operation}: {detail}"),
            Self::StoredIdentity {
                section,
                expected,
                actual,
            } => write!(
                formatter,
                "stored section identity mismatch for {section}: expected={expected} actual={actual}"
            ),
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

/// Incremental first-pass identity for one stored section.
///
/// The converter records this identity before the second pass calls
/// [`write_streaming`], so a planned section can be checked without retaining
/// its full bytes in memory.
pub struct StreamingSectionHasher {
    section: String,
    remaining: u64,
    hasher: Sha256,
}

impl StreamingSectionHasher {
    /// Start a section identity from its canonical name and exact planned
    /// stored length.
    pub fn new(section: &str, stored_len: u64) -> Result<Self, FnlpqWriteError> {
        let mut hasher = framed_stream_hasher("fnlpq-section-v1", 2);
        framed_stream_field_prefix(
            &mut hasher,
            u64::try_from(section.len()).map_err(|_| FnlpqWriteError::Arithmetic {
                field: "streaming section name length",
            })?,
        );
        hasher.update(section.as_bytes());
        framed_stream_field_prefix(&mut hasher, stored_len);
        Ok(Self {
            section: section.to_owned(),
            remaining: stored_len,
            hasher,
        })
    }

    /// Include one bounded chunk in the planned stored section.
    pub fn write(&mut self, bytes: &[u8]) -> Result<(), FnlpqWriteError> {
        let bytes_len = u64::try_from(bytes.len()).map_err(|_| FnlpqWriteError::Arithmetic {
            field: "streaming section write length",
        })?;
        if bytes_len > self.remaining {
            return Err(FnlpqWriteError::StoredIdentity {
                section: self.section.clone(),
                expected: format!("{} bytes", self.remaining),
                actual: format!("{bytes_len} additional bytes"),
            });
        }
        self.hasher.update(bytes);
        self.remaining -= bytes_len;
        Ok(())
    }

    /// Return the section's first-pass digest only after exact coverage.
    pub fn finish(self) -> Result<[u8; 32], FnlpqWriteError> {
        if self.remaining != 0 {
            return Err(FnlpqWriteError::StoredIdentity {
                section: self.section,
                expected: "exact planned length".to_owned(),
                actual: format!("underflow by {} bytes", self.remaining),
            });
        }
        Ok(self.hasher.finalize().into())
    }
}

/// Incremental first-pass identity for one logical tensor.
///
/// A conversion pass may write its data bytes panel-by-panel, but v1 frames
/// the scale and row-sum fields after the complete data field.  This state
/// machine preserves that order and the three planned lengths without
/// retaining the tensor image in memory.
pub struct LogicalTensorStreamingHasher {
    hasher: Sha256,
    data_remaining: u64,
    scale_remaining: u64,
    row_sum_remaining: u64,
    scale_started: bool,
    row_sum_started: bool,
}

impl LogicalTensorStreamingHasher {
    /// Start a framed v1 logical-tensor identity from fixed declaration
    /// metadata and its three precomputed Generic byte lengths.
    pub fn new(
        name: &str,
        canonical_dtype: &str,
        shape: &[u32],
        quantization: &str,
        data_len: u64,
        scale_len: u64,
        row_sum_len: u64,
    ) -> Result<Self, FnlpqWriteError> {
        let encoded_shape = encode_logical_tensor_shape(shape)?;
        let mut hasher = framed_stream_hasher("fnlpq-logical-tensor-v1", 7);
        for field in [
            name.as_bytes(),
            canonical_dtype.as_bytes(),
            encoded_shape.as_slice(),
            quantization.as_bytes(),
        ] {
            framed_stream_field_prefix(
                &mut hasher,
                u64::try_from(field.len()).map_err(|_| FnlpqWriteError::Arithmetic {
                    field: "logical tensor stream declaration length",
                })?,
            );
            hasher.update(field);
        }
        framed_stream_field_prefix(&mut hasher, data_len);
        Ok(Self {
            hasher,
            data_remaining: data_len,
            scale_remaining: scale_len,
            row_sum_remaining: row_sum_len,
            scale_started: false,
            row_sum_started: false,
        })
    }

    /// Append the next bounded Generic payload panel for this tensor.
    pub fn write_data(&mut self, bytes: &[u8]) -> Result<(), FnlpqWriteError> {
        if self.scale_started {
            return Err(logical_tensor_stream_error(
                "data",
                "payload bytes arrived after the scale field began",
            ));
        }
        write_logical_tensor_stream_bytes(&mut self.hasher, &mut self.data_remaining, "data", bytes)
    }

    /// Append the complete tensor's Generic scale sidecar after its payload.
    pub fn write_scale(&mut self, bytes: &[u8]) -> Result<(), FnlpqWriteError> {
        self.start_scale()?;
        if self.row_sum_started {
            return Err(logical_tensor_stream_error(
                "scale",
                "scale bytes arrived after the row-sum field began",
            ));
        }
        write_logical_tensor_stream_bytes(
            &mut self.hasher,
            &mut self.scale_remaining,
            "scale",
            bytes,
        )
    }

    /// Append the complete tensor's Generic row-sum sidecar after scales.
    pub fn write_row_sum(&mut self, bytes: &[u8]) -> Result<(), FnlpqWriteError> {
        self.start_row_sum()?;
        write_logical_tensor_stream_bytes(
            &mut self.hasher,
            &mut self.row_sum_remaining,
            "row_sum",
            bytes,
        )
    }

    /// Finish only when the data and both sidecars exactly match their
    /// precomputed lengths.
    pub fn finish(mut self) -> Result<[u8; 32], FnlpqWriteError> {
        self.start_scale()?;
        self.start_row_sum()?;
        if self.row_sum_remaining != 0 {
            return Err(logical_tensor_stream_error(
                "row_sum",
                &format!("underflow by {} planned bytes", self.row_sum_remaining),
            ));
        }
        Ok(self.hasher.finalize().into())
    }

    fn start_scale(&mut self) -> Result<(), FnlpqWriteError> {
        if self.data_remaining != 0 {
            return Err(logical_tensor_stream_error(
                "data",
                &format!("underflow by {} planned bytes", self.data_remaining),
            ));
        }
        if !self.scale_started {
            framed_stream_field_prefix(&mut self.hasher, self.scale_remaining);
            self.scale_started = true;
        }
        Ok(())
    }

    fn start_row_sum(&mut self) -> Result<(), FnlpqWriteError> {
        self.start_scale()?;
        if self.scale_remaining != 0 {
            return Err(logical_tensor_stream_error(
                "scale",
                &format!("underflow by {} planned bytes", self.scale_remaining),
            ));
        }
        if !self.row_sum_started {
            framed_stream_field_prefix(&mut self.hasher, self.row_sum_remaining);
            self.row_sum_started = true;
        }
        Ok(())
    }
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
    let encoded_shape = encode_logical_tensor_shape(shape)?;
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

fn encode_logical_tensor_shape(shape: &[u32]) -> Result<Vec<u8>, FnlpqWriteError> {
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
    Ok(encoded_shape)
}

fn logical_tensor_stream_error(field: &'static str, detail: &str) -> FnlpqWriteError {
    FnlpqWriteError::LogicalTensorStream {
        field,
        detail: detail.to_owned(),
    }
}

fn write_logical_tensor_stream_bytes(
    hasher: &mut Sha256,
    remaining: &mut u64,
    field: &'static str,
    bytes: &[u8],
) -> Result<(), FnlpqWriteError> {
    let bytes_len = u64::try_from(bytes.len()).map_err(|_| FnlpqWriteError::Arithmetic {
        field: "logical tensor stream write length",
    })?;
    if bytes_len > *remaining {
        return Err(logical_tensor_stream_error(
            field,
            &format!(
                "overflow: planned remaining={} received={bytes_len}",
                *remaining
            ),
        ));
    }
    hasher.update(bytes);
    *remaining -= bytes_len;
    Ok(())
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
    let section_hashes: Vec<[u8; 32]> = sections
        .iter()
        .map(|section| {
            framed_sha256(
                "fnlpq-section-v1",
                &[section.name.as_bytes(), &section.bytes],
            )
        })
        .collect::<Result<_, _>>()?;
    let streaming_sections = materialized_streaming_sections(&sections, &section_hashes)?;
    let section_by_name: BTreeMap<_, _> = streaming_sections
        .iter()
        .enumerate()
        .map(|(ordinal, section)| (section.name.as_str(), (ordinal, section)))
        .collect();
    let sections_by_kind: BTreeMap<_, _> = streaming_sections
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
            let section = sections
                .iter()
                .find(|section| section.kind == *kind)
                .expect("required materialized source section was validated");
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
    let license_section = sections
        .iter()
        .find(|section| section.kind == SectionKind::LicenseBundle)
        .expect("validated required singleton section");
    let license_bundle_sha256 =
        framed_sha256("fnlpq-license-bundle-v1", &[&license_section.bytes])?;
    let packing_set_sha256 =
        packing_set_digest(&input.recipe_id, &packing_sets, &streaming_sections)?;
    let header = build_header(
        &input.model_id,
        &input.revision,
        &input.recipe_id,
        &input.source_root_sha256,
        &input.logical_model_sha256,
        &streaming_sections,
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
    for (ordinal, section) in streaming_sections.iter().enumerate() {
        cursor = align_up(cursor, section.alignment)?;
        let stored_len = section.stored_len;
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
            stored_sha256: section.stored_sha256,
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

/// Adapt a small materialized writer input into the second-pass streaming
/// metadata form.
///
/// This helper intentionally uses [`write`] as the fixture oracle, so it is
/// not a production conversion path.  Production converters compute the same
/// section evidence in their bounded first pass and construct
/// [`FnlpqStreamingInput`] directly.
pub fn streaming_input_from_materialized(
    input: &FnlpqWriterInput,
) -> Result<FnlpqStreamingInput, FnlpqWriteError> {
    let written = write(input)?;
    let sections = canonical_sections(&input.sections)?;
    let section_hashes = sections
        .iter()
        .map(|section| {
            framed_sha256(
                "fnlpq-section-v1",
                &[section.name.as_bytes(), &section.bytes],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(FnlpqStreamingInput {
        model_id: input.model_id.clone(),
        revision: input.revision.clone(),
        recipe_id: input.recipe_id.clone(),
        source_root_sha256: input.source_root_sha256.clone(),
        logical_model_sha256: input.logical_model_sha256.clone(),
        sections: materialized_streaming_sections(&sections, &section_hashes)?,
        tensors: input.tensors.clone(),
        packing_sets: input.packing_sets.clone(),
        license_bundle_sha256: written.license_bundle_sha256,
    })
}

/// Write one canonical v1 envelope without allocating its whole-file image.
///
/// The caller supplies each canonical section in the writer's sorted order.
/// It may use a bounded source traversal or a small fixture buffer; either
/// way, the writer refuses an underflow, overflow, or digest mismatch before
/// reporting a completed artifact.
pub fn write_streaming<W, F>(
    input: &FnlpqStreamingInput,
    output: &mut W,
    mut emit_section: F,
) -> Result<StreamedFnlpq, FnlpqWriteError>
where
    W: Write,
    F: FnMut(&StreamingSection, &mut dyn Write) -> Result<(), FnlpqWriteError>,
{
    validate_authority("model_id", &input.model_id)?;
    validate_revision(&input.revision)?;
    validate_authority("recipe_id", &input.recipe_id)?;
    validate_digest("source_root_sha256", &input.source_root_sha256)?;
    validate_digest("logical_model_sha256", &input.logical_model_sha256)?;

    let sections = canonical_streaming_sections(&input.sections)?;
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
    let packing_set_sha256 = packing_set_digest(&input.recipe_id, &packing_sets, &sections)?;
    let header = build_header(
        &input.model_id,
        &input.revision,
        &input.recipe_id,
        &input.source_root_sha256,
        &input.logical_model_sha256,
        &sections,
        &tensors,
        &packing_sets,
        &packing_set_sha256,
        &input.license_bundle_sha256,
    )?;
    let header_bytes = canonjson::canonical_bytes(&header)
        .map_err(|error| FnlpqWriteError::CanonicalJson(error.to_string()))?;
    let header_len = u64::try_from(header_bytes.len()).map_err(|_| FnlpqWriteError::Arithmetic {
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
    let written_sections = resolve_written_sections(&sections, directory_end)?;
    let license_section_name = sections
        .iter()
        .find(|section| section.kind == SectionKind::LicenseBundle)
        .expect("validated required license section")
        .name
        .as_str();
    let file_len = written_sections
        .last()
        .map(|section| {
            section
                .file_offset
                .checked_add(section.stored_len)
                .ok_or(FnlpqWriteError::Arithmetic {
                    field: "section end",
                })
        })
        .transpose()?
        .unwrap_or(directory_end);

    let mut prelude_bytes = [0_u8; PRELUDE_BYTES];
    write_prelude(
        &mut prelude_bytes,
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
    let mut file_hasher = framed_stream_hasher("fnlpq-file-v1", 1);
    framed_stream_field_prefix(&mut file_hasher, file_len);
    let mut position = 0_u64;
    write_stream_bytes(output, &mut file_hasher, &prelude_bytes)?;
    position = position
        .checked_add(PRELUDE_BYTES as u64)
        .ok_or(FnlpqWriteError::Arithmetic {
            field: "streaming prelude position",
        })?;
    write_stream_bytes(output, &mut file_hasher, &header_bytes)?;
    position = position
        .checked_add(header_len)
        .ok_or(FnlpqWriteError::Arithmetic {
            field: "streaming header position",
        })?;
    for section in &written_sections {
        let mut directory_entry = [0_u8; SECTION_DIRECTORY_ENTRY_BYTES];
        write_directory_entry(&mut directory_entry, section);
        write_stream_bytes(output, &mut file_hasher, &directory_entry)?;
        position = position
            .checked_add(SECTION_DIRECTORY_ENTRY_BYTES as u64)
            .ok_or(FnlpqWriteError::Arithmetic {
                field: "streaming directory position",
            })?;
    }
    let mut emitted_license_bundle_sha256 = None;
    for (section, written) in sections.iter().zip(&written_sections) {
        let padding = written
            .file_offset
            .checked_sub(position)
            .ok_or(FnlpqWriteError::Arithmetic {
                field: "streaming section position",
            })?;
        write_stream_zeroes(output, &mut file_hasher, padding)?;
        position = written.file_offset;
        let mut section_writer = StreamingSectionWriter::new(output, &mut file_hasher, section);
        emit_section(section, &mut section_writer)?;
        if let Some(actual) = section_writer.finish()? {
            emitted_license_bundle_sha256 = Some(actual);
        }
        position = position
            .checked_add(section.stored_len)
            .ok_or(FnlpqWriteError::Arithmetic {
                field: "streaming section end",
            })?;
    }
    if position != file_len {
        return Err(FnlpqWriteError::Arithmetic {
            field: "streaming final file length",
        });
    }
    let emitted_license_bundle_sha256 =
        emitted_license_bundle_sha256.ok_or_else(|| FnlpqWriteError::Missing {
            field: "streamed license bundle digest",
            value: license_section_name.to_owned(),
        })?;
    if emitted_license_bundle_sha256 != input.license_bundle_sha256 {
        return Err(FnlpqWriteError::StoredIdentity {
            section: license_section_name.to_owned(),
            expected: hex_lower(&input.license_bundle_sha256),
            actual: hex_lower(&emitted_license_bundle_sha256),
        });
    }
    output.flush().map_err(|error| FnlpqWriteError::Io {
        operation: "flush streaming fnlpq output",
        detail: error.to_string(),
    })?;
    Ok(StreamedFnlpq {
        header_bytes,
        header_sha256,
        sections: written_sections,
        file_len,
        fnlpq_file_sha256: file_hasher.finalize().into(),
        packing_set_sha256,
        license_bundle_sha256: input.license_bundle_sha256,
    })
}

fn resolve_written_sections(
    sections: &[StreamingSection],
    directory_end: u64,
) -> Result<Vec<WrittenSection>, FnlpqWriteError> {
    let mut cursor = directory_end;
    let mut written = Vec::with_capacity(sections.len());
    for (ordinal, section) in sections.iter().enumerate() {
        cursor = align_up(cursor, section.alignment)?;
        let end = cursor
            .checked_add(section.stored_len)
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
        written.push(WrittenSection {
            ordinal: ordinal as u64,
            name: section.name.clone(),
            kind: section.kind,
            file_offset: cursor,
            stored_len: section.stored_len,
            alignment: section.alignment,
            stored_sha256: section.stored_sha256,
        });
        cursor = end;
    }
    Ok(written)
}

fn framed_stream_hasher(tag: &str, field_count: u64) -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update(tag.as_bytes());
    hasher.update([0]);
    hasher.update(field_count.to_le_bytes());
    hasher
}

fn framed_stream_field_prefix(hasher: &mut Sha256, field_len: u64) {
    hasher.update(field_len.to_le_bytes());
}

fn write_stream_bytes<W: Write>(
    output: &mut W,
    file_hasher: &mut Sha256,
    bytes: &[u8],
) -> Result<(), FnlpqWriteError> {
    output.write_all(bytes).map_err(|error| FnlpqWriteError::Io {
        operation: "write streaming fnlpq output",
        detail: error.to_string(),
    })?;
    file_hasher.update(bytes);
    Ok(())
}

fn write_stream_zeroes<W: Write>(
    output: &mut W,
    file_hasher: &mut Sha256,
    mut bytes: u64,
) -> Result<(), FnlpqWriteError> {
    const ZEROES: [u8; 4096] = [0; 4096];
    while bytes > 0 {
        let count = bytes.min(ZEROES.len() as u64) as usize;
        write_stream_bytes(output, file_hasher, &ZEROES[..count])?;
        bytes -= count as u64;
    }
    Ok(())
}

struct StreamingSectionWriter<'a, W: Write> {
    output: &'a mut W,
    file_hasher: &'a mut Sha256,
    section: &'a StreamingSection,
    section_hasher: Sha256,
    license_bundle_hasher: Option<Sha256>,
    remaining: u64,
}

impl<'a, W: Write> StreamingSectionWriter<'a, W> {
    fn new(
        output: &'a mut W,
        file_hasher: &'a mut Sha256,
        section: &'a StreamingSection,
    ) -> Self {
        let mut section_hasher = framed_stream_hasher("fnlpq-section-v1", 2);
        framed_stream_field_prefix(&mut section_hasher, section.name.len() as u64);
        section_hasher.update(section.name.as_bytes());
        framed_stream_field_prefix(&mut section_hasher, section.stored_len);
        let license_bundle_hasher = if section.kind == SectionKind::LicenseBundle {
            let mut hasher = framed_stream_hasher("fnlpq-license-bundle-v1", 1);
            framed_stream_field_prefix(&mut hasher, section.stored_len);
            Some(hasher)
        } else {
            None
        };
        Self {
            output,
            file_hasher,
            section,
            section_hasher,
            license_bundle_hasher,
            remaining: section.stored_len,
        }
    }

    fn finish(self) -> Result<Option<[u8; 32]>, FnlpqWriteError> {
        if self.remaining != 0 {
            return Err(FnlpqWriteError::StoredIdentity {
                section: self.section.name.clone(),
                expected: hex_lower(&self.section.stored_sha256),
                actual: format!("underflow-{}-bytes", self.remaining),
            });
        }
        let actual: [u8; 32] = self.section_hasher.finalize().into();
        if actual != self.section.stored_sha256 {
            return Err(FnlpqWriteError::StoredIdentity {
                section: self.section.name.clone(),
                expected: hex_lower(&self.section.stored_sha256),
                actual: hex_lower(&actual),
            });
        }
        Ok(self
            .license_bundle_hasher
            .map(|hasher| hasher.finalize().into()))
    }
}

impl<W: Write> Write for StreamingSectionWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let requested = u64::try_from(bytes.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "streaming section write length does not fit u64",
            )
        })?;
        if requested > self.remaining {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "streaming section {} exceeds planned {} bytes",
                    self.section.name, self.section.stored_len
                ),
            ));
        }
        let written = self.output.write(bytes)?;
        let written_u64 = u64::try_from(written).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "streaming section write count does not fit u64",
            )
        })?;
        self.file_hasher.update(&bytes[..written]);
        self.section_hasher.update(&bytes[..written]);
        if let Some(hasher) = &mut self.license_bundle_hasher {
            hasher.update(&bytes[..written]);
        }
        self.remaining -= written_u64;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
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

fn materialized_streaming_sections(
    sections: &[SectionPayload],
    section_hashes: &[[u8; 32]],
) -> Result<Vec<StreamingSection>, FnlpqWriteError> {
    if sections.len() != section_hashes.len() {
        return Err(FnlpqWriteError::Arithmetic {
            field: "materialized streaming section metadata",
        });
    }
    sections
        .iter()
        .zip(section_hashes)
        .map(|(section, stored_sha256)| {
            Ok(StreamingSection {
                name: section.name.clone(),
                kind: section.kind,
                stored_len: u64::try_from(section.bytes.len()).map_err(|_| {
                    FnlpqWriteError::Arithmetic {
                        field: "section stored length",
                    }
                })?,
                alignment: section.alignment,
                stored_sha256: *stored_sha256,
            })
        })
        .collect()
}

fn canonical_streaming_sections(
    input: &[StreamingSection],
) -> Result<Vec<StreamingSection>, FnlpqWriteError> {
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
        if section.stored_len > section.kind.section_cap() {
            return Err(FnlpqWriteError::Limit {
                field: "section stored length",
                observed: section.stored_len,
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
        left
            .kind
            .cmp(&right.kind)
            .then_with(|| left.name.as_bytes().cmp(right.name.as_bytes()))
    });
    Ok(sections)
}

fn canonical_tensors(
    input: &[TensorInput],
    sections_by_name: &BTreeMap<&str, (usize, &StreamingSection)>,
    sections_by_kind: &BTreeMap<SectionKind, (usize, &StreamingSection)>,
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
            if end > target.stored_len {
                return Err(FnlpqWriteError::Mapping {
                    tensor: tensor.name.clone(),
                    mapping: mapping_name,
                    reason: format!(
                        "range [{}, {end}) exceeds section length {}",
                        mapping.offset, target.stored_len
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
    sections_by_name: &BTreeMap<&str, (usize, &StreamingSection)>,
    sections_by_kind: &BTreeMap<SectionKind, (usize, &StreamingSection)>,
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
    model_id: &str,
    revision: &str,
    recipe_id: &str,
    source_root_sha256: &str,
    logical_model_sha256: &str,
    sections: &[StreamingSection],
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
        .map(|section| (section.name.as_str(), hex_lower(&section.stored_sha256)))
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
                        byte_cost: section.stored_len,
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
        logical_model_sha256: logical_model_sha256.to_owned(),
        materialized_sources,
        model: HeaderModel {
            model_id: model_id.to_owned(),
            revision: revision.to_owned(),
        },
        packing_set_sha256: hex_lower(packing_set_sha256),
        packing_sets: packing_sets_header,
        recipe_id: recipe_id.to_owned(),
        sections: sections_header,
        source_root_sha256: source_root_sha256.to_owned(),
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
    sections: &[StreamingSection],
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
                        byte_cost: sections[ordinal].stored_len,
                        section_name: name,
                        stored_sha256: hex_lower(&sections[ordinal].stored_sha256),
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
