//! Checked safetensors range-index surface.
//!
//! This module is deliberately the converter's *range* boundary.  It hashes
//! every declared source serially before parsing any authority-bearing JSON or
//! exposing a tensor range.  It never maps or materializes a whole shard: a
//! [`SafetensorsRangeIndex`] retains verified file handles and reads one tensor
//! or bounded row panel at a time.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::canonjson::{self, CanonJsonError, ParseLimits};

/// Largest authority-bearing safetensors JSON header accepted by the range
/// path.  The pinned shards use only 23,312 bytes of container headers total.
pub const MAX_HEADER_BYTES: u64 = 1_048_576;
/// Largest safetensors index JSON source accepted by the range path.
pub const MAX_INDEX_BYTES: u64 = 4 * 1024 * 1024;
/// Largest rank accepted by the current converter surface.
pub const MAX_TENSOR_RANK: usize = 8;
/// Default maximum allocation for one read call.
pub const DEFAULT_MAX_RANGE_BYTES: u64 = 64 * 1024 * 1024;
/// Pinned number of Nanbeige4.2-3B tensors.
pub const PINNED_TENSOR_COUNT: usize = 201;
/// Pinned logical safetensors payload excluding container metadata.
pub const PINNED_PAYLOAD_BYTES: u64 = 8_339_601_408;
/// Pinned total bytes across the two safetensors shards.
pub const PINNED_SHARD_BYTES: u64 = 8_339_624_720;
/// Pinned safetensors header/container bytes, never model payload bytes.
pub const PINNED_CONTAINER_HEADER_BYTES: u64 = 23_312;

/// A safetensors scalar type with a fixed stored-byte width.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SafetensorDtype {
    /// Boolean storage.
    Bool,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// IEEE binary16.
    F16,
    /// Brain floating point 16.
    Bf16,
    /// IEEE binary32.
    F32,
    /// IEEE binary64.
    F64,
    /// 8-bit floating point E4M3FN.
    F8E4M3Fn,
    /// 8-bit floating point E5M2.
    F8E5M2,
    /// 8-bit floating point E4M3FNUZ.
    F8E4M3FnUz,
    /// 8-bit floating point E5M2FNUZ.
    F8E5M2FnUz,
}

impl SafetensorDtype {
    fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "U16" => Self::U16,
            "I16" => Self::I16,
            "U32" => Self::U32,
            "I32" => Self::I32,
            "U64" => Self::U64,
            "I64" => Self::I64,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "F32" => Self::F32,
            "F64" => Self::F64,
            "F8_E4M3FN" => Self::F8E4M3Fn,
            "F8_E5M2" => Self::F8E5M2,
            "F8_E4M3FNUZ" => Self::F8E4M3FnUz,
            "F8_E5M2FNUZ" => Self::F8E5M2FnUz,
            _ => return None,
        })
    }

    /// Exact safetensors header spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::U16 => "U16",
            Self::I16 => "I16",
            Self::U32 => "U32",
            Self::I32 => "I32",
            Self::U64 => "U64",
            Self::I64 => "I64",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::F8E4M3Fn => "F8_E4M3FN",
            Self::F8E5M2 => "F8_E5M2",
            Self::F8E4M3FnUz => "F8_E4M3FNUZ",
            Self::F8E5M2FnUz => "F8_E5M2FNUZ",
        }
    }

    /// Stored bytes per logical scalar.
    pub const fn byte_width(self) -> u64 {
        match self {
            Self::Bool
            | Self::U8
            | Self::I8
            | Self::F8E4M3Fn
            | Self::F8E5M2
            | Self::F8E4M3FnUz
            | Self::F8E5M2FnUz => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

/// A source file whose length and digest must match before it may be parsed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceDigest {
    file_name: String,
    expected_bytes: u64,
    expected_sha256: String,
}

impl SourceDigest {
    /// Build a source-digest requirement from a lowercase SHA-256 hex string.
    pub fn new(
        file_name: impl Into<String>,
        expected_bytes: u64,
        expected_sha256: impl Into<String>,
    ) -> Result<Self, SafetensorsError> {
        let file_name = file_name.into();
        if !safe_file_name(&file_name) {
            return Err(SafetensorsError::UnsafeSourceName { file_name });
        }
        let expected_sha256 = expected_sha256.into();
        if !is_lower_sha256(&expected_sha256) {
            return Err(SafetensorsError::InvalidExpectedDigest {
                file_name,
                digest: expected_sha256,
            });
        }
        Ok(Self {
            file_name,
            expected_bytes,
            expected_sha256,
        })
    }

    /// Basename relative to the verified source closure root.
    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    /// Pinned/expected source byte length.
    pub const fn expected_bytes(&self) -> u64 {
        self.expected_bytes
    }

    /// Pinned/expected lowercase SHA-256 digest.
    pub fn expected_sha256(&self) -> &str {
        &self.expected_sha256
    }
}

/// The exact index plus shard closure used to build a range index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceClosure {
    index: SourceDigest,
    shards: Vec<SourceDigest>,
}

impl SourceClosure {
    /// Construct a closure with one index and at least one distinct shard.
    pub fn new(index: SourceDigest, shards: Vec<SourceDigest>) -> Result<Self, SafetensorsError> {
        if shards.is_empty() {
            return Err(SafetensorsError::EmptyShardSet);
        }
        let mut names = BTreeSet::new();
        names.insert(index.file_name.clone());
        for shard in &shards {
            if !names.insert(shard.file_name.clone()) {
                return Err(SafetensorsError::DuplicateSourceName {
                    file_name: shard.file_name.clone(),
                });
            }
        }
        Ok(Self { index, shards })
    }

    /// The source requirement for `model.safetensors.index.json`.
    pub fn index(&self) -> &SourceDigest {
        &self.index
    }

    /// The source requirements for each safetensors shard.
    pub fn shards(&self) -> &[SourceDigest] {
        &self.shards
    }

    /// The immutable Nanbeige4.2-3B safetensors source closure.
    pub fn pinned_nanbeige42() -> Result<Self, SafetensorsError> {
        Self::new(
            SourceDigest::new(
                "model.safetensors.index.json",
                16_519,
                "30d8da0fa8b97abc6d9eddfd017a0cb7a649bcbd58caa57804c56c767db5c0f1",
            )?,
            vec![
                SourceDigest::new(
                    "model-00001-of-00002.safetensors",
                    4_973_547_960,
                    "09d265d5ec837bc64462796b7f8c110be9a135a55ed7a6eb5d07e0e90c976a94",
                )?,
                SourceDigest::new(
                    "model-00002-of-00002.safetensors",
                    3_366_076_760,
                    "31019e7870a044f44bc3f7e981f8c5ecd42d341e5ca6cfdbfd07fb95d95be389",
                )?,
            ],
        )
    }
}

/// Resource limits and dtype requirements for range-index construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RangeIndexOptions {
    /// Cap applied before allocating a header buffer.
    pub max_header_bytes: u64,
    /// Cap applied before allocating one tensor or row-panel result buffer.
    pub max_range_bytes: u64,
    /// When set, every tensor must use this exact dtype.
    pub required_dtype: Option<SafetensorDtype>,
}

impl Default for RangeIndexOptions {
    fn default() -> Self {
        Self {
            max_header_bytes: MAX_HEADER_BYTES,
            max_range_bytes: DEFAULT_MAX_RANGE_BYTES,
            required_dtype: None,
        }
    }
}

/// Streaming source-verification receipt available to converter logging.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HashReceipt {
    /// Verified basename.
    pub file_name: String,
    /// Actual byte count hashed.
    pub bytes: u64,
    /// Actual lowercase SHA-256 digest.
    pub sha256: String,
    /// Serial hashing elapsed time; callers may emit this in debug logs.
    pub elapsed: Duration,
}

/// Public tensor facts safe to enumerate after index construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorCensusEntry {
    /// Exact source tensor name.
    pub name: String,
    /// Validated safetensors dtype.
    pub dtype: SafetensorDtype,
    /// Validated shape dimensions.
    pub shape: Vec<u64>,
    /// Exact declared payload length in bytes.
    pub len: u64,
}

/// A caller-provided tensor contract for census comparison.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorExpectation {
    /// Expected source tensor name.
    pub name: String,
    /// Expected source dtype.
    pub dtype: SafetensorDtype,
    /// Expected dimensions.
    pub shape: Vec<u64>,
    /// Expected stored payload length.
    pub len: u64,
}

/// An explicit census diff.  Nothing rounds a mismatch into a match.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CensusDiff {
    /// Expected names absent from the source closure.
    pub missing: Vec<String>,
    /// Present names with a dtype, shape, or length mismatch.
    pub shape_mismatch: Vec<String>,
    /// Source names absent from the expectation.
    pub extra: Vec<String>,
}

impl CensusDiff {
    /// Whether all expected and observed tensors agree exactly.
    pub const fn is_match(&self) -> bool {
        self.missing.is_empty() && self.shape_mismatch.is_empty() && self.extra.is_empty()
    }
}

/// A requested region of one tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowPanel {
    /// Read the entire named tensor, subject to the per-read byte cap.
    WholeTensor,
    /// Read contiguous rows selected by the first shape dimension.
    Rows {
        /// Zero-based first row.
        start_row: u64,
        /// Number of contiguous rows to read.
        row_count: u64,
    },
}

/// A checked absolute byte range in a verified source shard.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckedRange {
    /// Offset from the beginning of the shard file.
    pub file_offset: u64,
    /// Number of bytes in the range.
    pub len: u64,
}

/// A successfully parsed safetensors header with no readable file handle.
///
/// This type supports hostile-header testing and converter census inspection;
/// it cannot expose source bytes.  [`SafetensorsRangeIndex`] is the only type
/// that can read a range, and it is constructed only after source verification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedShard {
    shard_name: String,
    file_len: u64,
    data_start: u64,
    tensors: BTreeMap<String, TensorRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorRecord {
    facts: TensorCensusEntry,
    relative_offset: u64,
}

impl CheckedShard {
    /// Parse an in-memory synthetic shard for a hostile-header or differential
    /// fixture.  This function never exposes a file range.
    pub fn from_bytes(
        shard_name: impl Into<String>,
        bytes: &[u8],
    ) -> Result<Self, SafetensorsError> {
        let shard_name = shard_name.into();
        let file_len = u64::try_from(bytes.len()).map_err(|_| SafetensorsError::Arithmetic {
            invariant: "synthetic shard length to u64",
            tensor: None,
        })?;
        let header_len = decode_header_len(&shard_name, bytes)?;
        if header_len > MAX_HEADER_BYTES {
            return Err(SafetensorsError::HeaderTooLarge {
                shard: shard_name,
                declared: header_len,
                cap: MAX_HEADER_BYTES,
            });
        }
        let header_end = checked_add(8, header_len, "header prefix + header length", None)?;
        if header_end > file_len {
            return Err(SafetensorsError::TruncatedHeader {
                shard: shard_name,
                declared: header_len,
                available: file_len.saturating_sub(8),
            });
        }
        let header_end_usize =
            usize::try_from(header_end).map_err(|_| SafetensorsError::Arithmetic {
                invariant: "header end to usize",
                tensor: None,
            })?;
        let header_bytes =
            bytes
                .get(8..header_end_usize)
                .ok_or_else(|| SafetensorsError::TruncatedHeader {
                    shard: shard_name.clone(),
                    declared: header_len,
                    available: file_len.saturating_sub(8),
                })?;
        parse_header(&shard_name, file_len, header_len, header_bytes)
    }

    /// Source basename carried by this header.
    pub fn shard_name(&self) -> &str {
        &self.shard_name
    }

    /// Verified total shard length used to bound every declared data range.
    pub const fn file_len(&self) -> u64 {
        self.file_len
    }

    /// Validated tensors in lexicographic name order.
    pub fn census(&self) -> Vec<TensorCensusEntry> {
        self.tensors
            .values()
            .map(|record| record.facts.clone())
            .collect()
    }

    /// Return one tensor's public metadata.
    pub fn tensor(&self, name: &str) -> Option<&TensorCensusEntry> {
        self.tensors.get(name).map(|record| &record.facts)
    }

    /// Calculate, but do not read, one checked tensor or row-panel range.
    pub fn range_for(
        &self,
        tensor: &str,
        panel: RowPanel,
    ) -> Result<CheckedRange, SafetensorsError> {
        let record = self
            .tensors
            .get(tensor)
            .ok_or_else(|| SafetensorsError::UnknownTensor {
                tensor: tensor.to_owned(),
            })?;
        let (relative_offset, len) = match panel {
            RowPanel::WholeTensor => (record.relative_offset, record.facts.len),
            RowPanel::Rows {
                start_row,
                row_count,
            } => {
                let rows = *record.facts.shape.first().ok_or_else(|| {
                    SafetensorsError::RowPanelOnScalar {
                        tensor: tensor.to_owned(),
                    }
                })?;
                let panel_end = checked_add(start_row, row_count, "row panel end", Some(tensor))?;
                if panel_end > rows {
                    return Err(SafetensorsError::RowPanelOutOfBounds {
                        tensor: tensor.to_owned(),
                        start_row,
                        row_count,
                        rows,
                    });
                }
                if rows == 0 {
                    return Err(SafetensorsError::RowPanelOnZeroRows {
                        tensor: tensor.to_owned(),
                    });
                }
                let row_bytes = record.facts.len / rows;
                let byte_offset =
                    checked_mul(start_row, row_bytes, "row panel offset", Some(tensor))?;
                let len = checked_mul(row_count, row_bytes, "row panel length", Some(tensor))?;
                (
                    checked_add(
                        record.relative_offset,
                        byte_offset,
                        "row panel relative offset",
                        Some(tensor),
                    )?,
                    len,
                )
            }
        };
        Ok(CheckedRange {
            file_offset: checked_add(
                self.data_start,
                relative_offset,
                "data start + range offset",
                Some(tensor),
            )?,
            len,
        })
    }
}

struct VerifiedShard {
    header: CheckedShard,
    file: Mutex<File>,
}

/// The digest-verified, bounded safetensors source range index.
pub struct SafetensorsRangeIndex {
    shards: BTreeMap<String, VerifiedShard>,
    tensor_to_shard: BTreeMap<String, String>,
    receipts: Vec<HashReceipt>,
    max_range_bytes: u64,
}

impl SafetensorsRangeIndex {
    /// Open an arbitrary explicit source closure after streaming verification.
    pub fn open(root: impl AsRef<Path>, closure: SourceClosure) -> Result<Self, SafetensorsError> {
        Self::open_with_options(root, closure, RangeIndexOptions::default())
    }

    /// Open the immutable two-shard Nanbeige4.2-3B closure.  In addition to
    /// digest verification, this path requires the 201 BF16 tensor/payload
    /// contract before making any range available.
    pub fn open_pinned_nanbeige42(root: impl AsRef<Path>) -> Result<Self, SafetensorsError> {
        let options = RangeIndexOptions {
            required_dtype: Some(SafetensorDtype::Bf16),
            ..RangeIndexOptions::default()
        };
        let index = Self::open_with_options(root, SourceClosure::pinned_nanbeige42()?, options)?;
        if index.tensor_to_shard.len() != PINNED_TENSOR_COUNT {
            return Err(SafetensorsError::PinnedTensorCount {
                expected: PINNED_TENSOR_COUNT,
                actual: index.tensor_to_shard.len(),
            });
        }
        let payload = index.census().into_iter().try_fold(0_u64, |total, entry| {
            checked_add(total, entry.len, "pinned payload sum", Some(&entry.name))
        })?;
        if payload != PINNED_PAYLOAD_BYTES {
            return Err(SafetensorsError::PinnedPayloadBytes {
                expected: PINNED_PAYLOAD_BYTES,
                actual: payload,
            });
        }
        let shard_bytes = index
            .receipts
            .iter()
            .filter(|receipt| receipt.file_name.ends_with(".safetensors"))
            .try_fold(0_u64, |total, receipt| {
                checked_add(total, receipt.bytes, "pinned shard byte sum", None)
            })?;
        let metadata_bytes = shard_bytes.checked_sub(payload).ok_or(
            SafetensorsError::PinnedContainerAccounting {
                shard_bytes,
                payload_bytes: payload,
                observed_metadata_bytes: 0,
                expected_shard_bytes: PINNED_SHARD_BYTES,
                expected_metadata_bytes: PINNED_CONTAINER_HEADER_BYTES,
            },
        )?;
        if shard_bytes != PINNED_SHARD_BYTES || metadata_bytes != PINNED_CONTAINER_HEADER_BYTES {
            return Err(SafetensorsError::PinnedContainerAccounting {
                shard_bytes,
                payload_bytes: payload,
                observed_metadata_bytes: metadata_bytes,
                expected_shard_bytes: PINNED_SHARD_BYTES,
                expected_metadata_bytes: PINNED_CONTAINER_HEADER_BYTES,
            });
        }
        Ok(index)
    }

    /// Build a range index with caller-selected caps and dtype policy.
    pub fn open_with_options(
        root: impl AsRef<Path>,
        closure: SourceClosure,
        options: RangeIndexOptions,
    ) -> Result<Self, SafetensorsError> {
        if options.max_header_bytes == 0 || options.max_range_bytes == 0 {
            return Err(SafetensorsError::InvalidOptions);
        }
        let root = root.as_ref();
        let mut verified_files = BTreeMap::new();
        let mut receipts = Vec::with_capacity(closure.shards.len() + 1);

        // Deliberately hash the complete closure before parsing index/header
        // JSON.  A failure leaves no range-index object to expose.
        for source in std::iter::once(closure.index()).chain(closure.shards().iter()) {
            let path = source_path(root, source)?;
            let mut file = File::open(&path).map_err(|error| SafetensorsError::Io {
                source: source.file_name.clone(),
                operation: "open verified source",
                detail: error.to_string(),
            })?;
            let receipt = verify_source_file(source, &mut file)?;
            receipts.push(receipt);
            verified_files.insert(source.file_name.clone(), file);
        }

        let mut index_file = verified_files.remove(closure.index.file_name()).ok_or(
            SafetensorsError::MissingVerifiedSource {
                source: closure.index.file_name.clone(),
            },
        )?;
        let index_bytes = read_verified_index(closure.index(), &mut index_file)?;
        let mut headers = BTreeMap::new();
        let mut shards = BTreeMap::new();
        for source in closure.shards() {
            let mut file = verified_files.remove(source.file_name()).ok_or_else(|| {
                SafetensorsError::MissingVerifiedSource {
                    source: source.file_name.clone(),
                }
            })?;
            let header = parse_file_header(source, &mut file, options.max_header_bytes)?;
            if let Some(required_dtype) = options.required_dtype {
                for tensor in header.census() {
                    if tensor.dtype != required_dtype {
                        return Err(SafetensorsError::RequiredDtype {
                            shard: source.file_name.clone(),
                            tensor: tensor.name,
                            expected: required_dtype,
                            actual: tensor.dtype,
                        });
                    }
                }
            }
            headers.insert(source.file_name.clone(), header.clone());
            shards.insert(
                source.file_name.clone(),
                VerifiedShard {
                    header,
                    file: Mutex::new(file),
                },
            );
        }
        let tensor_to_shard =
            validate_index_mapping(closure.index.file_name(), &index_bytes, &headers)?;
        reject_oq1_tripwires(tensor_to_shard.keys())?;
        Ok(Self {
            shards,
            tensor_to_shard,
            receipts,
            max_range_bytes: options.max_range_bytes,
        })
    }

    /// Read exactly one tensor or bounded row panel from a verified shard.
    pub fn read_range(&self, tensor: &str, panel: RowPanel) -> Result<Vec<u8>, SafetensorsError> {
        let shard_name =
            self.tensor_to_shard
                .get(tensor)
                .ok_or_else(|| SafetensorsError::UnknownTensor {
                    tensor: tensor.to_owned(),
                })?;
        let shard =
            self.shards
                .get(shard_name)
                .ok_or_else(|| SafetensorsError::MissingVerifiedSource {
                    source: shard_name.clone(),
                })?;
        let range = shard.header.range_for(tensor, panel)?;
        if range.len > self.max_range_bytes {
            return Err(SafetensorsError::RangeReadTooLarge {
                tensor: tensor.to_owned(),
                requested: range.len,
                cap: self.max_range_bytes,
            });
        }
        let len = usize::try_from(range.len).map_err(|_| SafetensorsError::Arithmetic {
            invariant: "range read length to usize",
            tensor: Some(tensor.to_owned()),
        })?;
        let mut bytes = vec![0_u8; len];
        let mut file = shard
            .file
            .lock()
            .map_err(|_| SafetensorsError::FileHandlePoisoned {
                source: shard_name.clone(),
            })?;
        file.seek(SeekFrom::Start(range.file_offset))
            .map_err(|error| SafetensorsError::Io {
                source: shard_name.clone(),
                operation: "seek checked range",
                detail: error.to_string(),
            })?;
        file.read_exact(&mut bytes)
            .map_err(|error| SafetensorsError::Io {
                source: shard_name.clone(),
                operation: "read checked range",
                detail: error.to_string(),
            })?;
        Ok(bytes)
    }

    /// Enumerate source `(name, dtype, shape, len)` facts in name order.
    pub fn census(&self) -> Vec<TensorCensusEntry> {
        self.tensor_to_shard
            .iter()
            .filter_map(|(name, shard_name)| {
                self.shards
                    .get(shard_name)
                    .and_then(|shard| shard.header.tensor(name))
                    .cloned()
            })
            .collect()
    }

    /// Compare this source census with a machine-readable expected table.
    pub fn diff_census(
        &self,
        expected: &[TensorExpectation],
    ) -> Result<CensusDiff, SafetensorsError> {
        diff_census_entries(&self.census(), expected)
    }

    /// Per-source hash receipts for converter debug/event logging.
    pub fn hash_receipts(&self) -> &[HashReceipt] {
        &self.receipts
    }
}

/// Verify one in-memory fixture source.  Production construction uses the
/// streaming counterpart internally; this helper enables a wrong-digest test
/// without granting any range access.
pub fn verify_source_bytes(
    source: &SourceDigest,
    bytes: &[u8],
) -> Result<HashReceipt, SafetensorsError> {
    let started = Instant::now();
    let actual_bytes = u64::try_from(bytes.len()).map_err(|_| SafetensorsError::Arithmetic {
        invariant: "fixture source length to u64",
        tensor: None,
    })?;
    let actual_sha256 = digest_hex(bytes);
    verify_observed_source(source, actual_bytes, actual_sha256, started.elapsed())
}

/// Diff any already-validated census facts against an expected tensor table.
/// This is shared by the verified range index and hostile fixture tests.
pub fn diff_census_entries(
    actual: &[TensorCensusEntry],
    expected: &[TensorExpectation],
) -> Result<CensusDiff, SafetensorsError> {
    let mut expected_by_name = BTreeMap::new();
    for entry in expected {
        if expected_by_name.insert(entry.name.clone(), entry).is_some() {
            return Err(SafetensorsError::DuplicateExpectedTensor {
                tensor: entry.name.clone(),
            });
        }
    }
    let actual_by_name: BTreeMap<_, _> = actual
        .iter()
        .map(|entry| (entry.name.clone(), entry))
        .collect();
    let missing = expected_by_name
        .keys()
        .filter(|name| !actual_by_name.contains_key(*name))
        .cloned()
        .collect();
    let extra = actual_by_name
        .keys()
        .filter(|name| !expected_by_name.contains_key(*name))
        .cloned()
        .collect();
    let shape_mismatch = expected_by_name
        .iter()
        .filter_map(|(name, expected)| {
            actual_by_name.get(name).and_then(|actual| {
                (actual.dtype != expected.dtype
                    || actual.shape != expected.shape
                    || actual.len != expected.len)
                    .then(|| name.clone())
            })
        })
        .collect();
    Ok(CensusDiff {
        missing,
        shape_mismatch,
        extra,
    })
}

/// Parse and cross-check an index `weight_map` against checked shard headers.
/// This function cannot read tensor bytes and is available for hostile tests.
pub fn validate_index_mapping(
    index_name: &str,
    index_bytes: &[u8],
    headers: &BTreeMap<String, CheckedShard>,
) -> Result<BTreeMap<String, String>, SafetensorsError> {
    let index_text =
        std::str::from_utf8(index_bytes).map_err(|error| SafetensorsError::IndexUtf8 {
            index: index_name.to_owned(),
            detail: error.to_string(),
        })?;
    let value = parse_json(index_name, index_text)?;
    let root = value
        .as_object()
        .ok_or_else(|| SafetensorsError::IndexSchema {
            index: index_name.to_owned(),
            detail: "root must be an object".to_owned(),
        })?;
    let weight_map = root
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| SafetensorsError::IndexSchema {
            index: index_name.to_owned(),
            detail: "weight_map must be an object".to_owned(),
        })?;
    let mut header_names = BTreeMap::new();
    for (shard_name, shard) in headers {
        for entry in shard.census() {
            if header_names
                .insert(entry.name.clone(), shard_name.clone())
                .is_some()
            {
                return Err(SafetensorsError::DuplicateTensorName {
                    shard: shard_name.clone(),
                    tensor: entry.name,
                });
            }
        }
    }
    let mut mapping = BTreeMap::new();
    for (tensor, shard_value) in weight_map {
        if tensor.is_empty() {
            return Err(SafetensorsError::IndexSchema {
                index: index_name.to_owned(),
                detail: "weight_map tensor name must be non-empty".to_owned(),
            });
        }
        let shard_name = shard_value
            .as_str()
            .ok_or_else(|| SafetensorsError::IndexSchema {
                index: index_name.to_owned(),
                detail: format!("weight_map entry {tensor:?} must name a shard string"),
            })?;
        let shard =
            headers
                .get(shard_name)
                .ok_or_else(|| SafetensorsError::IndexMapsUnknownShard {
                    index: index_name.to_owned(),
                    tensor: tensor.clone(),
                    shard: shard_name.to_owned(),
                })?;
        if shard.tensor(tensor).is_none() {
            return Err(SafetensorsError::IndexShardMismatch {
                index: index_name.to_owned(),
                tensor: tensor.clone(),
                shard: shard_name.to_owned(),
            });
        }
        mapping.insert(tensor.clone(), shard_name.to_owned());
    }
    for (shard_name, shard) in headers {
        for entry in shard.census() {
            match mapping.get(&entry.name) {
                Some(mapped) if mapped == shard_name => {}
                Some(mapped) => {
                    return Err(SafetensorsError::IndexShardMismatch {
                        index: index_name.to_owned(),
                        tensor: entry.name,
                        shard: mapped.clone(),
                    });
                }
                None => {
                    return Err(SafetensorsError::IndexMissingTensor {
                        index: index_name.to_owned(),
                        shard: shard_name.clone(),
                        tensor: entry.name,
                    });
                }
            }
        }
    }
    Ok(mapping)
}

/// Typed malformed-header, source, and bounded-range failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SafetensorsError {
    /// A closure file name escaped the source root or was not one component.
    UnsafeSourceName { file_name: String },
    /// A source requirement contained non-canonical SHA-256 text.
    InvalidExpectedDigest { file_name: String, digest: String },
    /// A closure must contain at least one shard.
    EmptyShardSet,
    /// The index and shard list reused the same basename.
    DuplicateSourceName { file_name: String },
    /// I/O failed while reading a verified source or range.
    Io {
        source: String,
        operation: &'static str,
        detail: String,
    },
    /// The actual source length did not match the immutable requirement.
    SourceLength {
        source: String,
        expected: u64,
        actual: u64,
    },
    /// The actual source digest did not match the immutable requirement.
    SourceDigest {
        source: String,
        expected: String,
        actual: String,
    },
    /// The index source is too large for this bounded parser.
    IndexTooLarge {
        index: String,
        declared: u64,
        cap: u64,
    },
    /// A closure invariant was violated after verification.
    MissingVerifiedSource { source: String },
    /// A header prefix was shorter than eight bytes.
    TruncatedHeaderPrefix { shard: String, available: u64 },
    /// The header length exceeded the configured cap before allocation.
    HeaderTooLarge {
        shard: String,
        declared: u64,
        cap: u64,
    },
    /// A declared header did not fit in the verified shard length.
    TruncatedHeader {
        shard: String,
        declared: u64,
        available: u64,
    },
    /// Header bytes were not UTF-8 JSON text.
    HeaderUtf8 { shard: String, detail: String },
    /// Header or index JSON rejected duplicate keys, limits, or syntax.
    Json { source: String, detail: String },
    /// Duplicate object key.  At the header root this rejects a duplicate
    /// tensor name before any range can be exposed.
    DuplicateJsonKey { source: String, path: String },
    /// Header root or a tensor descriptor did not meet its schema.
    HeaderSchema {
        shard: String,
        tensor: Option<String>,
        detail: String,
    },
    /// An unrecognized safetensors dtype was declared.
    UnknownDtype {
        shard: String,
        tensor: String,
        dtype: String,
    },
    /// Shape dimensions exceeded the rank bound.
    RankTooLarge {
        shard: String,
        tensor: String,
        rank: usize,
        cap: usize,
    },
    /// Checked tensor-element multiplication overflowed.
    ShapeProductOverflow { shard: String, tensor: String },
    /// Checked shape-product times dtype-width overflowed.
    TensorByteLengthOverflow { shard: String, tensor: String },
    /// Declared offsets were reversed.
    RangeOrder {
        shard: String,
        tensor: String,
        start: u64,
        end: u64,
    },
    /// Declared range length disagreed with shape-product times dtype bytes.
    RangeLengthMismatch {
        shard: String,
        tensor: String,
        expected: u64,
        actual: u64,
    },
    /// A tensor range overlapped an earlier range.
    RangeOverlap {
        shard: String,
        tensor: String,
        offset: u64,
        prior_end: u64,
    },
    /// A tensor range left unexplained data between ranges.
    RangeGap {
        shard: String,
        tensor: String,
        expected_offset: u64,
        actual_offset: u64,
    },
    /// A tensor range exceeded the verified data section.
    RangeOutOfBounds {
        shard: String,
        tensor: String,
        end: u64,
        data_len: u64,
    },
    /// Data remained after the final declared tensor range.
    TrailingData {
        shard: String,
        offset: u64,
        data_len: u64,
    },
    /// The safetensors index was not UTF-8 JSON text.
    IndexUtf8 { index: String, detail: String },
    /// The safetensors index did not meet its schema.
    IndexSchema { index: String, detail: String },
    /// An index mapping named a shard outside the verified closure.
    IndexMapsUnknownShard {
        index: String,
        tensor: String,
        shard: String,
    },
    /// An index map and shard header disagreed about a tensor.
    IndexShardMismatch {
        index: String,
        tensor: String,
        shard: String,
    },
    /// A shard-header tensor was absent from the index map.
    IndexMissingTensor {
        index: String,
        shard: String,
        tensor: String,
    },
    /// A tensor name appeared in multiple shard headers.
    DuplicateTensorName { shard: String, tensor: String },
    /// A model-family tripwire invalidated the current one-model assumptions.
    DesignAssumptionTripwire { tensor: String },
    /// Options would create an unbounded or zero-cap range surface.
    InvalidOptions,
    /// The pinned model did not contain exactly 201 tensors.
    PinnedTensorCount { expected: usize, actual: usize },
    /// The pinned model did not contain its exact logical payload byte total.
    PinnedPayloadBytes { expected: u64, actual: u64 },
    /// Pinned shard bytes did not separate into the known payload and metadata
    /// totals; this is never treated as unexplained model data.
    PinnedContainerAccounting {
        shard_bytes: u64,
        payload_bytes: u64,
        observed_metadata_bytes: u64,
        expected_shard_bytes: u64,
        expected_metadata_bytes: u64,
    },
    /// A requested tensor was not indexed.
    UnknownTensor { tensor: String },
    /// Row panels are not meaningful for a rank-zero tensor.
    RowPanelOnScalar { tensor: String },
    /// Row panels cannot divide a zero-sized leading axis.
    RowPanelOnZeroRows { tensor: String },
    /// A row panel crossed the validated leading dimension.
    RowPanelOutOfBounds {
        tensor: String,
        start_row: u64,
        row_count: u64,
        rows: u64,
    },
    /// A single range request exceeded the explicit allocation cap.
    RangeReadTooLarge {
        tensor: String,
        requested: u64,
        cap: u64,
    },
    /// A verified file-handle mutex was poisoned by an earlier panic.
    FileHandlePoisoned { source: String },
    /// An expected census listed one tensor twice.
    DuplicateExpectedTensor { tensor: String },
    /// Checked arithmetic protected an address or product calculation.
    Arithmetic {
        invariant: &'static str,
        tensor: Option<String>,
    },
    /// All source tensors were expected to share one exact dtype.
    RequiredDtype {
        shard: String,
        tensor: String,
        expected: SafetensorDtype,
        actual: SafetensorDtype,
    },
}

impl fmt::Display for SafetensorsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "checked safetensors rejection: {self:?}")
    }
}

impl std::error::Error for SafetensorsError {}

fn parse_file_header(
    source: &SourceDigest,
    file: &mut File,
    max_header_bytes: u64,
) -> Result<CheckedShard, SafetensorsError> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| SafetensorsError::Io {
            source: source.file_name.clone(),
            operation: "rewind verified shard",
            detail: error.to_string(),
        })?;
    if source.expected_bytes < 8 {
        return Err(SafetensorsError::TruncatedHeaderPrefix {
            shard: source.file_name.clone(),
            available: source.expected_bytes,
        });
    }
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|error| SafetensorsError::Io {
            source: source.file_name.clone(),
            operation: "read header length",
            detail: error.to_string(),
        })?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > max_header_bytes {
        return Err(SafetensorsError::HeaderTooLarge {
            shard: source.file_name.clone(),
            declared: header_len,
            cap: max_header_bytes,
        });
    }
    let header_end = checked_add(8, header_len, "header prefix + header length", None)?;
    if header_end > source.expected_bytes {
        return Err(SafetensorsError::TruncatedHeader {
            shard: source.file_name.clone(),
            declared: header_len,
            available: source.expected_bytes.saturating_sub(8),
        });
    }
    let header_len_usize =
        usize::try_from(header_len).map_err(|_| SafetensorsError::Arithmetic {
            invariant: "header length to usize",
            tensor: None,
        })?;
    let mut header_bytes = vec![0_u8; header_len_usize];
    file.read_exact(&mut header_bytes)
        .map_err(|error| SafetensorsError::Io {
            source: source.file_name.clone(),
            operation: "read bounded header",
            detail: error.to_string(),
        })?;
    parse_header(
        source.file_name(),
        source.expected_bytes,
        header_len,
        &header_bytes,
    )
}

fn parse_header(
    shard_name: &str,
    file_len: u64,
    header_len: u64,
    header_bytes: &[u8],
) -> Result<CheckedShard, SafetensorsError> {
    let text = std::str::from_utf8(header_bytes).map_err(|error| SafetensorsError::HeaderUtf8 {
        shard: shard_name.to_owned(),
        detail: error.to_string(),
    })?;
    let value = parse_json(shard_name, text)?;
    let root = value
        .as_object()
        .ok_or_else(|| SafetensorsError::HeaderSchema {
            shard: shard_name.to_owned(),
            tensor: None,
            detail: "root must be an object".to_owned(),
        })?;
    let data_start = checked_add(8, header_len, "header data start", None)?;
    if data_start > file_len {
        return Err(SafetensorsError::TruncatedHeader {
            shard: shard_name.to_owned(),
            declared: header_len,
            available: file_len.saturating_sub(8),
        });
    }
    let data_len = file_len - data_start;
    let mut tensors = BTreeMap::new();
    let mut ranges = Vec::new();
    for (name, descriptor) in root {
        if name == "__metadata__" {
            if !descriptor.is_object() {
                return Err(SafetensorsError::HeaderSchema {
                    shard: shard_name.to_owned(),
                    tensor: None,
                    detail: "__metadata__ must be an object".to_owned(),
                });
            }
            continue;
        }
        if name.is_empty() {
            return Err(SafetensorsError::HeaderSchema {
                shard: shard_name.to_owned(),
                tensor: Some(name.clone()),
                detail: "tensor name must be non-empty".to_owned(),
            });
        }
        let record = parse_tensor_record(shard_name, name, descriptor)?;
        let end = checked_add(
            record.relative_offset,
            record.facts.len,
            "tensor offset + length",
            Some(name),
        )?;
        ranges.push((record.relative_offset, end, name.clone()));
        if tensors.insert(name.clone(), record).is_some() {
            return Err(SafetensorsError::DuplicateTensorName {
                shard: shard_name.to_owned(),
                tensor: name.clone(),
            });
        }
    }
    ranges.sort_unstable_by(|left, right| left.0.cmp(&right.0).then_with(|| left.2.cmp(&right.2)));
    let mut cursor = 0_u64;
    for (start, end, name) in ranges {
        if start < cursor {
            return Err(SafetensorsError::RangeOverlap {
                shard: shard_name.to_owned(),
                tensor: name,
                offset: start,
                prior_end: cursor,
            });
        }
        if start > cursor {
            return Err(SafetensorsError::RangeGap {
                shard: shard_name.to_owned(),
                tensor: name,
                expected_offset: cursor,
                actual_offset: start,
            });
        }
        if end > data_len {
            return Err(SafetensorsError::RangeOutOfBounds {
                shard: shard_name.to_owned(),
                tensor: name,
                end,
                data_len,
            });
        }
        cursor = end;
    }
    if cursor != data_len {
        return Err(SafetensorsError::TrailingData {
            shard: shard_name.to_owned(),
            offset: cursor,
            data_len,
        });
    }
    Ok(CheckedShard {
        shard_name: shard_name.to_owned(),
        file_len,
        data_start,
        tensors,
    })
}

fn parse_tensor_record(
    shard: &str,
    name: &str,
    descriptor: &Value,
) -> Result<TensorRecord, SafetensorsError> {
    let object = descriptor
        .as_object()
        .ok_or_else(|| SafetensorsError::HeaderSchema {
            shard: shard.to_owned(),
            tensor: Some(name.to_owned()),
            detail: "descriptor must be an object".to_owned(),
        })?;
    for field in ["dtype", "shape", "data_offsets"] {
        if !object.contains_key(field) {
            return Err(SafetensorsError::HeaderSchema {
                shard: shard.to_owned(),
                tensor: Some(name.to_owned()),
                detail: format!("missing required field {field}"),
            });
        }
    }
    if object.len() != 3 {
        return Err(SafetensorsError::HeaderSchema {
            shard: shard.to_owned(),
            tensor: Some(name.to_owned()),
            detail: "descriptor may contain only dtype, shape, and data_offsets".to_owned(),
        });
    }
    let dtype_value = object
        .get("dtype")
        .ok_or_else(|| SafetensorsError::HeaderSchema {
            shard: shard.to_owned(),
            tensor: Some(name.to_owned()),
            detail: "missing required field dtype".to_owned(),
        })?;
    let dtype_text = dtype_value
        .as_str()
        .ok_or_else(|| SafetensorsError::HeaderSchema {
            shard: shard.to_owned(),
            tensor: Some(name.to_owned()),
            detail: "dtype must be a string".to_owned(),
        })?;
    let dtype =
        SafetensorDtype::parse(dtype_text).ok_or_else(|| SafetensorsError::UnknownDtype {
            shard: shard.to_owned(),
            tensor: name.to_owned(),
            dtype: dtype_text.to_owned(),
        })?;
    let shape_value = object
        .get("shape")
        .ok_or_else(|| SafetensorsError::HeaderSchema {
            shard: shard.to_owned(),
            tensor: Some(name.to_owned()),
            detail: "missing required field shape".to_owned(),
        })?;
    let shape_values = shape_value
        .as_array()
        .ok_or_else(|| SafetensorsError::HeaderSchema {
            shard: shard.to_owned(),
            tensor: Some(name.to_owned()),
            detail: "shape must be an array of unsigned integers".to_owned(),
        })?;
    if shape_values.len() > MAX_TENSOR_RANK {
        return Err(SafetensorsError::RankTooLarge {
            shard: shard.to_owned(),
            tensor: name.to_owned(),
            rank: shape_values.len(),
            cap: MAX_TENSOR_RANK,
        });
    }
    let mut shape = Vec::with_capacity(shape_values.len());
    let mut elements = 1_u64;
    for value in shape_values {
        let dimension = value
            .as_u64()
            .ok_or_else(|| SafetensorsError::HeaderSchema {
                shard: shard.to_owned(),
                tensor: Some(name.to_owned()),
                detail: "shape dimensions must be unsigned integers".to_owned(),
            })?;
        elements = elements.checked_mul(dimension).ok_or_else(|| {
            SafetensorsError::ShapeProductOverflow {
                shard: shard.to_owned(),
                tensor: name.to_owned(),
            }
        })?;
        shape.push(dimension);
    }
    let expected_len = elements.checked_mul(dtype.byte_width()).ok_or_else(|| {
        SafetensorsError::TensorByteLengthOverflow {
            shard: shard.to_owned(),
            tensor: name.to_owned(),
        }
    })?;
    let offsets_value =
        object
            .get("data_offsets")
            .ok_or_else(|| SafetensorsError::HeaderSchema {
                shard: shard.to_owned(),
                tensor: Some(name.to_owned()),
                detail: "missing required field data_offsets".to_owned(),
            })?;
    let offsets = offsets_value
        .as_array()
        .ok_or_else(|| SafetensorsError::HeaderSchema {
            shard: shard.to_owned(),
            tensor: Some(name.to_owned()),
            detail: "data_offsets must be a two-element unsigned-integer array".to_owned(),
        })?;
    if offsets.len() != 2 {
        return Err(SafetensorsError::HeaderSchema {
            shard: shard.to_owned(),
            tensor: Some(name.to_owned()),
            detail: "data_offsets must contain exactly two values".to_owned(),
        });
    }
    let start_value = offsets
        .first()
        .ok_or_else(|| SafetensorsError::HeaderSchema {
            shard: shard.to_owned(),
            tensor: Some(name.to_owned()),
            detail: "data_offsets[0] must be an unsigned integer".to_owned(),
        })?;
    let start = start_value
        .as_u64()
        .ok_or_else(|| SafetensorsError::HeaderSchema {
            shard: shard.to_owned(),
            tensor: Some(name.to_owned()),
            detail: "data_offsets[0] must be an unsigned integer".to_owned(),
        })?;
    let end_value = offsets
        .get(1)
        .ok_or_else(|| SafetensorsError::HeaderSchema {
            shard: shard.to_owned(),
            tensor: Some(name.to_owned()),
            detail: "data_offsets[1] must be an unsigned integer".to_owned(),
        })?;
    let end = end_value
        .as_u64()
        .ok_or_else(|| SafetensorsError::HeaderSchema {
            shard: shard.to_owned(),
            tensor: Some(name.to_owned()),
            detail: "data_offsets[1] must be an unsigned integer".to_owned(),
        })?;
    if end < start {
        return Err(SafetensorsError::RangeOrder {
            shard: shard.to_owned(),
            tensor: name.to_owned(),
            start,
            end,
        });
    }
    let actual_len = end - start;
    if actual_len != expected_len {
        return Err(SafetensorsError::RangeLengthMismatch {
            shard: shard.to_owned(),
            tensor: name.to_owned(),
            expected: expected_len,
            actual: actual_len,
        });
    }
    Ok(TensorRecord {
        facts: TensorCensusEntry {
            name: name.to_owned(),
            dtype,
            shape,
            len: actual_len,
        },
        relative_offset: start,
    })
}

fn parse_json(source: &str, text: &str) -> Result<Value, SafetensorsError> {
    let limits = ParseLimits {
        max_depth: 16,
        max_string_bytes: 16 * 1024,
    };
    canonjson::parse_str_with_limits(text, limits).map_err(|error| match error {
        CanonJsonError::DuplicateKey { path } => SafetensorsError::DuplicateJsonKey {
            source: source.to_owned(),
            path: path.to_string(),
        },
        error => SafetensorsError::Json {
            source: source.to_owned(),
            detail: error.to_string(),
        },
    })
}

fn verify_source_file(
    source: &SourceDigest,
    file: &mut File,
) -> Result<HashReceipt, SafetensorsError> {
    let length = file
        .metadata()
        .map_err(|error| SafetensorsError::Io {
            source: source.file_name.clone(),
            operation: "stat verified source",
            detail: error.to_string(),
        })?
        .len();
    if length != source.expected_bytes {
        return Err(SafetensorsError::SourceLength {
            source: source.file_name.clone(),
            expected: source.expected_bytes,
            actual: length,
        });
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| SafetensorsError::Io {
            source: source.file_name.clone(),
            operation: "rewind source hash",
            detail: error.to_string(),
        })?;
    let started = Instant::now();
    let mut hasher = Sha256::new();
    let mut hashed_bytes = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| SafetensorsError::Io {
                source: source.file_name.clone(),
                operation: "stream source hash",
                detail: error.to_string(),
            })?;
        if read == 0 {
            break;
        }
        let chunk = buffer
            .get(..read)
            .ok_or_else(|| SafetensorsError::Arithmetic {
                invariant: "streaming read within hash buffer",
                tensor: None,
            })?;
        hasher.update(chunk);
        hashed_bytes = checked_add(
            hashed_bytes,
            u64::try_from(read).map_err(|_| SafetensorsError::Arithmetic {
                invariant: "streaming read length to u64",
                tensor: None,
            })?,
            "streaming hash byte count",
            None,
        )?;
    }
    let digest = hasher.finalize();
    let actual_sha256 = hex_digest(&digest);
    let receipt = verify_observed_source(source, hashed_bytes, actual_sha256, started.elapsed())?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| SafetensorsError::Io {
            source: source.file_name.clone(),
            operation: "rewind verified source",
            detail: error.to_string(),
        })?;
    Ok(receipt)
}

fn verify_observed_source(
    source: &SourceDigest,
    actual_bytes: u64,
    actual_sha256: String,
    elapsed: Duration,
) -> Result<HashReceipt, SafetensorsError> {
    if actual_bytes != source.expected_bytes {
        return Err(SafetensorsError::SourceLength {
            source: source.file_name.clone(),
            expected: source.expected_bytes,
            actual: actual_bytes,
        });
    }
    if actual_sha256 != source.expected_sha256 {
        return Err(SafetensorsError::SourceDigest {
            source: source.file_name.clone(),
            expected: source.expected_sha256.clone(),
            actual: actual_sha256,
        });
    }
    Ok(HashReceipt {
        file_name: source.file_name.clone(),
        bytes: actual_bytes,
        sha256: source.expected_sha256.clone(),
        elapsed,
    })
}

fn read_verified_index(
    source: &SourceDigest,
    file: &mut File,
) -> Result<Vec<u8>, SafetensorsError> {
    if source.expected_bytes > MAX_INDEX_BYTES {
        return Err(SafetensorsError::IndexTooLarge {
            index: source.file_name.clone(),
            declared: source.expected_bytes,
            cap: MAX_INDEX_BYTES,
        });
    }
    let len = usize::try_from(source.expected_bytes).map_err(|_| SafetensorsError::Arithmetic {
        invariant: "index length to usize",
        tensor: None,
    })?;
    let mut bytes = vec![0_u8; len];
    file.seek(SeekFrom::Start(0))
        .map_err(|error| SafetensorsError::Io {
            source: source.file_name.clone(),
            operation: "rewind verified index",
            detail: error.to_string(),
        })?;
    file.read_exact(&mut bytes)
        .map_err(|error| SafetensorsError::Io {
            source: source.file_name.clone(),
            operation: "read verified index",
            detail: error.to_string(),
        })?;
    Ok(bytes)
}

fn decode_header_len(shard: &str, bytes: &[u8]) -> Result<u64, SafetensorsError> {
    if bytes.len() < 8 {
        return Err(SafetensorsError::TruncatedHeaderPrefix {
            shard: shard.to_owned(),
            available: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        });
    }
    let prefix_bytes = bytes
        .get(..8)
        .ok_or_else(|| SafetensorsError::TruncatedHeaderPrefix {
            shard: shard.to_owned(),
            available: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        })?;
    let mut prefix = [0_u8; 8];
    prefix.copy_from_slice(prefix_bytes);
    Ok(u64::from_le_bytes(prefix))
}

fn reject_oq1_tripwires<'a>(
    names: impl IntoIterator<Item = &'a String>,
) -> Result<(), SafetensorsError> {
    for name in names {
        let lower = name.to_ascii_lowercase();
        if ["mhc", "ngram", "depth", "loopsplit"]
            .iter()
            .any(|family| lower.contains(family))
        {
            return Err(SafetensorsError::DesignAssumptionTripwire {
                tensor: name.clone(),
            });
        }
    }
    Ok(())
}

fn source_path(root: &Path, source: &SourceDigest) -> Result<PathBuf, SafetensorsError> {
    if !safe_file_name(source.file_name()) {
        return Err(SafetensorsError::UnsafeSourceName {
            file_name: source.file_name.clone(),
        });
    }
    Ok(root.join(source.file_name()))
}

fn safe_file_name(name: &str) -> bool {
    let path = Path::new(name);
    !name.is_empty()
        && path.components().count() == 1
        && matches!(path.components().next(), Some(Component::Normal(_)))
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_digest(&digest)
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(hex_digit(byte >> 4));
        output.push(hex_digit(byte & 0x0f));
    }
    output
}

fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => char::from(b'0' + nibble),
        10..=15 => char::from(b'a' + (nibble - 10)),
        _ => '?',
    }
}

fn checked_add(
    left: u64,
    right: u64,
    invariant: &'static str,
    tensor: Option<&str>,
) -> Result<u64, SafetensorsError> {
    left.checked_add(right)
        .ok_or_else(|| SafetensorsError::Arithmetic {
            invariant,
            tensor: tensor.map(str::to_owned),
        })
}

fn checked_mul(
    left: u64,
    right: u64,
    invariant: &'static str,
    tensor: Option<&str>,
) -> Result<u64, SafetensorsError> {
    left.checked_mul(right)
        .ok_or_else(|| SafetensorsError::Arithmetic {
            invariant,
            tensor: tensor.map(str::to_owned),
        })
}
