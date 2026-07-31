//! Checked owned-buffer reader for the frozen `.fnlpq` v1 envelope.
//!
//! This is deliberately the hostile-input counterpart of the canonical
//! writer.  The prelude is checked before a file path causes any
//! attacker-sized allocation; the bounded and hash-checked header is parsed
//! through the canonical JSON chokepoint before the directory is trusted.
//! `FNLP_MMAP=1` is rejected explicitly until a ratified safe suite mapping
//! surface exists, rather than adding an unreviewed unsafe dependency.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::artifact::format::{
    framed_sha256, ArchTarget, SectionKind, FORMAT_VERSION, MAGIC, MAX_ALIGNMENT, MAX_ENTRIES,
    MAX_FILE_BYTES, MAX_HEADER_BYTES, PRELUDE_BYTES, SECTION_DIRECTORY_ENTRY_BYTES,
};
use crate::canonjson::{self, ParseLimits};

const REQUIRED_KINDS: [SectionKind; 8] = [
    SectionKind::GenericTensorPayload,
    SectionKind::GenericTensorScales,
    SectionKind::GenericTensorRowSums,
    SectionKind::TokenizerModel,
    SectionKind::ModelConfig,
    SectionKind::TokenizerConfig,
    SectionKind::ChatTemplate,
    SectionKind::LicenseBundle,
];

/// Typed, location-bearing failures from the untrusted artifact boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FnlpqReadError {
    /// Filesystem I/O or regular-file checks failed before parsing.
    Io {
        operation: &'static str,
        detail: String,
    },
    /// The requested mmap mode has no ratified safe implementation yet.
    MmapUnavailable,
    /// The fixed prelude was truncated or violated its v1 contract.
    Prelude { field: &'static str, reason: String },
    /// The bounded canonical header was malformed or schema-invalid.
    Header { field: String, reason: String },
    /// A fixed directory entry was malformed.
    Directory {
        ordinal: usize,
        field: &'static str,
        reason: String,
    },
    /// A stored section or tensor mapping was malformed.
    Section {
        ordinal: usize,
        name: String,
        reason: String,
    },
    /// No declared packing set exists for the requested target.
    MissingPackingDerivation {
        requested_target: String,
        available_ids: Vec<String>,
    },
}

impl fmt::Display for FnlpqReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { operation, detail } => write!(formatter, "fnlpq {operation}: {detail}"),
            Self::MmapUnavailable => formatter.write_str(
                "FNLP_MMAP=1 requested, but no ratified safe mmap surface is available; use owned-buffer mode",
            ),
            Self::Prelude { field, reason } => write!(formatter, "fnlpq prelude.{field}: {reason}"),
            Self::Header { field, reason } => write!(formatter, "fnlpq header {field}: {reason}"),
            Self::Directory {
                ordinal,
                field,
                reason,
            } => write!(formatter, "fnlpq directory[{ordinal}].{field}: {reason}"),
            Self::Section {
                ordinal,
                name,
                reason,
            } => write!(
                formatter,
                "fnlpq section[{ordinal}] {name:?}: {reason}"
            ),
            Self::MissingPackingDerivation {
                requested_target,
                available_ids,
            } => write!(
                formatter,
                "required packing derivation missing for target {requested_target}; available sets: {}",
                available_ids.join(",")
            ),
        }
    }
}

impl Error for FnlpqReadError {}

/// A validated, owned `.fnlpq` file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FnlpqArtifact {
    bytes: Vec<u8>,
    prelude: CheckedPrelude,
    header: CheckedHeader,
    sections: Vec<CheckedSection>,
}

/// Fixed prelude fields after their version and file-length checks succeed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedPrelude {
    /// Exact canonical-header byte length.
    pub header_len: u64,
    /// Number of directory entries.
    pub section_count: u64,
    /// Number of logical tensor declarations.
    pub tensor_count: u64,
    /// Observed and declared regular-file length.
    pub file_len: u64,
    /// SHA-256 over exactly `header_len` header bytes.
    pub header_sha256: [u8; 32],
}

/// A directory record after checked range, alignment, and digest validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedSection {
    /// Header/directory ordinal.
    pub ordinal: u64,
    /// Unique authority-bearing section name.
    pub name: String,
    /// Closed v1 kind.
    pub kind: SectionKind,
    /// Absolute stored-byte offset.
    pub file_offset: u64,
    /// Exact stored and logical byte length in v1.
    pub stored_len: u64,
    /// Required power-of-two alignment.
    pub alignment: u64,
    /// Domain-framed stored-byte digest.
    pub stored_sha256: [u8; 32],
}

/// A validated logical tensor declaration and generic byte mappings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedTensor {
    /// Sorted unique tensor name.
    pub name: String,
    /// `bf16`, `f32`, or `i8` in v1.
    pub canonical_dtype: String,
    /// Bounded logical shape.
    pub shape: Vec<u32>,
    /// Converter-provided canonical logical identity.
    pub canonical_logical_sha256: String,
    /// Fixed generic quantization recipe ID.
    pub quantization: String,
    /// Mapping into the generic payload section.
    pub data: CheckedMapping,
    /// Mapping into the generic scale section.
    pub scale: CheckedMapping,
    /// Mapping into the generic row-sum section.
    pub row_sum: CheckedMapping,
}

/// A validated byte mapping inside one declared section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedMapping {
    /// Directory section ordinal.
    pub section_ordinal: u64,
    /// Offset relative to that section's stored bytes.
    pub offset: u64,
    /// Mapping byte length.
    pub len: u64,
}

/// A retained packing representation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedPackingSet {
    /// Unique packing-set ID.
    pub id: String,
    /// Closed target spelling.
    pub target: String,
    /// Declared, hash-checked section references.
    pub representations: Vec<CheckedPackingRepresentation>,
}

/// A packing-set representation bound to one stored directory section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedPackingRepresentation {
    /// Directory section ordinal.
    pub section_ordinal: u64,
    /// Exact retained stored-byte cost.
    pub byte_cost: u64,
    /// Stored-section digest also recorded in the header.
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CheckedHeader {
    model_id: String,
    revision: String,
    recipe_id: String,
    source_root_sha256: String,
    logical_model_sha256: String,
    packing_set_sha256: String,
    license_bundle_sha256: String,
    sources: Vec<HeaderSource>,
    sections: Vec<HeaderSection>,
    tensors: Vec<CheckedTensor>,
    packing_sets: Vec<CheckedPackingSet>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct HeaderSource {
    name: String,
    section_ordinal: u64,
    sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct HeaderSection {
    name: String,
    kind: SectionKind,
}

impl FnlpqArtifact {
    /// Validate an already-owned byte buffer without borrowing untrusted ranges.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, FnlpqReadError> {
        let prelude = validate_prelude(&bytes, bytes.len() as u64)?;
        let header_end = checked_metadata_end(&prelude)?;
        let header_start = PRELUDE_BYTES;
        let header_end_usize = usize_from_u64(header_end, "prelude.metadata_end")?;
        let header_bytes = bytes
            .get(header_start..header_start + prelude.header_len as usize)
            .ok_or_else(|| FnlpqReadError::Prelude {
                field: "header_len",
                reason: "header range is outside observed file bytes".to_owned(),
            })?;
        let observed_header_sha256: [u8; 32] = Sha256::digest(header_bytes).into();
        if observed_header_sha256 != prelude.header_sha256 {
            return Err(FnlpqReadError::Header {
                field: "prelude.header_sha256".to_owned(),
                reason: format!(
                    "mismatch expected={} actual={}",
                    hex_lower(&prelude.header_sha256),
                    hex_lower(&observed_header_sha256)
                ),
            });
        }
        let header = parse_header(header_bytes, &prelude)?;
        let directory_start = header_start + prelude.header_len as usize;
        let sections =
            parse_directory(&bytes, directory_start, header_end_usize, &prelude, &header)?;
        validate_header_relationships(&bytes, &header, &sections)?;
        Ok(Self {
            bytes,
            prelude,
            header,
            sections,
        })
    }

    /// Open and validate a regular artifact file in owned-buffer mode.
    ///
    /// `FNLP_MMAP=1` is an explicit typed refusal until the project has a
    /// ratified safe mapping surface.  It is never silently ignored.
    pub fn open_owned(path: impl AsRef<Path>) -> Result<Self, FnlpqReadError> {
        if env::var_os("FNLP_MMAP").as_deref() == Some(std::ffi::OsStr::new("1")) {
            return Err(FnlpqReadError::MmapUnavailable);
        }
        let path = path.as_ref();
        let link_metadata = fs::symlink_metadata(path).map_err(|error| FnlpqReadError::Io {
            operation: "symlink_metadata",
            detail: error.to_string(),
        })?;
        if link_metadata.file_type().is_symlink() {
            return Err(FnlpqReadError::Io {
                operation: "regular-file check",
                detail: "symlink artifacts are forbidden".to_owned(),
            });
        }
        if !link_metadata.file_type().is_file() {
            return Err(FnlpqReadError::Io {
                operation: "regular-file check",
                detail: "artifact must be a regular file".to_owned(),
            });
        }
        let mut file = File::open(path).map_err(|error| FnlpqReadError::Io {
            operation: "open",
            detail: error.to_string(),
        })?;
        let observed_len = file
            .metadata()
            .map_err(|error| FnlpqReadError::Io {
                operation: "metadata",
                detail: error.to_string(),
            })?
            .len();
        let mut fixed = [0_u8; PRELUDE_BYTES];
        file.read_exact(&mut fixed)
            .map_err(|error| FnlpqReadError::Io {
                operation: "read fixed prelude",
                detail: error.to_string(),
            })?;
        let prelude = validate_prelude(&fixed, observed_len)?;
        // Read and authenticate the only bounded variable-sized object before
        // reserving the whole owned artifact buffer.
        let mut header_bytes = vec![0_u8; prelude.header_len as usize];
        file.read_exact(&mut header_bytes)
            .map_err(|error| FnlpqReadError::Io {
                operation: "read bounded header",
                detail: error.to_string(),
            })?;
        let header_sha256: [u8; 32] = Sha256::digest(&header_bytes).into();
        if header_sha256 != prelude.header_sha256 {
            return Err(FnlpqReadError::Header {
                field: "prelude.header_sha256".to_owned(),
                reason: "mismatch while preflighting bounded header".to_owned(),
            });
        }
        let _ = parse_header(&header_bytes, &prelude)?;
        let file_len = usize_from_u64(prelude.file_len, "prelude.file_len")?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(file_len)
            .map_err(|error| FnlpqReadError::Io {
                operation: "reserve owned artifact buffer",
                detail: error.to_string(),
            })?;
        bytes.resize(file_len, 0);
        file.seek(SeekFrom::Start(0))
            .map_err(|error| FnlpqReadError::Io {
                operation: "rewind after header preflight",
                detail: error.to_string(),
            })?;
        file.read_exact(&mut bytes)
            .map_err(|error| FnlpqReadError::Io {
                operation: "read owned artifact buffer",
                detail: error.to_string(),
            })?;
        let final_len = file
            .metadata()
            .map_err(|error| FnlpqReadError::Io {
                operation: "post-read metadata",
                detail: error.to_string(),
            })?
            .len();
        if final_len != observed_len {
            return Err(FnlpqReadError::Io {
                operation: "regular-file length stability",
                detail: format!("changed during read: before={observed_len} after={final_len}"),
            });
        }
        Self::from_bytes(bytes)
    }

    /// Checked fixed prelude for diagnostics and receipts.
    pub fn prelude(&self) -> &CheckedPrelude {
        &self.prelude
    }

    /// Model identifier declared by the closed canonical header schema.
    pub fn model_id(&self) -> &str {
        &self.header.model_id
    }

    /// Immutable source revision declared by the closed canonical header schema.
    pub fn revision(&self) -> &str {
        &self.header.revision
    }

    /// Immutable converter recipe ID.
    pub fn recipe_id(&self) -> &str {
        &self.header.recipe_id
    }

    /// Canonical source-manifest identity supplied by the conversion receipt.
    pub fn source_root_sha256(&self) -> &str {
        &self.header.source_root_sha256
    }

    /// Packing-independent computational model identity.
    pub fn logical_model_sha256(&self) -> &str {
        &self.header.logical_model_sha256
    }

    /// Identity of the retained physical representation set.
    pub fn packing_set_sha256(&self) -> &str {
        &self.header.packing_set_sha256
    }

    /// Exact legal-bundle identity, deliberately distinct from model semantics.
    pub fn license_bundle_sha256(&self) -> &str {
        &self.header.license_bundle_sha256
    }

    /// Validated logical tensor declarations in ASCII-name order.
    pub fn tensors(&self) -> &[CheckedTensor] {
        &self.header.tensors
    }

    /// Checked binary directory entries in physical file order.
    pub fn sections(&self) -> &[CheckedSection] {
        &self.sections
    }

    /// Borrow exact stored bytes only after all format invariants have passed.
    pub fn section_bytes(&self, ordinal: u64) -> Option<&[u8]> {
        let section = self.sections.get(ordinal as usize)?;
        let start = section.file_offset as usize;
        let end = start.checked_add(section.stored_len as usize)?;
        self.bytes.get(start..end)
    }

    /// Return the declared packing set for one observed target.  No native
    /// target silently falls back to a differently-qualified representation.
    pub fn select_packing(&self, target: ArchTarget) -> Result<&CheckedPackingSet, FnlpqReadError> {
        let requested_target = target.header_name().to_owned();
        self.header
            .packing_sets
            .iter()
            .find(|set| set.target == requested_target)
            .ok_or_else(|| FnlpqReadError::MissingPackingDerivation {
                requested_target,
                available_ids: self
                    .header
                    .packing_sets
                    .iter()
                    .map(|set| set.id.clone())
                    .collect(),
            })
    }
}

fn validate_prelude(bytes: &[u8], observed_len: u64) -> Result<CheckedPrelude, FnlpqReadError> {
    if bytes.len() < PRELUDE_BYTES {
        return Err(FnlpqReadError::Prelude {
            field: "range [0,80)",
            reason: format!("truncated prelude: observed {} bytes", bytes.len()),
        });
    }
    if bytes[..8] != MAGIC {
        return Err(FnlpqReadError::Prelude {
            field: "magic",
            reason: "expected FNLPQ\\0\\0\\x01".to_owned(),
        });
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("fixed prelude range"));
    if version != FORMAT_VERSION {
        return Err(FnlpqReadError::Prelude {
            field: "format_version",
            reason: format!("expected {FORMAT_VERSION}, found {version}"),
        });
    }
    let required_flags = u32::from_le_bytes(bytes[12..16].try_into().expect("fixed prelude range"));
    if required_flags != 0 {
        return Err(FnlpqReadError::Prelude {
            field: "required_flags",
            reason: format!("unknown required flags 0x{required_flags:08x}"),
        });
    }
    let header_len = u64::from_le_bytes(bytes[16..24].try_into().expect("fixed prelude range"));
    if header_len > MAX_HEADER_BYTES {
        return Err(FnlpqReadError::Prelude {
            field: "header_len",
            reason: format!("cap={MAX_HEADER_BYTES} observed={header_len}"),
        });
    }
    let section_count = u64::from_le_bytes(bytes[24..32].try_into().expect("fixed prelude range"));
    if section_count == 0 || section_count > MAX_ENTRIES as u64 {
        return Err(FnlpqReadError::Prelude {
            field: "section_count",
            reason: format!("must be 1..={MAX_ENTRIES}, found {section_count}"),
        });
    }
    let tensor_count = u64::from_le_bytes(bytes[32..40].try_into().expect("fixed prelude range"));
    if tensor_count == 0 || tensor_count > MAX_ENTRIES as u64 {
        return Err(FnlpqReadError::Prelude {
            field: "tensor_count",
            reason: format!("must be 1..={MAX_ENTRIES}, found {tensor_count}"),
        });
    }
    let file_len = u64::from_le_bytes(bytes[40..48].try_into().expect("fixed prelude range"));
    if file_len > MAX_FILE_BYTES {
        return Err(FnlpqReadError::Prelude {
            field: "file_len",
            reason: format!("cap={MAX_FILE_BYTES} observed={file_len}"),
        });
    }
    if file_len != observed_len {
        return Err(FnlpqReadError::Prelude {
            field: "file_len",
            reason: format!("expected={file_len} observed={observed_len}"),
        });
    }
    let mut header_sha256 = [0_u8; 32];
    header_sha256.copy_from_slice(&bytes[48..80]);
    let checked = CheckedPrelude {
        header_len,
        section_count,
        tensor_count,
        file_len,
        header_sha256,
    };
    let metadata_end = checked_metadata_end(&checked)?;
    if metadata_end > file_len {
        return Err(FnlpqReadError::Prelude {
            field: "header_len/section_count",
            reason: format!("metadata end {metadata_end} exceeds file_len {file_len}"),
        });
    }
    Ok(checked)
}

fn checked_metadata_end(prelude: &CheckedPrelude) -> Result<u64, FnlpqReadError> {
    let directory_len = prelude
        .section_count
        .checked_mul(SECTION_DIRECTORY_ENTRY_BYTES as u64)
        .ok_or_else(|| FnlpqReadError::Prelude {
            field: "section_count",
            reason: "directory byte-count overflow".to_owned(),
        })?;
    (PRELUDE_BYTES as u64)
        .checked_add(prelude.header_len)
        .and_then(|value| value.checked_add(directory_len))
        .ok_or_else(|| FnlpqReadError::Prelude {
            field: "header_len/section_count",
            reason: "metadata end overflow".to_owned(),
        })
}

fn parse_header(bytes: &[u8], prelude: &CheckedPrelude) -> Result<CheckedHeader, FnlpqReadError> {
    if bytes.starts_with(&[0xef, 0xbb, 0xbf]) {
        return Err(header_error("bytes", "UTF-8 BOM is forbidden"));
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|error| header_error("bytes", format!("invalid UTF-8: {error}")))?;
    let limits = ParseLimits {
        max_depth: 64,
        max_string_bytes: MAX_HEADER_BYTES as usize,
    };
    let canonical = canonjson::canonicalize_str(text, limits)
        .map_err(|error| header_error("canonical_json", error.to_string()))?;
    if canonical != bytes {
        return Err(header_error(
            "canonical_json",
            "header is not the exact pinned canonical byte form",
        ));
    }
    let value = canonjson::parse_str_with_limits(text, limits)
        .map_err(|error| header_error("canonical_json", error.to_string()))?;
    let root = exact_object(
        &value,
        "$",
        &[
            "header_schema",
            "license_bundle_sha256",
            "limits_profile",
            "logical_model_sha256",
            "materialized_sources",
            "model",
            "packing_set_sha256",
            "packing_sets",
            "recipe_id",
            "sections",
            "source_root_sha256",
            "tensors",
        ],
    )?;
    exact_string(root, "header_schema", "$")
        .and_then(|value| ensure_exact(value, "fnlpq-header-v1", "header_schema"))?;
    exact_string(root, "limits_profile", "$")
        .and_then(|value| ensure_exact(value, "fnlpq-limits-v1", "limits_profile"))?;
    let model = exact_object(
        value_at(root, "model", "$")?,
        "$/model",
        &["model_id", "revision"],
    )?;
    let model_id = authority(
        exact_string(model, "model_id", "$/model")?,
        "model.model_id",
    )?;
    let revision = lowercase_hex(
        exact_string(model, "revision", "$/model")?,
        40,
        "model.revision",
    )?;
    let recipe_id = authority(exact_string(root, "recipe_id", "$")?, "recipe_id")?;
    let source_root_sha256 = lowercase_hex(
        exact_string(root, "source_root_sha256", "$")?,
        64,
        "source_root_sha256",
    )?;
    let logical_model_sha256 = lowercase_hex(
        exact_string(root, "logical_model_sha256", "$")?,
        64,
        "logical_model_sha256",
    )?;
    let packing_set_sha256 = lowercase_hex(
        exact_string(root, "packing_set_sha256", "$")?,
        64,
        "packing_set_sha256",
    )?;
    let license_bundle_sha256 = lowercase_hex(
        exact_string(root, "license_bundle_sha256", "$")?,
        64,
        "license_bundle_sha256",
    )?;
    let sections = parse_header_sections(value_at(root, "sections", "$")?)?;
    if sections.len() != prelude.section_count as usize {
        return Err(header_error(
            "sections",
            format!(
                "length {} differs from prelude.section_count {}",
                sections.len(),
                prelude.section_count
            ),
        ));
    }
    let tensors = parse_tensors(value_at(root, "tensors", "$")?)?;
    if tensors.len() != prelude.tensor_count as usize {
        return Err(header_error(
            "tensors",
            format!(
                "length {} differs from prelude.tensor_count {}",
                tensors.len(),
                prelude.tensor_count
            ),
        ));
    }
    let sources = parse_sources(value_at(root, "materialized_sources", "$")?)?;
    let packing_sets = parse_packing_sets(value_at(root, "packing_sets", "$")?)?;
    Ok(CheckedHeader {
        model_id,
        revision,
        recipe_id,
        source_root_sha256,
        logical_model_sha256,
        packing_set_sha256,
        license_bundle_sha256,
        sources,
        sections,
        tensors,
        packing_sets,
    })
}

fn parse_header_sections(value: &Value) -> Result<Vec<HeaderSection>, FnlpqReadError> {
    let array = array_at(value, "sections")?;
    if array.is_empty() || array.len() > MAX_ENTRIES {
        return Err(header_error(
            "sections",
            format!("length must be 1..={MAX_ENTRIES}"),
        ));
    }
    let mut names = BTreeSet::new();
    let mut kinds = BTreeMap::new();
    let mut output = Vec::with_capacity(array.len());
    for (index, item) in array.iter().enumerate() {
        let path = format!("sections/{index}");
        let object = exact_object(item, &path, &["kind", "name", "ordinal", "required"])?;
        let ordinal = exact_u64(object, "ordinal", &path)?;
        if ordinal != index as u64 {
            return Err(header_error(
                format!("{path}/ordinal"),
                format!("must be contiguous ordinal {index}"),
            ));
        }
        let name = authority(exact_string(object, "name", &path)?, "section.name")?;
        if !names.insert(name.clone()) {
            return Err(header_error(
                format!("{path}/name"),
                "duplicate section name",
            ));
        }
        let kind = section_kind_name(exact_string(object, "kind", &path)?, &path)?;
        *kinds.entry(kind).or_insert(0_usize) += 1;
        let required = exact_bool(object, "required", &path)?;
        if required != required_singleton(kind) {
            return Err(header_error(
                format!("{path}/required"),
                "must match v1 section-kind cardinality",
            ));
        }
        output.push(HeaderSection {
            name,
            kind,
        });
    }
    for kind in REQUIRED_KINDS {
        if kinds.get(&kind).copied() != Some(1) {
            return Err(header_error(
                "sections",
                format!(
                    "required section kind {} must occur exactly once",
                    kind.header_name()
                ),
            ));
        }
    }
    Ok(output)
}

fn parse_sources(value: &Value) -> Result<Vec<HeaderSource>, FnlpqReadError> {
    const EXPECTED: [&str; 4] = [
        "model_config",
        "tokenizer_model",
        "tokenizer_config",
        "chat_template",
    ];
    let array = array_at(value, "materialized_sources")?;
    if array.len() != EXPECTED.len() {
        return Err(header_error(
            "materialized_sources",
            "must contain exactly four materialized source entries",
        ));
    }
    let mut output = Vec::with_capacity(array.len());
    for (index, item) in array.iter().enumerate() {
        let path = format!("materialized_sources/{index}");
        let object = exact_object(item, &path, &["name", "section_ordinal", "sha256"])?;
        let name = authority(
            exact_string(object, "name", &path)?,
            "materialized_source.name",
        )?;
        if name != EXPECTED[index] {
            return Err(header_error(
                format!("{path}/name"),
                format!("expected {}", EXPECTED[index]),
            ));
        }
        output.push(HeaderSource {
            name,
            section_ordinal: exact_u64(object, "section_ordinal", &path)?,
            sha256: lowercase_hex(exact_string(object, "sha256", &path)?, 64, "source.sha256")?,
        });
    }
    Ok(output)
}

fn parse_tensors(value: &Value) -> Result<Vec<CheckedTensor>, FnlpqReadError> {
    let array = array_at(value, "tensors")?;
    if array.is_empty() || array.len() > MAX_ENTRIES {
        return Err(header_error(
            "tensors",
            format!("length must be 1..={MAX_ENTRIES}"),
        ));
    }
    let mut previous = None::<String>;
    let mut output = Vec::with_capacity(array.len());
    for (index, item) in array.iter().enumerate() {
        let path = format!("tensors/{index}");
        let object = exact_object(
            item,
            &path,
            &[
                "canonical_dtype",
                "canonical_logical_sha256",
                "generic",
                "name",
                "shape",
            ],
        )?;
        let name = authority(exact_string(object, "name", &path)?, "tensor.name")?;
        if previous
            .as_ref()
            .is_some_and(|previous| previous.as_bytes() >= name.as_bytes())
        {
            return Err(header_error(
                format!("{path}/name"),
                "tensor names must be strictly increasing ASCII order",
            ));
        }
        previous = Some(name.clone());
        let canonical_dtype = exact_string(object, "canonical_dtype", &path)?.to_owned();
        if !matches!(canonical_dtype.as_str(), "bf16" | "f32" | "i8") {
            return Err(header_error(
                format!("{path}/canonical_dtype"),
                "unknown v1 dtype",
            ));
        }
        let shape_values = array_at(value_at(object, "shape", &path)?, &format!("{path}/shape"))?;
        if shape_values.is_empty() || shape_values.len() > 8 {
            return Err(header_error(format!("{path}/shape"), "rank must be 1..=8"));
        }
        let mut element_count = 1_u64;
        let mut shape = Vec::with_capacity(shape_values.len());
        for (dimension_index, dimension) in shape_values.iter().enumerate() {
            let number = dimension.as_u64().ok_or_else(|| {
                header_error(
                    format!("{path}/shape/{dimension_index}"),
                    "must be an unsigned integer",
                )
            })?;
            if number == 0 || number > u32::MAX as u64 {
                return Err(header_error(
                    format!("{path}/shape/{dimension_index}"),
                    "must be 1..=4294967295",
                ));
            }
            element_count = element_count.checked_mul(number).ok_or_else(|| {
                header_error(format!("{path}/shape"), "element product overflows u64")
            })?;
            shape.push(number as u32);
        }
        let _ = element_count;
        let generic = exact_object(
            value_at(object, "generic", &path)?,
            &format!("{path}/generic"),
            &["data", "quantization", "row_sum", "scale"],
        )?;
        let quantization = authority(
            exact_string(generic, "quantization", &format!("{path}/generic"))?,
            "tensor.generic.quantization",
        )?;
        output.push(CheckedTensor {
            name,
            canonical_dtype,
            shape,
            canonical_logical_sha256: lowercase_hex(
                exact_string(object, "canonical_logical_sha256", &path)?,
                64,
                "tensor.canonical_logical_sha256",
            )?,
            quantization,
            data: parse_mapping(
                value_at(generic, "data", &format!("{path}/generic"))?,
                &path,
                "data",
            )?,
            scale: parse_mapping(
                value_at(generic, "scale", &format!("{path}/generic"))?,
                &path,
                "scale",
            )?,
            row_sum: parse_mapping(
                value_at(generic, "row_sum", &format!("{path}/generic"))?,
                &path,
                "row_sum",
            )?,
        });
    }
    Ok(output)
}

fn parse_mapping(
    value: &Value,
    tensor_path: &str,
    mapping_name: &str,
) -> Result<CheckedMapping, FnlpqReadError> {
    let path = format!("{tensor_path}/generic/{mapping_name}");
    let object = exact_object(value, &path, &["length", "offset", "section_ordinal"])?;
    Ok(CheckedMapping {
        len: exact_u64(object, "length", &path)?,
        offset: exact_u64(object, "offset", &path)?,
        section_ordinal: exact_u64(object, "section_ordinal", &path)?,
    })
}

fn parse_packing_sets(value: &Value) -> Result<Vec<CheckedPackingSet>, FnlpqReadError> {
    let array = array_at(value, "packing_sets")?;
    if array.is_empty() || array.len() > 16 {
        return Err(header_error("packing_sets", "length must be 1..=16"));
    }
    let mut previous = None::<String>;
    let mut output = Vec::with_capacity(array.len());
    for (index, item) in array.iter().enumerate() {
        let path = format!("packing_sets/{index}");
        let object = exact_object(item, &path, &["id", "representations", "target"])?;
        let id = authority(exact_string(object, "id", &path)?, "packing_set.id")?;
        if previous
            .as_ref()
            .is_some_and(|previous| previous.as_bytes() >= id.as_bytes())
        {
            return Err(header_error(
                format!("{path}/id"),
                "IDs must be strictly sorted",
            ));
        }
        previous = Some(id.clone());
        let target = exact_string(object, "target", &path)?.to_owned();
        if !is_known_target(&target) {
            return Err(header_error(format!("{path}/target"), "unknown v1 target"));
        }
        let representation_values = array_at(
            value_at(object, "representations", &path)?,
            &format!("{path}/representations"),
        )?;
        if representation_values.is_empty() || representation_values.len() > 16 {
            return Err(header_error(
                format!("{path}/representations"),
                "length must be 1..=16",
            ));
        }
        let mut representations = Vec::with_capacity(representation_values.len());
        for (representation_index, representation) in representation_values.iter().enumerate() {
            let representation_path = format!("{path}/representations/{representation_index}");
            let representation = exact_object(
                representation,
                &representation_path,
                &["byte_cost", "section_ordinal", "sha256"],
            )?;
            representations.push(CheckedPackingRepresentation {
                byte_cost: exact_u64(representation, "byte_cost", &representation_path)?,
                section_ordinal: exact_u64(
                    representation,
                    "section_ordinal",
                    &representation_path,
                )?,
                sha256: lowercase_hex(
                    exact_string(representation, "sha256", &representation_path)?,
                    64,
                    "packing_representation.sha256",
                )?,
            });
        }
        output.push(CheckedPackingSet {
            id,
            target,
            representations,
        });
    }
    Ok(output)
}

fn parse_directory(
    bytes: &[u8],
    directory_start: usize,
    directory_end: usize,
    prelude: &CheckedPrelude,
    header: &CheckedHeader,
) -> Result<Vec<CheckedSection>, FnlpqReadError> {
    let mut output = Vec::with_capacity(prelude.section_count as usize);
    let mut previous_end = directory_end as u64;
    for ordinal in 0..prelude.section_count as usize {
        let start = directory_start
            .checked_add(
                ordinal
                    .checked_mul(SECTION_DIRECTORY_ENTRY_BYTES)
                    .ok_or_else(|| {
                        directory_error(ordinal, "range", "directory offset overflow")
                    })?,
            )
            .ok_or_else(|| directory_error(ordinal, "range", "directory offset overflow"))?;
        let entry = bytes
            .get(start..start + SECTION_DIRECTORY_ENTRY_BYTES)
            .ok_or_else(|| directory_error(ordinal, "range", "truncated directory entry"))?;
        let kind = section_kind_code(
            u32::from_le_bytes(entry[..4].try_into().expect("fixed entry")),
            ordinal,
        )?;
        let flags = u32::from_le_bytes(entry[4..8].try_into().expect("fixed entry"));
        if flags != 0 {
            return Err(directory_error(ordinal, "flags", "unknown required flags"));
        }
        let name_index = u64::from_le_bytes(entry[8..16].try_into().expect("fixed entry"));
        if name_index != ordinal as u64 {
            return Err(directory_error(
                ordinal,
                "name_index",
                "must equal header section ordinal",
            ));
        }
        let file_offset = u64::from_le_bytes(entry[16..24].try_into().expect("fixed entry"));
        let stored_len = u64::from_le_bytes(entry[24..32].try_into().expect("fixed entry"));
        let logical_len = u64::from_le_bytes(entry[32..40].try_into().expect("fixed entry"));
        if logical_len != stored_len {
            return Err(directory_error(
                ordinal,
                "logical_len",
                "compression is not supported in v1",
            ));
        }
        if stored_len > section_cap(kind) {
            return Err(directory_error(
                ordinal,
                "stored_len",
                "section length exceeds its v1 cap",
            ));
        }
        let alignment = u64::from_le_bytes(entry[40..48].try_into().expect("fixed entry"));
        if !valid_alignment(alignment) {
            return Err(directory_error(
                ordinal,
                "alignment",
                "must be a power of two in 1..=4096",
            ));
        }
        if file_offset % alignment != 0 {
            return Err(directory_error(
                ordinal,
                "file_offset",
                "does not meet declared alignment",
            ));
        }
        let expected_offset = align_up(previous_end, alignment)
            .map_err(|reason| directory_error(ordinal, "file_offset", reason))?;
        if file_offset != expected_offset {
            return Err(directory_error(
                ordinal,
                "file_offset",
                format!("expected minimum aligned offset {expected_offset}, found {file_offset}"),
            ));
        }
        let file_end = file_offset
            .checked_add(stored_len)
            .ok_or_else(|| directory_error(ordinal, "file_offset/stored_len", "range overflow"))?;
        if file_end > prelude.file_len {
            return Err(directory_error(
                ordinal,
                "file_offset/stored_len",
                "range exceeds prelude.file_len",
            ));
        }
        let gap_start = usize_from_u64(previous_end, "directory gap start")?;
        let gap_end = usize_from_u64(file_offset, "directory gap end")?;
        if !bytes
            .get(gap_start..gap_end)
            .ok_or_else(|| directory_error(ordinal, "gap", "gap outside file"))?
            .iter()
            .all(|byte| *byte == 0)
        {
            return Err(directory_error(
                ordinal,
                "gap",
                format!("nonzero alignment gap [{previous_end},{file_offset})"),
            ));
        }
        let payload_start = usize_from_u64(file_offset, "section file_offset")?;
        let payload_end = usize_from_u64(file_end, "section file_end")?;
        let payload = bytes
            .get(payload_start..payload_end)
            .ok_or_else(|| directory_error(ordinal, "range", "payload outside file"))?;
        let mut stored_sha256 = [0_u8; 32];
        stored_sha256.copy_from_slice(&entry[48..80]);
        let header_section = header.sections.get(ordinal).ok_or_else(|| {
            directory_error(ordinal, "name_index", "header section ordinal missing")
        })?;
        if header_section.kind != kind {
            return Err(directory_error(
                ordinal,
                "kind",
                format!(
                    "disagrees with header kind {}",
                    header_section.kind.header_name()
                ),
            ));
        }
        let observed_sha256 = framed_sha256(
            "fnlpq-section-v1",
            &[header_section.name.as_bytes(), payload],
        )
        .map_err(|error| directory_error(ordinal, "stored_sha256", error.to_string()))?;
        if observed_sha256 != stored_sha256 {
            return Err(directory_error(
                ordinal,
                "stored_sha256",
                format!(
                    "mismatch expected={} actual={}",
                    hex_lower(&stored_sha256),
                    hex_lower(&observed_sha256)
                ),
            ));
        }
        output.push(CheckedSection {
            ordinal: ordinal as u64,
            name: header_section.name.clone(),
            kind,
            file_offset,
            stored_len,
            alignment,
            stored_sha256,
        });
        previous_end = file_end;
    }
    if previous_end != prelude.file_len {
        return Err(FnlpqReadError::Directory {
            ordinal: prelude.section_count as usize - 1,
            field: "trailing_padding",
            reason: format!(
                "section end {previous_end} differs from file_len {}",
                prelude.file_len
            ),
        });
    }
    Ok(output)
}

fn validate_header_relationships(
    bytes: &[u8],
    header: &CheckedHeader,
    sections: &[CheckedSection],
) -> Result<(), FnlpqReadError> {
    for source in &header.sources {
        let section = section_for(sections, source.section_ordinal, "materialized source")?;
        let expected_kind = match source.name.as_str() {
            "model_config" => SectionKind::ModelConfig,
            "tokenizer_model" => SectionKind::TokenizerModel,
            "tokenizer_config" => SectionKind::TokenizerConfig,
            "chat_template" => SectionKind::ChatTemplate,
            _ => unreachable!("parse_sources validates the closed source name set"),
        };
        if section.kind != expected_kind {
            return Err(section_error(
                section,
                format!("materialized source {} points to wrong kind", source.name),
            ));
        }
        if source.sha256 != hex_lower(&section.stored_sha256) {
            return Err(section_error(
                section,
                format!(
                    "materialized source {} digest differs from directory",
                    source.name
                ),
            ));
        }
    }
    let mut ranges: BTreeMap<u64, Vec<(&str, u64, u64)>> = BTreeMap::new();
    for tensor in &header.tensors {
        let mappings = [
            ("data", &tensor.data, SectionKind::GenericTensorPayload),
            ("scale", &tensor.scale, SectionKind::GenericTensorScales),
            (
                "row_sum",
                &tensor.row_sum,
                SectionKind::GenericTensorRowSums,
            ),
        ];
        for (mapping_name, mapping, expected_kind) in mappings {
            let section = section_for(sections, mapping.section_ordinal, mapping_name)?;
            if section.kind != expected_kind {
                return Err(section_error(
                    section,
                    format!(
                        "tensor {} {mapping_name} mapping targets wrong section kind",
                        tensor.name
                    ),
                ));
            }
            let end = mapping.offset.checked_add(mapping.len).ok_or_else(|| {
                section_error(
                    section,
                    format!("tensor {} {mapping_name} range overflow", tensor.name),
                )
            })?;
            if end > section.stored_len {
                return Err(section_error(
                    section,
                    format!(
                        "tensor {} {mapping_name} range [{}, {end}) exceeds section length {}",
                        tensor.name, mapping.offset, section.stored_len
                    ),
                ));
            }
            if mapping.len > 0 {
                ranges.entry(mapping.section_ordinal).or_default().push((
                    tensor.name.as_str(),
                    mapping.offset,
                    end,
                ));
            }
            if mapping_name == "scale" {
                validate_scale_range(bytes, section, mapping, &tensor.name)?;
            }
        }
    }
    for (ordinal, mut claimed) in ranges {
        claimed.sort_by_key(|(_, start, _)| *start);
        for pair in claimed.windows(2) {
            if pair[0].2 > pair[1].1 {
                let section = section_for(sections, ordinal, "tensor overlap")?;
                return Err(section_error(
                    section,
                    format!(
                        "tensor mappings overlap: {} [{}, {}) and {} [{}, {})",
                        pair[0].0, pair[0].1, pair[0].2, pair[1].0, pair[1].1, pair[1].2
                    ),
                ));
            }
        }
    }
    validate_packing_sets(header, sections)
}

fn validate_scale_range(
    bytes: &[u8],
    section: &CheckedSection,
    mapping: &CheckedMapping,
    tensor_name: &str,
) -> Result<(), FnlpqReadError> {
    if mapping.len % 4 != 0 {
        return Err(section_error(
            section,
            format!(
                "tensor {tensor_name} scale length {} is not f32-aligned",
                mapping.len
            ),
        ));
    }
    let start = section
        .file_offset
        .checked_add(mapping.offset)
        .ok_or_else(|| section_error(section, "scale absolute offset overflow"))?;
    let end = start
        .checked_add(mapping.len)
        .ok_or_else(|| section_error(section, "scale absolute end overflow"))?;
    let values = bytes
        .get(usize_from_u64(start, "scale start")?..usize_from_u64(end, "scale end")?)
        .ok_or_else(|| section_error(section, "scale range outside owned bytes"))?;
    for (index, bits) in values.chunks_exact(4).enumerate() {
        let value = f32::from_bits(u32::from_le_bytes(bits.try_into().expect("exact chunks")));
        if !value.is_finite() || value <= 0.0 {
            return Err(section_error(
                section,
                format!("tensor {tensor_name} nonfinite/nonpositive scale index={index}"),
            ));
        }
    }
    Ok(())
}

fn validate_packing_sets(
    header: &CheckedHeader,
    sections: &[CheckedSection],
) -> Result<(), FnlpqReadError> {
    let generic = header
        .packing_sets
        .iter()
        .find(|set| set.id == "generic")
        .ok_or_else(|| header_error("packing_sets", "mandatory generic packing set is absent"))?;
    if generic.target != "generic" {
        return Err(header_error(
            "packing_sets/generic/target",
            "must be generic",
        ));
    }
    let mut native_seen = BTreeSet::new();
    for set in &header.packing_sets {
        let mut seen = BTreeSet::new();
        for representation in &set.representations {
            if !seen.insert(representation.section_ordinal) {
                return Err(header_error(
                    format!("packing_sets/{}/representations", set.id),
                    "duplicate section ordinal",
                ));
            }
            let section = section_for(
                sections,
                representation.section_ordinal,
                "packing representation",
            )?;
            if representation.byte_cost != section.stored_len {
                return Err(section_error(
                    section,
                    "packing byte_cost differs from stored_len",
                ));
            }
            if representation.sha256 != hex_lower(&section.stored_sha256) {
                return Err(section_error(
                    section,
                    "packing digest differs from directory digest",
                ));
            }
            if set.id == "generic" {
                if !matches!(
                    section.kind,
                    SectionKind::GenericTensorPayload
                        | SectionKind::GenericTensorScales
                        | SectionKind::GenericTensorRowSums
                ) {
                    return Err(section_error(
                        section,
                        "generic set references a non-generic section",
                    ));
                }
            } else if section.kind != SectionKind::NativePackingPayload {
                return Err(section_error(
                    section,
                    "native set references a non-native section",
                ));
            } else if !native_seen.insert(section.ordinal) {
                return Err(section_error(
                    section,
                    "native payload must occur in exactly one packing representation",
                ));
            }
        }
    }
    let generic_kinds: BTreeSet<_> = generic
        .representations
        .iter()
        .map(|representation| {
            section_for(
                sections,
                representation.section_ordinal,
                "generic representation",
            )
            .map(|section| section.kind)
        })
        .collect::<Result<_, _>>()?;
    let expected: BTreeSet<_> = [
        SectionKind::GenericTensorPayload,
        SectionKind::GenericTensorScales,
        SectionKind::GenericTensorRowSums,
    ]
    .into_iter()
    .collect();
    if generic_kinds != expected {
        return Err(header_error(
            "packing_sets/generic/representations",
            "must bind exactly generic payload, scales, and row sums",
        ));
    }
    for section in sections
        .iter()
        .filter(|section| section.kind == SectionKind::NativePackingPayload)
    {
        if !native_seen.contains(&section.ordinal) {
            return Err(section_error(
                section,
                "native payload is not declared by a packing set",
            ));
        }
    }
    Ok(())
}

fn section_for<'a>(
    sections: &'a [CheckedSection],
    ordinal: u64,
    context: &str,
) -> Result<&'a CheckedSection, FnlpqReadError> {
    sections
        .get(ordinal as usize)
        .ok_or_else(|| FnlpqReadError::Header {
            field: context.to_owned(),
            reason: format!("section ordinal {ordinal} is absent"),
        })
}

fn exact_object<'a>(
    value: &'a Value,
    path: &str,
    expected_keys: &[&str],
) -> Result<&'a Map<String, Value>, FnlpqReadError> {
    let object = value
        .as_object()
        .ok_or_else(|| header_error(path, "must be an object"))?;
    for key in expected_keys {
        if !object.contains_key(*key) {
            return Err(header_error(path, format!("missing required member {key}")));
        }
    }
    for key in object.keys() {
        if !expected_keys.contains(&key.as_str()) {
            return Err(header_error(path, format!("unknown member {key}")));
        }
    }
    if object.len() != expected_keys.len() {
        return Err(header_error(
            path,
            "duplicate or unknown members are forbidden",
        ));
    }
    Ok(object)
}

fn value_at<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<&'a Value, FnlpqReadError> {
    object
        .get(key)
        .ok_or_else(|| header_error(path, format!("missing required member {key}")))
}

fn array_at<'a>(value: &'a Value, path: &str) -> Result<&'a Vec<Value>, FnlpqReadError> {
    value
        .as_array()
        .ok_or_else(|| header_error(path, "must be an array"))
}

fn exact_string<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<&'a str, FnlpqReadError> {
    value_at(object, key, path)?
        .as_str()
        .ok_or_else(|| header_error(format!("{path}/{key}"), "must be a string"))
}

fn exact_u64(object: &Map<String, Value>, key: &str, path: &str) -> Result<u64, FnlpqReadError> {
    value_at(object, key, path)?
        .as_u64()
        .ok_or_else(|| header_error(format!("{path}/{key}"), "must be an unsigned integer"))
}

fn exact_bool(object: &Map<String, Value>, key: &str, path: &str) -> Result<bool, FnlpqReadError> {
    value_at(object, key, path)?
        .as_bool()
        .ok_or_else(|| header_error(format!("{path}/{key}"), "must be a boolean"))
}

fn ensure_exact(value: &str, expected: &str, field: &str) -> Result<(), FnlpqReadError> {
    if value == expected {
        Ok(())
    } else {
        Err(header_error(
            field,
            format!("expected {expected:?}, found {value:?}"),
        ))
    }
}

fn authority(value: &str, field: &str) -> Result<String, FnlpqReadError> {
    let mut bytes = value.bytes();
    let first = bytes.next();
    if value.len() > 128
        || !matches!(first, Some(byte) if byte.is_ascii_alphanumeric())
        || !bytes.all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'+' | b'-')
        })
    {
        return Err(header_error(
            field,
            "must be a printable ASCII authority ID",
        ));
    }
    Ok(value.to_owned())
}

fn lowercase_hex(value: &str, width: usize, field: &str) -> Result<String, FnlpqReadError> {
    if value.len() != width
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(header_error(
            field,
            format!("must be lowercase hexadecimal with width {width}"),
        ));
    }
    Ok(value.to_owned())
}

fn section_kind_name(value: &str, path: &str) -> Result<SectionKind, FnlpqReadError> {
    match value {
        "GENERIC_TENSOR_PAYLOAD" => Ok(SectionKind::GenericTensorPayload),
        "GENERIC_TENSOR_SCALES" => Ok(SectionKind::GenericTensorScales),
        "GENERIC_TENSOR_ROW_SUMS" => Ok(SectionKind::GenericTensorRowSums),
        "TOKENIZER_MODEL" => Ok(SectionKind::TokenizerModel),
        "MODEL_CONFIG" => Ok(SectionKind::ModelConfig),
        "TOKENIZER_CONFIG" => Ok(SectionKind::TokenizerConfig),
        "CHAT_TEMPLATE" => Ok(SectionKind::ChatTemplate),
        "LICENSE_BUNDLE" => Ok(SectionKind::LicenseBundle),
        "NATIVE_PACKING_PAYLOAD" => Ok(SectionKind::NativePackingPayload),
        _ => Err(header_error(
            format!("{path}/kind"),
            "unknown v1 section kind",
        )),
    }
}

fn section_kind_code(value: u32, ordinal: usize) -> Result<SectionKind, FnlpqReadError> {
    match value {
        1 => Ok(SectionKind::GenericTensorPayload),
        2 => Ok(SectionKind::GenericTensorScales),
        3 => Ok(SectionKind::GenericTensorRowSums),
        4 => Ok(SectionKind::TokenizerModel),
        5 => Ok(SectionKind::ModelConfig),
        6 => Ok(SectionKind::TokenizerConfig),
        7 => Ok(SectionKind::ChatTemplate),
        8 => Ok(SectionKind::LicenseBundle),
        9 => Ok(SectionKind::NativePackingPayload),
        _ => Err(directory_error(ordinal, "kind", "unknown v1 section kind")),
    }
}

fn required_singleton(kind: SectionKind) -> bool {
    kind != SectionKind::NativePackingPayload
}

fn section_cap(kind: SectionKind) -> u64 {
    match kind {
        SectionKind::TokenizerModel => 64 * 1024 * 1024,
        SectionKind::ModelConfig | SectionKind::TokenizerConfig => 16 * 1024 * 1024,
        SectionKind::ChatTemplate => 8 * 1024 * 1024,
        SectionKind::LicenseBundle => 1024 * 1024,
        _ => MAX_FILE_BYTES,
    }
}

fn is_known_target(value: &str) -> bool {
    matches!(
        value,
        "generic" | "aarch64-sdot" | "aarch64-i8mm" | "x86-vnni-256" | "x86-vnni-512" | "x86-avx2"
    )
}

fn valid_alignment(value: u64) -> bool {
    value != 0 && value <= MAX_ALIGNMENT && value.is_power_of_two()
}

fn align_up(value: u64, alignment: u64) -> Result<u64, String> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| "alignment addition overflow".to_owned())
    }
}

fn usize_from_u64(value: u64, field: &str) -> Result<usize, FnlpqReadError> {
    usize::try_from(value).map_err(|_| FnlpqReadError::Prelude {
        field: "usize conversion",
        reason: format!("{field}={value} is not addressable on this host"),
    })
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn header_error(field: impl Into<String>, reason: impl Into<String>) -> FnlpqReadError {
    FnlpqReadError::Header {
        field: field.into(),
        reason: reason.into(),
    }
}

fn directory_error(
    ordinal: usize,
    field: &'static str,
    reason: impl Into<String>,
) -> FnlpqReadError {
    FnlpqReadError::Directory {
        ordinal,
        field,
        reason: reason.into(),
    }
}

fn section_error(section: &CheckedSection, reason: impl Into<String>) -> FnlpqReadError {
    FnlpqReadError::Section {
        ordinal: section.ordinal as usize,
        name: section.name.clone(),
        reason: reason.into(),
    }
}
