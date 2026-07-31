//! Host-native packing derivation from a checked Generic `.fnlpq` root.
//!
//! The published Generic root remains the reconstruction authority.  A native
//! cache is a separate, content-addressed envelope that retains the checked
//! Generic sections and adds one target-specific packed payload.  The first
//! tile table deliberately preserves logical row order inside a bounded native
//! payload: it establishes cache identity and byte-differential recovery
//! before a measured ISA kernel promotes a different tile table.

use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::format::{
    framed_sha256, validate_authority_identifier, write, ArchTarget, CanonicalDtype,
    FnlpqWriterInput, PackingSetInput, SectionKind, SectionPayload, SectionRange, TensorInput,
};
use super::reader::{CheckedMapping, FnlpqArtifact, FnlpqReadError};

/// The initial deterministic native payload table.  It preserves logical row
/// order; measured target kernels must introduce a new version rather than
/// changing bytes under this identifier.
pub const TILE_TABLE_VERSION_V1: &str = "tile-table-v1";
const NATIVE_PAYLOAD_MAGIC: [u8; 8] = *b"FNLPNTV1";
const NATIVE_PAYLOAD_ALIGNMENT: u64 = 64;

/// Every architecture-specific representation that v1 may derive locally.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum NativePackingTarget {
    /// AArch64 signed-dot product packing.
    Aarch64Sdot,
    /// AArch64 I8MM packing.
    Aarch64I8mm,
    /// x86 VNNI 256-bit packing.
    X86Vnni256,
    /// x86 VNNI 512-bit packing.
    X86Vnni512,
    /// Exact x86 AVX2 packing.
    X86Avx2,
}

impl NativePackingTarget {
    /// All closed v1 derivation targets in stable command order.
    pub const ALL: [Self; 5] = [
        Self::Aarch64Sdot,
        Self::Aarch64I8mm,
        Self::X86Vnni256,
        Self::X86Vnni512,
        Self::X86Avx2,
    ];

    /// Parse the stable `fnlp models derive --arch` spelling.
    pub fn parse(value: &str) -> Result<Self, PackingError> {
        match value {
            "aarch64-sdot" => Ok(Self::Aarch64Sdot),
            "aarch64-i8mm" => Ok(Self::Aarch64I8mm),
            "x86-vnni256" => Ok(Self::X86Vnni256),
            "x86-vnni512" => Ok(Self::X86Vnni512),
            "x86-avx2" => Ok(Self::X86Avx2),
            actual => Err(PackingError::UnsupportedTarget {
                actual: actual.to_owned(),
            }),
        }
    }

    /// Stable CLI spelling and portable filename component.
    pub const fn cli_name(self) -> &'static str {
        match self {
            Self::Aarch64Sdot => "aarch64-sdot",
            Self::Aarch64I8mm => "aarch64-i8mm",
            Self::X86Vnni256 => "x86-vnni256",
            Self::X86Vnni512 => "x86-vnni512",
            Self::X86Avx2 => "x86-avx2",
        }
    }

    /// Corresponding closed envelope target.
    pub const fn arch_target(self) -> ArchTarget {
        match self {
            Self::Aarch64Sdot => ArchTarget::Aarch64Sdot,
            Self::Aarch64I8mm => ArchTarget::Aarch64I8mm,
            Self::X86Vnni256 => ArchTarget::X86Vnni256,
            Self::X86Vnni512 => ArchTarget::X86Vnni512,
            Self::X86Avx2 => ArchTarget::X86Avx2,
        }
    }

    /// Deterministic packing identity, including the tile-table version.
    pub fn packing_id(self, tile_table_version: &str) -> Result<String, PackingError> {
        validate_tile_table_version(tile_table_version)?;
        let id = format!("{}-{tile_table_version}", self.cli_name());
        validate_authority_identifier("native packing id", &id).map_err(PackingError::Format)?;
        Ok(id)
    }
}

/// Immutable cache address for one Generic root and native representation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeCacheAddress {
    /// SHA-256 of the complete Generic root bytes.
    pub whole_artifact_sha256: String,
    /// Target/table-specific native representation identity.
    pub packing_id: String,
    /// Versioned tile-table identity included in `packing_id` and the address.
    pub tile_table_version: String,
    /// Domain-framed cache directory component derived from the exact triple.
    pub content_address: String,
}

impl NativeCacheAddress {
    /// Construct the one cache address for the contract's identity triple.
    pub fn new(
        whole_artifact_sha256: String,
        packing_id: String,
        tile_table_version: String,
    ) -> Result<Self, PackingError> {
        validate_lower_sha256("whole artifact", &whole_artifact_sha256)?;
        validate_authority_identifier("native packing id", &packing_id)
            .map_err(PackingError::Format)?;
        validate_tile_table_version(&tile_table_version)?;
        if !packing_id.ends_with(&tile_table_version) {
            return Err(PackingError::Address {
                detail: "packing_id must include tile_table_version".to_owned(),
            });
        }
        let content_address = hex_lower(
            &framed_sha256(
                "fnlp-native-cache-v1",
                &[
                    whole_artifact_sha256.as_bytes(),
                    packing_id.as_bytes(),
                    tile_table_version.as_bytes(),
                ],
            )
            .map_err(PackingError::Format)?,
        );
        Ok(Self {
            whole_artifact_sha256,
            packing_id,
            tile_table_version,
            content_address,
        })
    }

    /// Cache destination beneath the caller's model root.  This has no
    /// side-effect; the ratified filesystem transaction owns creation.
    pub fn cache_path(&self, model_root: impl AsRef<Path>) -> PathBuf {
        model_root
            .as_ref()
            .join("native")
            .join(&self.content_address)
            .join(format!("{}.fnlpq", self.packing_id))
    }
}

/// Honest local footprint of one derived cache while its Generic root remains
/// retained for reconstruction and provenance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeDerivationFootprint {
    /// Bytes of the immutable Generic root already present locally.
    pub generic_root_bytes: u64,
    /// Bytes solely in the target-specific native representation.
    pub native_packing_set_bytes: u64,
    /// Peak payload bytes retained by the current materialized derive path.
    /// This counts exact byte buffers, not allocator or vector metadata.
    pub peak_derivation_bytes: u64,
    /// Generic root plus complete derived cache after a successful activation.
    pub steady_state_retained_bytes: u64,
}

/// Complete in-memory result of a deterministic native derivation.  A later
/// ratified filesystem layer may stage `bytes` at `address.cache_path(root)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivedNativeArtifact {
    /// Content address of the disposable native cache.
    pub address: NativeCacheAddress,
    /// Target that owns the derived representation.
    pub target: NativePackingTarget,
    /// Full checked derived envelope bytes.
    pub bytes: Vec<u8>,
    /// `D("fnlpq-file-v1", bytes)`; this necessarily differs from the
    /// Generic root's file identity.
    pub fnlpq_file_sha256: String,
    /// Packing-set digest from the derived physical envelope.
    pub packing_set_sha256: String,
    /// Packing-independent logical-model identity preserved from the root.
    pub logical_model_sha256: String,
    /// Five-quantity derivation accounting subset for the local cache.
    pub footprint: NativeDerivationFootprint,
}

/// Native derivation and byte-differential refusals.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PackingError {
    /// The requested CLI architecture has no v1 native packing contract.
    UnsupportedTarget { actual: String },
    /// An id or digest failed a closed authority grammar.
    Address { detail: String },
    /// The source root is not a Generic-only reconstruction authority.
    GenericRoot { detail: String },
    /// The checked reader rejected the source or derived envelope.
    Reader(String),
    /// The canonical writer rejected reconstructed envelope records.
    Format(super::format::FnlpqWriteError),
    /// Native payload bytes are truncated, non-canonical, or inconsistent.
    NativePayload { detail: String },
    /// Derived bytes no longer reconstruct the root's exact logical tensor.
    LogicalMismatch {
        tensor: String,
        component: &'static str,
    },
    /// A native representation is absent and dispatch must not silently fall
    /// back to another architecture.
    MissingDerivation { command: String },
    /// Checked accounting would overflow a u64 contract value.
    Arithmetic { invariant: &'static str },
}

impl fmt::Display for PackingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedTarget { actual } => write!(
                formatter,
                "unsupported native packing target {actual:?}; expected aarch64-sdot, aarch64-i8mm, x86-vnni256, x86-vnni512, or x86-avx2"
            ),
            Self::Address { detail } => write!(formatter, "invalid native cache address: {detail}"),
            Self::GenericRoot { detail } => write!(formatter, "Generic root refusal: {detail}"),
            Self::Reader(detail) => write!(formatter, "checked fnlpq reader: {detail}"),
            Self::Format(error) => write!(formatter, "canonical fnlpq writer: {error}"),
            Self::NativePayload { detail } => write!(formatter, "native payload: {detail}"),
            Self::LogicalMismatch { tensor, component } => {
                write!(formatter, "native reconstruction differs for tensor {tensor} {component}")
            }
            Self::MissingDerivation { command } => {
                write!(formatter, "required native packing is absent; derive it with {command}")
            }
            Self::Arithmetic { invariant } => {
                write!(formatter, "checked arithmetic overflow for {invariant}")
            }
        }
    }
}

impl Error for PackingError {}

/// Derive one deterministic target representation from a read-only Generic
/// root.  It neither chooses a dispatch default nor performs filesystem I/O.
pub fn derive_native_packing(
    generic_bytes: Vec<u8>,
    target: NativePackingTarget,
    tile_table_version: &str,
) -> Result<DerivedNativeArtifact, PackingError> {
    let generic_sha256 = hex_lower(&Sha256::digest(&generic_bytes));
    let generic_root_bytes =
        u64::try_from(generic_bytes.len()).map_err(|_| PackingError::Arithmetic {
            invariant: "Generic root bytes",
        })?;
    let generic = FnlpqArtifact::from_bytes(generic_bytes).map_err(reader_error)?;
    ensure_generic_root(&generic)?;
    let packing_id = target.packing_id(tile_table_version)?;
    let address = NativeCacheAddress::new(
        generic_sha256.clone(),
        packing_id.clone(),
        tile_table_version.to_owned(),
    )?;
    let logical = logical_tensors(&generic)?;
    let logical_copy_bytes = logical_tensor_bytes(&logical)?;
    let native_payload = encode_native_payload(&logical, target, &packing_id, tile_table_version)?;
    let mut sections = copy_sections(&generic)?;
    let native_section_name = format!("native-{}", target.cli_name());
    sections.push(SectionPayload::new(
        native_section_name.clone(),
        SectionKind::NativePackingPayload,
        native_payload.clone(),
        NATIVE_PAYLOAD_ALIGNMENT,
    ));
    let tensors = copy_tensor_inputs(&generic)?;
    let generic_section_names = generic
        .sections()
        .iter()
        .filter(|section| {
            matches!(
                section.kind,
                SectionKind::GenericTensorPayload
                    | SectionKind::GenericTensorScales
                    | SectionKind::GenericTensorRowSums
            )
        })
        .map(|section| section.name.clone())
        .collect();
    let written = write(&FnlpqWriterInput {
        model_id: generic.model_id().to_owned(),
        revision: generic.revision().to_owned(),
        recipe_id: generic.recipe_id().to_owned(),
        source_root_sha256: generic.source_root_sha256().to_owned(),
        logical_model_sha256: generic.logical_model_sha256().to_owned(),
        sections,
        tensors,
        packing_sets: vec![
            PackingSetInput {
                id: "generic".to_owned(),
                target: ArchTarget::Generic,
                section_names: generic_section_names,
            },
            PackingSetInput {
                id: packing_id,
                target: target.arch_target(),
                section_names: vec![native_section_name],
            },
        ],
    })
    .map_err(PackingError::Format)?;
    let derived_raw_sha256 = hex_lower(&Sha256::digest(&written.bytes));
    if derived_raw_sha256 == generic_sha256 {
        return Err(PackingError::GenericRoot {
            detail: "derived physical bytes equal the Generic root".to_owned(),
        });
    }
    let derived = FnlpqArtifact::from_bytes(written.bytes.clone()).map_err(reader_error)?;
    verify_derived_packing(&generic, &derived, target, tile_table_version)?;
    let derived_file_bytes =
        u64::try_from(written.bytes.len()).map_err(|_| PackingError::Arithmetic {
            invariant: "derived artifact bytes",
        })?;
    let native_packing_set_bytes =
        u64::try_from(native_payload.len()).map_err(|_| PackingError::Arithmetic {
            invariant: "native packing payload bytes",
        })?;
    let header_bytes =
        u64::try_from(written.header_bytes.len()).map_err(|_| PackingError::Arithmetic {
            invariant: "derived header bytes",
        })?;
    let retained =
        generic_root_bytes
            .checked_add(derived_file_bytes)
            .ok_or(PackingError::Arithmetic {
                invariant: "derived retained footprint",
            })?;
    // `write` retains the checked Generic root, decoded logical copies, the
    // native payload, its input section copies, its canonical section copies,
    // and the resulting envelope/header at peak.  The accounting deliberately
    // excludes allocator bookkeeping, which cannot be made architecture
    // independent here.
    let copied_sections = generic_root_bytes
        .checked_add(native_packing_set_bytes)
        .ok_or(PackingError::Arithmetic {
            invariant: "derived copied section bytes",
        })?;
    let writer_peak_payload_bytes = checked_sum(
        &[
            generic_root_bytes,
            logical_copy_bytes,
            native_packing_set_bytes,
            copied_sections,
            copied_sections,
            derived_file_bytes,
            header_bytes,
        ],
        "derived peak payload footprint",
    )?;
    let verification_peak_payload_bytes = checked_sum(
        &[
            generic_root_bytes,
            logical_copy_bytes,
            native_packing_set_bytes,
            derived_file_bytes,
            derived_file_bytes,
            header_bytes,
        ],
        "derived verification payload footprint",
    )?;
    let peak_derivation_bytes = writer_peak_payload_bytes.max(verification_peak_payload_bytes);
    let fnlpq_file_sha256 = hex_lower(&written.fnlpq_file_sha256);
    let packing_set_sha256 = derived.packing_set_sha256().to_owned();
    let logical_model_sha256 = derived.logical_model_sha256().to_owned();
    Ok(DerivedNativeArtifact {
        address,
        target,
        bytes: written.bytes,
        fnlpq_file_sha256,
        packing_set_sha256,
        logical_model_sha256,
        footprint: NativeDerivationFootprint {
            generic_root_bytes,
            native_packing_set_bytes,
            peak_derivation_bytes,
            steady_state_retained_bytes: retained,
        },
    })
}

/// Refuse a dispatch request whose matching target representation is not
/// installed.  The error names the exact derive command; it never offers a
/// differently-qualified target as a fallback.
pub fn require_native_packing(
    artifact: &FnlpqArtifact,
    target: NativePackingTarget,
    generic_path: impl AsRef<Path>,
) -> Result<(), PackingError> {
    match artifact.select_packing(target.arch_target()) {
        Ok(_) => Ok(()),
        Err(_) => Err(PackingError::MissingDerivation {
            command: format!(
                "fnlp models derive --generic {} --arch {}",
                generic_path.as_ref().display(),
                target.cli_name()
            ),
        }),
    }
}

/// Verify the derived envelope and its native payload against an already
/// checked Generic root.  This is the byte-differential gate used by derive
/// before any filesystem transaction may publish the cache.
pub fn verify_derived_packing(
    generic: &FnlpqArtifact,
    derived: &FnlpqArtifact,
    target: NativePackingTarget,
    tile_table_version: &str,
) -> Result<(), PackingError> {
    ensure_generic_root(generic)?;
    validate_tile_table_version(tile_table_version)?;
    if generic.logical_model_sha256() != derived.logical_model_sha256() {
        return Err(PackingError::GenericRoot {
            detail: "derived logical_model_sha256 differs from Generic root".to_owned(),
        });
    }
    if generic.packing_set_sha256() == derived.packing_set_sha256() {
        return Err(PackingError::GenericRoot {
            detail: "derived packing_set_sha256 did not change".to_owned(),
        });
    }
    let generic_tensors = logical_tensors(generic)?;
    let derived_tensors = logical_tensors(derived)?;
    compare_logical_tensors(&generic_tensors, &derived_tensors)?;
    let packing_id = target.packing_id(tile_table_version)?;
    let native_section = derived
        .sections()
        .iter()
        .find(|section| {
            section.kind == SectionKind::NativePackingPayload
                && section.name == format!("native-{}", target.cli_name())
        })
        .ok_or_else(|| PackingError::NativePayload {
            detail: format!("missing {} native section", target.cli_name()),
        })?;
    let bytes = derived
        .section_bytes(native_section.ordinal)
        .ok_or_else(|| PackingError::NativePayload {
            detail: "checked native section bytes became unavailable".to_owned(),
        })?;
    let native = decode_native_payload(bytes)?;
    if native.target != target
        || native.packing_id != packing_id
        || native.tile_table_version != tile_table_version
    {
        return Err(PackingError::NativePayload {
            detail: "target, packing id, or tile table version disagrees with derivation request"
                .to_owned(),
        });
    }
    compare_logical_tensors(&generic_tensors, &native.tensors)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LogicalTensorBytes {
    name: String,
    canonical_dtype: String,
    shape: Vec<u32>,
    quantization: String,
    data: Vec<u8>,
    scale: Vec<u8>,
    row_sum: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NativePayload {
    target: NativePackingTarget,
    packing_id: String,
    tile_table_version: String,
    tensors: Vec<LogicalTensorBytes>,
}

fn ensure_generic_root(artifact: &FnlpqArtifact) -> Result<(), PackingError> {
    if artifact
        .sections()
        .iter()
        .any(|section| section.kind == SectionKind::NativePackingPayload)
    {
        return Err(PackingError::GenericRoot {
            detail: "root already contains a native packing payload".to_owned(),
        });
    }
    artifact
        .select_packing(ArchTarget::Generic)
        .map_err(reader_error)?;
    Ok(())
}

fn copy_sections(artifact: &FnlpqArtifact) -> Result<Vec<SectionPayload>, PackingError> {
    artifact
        .sections()
        .iter()
        .map(|section| {
            let bytes = artifact.section_bytes(section.ordinal).ok_or_else(|| {
                PackingError::Reader(format!("section {} bytes became unavailable", section.name))
            })?;
            Ok(SectionPayload::new(
                section.name.clone(),
                section.kind,
                bytes.to_vec(),
                section.alignment,
            ))
        })
        .collect()
}

fn copy_tensor_inputs(artifact: &FnlpqArtifact) -> Result<Vec<TensorInput>, PackingError> {
    artifact
        .tensors()
        .iter()
        .map(|tensor| {
            Ok(TensorInput {
                name: tensor.name.clone(),
                canonical_dtype: canonical_dtype(&tensor.canonical_dtype)?,
                shape: tensor.shape.clone(),
                canonical_logical_sha256: tensor.canonical_logical_sha256.clone(),
                quantization: tensor.quantization.clone(),
                data: mapping_range(artifact, &tensor.data)?,
                scale: mapping_range(artifact, &tensor.scale)?,
                row_sum: mapping_range(artifact, &tensor.row_sum)?,
            })
        })
        .collect()
}

fn canonical_dtype(value: &str) -> Result<CanonicalDtype, PackingError> {
    match value {
        "bf16" => Ok(CanonicalDtype::Bf16),
        "f32" => Ok(CanonicalDtype::F32),
        "i8" => Ok(CanonicalDtype::I8),
        actual => Err(PackingError::GenericRoot {
            detail: format!("unsupported checked canonical dtype {actual}"),
        }),
    }
}

fn mapping_range(
    artifact: &FnlpqArtifact,
    mapping: &CheckedMapping,
) -> Result<SectionRange, PackingError> {
    Ok(SectionRange::new(
        section_for_mapping(artifact, mapping)?.name.clone(),
        mapping.offset,
        mapping.len,
    ))
}

fn logical_tensors(artifact: &FnlpqArtifact) -> Result<Vec<LogicalTensorBytes>, PackingError> {
    artifact
        .tensors()
        .iter()
        .map(|tensor| {
            Ok(LogicalTensorBytes {
                name: tensor.name.clone(),
                canonical_dtype: tensor.canonical_dtype.clone(),
                shape: tensor.shape.clone(),
                quantization: tensor.quantization.clone(),
                data: mapping_bytes(artifact, &tensor.data)?.to_vec(),
                scale: mapping_bytes(artifact, &tensor.scale)?.to_vec(),
                row_sum: mapping_bytes(artifact, &tensor.row_sum)?.to_vec(),
            })
        })
        .collect()
}

fn logical_tensor_bytes(tensors: &[LogicalTensorBytes]) -> Result<u64, PackingError> {
    let mut total = 0_u64;
    for tensor in tensors {
        for bytes in [&tensor.data, &tensor.scale, &tensor.row_sum] {
            let len = u64::try_from(bytes.len()).map_err(|_| PackingError::Arithmetic {
                invariant: "logical tensor copy bytes",
            })?;
            total = total.checked_add(len).ok_or(PackingError::Arithmetic {
                invariant: "logical tensor copy bytes",
            })?;
        }
    }
    Ok(total)
}

fn section_for_mapping<'a>(
    artifact: &'a FnlpqArtifact,
    mapping: &CheckedMapping,
) -> Result<&'a super::reader::CheckedSection, PackingError> {
    artifact
        .sections()
        .iter()
        .find(|section| section.ordinal == mapping.section_ordinal)
        .ok_or_else(|| {
            PackingError::Reader(format!(
                "mapping references unavailable section ordinal {}",
                mapping.section_ordinal
            ))
        })
}

fn mapping_bytes<'a>(
    artifact: &'a FnlpqArtifact,
    mapping: &CheckedMapping,
) -> Result<&'a [u8], PackingError> {
    let section = section_for_mapping(artifact, mapping)?;
    let section_bytes = artifact.section_bytes(section.ordinal).ok_or_else(|| {
        PackingError::Reader(format!("section {} bytes became unavailable", section.name))
    })?;
    let start = usize::try_from(mapping.offset).map_err(|_| PackingError::Arithmetic {
        invariant: "logical mapping offset",
    })?;
    let len = usize::try_from(mapping.len).map_err(|_| PackingError::Arithmetic {
        invariant: "logical mapping length",
    })?;
    let end = start.checked_add(len).ok_or(PackingError::Arithmetic {
        invariant: "logical mapping end",
    })?;
    section_bytes.get(start..end).ok_or_else(|| {
        PackingError::Reader(format!(
            "mapping range [{start},{end}) is unavailable in checked section {}",
            section.name
        ))
    })
}

fn encode_native_payload(
    tensors: &[LogicalTensorBytes],
    target: NativePackingTarget,
    packing_id: &str,
    tile_table_version: &str,
) -> Result<Vec<u8>, PackingError> {
    validate_tile_table_version(tile_table_version)?;
    validate_authority_identifier("native packing id", packing_id).map_err(PackingError::Format)?;
    let expected_id = target.packing_id(tile_table_version)?;
    if packing_id != expected_id {
        return Err(PackingError::NativePayload {
            detail: "packing id does not match target/tile-table pair".to_owned(),
        });
    }
    let tensor_count = u32::try_from(tensors.len()).map_err(|_| PackingError::Arithmetic {
        invariant: "native tensor count",
    })?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&NATIVE_PAYLOAD_MAGIC);
    push_bytes(&mut bytes, target.cli_name().as_bytes())?;
    push_bytes(&mut bytes, packing_id.as_bytes())?;
    push_bytes(&mut bytes, tile_table_version.as_bytes())?;
    bytes.extend_from_slice(&tensor_count.to_le_bytes());
    for tensor in tensors {
        push_bytes(&mut bytes, tensor.name.as_bytes())?;
        push_bytes(&mut bytes, tensor.canonical_dtype.as_bytes())?;
        let rank = u32::try_from(tensor.shape.len()).map_err(|_| PackingError::Arithmetic {
            invariant: "native tensor rank",
        })?;
        bytes.extend_from_slice(&rank.to_le_bytes());
        for dimension in &tensor.shape {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        push_bytes(&mut bytes, tensor.quantization.as_bytes())?;
        push_bytes(&mut bytes, &tensor.data)?;
        push_bytes(&mut bytes, &tensor.scale)?;
        push_bytes(&mut bytes, &tensor.row_sum)?;
    }
    Ok(bytes)
}

fn decode_native_payload(bytes: &[u8]) -> Result<NativePayload, PackingError> {
    let mut cursor = PayloadCursor::new(bytes);
    if cursor.read_exact(NATIVE_PAYLOAD_MAGIC.len(), "magic")? != NATIVE_PAYLOAD_MAGIC.as_slice() {
        return Err(PackingError::NativePayload {
            detail: "bad native payload magic".to_owned(),
        });
    }
    let target = NativePackingTarget::parse(cursor.read_authority("target")?)?;
    let packing_id = cursor.read_authority("packing id")?.to_owned();
    let tile_table_version = cursor.read_authority("tile table version")?.to_owned();
    let expected_id = target.packing_id(&tile_table_version)?;
    if packing_id != expected_id {
        return Err(PackingError::NativePayload {
            detail: "native payload packing id does not bind target/tile-table version".to_owned(),
        });
    }
    let count = usize::try_from(cursor.read_u32("tensor count")?).map_err(|_| {
        PackingError::Arithmetic {
            invariant: "native tensor count to usize",
        }
    })?;
    let mut tensors = Vec::with_capacity(count);
    for _ in 0..count {
        let name = cursor.read_authority("tensor name")?.to_owned();
        let canonical_dtype = cursor.read_authority("canonical dtype")?.to_owned();
        let rank = usize::try_from(cursor.read_u32("tensor rank")?).map_err(|_| {
            PackingError::Arithmetic {
                invariant: "native tensor rank to usize",
            }
        })?;
        if !(1..=8).contains(&rank) {
            return Err(PackingError::NativePayload {
                detail: format!("tensor {name} rank {rank} is outside 1..=8"),
            });
        }
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            let dimension = cursor.read_u32("tensor shape dimension")?;
            if dimension == 0 {
                return Err(PackingError::NativePayload {
                    detail: format!("tensor {name} has a zero shape dimension"),
                });
            }
            shape.push(dimension);
        }
        let quantization = cursor.read_authority("quantization")?.to_owned();
        let data = cursor.read_bytes("data")?.to_vec();
        let scale = cursor.read_bytes("scale")?.to_vec();
        let row_sum = cursor.read_bytes("row sum")?.to_vec();
        tensors.push(LogicalTensorBytes {
            name,
            canonical_dtype,
            shape,
            quantization,
            data,
            scale,
            row_sum,
        });
    }
    if cursor.remaining() != 0 {
        return Err(PackingError::NativePayload {
            detail: format!("{} trailing native payload bytes", cursor.remaining()),
        });
    }
    Ok(NativePayload {
        target,
        packing_id,
        tile_table_version,
        tensors,
    })
}

fn compare_logical_tensors(
    expected: &[LogicalTensorBytes],
    actual: &[LogicalTensorBytes],
) -> Result<(), PackingError> {
    if expected.len() != actual.len() {
        return Err(PackingError::GenericRoot {
            detail: format!(
                "logical tensor count differs: expected={} actual={}",
                expected.len(),
                actual.len()
            ),
        });
    }
    for (expected, actual) in expected.iter().zip(actual) {
        if expected.name != actual.name {
            return Err(PackingError::LogicalMismatch {
                tensor: expected.name.clone(),
                component: "name",
            });
        }
        for (component, matches) in [
            (
                "canonical_dtype",
                expected.canonical_dtype == actual.canonical_dtype,
            ),
            ("shape", expected.shape == actual.shape),
            ("quantization", expected.quantization == actual.quantization),
            ("data", expected.data == actual.data),
            ("scale", expected.scale == actual.scale),
            ("row_sum", expected.row_sum == actual.row_sum),
        ] {
            if !matches {
                return Err(PackingError::LogicalMismatch {
                    tensor: expected.name.clone(),
                    component,
                });
            }
        }
    }
    Ok(())
}

fn validate_tile_table_version(value: &str) -> Result<(), PackingError> {
    validate_authority_identifier("tile_table_version", value).map_err(PackingError::Format)
}

fn validate_lower_sha256(field: &str, value: &str) -> Result<(), PackingError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PackingError::Address {
            detail: format!("{field} must be lowercase SHA-256"),
        });
    }
    Ok(())
}

fn checked_sum(values: &[u64], invariant: &'static str) -> Result<u64, PackingError> {
    values.iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(*value)
            .ok_or(PackingError::Arithmetic { invariant })
    })
}

fn push_bytes(output: &mut Vec<u8>, bytes: &[u8]) -> Result<(), PackingError> {
    let len = u64::try_from(bytes.len()).map_err(|_| PackingError::Arithmetic {
        invariant: "native payload field length",
    })?;
    output.extend_from_slice(&len.to_le_bytes());
    output.extend_from_slice(bytes);
    Ok(())
}

fn reader_error(error: FnlpqReadError) -> PackingError {
    PackingError::Reader(error.to_string())
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut rendered = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        rendered.push(char::from(HEX[usize::from(byte >> 4)]));
        rendered.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    rendered
}

struct PayloadCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PayloadCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn read_exact(&mut self, len: usize, field: &str) -> Result<&'a [u8], PackingError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(PackingError::Arithmetic {
                invariant: "native payload cursor end",
            })?;
        let bytes =
            self.bytes
                .get(self.offset..end)
                .ok_or_else(|| PackingError::NativePayload {
                    detail: format!("truncated {field}"),
                })?;
        self.offset = end;
        Ok(bytes)
    }

    fn read_u32(&mut self, field: &str) -> Result<u32, PackingError> {
        let bytes = self.read_exact(4, field)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self, field: &str) -> Result<u64, PackingError> {
        let bytes = self.read_exact(8, field)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_bytes(&mut self, field: &str) -> Result<&'a [u8], PackingError> {
        let len = usize::try_from(self.read_u64(field)?).map_err(|_| PackingError::Arithmetic {
            invariant: "native payload field length to usize",
        })?;
        self.read_exact(len, field)
    }

    fn read_authority(&mut self, field: &str) -> Result<&'a str, PackingError> {
        let bytes = self.read_bytes(field)?;
        let value = std::str::from_utf8(bytes).map_err(|_| PackingError::NativePayload {
            detail: format!("{field} is not UTF-8"),
        })?;
        validate_authority_identifier("native payload authority", value)
            .map_err(PackingError::Format)?;
        Ok(value)
    }
}
