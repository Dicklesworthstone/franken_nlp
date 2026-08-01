//! Bounded, fail-closed source preparation for `fnlp convert`.
//!
//! This module owns the conversion-side trust boundary: all ten source files
//! are hashed in manifest order before a safetensors header is parsed, the
//! 201-tensor inventory is checked before routing, and every panel/output
//! range is sized with checked arithmetic.  It deliberately does *not* use
//! the in-memory `.fnlpq` serializer as a production writer: that serializer
//! remains a small-fixture oracle until the envelope grows its streaming sink.

use std::collections::BTreeSet;
use std::fmt;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::artifact::quantize::{encode_generic_panel, GenericPanelBytes, QuantizeError};
use crate::artifact::safetensors::{
    CensusDiff, RowPanel, SafetensorDtype, SafetensorsError, SafetensorsRangeIndex, SourceClosure,
    SourceDigest, TensorCensusEntry, TensorExpectation, diff_census_entries,
};
use crate::canonjson::{self, ParseLimits};

/// The complete source closure, including tokenizer/configuration files.
pub const PINNED_SOURCE_CLOSURE_BYTES: u64 = 8_360_887_509;
/// Logical BF16 payload represented by the 201 source tensors.
pub const PINNED_LOGICAL_PAYLOAD_BYTES: u64 = 8_339_601_408;
/// The exact tensor count accepted by the Nanbeige4.2-3B converter.
pub const PINNED_TENSOR_COUNT: usize = 201;
/// Maximum source bytes held by a single range read unless the caller chooses
/// a stricter panel budget.
pub const DEFAULT_PANEL_BYTES: u64 = 64 * 1024 * 1024;
/// The immutable source revision accepted by the initial converter recipe.
pub const PINNED_REVISION: &str = "f56ec5a9650268aa098496734743c25ea778bd2d";
/// Digest of the canonical checked-in ten-file source manifest.
pub const PINNED_SOURCE_MANIFEST_SHA256: &str =
    "9263a395e9a9c948cb66aeb9ef05147b5e6e7f848defcafd0ce6b2e31aacddca";
/// The first admitted portable conversion recipe.  It is intentionally a
/// recipe identity, never a loose `--quant` bit-width switch.
pub const PINNED_CONVERSION_RECIPE: &str = "nanbeige42-int8-v1";
/// Canonical Generic encoding for every quantized stage in the first recipe.
pub const PORTABLE_QUANT_V1: &str = "portable-quant-v1";
/// Canonical Generic encoding for source-preserved BF16 tensors.
pub const BF16_VERBATIM_V1: &str = "bf16-verbatim-v1";
/// The only canonical JSON schema emitted for conversion receipts.
pub const CONVERSION_RECEIPT_SCHEMA: &str = "fnlp-conversion-receipt-v1";

/// The only source-to-artifact target admitted by `fnlp convert`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConvertArch {
    /// Canonical cross-host generic representation.
    Generic,
}

impl ConvertArch {
    /// Parse the stable CLI spelling while refusing native derived packings.
    pub fn parse(value: &str) -> Result<Self, ConverterError> {
        match value {
            "generic" => Ok(Self::Generic),
            actual => Err(ConverterError::InvalidConvertArgument {
                argument: "--arch",
                expected: "generic".to_owned(),
                actual: actual.to_owned(),
            }),
        }
    }

    /// Stable CLI/receipt spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic",
        }
    }
}

/// Parsed `fnlp convert` arguments, independent of the CLI parser so unit
/// tests and robot dispatch share the same fail-closed validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConvertRequest {
    /// Pinned source closure directory.
    pub source_dir: PathBuf,
    /// Authenticated ten-file source manifest.
    pub source_manifest: PathBuf,
    /// Versioned conversion recipe id supplied via `--recipe`.
    pub recipe_id: String,
    /// Canonical target only.
    pub arch: ConvertArch,
    /// Final no-clobber `.fnlpq` destination.
    pub output: PathBuf,
    /// Interactive confirmation bypass.
    pub yes: bool,
    /// Reject all unrelated directory entries when true.
    pub strict_source_dir: bool,
    /// Emit versioned progress events on the robot data channel.
    pub robot: bool,
}

impl ConvertRequest {
    /// Validate immutable CLI semantics before hashing a potentially large
    /// source closure.  `--quant` is deliberately not represented anywhere in
    /// this request type.
    pub fn validate(&self) -> Result<(), ConverterError> {
        if self.recipe_id != PINNED_CONVERSION_RECIPE {
            return Err(ConverterError::InvalidConvertArgument {
                argument: "--recipe",
                expected: PINNED_CONVERSION_RECIPE.to_owned(),
                actual: self.recipe_id.clone(),
            });
        }
        if self.output.as_os_str().is_empty() {
            return Err(ConverterError::InvalidConvertArgument {
                argument: "-o/--output",
                expected: "a non-empty .fnlpq destination".to_owned(),
                actual: "empty".to_owned(),
            });
        }
        if self.output.extension().and_then(|value| value.to_str()) != Some("fnlpq") {
            return Err(ConverterError::InvalidConvertArgument {
                argument: "-o/--output",
                expected: "a .fnlpq destination".to_owned(),
                actual: self.output.display().to_string(),
            });
        }
        Ok(())
    }
}

/// One file included in the immutable conversion source closure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceFileSpec {
    /// Portable single-component source basename.
    pub name: String,
    /// Exact source byte length.
    pub bytes: u64,
    /// Expected lowercase SHA-256 digest.
    pub sha256: String,
}

/// Parsed, duplicate-key-rejecting source-closure manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConversionSourceManifest {
    /// Frozen manifest schema revision.
    pub schema_version: u64,
    /// Upstream model identity.
    pub model: String,
    /// Immutable upstream source revision.
    pub revision: String,
    /// Factual truth-pack state retained in the receipt.
    pub observed_state: String,
    /// Sum of all ten exact source files.
    pub closure_total_bytes: u64,
    /// Sum of safetensors payload ranges only.
    pub logical_safetensors_payload_bytes: u64,
    /// Safetensors container-header accounting, never model payload.
    pub safetensors_container_header_bytes: u64,
    /// Files in their canonical manifest/hash order.
    pub files: Vec<SourceFileSpec>,
}

impl ConversionSourceManifest {
    /// Parse a bounded canonical-JSON manifest and reject an ambiguous schema.
    pub fn parse(input: &str) -> Result<Self, ConverterError> {
        let value = canonjson::parse_str_with_limits(
            input,
            ParseLimits {
                max_depth: 8,
                max_string_bytes: 4096,
            },
        )
        .map_err(ConverterError::ManifestJson)?;
        let object = required_object(&value, "$", &MANIFEST_KEYS)?;
        let files_value = object
            .get("files")
            .ok_or_else(|| ConverterError::ManifestSchema {
                path: "$/files".to_owned(),
                detail: "missing required field".to_owned(),
            })?;
        let files = required_array(files_value, "$/files")?
            .iter()
            .enumerate()
            .map(|(index, file)| parse_source_file(file, index))
            .collect::<Result<Vec<_>, _>>()?;
        let manifest = Self {
            schema_version: required_u64(object, "schema_version", "$")?,
            model: required_string(object, "model", "$")?,
            revision: required_string(object, "revision", "$")?,
            observed_state: required_string(object, "observed_state", "$")?,
            closure_total_bytes: required_u64(object, "closure_total_bytes", "$")?,
            logical_safetensors_payload_bytes: required_u64(
                object,
                "logical_safetensors_payload_bytes",
                "$",
            )?,
            safetensors_container_header_bytes: required_u64(
                object,
                "safetensors_container_header_bytes",
                "$",
            )?,
            files,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// Read a bounded manifest without giving a hostile input an unbounded
    /// allocation path.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConverterError> {
        let path = path.as_ref();
        let metadata = fs::metadata(path).map_err(|error| ConverterError::Io {
            path: path.to_path_buf(),
            operation: "stat source manifest",
            detail: error.to_string(),
        })?;
        if metadata.len() > 1024 * 1024 {
            return Err(ConverterError::ManifestTooLarge {
                path: path.to_path_buf(),
                observed: metadata.len(),
                cap: 1024 * 1024,
            });
        }
        let input = fs::read_to_string(path).map_err(|error| ConverterError::Io {
            path: path.to_path_buf(),
            operation: "read source manifest",
            detail: error.to_string(),
        })?;
        Self::parse(&input)
    }

    /// Read and authenticate the only accepted pinned ten-file manifest.
    pub fn load_pinned(path: impl AsRef<Path>) -> Result<Self, ConverterError> {
        let path = path.as_ref();
        let metadata = fs::metadata(path).map_err(|error| ConverterError::Io {
            path: path.to_path_buf(),
            operation: "stat pinned source manifest",
            detail: error.to_string(),
        })?;
        if metadata.len() > 1024 * 1024 {
            return Err(ConverterError::ManifestTooLarge {
                path: path.to_path_buf(),
                observed: metadata.len(),
                cap: 1024 * 1024,
            });
        }
        let bytes = fs::read(path).map_err(|error| ConverterError::Io {
            path: path.to_path_buf(),
            operation: "read pinned source manifest",
            detail: error.to_string(),
        })?;
        let actual = sha256_hex(&bytes);
        if actual != PINNED_SOURCE_MANIFEST_SHA256 {
            return Err(ConverterError::PinnedManifestDigest {
                path: path.to_path_buf(),
                expected: PINNED_SOURCE_MANIFEST_SHA256.to_owned(),
                actual,
            });
        }
        let input =
            std::str::from_utf8(&bytes).map_err(|error| ConverterError::ManifestSchema {
                path: "$".to_owned(),
                detail: format!("manifest is not UTF-8: {error}"),
            })?;
        let manifest = Self::parse(input)?;
        manifest.validate_pinned()?;
        Ok(manifest)
    }

    /// Convert the manifest's index plus two weight shards into the checked
    /// range-index closure.  All other source files are verified first by
    /// [`verify_source_closure`].
    pub fn safetensors_closure(&self) -> Result<SourceClosure, ConverterError> {
        let index = self
            .files
            .iter()
            .find(|file| file.name == "model.safetensors.index.json")
            .ok_or_else(|| ConverterError::ManifestSchema {
                path: "$/files".to_owned(),
                detail: "missing model.safetensors.index.json".to_owned(),
            })?;
        let index = source_digest(index)?;
        let mut shards = self
            .files
            .iter()
            .filter(|file| file.name.ends_with(".safetensors"))
            .map(source_digest)
            .collect::<Result<Vec<_>, _>>()?;
        shards.sort_by(|left, right| left.file_name().cmp(right.file_name()));
        SourceClosure::new(index, shards).map_err(ConverterError::Safetensors)
    }

    fn validate(&self) -> Result<(), ConverterError> {
        if self.schema_version != 1 {
            return Err(ConverterError::ManifestSchema {
                path: "$/schema_version".to_owned(),
                detail: format!("expected 1, observed {}", self.schema_version),
            });
        }
        if self.files.len() != 10 {
            return Err(ConverterError::ManifestSchema {
                path: "$/files".to_owned(),
                detail: format!("expected exactly 10 files, observed {}", self.files.len()),
            });
        }
        let mut names = BTreeSet::new();
        let total = self.files.iter().try_fold(0_u64, |sum, file| {
            if !is_safe_basename(&file.name) {
                return Err(ConverterError::ManifestSchema {
                    path: "$/files".to_owned(),
                    detail: format!("unsafe file name {:?}", file.name),
                });
            }
            if !is_lower_sha256(&file.sha256) {
                return Err(ConverterError::ManifestSchema {
                    path: "$/files".to_owned(),
                    detail: format!("invalid SHA-256 for {}", file.name),
                });
            }
            if !names.insert(file.name.clone()) {
                return Err(ConverterError::ManifestSchema {
                    path: "$/files".to_owned(),
                    detail: format!("duplicate closure member {}", file.name),
                });
            }
            sum.checked_add(file.bytes)
                .ok_or(ConverterError::Arithmetic {
                    invariant: "source closure byte sum",
                })
        })?;
        if total != self.closure_total_bytes {
            return Err(ConverterError::ManifestSchema {
                path: "$/closure_total_bytes".to_owned(),
                detail: format!(
                    "expected sum {total}, observed {}",
                    self.closure_total_bytes
                ),
            });
        }
        Ok(())
    }

    fn validate_pinned(&self) -> Result<(), ConverterError> {
        if self.model != "Nanbeige4.2-3B"
            || self.revision != PINNED_REVISION
            || self.closure_total_bytes != PINNED_SOURCE_CLOSURE_BYTES
            || self.logical_safetensors_payload_bytes != PINNED_LOGICAL_PAYLOAD_BYTES
            || self.safetensors_container_header_bytes != 23_312
            || self.observed_state != "OBSERVED@pin"
        {
            return Err(ConverterError::PinnedManifestContract {
                detail: "model/revision/byte accounting differs from the frozen source closure"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

const MANIFEST_KEYS: [&str; 8] = [
    "closure_total_bytes",
    "files",
    "logical_safetensors_payload_bytes",
    "model",
    "observed_state",
    "revision",
    "safetensors_container_header_bytes",
    "schema_version",
];

const SOURCE_FILE_KEYS: [&str; 3] = ["bytes", "name", "sha256"];

/// Result of serial, manifest-order source verification.  No tensor parser is
/// reachable until this value exists.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedSourceClosure {
    /// Authenticated source manifest.
    pub manifest: ConversionSourceManifest,
    /// Per-file observed identities in canonical manifest order.
    pub files: Vec<VerifiedSourceFile>,
    /// Domain-framed root identity carried into the `.fnlpq` header/receipt.
    pub source_root_sha256: String,
}

/// One source-file verification receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedSourceFile {
    /// Source basename.
    pub name: String,
    /// Observed exact byte length.
    pub bytes: u64,
    /// Observed lowercase SHA-256.
    pub sha256: String,
}

/// Verify every declared member before safetensors/header parsing.
///
/// `strict_source_dir=false` permits unrelated ordinary files.  It never
/// permits a second shard/index or a configuration/tokenizer lookalike: such
/// files could alter loading semantics and are refused in both modes.
pub fn verify_source_closure(
    source_dir: impl AsRef<Path>,
    manifest: ConversionSourceManifest,
    strict_source_dir: bool,
) -> Result<VerifiedSourceClosure, ConverterError> {
    manifest.validate()?;
    let source_dir = source_dir.as_ref();
    verify_source_directory(source_dir, &manifest, strict_source_dir)?;
    let mut files = Vec::with_capacity(manifest.files.len());
    for expected in &manifest.files {
        files.push(verify_source_member(source_dir, expected)?);
    }
    let source_root_sha256 = source_root_digest(&files)?;
    Ok(VerifiedSourceClosure {
        manifest,
        files,
        source_root_sha256,
    })
}

fn verify_source_directory(
    source_dir: &Path,
    manifest: &ConversionSourceManifest,
    strict_source_dir: bool,
) -> Result<(), ConverterError> {
    let expected: BTreeSet<_> = manifest
        .files
        .iter()
        .map(|file| file.name.as_str())
        .collect();
    let entries = fs::read_dir(source_dir).map_err(|error| ConverterError::Io {
        path: source_dir.to_path_buf(),
        operation: "list source directory",
        detail: error.to_string(),
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| ConverterError::Io {
            path: source_dir.to_path_buf(),
            operation: "read source directory entry",
            detail: error.to_string(),
        })?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if expected.contains(name.as_str()) {
            continue;
        }
        if is_semantic_conflict(&name) {
            return Err(ConverterError::SemanticSourceExtra {
                path: entry.path(),
                name,
                next_command: "remove the conflicting source artifact, then rerun fnlp convert"
                    .to_owned(),
            });
        }
        if strict_source_dir {
            return Err(ConverterError::StrictSourceExtra {
                path: entry.path(),
                name,
                next_command: "remove the extra file or rerun without --strict-source-dir"
                    .to_owned(),
            });
        }
    }
    Ok(())
}

fn verify_source_member(
    source_dir: &Path,
    expected: &SourceFileSpec,
) -> Result<VerifiedSourceFile, ConverterError> {
    let path = source_dir.join(&expected.name);
    let metadata = fs::symlink_metadata(&path).map_err(|error| ConverterError::SourceMissing {
        path: path.clone(),
        expected: expected.name.clone(),
        detail: error.to_string(),
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(ConverterError::SourceNotRegular {
            path,
            expected: expected.name.clone(),
        });
    }
    if metadata.len() != expected.bytes {
        return Err(ConverterError::SourceLength {
            path,
            expected: expected.bytes,
            actual: metadata.len(),
            next_command: "restore the pinned source closure and rerun fnlp convert".to_owned(),
        });
    }
    let mut file = File::open(&path).map_err(|error| ConverterError::Io {
        path: path.clone(),
        operation: "open source member",
        detail: error.to_string(),
    })?;
    let actual = stream_sha256(&mut file, &path)?;
    if actual != expected.sha256 {
        return Err(ConverterError::SourceDigest {
            path,
            expected: expected.sha256.clone(),
            actual,
            next_command: "restore the pinned source closure and rerun fnlp convert".to_owned(),
        });
    }
    Ok(VerifiedSourceFile {
        name: expected.name.clone(),
        bytes: expected.bytes,
        sha256: expected.sha256.clone(),
    })
}

/// Exact storage policy selected by the frozen converter route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageStage {
    /// Copy source BF16 bytes exactly into the generic logical payload.
    Bf16Verbatim,
    /// MLP projection quantization, owned by portable-quant stage 2a.
    Int8Stage2A,
    /// Q/K/V/O projection quantization, owned by portable-quant stage 2b.
    Int8Stage2B,
    /// Final projection quantization, owned by portable-quant stage 2c.
    Int8Stage2C,
}

impl StorageStage {
    /// Stable stage spelling for diagnostics and receipts.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16Verbatim => "bf16-verbatim",
            Self::Int8Stage2A => "int8-stage-2a",
            Self::Int8Stage2B => "int8-stage-2b",
            Self::Int8Stage2C => "int8-stage-2c",
        }
    }
}

/// One total mapping from a source tensor to an internal logical tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorRoute {
    /// Source HF dotted name.
    pub source_name: String,
    /// Canonical internal logical tensor name.
    pub internal_name: String,
    /// Storage/quantization stage.
    pub stage: StorageStage,
}

/// Map one source tensor under the one-model conversion contract.
pub fn remap_tensor_name(name: &str) -> Result<TensorRoute, ConverterError> {
    if name.contains("bias") {
        return Err(ConverterError::BiasTensor {
            tensor: name.to_owned(),
        });
    }
    let route = match name {
        "model.embed_tokens.weight" => ("embed".to_owned(), StorageStage::Bf16Verbatim),
        "model.norm.weight" => ("final_norm".to_owned(), StorageStage::Bf16Verbatim),
        "lm_head.weight" => ("lm_head".to_owned(), StorageStage::Int8Stage2C),
        _ => remap_layer_tensor(name)?,
    };
    Ok(TensorRoute {
        source_name: name.to_owned(),
        internal_name: route.0,
        stage: route.1,
    })
}

fn remap_layer_tensor(name: &str) -> Result<(String, StorageStage), ConverterError> {
    let prefix = "model.layers.";
    let remainder =
        name.strip_prefix(prefix)
            .ok_or_else(|| ConverterError::UnknownTensorRoute {
                tensor: name.to_owned(),
            })?;
    let (layer, suffix) =
        remainder
            .split_once('.')
            .ok_or_else(|| ConverterError::UnknownTensorRoute {
                tensor: name.to_owned(),
            })?;
    let layer = layer
        .parse::<u8>()
        .map_err(|_| ConverterError::UnknownTensorRoute {
            tensor: name.to_owned(),
        })?;
    if layer >= 22 {
        return Err(ConverterError::UnknownTensorRoute {
            tensor: name.to_owned(),
        });
    }
    // The frozen fnlpq v1 authority grammar excludes bracket indexing.  Keep
    // the internal route explicit but spell its layer index with identifier
    // characters so the planned tensor name is writer-valid before traversal.
    let base = format!("layer.{layer}");
    match suffix {
        "input_layernorm.weight" => Ok((format!("{base}.norm1"), StorageStage::Bf16Verbatim)),
        "post_attention_layernorm.weight" => {
            Ok((format!("{base}.norm2"), StorageStage::Bf16Verbatim))
        }
        "self_attn.q_proj.weight" => Ok((format!("{base}.attn.q"), StorageStage::Int8Stage2B)),
        "self_attn.k_proj.weight" => Ok((format!("{base}.attn.k"), StorageStage::Int8Stage2B)),
        "self_attn.v_proj.weight" => Ok((format!("{base}.attn.v"), StorageStage::Int8Stage2B)),
        "self_attn.o_proj.weight" => Ok((format!("{base}.attn.o"), StorageStage::Int8Stage2B)),
        "mlp.gate_proj.weight" => Ok((format!("{base}.mlp.gate"), StorageStage::Int8Stage2A)),
        "mlp.up_proj.weight" => Ok((format!("{base}.mlp.up"), StorageStage::Int8Stage2A)),
        "mlp.down_proj.weight" => Ok((format!("{base}.mlp.down"), StorageStage::Int8Stage2A)),
        _ => Err(ConverterError::UnknownTensorRoute {
            tensor: name.to_owned(),
        }),
    }
}

/// Generate the exact 201-tensor inventory from frozen dimensions.  This is
/// intentionally code, rather than a missing/optional fixture: conversion
/// must not proceed if a census artifact has not been materialized.
pub fn expected_nanbeige42_census() -> Vec<TensorExpectation> {
    const HIDDEN: u64 = 3_072;
    const INTERMEDIATE: u64 = 10_752;
    const VOCAB: u64 = 166_144;
    const Q: u64 = 6_144;
    const KV: u64 = 1_024;
    let mut output = vec![
        expectation("lm_head.weight", &[VOCAB, HIDDEN]),
        expectation("model.embed_tokens.weight", &[VOCAB, HIDDEN]),
        expectation("model.norm.weight", &[HIDDEN]),
    ];
    for layer in 0..22 {
        let prefix = format!("model.layers.{layer}");
        output.extend([
            expectation(&format!("{prefix}.input_layernorm.weight"), &[HIDDEN]),
            expectation(
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[HIDDEN],
            ),
            expectation(&format!("{prefix}.self_attn.q_proj.weight"), &[Q, HIDDEN]),
            expectation(&format!("{prefix}.self_attn.k_proj.weight"), &[KV, HIDDEN]),
            expectation(&format!("{prefix}.self_attn.v_proj.weight"), &[KV, HIDDEN]),
            expectation(&format!("{prefix}.self_attn.o_proj.weight"), &[HIDDEN, Q]),
            expectation(
                &format!("{prefix}.mlp.gate_proj.weight"),
                &[INTERMEDIATE, HIDDEN],
            ),
            expectation(
                &format!("{prefix}.mlp.up_proj.weight"),
                &[INTERMEDIATE, HIDDEN],
            ),
            expectation(
                &format!("{prefix}.mlp.down_proj.weight"),
                &[HIDDEN, INTERMEDIATE],
            ),
        ]);
    }
    output.sort_by(|left, right| left.name.cmp(&right.name));
    debug_assert_eq!(output.len(), PINNED_TENSOR_COUNT);
    output
}

/// Diff a fully parsed range-index census against the converter's complete
/// table.  Any OQ-1 family is a design-assumption abort, not a tolerated
/// generic `EXTRA` row.
pub fn validate_nanbeige42_census(
    actual: &[TensorCensusEntry],
) -> Result<CensusDiff, ConverterError> {
    if let Some(tensor) = actual.iter().find(|entry| is_oq1_tripwire(&entry.name)) {
        return Err(ConverterError::Oq1Tripwire {
            tensor: tensor.name.clone(),
        });
    }
    let diff = diff_census_entries(actual, &expected_nanbeige42_census())
        .map_err(ConverterError::Safetensors)?;
    if diff.is_match() {
        Ok(diff)
    } else {
        Err(ConverterError::CensusMismatch { diff })
    }
}

/// Fully checked input facts that may enter the serial panel pipeline.
///
/// Construction order is intentional: the ten-file closure is authenticated
/// before the safetensors parser is constructed, then the parsed census is
/// rejected before any tensor is routed or decoded.
pub struct PreparedConversionInput {
    /// Ten-file source closure evidence in manifest order.
    pub source: VerifiedSourceClosure,
    /// Checked source facts in canonical tensor-name order.
    pub census: Vec<TensorCensusEntry>,
    /// One complete route for every census entry.
    pub routes: Vec<TensorRoute>,
    /// One bounded row-panel plan for every source tensor.
    pub panels: Vec<PanelPlan>,
    /// Checked total BF16 source payload across the complete census.  This is
    /// the fixed denominator for progress and ETA accounting.
    pub logical_payload_bytes: u64,
    /// Domain-framed exact source census identity.
    pub census_sha256: String,
    /// The verified safetensors handles that supplied this census.
    ///
    /// This intentionally remains private: later conversion passes must reuse
    /// this authenticated range session rather than reopen mutable shard
    /// pathnames after preparation.
    range_index: SafetensorsRangeIndex,
}

/// One tensor's precomputed destinations in the three Generic payload
/// sections.  The plan carries only small layout metadata; it never retains a
/// source tensor or a quantized tensor image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenericTensorLayout {
    /// Exact source tensor name in the checked safetensors census.
    pub source_name: String,
    /// Canonical internal tensor identity used to route converter panels.
    ///
    /// The artifact header deliberately uses `source_name`: its frozen
    /// census is an external authority, while this route name is local
    /// converter implementation detail.
    pub internal_name: String,
    /// Frozen converter route that determines the Generic representation.
    pub stage: StorageStage,
    /// Exact source shape, retained for a later logical-digest pass.
    pub shape: Vec<u64>,
    /// Versioned Generic encoding identifier for this logical tensor.
    pub quantization: String,
    /// Destination range in `GENERIC_TENSOR_PAYLOAD`.
    pub data: OutputRange,
    /// Destination range in `GENERIC_TENSOR_SCALES`.
    pub scale: OutputRange,
    /// Destination range in `GENERIC_TENSOR_ROW_SUMS`.
    pub row_sum: OutputRange,
}

/// Exact section byte totals and per-tensor mappings calculated before any
/// output file is created.  The later streaming-envelope bridge uses this to
/// allocate header/directory metadata and to enforce one-write coverage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenericPayloadPlan {
    /// Mappings in checked census order.
    pub tensors: Vec<GenericTensorLayout>,
    /// Exact stored bytes for `GENERIC_TENSOR_PAYLOAD`.
    pub payload_bytes: u64,
    /// Exact stored bytes for `GENERIC_TENSOR_SCALES`.
    pub scale_bytes: u64,
    /// Exact stored bytes for `GENERIC_TENSOR_ROW_SUMS`.
    pub row_sum_bytes: u64,
}

/// Precompute every bounded Generic tensor destination before source-panel
/// traversal or staging-file creation.
///
/// BF16-verbatim routes retain their exact source byte count and claim no
/// scale/row-sum bytes.  The portable int8 routes emit one signed byte per
/// BF16 scalar plus one little-endian f32 scale and i32 row sum per output
/// row.  This is only a layout plan: the later first pass still hashes and
/// validates every emitted byte before it can construct a streaming envelope.
pub fn plan_generic_payload(
    census: &[TensorCensusEntry],
    routes: &[TensorRoute],
) -> Result<GenericPayloadPlan, ConverterError> {
    if census.len() != routes.len() {
        return Err(ConverterError::PipelinePlanCount {
            census: census.len(),
            routes: routes.len(),
            panels: 0,
        });
    }

    let mut tensors = Vec::with_capacity(census.len());
    let mut internal_names = BTreeSet::new();
    let mut payload_offset = 0_u64;
    let mut scale_offset = 0_u64;
    let mut row_sum_offset = 0_u64;

    for (entry, route) in census.iter().zip(routes) {
        let expected_route = remap_tensor_name(&entry.name)?;
        if route != &expected_route {
            return Err(ConverterError::PipelinePlanAlignment {
                tensor: entry.name.clone(),
                detail: "route differs from canonical mapping".to_owned(),
            });
        }
        if entry.dtype != SafetensorDtype::Bf16 {
            return Err(ConverterError::UnexpectedDtype {
                tensor: entry.name.clone(),
                expected: SafetensorDtype::Bf16,
                actual: entry.dtype,
            });
        }
        if !internal_names.insert(route.internal_name.clone()) {
            return Err(ConverterError::PipelinePlanAlignment {
                tensor: entry.name.clone(),
                detail: format!("duplicate internal tensor name {}", route.internal_name),
            });
        }
        let rows = *entry
            .shape
            .first()
            .ok_or_else(|| ConverterError::InvalidPanelPlan {
                tensor: entry.name.clone(),
                detail: "rank-zero tensor has no Generic row layout".to_owned(),
            })?;
        let scalar_count = entry.len.checked_div(2).ok_or(ConverterError::Arithmetic {
            invariant: "BF16 scalar count",
        })?;
        if scalar_count
            .checked_mul(2)
            .ok_or(ConverterError::Arithmetic {
                invariant: "BF16 scalar byte reconstruction",
            })?
            != entry.len
        {
            return Err(ConverterError::InvalidPanelPlan {
                tensor: entry.name.clone(),
                detail: "BF16 source byte length is not scalar-aligned".to_owned(),
            });
        }
        let (data_len, scale_len, row_sum_len, quantization) = match route.stage {
            StorageStage::Bf16Verbatim => (entry.len, 0, 0, BF16_VERBATIM_V1),
            StorageStage::Int8Stage2A | StorageStage::Int8Stage2B | StorageStage::Int8Stage2C => {
                let sidecar_len = rows.checked_mul(4).ok_or(ConverterError::Arithmetic {
                    invariant: "portable int8 row sidecar bytes",
                })?;
                (scalar_count, sidecar_len, sidecar_len, PORTABLE_QUANT_V1)
            }
        };
        let data = OutputRange {
            name: format!("{}.data", route.internal_name),
            offset: payload_offset,
            len: data_len,
        };
        let scale = OutputRange {
            name: format!("{}.scale", route.internal_name),
            offset: scale_offset,
            len: scale_len,
        };
        let row_sum = OutputRange {
            name: format!("{}.row_sum", route.internal_name),
            offset: row_sum_offset,
            len: row_sum_len,
        };
        payload_offset =
            payload_offset
                .checked_add(data_len)
                .ok_or(ConverterError::Arithmetic {
                    invariant: "Generic payload section bytes",
                })?;
        scale_offset = scale_offset
            .checked_add(scale_len)
            .ok_or(ConverterError::Arithmetic {
                invariant: "Generic scale section bytes",
            })?;
        row_sum_offset =
            row_sum_offset
                .checked_add(row_sum_len)
                .ok_or(ConverterError::Arithmetic {
                    invariant: "Generic row-sum section bytes",
                })?;
        tensors.push(GenericTensorLayout {
            source_name: entry.name.clone(),
            internal_name: route.internal_name.clone(),
            stage: route.stage,
            shape: entry.shape.clone(),
            quantization: quantization.to_owned(),
            data,
            scale,
            row_sum,
        });
    }
    Ok(GenericPayloadPlan {
        tensors,
        payload_bytes: payload_offset,
        scale_bytes: scale_offset,
        row_sum_bytes: row_sum_offset,
    })
}

/// Execute the refusal-first converter preparation path.
pub fn prepare_nanbeige42_input(
    source_dir: impl AsRef<Path>,
    manifest: ConversionSourceManifest,
    strict_source_dir: bool,
    panel_cap: u64,
) -> Result<PreparedConversionInput, ConverterError> {
    let source_dir = source_dir.as_ref();
    let source = verify_source_closure(source_dir, manifest, strict_source_dir)?;
    let range_index = SafetensorsRangeIndex::open_pinned_nanbeige42(source_dir)
        .map_err(ConverterError::Safetensors)?;
    let census = range_index.census();
    validate_nanbeige42_census(&census)?;
    let routes = census
        .iter()
        .map(|entry| remap_tensor_name(&entry.name))
        .collect::<Result<Vec<_>, _>>()?;
    let panels = census
        .iter()
        .map(|entry| PanelPlan::for_tensor(entry, panel_cap))
        .collect::<Result<Vec<_>, _>>()?;
    let logical_payload_bytes = validate_pinned_logical_payload_bytes(&census)?;
    let census_sha256 = census_digest(&census)?;
    Ok(PreparedConversionInput {
        source,
        census,
        routes,
        panels,
        logical_payload_bytes,
        census_sha256,
        range_index,
    })
}

/// Prepare the one admitted `fnlp convert` request before any output-side
/// allocation or staging-file creation.
///
/// This is deliberately the command boundary rather than a convenience
/// wrapper: immutable CLI semantics are rejected before any source I/O, the
/// checked pinned manifest is authenticated before the source directory is
/// traversed, and the full closure/census route is established before a later
/// streaming envelope may plan output bytes.  Successful preparation is not a
/// conversion result and does not authorize creation of `request.output`.
pub fn prepare_convert_request(
    request: &ConvertRequest,
    panel_cap: u64,
) -> Result<PreparedConversionInput, ConverterError> {
    request.validate()?;
    let manifest = ConversionSourceManifest::load_pinned(&request.source_manifest)?;
    prepare_nanbeige42_input(
        &request.source_dir,
        manifest,
        request.strict_source_dir,
        panel_cap,
    )
}

/// Sum a validated source census and bind its payload total to the pinned
/// model.  Call this after [`validate_nanbeige42_census`]: it gives progress
/// and preflight code one fixed logical-byte denominator rather than allowing
/// a second, independently maintained total.
pub fn validate_pinned_logical_payload_bytes(
    census: &[TensorCensusEntry],
) -> Result<u64, ConverterError> {
    let actual = census.iter().try_fold(0_u64, |total, tensor| {
        total
            .checked_add(tensor.len)
            .ok_or(ConverterError::Arithmetic {
                invariant: "pinned logical payload census sum",
            })
    })?;
    if actual != PINNED_LOGICAL_PAYLOAD_BYTES {
        return Err(ConverterError::CensusPayloadBytes {
            expected: PINNED_LOGICAL_PAYLOAD_BYTES,
            actual,
        });
    }
    Ok(actual)
}

impl PreparedConversionInput {
    /// Read one bounded BF16 panel through the sealed safetensors session that
    /// was authenticated while preparing this conversion.
    ///
    /// Callers cannot replace the source directory or reopen a shard between
    /// preflight and emission. The range index retains the verified file
    /// handles, including the parsed tensor-to-shard authority.
    pub(crate) fn read_verified_panel(
        &self,
        tensor: &str,
        panel: RowPanel,
    ) -> Result<Vec<u8>, ConverterError> {
        self.range_index
            .read_range(tensor, panel)
            .map_err(ConverterError::Safetensors)
    }

    /// Calculate the pre-conversion footprint from actual parsed panel plans.
    /// The stage/output planner supplies its independently bounded buffers.
    pub fn preflight(
        &self,
        staged_output_bytes: u64,
        quant_packing_scratch_bytes: u64,
        output_buffer_bytes: u64,
        parser_metadata_bytes: u64,
        margin_bytes: u64,
    ) -> Result<ConversionPreflight, ConverterError> {
        let largest_source_panel_bytes = self
            .panels
            .iter()
            .map(|plan| plan.largest_source_panel_bytes)
            .max()
            .ok_or(ConverterError::Arithmetic {
                invariant: "largest source panel with empty census",
            })?;
        let largest_f32_panel_bytes = self
            .panels
            .iter()
            .map(|plan| plan.largest_f32_panel_bytes)
            .max()
            .ok_or(ConverterError::Arithmetic {
                invariant: "largest f32 panel with empty census",
            })?;
        let peak_rss = PeakRssFormula {
            largest_source_panel_bytes,
            largest_f32_panel_bytes,
            quant_packing_scratch_bytes,
            output_buffer_bytes,
            parser_metadata_bytes,
            margin_bytes,
        };
        let final_disk_bytes =
            staged_output_bytes
                .checked_add(margin_bytes)
                .ok_or(ConverterError::Arithmetic {
                    invariant: "preflight final disk bytes",
                })?;
        Ok(ConversionPreflight {
            closure_bytes_to_read: self.source.manifest.closure_total_bytes,
            staged_output_bytes,
            peak_rss,
            final_disk_bytes,
        })
    }
}

/// A machine-readable conversion receipt prepared before activation.  It has
/// no optional identity fields: an incomplete receipt is not serializable.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConversionReceipt {
    /// Fixed schema discriminator for a retained conversion receipt.
    pub receipt_schema: String,
    /// Framed ordered source-file identity.
    pub source_root_sha256: String,
    /// Framed complete 201-tensor source census identity.
    pub census_sha256: String,
    /// Exact converter source revision/commit supplied by the build layer.
    pub converter_commit: String,
    /// Frozen recipe identity passed via `--recipe`.
    pub recipe_id: String,
    /// Versioned rounding contract identity.
    pub rounding_id: String,
    /// Versioned generic packing contract identity.
    pub packing_id: String,
    /// Measured/proxy peak enforced against the printed formula.
    pub measured_peak_rss_bytes: u64,
    /// Largest explicitly measured conversion scratch allocation.
    pub measured_scratch_bytes: u64,
    /// Printed formula cap.
    pub peak_rss_cap_bytes: u64,
    /// Exact final disk requirement observed by preflight.
    pub final_disk_bytes: u64,
    /// Observed durable staged artifact length before activation.
    pub measured_disk_bytes: u64,
    /// Exact activated artifact length.
    pub output_len: u64,
    /// Exact completed `.fnlpq` file SHA-256.
    pub output_sha256: String,
    /// Exact embedded Apache-2.0/attribution/notice identity.
    pub license_bundle_sha256: String,
}

impl ConversionReceipt {
    /// Reject incomplete or non-canonical identities before the receipt can be
    /// written alongside an activation journal.
    pub fn validate(&self) -> Result<(), ConverterError> {
        if self.receipt_schema != CONVERSION_RECEIPT_SCHEMA {
            return Err(ConverterError::ReceiptField {
                field: "receipt_schema",
                detail: format!("expected {CONVERSION_RECEIPT_SCHEMA}"),
            });
        }
        for (field, value) in [
            ("source_root_sha256", self.source_root_sha256.as_str()),
            ("census_sha256", self.census_sha256.as_str()),
            ("output_sha256", self.output_sha256.as_str()),
            ("license_bundle_sha256", self.license_bundle_sha256.as_str()),
        ] {
            if !is_lower_sha256(value) {
                return Err(ConverterError::ReceiptField {
                    field,
                    detail: "must be a lowercase SHA-256".to_owned(),
                });
            }
        }
        if !is_lower_git_commit(&self.converter_commit) {
            return Err(ConverterError::ReceiptField {
                field: "converter_commit",
                detail: "must be a lowercase 40-character Git commit".to_owned(),
            });
        }
        for (field, value) in [
            ("recipe_id", self.recipe_id.as_str()),
            ("rounding_id", self.rounding_id.as_str()),
            ("packing_id", self.packing_id.as_str()),
        ] {
            if value.is_empty() || !value.is_ascii() {
                return Err(ConverterError::ReceiptField {
                    field,
                    detail: "must be non-empty ASCII".to_owned(),
                });
            }
        }
        if self.measured_peak_rss_bytes > self.peak_rss_cap_bytes {
            return Err(ConverterError::PeakRssExceeded {
                observed: self.measured_peak_rss_bytes,
                cap: self.peak_rss_cap_bytes,
            });
        }
        if self.measured_scratch_bytes > self.measured_peak_rss_bytes {
            return Err(ConverterError::ReceiptField {
                field: "measured_scratch_bytes",
                detail: "must not exceed measured_peak_rss_bytes".to_owned(),
            });
        }
        if self.measured_disk_bytes != self.output_len {
            return Err(ConverterError::ReceiptField {
                field: "measured_disk_bytes",
                detail: "must equal output_len for the single staged artifact".to_owned(),
            });
        }
        if self.final_disk_bytes < self.measured_disk_bytes {
            return Err(ConverterError::ReceiptField {
                field: "final_disk_bytes",
                detail: "must cover measured_disk_bytes".to_owned(),
            });
        }
        Ok(())
    }

    /// Canonical NDJSON-ready receipt bytes, after completeness validation.
    pub fn canonical_json(&self) -> Result<String, ConverterError> {
        self.validate()?;
        canonjson::canonical_string(self).map_err(ConverterError::ReceiptJson)
    }

    /// Parse only canonical JSON receipt bytes through the duplicate-key
    /// rejecting repository boundary.
    pub fn parse_canonical_json(input: &str) -> Result<Self, ConverterError> {
        let value = canonjson::parse_str_with_limits(input, ParseLimits::default())
            .map_err(ConverterError::ReceiptJson)?;
        let canonical = canonjson::canonical_bytes(&value).map_err(ConverterError::ReceiptJson)?;
        if canonical.as_slice() != input.as_bytes() {
            return Err(ConverterError::ReceiptNonCanonical);
        }
        let receipt = serde_json::from_value(value).map_err(|error| ConverterError::ReceiptParse {
            detail: error.to_string(),
        })?;
        receipt.validate()?;
        Ok(receipt)
    }
}

/// A bounded source-panel plan for a single BF16 tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PanelPlan {
    /// Source tensor name.
    pub tensor: String,
    /// Exact number of bytes per first-dimension row.
    pub bytes_per_row: u64,
    /// Number of rows in every full panel.
    pub rows_per_panel: u64,
    /// Total panels required for this tensor.
    pub panel_count: u64,
    /// Largest source byte allocation for this tensor.
    pub largest_source_panel_bytes: u64,
    /// Largest decoded f32 work panel allocation.
    pub largest_f32_panel_bytes: u64,
}

/// One lazy, source-order stream of bounded row panels for a tensor.
///
/// This keeps panel bookkeeping proportional to the current panel rather than
/// to the number of rows in a strict-cap conversion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RowPanelIter {
    total_rows: u64,
    next_row: u64,
    rows_per_panel: u64,
}

impl Iterator for RowPanelIter {
    type Item = RowPanel;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_row >= self.total_rows {
            return None;
        }
        let row_count = self.rows_per_panel.min(self.total_rows - self.next_row);
        let panel = RowPanel::Rows {
            start_row: self.next_row,
            row_count,
        };
        self.next_row += row_count;
        Some(panel)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_rows = self.total_rows - self.next_row;
        let remaining_panels = remaining_rows / self.rows_per_panel
            + u64::from(remaining_rows % self.rows_per_panel != 0);
        match usize::try_from(remaining_panels) {
            Ok(exact) => (exact, Some(exact)),
            Err(_) => (usize::MAX, None),
        }
    }
}

impl PanelPlan {
    /// Derive a panel plan before allocating source or f32 work bytes.
    pub fn for_tensor(entry: &TensorCensusEntry, panel_cap: u64) -> Result<Self, ConverterError> {
        if entry.dtype != SafetensorDtype::Bf16 {
            return Err(ConverterError::UnexpectedDtype {
                tensor: entry.name.clone(),
                expected: SafetensorDtype::Bf16,
                actual: entry.dtype,
            });
        }
        if entry.shape.is_empty() || panel_cap == 0 {
            return Err(ConverterError::InvalidPanelPlan {
                tensor: entry.name.clone(),
                detail: "rank-zero tensor or zero panel cap".to_owned(),
            });
        }
        let rows = entry.shape[0];
        if rows == 0 || entry.shape.iter().any(|dimension| *dimension == 0) {
            return Err(ConverterError::InvalidPanelPlan {
                tensor: entry.name.clone(),
                detail: "zero dimensions are not valid model tensor shapes".to_owned(),
            });
        }
        let columns = entry.shape[1..]
            .iter()
            .try_fold(1_u64, |value, dimension| {
                value
                    .checked_mul(*dimension)
                    .ok_or(ConverterError::Arithmetic {
                        invariant: "panel column product",
                    })
            })?;
        let bytes_per_row = columns.checked_mul(2).ok_or(ConverterError::Arithmetic {
            invariant: "BF16 panel row bytes",
        })?;
        let expected_len = rows
            .checked_mul(bytes_per_row)
            .ok_or(ConverterError::Arithmetic {
                invariant: "BF16 tensor byte length for panel plan",
            })?;
        if entry.len != expected_len {
            return Err(ConverterError::InvalidPanelPlan {
                tensor: entry.name.clone(),
                detail: format!(
                    "shape implies {expected_len} BF16 bytes but census declares {}",
                    entry.len
                ),
            });
        }
        if bytes_per_row > panel_cap {
            return Err(ConverterError::RowExceedsPanelCap {
                tensor: entry.name.clone(),
                row_bytes: bytes_per_row,
                cap: panel_cap,
            });
        }
        let rows_per_panel = panel_cap / bytes_per_row;
        let panel_count = div_ceil(rows, rows_per_panel)?;
        let largest_rows = rows.min(rows_per_panel);
        let largest_source_panel_bytes =
            largest_rows
                .checked_mul(bytes_per_row)
                .ok_or(ConverterError::Arithmetic {
                    invariant: "largest source panel bytes",
                })?;
        let largest_f32_panel_bytes = largest_source_panel_bytes
            .checked_div(2)
            .and_then(|scalars| scalars.checked_mul(4))
            .ok_or(ConverterError::Arithmetic {
                invariant: "largest f32 panel bytes",
            })?;
        Ok(Self {
            tensor: entry.name.clone(),
            bytes_per_row,
            rows_per_panel,
            panel_count,
            largest_source_panel_bytes,
            largest_f32_panel_bytes,
        })
    }

    /// Produce a checked, lazy stream of row ranges in source order.
    pub fn row_panels(&self, total_rows: u64) -> Result<RowPanelIter, ConverterError> {
        if self.rows_per_panel == 0 {
            return Err(ConverterError::InvalidPanelPlan {
                tensor: self.tensor.clone(),
                detail: "zero rows per panel cannot produce a stream".to_owned(),
            });
        }
        let observed_panel_count = div_ceil(total_rows, self.rows_per_panel)?;
        if observed_panel_count != self.panel_count {
            return Err(ConverterError::InvalidPanelPlan {
                tensor: self.tensor.clone(),
                detail: format!(
                    "row count {total_rows} requires {observed_panel_count} panels, plan declares {}",
                    self.panel_count
                ),
            });
        }
        Ok(RowPanelIter {
            total_rows,
            next_row: 0,
            rows_per_panel: self.rows_per_panel,
        })
    }
}

/// Safely decode one little-endian BF16 source panel to the bounded f32 work
/// panel.  The caller owns the panel cap; this routine never widens a shard.
pub fn decode_bf16_panel(source: &[u8]) -> Result<Vec<f32>, ConverterError> {
    let chunks = source.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        return Err(ConverterError::MalformedBf16Panel {
            observed_bytes: source.len(),
        });
    }
    Ok(chunks
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
        .collect())
}

/// Read, decode, and consume a verified BF16 tensor one planned row panel at
/// a time. The source and f32 work buffers are local to each loop iteration,
/// so neither a whole tensor nor a whole shard can remain resident while the
/// next panel is read.
///
/// The caller owns quantization/packing and staging in `consume`; this bridge
/// owns the bounded source-range and BF16-to-f32 boundary only.
pub fn stream_verified_bf16_panels<C>(
    source: &SafetensorsRangeIndex,
    entry: &TensorCensusEntry,
    plan: &PanelPlan,
    consume: C,
) -> Result<(), ConverterError>
where
    C: FnMut(RowPanel, &[u8], &[f32]) -> Result<(), ConverterError>,
{
    stream_bf16_row_panels(
        entry,
        plan,
        |panel| {
            source
                .read_range(&entry.name, panel)
                .map_err(ConverterError::Safetensors)
        },
        consume,
    )
}

/// Observed accounting from a completed bounded BF16 panel traversal.
///
/// `f32_work_bytes` is the total work presented to the recipe sink over the
/// full run, not retained memory.  The concurrent allocation remains bounded
/// by [`PanelPlan::largest_f32_panel_bytes`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bf16PanelStreamReport {
    /// Number of source tensors whose complete planned panel sequence ran.
    pub tensors: u64,
    /// Number of source panels read and consumed.
    pub panels: u64,
    /// Total BF16 source bytes passed through the reader boundary.
    pub source_bytes: u64,
    /// Total f32 work bytes passed to the recipe sink.
    pub f32_work_bytes: u64,
}

/// Apply the frozen route's bounded Generic transform to one complete-row
/// source panel.  This does not retain a tensor or write an artifact; its
/// returned bytes are the quantization module's sole Generic-panel authority.
pub fn transform_routed_panel(
    entry: &TensorCensusEntry,
    route: &TensorRoute,
    panel: RowPanel,
    source_bytes: &[u8],
    f32_work: &[f32],
) -> Result<GenericPanelBytes, ConverterError> {
    if route != &remap_tensor_name(&entry.name)? {
        return Err(ConverterError::PipelinePlanAlignment {
            tensor: entry.name.clone(),
            detail: "route differs from canonical mapping".to_owned(),
        });
    }
    let RowPanel::Rows { row_count, .. } = panel else {
        return Err(ConverterError::PipelinePlanAlignment {
            tensor: entry.name.clone(),
            detail: "routed transform requires a complete-row panel, not a whole-tensor panel"
                .to_owned(),
        });
    };
    let columns = entry.shape[1..].iter().try_fold(1_u64, |product, dimension| {
        product.checked_mul(*dimension).ok_or(ConverterError::Arithmetic {
            invariant: "routed panel column product",
        })
    })?;
    let rows = usize::try_from(row_count).map_err(|_| ConverterError::Arithmetic {
        invariant: "routed panel rows to usize",
    })?;
    let columns = usize::try_from(columns).map_err(|_| ConverterError::Arithmetic {
        invariant: "routed panel columns to usize",
    })?;
    encode_generic_panel(route.stage, source_bytes, f32_work, rows, columns)
        .map_err(ConverterError::Quantize)
}

/// Traverse a fully prepared census in its canonical order, keeping only one
/// source panel and its decoded f32 work panel live at a time.
///
/// The reader is supplied by the checked safetensors range-index boundary and
/// the consumer is supplied by the recipe-specific quantization/staging
/// boundary.  This module owns the intervening shape, route, panel-length,
/// and BF16 decoding checks.  It intentionally has no output writer: a
/// canonical streaming envelope must first precompute its exact ranges.
pub fn stream_routed_bf16_panels<R, C>(
    census: &[TensorCensusEntry],
    routes: &[TensorRoute],
    panels: &[PanelPlan],
    mut read_panel: R,
    mut consume: C,
) -> Result<Bf16PanelStreamReport, ConverterError>
where
    R: FnMut(&TensorCensusEntry, RowPanel) -> Result<Vec<u8>, ConverterError>,
    C: FnMut(&TensorCensusEntry, &TensorRoute, RowPanel, &[u8], &[f32]) -> Result<(), ConverterError>,
{
    if census.len() != routes.len() || census.len() != panels.len() {
        return Err(ConverterError::PipelinePlanCount {
            census: census.len(),
            routes: routes.len(),
            panels: panels.len(),
        });
    }
    let mut report = Bf16PanelStreamReport {
        tensors: 0,
        panels: 0,
        source_bytes: 0,
        f32_work_bytes: 0,
    };
    for ((entry, route), plan) in census.iter().zip(routes).zip(panels) {
        let expected_route = remap_tensor_name(&entry.name)?;
        if route != &expected_route {
            return Err(ConverterError::PipelinePlanAlignment {
                tensor: entry.name.clone(),
                detail: format!(
                    "route={}/{} differs from canonical={}/{}",
                    route.internal_name,
                    route.stage.as_str(),
                    expected_route.internal_name,
                    expected_route.stage.as_str(),
                ),
            });
        }
        if plan.tensor != entry.name {
            return Err(ConverterError::PipelinePlanAlignment {
                tensor: entry.name.clone(),
                detail: format!("panel belongs to {}", plan.tensor),
            });
        }
        stream_bf16_row_panels(
            entry,
            plan,
            |panel| read_panel(entry, panel),
            |panel, source_bytes, f32_work| {
                consume(entry, route, panel, source_bytes, f32_work)?;
                let source_bytes_len = u64::try_from(source_bytes.len()).map_err(|_| {
                    ConverterError::Arithmetic {
                        invariant: "routed panel source bytes to u64",
                    }
                })?;
                let f32_work_bytes = u64::try_from(f32_work.len())
                    .map_err(|_| ConverterError::Arithmetic {
                        invariant: "routed panel f32 elements to u64",
                    })?
                    .checked_mul(4)
                    .ok_or(ConverterError::Arithmetic {
                        invariant: "routed panel f32 work bytes",
                    })?;
                report.panels = report
                    .panels
                    .checked_add(1)
                    .ok_or(ConverterError::Arithmetic {
                        invariant: "routed panel count",
                    })?;
                report.source_bytes = report
                    .source_bytes
                    .checked_add(source_bytes_len)
                    .ok_or(ConverterError::Arithmetic {
                        invariant: "routed source byte count",
                    })?;
                report.f32_work_bytes = report
                    .f32_work_bytes
                    .checked_add(f32_work_bytes)
                    .ok_or(ConverterError::Arithmetic {
                        invariant: "routed f32 work byte count",
                    })?;
                Ok(())
            },
        )?;
        report.tensors = report
            .tensors
            .checked_add(1)
            .ok_or(ConverterError::Arithmetic {
                invariant: "routed tensor count",
            })?;
    }
    Ok(report)
}

/// Feed one already-checked source range index through a prepared conversion
/// plan.  The range index is the only production reader admitted here; it
/// preserves its own source-digest and range-length checks while the routed
/// traversal preserves converter plan alignment and bounded BF16 decoding.
pub fn stream_prepared_bf16_panels<C>(
    source: &SafetensorsRangeIndex,
    prepared: &PreparedConversionInput,
    consume: C,
) -> Result<Bf16PanelStreamReport, ConverterError>
where
    C: FnMut(&TensorCensusEntry, &TensorRoute, RowPanel, &[u8], &[f32]) -> Result<(), ConverterError>,
{
    stream_routed_bf16_panels(
        &prepared.census,
        &prepared.routes,
        &prepared.panels,
        |entry, panel| {
            source
                .read_range(&entry.name, panel)
                .map_err(ConverterError::Safetensors)
        },
        consume,
    )
}

fn stream_bf16_row_panels<R, C>(
    entry: &TensorCensusEntry,
    plan: &PanelPlan,
    mut read_panel: R,
    mut consume: C,
) -> Result<(), ConverterError>
where
    R: FnMut(RowPanel) -> Result<Vec<u8>, ConverterError>,
    C: FnMut(RowPanel, &[u8], &[f32]) -> Result<(), ConverterError>,
{
    if entry.dtype != SafetensorDtype::Bf16 {
        return Err(ConverterError::UnexpectedDtype {
            tensor: entry.name.clone(),
            expected: SafetensorDtype::Bf16,
            actual: entry.dtype,
        });
    }
    if plan.tensor != entry.name {
        return Err(ConverterError::InvalidPanelPlan {
            tensor: entry.name.clone(),
            detail: format!(
                "plan belongs to {} rather than requested tensor {}",
                plan.tensor, entry.name
            ),
        });
    }
    let total_rows = *entry
        .shape
        .first()
        .ok_or_else(|| ConverterError::InvalidPanelPlan {
            tensor: entry.name.clone(),
            detail: "rank-zero tensor has no row-panel stream".to_owned(),
        })?;

    for panel in plan.row_panels(total_rows)? {
        let RowPanel::Rows {
            start_row,
            row_count,
        } = panel
        else {
            return Err(ConverterError::InvalidPanelPlan {
                tensor: entry.name.clone(),
                detail: "converter row plan produced a whole-tensor request".to_owned(),
            });
        };
        let expected_source_bytes =
            row_count
                .checked_mul(plan.bytes_per_row)
                .ok_or(ConverterError::Arithmetic {
                    invariant: "streamed BF16 panel byte length",
                })?;
        let source_bytes = read_panel(panel)?;
        let observed_source_bytes =
            u64::try_from(source_bytes.len()).map_err(|_| ConverterError::Arithmetic {
                invariant: "streamed BF16 panel length to u64",
            })?;
        if observed_source_bytes != expected_source_bytes {
            return Err(ConverterError::PanelReadLength {
                tensor: entry.name.clone(),
                start_row,
                row_count,
                expected: expected_source_bytes,
                actual: observed_source_bytes,
            });
        }
        if observed_source_bytes > plan.largest_source_panel_bytes {
            return Err(ConverterError::InvalidPanelPlan {
                tensor: entry.name.clone(),
                detail: format!(
                    "source panel {observed_source_bytes} bytes exceeds planned maximum {}",
                    plan.largest_source_panel_bytes
                ),
            });
        }
        let f32_work = decode_bf16_panel(&source_bytes)?;
        let f32_work_bytes = u64::try_from(f32_work.len())
            .map_err(|_| ConverterError::Arithmetic {
                invariant: "streamed f32 panel element count to u64",
            })?
            .checked_mul(4)
            .ok_or(ConverterError::Arithmetic {
                invariant: "streamed f32 panel byte length",
            })?;
        if f32_work_bytes > plan.largest_f32_panel_bytes {
            return Err(ConverterError::InvalidPanelPlan {
                tensor: entry.name.clone(),
                detail: format!(
                    "f32 work panel {f32_work_bytes} bytes exceeds planned maximum {}",
                    plan.largest_f32_panel_bytes
                ),
            });
        }
        consume(panel, &source_bytes, &f32_work)?;
    }
    Ok(())
}

/// A precomputed, non-overlapping output range in the staging artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputRange {
    /// Named logical destination.
    pub name: String,
    /// Absolute offset in the staged byte stream.
    pub offset: u64,
    /// Exact destination byte length.
    pub len: u64,
}

/// Ordered output-range directory used by the streaming sink.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputRangePlan {
    /// Ranges in canonical manifest/tensor order.
    pub ranges: Vec<OutputRange>,
    /// Exact final staged file byte length.
    pub file_len: u64,
}

impl OutputRangePlan {
    /// Construct adjacent, fully covered output ranges before creating a file.
    pub fn contiguous(named_lengths: &[(String, u64)]) -> Result<Self, ConverterError> {
        let mut ranges = Vec::with_capacity(named_lengths.len());
        let mut offset = 0_u64;
        let mut names = BTreeSet::new();
        for (name, len) in named_lengths {
            if !names.insert(name.clone()) {
                return Err(ConverterError::DuplicateOutputRange { name: name.clone() });
            }
            let end = offset.checked_add(*len).ok_or(ConverterError::Arithmetic {
                invariant: "output range end",
            })?;
            ranges.push(OutputRange {
                name: name.clone(),
                offset,
                len: *len,
            });
            offset = end;
        }
        Ok(Self {
            ranges,
            file_len: offset,
        })
    }

    /// Reject overlaps/gaps/out-of-order directory records from a receipt or
    /// a downstream envelope planner.
    pub fn validate(&self) -> Result<(), ConverterError> {
        let mut cursor = 0_u64;
        let mut names = BTreeSet::new();
        for range in &self.ranges {
            if !names.insert(range.name.clone()) {
                return Err(ConverterError::DuplicateOutputRange {
                    name: range.name.clone(),
                });
            }
            if range.offset != cursor {
                return Err(ConverterError::OutputRangeLayout {
                    name: range.name.clone(),
                    expected_offset: cursor,
                    actual_offset: range.offset,
                });
            }
            cursor = cursor
                .checked_add(range.len)
                .ok_or(ConverterError::Arithmetic {
                    invariant: "output range validation",
                })?;
        }
        if cursor != self.file_len {
            return Err(ConverterError::OutputFileLength {
                expected: cursor,
                actual: self.file_len,
            });
        }
        Ok(())
    }
}

/// The printed/enforced peak-RSS accounting formula.  This is deliberately a
/// byte formula, not an uncheckable promise about final file size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PeakRssFormula {
    /// Largest verified source panel allocation.
    pub largest_source_panel_bytes: u64,
    /// Largest f32 decoded work panel allocation.
    pub largest_f32_panel_bytes: u64,
    /// Recipe-owned bounded quantization/packing scratch.
    pub quant_packing_scratch_bytes: u64,
    /// Bounded staging write buffer.
    pub output_buffer_bytes: u64,
    /// Parser/index/header metadata retained while streaming.
    pub parser_metadata_bytes: u64,
    /// Explicit safety margin.
    pub margin_bytes: u64,
}

impl PeakRssFormula {
    /// Calculate the hard cap with checked addition.
    pub fn total_bytes(self) -> Result<u64, ConverterError> {
        [
            self.largest_source_panel_bytes,
            self.largest_f32_panel_bytes,
            self.quant_packing_scratch_bytes,
            self.output_buffer_bytes,
            self.parser_metadata_bytes,
            self.margin_bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |sum, component| {
            sum.checked_add(component)
                .ok_or(ConverterError::Arithmetic {
                    invariant: "peak RSS formula",
                })
        })
    }

    /// Refuse a measured/proxy peak beyond the printed formula.
    pub fn enforce(self, observed_peak_rss_bytes: u64) -> Result<(), ConverterError> {
        let cap = self.total_bytes()?;
        if observed_peak_rss_bytes > cap {
            return Err(ConverterError::PeakRssExceeded {
                observed: observed_peak_rss_bytes,
                cap,
            });
        }
        Ok(())
    }
}

impl fmt::Display for PeakRssFormula {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total = self.total_bytes().map_err(|_| fmt::Error)?;
        write!(
            formatter,
            "largest-source-panel={} + largest-f32-panel={} + quant-packing-scratch={} + output-buffer={} + parser-metadata={} + margin={} = {} bytes",
            self.largest_source_panel_bytes,
            self.largest_f32_panel_bytes,
            self.quant_packing_scratch_bytes,
            self.output_buffer_bytes,
            self.parser_metadata_bytes,
            self.margin_bytes,
            total,
        )
    }
}

/// Values printed before any conversion-side output allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConversionPreflight {
    /// Total verified ten-file source closure bytes.
    pub closure_bytes_to_read: u64,
    /// Exact output file length planned by the streaming envelope planner.
    pub staged_output_bytes: u64,
    /// Peak scratch/RSS formula.
    pub peak_rss: PeakRssFormula,
    /// Final disk requirement after atomic activation.
    pub final_disk_bytes: u64,
}

impl ConversionPreflight {
    /// Format the required human stderr preflight block.
    pub fn stderr_block(&self) -> Result<String, ConverterError> {
        Ok(format!(
            "CONVERT PREFLIGHT closure-bytes={} staged-output-bytes={} peak-rss={} final-disk-bytes={}",
            self.closure_bytes_to_read,
            self.staged_output_bytes,
            self.peak_rss,
            self.final_disk_bytes,
        ))
    }
}

/// Per-tensor progress record.  The display format is intentionally stable
/// for parsers and keeps stdout free for robot NDJSON/data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConversionProgress {
    /// One-based canonical tensor ordinal.
    pub ordinal: usize,
    /// Total tensor count.
    pub total: usize,
    /// Exact source tensor name.
    pub name: String,
    /// Source dimensions.
    pub shape: Vec<u64>,
    /// Number of bounded source panels.
    pub panels: u64,
    /// Cumulative staged bytes written.
    pub bytes_written: u64,
    /// Highest observed/proxy RSS during this run.
    pub running_peak_rss_bytes: u64,
    /// Remaining wall-clock estimate derived from logical payload progress.
    pub eta: Option<Duration>,
}

impl fmt::Display for ConversionProgress {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let eta = self
            .eta
            .map(|value| format!("{}s", value.as_secs()))
            .unwrap_or_else(|| "unknown".to_owned());
        write!(
            formatter,
            "[{}/{}] {} shape={:?} panels={} bytes-written={} running-peak-RSS={} ETA={eta}",
            self.ordinal,
            self.total,
            self.name,
            self.shape,
            self.panels,
            self.bytes_written,
            self.running_peak_rss_bytes,
        )
    }
}

/// Typed conversion failures.  Every source/census refusal retains observed
/// and expected values plus a next command where an operator can act.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConverterError {
    /// `fnlp convert` was invoked outside its one-recipe/Generic contract.
    InvalidConvertArgument {
        argument: &'static str,
        expected: String,
        actual: String,
    },
    /// Bounded canonical JSON parser rejection.
    ManifestJson(canonjson::CanonJsonError),
    /// Bounded canonical JSON parser rejection for a conversion receipt.
    ReceiptJson(canonjson::CanonJsonError),
    /// Manifest schema/type/key validation failure.
    ManifestSchema { path: String, detail: String },
    /// Manifest itself exceeded its bounded input cap.
    ManifestTooLarge {
        path: PathBuf,
        observed: u64,
        cap: u64,
    },
    /// Pinned manifest bytes differed before parsing.
    PinnedManifestDigest {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    /// Pinned source manifest fields did not match the frozen closure.
    PinnedManifestContract { detail: String },
    /// Generic filesystem failure.
    Io {
        path: PathBuf,
        operation: &'static str,
        detail: String,
    },
    /// Required member missing before any tensor parser can run.
    SourceMissing {
        path: PathBuf,
        expected: String,
        detail: String,
    },
    /// Required member was a symlink/device/directory instead of a file.
    SourceNotRegular { path: PathBuf, expected: String },
    /// Required member length mismatch.
    SourceLength {
        path: PathBuf,
        expected: u64,
        actual: u64,
        next_command: String,
    },
    /// Required member digest mismatch.
    SourceDigest {
        path: PathBuf,
        expected: String,
        actual: String,
        next_command: String,
    },
    /// Extra file always capable of changing source semantics.
    SemanticSourceExtra {
        path: PathBuf,
        name: String,
        next_command: String,
    },
    /// Extra file rejected only when strict directory mode is requested.
    StrictSourceExtra {
        path: PathBuf,
        name: String,
        next_command: String,
    },
    /// Range-index source/parser error after closure verification.
    Safetensors(SafetensorsError),
    /// OQ-1 architecture tripwire.
    Oq1Tripwire { tensor: String },
    /// Census differs in named categories.
    CensusMismatch { diff: CensusDiff },
    /// The checked full census did not add up to the frozen logical source
    /// payload.  This protects progress/ETA and preflight accounting against
    /// a second, drifting byte total.
    CensusPayloadBytes { expected: u64, actual: u64 },
    /// A caller attempted to traverse differently sized prepared inputs.
    PipelinePlanCount {
        census: usize,
        routes: usize,
        panels: usize,
    },
    /// A route or panel no longer corresponds to its prepared source tensor.
    PipelinePlanAlignment { tensor: String, detail: String },
    /// Portable panel quantization rejected the bounded f32 work slice.
    Quantize(QuantizeError),
    /// An unapproved bias tensor was discovered.
    BiasTensor { tensor: String },
    /// A source tensor has no complete mapping.
    UnknownTensorRoute { tensor: String },
    /// Dtype differs from the BF16 source contract.
    UnexpectedDtype {
        tensor: String,
        expected: SafetensorDtype,
        actual: SafetensorDtype,
    },
    /// Invalid panel-plan inputs.
    InvalidPanelPlan { tensor: String, detail: String },
    /// One source row alone exceeds the bounded panel allocation.
    RowExceedsPanelCap {
        tensor: String,
        row_bytes: u64,
        cap: u64,
    },
    /// BF16 byte count cannot be decoded exactly.
    MalformedBf16Panel { observed_bytes: usize },
    /// The verified range reader returned a panel outside the precomputed
    /// row-panel byte contract.
    PanelReadLength {
        tensor: String,
        start_row: u64,
        row_count: u64,
        expected: u64,
        actual: u64,
    },
    /// Checked byte/count arithmetic overflowed.
    Arithmetic { invariant: &'static str },
    /// Duplicate logical output section/range name.
    DuplicateOutputRange { name: String },
    /// Planned output range was not contiguous.
    OutputRangeLayout {
        name: String,
        expected_offset: u64,
        actual_offset: u64,
    },
    /// Staged file coverage disagreed with its precomputed file length.
    OutputFileLength { expected: u64, actual: u64 },
    /// A write did not exactly match its next precomputed range.
    StagingWriteRange {
        name: String,
        expected_offset: u64,
        observed_offset: u64,
        expected_len: u64,
        observed_len: u64,
    },
    /// Observed RSS/proxy was over the printed hard cap.
    PeakRssExceeded { observed: u64, cap: u64 },
    /// Atomic activation would clobber an existing artifact.
    ActivationTargetExists { path: PathBuf },
    /// The filesystem transaction substrate has not yet supplied a portable
    /// atomic no-clobber activation primitive.  A plain `rename` is forbidden
    /// because it can overwrite a concurrent destination on Unix.
    ActivationProtocolUnavailable { destination: PathBuf },
    /// Required machine receipt field was absent or malformed.
    ReceiptField { field: &'static str, detail: String },
    /// A conversion receipt did not use the one canonical JSON spelling.
    ReceiptNonCanonical,
    /// A canonical conversion-receipt JSON value did not match its typed schema.
    ReceiptParse { detail: String },
}

impl fmt::Display for ConverterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConvertArgument {
                argument,
                expected,
                actual,
            } => write!(
                formatter,
                "convert argument {argument} mismatch: expected={expected} observed={actual}; next=run fnlp convert --recipe {PINNED_CONVERSION_RECIPE} --arch generic"
            ),
            Self::ManifestJson(error) => write!(formatter, "source manifest JSON: {error}"),
            Self::ReceiptJson(error) => write!(formatter, "conversion receipt JSON: {error}"),
            Self::ManifestSchema { path, detail } => {
                write!(formatter, "source manifest {path}: {detail}")
            }
            Self::ManifestTooLarge {
                path,
                observed,
                cap,
            } => write!(
                formatter,
                "source manifest {} is {observed} bytes; cap is {cap}",
                path.display()
            ),
            Self::PinnedManifestDigest {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "pinned source manifest {} digest mismatch: expected={expected} observed={actual}; next=restore docs/truth-pack/nanbeige4.2-3b.source.json",
                path.display()
            ),
            Self::PinnedManifestContract { detail } => write!(
                formatter,
                "pinned source manifest contract mismatch: {detail}"
            ),
            Self::Io {
                path,
                operation,
                detail,
            } => write!(formatter, "{operation} {}: {detail}", path.display()),
            Self::SourceMissing {
                path,
                expected,
                detail,
            } => write!(
                formatter,
                "source member {expected} missing at {}: {detail}; next=restore the pinned source closure and rerun fnlp convert",
                path.display()
            ),
            Self::SourceNotRegular { path, expected } => write!(
                formatter,
                "source member {expected} is not a regular non-symlink file at {}; next=restore the pinned source closure and rerun fnlp convert",
                path.display()
            ),
            Self::SourceLength {
                path,
                expected,
                actual,
                next_command,
            } => write!(
                formatter,
                "source length mismatch {}: expected={expected} observed={actual}; next={next_command}",
                path.display()
            ),
            Self::SourceDigest {
                path,
                expected,
                actual,
                next_command,
            } => write!(
                formatter,
                "source digest mismatch {}: expected={expected} observed={actual}; next={next_command}",
                path.display()
            ),
            Self::SemanticSourceExtra {
                path,
                name,
                next_command,
            } => write!(
                formatter,
                "semantics-altering extra source file {name} at {}; next={next_command}",
                path.display()
            ),
            Self::StrictSourceExtra {
                path,
                name,
                next_command,
            } => write!(
                formatter,
                "strict source directory rejects extra {name} at {}; next={next_command}",
                path.display()
            ),
            Self::Safetensors(error) => write!(formatter, "checked safetensors source: {error}"),
            Self::Oq1Tripwire { tensor } => write!(
                formatter,
                "OQ-1 design-assumption abort: unexpected tensor family in {tensor}"
            ),
            Self::CensusMismatch { diff } => write!(
                formatter,
                "tensor census mismatch: MISSING={:?} SHAPE-MISMATCH={:?} EXTRA={:?}",
                diff.missing, diff.shape_mismatch, diff.extra
            ),
            Self::CensusPayloadBytes { expected, actual } => write!(
                formatter,
                "pinned logical source payload mismatch: expected={expected} observed={actual}"
            ),
            Self::PipelinePlanCount {
                census,
                routes,
                panels,
            } => write!(
                formatter,
                "prepared pipeline length mismatch: census={census} routes={routes} panels={panels}"
            ),
            Self::PipelinePlanAlignment { tensor, detail } => write!(
                formatter,
                "prepared pipeline alignment for {tensor}: {detail}"
            ),
            Self::Quantize(error) => write!(formatter, "portable panel quantization: {error}"),
            Self::BiasTensor { tensor } => write!(
                formatter,
                "bias tensor {tensor} is forbidden by the Nanbeige4.2-3B source contract"
            ),
            Self::UnknownTensorRoute { tensor } => write!(
                formatter,
                "no complete converter remap for source tensor {tensor}"
            ),
            Self::UnexpectedDtype {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "tensor {tensor} dtype mismatch: expected={} observed={}",
                expected.as_str(),
                actual.as_str()
            ),
            Self::InvalidPanelPlan { tensor, detail } => {
                write!(formatter, "invalid panel plan for {tensor}: {detail}")
            }
            Self::RowExceedsPanelCap {
                tensor,
                row_bytes,
                cap,
            } => write!(
                formatter,
                "tensor {tensor} row is {row_bytes} bytes, over panel cap {cap}"
            ),
            Self::MalformedBf16Panel { observed_bytes } => write!(
                formatter,
                "BF16 panel byte length must be even, observed {observed_bytes}"
            ),
            Self::PanelReadLength {
                tensor,
                start_row,
                row_count,
                expected,
                actual,
            } => write!(
                formatter,
                "streamed panel for {tensor} rows=[{start_row}, {}) has length {actual}, expected {expected}",
                start_row.saturating_add(*row_count)
            ),
            Self::Arithmetic { invariant } => {
                write!(formatter, "checked arithmetic overflow: {invariant}")
            }
            Self::DuplicateOutputRange { name } => {
                write!(formatter, "duplicate precomputed output range {name}")
            }
            Self::OutputRangeLayout {
                name,
                expected_offset,
                actual_offset,
            } => write!(
                formatter,
                "output range {name} offset mismatch: expected={expected_offset} observed={actual_offset}"
            ),
            Self::OutputFileLength { expected, actual } => write!(
                formatter,
                "staging coverage mismatch: expected={expected} observed={actual}"
            ),
            Self::StagingWriteRange {
                name,
                expected_offset,
                observed_offset,
                expected_len,
                observed_len,
            } => write!(
                formatter,
                "staging write {name} mismatch: expected offset/len={expected_offset}/{expected_len}, observed={observed_offset}/{observed_len}"
            ),
            Self::PeakRssExceeded { observed, cap } => write!(
                formatter,
                "peak RSS gate exceeded: observed={observed} formula-cap={cap}"
            ),
            Self::ActivationTargetExists { path } => write!(
                formatter,
                "refusing to overwrite existing artifact {}; quarantine it explicitly, then rerun fnlp convert",
                path.display()
            ),
            Self::ActivationProtocolUnavailable { destination } => write!(
                formatter,
                "activation of {} is blocked: no portable atomic no-clobber filesystem transaction is installed; next=complete the fs_tx activation contract",
                destination.display()
            ),
            Self::ReceiptField { field, detail } => {
                write!(formatter, "conversion receipt field {field}: {detail}")
            }
            Self::ReceiptNonCanonical => {
                formatter.write_str("conversion receipt bytes are not canonical JSON")
            }
            Self::ReceiptParse { detail } => {
                write!(formatter, "conversion receipt schema: {detail}")
            }
        }
    }
}

impl std::error::Error for ConverterError {}

fn parse_source_file(value: &Value, index: usize) -> Result<SourceFileSpec, ConverterError> {
    let path = format!("$/files/{index}");
    let object = required_object(value, &path, &SOURCE_FILE_KEYS)?;
    Ok(SourceFileSpec {
        name: required_string(object, "name", &path)?,
        bytes: required_u64(object, "bytes", &path)?,
        sha256: required_string(object, "sha256", &path)?,
    })
}

fn required_object<'a>(
    value: &'a Value,
    path: &str,
    allowed: &[&str],
) -> Result<&'a Map<String, Value>, ConverterError> {
    let object = value
        .as_object()
        .ok_or_else(|| ConverterError::ManifestSchema {
            path: path.to_owned(),
            detail: "expected object".to_owned(),
        })?;
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(ConverterError::ManifestSchema {
                path: path.to_owned(),
                detail: format!("unknown field {key:?}"),
            });
        }
    }
    for key in allowed {
        if !object.contains_key(*key) {
            return Err(ConverterError::ManifestSchema {
                path: path.to_owned(),
                detail: format!("missing required field {key:?}"),
            });
        }
    }
    Ok(object)
}

fn required_array<'a>(value: &'a Value, path: &str) -> Result<&'a Vec<Value>, ConverterError> {
    value
        .as_array()
        .ok_or_else(|| ConverterError::ManifestSchema {
            path: path.to_owned(),
            detail: "expected array".to_owned(),
        })
}

fn required_string(
    object: &Map<String, Value>,
    key: &str,
    parent: &str,
) -> Result<String, ConverterError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| ConverterError::ManifestSchema {
            path: format!("{parent}/{key}"),
            detail: "expected string".to_owned(),
        })
}

fn required_u64(
    object: &Map<String, Value>,
    key: &str,
    parent: &str,
) -> Result<u64, ConverterError> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| ConverterError::ManifestSchema {
            path: format!("{parent}/{key}"),
            detail: "expected non-negative integer".to_owned(),
        })
}

fn source_digest(file: &SourceFileSpec) -> Result<SourceDigest, ConverterError> {
    SourceDigest::new(&file.name, file.bytes, &file.sha256).map_err(ConverterError::Safetensors)
}

fn expectation(name: &str, shape: &[u64]) -> TensorExpectation {
    let elements = shape.iter().copied().product::<u64>();
    TensorExpectation {
        name: name.to_owned(),
        dtype: SafetensorDtype::Bf16,
        shape: shape.to_vec(),
        len: elements * 2,
    }
}

fn is_semantic_conflict(name: &str) -> bool {
    name.ends_with(".safetensors")
        || name.ends_with(".index.json")
        || matches!(
            name,
            "config.json"
                | "generation_config.json"
                | "tokenizer.json"
                | "tokenizer.model"
                | "tokenizer_config.json"
                | "special_tokens_map.json"
        )
}

fn is_oq1_tripwire(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    ["mhc", "ngram", "depth", "loopsplit"]
        .iter()
        .any(|needle| lower.contains(needle))
}

fn is_safe_basename(name: &str) -> bool {
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

fn is_lower_git_commit(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn stream_sha256(file: &mut File, path: &Path) -> Result<String, ConverterError> {
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| ConverterError::Io {
            path: path.to_path_buf(),
            operation: "stream-hash source member",
            detail: error.to_string(),
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn source_root_digest(files: &[VerifiedSourceFile]) -> Result<String, ConverterError> {
    let mut hasher = Sha256::new();
    hasher.update(b"fnlpq-source-root-v1\0");
    let count = u64::try_from(files.len()).map_err(|_| ConverterError::Arithmetic {
        invariant: "source-root file count",
    })?;
    hasher.update(count.to_le_bytes());
    for file in files {
        let name_len = u64::try_from(file.name.len()).map_err(|_| ConverterError::Arithmetic {
            invariant: "source-root file name length",
        })?;
        hasher.update(name_len.to_le_bytes());
        hasher.update(file.name.as_bytes());
        hasher.update(file.bytes.to_le_bytes());
        hasher.update(file.sha256.as_bytes());
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn census_digest(census: &[TensorCensusEntry]) -> Result<String, ConverterError> {
    let mut hasher = Sha256::new();
    hasher.update(b"fnlpq-census-v1\0");
    let count = u64::try_from(census.len()).map_err(|_| ConverterError::Arithmetic {
        invariant: "census tensor count",
    })?;
    hasher.update(count.to_le_bytes());
    for tensor in census {
        let name_len =
            u64::try_from(tensor.name.len()).map_err(|_| ConverterError::Arithmetic {
                invariant: "census tensor name length",
            })?;
        hasher.update(name_len.to_le_bytes());
        hasher.update(tensor.name.as_bytes());
        hasher.update(tensor.dtype.as_str().as_bytes());
        let rank = u64::try_from(tensor.shape.len()).map_err(|_| ConverterError::Arithmetic {
            invariant: "census tensor rank",
        })?;
        hasher.update(rank.to_le_bytes());
        for dimension in &tensor.shape {
            hasher.update(dimension.to_le_bytes());
        }
        hasher.update(tensor.len.to_le_bytes());
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn div_ceil(numerator: u64, denominator: u64) -> Result<u64, ConverterError> {
    if denominator == 0 {
        return Err(ConverterError::Arithmetic {
            invariant: "division by zero in panel planner",
        });
    }
    let quotient = numerator / denominator;
    quotient
        .checked_add(u64::from(numerator % denominator != 0))
        .ok_or(ConverterError::Arithmetic {
            invariant: "panel division ceiling",
        })
}

#[cfg(test)]
mod tests {
    use super::{
        ConversionProgress, ConverterError, OutputRangePlan, PanelPlan, PeakRssFormula,
        StorageStage, TensorCensusEntry, decode_bf16_panel, expected_nanbeige42_census,
        remap_tensor_name, stream_bf16_row_panels, stream_routed_bf16_panels,
    };
    use crate::artifact::safetensors::{RowPanel, SafetensorDtype};

    #[test]
    fn census_has_the_full_pinned_mapping_surface() {
        let census = expected_nanbeige42_census();
        assert_eq!(census.len(), 201);
        for tensor in &census {
            assert!(remap_tensor_name(&tensor.name).is_ok(), "{}", tensor.name);
        }
    }

    #[test]
    fn remap_preserves_stage_boundaries() {
        assert_eq!(
            remap_tensor_name("model.layers.7.mlp.gate_proj.weight")
                .expect("route")
                .stage,
            StorageStage::Int8Stage2A
        );
        assert_eq!(
            remap_tensor_name("model.layers.7.self_attn.q_proj.weight")
                .expect("route")
                .stage,
            StorageStage::Int8Stage2B
        );
        assert_eq!(
            remap_tensor_name("model.embed_tokens.weight")
                .expect("route")
                .stage,
            StorageStage::Bf16Verbatim
        );
    }

    #[test]
    fn remap_uses_fnlpq_authority_safe_layer_names() {
        let route =
            remap_tensor_name("model.layers.7.self_attn.k_proj.weight").expect("known route");
        assert_eq!(route.internal_name, "layer.7.attn.k");
        crate::artifact::format::validate_authority_identifier("tensor.name", &route.internal_name)
            .expect("converter route must satisfy the frozen fnlpq authority grammar");
    }

    #[test]
    fn panel_plan_never_exceeds_its_source_cap() {
        let entry = TensorCensusEntry {
            name: "small".to_owned(),
            dtype: SafetensorDtype::Bf16,
            shape: vec![17, 8],
            len: 17 * 8 * 2,
        };
        let plan = PanelPlan::for_tensor(&entry, 64).expect("bounded plan");
        assert_eq!(plan.rows_per_panel, 4);
        assert_eq!(plan.panel_count, 5);
        assert!(plan.largest_source_panel_bytes <= 64);
        assert_eq!(plan.row_panels(17).expect("row panels").count(), 5);
    }

    #[test]
    fn row_panel_stream_keeps_strict_cap_bookkeeping_lazy() {
        let entry = TensorCensusEntry {
            name: "synthetic.weight".to_owned(),
            dtype: SafetensorDtype::Bf16,
            shape: vec![10_000, 1],
            len: 20_000,
        };
        let plan = PanelPlan::for_tensor(&entry, 2).expect("one BF16 value per panel");
        let mut panels = plan.row_panels(10_000).expect("lazy row stream");

        assert_eq!(panels.size_hint(), (10_000, Some(10_000)));
        assert_eq!(
            panels.next(),
            Some(RowPanel::Rows {
                start_row: 0,
                row_count: 1,
            })
        );
        assert_eq!(panels.count(), 9_999);
    }

    #[test]
    fn row_panel_stream_refuses_a_row_count_outside_its_plan() {
        let entry = TensorCensusEntry {
            name: "synthetic.weight".to_owned(),
            dtype: SafetensorDtype::Bf16,
            shape: vec![17, 1],
            len: 34,
        };
        let plan = PanelPlan::for_tensor(&entry, 8).expect("four rows per panel");

        let error = plan
            .row_panels(16)
            .expect_err("different row count must not reuse a panel plan");
        assert!(error.to_string().contains("requires 4 panels"));
    }

    #[test]
    fn streamed_bf16_panels_read_decode_and_release_one_panel_at_a_time() {
        let entry = TensorCensusEntry {
            name: "synthetic.weight".to_owned(),
            dtype: SafetensorDtype::Bf16,
            shape: vec![5, 2],
            len: 20,
        };
        let plan = PanelPlan::for_tensor(&entry, 8).expect("two rows per bounded panel");
        let mut observed = Vec::new();

        stream_bf16_row_panels(
            &entry,
            &plan,
            |panel| {
                let RowPanel::Rows { row_count, .. } = panel else {
                    panic!("panel plan must yield row panels");
                };
                let scalar_count = usize::try_from(row_count * 2).expect("tiny fixture");
                Ok(vec![0x80, 0x3f].repeat(scalar_count))
            },
            |panel, source_bytes, f32_work| {
                let RowPanel::Rows {
                    start_row,
                    row_count,
                } = panel
                else {
                    panic!("panel plan must yield row panels");
                };
                assert_eq!(source_bytes.len() / 2, f32_work.len());
                assert!(source_bytes.len() <= 8);
                assert!(f32_work.len() * 4 <= 16);
                observed.push((start_row, row_count, source_bytes.len(), f32_work.len()));
                Ok(())
            },
        )
        .expect("every source range is decoded and consumed before the next panel");

        assert_eq!(observed, vec![(0, 2, 8, 4), (2, 2, 8, 4), (4, 1, 4, 2)]);
    }

    #[test]
    fn streamed_bf16_panels_refuse_reader_lengths_outside_the_plan() {
        let entry = TensorCensusEntry {
            name: "synthetic.weight".to_owned(),
            dtype: SafetensorDtype::Bf16,
            shape: vec![1, 2],
            len: 4,
        };
        let plan = PanelPlan::for_tensor(&entry, 4).expect("one bounded row");

        let error = stream_bf16_row_panels(&entry, &plan, |_| Ok(vec![0; 3]), |_, _, _| Ok(()))
            .expect_err("a short range must not enter the BF16 work stage");
        assert!(matches!(
            error,
            ConverterError::PanelReadLength {
                tensor,
                start_row: 0,
                row_count: 1,
                expected: 4,
                actual: 3,
            } if tensor == "synthetic.weight"
        ));
    }

    #[test]
    fn routed_panel_stream_keeps_canonical_tensor_and_panel_order() {
        let census = vec![
            TensorCensusEntry {
                name: "model.embed_tokens.weight".to_owned(),
                dtype: SafetensorDtype::Bf16,
                shape: vec![3, 2],
                len: 12,
            },
            TensorCensusEntry {
                name: "model.norm.weight".to_owned(),
                dtype: SafetensorDtype::Bf16,
                shape: vec![2, 2],
                len: 8,
            },
        ];
        let routes = census
            .iter()
            .map(|entry| remap_tensor_name(&entry.name).expect("known route"))
            .collect::<Vec<_>>();
        let panels = census
            .iter()
            .map(|entry| PanelPlan::for_tensor(entry, 4).expect("one row per panel"))
            .collect::<Vec<_>>();
        let mut observed = Vec::new();

        let report = stream_routed_bf16_panels(
            &census,
            &routes,
            &panels,
            |entry, panel| {
                let RowPanel::Rows { row_count, .. } = panel else {
                    panic!("prepared plan must yield row panels");
                };
                let scalar_count = usize::try_from(row_count * entry.shape[1]).expect("tiny");
                Ok(vec![0x80, 0x3f].repeat(scalar_count))
            },
            |entry, route, panel, source_bytes, f32_work| {
                let RowPanel::Rows { start_row, .. } = panel else {
                    panic!("prepared plan must yield row panels");
                };
                observed.push((
                    entry.name.clone(),
                    route.internal_name.clone(),
                    start_row,
                    source_bytes.len(),
                    f32_work.len(),
                ));
                Ok(())
            },
        )
        .expect("the prepared route stream is serial and bounded");

        assert_eq!(report.tensors, 2);
        assert_eq!(report.panels, 5);
        assert_eq!(report.source_bytes, 20);
        assert_eq!(report.f32_work_bytes, 40);
        assert_eq!(
            observed,
            vec![
                ("model.embed_tokens.weight".to_owned(), "embed".to_owned(), 0, 4, 2),
                ("model.embed_tokens.weight".to_owned(), "embed".to_owned(), 1, 4, 2),
                ("model.embed_tokens.weight".to_owned(), "embed".to_owned(), 2, 4, 2),
                ("model.norm.weight".to_owned(), "final_norm".to_owned(), 0, 4, 2),
                ("model.norm.weight".to_owned(), "final_norm".to_owned(), 1, 4, 2),
            ]
        );
    }

    #[test]
    fn routed_panel_stream_refuses_a_stale_route_before_reading() {
        let census = vec![TensorCensusEntry {
            name: "model.embed_tokens.weight".to_owned(),
            dtype: SafetensorDtype::Bf16,
            shape: vec![1, 1],
            len: 2,
        }];
        let routes = vec![remap_tensor_name("model.norm.weight").expect("known route")];
        let panels = vec![PanelPlan::for_tensor(&census[0], 2).expect("one row")];

        assert!(matches!(
            stream_routed_bf16_panels(
                &census,
                &routes,
                &panels,
                |_, _| panic!("stale route must reject before source reads"),
                |_, _, _, _, _| panic!("stale route must reject before consumption"),
            ),
            Err(ConverterError::PipelinePlanAlignment { tensor, .. })
                if tensor == "model.embed_tokens.weight"
        ));
    }

    #[test]
    fn panel_division_ceiling_accepts_maximum_valid_dimensions() {
        assert_eq!(
            super::div_ceil(u64::MAX, 1).expect("one-row panels"),
            u64::MAX
        );
        assert_eq!(
            super::div_ceil(u64::MAX, 2).expect("two-row panels"),
            (u64::MAX / 2) + 1
        );
        assert_eq!(
            super::div_ceil(u64::MAX, u64::MAX).expect("one maximum-size panel"),
            1
        );
    }

    #[test]
    fn bf16_decode_is_exact_at_the_bit_boundary() {
        let decoded = decode_bf16_panel(&[0x80, 0x3f, 0x00, 0xc0]).expect("BF16");
        assert_eq!(decoded, vec![1.0, -2.0]);
    }

    #[test]
    fn output_ranges_are_adjacent_and_complete() {
        let plan = OutputRangePlan::contiguous(&[("a".to_owned(), 3), ("b".to_owned(), 5)])
            .expect("ranges");
        assert_eq!(plan.file_len, 8);
        plan.validate().expect("contiguous range plan");
    }

    #[test]
    fn peak_formula_is_a_hard_checked_cap() {
        let formula = PeakRssFormula {
            largest_source_panel_bytes: 4,
            largest_f32_panel_bytes: 8,
            quant_packing_scratch_bytes: 2,
            output_buffer_bytes: 1,
            parser_metadata_bytes: 3,
            margin_bytes: 2,
        };
        assert_eq!(formula.total_bytes().expect("total"), 20);
        assert!(formula.enforce(20).is_ok());
        assert!(formula.enforce(21).is_err());
    }

    #[test]
    fn progress_line_has_the_machine_contract_fields() {
        let line = ConversionProgress {
            ordinal: 1,
            total: 201,
            name: "model.embed_tokens.weight".to_owned(),
            shape: vec![166_144, 3_072],
            panels: 16,
            bytes_written: 64,
            running_peak_rss_bytes: 128,
            eta: None,
        }
        .to_string();
        assert!(line.contains("[1/201]"));
        assert!(line.contains("bytes-written=64"));
        assert!(line.contains("running-peak-RSS=128"));
    }
}
