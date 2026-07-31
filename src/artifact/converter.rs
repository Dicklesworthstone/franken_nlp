//! Bounded, fail-closed source preparation for `fnlp convert`.
//!
//! This module owns the conversion-side trust boundary: all ten source files
//! are hashed in manifest order before a safetensors header is parsed, the
//! 201-tensor inventory is checked before routing, and every panel/output
//! range is sized with checked arithmetic.  It deliberately does *not* use
//! the in-memory `.fnlpq` serializer as a production writer: that serializer
//! remains a small-fixture oracle until the envelope grows its streaming sink.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::artifact::safetensors::{
    diff_census_entries, CensusDiff, RowPanel, SafetensorDtype, SafetensorsError,
    SafetensorsRangeIndex, SourceClosure, SourceDigest, TensorCensusEntry, TensorExpectation,
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
    let base = format!("layer[{layer}]");
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedConversionInput {
    /// Ten-file source closure evidence in manifest order.
    pub source: VerifiedSourceClosure,
    /// Checked source facts in canonical tensor-name order.
    pub census: Vec<TensorCensusEntry>,
    /// One complete route for every census entry.
    pub routes: Vec<TensorRoute>,
    /// One bounded row-panel plan for every source tensor.
    pub panels: Vec<PanelPlan>,
    /// Domain-framed exact source census identity.
    pub census_sha256: String,
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
    let census_sha256 = census_digest(&census)?;
    Ok(PreparedConversionInput {
        source,
        census,
        routes,
        panels,
        census_sha256,
    })
}

impl PreparedConversionInput {
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
        let final_disk_bytes = staged_output_bytes
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
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConversionReceipt {
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
    /// Printed formula cap.
    pub peak_rss_cap_bytes: u64,
    /// Exact final disk requirement observed by preflight.
    pub final_disk_bytes: u64,
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
        for (field, value) in [
            ("converter_commit", self.converter_commit.as_str()),
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
        Ok(())
    }

    /// Canonical NDJSON-ready receipt bytes, after completeness validation.
    pub fn canonical_json(&self) -> Result<String, ConverterError> {
        self.validate()?;
        canonjson::canonical_string(self).map_err(ConverterError::ManifestJson)
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

    /// Produce checked row ranges in source order.
    pub fn row_panels(&self, total_rows: u64) -> Result<Vec<RowPanel>, ConverterError> {
        let mut output = Vec::new();
        let mut start_row = 0_u64;
        while start_row < total_rows {
            let row_count = self.rows_per_panel.min(total_rows - start_row);
            output.push(RowPanel::Rows {
                start_row,
                row_count,
            });
            start_row = start_row
                .checked_add(row_count)
                .ok_or(ConverterError::Arithmetic {
                    invariant: "panel row cursor",
                })?;
        }
        Ok(output)
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
        for range in &self.ranges {
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

/// A serial staging sink.  It has no overwrite path: callers must precompute
/// a contiguous directory, and every byte is written exactly once in order.
pub struct CreateNewStagingFile {
    path: PathBuf,
    file: File,
    expected_file_len: u64,
    next_offset: u64,
}

impl CreateNewStagingFile {
    /// Create an exact-length staging file without replacing any existing
    /// file.  A failed/abandoned stage is deliberately retained for audit; no
    /// drop implementation deletes it.
    pub fn create_new(
        path: impl Into<PathBuf>,
        expected_file_len: u64,
    ) -> Result<Self, ConverterError> {
        let path = path.into();
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| ConverterError::Io {
                path: path.clone(),
                operation: "create_new conversion staging file",
                detail: error.to_string(),
            })?;
        file.set_len(expected_file_len)
            .map_err(|error| ConverterError::Io {
                path: path.clone(),
                operation: "set conversion staging length",
                detail: error.to_string(),
            })?;
        Ok(Self {
            path,
            file,
            expected_file_len,
            next_offset: 0,
        })
    }

    /// Write the next precomputed output range.  Seeking to arbitrary ranges
    /// is deliberately not exposed, preventing overlaps or holes.
    pub fn write_next(&mut self, range: &OutputRange, bytes: &[u8]) -> Result<(), ConverterError> {
        let observed_len = u64::try_from(bytes.len()).map_err(|_| ConverterError::Arithmetic {
            invariant: "staging byte slice length",
        })?;
        if range.offset != self.next_offset || observed_len != range.len {
            return Err(ConverterError::StagingWriteRange {
                name: range.name.clone(),
                expected_offset: self.next_offset,
                observed_offset: range.offset,
                expected_len: range.len,
                observed_len,
            });
        }
        self.file
            .seek(SeekFrom::Start(range.offset))
            .and_then(|_| self.file.write_all(bytes))
            .map_err(|error| ConverterError::Io {
                path: self.path.clone(),
                operation: "write conversion staging range",
                detail: error.to_string(),
            })?;
        self.next_offset =
            self.next_offset
                .checked_add(range.len)
                .ok_or(ConverterError::Arithmetic {
                    invariant: "staging next offset",
                })?;
        Ok(())
    }

    /// Sync and prove complete coverage before a separate activation layer
    /// performs its no-clobber atomic rename.
    pub fn finish(mut self) -> Result<PathBuf, ConverterError> {
        if self.next_offset != self.expected_file_len {
            return Err(ConverterError::OutputFileLength {
                expected: self.expected_file_len,
                actual: self.next_offset,
            });
        }
        self.file
            .flush()
            .and_then(|_| self.file.sync_all())
            .map_err(|error| ConverterError::Io {
                path: self.path.clone(),
                operation: "sync conversion staging file",
                detail: error.to_string(),
            })?;
        Ok(self.path)
    }
}

/// Atomically publish a fully verified stage only when it cannot replace an
/// installed artifact.  Existing/invalid targets are left untouched for the
/// caller's explicit quarantine workflow.
pub fn activate_no_clobber(
    stage: impl AsRef<Path>,
    destination: impl AsRef<Path>,
) -> Result<(), ConverterError> {
    let stage = stage.as_ref();
    let destination = destination.as_ref();
    if fs::symlink_metadata(destination).is_ok() {
        return Err(ConverterError::ActivationTargetExists {
            path: destination.to_path_buf(),
        });
    }
    fs::rename(stage, destination).map_err(|error| ConverterError::Io {
        path: destination.to_path_buf(),
        operation: "atomically activate verified conversion stage",
        detail: error.to_string(),
    })
}

/// Typed conversion failures.  Every source/census refusal retains observed
/// and expected values plus a next command where an operator can act.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConverterError {
    /// Bounded canonical JSON parser rejection.
    ManifestJson(canonjson::CanonJsonError),
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
    /// Required machine receipt field was absent or malformed.
    ReceiptField { field: &'static str, detail: String },
}

impl fmt::Display for ConverterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ManifestJson(error) => write!(formatter, "source manifest JSON: {error}"),
            Self::ManifestSchema { path, detail } => write!(formatter, "source manifest {path}: {detail}"),
            Self::ManifestTooLarge { path, observed, cap } => write!(formatter, "source manifest {} is {observed} bytes; cap is {cap}", path.display()),
            Self::PinnedManifestDigest { path, expected, actual } => write!(formatter, "pinned source manifest {} digest mismatch: expected={expected} observed={actual}; next=restore docs/truth-pack/nanbeige4.2-3b.source.json", path.display()),
            Self::PinnedManifestContract { detail } => write!(formatter, "pinned source manifest contract mismatch: {detail}"),
            Self::Io { path, operation, detail } => write!(formatter, "{operation} {}: {detail}", path.display()),
            Self::SourceMissing { path, expected, detail } => write!(formatter, "source member {expected} missing at {}: {detail}; next=restore the pinned source closure and rerun fnlp convert", path.display()),
            Self::SourceNotRegular { path, expected } => write!(formatter, "source member {expected} is not a regular non-symlink file at {}; next=restore the pinned source closure and rerun fnlp convert", path.display()),
            Self::SourceLength { path, expected, actual, next_command } => write!(formatter, "source length mismatch {}: expected={expected} observed={actual}; next={next_command}", path.display()),
            Self::SourceDigest { path, expected, actual, next_command } => write!(formatter, "source digest mismatch {}: expected={expected} observed={actual}; next={next_command}", path.display()),
            Self::SemanticSourceExtra { path, name, next_command } => write!(formatter, "semantics-altering extra source file {name} at {}; next={next_command}", path.display()),
            Self::StrictSourceExtra { path, name, next_command } => write!(formatter, "strict source directory rejects extra {name} at {}; next={next_command}", path.display()),
            Self::Safetensors(error) => write!(formatter, "checked safetensors source: {error}"),
            Self::Oq1Tripwire { tensor } => write!(formatter, "OQ-1 design-assumption abort: unexpected tensor family in {tensor}"),
            Self::CensusMismatch { diff } => write!(formatter, "tensor census mismatch: MISSING={:?} SHAPE-MISMATCH={:?} EXTRA={:?}", diff.missing, diff.shape_mismatch, diff.extra),
            Self::BiasTensor { tensor } => write!(formatter, "bias tensor {tensor} is forbidden by the Nanbeige4.2-3B source contract"),
            Self::UnknownTensorRoute { tensor } => write!(formatter, "no complete converter remap for source tensor {tensor}"),
            Self::UnexpectedDtype { tensor, expected, actual } => write!(formatter, "tensor {tensor} dtype mismatch: expected={} observed={}", expected.as_str(), actual.as_str()),
            Self::InvalidPanelPlan { tensor, detail } => write!(formatter, "invalid panel plan for {tensor}: {detail}"),
            Self::RowExceedsPanelCap { tensor, row_bytes, cap } => write!(formatter, "tensor {tensor} row is {row_bytes} bytes, over panel cap {cap}"),
            Self::MalformedBf16Panel { observed_bytes } => write!(formatter, "BF16 panel byte length must be even, observed {observed_bytes}"),
            Self::Arithmetic { invariant } => write!(formatter, "checked arithmetic overflow: {invariant}"),
            Self::DuplicateOutputRange { name } => write!(formatter, "duplicate precomputed output range {name}"),
            Self::OutputRangeLayout { name, expected_offset, actual_offset } => write!(formatter, "output range {name} offset mismatch: expected={expected_offset} observed={actual_offset}"),
            Self::OutputFileLength { expected, actual } => write!(formatter, "staging coverage mismatch: expected={expected} observed={actual}"),
            Self::StagingWriteRange { name, expected_offset, observed_offset, expected_len, observed_len } => write!(formatter, "staging write {name} mismatch: expected offset/len={expected_offset}/{expected_len}, observed={observed_offset}/{observed_len}"),
            Self::PeakRssExceeded { observed, cap } => write!(formatter, "peak RSS gate exceeded: observed={observed} formula-cap={cap}"),
            Self::ActivationTargetExists { path } => write!(formatter, "refusing to overwrite existing artifact {}; quarantine it explicitly, then rerun fnlp convert", path.display()),
            Self::ReceiptField { field, detail } => {
                write!(formatter, "conversion receipt field {field}: {detail}")
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
        let name_len = u64::try_from(tensor.name.len()).map_err(|_| ConverterError::Arithmetic {
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
    numerator
        .checked_add(denominator - 1)
        .and_then(|value| value.checked_div(denominator))
        .ok_or(ConverterError::Arithmetic {
            invariant: "panel division ceiling",
        })
}

#[cfg(test)]
mod tests {
    use super::{
        decode_bf16_panel, expected_nanbeige42_census, remap_tensor_name, ConversionProgress,
        OutputRangePlan, PanelPlan, PeakRssFormula, StorageStage, TensorCensusEntry,
    };
    use crate::artifact::safetensors::SafetensorDtype;

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
        assert_eq!(plan.row_panels(17).expect("row panels").len(), 5);
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
