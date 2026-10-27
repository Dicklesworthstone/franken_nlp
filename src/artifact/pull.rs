//! Verified local reconstruction for immutable release parts.
//!
//! Network acquisition and model-root activation remain separate authority
//! boundaries. This module only consumes an explicit local release manifest
//! whose domain-separated digest was obtained through an independent trusted
//! channel, verifies every local part, reconstructs one staged Generic fnlpq,
//! verifies the complete artifact and its internal identities, and then
//! publishes the final pathname with a no-replace hard link.
//!
//! A failed reconstruction deliberately leaves the caller-named staging file
//! in place for forensic inspection. This module never deletes or overwrites
//! caller files and never treats a checksum as publisher authentication.

use std::{
    error::Error,
    fmt,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

use crate::artifact::{
    manifest::{
        release_manifest_sha256, ManifestError, ReleaseManifest,
        NANBEIGE42_INT8_V1_COMPATIBILITY, MAX_RELEASE_MANIFEST_BYTES,
    },
    reader::{FnlpqRangeReader, FnlpqReadError},
};

const COPY_CHUNK_BYTES: usize = 1024 * 1024;

/// Explicit offline reconstruction request.
///
/// `expected_release_manifest_sha256` is the domain-separated manifest digest
/// from an independently trusted channel. The manifest file and release parts
/// themselves are therefore not allowed to choose their own authority root.
#[derive(Clone, Debug)]
pub struct LocalReconstructRequest {
    pub manifest: PathBuf,
    pub expected_release_manifest_sha256: String,
    pub parts_dir: PathBuf,
    /// Caller-owned create-new staging pathname. It remains after success.
    pub staging_output: PathBuf,
    /// Final create-new pathname. Publication uses `hard_link`, never replace.
    pub output: PathBuf,
}

/// Successful reconstruction evidence.
///
/// `directory_durability_confirmed` is intentionally false: safe portable
/// std does not provide the owner/no-follow directory fsync surface required by
/// the product's activation contract. This report is therefore not an
/// activation receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalReconstructReport {
    pub release_manifest_sha256: String,
    pub artifact_sha256: String,
    pub artifact_bytes: u64,
    pub part_count: usize,
    pub logical_model_sha256: String,
    pub output: PathBuf,
    pub staging_output: PathBuf,
    pub directory_durability_confirmed: bool,
}

/// Typed refusal for local reconstruction. Default formatting avoids embedding
/// arbitrary file contents or network responses.
#[derive(Debug)]
pub enum PullError {
    InvalidExpectedManifestDigest,
    ManifestIo { operation: &'static str, detail: String },
    Manifest(ManifestError),
    ManifestDigestMismatch { expected: String, observed: String },
    InactiveManifest { state: String },
    InvalidPath(&'static str),
    PartIo { part: usize, operation: &'static str, detail: String },
    PartLength { part: usize, expected: u64, observed: u64 },
    PartDigest { part: usize, expected: String, observed: String },
    ArtifactLength { expected: u64, observed: u64 },
    ArtifactDigest { expected: String, observed: String },
    Container(FnlpqReadError),
    ContainerIdentity(&'static str),
    Publish { operation: &'static str, detail: String },
}

impl fmt::Display for PullError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidExpectedManifestDigest => {
                f.write_str("local pull refused: expected manifest digest must be lowercase SHA-256")
            }
            Self::ManifestIo { operation, detail } => {
                write!(f, "local pull manifest {operation} failed: {detail}")
            }
            Self::Manifest(error) => write!(f, "local pull manifest refused: {error}"),
            Self::ManifestDigestMismatch { expected, observed } => write!(
                f,
                "local pull manifest digest mismatch expected={expected} observed={observed}"
            ),
            Self::InactiveManifest { state } => {
                write!(f, "local pull refused: manifest lifecycle state is {state}")
            }
            Self::InvalidPath(reason) => write!(f, "local pull path refused: {reason}"),
            Self::PartIo { part, operation, detail } => {
                write!(f, "local pull part {part} {operation} failed: {detail}")
            }
            Self::PartLength { part, expected, observed } => write!(
                f,
                "local pull part {part} length mismatch expected={expected} observed={observed}"
            ),
            Self::PartDigest { part, expected, observed } => write!(
                f,
                "local pull part {part} digest mismatch expected={expected} observed={observed}"
            ),
            Self::ArtifactLength { expected, observed } => write!(
                f,
                "local pull artifact length mismatch expected={expected} observed={observed}"
            ),
            Self::ArtifactDigest { expected, observed } => write!(
                f,
                "local pull artifact digest mismatch expected={expected} observed={observed}"
            ),
            Self::Container(error) => write!(f, "local pull fnlpq verification failed: {error}"),
            Self::ContainerIdentity(field) => {
                write!(f, "local pull fnlpq identity disagrees with manifest field {field}")
            }
            Self::Publish { operation, detail } => {
                write!(f, "local pull publish {operation} failed: {detail}")
            }
        }
    }
}

impl Error for PullError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Manifest(error) => Some(error),
            Self::Container(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ManifestError> for PullError {
    fn from(error: ManifestError) -> Self {
        Self::Manifest(error)
    }
}

impl From<FnlpqReadError> for PullError {
    fn from(error: FnlpqReadError) -> Self {
        Self::Container(error)
    }
}

/// Reconstruct one explicit local release package without network access.
///
/// Ordering is deliberate:
/// 1. authenticate and compatibility-check the manifest before opening parts;
/// 2. create the staging output with create-new semantics;
/// 3. verify every part while copying in canonical order;
/// 4. sync and re-read the complete staged artifact;
/// 5. validate the fnlpq container and manifest-bound identities;
/// 6. atomically create the final name with a no-replace hard link.
///
/// This does not activate the model and does not grant model-root authority.
pub fn reconstruct_local_release(
    request: &LocalReconstructRequest,
) -> Result<LocalReconstructReport, PullError> {
    validate_request_paths(request)?;
    validate_sha256_text(&request.expected_release_manifest_sha256)
        .ok_or(PullError::InvalidExpectedManifestDigest)?;

    let manifest_bytes = read_manifest(&request.manifest)?;
    let observed_manifest_sha256 = release_manifest_sha256(&manifest_bytes);
    if observed_manifest_sha256 != request.expected_release_manifest_sha256 {
        return Err(PullError::ManifestDigestMismatch {
            expected: request.expected_release_manifest_sha256.clone(),
            observed: observed_manifest_sha256,
        });
    }

    let manifest = ReleaseManifest::parse(&manifest_bytes)?;
    manifest.validate_for(NANBEIGE42_INT8_V1_COMPATIBILITY)?;
    if manifest.lifecycle.state != "active" {
        return Err(PullError::InactiveManifest {
            state: manifest.lifecycle.state.clone(),
        });
    }

    let mut staging = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&request.staging_output)
        .map_err(|error| PullError::Publish {
            operation: "create-new staging output",
            detail: error.to_string(),
        })?;

    let mut artifact_hasher = Sha256::new();
    let mut artifact_bytes = 0_u64;
    let mut buffer = vec![0_u8; COPY_CHUNK_BYTES];

    for (index, part) in manifest.parts.iter().enumerate() {
        let path = request.parts_dir.join(&part.name);
        let metadata = fs::symlink_metadata(&path).map_err(|error| PullError::PartIo {
            part: index,
            operation: "symlink_metadata",
            detail: error.to_string(),
        })?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Err(PullError::PartIo {
                part: index,
                operation: "regular-file check",
                detail: "release part must be a non-symlink regular file".to_owned(),
            });
        }
        if metadata.len() != part.bytes {
            return Err(PullError::PartLength {
                part: index,
                expected: part.bytes,
                observed: metadata.len(),
            });
        }

        let mut file = File::open(&path).map_err(|error| PullError::PartIo {
            part: index,
            operation: "open",
            detail: error.to_string(),
        })?;
        let mut part_hasher = Sha256::new();
        let mut part_bytes = 0_u64;
        loop {
            let read = file.read(&mut buffer).map_err(|error| PullError::PartIo {
                part: index,
                operation: "read",
                detail: error.to_string(),
            })?;
            if read == 0 {
                break;
            }
            part_bytes = part_bytes
                .checked_add(read as u64)
                .ok_or(PullError::PartLength {
                    part: index,
                    expected: part.bytes,
                    observed: u64::MAX,
                })?;
            if part_bytes > part.bytes {
                return Err(PullError::PartLength {
                    part: index,
                    expected: part.bytes,
                    observed: part_bytes,
                });
            }
            part_hasher.update(&buffer[..read]);
            artifact_hasher.update(&buffer[..read]);
            staging
                .write_all(&buffer[..read])
                .map_err(|error| PullError::Publish {
                    operation: "write staging output",
                    detail: error.to_string(),
                })?;
        }
        if part_bytes != part.bytes {
            return Err(PullError::PartLength {
                part: index,
                expected: part.bytes,
                observed: part_bytes,
            });
        }
        let observed = hex_lower(&part_hasher.finalize());
        if observed != part.sha256 {
            return Err(PullError::PartDigest {
                part: index,
                expected: part.sha256.clone(),
                observed,
            });
        }
        artifact_bytes = artifact_bytes
            .checked_add(part_bytes)
            .ok_or(PullError::ArtifactLength {
                expected: manifest.artifact.bytes,
                observed: u64::MAX,
            })?;
    }

    if artifact_bytes != manifest.artifact.bytes {
        return Err(PullError::ArtifactLength {
            expected: manifest.artifact.bytes,
            observed: artifact_bytes,
        });
    }
    let artifact_sha256 = hex_lower(&artifact_hasher.finalize());
    if artifact_sha256 != manifest.artifact.sha256 {
        return Err(PullError::ArtifactDigest {
            expected: manifest.artifact.sha256.clone(),
            observed: artifact_sha256,
        });
    }

    staging.flush().map_err(|error| PullError::Publish {
        operation: "flush staging output",
        detail: error.to_string(),
    })?;
    staging.sync_all().map_err(|error| PullError::Publish {
        operation: "sync staging output",
        detail: error.to_string(),
    })?;
    drop(staging);

    let reader = FnlpqRangeReader::open(&request.staging_output)?;
    verify_container_identity(&reader, &manifest)?;

    // std::fs::hard_link is a no-replace publication primitive: it fails when
    // the destination exists and gives the final name the exact verified inode.
    // Requiring one parent directory makes cross-device failure impossible by
    // construction and keeps the caller-visible staging artifact explicit.
    fs::hard_link(&request.staging_output, &request.output).map_err(|error| PullError::Publish {
        operation: "create final no-replace hard link",
        detail: error.to_string(),
    })?;

    Ok(LocalReconstructReport {
        release_manifest_sha256: request.expected_release_manifest_sha256.clone(),
        artifact_sha256: manifest.artifact.sha256.clone(),
        artifact_bytes,
        part_count: manifest.parts.len(),
        logical_model_sha256: manifest.digests.logical_model_sha256.clone(),
        output: request.output.clone(),
        staging_output: request.staging_output.clone(),
        directory_durability_confirmed: false,
    })
}

fn read_manifest(path: &Path) -> Result<Vec<u8>, PullError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| PullError::ManifestIo {
        operation: "symlink_metadata",
        detail: error.to_string(),
    })?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(PullError::ManifestIo {
            operation: "regular-file check",
            detail: "manifest must be a non-symlink regular file".to_owned(),
        });
    }
    if metadata.len() > MAX_RELEASE_MANIFEST_BYTES as u64 {
        return Err(PullError::ManifestIo {
            operation: "size check",
            detail: format!("manifest exceeds {MAX_RELEASE_MANIFEST_BYTES} bytes"),
        });
    }
    let mut file = File::open(path).map_err(|error| PullError::ManifestIo {
        operation: "open",
        detail: error.to_string(),
    })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(metadata.len() as usize)
        .map_err(|_| PullError::ManifestIo {
            operation: "allocation",
            detail: "manifest allocation refused".to_owned(),
        })?;
    file.read_to_end(&mut bytes)
        .map_err(|error| PullError::ManifestIo {
            operation: "read",
            detail: error.to_string(),
        })?;
    if bytes.len() > MAX_RELEASE_MANIFEST_BYTES {
        return Err(PullError::ManifestIo {
            operation: "size check",
            detail: format!("manifest exceeds {MAX_RELEASE_MANIFEST_BYTES} bytes"),
        });
    }
    Ok(bytes)
}

fn validate_request_paths(request: &LocalReconstructRequest) -> Result<(), PullError> {
    if request.manifest.as_os_str().is_empty()
        || request.parts_dir.as_os_str().is_empty()
        || request.staging_output.as_os_str().is_empty()
        || request.output.as_os_str().is_empty()
    {
        return Err(PullError::InvalidPath("all paths must be explicit and non-empty"));
    }
    if request.staging_output == request.output {
        return Err(PullError::InvalidPath(
            "staging output and final output must be different paths",
        ));
    }
    let staging_parent = request
        .staging_output
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let output_parent = request.output.parent().unwrap_or_else(|| Path::new("."));
    if staging_parent != output_parent {
        return Err(PullError::InvalidPath(
            "staging output and final output must have the same parent directory",
        ));
    }
    if request.staging_output.file_name().is_none() || request.output.file_name().is_none() {
        return Err(PullError::InvalidPath("output paths must name files"));
    }
    Ok(())
}

fn verify_container_identity(
    reader: &FnlpqRangeReader,
    manifest: &ReleaseManifest,
) -> Result<(), PullError> {
    if reader.prelude().file_len != manifest.artifact.bytes {
        return Err(PullError::ContainerIdentity("artifact.bytes"));
    }
    if reader.model_id() != manifest.model_id {
        return Err(PullError::ContainerIdentity("model_id"));
    }
    if reader.revision() != manifest.source_revision {
        return Err(PullError::ContainerIdentity("source_revision"));
    }
    if reader.recipe_id() != manifest.compatibility.recipe_id {
        return Err(PullError::ContainerIdentity("compatibility.recipe_id"));
    }
    if reader.source_root_sha256() != manifest.source.source_root_sha256 {
        return Err(PullError::ContainerIdentity("source.source_root_sha256"));
    }
    if reader.logical_model_sha256() != manifest.digests.logical_model_sha256 {
        return Err(PullError::ContainerIdentity("digests.logical_model_sha256"));
    }
    if reader.packing_set_sha256() != manifest.digests.packing_set_sha256 {
        return Err(PullError::ContainerIdentity("digests.packing_set_sha256"));
    }
    if reader.license_bundle_sha256() != manifest.license_bundle_sha256 {
        return Err(PullError::ContainerIdentity("license_bundle_sha256"));
    }
    Ok(())
}

fn validate_sha256_text(value: &str) -> Option<()> {
    (value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    .then_some(())
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(staging: &str, output: &str) -> LocalReconstructRequest {
        LocalReconstructRequest {
            manifest: "release.manifest.json".into(),
            expected_release_manifest_sha256: "ab".repeat(32),
            parts_dir: "parts".into(),
            staging_output: staging.into(),
            output: output.into(),
        }
    }

    #[test]
    fn trusted_manifest_pin_requires_exact_lowercase_sha256() {
        assert!(validate_sha256_text(&"00".repeat(32)).is_some());
        assert!(validate_sha256_text(&"af".repeat(32)).is_some());
        assert!(validate_sha256_text(&"AF".repeat(32)).is_none());
        assert!(validate_sha256_text(&"0".repeat(63)).is_none());
        assert!(validate_sha256_text(&"g0".repeat(32)).is_none());
    }

    #[test]
    fn staging_and_final_names_are_distinct_and_same_directory() {
        assert!(validate_request_paths(&request("cache/model.stage", "cache/model.fnlpq")).is_ok());
        assert!(validate_request_paths(&request("model.stage", "model.fnlpq")).is_ok());
        assert!(validate_request_paths(&request("model.fnlpq", "model.fnlpq")).is_err());
        assert!(validate_request_paths(&request("a/model.stage", "b/model.fnlpq")).is_err());
        assert!(validate_request_paths(&request("", "model.fnlpq")).is_err());
    }

    #[test]
    fn hex_rendering_is_fixed_lowercase() {
        assert_eq!(hex_lower(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
    }
}
