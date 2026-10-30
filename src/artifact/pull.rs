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
//!
//! Explicit resume authenticates every byte already present before appending.
//! Paths and file contents must remain stable in a caller-controlled workspace
//! for the invocation. This is not the hostile/shared-root transaction profile
//! or a model activation capability; same-user concurrent mutation is outside
//! this local candidate operation's authority.

use std::{
    error::Error,
    fmt,
    fs::{self, File, Metadata, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

use crate::artifact::{
    manifest::{
        release_manifest_sha256, ManifestError, ReleaseManifest, ReleasePart,
        NANBEIGE42_INT8_V1_COMPATIBILITY, MAX_RELEASE_MANIFEST_BYTES,
    },
    reader::{FnlpqRangeReader, FnlpqReadError},
};
use crate::native_engine::decode::DecodeCancellationKind;

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

/// Explicit restart and new-copy limits for one local reconstruction.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LocalReconstructOptions {
    /// Open an existing regular staging file and authenticate its entire prefix
    /// before appending. False preserves the create-new-only default.
    pub resume: bool,
    /// Maximum bytes newly appended by this invocation. Verification reads do
    /// not spend this allowance. Zero permits verification/publication of an
    /// already complete staging file, but cannot add bytes to an incomplete one.
    pub max_new_bytes: Option<u64>,
}

/// Bounded reconstruction phase reported at cooperative checkpoints.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalReconstructPhase {
    VerifyPrefix,
    Copy,
    VerifyArtifact,
    Publish,
}

/// Metadata-only progress; byte counters never contain input contents.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalReconstructProgress {
    pub phase: LocalReconstructPhase,
    pub part_index: Option<usize>,
    /// Retained bytes whose containing source parts have passed their digests.
    pub verified_prefix_bytes: u64,
    /// Bytes appended during this invocation, excluding the retained prefix.
    pub copied_bytes: u64,
    /// Manifest-declared complete artifact length, not a completion claim.
    pub artifact_bytes: u64,
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
    pub resumed_bytes: u64,
    pub copied_bytes: u64,
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
    ResumeLength { expected_max: u64, observed: u64 },
    ResumePrefixMismatch { part: usize, offset: u64 },
    StagingChanged(&'static str),
    Cancelled { cause: DecodeCancellationKind, progress: LocalReconstructProgress },
    CopyBudgetExceeded {
        limit: u64,
        copied_bytes: u64,
        retained_bytes: u64,
        artifact_bytes: u64,
    },
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
            Self::ResumeLength { expected_max, observed } => write!(
                f,
                "local pull resume refused: staging length {observed} exceeds artifact length {expected_max}"
            ),
            Self::ResumePrefixMismatch { part, offset } => write!(
                f,
                "local pull resume refused: staging prefix differs from part {part} at artifact offset {offset}"
            ),
            Self::StagingChanged(reason) => {
                write!(f, "local pull staging changed during reconstruction: {reason}")
            }
            Self::Cancelled { cause, progress } => write!(
                f,
                "local pull cancelled: {cause:?}; phase={:?} verified_prefix_bytes={} copied_bytes={}",
                progress.phase, progress.verified_prefix_bytes, progress.copied_bytes
            ),
            Self::CopyBudgetExceeded { limit, copied_bytes, retained_bytes, artifact_bytes } => write!(
                f,
                "local pull copy budget exhausted: limit={limit} copied_bytes={copied_bytes} retained_bytes={retained_bytes} artifact_bytes={artifact_bytes}; staging preserved for explicit resume"
            ),
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
    reconstruct_local_release_with_options(request, &LocalReconstructOptions::default(), |_| None)
}

/// Reconstruct or explicitly resume one local package with bounded copy work.
///
/// Resume never trusts the staging filename, length, or an unauthenticated
/// journal as evidence of its contents. Every retained byte is compared with
/// its source part, and each such source part is hashed in full, before the
/// first append. Even an incomplete last part must pass that preflight. A
/// mismatch refuses the operation without truncation or repair. Source parts
/// remain required when resuming an already complete staging file.
///
/// Checkpoints occur before staging/part work and between bounded reads/writes,
/// including prefix, whole-file and container verification. A synchronous
/// filesystem call cannot be preempted. Cancellation and copy-budget failure
/// preserve staging and return no successful report; only final publication
/// produces success. The byte budget counts new copied bytes, not verification
/// reads, and is not a claim about total IO, wall time, or durable activation.
pub fn reconstruct_local_release_with_options(
    request: &LocalReconstructRequest,
    options: &LocalReconstructOptions,
    mut checkpoint: impl FnMut(LocalReconstructProgress) -> Option<DecodeCancellationKind>,
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

    let mut progress = LocalReconstructProgress {
        phase: if options.resume {
            LocalReconstructPhase::VerifyPrefix
        } else {
            LocalReconstructPhase::Copy
        },
        part_index: None,
        verified_prefix_bytes: 0,
        copied_bytes: 0,
        artifact_bytes: manifest.artifact.bytes,
    };
    reconstruction_checkpoint(progress, &mut checkpoint)?;
    let mut staging = open_staging(&request.staging_output, options.resume)?;
    let resumed_bytes = staging.metadata().map_err(|error| PullError::Publish {
        operation: "inspect opened staging length",
        detail: error.to_string(),
    })?.len();
    if resumed_bytes > manifest.artifact.bytes {
        return Err(PullError::ResumeLength {
            expected_max: manifest.artifact.bytes,
            observed: resumed_bytes,
        });
    }
    let mut buffer = vec![0_u8; COPY_CHUNK_BYTES];
    if resumed_bytes != 0 {
        verify_retained_prefix(
            &mut staging, &manifest, &request.parts_dir, resumed_bytes,
            &mut buffer, &mut progress, &mut checkpoint,
        )?;
    }
    check_staging_path(&staging, &request.staging_output, resumed_bytes)?;
    progress.phase = LocalReconstructPhase::Copy;
    progress.part_index = None;
    copy_remaining_parts(
        &mut staging, &manifest, &request.parts_dir, resumed_bytes, options,
        &mut progress, &mut checkpoint,
    )?;

    staging.flush().map_err(|error| PullError::Publish {
        operation: "flush staging output",
        detail: error.to_string(),
    })?;
    staging.sync_all().map_err(|error| PullError::Publish {
        operation: "sync staging output",
        detail: error.to_string(),
    })?;
    progress.phase = LocalReconstructPhase::VerifyArtifact;
    progress.part_index = None;
    verify_staged_artifact(
        &mut staging, &manifest, &mut buffer, progress, &mut checkpoint,
    )?;
    let pinned_file = staging.try_clone().map_err(|error| PullError::Publish {
        operation: "clone held staging descriptor",
        detail: error.to_string(),
    })?;
    let reader = match FnlpqRangeReader::from_file_with_checkpoint(
        pinned_file, || checkpoint(progress),
    ) {
        Ok(reader) => reader,
        Err(FnlpqReadError::Cancelled(cause)) => {
            return Err(PullError::Cancelled { cause, progress });
        }
        Err(error) => return Err(PullError::Container(error)),
    };
    verify_container_identity(&reader, &manifest)?;

    progress.phase = LocalReconstructPhase::Publish;
    reconstruction_checkpoint(progress, &mut checkpoint)?;
    check_staging_path(&staging, &request.staging_output, manifest.artifact.bytes)?;
    // Hard-link publication cannot replace an existing destination. The held
    // descriptor stays alive through validation and publication; path checks
    // also detect observed replacement. They are not a no-follow transaction
    // against concurrent hostile mutation of the caller-controlled workspace.
    fs::hard_link(&request.staging_output, &request.output).map_err(|error| PullError::Publish {
        operation: "create final no-replace hard link",
        detail: error.to_string(),
    })?;
    check_staging_path(&staging, &request.output, manifest.artifact.bytes)?;

    Ok(LocalReconstructReport {
        release_manifest_sha256: request.expected_release_manifest_sha256.clone(),
        artifact_sha256: manifest.artifact.sha256.clone(),
        artifact_bytes: manifest.artifact.bytes,
        part_count: manifest.parts.len(),
        resumed_bytes,
        copied_bytes: progress.copied_bytes,
        logical_model_sha256: manifest.digests.logical_model_sha256.clone(),
        output: request.output.clone(),
        staging_output: request.staging_output.clone(),
        directory_durability_confirmed: false,
    })
}

fn reconstruction_checkpoint(
    progress: LocalReconstructProgress,
    checkpoint: &mut impl FnMut(LocalReconstructProgress) -> Option<DecodeCancellationKind>,
) -> Result<(), PullError> {
    match checkpoint(progress) {
        Some(cause) => Err(PullError::Cancelled { cause, progress }),
        None => Ok(()),
    }
}

fn open_staging(path: &Path, resume: bool) -> Result<File, PullError> {
    let retained = if resume {
        let metadata = fs::symlink_metadata(path).map_err(|error| PullError::Publish {
            operation: "inspect existing resume staging",
            detail: error.to_string(),
        })?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(PullError::InvalidPath(
                "resume staging must be an existing non-symlink regular file",
            ));
        }
        Some(metadata)
    } else {
        None
    };
    let file = OpenOptions::new()
        .read(true)
        .append(true)
        .create_new(!resume)
        .open(path)
        .map_err(|error| PullError::Publish {
            operation: if resume { "open existing resume staging" } else { "create-new staging output" },
            detail: error.to_string(),
        })?;
    let opened = file.metadata().map_err(|error| PullError::Publish {
        operation: "inspect opened staging",
        detail: error.to_string(),
    })?;
    if !opened.is_file() {
        return Err(PullError::InvalidPath("staging handle must refer to a regular file"));
    }
    if retained.as_ref().is_some_and(|before| {
        before.len() != opened.len() || !same_file_identity(before, &opened)
    }) {
        return Err(PullError::StagingChanged("resume target changed while opening"));
    }
    Ok(file)
}

// These checks detect observed pathname replacement on Unix. Other targets
// retain the regular-file/length checks and the caller-controlled stable-path
// premise; metadata comparisons are never claimed as a hostile-root protocol.
#[cfg(unix)]
fn same_file_identity(left: &Metadata, right: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file_identity(_left: &Metadata, _right: &Metadata) -> bool {
    true
}

fn check_staging_path(file: &File, path: &Path, expected_len: u64) -> Result<(), PullError> {
    let held = file.metadata().map_err(|error| PullError::Publish {
        operation: "inspect held staging",
        detail: error.to_string(),
    })?;
    let named = fs::symlink_metadata(path).map_err(|error| PullError::Publish {
        operation: "inspect named staging",
        detail: error.to_string(),
    })?;
    if !held.is_file() || !named.is_file() || named.file_type().is_symlink()
        || !same_file_identity(&held, &named)
    {
        return Err(PullError::StagingChanged("held and named regular files disagree"));
    }
    if held.len() != expected_len || named.len() != expected_len {
        return Err(PullError::StagingChanged("staging length changed unexpectedly"));
    }
    Ok(())
}

fn open_part(parts_dir: &Path, index: usize, part: &ReleasePart) -> Result<File, PullError> {
    let path = parts_dir.join(&part.name);
    let before = fs::symlink_metadata(&path).map_err(|error| PullError::PartIo {
        part: index, operation: "symlink_metadata", detail: error.to_string(),
    })?;
    if !before.is_file() || before.file_type().is_symlink() {
        return Err(PullError::PartIo {
            part: index,
            operation: "regular-file check",
            detail: "release part must be a non-symlink regular file".to_owned(),
        });
    }
    let file = File::open(&path).map_err(|error| PullError::PartIo {
        part: index, operation: "open", detail: error.to_string(),
    })?;
    let after = file.metadata().map_err(|error| PullError::PartIo {
        part: index, operation: "opened metadata", detail: error.to_string(),
    })?;
    if !after.is_file() || !same_file_identity(&before, &after) {
        return Err(PullError::PartIo {
            part: index,
            operation: "opened regular-file check",
            detail: "release part changed while opening".to_owned(),
        });
    }
    if before.len() != part.bytes || after.len() != part.bytes {
        return Err(PullError::PartLength {
            part: index, expected: part.bytes,
            observed: if after.len() != part.bytes { after.len() } else { before.len() },
        });
    }
    Ok(file)
}

fn read_part_chunk(
    file: &mut File,
    buffer: &mut [u8],
    index: usize,
    part: &ReleasePart,
    read_bytes: &mut u64,
) -> Result<usize, PullError> {
    let read = file.read(buffer).map_err(|error| PullError::PartIo {
        part: index, operation: "read", detail: error.to_string(),
    })?;
    *read_bytes = read_bytes.checked_add(read as u64).ok_or(PullError::PartLength {
        part: index, expected: part.bytes, observed: u64::MAX,
    })?;
    if *read_bytes > part.bytes {
        return Err(PullError::PartLength {
            part: index, expected: part.bytes, observed: *read_bytes,
        });
    }
    Ok(read)
}

fn verify_part(
    index: usize,
    part: &ReleasePart,
    read_bytes: u64,
    hasher: Sha256,
) -> Result<(), PullError> {
    if read_bytes != part.bytes {
        return Err(PullError::PartLength {
            part: index, expected: part.bytes, observed: read_bytes,
        });
    }
    let observed = hex_lower(&hasher.finalize());
    if observed != part.sha256 {
        return Err(PullError::PartDigest {
            part: index, expected: part.sha256.clone(), observed,
        });
    }
    Ok(())
}

fn verify_retained_prefix(
    staging: &mut File,
    manifest: &ReleaseManifest,
    parts_dir: &Path,
    retained_bytes: u64,
    buffer: &mut [u8],
    progress: &mut LocalReconstructProgress,
    checkpoint: &mut impl FnMut(LocalReconstructProgress) -> Option<DecodeCancellationKind>,
) -> Result<(), PullError> {
    staging.seek(SeekFrom::Start(0)).map_err(|error| PullError::Publish {
        operation: "seek retained staging prefix", detail: error.to_string(),
    })?;
    let mut retained = vec![0_u8; buffer.len()];
    let mut part_start = 0_u64;
    for (index, part) in manifest.parts.iter().enumerate() {
        if part_start >= retained_bytes {
            break;
        }
        progress.part_index = Some(index);
        reconstruction_checkpoint(*progress, checkpoint)?;
        let mut source = open_part(parts_dir, index, part)?;
        let prefix_in_part = (retained_bytes - part_start).min(part.bytes);
        let mut part_bytes = 0_u64;
        let mut hasher = Sha256::new();
        loop {
            reconstruction_checkpoint(*progress, checkpoint)?;
            let before = part_bytes;
            let read = read_part_chunk(&mut source, buffer, index, part, &mut part_bytes)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            let compare = prefix_in_part.saturating_sub(before).min(read as u64) as usize;
            if compare != 0 {
                staging.read_exact(&mut retained[..compare]).map_err(|error| PullError::Publish {
                    operation: "read retained staging prefix", detail: error.to_string(),
                })?;
                if let Some(mismatch) = retained[..compare].iter()
                    .zip(&buffer[..compare]).position(|(left, right)| left != right)
                {
                    return Err(PullError::ResumePrefixMismatch {
                        part: index, offset: part_start + before + mismatch as u64,
                    });
                }
            }
        }
        verify_part(index, part, part_bytes, hasher)?;
        progress.verified_prefix_bytes = part_start + prefix_in_part;
        reconstruction_checkpoint(*progress, checkpoint)?;
        part_start = part_start.checked_add(part.bytes).ok_or(PullError::ArtifactLength {
            expected: manifest.artifact.bytes, observed: u64::MAX,
        })?;
    }
    if progress.verified_prefix_bytes != retained_bytes {
        return Err(PullError::StagingChanged("retained prefix exceeds the verified part sequence"));
    }
    Ok(())
}

fn copy_remaining_parts(
    staging: &mut File,
    manifest: &ReleaseManifest,
    parts_dir: &Path,
    retained_bytes: u64,
    options: &LocalReconstructOptions,
    progress: &mut LocalReconstructProgress,
    checkpoint: &mut impl FnMut(LocalReconstructProgress) -> Option<DecodeCancellationKind>,
) -> Result<(), PullError> {
    let mut buffer = vec![0_u8; COPY_CHUNK_BYTES];
    let mut part_start = 0_u64;
    for (index, part) in manifest.parts.iter().enumerate() {
        let part_end = part_start.checked_add(part.bytes).ok_or(PullError::ArtifactLength {
            expected: manifest.artifact.bytes, observed: u64::MAX,
        })?;
        if part_end <= retained_bytes {
            part_start = part_end;
            continue;
        }
        progress.part_index = Some(index);
        reconstruction_checkpoint(*progress, checkpoint)?;
        let mut source = open_part(parts_dir, index, part)?;
        let mut part_bytes = 0_u64;
        let mut hasher = Sha256::new();
        loop {
            reconstruction_checkpoint(*progress, checkpoint)?;
            let before = part_bytes;
            let read = read_part_chunk(&mut source, &mut buffer, index, part, &mut part_bytes)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            let skip = retained_bytes.saturating_sub(part_start + before).min(read as u64) as usize;
            let available = read - skip;
            let allowed = options.max_new_bytes
                .map_or(available as u64, |limit| limit.saturating_sub(progress.copied_bytes))
                .min(available as u64) as usize;
            if allowed != 0 {
                staging.write_all(&buffer[skip..skip + allowed]).map_err(|error| PullError::Publish {
                    operation: "append staging output", detail: error.to_string(),
                })?;
                progress.copied_bytes = progress.copied_bytes.checked_add(allowed as u64)
                    .ok_or(PullError::ArtifactLength {
                        expected: manifest.artifact.bytes, observed: u64::MAX,
                    })?;
                reconstruction_checkpoint(*progress, checkpoint)?;
            }
            if allowed < available {
                return Err(PullError::CopyBudgetExceeded {
                    limit: options.max_new_bytes.unwrap_or(0),
                    copied_bytes: progress.copied_bytes,
                    retained_bytes: retained_bytes.checked_add(progress.copied_bytes)
                        .ok_or(PullError::ArtifactLength {
                            expected: manifest.artifact.bytes, observed: u64::MAX,
                        })?,
                    artifact_bytes: manifest.artifact.bytes,
                });
            }
        }
        verify_part(index, part, part_bytes, hasher)?;
        part_start = part_end;
    }
    let completed = retained_bytes.checked_add(progress.copied_bytes)
        .ok_or(PullError::ArtifactLength {
            expected: manifest.artifact.bytes, observed: u64::MAX,
        })?;
    if completed != manifest.artifact.bytes {
        return Err(PullError::ArtifactLength {
            expected: manifest.artifact.bytes, observed: completed,
        });
    }
    Ok(())
}

fn verify_staged_artifact(
    staging: &mut File,
    manifest: &ReleaseManifest,
    buffer: &mut [u8],
    progress: LocalReconstructProgress,
    checkpoint: &mut impl FnMut(LocalReconstructProgress) -> Option<DecodeCancellationKind>,
) -> Result<(), PullError> {
    reconstruction_checkpoint(progress, checkpoint)?;
    staging.seek(SeekFrom::Start(0)).map_err(|error| PullError::Publish {
        operation: "seek complete staging artifact", detail: error.to_string(),
    })?;
    let mut bytes = 0_u64;
    let mut hasher = Sha256::new();
    loop {
        reconstruction_checkpoint(progress, checkpoint)?;
        let read = staging.read(buffer).map_err(|error| PullError::Publish {
            operation: "rehash complete staging artifact", detail: error.to_string(),
        })?;
        if read == 0 {
            break;
        }
        bytes = bytes.checked_add(read as u64).ok_or(PullError::ArtifactLength {
            expected: manifest.artifact.bytes, observed: u64::MAX,
        })?;
        if bytes > manifest.artifact.bytes {
            return Err(PullError::ArtifactLength {
                expected: manifest.artifact.bytes, observed: bytes,
            });
        }
        hasher.update(&buffer[..read]);
    }
    if bytes != manifest.artifact.bytes {
        return Err(PullError::ArtifactLength {
            expected: manifest.artifact.bytes, observed: bytes,
        });
    }
    let observed = hex_lower(&hasher.finalize());
    if observed != manifest.artifact.sha256 {
        return Err(PullError::ArtifactDigest {
            expected: manifest.artifact.sha256.clone(), observed,
        });
    }
    Ok(())
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
    let file = File::open(path).map_err(|error| PullError::ManifestIo {
        operation: "open",
        detail: error.to_string(),
    })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(MAX_RELEASE_MANIFEST_BYTES + 1)
        .map_err(|_| PullError::ManifestIo {
            operation: "allocation",
            detail: "manifest allocation refused".to_owned(),
        })?;
    file.take(MAX_RELEASE_MANIFEST_BYTES as u64 + 1).read_to_end(&mut bytes)
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
    use std::sync::atomic::{AtomicU64, Ordering};
    use crate::artifact::manifest::{
        ReleaseArtifact, ReleaseCompatibility, ReleaseDigests, ReleaseLifecycle,
        ReleaseSourceClosure,
    };

    static NEXT_TRANSFER_FIXTURE: AtomicU64 = AtomicU64::new(0);

    // Only the private byte-transfer helpers consume this scaled inventory.
    // Public reconstruction still passes ReleaseManifest::parse/validate_for
    // and therefore cannot select a test-sized nonfinal release part.
    fn transfer_fixture() -> (PathBuf, ReleaseManifest, Vec<u8>) {
        let sequence = NEXT_TRANSFER_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "fnlp-local-transfer-{}-{sequence}", std::process::id(),
        ));
        fs::create_dir(&root).expect("new transfer fixture directory");
        let mut combined = Vec::new();
        let mut parts = Vec::new();
        for (index, bytes) in [b"abc".as_slice(), b"DEFG", b"hijkl"].iter().enumerate() {
            let name = format!("fixture.part{index:02}");
            let mut file = OpenOptions::new().write(true).create_new(true)
                .open(root.join(&name)).expect("create unique source part");
            file.write_all(bytes).expect("write source part");
            combined.extend_from_slice(bytes);
            parts.push(ReleasePart {
                id: index as u32, name, bytes: bytes.len() as u64,
                sha256: hex_lower(&Sha256::digest(bytes)), mirrors: Vec::new(),
            });
        }
        let manifest = ReleaseManifest {
            manifest_schema: String::new(), model_id: String::new(),
            source_revision: String::new(), artifact_id: String::new(),
            release_tag: String::new(),
            artifact: ReleaseArtifact {
                logical_name: "fixture.fnlpq".to_owned(), bytes: combined.len() as u64,
                sha256: hex_lower(&Sha256::digest(&combined)),
            },
            parts,
            source: ReleaseSourceClosure { files: Vec::new(), source_root_sha256: String::new() },
            compatibility: ReleaseCompatibility {
                recipe_id: String::new(), converter_id: String::new(),
                fnlpq_format: String::new(), pull_api: String::new(),
            },
            digests: ReleaseDigests {
                tokenizer_sha256: String::new(), template_sha256: String::new(),
                census_sha256: String::new(), logical_model_sha256: String::new(),
                packing_set_sha256: String::new(),
            },
            license_bundle_sha256: String::new(), packing_policy: String::new(),
            lifecycle: ReleaseLifecycle {
                state: "active".to_owned(), superseded_by: None, revocation_reason: None,
            },
        };
        (root, manifest, combined)
    }

    fn transfer_progress(bytes: u64) -> LocalReconstructProgress {
        LocalReconstructProgress {
            phase: LocalReconstructPhase::VerifyPrefix, part_index: None,
            verified_prefix_bytes: 0, copied_bytes: 0, artifact_bytes: bytes,
        }
    }

    #[test]
    fn scaled_multipart_resume_preserves_every_boundary_and_partial_part() {
        let (root, manifest, combined) = transfer_fixture();
        for retained in [0, 1, 3, 4, 7, 8, 12] {
            let path = root.join(format!("retained-{retained}.stage"));
            let mut stage = open_staging(&path, false).expect("create stage");
            stage.write_all(&combined[..retained]).expect("seed prefix");
            let mut progress = transfer_progress(combined.len() as u64);
            let mut buffer = [0_u8; 2];
            verify_retained_prefix(
                &mut stage, &manifest, &root, retained as u64, &mut buffer,
                &mut progress, &mut |_| None,
            ).expect("authenticate complete prefix");
            assert_eq!(progress.verified_prefix_bytes, retained as u64);
            progress.phase = LocalReconstructPhase::Copy;
            copy_remaining_parts(
                &mut stage, &manifest, &root, retained as u64,
                &LocalReconstructOptions::default(), &mut progress, &mut |_| None,
            ).expect("append exact suffix across parts");
            assert_eq!(progress.copied_bytes, (combined.len() - retained) as u64);
            assert_eq!(fs::read(&path).expect("read completed stage"), combined);
            verify_staged_artifact(
                &mut stage, &manifest, &mut buffer, progress, &mut |_| None,
            ).expect("independent whole-file rehash");
        }
    }

    #[test]
    fn copy_budget_spanning_parts_retains_exact_resumable_prefix() {
        let (root, manifest, combined) = transfer_fixture();
        let path = root.join("budget.stage");
        let mut stage = open_staging(&path, false).expect("create stage");
        stage.write_all(&combined[..2]).expect("seed prefix");
        let mut progress = transfer_progress(combined.len() as u64);
        let mut buffer = [0_u8; 2];
        verify_retained_prefix(
            &mut stage, &manifest, &root, 2, &mut buffer, &mut progress, &mut |_| None,
        ).expect("authenticate original prefix");
        progress.phase = LocalReconstructPhase::Copy;
        let error = copy_remaining_parts(
            &mut stage, &manifest, &root, 2,
            &LocalReconstructOptions { resume: true, max_new_bytes: Some(4) },
            &mut progress, &mut |_| None,
        ).expect_err("new-byte allowance ends in part one");
        assert!(matches!(error, PullError::CopyBudgetExceeded {
            limit: 4, copied_bytes: 4, retained_bytes: 6, ..
        }));
        assert_eq!(fs::read(&path).expect("read retained prefix"), combined[..6]);
        let mut resumed = transfer_progress(combined.len() as u64);
        verify_retained_prefix(
            &mut stage, &manifest, &root, 6, &mut buffer, &mut resumed, &mut |_| None,
        ).expect("rehash both retained parts");
        copy_remaining_parts(
            &mut stage, &manifest, &root, 6, &LocalReconstructOptions::default(),
            &mut resumed, &mut |_| None,
        ).expect("finish remaining parts");
        assert_eq!(fs::read(&path).expect("read completed stage"), combined);
    }

    #[test]
    fn damaged_later_source_part_cannot_extend_verified_earlier_prefix() {
        let (root, manifest, combined) = transfer_fixture();
        let path = root.join("damaged-source.stage");
        let mut stage = open_staging(&path, false).expect("create stage");
        stage.write_all(&combined[..8]).expect("seed prefix through part two");
        let mut source = OpenOptions::new().write(true)
            .open(root.join(&manifest.parts[2].name)).expect("open fixture source");
        source.seek(SeekFrom::End(-1)).expect("seek unseen source suffix");
        source.write_all(b"X").expect("mutate source suffix only");
        let mut progress = transfer_progress(combined.len() as u64);
        let error = verify_retained_prefix(
            &mut stage, &manifest, &root, 8, &mut [0_u8; 2],
            &mut progress, &mut |_| None,
        ).expect_err("whole source part digest is required before any append");
        assert!(matches!(error, PullError::PartDigest { part: 2, .. }));
        assert_eq!(fs::read(path).expect("read untouched stage"), combined[..8]);
        assert_eq!(progress.copied_bytes, 0);
    }

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
