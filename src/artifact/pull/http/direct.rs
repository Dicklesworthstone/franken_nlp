//! Acquire an explicit candidate artifact without retaining release-part files.
//!
//! Each response range appends directly to one caller-controlled staging file.
//! Complete retained parts are authenticated before the first request. A
//! retained incomplete part is rehashed, but is not trusted until its complete
//! manifest digest passes. Failed mirrors restore the exact pre-attempt file
//! length, including the already authenticated earlier parts; spent network
//! bytes are never refunded. A successful bounded partial remains resumable.
//!
//! This is the same stable-path candidate workspace contract as local
//! reconstruction, not a hostile model-root transaction or activation journal.
//! Staging remains after every outcome. Final publication uses the shared
//! whole-artifact/container checks and no-replace hard-link boundary.

use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::PathBuf,
};

use asupersync::{CancelKind, cx::Cx, http::HttpClient};
use sha2::{Digest, Sha256};

use super::{
    COPY_BUFFER_BYTES, NetworkPartsOptions, NetworkPullError, USER_AGENT,
    checkpoint, drive_attempt, exhausted_mirrors, fetch_range, hex_lower,
    next_range_end, staging_error, validate_options,
};
use super::super::{
    LocalReconstructPhase, LocalReconstructProgress, PullError,
    check_staging_path, open_staging, read_pinned_manifest,
    verify_and_publish_staged_artifact, verify_local_release_cache,
};
use crate::{
    artifact::manifest::{ReleaseManifest, ReleasePart},
    native_engine::decode::DecodeCancellationKind,
};

#[derive(Clone, Debug)]
pub struct NetworkArtifactRequest {
    pub manifest: PathBuf,
    pub expected_release_manifest_sha256: String,
    /// Explicit caller-controlled staging file; retained after every outcome.
    pub staging_output: PathBuf,
    /// Same-parent final pathname. Existing files must pass full verification.
    pub output: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NetworkArtifactReport {
    pub release_manifest_sha256: String,
    pub artifact_sha256: String,
    pub logical_model_sha256: String,
    pub artifact_bytes: u64,
    pub part_count: usize,
    /// True only after independently rechecking an existing complete artifact.
    pub cache_hit: bool,
    /// Staging bytes retained from before this invocation (zero on a cache hit).
    pub resumed_bytes: u64,
    /// All newly received body bytes, including discarded mirror attempts.
    pub downloaded_bytes: u64,
    /// Newly appended artifact bytes, excluding discarded mirror attempts.
    pub retained_new_bytes: u64,
    pub output: PathBuf,
    pub staging_output: PathBuf,
    pub directory_durability_confirmed: bool,
    pub activation_authority: bool,
}

/// Stream verified release parts into one explicit staging artifact.
///
/// No parts directory is read or created. `options.resume` permits an existing
/// regular staging file; a missing staging path always uses create-new. Final
/// cache verification precedes staging and network access, including when
/// resume is disabled or the new-byte allowance is zero. The network allowance
/// and per-mirror timeout have exactly the existing part transport semantics.
///
/// The caller must admit local verification and network workspace before this
/// call. `NlpEngine::acquire_release_artifact_candidate` supplies those modeled
/// ledger reservations. Synchronous filesystem operations are cooperative.
/// Cancellation after publication can leave a verified final pathname without
/// a success receipt; retry authenticates that pathname as a cache hit.
pub async fn acquire_release_artifact(
    cx: &Cx,
    request: &NetworkArtifactRequest,
    options: NetworkPartsOptions,
) -> Result<NetworkArtifactReport, NetworkPullError> {
    validate_options(options)?;
    checkpoint(cx, None)?;
    validate_request(request)?;
    let cached = verify_local_release_cache(
        &request.manifest, &request.expected_release_manifest_sha256,
        &request.output, |_| local_cancellation(cx),
    ).map_err(|error| local_error(cx, error))?;
    if let Some(cached) = cached {
        checkpoint(cx, None)?;
        return Ok(NetworkArtifactReport {
            release_manifest_sha256: cached.release_manifest_sha256,
            artifact_sha256: cached.artifact_sha256,
            logical_model_sha256: cached.logical_model_sha256,
            artifact_bytes: cached.artifact_bytes,
            part_count: cached.part_count,
            cache_hit: true,
            resumed_bytes: 0,
            downloaded_bytes: 0,
            retained_new_bytes: 0,
            output: request.output.clone(),
            staging_output: request.staging_output.clone(),
            directory_durability_confirmed: false,
            activation_authority: false,
        });
    }

    // Authenticate again rather than allowing a changed local manifest to
    // change the transfer authority between cache inspection and staging.
    let manifest = read_pinned_manifest(
        &request.manifest, &request.expected_release_manifest_sha256,
    )?;
    checkpoint(cx, None)?;
    let retained = match fs::symlink_metadata(&request.staging_output) {
        Ok(_) => options.resume,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(PullError::Publish {
            operation: "inspect direct acquisition staging", detail: error.to_string(),
        }.into()),
    };
    let mut staging = open_staging(&request.staging_output, retained)?;
    let resumed_bytes = staging.metadata().map_err(|error| PullError::Publish {
        operation: "inspect direct acquisition staging length", detail: error.to_string(),
    })?.len();
    check_staging_path(&staging, &request.staging_output, resumed_bytes)?;
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    let prefix = authenticate_prefix(
        &mut staging, &manifest, resumed_bytes, &mut buffer,
        &mut |part| checkpoint(cx, part),
    )?;
    check_staging_path(&staging, &request.staging_output, resumed_bytes)?;
    let mut downloaded_bytes = 0_u64;
    if prefix.complete_parts < manifest.parts.len() {
        let max_part = manifest.parts.iter().map(|part| part.bytes).max().unwrap_or(1);
        let max_body = usize::try_from(max_part)
            .map_err(|_| NetworkPullError::InvalidOptions("part length does not fit host usize"))?;
        let client = HttpClient::builder()
            .no_redirects().no_retries().no_cookie_store().user_agent(USER_AGENT)
            .max_body_size(max_body).request_timeout(options.request_timeout).build();
        let mut artifact_offset = prefix.verified_bytes;
        for (index, part) in manifest.parts.iter().enumerate().skip(prefix.complete_parts) {
            checkpoint(cx, Some(index))?;
            let start = if index == prefix.complete_parts {
                resumed_bytes - artifact_offset
            } else { 0 };
            let prefix_hasher = if index == prefix.complete_parts {
                prefix.partial_hasher.clone()
            } else { Sha256::new() };
            if part.mirrors.is_empty() {
                return Err(NetworkPullError::NoMirrors { part: index });
            }
            let mut success = None;
            let mut last_error = None;
            for (mirror_index, mirror) in part.mirrors.iter().enumerate() {
                checkpoint(cx, Some(index))?;
                let end = artifact_range_end(index, artifact_offset, start, part.bytes,
                    manifest.artifact.bytes, options.max_new_bytes, downloaded_bytes)?;
                let base_len = artifact_offset.checked_add(start)
                    .ok_or(NetworkPullError::InvalidOptions("artifact offset overflow"))?;
                check_staging_path(&staging, &request.staging_output, base_len)?;
                staging.seek(SeekFrom::Start(base_len))
                    .map_err(|error| staging_error(index, "seek-artifact-append", error))?;
                let mut hasher = prefix_hasher.clone();
                let result = drive_attempt(cx, index, mirror_index, options.request_timeout,
                    fetch_range(cx, &client, index, mirror_index, mirror, start, end, part.bytes,
                        &mut staging, &mut hasher, &mut downloaded_bytes, options)).await;
                match finish_artifact_attempt(index, mirror_index, part, artifact_offset,
                    start, end, &mut staging, &hasher, result) {
                    Ok(received) => {
                        success = Some(received);
                        break;
                    }
                    Err(error @ (NetworkPullError::Cancelled { .. }
                        | NetworkPullError::Staging { .. })) => return Err(error),
                    Err(error) => last_error = Some(error),
                }
            }
            let Some(received) = success else {
                artifact_range_end(index, artifact_offset, start, part.bytes,
                    manifest.artifact.bytes, options.max_new_bytes, downloaded_bytes)?;
                return Err(exhausted_mirrors(index, part.mirrors.len(), last_error));
            };
            staging.flush().map_err(|error| staging_error(index, "flush-artifact-part", error))?;
            staging.sync_all().map_err(|error| staging_error(index, "sync-artifact-part", error))?;
            let retained_part_bytes = start.checked_add(received)
                .ok_or(NetworkPullError::InvalidOptions("retained part length overflow"))?;
            if retained_part_bytes < part.bytes {
                return Err(NetworkPullError::DownloadBudgetExceeded {
                    limit: options.max_new_bytes.unwrap_or(downloaded_bytes),
                    downloaded_bytes,
                    retained_bytes: artifact_offset + retained_part_bytes,
                    total_bytes: manifest.artifact.bytes, part: index,
                });
            }
            artifact_offset = artifact_offset.checked_add(part.bytes)
                .ok_or(NetworkPullError::InvalidOptions("artifact offset overflow"))?;
        }
    }
    checkpoint(cx, None)?;
    let retained_new_bytes = manifest.artifact.bytes.checked_sub(resumed_bytes)
        .ok_or(NetworkPullError::InvalidOptions("retained artifact length overflow"))?;
    let mut progress = LocalReconstructProgress {
        phase: LocalReconstructPhase::VerifyArtifact,
        part_index: None,
        verified_prefix_bytes: resumed_bytes,
        copied_bytes: retained_new_bytes,
        artifact_bytes: manifest.artifact.bytes,
    };
    verify_and_publish_staged_artifact(
        &mut staging, &manifest, &request.staging_output, &request.output,
        &mut buffer, &mut progress, &mut |_| local_cancellation(cx),
    ).map_err(|error| local_error(cx, error))?;
    checkpoint(cx, None)?;
    Ok(NetworkArtifactReport {
        release_manifest_sha256: request.expected_release_manifest_sha256.clone(),
        artifact_sha256: manifest.artifact.sha256.clone(),
        logical_model_sha256: manifest.digests.logical_model_sha256.clone(),
        artifact_bytes: manifest.artifact.bytes,
        part_count: manifest.parts.len(),
        cache_hit: false,
        resumed_bytes,
        downloaded_bytes,
        retained_new_bytes,
        output: request.output.clone(),
        staging_output: request.staging_output.clone(),
        directory_durability_confirmed: false,
        activation_authority: false,
    })
}

fn validate_request(request: &NetworkArtifactRequest) -> Result<(), NetworkPullError> {
    if request.manifest.as_os_str().is_empty() || request.staging_output.as_os_str().is_empty()
        || request.output.as_os_str().is_empty() || request.manifest.file_name().is_none()
        || request.staging_output.file_name().is_none() || request.output.file_name().is_none()
    {
        return Err(NetworkPullError::InvalidOptions("manifest, staging and output must be explicit file paths"));
    }
    if request.staging_output == request.output || request.staging_output == request.manifest
        || request.output == request.manifest
    {
        return Err(NetworkPullError::InvalidOptions("manifest, staging and output paths must be distinct"));
    }
    if request.staging_output.parent() != request.output.parent() {
        return Err(NetworkPullError::InvalidOptions("staging and output must have the same parent directory"));
    }
    Ok(())
}

struct RetainedPrefix {
    complete_parts: usize,
    verified_bytes: u64,
    partial_hasher: Sha256,
}

fn authenticate_prefix(
    staging: &mut File, manifest: &ReleaseManifest, length: u64, buffer: &mut [u8],
    check: &mut impl FnMut(Option<usize>) -> Result<(), NetworkPullError>,
) -> Result<RetainedPrefix, NetworkPullError> {
    check(None)?;
    if length > manifest.artifact.bytes {
        return Err(PullError::ResumeLength {
            expected_max: manifest.artifact.bytes, observed: length,
        }.into());
    }
    let mut prefix = RetainedPrefix {
        complete_parts: 0, verified_bytes: 0, partial_hasher: Sha256::new(),
    };
    staging.seek(SeekFrom::Start(0))
        .map_err(|error| staging_error(0, "seek-artifact-prefix", error))?;
    for (index, part) in manifest.parts.iter().enumerate() {
        if prefix.verified_bytes == length { break; }
        check(Some(index))?;
        let retained = (length - prefix.verified_bytes).min(part.bytes);
        let mut remaining = retained;
        let mut hasher = Sha256::new();
        while remaining != 0 {
            check(Some(index))?;
            let want = usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
            let read = staging.read(&mut buffer[..want])
                .map_err(|error| staging_error(index, "read-artifact-prefix", error))?;
            if read == 0 {
                return Err(NetworkPullError::Staging { part: index, operation: "read-artifact-prefix",
                    detail: "retained artifact ended before its metadata length".to_owned() });
            }
            hasher.update(&buffer[..read]);
            remaining -= read as u64;
        }
        if retained < part.bytes {
            prefix.partial_hasher = hasher;
            break;
        }
        check(Some(index))?;
        let observed = hex_lower(&hasher.finalize());
        if observed != part.sha256 {
            return Err(NetworkPullError::Digest { part: index, expected: part.sha256.clone(), observed });
        }
        prefix.verified_bytes += part.bytes;
        prefix.complete_parts += 1;
    }
    check(None)?;
    staging.seek(SeekFrom::Start(length))
        .map_err(|error| staging_error(prefix.complete_parts, "seek-retained-artifact-end", error))?;
    Ok(prefix)
}

fn artifact_range_end(
    part: usize, artifact_offset: u64, start: u64, part_bytes: u64, artifact_bytes: u64,
    limit: Option<u64>, downloaded_bytes: u64,
) -> Result<u64, NetworkPullError> {
    next_range_end(part, start, part_bytes, artifact_bytes, limit, downloaded_bytes)
        .map_err(|error| match error {
            NetworkPullError::DownloadBudgetExceeded {
                limit, downloaded_bytes, retained_bytes, total_bytes, part,
            } => NetworkPullError::DownloadBudgetExceeded {
                limit, downloaded_bytes, retained_bytes: artifact_offset + retained_bytes,
                total_bytes, part,
            },
            error => error,
        })
}

/// Remote ranges are part-relative; truncation and seek must be artifact-relative.
fn finish_artifact_attempt(
    index: usize, mirror: usize, part: &ReleasePart, artifact_offset: u64,
    start: u64, end: u64, file: &mut File, hasher: &Sha256,
    result: Result<u64, NetworkPullError>,
) -> Result<u64, NetworkPullError> {
    let checked = result.and_then(|received| {
        if end.checked_sub(start).and_then(|bytes| bytes.checked_add(1)) != Some(received) {
            return Err(NetworkPullError::Response { part: index, mirror,
                detail: "body length differs from requested range" });
        }
        if end.checked_add(1) == Some(part.bytes) {
            let observed = hex_lower(&hasher.clone().finalize());
            if observed != part.sha256 {
                return Err(NetworkPullError::Digest { part: index, expected: part.sha256.clone(), observed });
            }
        }
        Ok(received)
    });
    if checked.is_err() {
        let base_len = artifact_offset.checked_add(start)
            .ok_or(NetworkPullError::InvalidOptions("artifact rollback offset overflow"))?;
        file.set_len(base_len).map_err(|error| staging_error(index, "rollback-artifact-attempt", error))?;
        file.seek(SeekFrom::Start(base_len)).map_err(|error| staging_error(index, "seek-artifact-rollback", error))?;
        file.sync_all().map_err(|error| staging_error(index, "sync-artifact-rollback", error))?;
    }
    checked
}

fn local_cancellation(cx: &Cx) -> Option<DecodeCancellationKind> {
    if cx.checkpoint().is_ok() { return None; }
    Some(match cx.cancel_reason().map(|reason| reason.kind) {
        Some(CancelKind::User) => DecodeCancellationKind::User,
        Some(CancelKind::Timeout) => DecodeCancellationKind::Timeout,
        Some(CancelKind::Deadline) => DecodeCancellationKind::Deadline,
        Some(CancelKind::PollQuota) => DecodeCancellationKind::PollQuota,
        Some(CancelKind::CostBudget) => DecodeCancellationKind::CostBudget,
        Some(CancelKind::FailFast) | None => DecodeCancellationKind::FailFast,
        Some(CancelKind::RaceLost) => DecodeCancellationKind::RaceLost,
        Some(CancelKind::ParentCancelled) => DecodeCancellationKind::ParentCancelled,
        Some(CancelKind::ResourceUnavailable) => DecodeCancellationKind::ResourceUnavailable,
        Some(CancelKind::Shutdown) => DecodeCancellationKind::Shutdown,
        Some(CancelKind::LinkedExit) => DecodeCancellationKind::LinkedExit,
    })
}

fn local_error(cx: &Cx, error: PullError) -> NetworkPullError {
    match error {
        PullError::Cancelled { progress, .. } => NetworkPullError::Cancelled {
            cause: cx.cancel_reason().map(|reason| reason.kind), part: progress.part_index,
        },
        error => NetworkPullError::Manifest(error),
    }
}

#[cfg(test)]
mod tests;
