//! Candidate HTTPS acquisition for immutable release parts.
//!
//! This is the real pinned asupersync transport, but deliberately not an
//! activation capability. The caller supplies an explicit structured Cx; the
//! manifest is authenticated and compatibility-checked before a network
//! request. Every request is a bounded byte range with identity encoding.
//! Mirrors are attempted sequentially in manifest order. A failed attempt
//! rolls the staging file back to the exact pre-attempt length.
//!
//! Filesystem calls are synchronous and checkpointed between bounded network
//! frames. This candidate therefore does not claim the final G0 HTTPS/platform
//! qualification; it closes the missing executable acquisition path without
//! weakening those gates.

use std::{
    error::Error,
    fmt,
    fs::{self, File, OpenOptions},
    future::poll_fn,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    pin::Pin,
    time::Duration,
};

use asupersync::{
    CancelKind,
    bytes::Buf,
    cx::Cx,
    http::{Body, Frame, HttpClient, Method},
};
use sha2::{Digest, Sha256};

use super::{PullError, read_manifest, validate_sha256_text};
use crate::artifact::{
    manifest::{
        MAX_RELEASE_MANIFEST_PARTS, NANBEIGE42_INT8_V1_COMPATIBILITY, ReleaseManifest, ReleasePart,
        release_manifest_sha256,
    },
    reader::FnlpqRangeReader,
};

const USER_AGENT: &str = "franken_nlp-agent (+https://github.com/Dicklesworthstone/franken_nlp)";
const STAGING_SUFFIX: &str = ".partial";
const COPY_BUFFER_BYTES: usize = 1024 * 1024;
const DEFAULT_REDIRECTS: u32 = 5;

#[derive(Clone, Debug)]
pub struct NetworkPartsRequest {
    pub manifest: PathBuf,
    pub expected_release_manifest_sha256: String,
    /// Caller-controlled cache/workspace. This is not the hostile model root.
    pub parts_dir: PathBuf,
}

#[derive(Clone, Copy, Debug)]
pub struct NetworkPartsOptions {
    /// Reuse and extend existing `.partial` files. Final verified parts are
    /// always reused after a complete digest check, regardless of this flag.
    pub resume: bool,
    /// Maximum newly downloaded bytes across this invocation. A successful
    /// partial range is retained and returns DownloadBudgetExceeded.
    pub max_new_bytes: Option<u64>,
    /// Total timeout for each individual HTTP request, meet-composed with Cx.
    pub request_timeout: Duration,
    /// Maximum explicit HTTPS redirect hops for one mirror.
    pub max_redirects: u32,
}

impl Default for NetworkPartsOptions {
    fn default() -> Self {
        Self {
            resume: true,
            max_new_bytes: None,
            request_timeout: Duration::from_secs(300),
            max_redirects: DEFAULT_REDIRECTS,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NetworkPartState {
    Reused,
    Downloaded,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NetworkPartReport {
    pub part: usize,
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
    pub state: NetworkPartState,
    pub mirror_index: Option<usize>,
    pub resumed_bytes: u64,
    pub downloaded_bytes: u64,
    pub path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NetworkPartsReport {
    pub release_manifest_sha256: String,
    pub parts: Vec<NetworkPartReport>,
    pub reused_bytes: u64,
    pub downloaded_bytes: u64,
    pub complete_bytes: u64,
    pub parts_dir: PathBuf,
    /// Explicitly false: acquisition does not grant model-root activation.
    pub activation_authority: bool,
}

#[derive(Debug)]
pub enum NetworkPullError {
    Manifest(PullError),
    InvalidOptions(&'static str),
    PartsDirectory { operation: &'static str, detail: String },
    PartPath { part: usize, detail: &'static str },
    ExistingPart { part: usize, detail: &'static str },
    Staging { part: usize, operation: &'static str, detail: String },
    NoMirrors { part: usize },
    MirrorExhausted { part: usize, attempts: usize, last: String },
    Redirect { part: usize, mirror: usize, detail: &'static str },
    Response { part: usize, mirror: usize, detail: &'static str },
    Body { part: usize, mirror: usize, detail: String },
    Digest { part: usize, expected: String, observed: String },
    DownloadBudgetExceeded {
        limit: u64,
        downloaded_bytes: u64,
        retained_bytes: u64,
        total_bytes: u64,
        part: usize,
    },
    Cancelled { cause: Option<CancelKind>, part: Option<usize> },
}

impl fmt::Display for NetworkPullError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(error) => write!(f, "network pull manifest refused: {error}"),
            Self::InvalidOptions(detail) => write!(f, "network pull options refused: {detail}"),
            Self::PartsDirectory { operation, detail } => write!(f, "network pull parts directory {operation}: {detail}"),
            Self::PartPath { part, detail } => write!(f, "network pull part {part} path refused: {detail}"),
            Self::ExistingPart { part, detail } => write!(f, "network pull part {part} existing final refused: {detail}"),
            Self::Staging { part, operation, detail } => write!(f, "network pull part {part} staging {operation}: {detail}"),
            Self::NoMirrors { part } => write!(f, "network pull part {part} has no mirrors"),
            Self::MirrorExhausted { part, attempts, last } => write!(f, "network pull part {part} exhausted {attempts} mirrors: {last}"),
            Self::Redirect { part, mirror, detail } => write!(f, "network pull part {part} mirror {mirror} redirect refused: {detail}"),
            Self::Response { part, mirror, detail } => write!(f, "network pull part {part} mirror {mirror} response refused: {detail}"),
            Self::Body { part, mirror, detail } => write!(f, "network pull part {part} mirror {mirror} body failed: {detail}"),
            Self::Digest { part, expected, observed } => write!(f, "network pull part {part} digest mismatch expected={expected} observed={observed}"),
            Self::DownloadBudgetExceeded { limit, downloaded_bytes, retained_bytes, total_bytes, part } =>
                write!(f, "network pull byte budget exhausted part={part} limit={limit} downloaded={downloaded_bytes} retained={retained_bytes} total={total_bytes}; partial staging retained"),
            Self::Cancelled { cause, part } => write!(f, "network pull cancelled cause={cause:?} part={part:?}"),
        }
    }
}
impl Error for NetworkPullError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Manifest(error) => Some(error), _ => None }
    }
}
impl From<PullError> for NetworkPullError {
    fn from(error: PullError) -> Self { Self::Manifest(error) }
}

/// Acquire every immutable part named by an independently pinned manifest.
///
/// Success means every final part is a regular file whose exact complete
/// SHA-256 matches the manifest. It does not reconstruct, pack or activate the
/// model. Callers may feed the resulting directory to the existing local
/// reconstruction path.
pub async fn download_release_parts(
    cx: &Cx,
    request: &NetworkPartsRequest,
    options: NetworkPartsOptions,
) -> Result<NetworkPartsReport, NetworkPullError> {
    validate_options(options)?;
    checkpoint(cx, None)?;
    if !validate_sha256_text(&request.expected_release_manifest_sha256).is_some() {
        return Err(NetworkPullError::Manifest(PullError::InvalidExpectedManifestDigest));
    }
    if request.manifest.as_os_str().is_empty() || request.parts_dir.as_os_str().is_empty() {
        return Err(NetworkPullError::InvalidOptions("manifest and parts_dir must be explicit"));
    }
    let manifest_bytes = read_manifest(&request.manifest)?;
    let observed = release_manifest_sha256(&manifest_bytes);
    if observed != request.expected_release_manifest_sha256 {
        return Err(NetworkPullError::Manifest(PullError::ManifestDigestMismatch {
            expected: request.expected_release_manifest_sha256.clone(), observed,
        }));
    }
    let manifest = ReleaseManifest::parse(&manifest_bytes)
        .map_err(PullError::Manifest)
        .map_err(NetworkPullError::Manifest)?;
    manifest.validate_for(NANBEIGE42_INT8_V1_COMPATIBILITY)
        .map_err(PullError::Manifest)
        .map_err(NetworkPullError::Manifest)?;
    if manifest.lifecycle.state != "active" {
        return Err(NetworkPullError::Manifest(PullError::InactiveManifest {
            state: manifest.lifecycle.state.clone(),
        }));
    }
    validate_parts_dir(&request.parts_dir)?;
    checkpoint(cx, None)?;

    let max_part = manifest.parts.iter().map(|part| part.bytes).max().unwrap_or(1);
    let max_body = usize::try_from(max_part)
        .map_err(|_| NetworkPullError::InvalidOptions("part length does not fit host usize"))?;
    let client = HttpClient::builder()
        .no_redirects()
        .no_retries()
        .no_cookie_store()
        .user_agent(USER_AGENT)
        .max_body_size(max_body)
        .request_timeout(options.request_timeout)
        .build();

    let mut reports = Vec::new();
    reports.try_reserve_exact(manifest.parts.len())
        .map_err(|_| NetworkPullError::InvalidOptions("part report allocation refused"))?;
    let mut downloaded_total = 0_u64;
    let mut reused_total = 0_u64;

    for (index, part) in manifest.parts.iter().enumerate() {
        checkpoint(cx, Some(index))?;
        let final_path = request.parts_dir.join(&part.name);
        if final_path.file_name().and_then(|name| name.to_str()) != Some(part.name.as_str()) {
            return Err(NetworkPullError::PartPath { part: index, detail: "manifest part must remain a portable basename" });
        }
        if final_path.exists() {
            verify_final(index, part, &final_path)?;
            reused_total = reused_total.checked_add(part.bytes)
                .ok_or(NetworkPullError::InvalidOptions("reused byte arithmetic overflow"))?;
            reports.push(NetworkPartReport {
                part: index, name: part.name.clone(), bytes: part.bytes, sha256: part.sha256.clone(),
                state: NetworkPartState::Reused, mirror_index: None, resumed_bytes: part.bytes,
                downloaded_bytes: 0, path: final_path,
            });
            continue;
        }
        let staging_path = request.parts_dir.join(format!("{}{}", part.name, STAGING_SUFFIX));
        let mut staging = open_partial(index, &staging_path, options.resume)?;
        let start = staging.metadata().map_err(|error| staging_error(index, "inspect", error))?.len();
        if start > part.bytes {
            return Err(NetworkPullError::Staging { part: index, operation: "length", detail: "partial exceeds declared part length".to_owned() });
        }
        let mut prefix_hasher = hash_prefix(index, &mut staging, start, cx)?;
        let resumed = start;
        if start == part.bytes {
            let observed = hex_lower(&prefix_hasher.clone().finalize());
            if observed != part.sha256 {
                return Err(NetworkPullError::Digest { part: index, expected: part.sha256.clone(), observed });
            }
            publish_part(index, &staging_path, &final_path)?;
            reused_total = reused_total.checked_add(part.bytes)
                .ok_or(NetworkPullError::InvalidOptions("reused byte arithmetic overflow"))?;
            reports.push(NetworkPartReport {
                part: index, name: part.name.clone(), bytes: part.bytes, sha256: part.sha256.clone(),
                state: NetworkPartState::Reused, mirror_index: None, resumed_bytes: part.bytes,
                downloaded_bytes: 0, path: final_path,
            });
            continue;
        }

        let allowance = options.max_new_bytes.map(|limit| limit.saturating_sub(downloaded_total));
        if allowance == Some(0) {
            return Err(NetworkPullError::DownloadBudgetExceeded {
                limit: options.max_new_bytes.unwrap_or(0), downloaded_bytes: downloaded_total,
                retained_bytes: start, total_bytes: manifest.artifact.bytes, part: index,
            });
        }
        let request_end = match allowance {
            Some(allowed) => part.bytes.saturating_sub(1).min(start.saturating_add(allowed).saturating_sub(1)),
            None => part.bytes - 1,
        };
        let segment_len = request_end - start + 1;
        let base_len = start;
        let mut success = None;
        let mut last_error = String::from("no mirror attempted");
        if part.mirrors.is_empty() { return Err(NetworkPullError::NoMirrors { part: index }); }

        for (mirror_index, mirror) in part.mirrors.iter().enumerate() {
            checkpoint(cx, Some(index))?;
            staging.set_len(base_len).map_err(|error| staging_error(index, "rollback-before-attempt", error))?;
            staging.seek(SeekFrom::Start(base_len)).map_err(|error| staging_error(index, "seek-before-attempt", error))?;
            let mut hasher = prefix_hasher.clone();
            match fetch_range(cx, &client, index, mirror_index, mirror, start, request_end, part.bytes,
                &mut staging, &mut hasher, options).await {
                Ok(received) if received == segment_len => {
                    success = Some((mirror_index, hasher));
                    break;
                }
                Ok(received) => {
                    last_error = format!("short body received={received} expected={segment_len}");
                }
                Err(NetworkPullError::Cancelled { cause, .. }) => {
                    staging.set_len(base_len).map_err(|error| staging_error(index, "rollback-cancelled-attempt", error))?;
                    return Err(NetworkPullError::Cancelled { cause, part: Some(index) });
                }
                Err(error) => last_error = error.to_string(),
            }
        }
        let Some((mirror_index, hasher)) = success else {
            staging.set_len(base_len).map_err(|error| staging_error(index, "rollback-exhausted-attempts", error))?;
            return Err(NetworkPullError::MirrorExhausted {
                part: index, attempts: part.mirrors.len(), last: last_error,
            });
        };
        staging.flush().map_err(|error| staging_error(index, "flush", error))?;
        staging.sync_all().map_err(|error| staging_error(index, "sync", error))?;
        let new_len = base_len.checked_add(segment_len)
            .ok_or(NetworkPullError::InvalidOptions("part length arithmetic overflow"))?;
        downloaded_total = downloaded_total.checked_add(segment_len)
            .ok_or(NetworkPullError::InvalidOptions("downloaded byte arithmetic overflow"))?;
        prefix_hasher = hasher;

        if new_len < part.bytes {
            return Err(NetworkPullError::DownloadBudgetExceeded {
                limit: options.max_new_bytes.unwrap_or(downloaded_total),
                downloaded_bytes: downloaded_total, retained_bytes: new_len,
                total_bytes: manifest.artifact.bytes, part: index,
            });
        }
        let observed = hex_lower(&prefix_hasher.finalize());
        if observed != part.sha256 {
            return Err(NetworkPullError::Digest { part: index, expected: part.sha256.clone(), observed });
        }
        publish_part(index, &staging_path, &final_path)?;
        reports.push(NetworkPartReport {
            part: index, name: part.name.clone(), bytes: part.bytes, sha256: part.sha256.clone(),
            state: NetworkPartState::Downloaded, mirror_index: Some(mirror_index), resumed_bytes: resumed,
            downloaded_bytes: segment_len, path: final_path,
        });
    }
    checkpoint(cx, None)?;
    Ok(NetworkPartsReport {
        release_manifest_sha256: request.expected_release_manifest_sha256.clone(),
        parts: reports, reused_bytes: reused_total, downloaded_bytes: downloaded_total,
        complete_bytes: manifest.artifact.bytes, parts_dir: request.parts_dir.clone(),
        activation_authority: false,
    })
}

async fn fetch_range(
    cx: &Cx,
    client: &HttpClient,
    part: usize,
    mirror: usize,
    initial_url: &str,
    start: u64,
    end: u64,
    total: u64,
    file: &mut File,
    hasher: &mut Sha256,
    options: NetworkPartsOptions,
) -> Result<u64, NetworkPullError> {
    let mut url = initial_url.to_owned();
    for redirects in 0..=options.max_redirects {
        checkpoint(cx, Some(part))?;
        if !url.starts_with("https://") {
            return Err(NetworkPullError::Redirect { part, mirror, detail: "only HTTPS targets are permitted" });
        }
        let response = client.request_streaming(
            cx,
            Method::Get,
            &url,
            vec![
                ("Range".to_owned(), format!("bytes={start}-{end}")),
                ("Accept-Encoding".to_owned(), "identity".to_owned()),
            ],
            Vec::new(),
        ).await.map_err(|error| NetworkPullError::Body { part, mirror, detail: error.to_string() })?;
        if matches!(response.head.status, 301 | 302 | 303 | 307 | 308) {
            if redirects == options.max_redirects {
                return Err(NetworkPullError::Redirect { part, mirror, detail: "redirect limit exceeded" });
            }
            let location = unique_header(&response.head.headers, "location")
                .map_err(|_| NetworkPullError::Redirect { part, mirror, detail: "duplicate Location header" })?
                .ok_or(NetworkPullError::Redirect { part, mirror, detail: "redirect missing Location" })?;
            if !location.starts_with("https://") {
                return Err(NetworkPullError::Redirect { part, mirror, detail: "redirect must be absolute HTTPS URL" });
            }
            url = location.to_owned();
            continue;
        }
        validate_response_head(part, mirror, response.head.status, &response.head.headers, start, end, total)?;
        let mut received = 0_u64;
        let mut body = response.body;
        while let Some(frame) = poll_fn(|task_cx| Pin::new(&mut body).poll_frame(task_cx)).await {
            checkpoint(cx, Some(part))?;
            let frame = frame.map_err(|error| NetworkPullError::Body { part, mirror, detail: error.to_string() })?;
            match frame {
                Frame::Data(mut data) => {
                    while data.remaining() != 0 {
                        let bytes = data.chunk();
                        if bytes.is_empty() {
                            return Err(NetworkPullError::Response { part, mirror, detail: "body yielded empty data chunk" });
                        }
                        let next = received.checked_add(bytes.len() as u64)
                            .ok_or(NetworkPullError::Response { part, mirror, detail: "body length overflow" })?;
                        if next > end - start + 1 {
                            return Err(NetworkPullError::Response { part, mirror, detail: "body exceeds requested range" });
                        }
                        file.write_all(bytes).map_err(|error| staging_error(part, "write-range", error))?;
                        hasher.update(bytes);
                        received = next;
                        let consumed = bytes.len();
                        data.advance(consumed);
                    }
                }
                Frame::Trailers(_) => {
                    return Err(NetworkPullError::Response { part, mirror, detail: "range response trailers are not accepted" });
                }
            }
        }
        if received != end - start + 1 {
            return Err(NetworkPullError::Response { part, mirror, detail: "body length differs from requested range" });
        }
        return Ok(received);
    }
    Err(NetworkPullError::Redirect { part, mirror, detail: "redirect state exhausted" })
}

fn validate_response_head(
    part: usize, mirror: usize, status: u16, headers: &[(String, String)],
    start: u64, end: u64, total: u64,
) -> Result<(), NetworkPullError> {
    if status != 206 {
        return Err(NetworkPullError::Response { part, mirror, detail: "range request requires status 206" });
    }
    let content_encoding = unique_header(headers, "content-encoding")
        .map_err(|_| NetworkPullError::Response { part, mirror, detail: "duplicate Content-Encoding" })?;
    if content_encoding.is_some_and(|value| !value.eq_ignore_ascii_case("identity")) {
        return Err(NetworkPullError::Response { part, mirror, detail: "unexpected Content-Encoding" });
    }
    if unique_header(headers, "transfer-encoding")
        .map_err(|_| NetworkPullError::Response { part, mirror, detail: "duplicate Transfer-Encoding" })?
        .is_some() {
        return Err(NetworkPullError::Response { part, mirror, detail: "Transfer-Encoding is not accepted for authenticated ranges" });
    }
    let expected_len = end.checked_sub(start).and_then(|n| n.checked_add(1))
        .ok_or(NetworkPullError::Response { part, mirror, detail: "invalid requested range" })?;
    let length = unique_header(headers, "content-length")
        .map_err(|_| NetworkPullError::Response { part, mirror, detail: "duplicate Content-Length" })?
        .ok_or(NetworkPullError::Response { part, mirror, detail: "missing Content-Length" })?;
    if length.parse::<u64>().ok() != Some(expected_len) {
        return Err(NetworkPullError::Response { part, mirror, detail: "Content-Length differs from requested range" });
    }
    let content_range = unique_header(headers, "content-range")
        .map_err(|_| NetworkPullError::Response { part, mirror, detail: "duplicate Content-Range" })?
        .ok_or(NetworkPullError::Response { part, mirror, detail: "missing Content-Range" })?;
    if parse_content_range(content_range) != Some((start, end, total)) {
        return Err(NetworkPullError::Response { part, mirror, detail: "Content-Range differs from requested range" });
    }
    Ok(())
}

fn parse_content_range(value: &str) -> Option<(u64, u64, u64)> {
    let rest = value.strip_prefix("bytes ")?;
    let (range, total) = rest.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    let start = parse_decimal(start)?;
    let end = parse_decimal(end)?;
    let total = parse_decimal(total)?;
    (start <= end && end < total).then_some((start, end, total))
}

fn parse_decimal(value: &str) -> Option<u64> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0')) {
        return None;
    }
    value.parse().ok()
}

fn unique_header<'a>(headers: &'a [(String, String)], name: &str) -> Result<Option<&'a str>, ()> {
    let mut found = None;
    for (header, value) in headers {
        if header.eq_ignore_ascii_case(name) {
            if found.is_some() { return Err(()); }
            found = Some(value.as_str());
        }
    }
    Ok(found)
}

fn validate_options(options: NetworkPartsOptions) -> Result<(), NetworkPullError> {
    if options.request_timeout.is_zero() || options.max_redirects > 16 {
        return Err(NetworkPullError::InvalidOptions("request timeout must be nonzero and redirects <=16"));
    }
    Ok(())
}

fn validate_parts_dir(path: &Path) -> Result<(), NetworkPullError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| NetworkPullError::PartsDirectory {
        operation: "inspect", detail: error.to_string(),
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(NetworkPullError::PartsDirectory {
            operation: "type-check", detail: "parts_dir must be an existing non-symlink directory".to_owned(),
        });
    }
    Ok(())
}

fn open_partial(part: usize, path: &Path, resume: bool) -> Result<File, NetworkPullError> {
    if path.exists() {
        if !resume {
            return Err(NetworkPullError::Staging { part, operation: "open", detail: "partial exists and resume is disabled".to_owned() });
        }
        let metadata = fs::symlink_metadata(path).map_err(|error| staging_error(part, "inspect", error))?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(NetworkPullError::Staging { part, operation: "type-check", detail: "partial must be a non-symlink regular file".to_owned() });
        }
        OpenOptions::new().read(true).write(true).open(path)
            .map_err(|error| staging_error(part, "open-resume", error))
    } else {
        OpenOptions::new().read(true).write(true).create_new(true).open(path)
            .map_err(|error| staging_error(part, "create-new", error))
    }
}

fn hash_prefix(part: usize, file: &mut File, length: u64, cx: &Cx) -> Result<Sha256, NetworkPullError> {
    file.seek(SeekFrom::Start(0)).map_err(|error| staging_error(part, "seek-prefix", error))?;
    let mut remaining = length;
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    let mut hasher = Sha256::new();
    while remaining != 0 {
        checkpoint(cx, Some(part))?;
        let want = usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
        let read = file.read(&mut buffer[..want]).map_err(|error| staging_error(part, "read-prefix", error))?;
        if read == 0 {
            return Err(NetworkPullError::Staging { part, operation: "read-prefix", detail: "partial ended before its metadata length".to_owned() });
        }
        hasher.update(&buffer[..read]);
        remaining -= read as u64;
    }
    file.seek(SeekFrom::Start(length)).map_err(|error| staging_error(part, "seek-append", error))?;
    Ok(hasher)
}

fn verify_final(part: usize, expected: &ReleasePart, path: &Path) -> Result<(), NetworkPullError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| NetworkPullError::ExistingPart {
        part, detail: if error.kind() == std::io::ErrorKind::NotFound { "final disappeared" } else { "cannot inspect final" },
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != expected.bytes {
        return Err(NetworkPullError::ExistingPart { part, detail: "final is not the declared regular file length" });
    }
    let mut file = File::open(path).map_err(|_| NetworkPullError::ExistingPart { part, detail: "cannot open final" })?;
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    let mut hasher = Sha256::new();
    loop {
        let read = file.read(&mut buffer).map_err(|_| NetworkPullError::ExistingPart { part, detail: "cannot hash final" })?;
        if read == 0 { break; }
        hasher.update(&buffer[..read]);
    }
    let observed = hex_lower(&hasher.finalize());
    if observed != expected.sha256 {
        return Err(NetworkPullError::Digest { part, expected: expected.sha256.clone(), observed });
    }
    Ok(())
}

fn publish_part(part: usize, staging: &Path, final_path: &Path) -> Result<(), NetworkPullError> {
    fs::hard_link(staging, final_path).map_err(|error| staging_error(part, "publish-no-replace-hard-link", error))?;
    Ok(())
}

fn staging_error(part: usize, operation: &'static str, error: std::io::Error) -> NetworkPullError {
    NetworkPullError::Staging { part, operation, detail: error.to_string() }
}

fn checkpoint(cx: &Cx, part: Option<usize>) -> Result<(), NetworkPullError> {
    if cx.checkpoint().is_err() {
        return Err(NetworkPullError::Cancelled {
            cause: cx.cancel_reason().map(|reason| reason.kind), part,
        });
    }
    Ok(())
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

    #[test]
    fn range_response_requires_exact_206_identity_length_and_total() {
        let headers = vec![
            ("Content-Range".to_owned(), "bytes 10-19/100".to_owned()),
            ("Content-Length".to_owned(), "10".to_owned()),
            ("Content-Encoding".to_owned(), "identity".to_owned()),
        ];
        assert!(validate_response_head(0, 0, 206, &headers, 10, 19, 100).is_ok());
        assert!(validate_response_head(0, 0, 200, &headers, 10, 19, 100).is_err());
        let mut wrong = headers.clone(); wrong[0].1 = "bytes 11-20/100".to_owned();
        assert!(validate_response_head(0, 0, 206, &wrong, 10, 19, 100).is_err());
        let mut encoded = headers.clone(); encoded[2].1 = "gzip".to_owned();
        assert!(validate_response_head(0, 0, 206, &encoded, 10, 19, 100).is_err());
        let mut short = headers.clone(); short[1].1 = "9".to_owned();
        assert!(validate_response_head(0, 0, 206, &short, 10, 19, 100).is_err());
    }

    #[test]
    fn range_parser_is_canonical_and_overflow_safe() {
        assert_eq!(parse_content_range("bytes 0-9/10"), Some((0, 9, 10)));
        assert_eq!(parse_content_range("bytes 10-19/100"), Some((10, 19, 100)));
        for invalid in ["bytes 01-9/10", "bytes 9-1/10", "bytes 0-10/10",
            "bytes */10", "bytes 0-1/*", "bytes 0-1/01", "Bytes 0-1/2",
            "bytes 0-18446744073709551616/18446744073709551617"] {
            assert_eq!(parse_content_range(invalid), None, "{invalid}");
        }
    }

    #[test]
    fn duplicate_authority_headers_are_refused() {
        let headers = vec![
            ("Content-Range".to_owned(), "bytes 0-0/1".to_owned()),
            ("content-range".to_owned(), "bytes 0-0/1".to_owned()),
            ("Content-Length".to_owned(), "1".to_owned()),
        ];
        assert!(validate_response_head(0, 0, 206, &headers, 0, 0, 1).is_err());
    }

    #[test]
    fn options_are_finite_and_redirects_bounded() {
        assert!(validate_options(NetworkPartsOptions::default()).is_ok());
        let mut options = NetworkPartsOptions::default();
        options.request_timeout = Duration::ZERO;
        assert!(validate_options(options).is_err());
        options.request_timeout = Duration::from_secs(1);
        options.max_redirects = 17;
        assert!(validate_options(options).is_err());
    }
}
