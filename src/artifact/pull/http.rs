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
//! frames. Pending transfers also check cancellation on a 100 ms runtime timer;
//! synchronous filesystem work remains cooperative. This candidate therefore
//! does not claim the final G0 HTTPS/platform
//! qualification; it closes the missing executable acquisition path without
//! weakening those gates.

pub mod direct;

use std::{
    error::Error,
    fmt,
    fs::{self, File, OpenOptions},
    future::{Future, poll_fn},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    pin::Pin,
    task::Poll,
    time::Duration,
};

use asupersync::{
    CancelKind,
    bytes::Buf,
    cx::Cx,
    http::{Body, Frame, HttpClient, Method, ParsedUrl, Scheme},
};
use sha2::{Digest, Sha256};

use super::{PullError, read_manifest, validate_sha256_text};
use crate::artifact::{
    manifest::{
        NANBEIGE42_INT8_V1_COMPATIBILITY, ReleaseManifest, ReleasePart,
        release_manifest_sha256,
    },
};

const USER_AGENT: &str = "franken_nlp-agent (+https://github.com/Dicklesworthstone/franken_nlp)";
const STAGING_SUFFIX: &str = ".partial";
const COPY_BUFFER_BYTES: usize = 1024 * 1024;
const DEFAULT_REDIRECTS: u32 = 5;
const CANCELLATION_CHECK_INTERVAL: Duration = Duration::from_millis(100);
// Signed asset URLs can be longer than the manifest's immutable mirror URLs.
// Bound them independently without altering their path or query bytes.
const MAX_DOWNLOAD_URL_BYTES: usize = 16 * 1024;
const GITHUB_RELEASE_REDIRECT_HOSTS: [&str; 3] = [
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
    "github-releases.githubusercontent.com",
];

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
    /// Allowance for requested part-body bytes across ALL mirror attempts.
    /// Received bytes remain charged when an attempt fails or rolls back. A
    /// successful partial range is retained and returns DownloadBudgetExceeded.
    /// This is not a wire-byte cap: HTTP/TLS overhead and unsolicited bytes
    /// from a server that violates its requested range cannot be prevented.
    pub max_new_bytes: Option<u64>,
    /// Total timeout for one mirror attempt, including redirects, response
    /// headers and the complete streaming body, meet-composed with Cx.
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
    /// Body bytes received for this part, including discarded mirror attempts.
    pub downloaded_bytes: u64,
    pub path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NetworkPartsReport {
    pub release_manifest_sha256: String,
    pub parts: Vec<NetworkPartReport>,
    pub reused_bytes: u64,
    /// Body bytes received across all parts and attempts, including failures.
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
    RequestTimeout { part: usize, mirror: usize },
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
            Self::RequestTimeout { part, mirror } => write!(f, "network pull part {part} mirror {mirror} complete-response timeout"),
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
            verify_final(index, part, &final_path, || checkpoint(cx, Some(index)))?;
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
        let prefix_hasher = hash_prefix(index, &mut staging, start, cx)?;
        let resumed = start;
        if start == part.bytes {
            let observed = hex_lower(&prefix_hasher.clone().finalize());
            if observed != part.sha256 {
                return Err(NetworkPullError::Digest { part: index, expected: part.sha256.clone(), observed });
            }
            publish_part(index, &staging_path, &final_path)?;
            verify_final(index, part, &final_path, || checkpoint(cx, Some(index)))?;
            reused_total = reused_total.checked_add(part.bytes)
                .ok_or(NetworkPullError::InvalidOptions("reused byte arithmetic overflow"))?;
            reports.push(NetworkPartReport {
                part: index, name: part.name.clone(), bytes: part.bytes, sha256: part.sha256.clone(),
                state: NetworkPartState::Reused, mirror_index: None, resumed_bytes: part.bytes,
                downloaded_bytes: 0, path: final_path,
            });
            continue;
        }

        let base_len = start;
        let downloaded_before_part = downloaded_total;
        let mut success = None;
        let mut last_error = None;
        if part.mirrors.is_empty() { return Err(NetworkPullError::NoMirrors { part: index }); }

        for (mirror_index, mirror) in part.mirrors.iter().enumerate() {
            checkpoint(cx, Some(index))?;
            // A failed attempt spends the enclosing budget even though its
            // bytes are discarded. Recompute the next Range before contacting
            // each mirror; never reuse the first mirror's larger allowance.
            let request_end = next_range_end(index, start, part.bytes,
                manifest.artifact.bytes, options.max_new_bytes, downloaded_total)?;
            let mut hasher = prefix_hasher.clone();
            let result = drive_attempt(cx, index, mirror_index, options.request_timeout,
                fetch_range(cx, &client, index, mirror_index, mirror, start, request_end, part.bytes,
                    &mut staging, &mut hasher, &mut downloaded_total, options)).await;
            // Digest validation belongs to the attempt, before short-circuiting
            // mirror fallback. Failure restores both length and cursor to the
            // pre-attempt prefix; a corrupt complete suffix cannot poison resume.
            match finish_attempt(index, mirror_index, part, base_len, request_end,
                &mut staging, &hasher, result) {
                Ok(received) => {
                    success = Some((mirror_index, received));
                    break;
                }
                // Retrying cannot repair cancellation or a local storage
                // failure. In particular, rollback failure must stop here.
                Err(error @ (NetworkPullError::Cancelled { .. } | NetworkPullError::Staging { .. })) => return Err(error),
                Err(error) => last_error = Some(error),
            }
        }
        let Some((mirror_index, segment_len)) = success else {
            // Give exhausted enclosing work its typed budget outcome even if
            // the last available mirror consumed the final allowance.
            next_range_end(index, start, part.bytes, manifest.artifact.bytes,
                options.max_new_bytes, downloaded_total)?;
            return Err(exhausted_mirrors(index, part.mirrors.len(), last_error));
        };
        staging.flush().map_err(|error| staging_error(index, "flush", error))?;
        staging.sync_all().map_err(|error| staging_error(index, "sync", error))?;
        let new_len = base_len.checked_add(segment_len)
            .ok_or(NetworkPullError::InvalidOptions("part length arithmetic overflow"))?;
        if new_len < part.bytes {
            return Err(NetworkPullError::DownloadBudgetExceeded {
                limit: options.max_new_bytes.unwrap_or(downloaded_total),
                downloaded_bytes: downloaded_total, retained_bytes: new_len,
                total_bytes: manifest.artifact.bytes, part: index,
            });
        }
        publish_part(index, &staging_path, &final_path)?;
        verify_final(index, part, &final_path, || checkpoint(cx, Some(index)))?;
        reports.push(NetworkPartReport {
            part: index, name: part.name.clone(), bytes: part.bytes, sha256: part.sha256.clone(),
            state: NetworkPartState::Downloaded, mirror_index: Some(mirror_index), resumed_bytes: resumed,
            downloaded_bytes: downloaded_total - downloaded_before_part, path: final_path,
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

/// At selected asupersync revision 362dc5b174427f66cfa76ab2bdd68cce1a95c6cc,
/// `HttpClient::request_streaming` ends its timeout at the response head.
/// Own one timer around the entire attempt instead: a server that supplies
/// headers and then never yields a body frame must still release its file
/// borrow and return through `finish_attempt` before another mirror starts.
/// Redirects and body frames share this deadline; neither renews it.
///
/// This pin exposes no public Cx cancellation notification future or Waker
/// registration. A separate runtime timer checkpoints pending work every
/// 100 ms, subject to runtime scheduling and cooperative synchronous work.
/// The timer does not cancel/recreate the transfer or renew its deadline.
async fn drive_attempt<T>(
    cx: &Cx,
    part: usize,
    mirror: usize,
    timeout: Duration,
    future: impl Future<Output = Result<T, NetworkPullError>>,
) -> Result<T, NetworkPullError> {
    checkpoint(cx, Some(part))?;
    let parent_deadline = cx.budget().deadline;
    let deadline = cx.budget_for_timeout(timeout).deadline
        .ok_or(NetworkPullError::InvalidOptions("mirror attempt requires a deadline"))?;
    let parent_limited = parent_deadline.is_some_and(|parent| parent <= deadline);
    let expired = || if parent_limited {
        NetworkPullError::Cancelled { cause: Some(CancelKind::Deadline), part: Some(part) }
    } else {
        NetworkPullError::RequestTimeout { part, mirror }
    };
    if deadline <= cx.now() { return Err(expired()); }
    let result = {
        let mut attempt = std::pin::pin!(future);
        let mut cancellation = std::pin::pin!(async {
            loop {
                asupersync::time::sleep(cx.now(), CANCELLATION_CHECK_INTERVAL).await;
                if let Err(error) = checkpoint(cx, Some(part)) { break error; }
            }
        });
        let observed = poll_fn(|task_cx| {
            if let Poll::Ready(error) = cancellation.as_mut().poll(task_cx) {
                return Poll::Ready(Err(error));
            }
            attempt.as_mut().poll(task_cx)
        });
        asupersync::time::timeout_at(deadline, observed).await
        // Both owned futures are dropped here, before rollback or retry.
    };
    // Preserve any real parent cancellation, including its precise kind.
    // Synchronous file work is cooperative; detect a late ready result too.
    checkpoint(cx, Some(part))?;
    match result {
        Ok(result) if cx.now() <= deadline => result,
        _ => Err(expired()),
    }
}

fn exhausted_mirrors(part: usize, attempts: usize, last: Option<NetworkPullError>) -> NetworkPullError {
    match last {
        // Keep the typed terminal timeout for the CLI's budget exit code.
        // An earlier timeout remains retryable when another mirror succeeds.
        Some(error @ NetworkPullError::RequestTimeout { .. }) => error,
        last => NetworkPullError::MirrorExhausted { part, attempts,
            last: last.map_or_else(|| "no mirror attempted".to_owned(), |error| error.to_string()) },
    }
}

fn next_range_end(
    part: usize, start: u64, part_bytes: u64, artifact_bytes: u64,
    limit: Option<u64>, downloaded_bytes: u64,
) -> Result<u64, NetworkPullError> {
    let remaining = part_bytes.checked_sub(start).filter(|&bytes| bytes != 0)
        .ok_or(NetworkPullError::InvalidOptions("invalid incomplete part range"))?;
    let length = limit.map_or(remaining, |limit| remaining.min(limit.saturating_sub(downloaded_bytes)));
    if length == 0 {
        return Err(NetworkPullError::DownloadBudgetExceeded {
            limit: limit.unwrap_or(0), downloaded_bytes, retained_bytes: start,
            total_bytes: artifact_bytes, part,
        });
    }
    // length <= part_bytes - start, so this addition cannot overflow.
    Ok(start + length - 1)
}

/// Complete or roll back one real transfer, including its digest gate.
fn finish_attempt(
    index: usize, mirror: usize, part: &ReleasePart, start: u64, end: u64,
    file: &mut File, hasher: &Sha256, result: Result<u64, NetworkPullError>,
) -> Result<u64, NetworkPullError> {
    let checked = result.and_then(|received| {
        if end.checked_sub(start).and_then(|bytes| bytes.checked_add(1)) != Some(received) {
            return Err(NetworkPullError::Response { part: index, mirror, detail: "body length differs from requested range" });
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
        file.set_len(start).map_err(|error| staging_error(index, "rollback-attempt", error))?;
        file.seek(SeekFrom::Start(start)).map_err(|error| staging_error(index, "seek-after-rollback", error))?;
        file.sync_all().map_err(|error| staging_error(index, "sync-rollback", error))?;
    }
    checked
}

fn charge_body_bytes(downloaded: &mut u64, bytes: usize) -> Result<(), NetworkPullError> {
    let bytes = u64::try_from(bytes)
        .map_err(|_| NetworkPullError::InvalidOptions("body byte length does not fit u64"))?;
    *downloaded = downloaded.checked_add(bytes)
        .ok_or(NetworkPullError::InvalidOptions("downloaded byte arithmetic overflow"))?;
    Ok(())
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
    downloaded_total: &mut u64,
    options: NetworkPartsOptions,
) -> Result<u64, NetworkPullError> {
    let redirect_policy = MirrorRedirectPolicy::new(initial_url)
        .map_err(|detail| NetworkPullError::Redirect { part, mirror, detail })?;
    let mut url = initial_url.to_owned();
    for redirects in 0..=options.max_redirects {
        checkpoint(cx, Some(part))?;
        let response = redirect_policy.with_authorized_target(&url, |target| {
            client.request_streaming(
                cx,
                Method::Get,
                target,
                vec![
                    ("Range".to_owned(), format!("bytes={start}-{end}")),
                    ("Accept-Encoding".to_owned(), "identity".to_owned()),
                ],
                Vec::new(),
            )
        }).map_err(|detail| NetworkPullError::Redirect { part, mirror, detail })?.await;
        // A cancelled request must not become a retryable transport error.
        checkpoint(cx, Some(part))?;
        let response = response.map_err(|error| NetworkPullError::Body { part, mirror, detail: error.to_string() })?;
        if matches!(response.head.status, 301 | 302 | 303 | 307 | 308) {
            if redirects == options.max_redirects {
                return Err(NetworkPullError::Redirect { part, mirror, detail: "redirect limit exceeded" });
            }
            let location = unique_header(&response.head.headers, "location")
                .map_err(|_| NetworkPullError::Redirect { part, mirror, detail: "duplicate Location header" })?
                .ok_or(NetworkPullError::Redirect { part, mirror, detail: "redirect missing Location" })?;
            // Validate before copying Location or constructing the next
            // request. A redirect cannot grant authority for a later hop.
            url = redirect_policy.with_authorized_target(location, str::to_owned)
                .map_err(|detail| NetworkPullError::Redirect { part, mirror, detail })?;
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
                    // Charge the entire delivered frame before inspecting or
                    // writing it, including overlong bytes and failed writes.
                    charge_body_bytes(downloaded_total, data.remaining())?;
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
        checkpoint(cx, Some(part))?;
        if received != end - start + 1 {
            return Err(NetworkPullError::Response { part, mirror, detail: "body length differs from requested range" });
        }
        return Ok(received);
    }
    Err(NetworkPullError::Redirect { part, mirror, detail: "redirect state exhausted" })
}

/// Every hop remains bound to the independently pinned mirror's original host.
/// Public github.com release URLs may also use the exact documented GitHub
/// asset hosts. Custom mirrors retain same-host redirects; neither a sibling
/// mirror nor a CDN response can widen this policy.
struct MirrorRedirectPolicy {
    original_host: String,
}

impl MirrorRedirectPolicy {
    fn new(mirror: &str) -> Result<Self, &'static str> {
        Ok(Self { original_host: parse_download_target(mirror)?.host })
    }

    /// Construct transport work only after validating the exact URL that is
    /// passed to it. The callback also lets model-free tests prove refusal
    /// happens before any request is constructed; retaining a Location uses
    /// the same guard after its bounded transport parse.
    fn with_authorized_target<'url, T>(
        &self,
        url: &'url str,
        send: impl FnOnce(&'url str) -> T,
    ) -> Result<T, &'static str> {
        let target = parse_download_target(url)?;
        let same_host = target.host.eq_ignore_ascii_case(&self.original_host);
        let github_asset = self.original_host.eq_ignore_ascii_case("github.com")
            && GITHUB_RELEASE_REDIRECT_HOSTS.iter()
                .any(|host| target.host.eq_ignore_ascii_case(host));
        if !same_host && !github_asset {
            return Err("target host is outside the original mirror's redirect allowlist");
        }
        Ok(send(url))
    }
}

/// Use the transport's parser, with the product's stricter URL authority
/// policy applied first. The selected asupersync pin permits explicit ports
/// and retains fragments in ParsedUrl::path; neither is allowed here. Errors
/// deliberately omit the URL, whose signed query may contain credentials.
fn parse_download_target(url: &str) -> Result<ParsedUrl, &'static str> {
    if url.len() > MAX_DOWNLOAD_URL_BYTES || !url.is_ascii()
        || url.bytes().any(|byte| byte <= b' ' || byte == 0x7f || matches!(byte, b'\\' | b'#'))
    {
        return Err("target must be bounded ASCII without whitespace, controls, backslashes or fragments");
    }
    let rest = url.strip_prefix("https://")
        .ok_or("target must be an absolute HTTPS URL")?;
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    if authority.is_empty() || authority.len() > 253
        || authority.split('.').any(|label| label.is_empty() || label.len() > 63
            || label.starts_with('-') || label.ends_with('-')
            || !label.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'))
    {
        return Err("target authority must be a hostname without credentials, ports or escapes");
    }
    let parsed = ParsedUrl::parse(url).map_err(|_| "target URL is invalid for the selected transport")?;
    if parsed.scheme != Scheme::Https || parsed.port != 443 || parsed.host != authority {
        return Err("target authority disagrees with the selected HTTPS transport");
    }
    Ok(parsed)
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

fn verify_final(part: usize, expected: &ReleasePart, path: &Path,
    mut check: impl FnMut() -> Result<(), NetworkPullError>) -> Result<(), NetworkPullError> {
    check()?;
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
        check()?;
        let read = file.read(&mut buffer).map_err(|_| NetworkPullError::ExistingPart { part, detail: "cannot hash final" })?;
        if read == 0 { break; }
        hasher.update(&buffer[..read]);
    }
    check()?;
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
    use std::{cell::Cell, sync::Arc, task::{Context, Waker}};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ATTEMPT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct AttemptDrop<'a>(&'a Cell<bool>);
    impl Drop for AttemptDrop<'_> {
        fn drop(&mut self) { self.0.set(true); }
    }

    fn virtual_attempt_runtime() -> (asupersync::runtime::Runtime, Arc<asupersync::time::VirtualClock>) {
        let clock = Arc::new(asupersync::time::VirtualClock::starting_at(
            asupersync::types::Time::from_secs(17),
        ));
        let driver = asupersync::time::TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
            .with_timer_driver(driver).build().unwrap();
        (runtime, clock)
    }

    fn attempt_path() -> PathBuf {
        let sequence = NEXT_ATTEMPT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "fnlp-network-attempt-{}-{sequence}", std::process::id(),
        ))
    }

    fn attempt_fixture(prefix: &[u8], complete: &[u8]) -> (File, ReleasePart, Sha256) {
        let mut file = OpenOptions::new().read(true).write(true).create_new(true)
            .open(attempt_path()).expect("create unique attempt fixture");
        file.write_all(prefix).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(prefix);
        let part = ReleasePart { id: 0, name: "fixture.part00".to_owned(),
            bytes: complete.len() as u64, sha256: hex_lower(&Sha256::digest(complete)),
            mirrors: vec!["https://first.invalid/part".to_owned(), "https://second.invalid/part".to_owned()] };
        (file, part, hasher)
    }

    #[test]
    fn stalled_body_times_out_drops_the_attempt_and_rolls_back_without_refunding_bytes() {
        use std::sync::atomic::AtomicBool;
        struct DropSignal<'a>(&'a AtomicBool);
        impl Drop for DropSignal<'_> {
            fn drop(&mut self) { self.0.store(true, Ordering::SeqCst); }
        }
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread().build().unwrap();
        let (mut file, part, mut hasher) = attempt_fixture(b"ab", b"abcdef");
        let mut downloaded = 0;
        file.write_all(b"cd").unwrap(); hasher.update(b"cd");
        charge_body_bytes(&mut downloaded, 2).unwrap();
        let dropped = AtomicBool::new(false);
        let guard = DropSignal(&dropped);
        let result = runtime.block_on(async {
            let cx = Cx::current().unwrap();
            drive_attempt(&cx, 0, 0, Duration::from_millis(100), async move {
                let _guard = guard;
                std::future::pending::<Result<u64, NetworkPullError>>().await
            }).await
        });
        assert!(dropped.load(Ordering::SeqCst), "the timed-out attempt must be dropped before retry");
        assert!(matches!(result, Err(NetworkPullError::RequestTimeout { part: 0, mirror: 0 })));
        assert!(matches!(finish_attempt(0, 0, &part, 2, 5, &mut file, &hasher, result),
            Err(NetworkPullError::RequestTimeout { .. })));
        assert_eq!(file.metadata().unwrap().len(), 2);
        assert_eq!(file.stream_position().unwrap(), 2);
        assert_eq!(downloaded, 2, "timeout never refunds consumed network bytes");
        assert_eq!(next_range_end(0, 2, 6, 6, Some(4), downloaded).unwrap(), 3);
    }

    #[test]
    fn silent_attempt_observes_each_cancellation_kind_at_the_next_timer_check() {
        assert_eq!(CANCELLATION_CHECK_INTERVAL, Duration::from_millis(100));
        for cause in [CancelKind::User, CancelKind::Timeout, CancelKind::Deadline,
            CancelKind::PollQuota, CancelKind::CostBudget, CancelKind::FailFast,
            CancelKind::RaceLost, CancelKind::ParentCancelled, CancelKind::ResourceUnavailable,
            CancelKind::Shutdown, CancelKind::LinkedExit] {
            let (runtime, clock) = virtual_attempt_runtime();
            runtime.block_on(async {
                let cx = Cx::current().unwrap();
                assert_eq!(cx.now(), asupersync::types::Time::from_secs(17));
                let dropped = Cell::new(false);
                let mut task = Context::from_waker(Waker::noop());
                let mut attempt = std::pin::pin!(drive_attempt(&cx, 2, 1, Duration::from_secs(60), async {
                    let _guard = AttemptDrop(&dropped);
                    std::future::pending::<Result<u64, NetworkPullError>>().await
                }));
                assert!(attempt.as_mut().poll(&mut task).is_pending());
                cx.cancel_with(cause, None);
                clock.advance(50_000_000);
                assert!(attempt.as_mut().poll(&mut task).is_pending());
                assert!(!dropped.get(), "pending timer checks must retain the transfer");
                clock.advance(50_000_000);
                let result = match attempt.as_mut().poll(&mut task) {
                    Poll::Ready(result) => result,
                    Poll::Pending => panic!("silent cancellation must not wait for the 60 second deadline"),
                };
                assert!(matches!(result, Err(NetworkPullError::Cancelled { cause: Some(observed), part: Some(2) })
                    if observed == cause));
                assert!(dropped.get(), "cancellation must release the owned transfer before returning");
            });
        }
    }

    #[test]
    fn silent_cancellation_rolls_back_without_refunding_received_bytes() {
        let (runtime, clock) = virtual_attempt_runtime();
        let (mut file, part, mut hasher) = attempt_fixture(b"ab", b"abcdef");
        let mut downloaded = 0;
        let dropped = Cell::new(false);
        let result = runtime.block_on(async {
            let cx = Cx::current().unwrap();
            assert_eq!(cx.now(), asupersync::types::Time::from_secs(17));
            let mut task = Context::from_waker(Waker::noop());
            let mut attempt = std::pin::pin!(drive_attempt(&cx, 0, 0, Duration::from_secs(60), async {
                let _guard = AttemptDrop(&dropped);
                file.write_all(b"cd").unwrap(); hasher.update(b"cd");
                charge_body_bytes(&mut downloaded, 2).unwrap();
                std::future::pending::<Result<u64, NetworkPullError>>().await
            }));
            assert!(attempt.as_mut().poll(&mut task).is_pending());
            cx.cancel_with(CancelKind::User, None);
            clock.advance(100_000_000);
            match attempt.as_mut().poll(&mut task) {
                Poll::Ready(result) => result,
                Poll::Pending => panic!("cancelled transfer must finish on the cancellation timer"),
            }
        });
        assert!(dropped.get());
        assert!(matches!(finish_attempt(0, 0, &part, 2, 5, &mut file, &hasher, result),
            Err(NetworkPullError::Cancelled { cause: Some(CancelKind::User), part: Some(0) })));
        assert_eq!(file.metadata().unwrap().len(), 2);
        assert_eq!(file.stream_position().unwrap(), 2);
        assert_eq!(downloaded, 2, "cancellation never refunds consumed network bytes");
        assert_eq!(next_range_end(0, 2, 6, 6, Some(4), downloaded).unwrap(), 3);
    }

    #[test]
    fn cancellation_ticks_keep_one_transfer_and_never_renew_the_attempt_deadline() {
        let (runtime, clock) = virtual_attempt_runtime();
        runtime.block_on(async {
            let cx = Cx::current().unwrap();
            assert_eq!(cx.now(), asupersync::types::Time::from_secs(17));
            let starts = Cell::new(0);
            let polls = Cell::new(0);
            let dropped = Cell::new(false);
            let mut task = Context::from_waker(Waker::noop());
            let mut attempt = std::pin::pin!(drive_attempt(&cx, 0, 0, Duration::from_millis(250), async {
                starts.set(starts.get() + 1);
                let _guard = AttemptDrop(&dropped);
                poll_fn(|_| {
                    polls.set(polls.get() + 1);
                    if polls.get() == 4 { Poll::Ready(Ok(4_u64)) } else { Poll::Pending }
                }).await
            }));
            assert!(attempt.as_mut().poll(&mut task).is_pending());
            for _ in 0..2 {
                clock.advance(100_000_000);
                assert!(attempt.as_mut().poll(&mut task).is_pending());
                assert_eq!(starts.get(), 1);
                assert!(!dropped.get(), "a timer tick must not drop or restart the transfer");
            }
            clock.advance(50_000_001);
            assert!(matches!(attempt.as_mut().poll(&mut task),
                Poll::Ready(Err(NetworkPullError::RequestTimeout { part: 0, mirror: 0 }))));
            assert_eq!(polls.get(), 3, "expired work must not receive a late completion poll");
            assert!(dropped.get());
        });
    }

    #[test]
    fn complete_attempt_preserves_the_result_and_transport_error() {
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread().build().unwrap();
        runtime.block_on(async {
            let cx = Cx::current().unwrap();
            assert_eq!(drive_attempt(&cx, 2, 3, Duration::from_secs(60), async { Ok(17_u64) }).await.unwrap(), 17);
            let result = drive_attempt(&cx, 2, 3, Duration::from_secs(60), async {
                Err::<u64, _>(NetworkPullError::Response { part: 2, mirror: 3, detail: "invalid range" })
            }).await;
            assert!(matches!(result, Err(NetworkPullError::Response { part: 2, mirror: 3, .. })));
        });
    }

    #[test]
    fn an_expired_attempt_does_not_start_request_or_body_work() {
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread().build().unwrap();
        runtime.block_on(async {
            let cx = Cx::current().unwrap();
            let result = drive_attempt(&cx, 1, 2, Duration::ZERO, async {
                panic!("expired work must never be polled");
                #[allow(unreachable_code)] Ok::<u64, NetworkPullError>(0)
            }).await;
            assert!(matches!(result, Err(NetworkPullError::RequestTimeout { part: 1, mirror: 2 })));
        });
    }

    #[test]
    fn parent_cancellation_wins_over_a_ready_attempt_without_losing_its_kind() {
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread().build().unwrap();
        for cause in [CancelKind::User, CancelKind::Timeout, CancelKind::Deadline,
            CancelKind::PollQuota, CancelKind::CostBudget, CancelKind::FailFast,
            CancelKind::RaceLost, CancelKind::ParentCancelled, CancelKind::ResourceUnavailable,
            CancelKind::Shutdown, CancelKind::LinkedExit] {
            runtime.block_on(async {
                let cx = Cx::current().unwrap();
                let result = drive_attempt(&cx, 2, 1, Duration::from_secs(60), async {
                    cx.cancel_with(cause, None);
                    Ok(4_u64)
                }).await;
                assert!(matches!(result, Err(NetworkPullError::Cancelled { cause: Some(observed), part: Some(2) })
                    if observed == cause));
            });
        }
    }

    #[test]
    fn exhausted_mirror_timeout_retains_its_typed_budget_outcome() {
        assert!(matches!(exhausted_mirrors(2, 4,
            Some(NetworkPullError::RequestTimeout { part: 2, mirror: 3 })),
            NetworkPullError::RequestTimeout { part: 2, mirror: 3 }));
        assert!(matches!(exhausted_mirrors(2, 4,
            Some(NetworkPullError::Response { part: 2, mirror: 3, detail: "invalid range" })),
            NetworkPullError::MirrorExhausted { part: 2, attempts: 4, .. }));
    }

    #[test]
    fn corrupt_complete_mirror_rolls_back_before_a_valid_mirror_succeeds() {
        let (mut file, part, prefix) = attempt_fixture(b"ab", b"abcdef");
        let mut downloaded = 0;
        let mut attempted = Vec::new();
        for (mirror, bytes) in [b"XXXX".as_slice(), b"cdef"].into_iter().enumerate() {
            attempted.push(mirror);
            let end = next_range_end(0, 2, part.bytes, part.bytes, Some(8), downloaded).unwrap();
            let mut hasher = prefix.clone();
            charge_body_bytes(&mut downloaded, bytes.len()).unwrap();
            file.write_all(bytes).unwrap(); hasher.update(bytes);
            match finish_attempt(0, mirror, &part, 2, end, &mut file, &hasher, Ok(bytes.len() as u64)) {
                Ok(received) => { assert_eq!(received, 4); break; }
                Err(NetworkPullError::Digest { .. }) => {
                    assert_eq!(mirror, 0);
                    assert_eq!(file.metadata().unwrap().len(), 2);
                    assert_eq!(file.stream_position().unwrap(), 2);
                    assert_eq!(downloaded, 4, "rollback never refunds transferred bytes");
                }
                Err(error) => panic!("unexpected failure: {error}"),
            }
        }
        assert_eq!(attempted, [0, 1]);
        assert_eq!(downloaded, 8);
        file.rewind().unwrap();
        let mut bytes = Vec::new(); file.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"abcdef");
    }

    #[test]
    fn a_failed_attempt_shrinks_the_next_range_without_refunding_its_bytes() {
        let (mut file, part, prefix) = attempt_fixture(b"ab", b"abcdef");
        let mut downloaded = 0;
        let end = next_range_end(0, 2, part.bytes, 12, Some(5), downloaded).unwrap();
        assert_eq!(end, 5);
        let mut hasher = prefix.clone();
        charge_body_bytes(&mut downloaded, 3).unwrap();
        file.write_all(b"bad").unwrap(); hasher.update(b"bad");
        let failure = NetworkPullError::Body { part: 0, mirror: 0, detail: "interrupted".to_owned() };
        assert!(matches!(finish_attempt(0, 0, &part, 2, end, &mut file, &hasher, Err(failure)),
            Err(NetworkPullError::Body { .. })));
        assert_eq!(file.metadata().unwrap().len(), 2);
        let end = next_range_end(0, 2, part.bytes, 12, Some(5), downloaded).unwrap();
        assert_eq!(end, 3, "only two body bytes remain for the next mirror");
        let mut hasher = prefix;
        charge_body_bytes(&mut downloaded, 2).unwrap();
        file.write_all(b"cd").unwrap(); hasher.update(b"cd");
        assert_eq!(finish_attempt(0, 1, &part, 2, end, &mut file, &hasher, Ok(2)).unwrap(), 2);
        assert_eq!(file.metadata().unwrap().len(), 4, "a valid partial is retained for explicit resume");
        assert!(matches!(next_range_end(0, 4, part.bytes, 12, Some(5), downloaded),
            Err(NetworkPullError::DownloadBudgetExceeded {
                limit: 5, downloaded_bytes: 5, retained_bytes: 4, total_bytes: 12, part: 0,
            })));
    }

    #[test]
    fn corrupt_body_cannot_buy_another_attempt_with_an_exhausted_budget() {
        let (mut file, part, mut hasher) = attempt_fixture(b"ab", b"abcdef");
        let mut downloaded = 0;
        charge_body_bytes(&mut downloaded, 4).unwrap();
        file.write_all(b"bad!").unwrap(); hasher.update(b"bad!");
        assert!(matches!(finish_attempt(0, 0, &part, 2, 5, &mut file, &hasher, Ok(4)),
            Err(NetworkPullError::Digest { .. })));
        assert!(matches!(next_range_end(0, 2, part.bytes, part.bytes, Some(4), downloaded),
            Err(NetworkPullError::DownloadBudgetExceeded { downloaded_bytes: 4, retained_bytes: 2, .. })));
        assert_eq!(file.metadata().unwrap().len(), 2);
    }

    #[test]
    fn short_overlong_cancelled_and_storage_failed_attempts_restore_the_prefix() {
        for kind in 0..4 {
            let (mut file, part, mut hasher) = attempt_fixture(b"ab", b"abcdef");
            let bytes = if kind == 1 { b"cdefEXTRA".as_slice() } else { b"cd".as_slice() };
            let mut downloaded = 0;
            charge_body_bytes(&mut downloaded, bytes.len()).unwrap();
            file.write_all(bytes).unwrap(); hasher.update(bytes);
            let result = match kind {
                0 | 1 => Ok(bytes.len() as u64),
                2 => Err(NetworkPullError::Cancelled { cause: Some(CancelKind::Deadline), part: Some(0) }),
                _ => Err(NetworkPullError::Staging { part: 0, operation: "write-range", detail: "full".to_owned() }),
            };
            let error = finish_attempt(0, 0, &part, 2, 5, &mut file, &hasher, result).unwrap_err();
            match kind {
                0 | 1 => assert!(matches!(error, NetworkPullError::Response { .. })),
                2 => assert!(matches!(error, NetworkPullError::Cancelled { cause: Some(CancelKind::Deadline), .. })),
                _ => assert!(matches!(error, NetworkPullError::Staging { operation: "write-range", .. })),
            }
            assert_eq!(file.metadata().unwrap().len(), 2);
            assert_eq!(file.stream_position().unwrap(), 2);
            assert_eq!(downloaded, bytes.len() as u64);
            file.rewind().unwrap();
            let mut prefix = Vec::new(); file.read_to_end(&mut prefix).unwrap();
            assert_eq!(prefix, b"ab");
        }
    }

    #[test]
    fn transfer_budget_and_range_arithmetic_refuse_overflow_and_empty_ranges() {
        assert_eq!(next_range_end(0, u64::MAX - 2, u64::MAX, u64::MAX, None, 0).unwrap(), u64::MAX - 1);
        assert_eq!(next_range_end(0, 5, 10, 10, Some(1), 0).unwrap(), 5);
        for (start, total) in [(0, 0), (10, 10), (11, 10)] {
            assert!(matches!(next_range_end(0, start, total, total, None, 0), Err(NetworkPullError::InvalidOptions(_))));
        }
        for downloaded in [4, 5, u64::MAX] {
            assert!(matches!(next_range_end(0, 2, 6, 6, Some(4), downloaded),
                Err(NetworkPullError::DownloadBudgetExceeded { .. })));
        }
        let mut downloaded = u64::MAX;
        assert!(charge_body_bytes(&mut downloaded, 1).is_err());
        assert_eq!(downloaded, u64::MAX);
    }

    #[test]
    fn cached_part_verification_checks_cancellation_before_open_and_between_reads() {
        let path = attempt_path();
        let bytes = vec![b'x'; COPY_BUFFER_BYTES + 1];
        let part = ReleasePart { id: 0, name: "fixture.part00".to_owned(), bytes: bytes.len() as u64,
            sha256: hex_lower(&Sha256::digest(&bytes)), mirrors: Vec::new() };
        let cancelled = || NetworkPullError::Cancelled { cause: Some(CancelKind::Deadline), part: Some(0) };
        assert!(matches!(verify_final(0, &part, &path, || Err(cancelled())),
            Err(NetworkPullError::Cancelled { cause: Some(CancelKind::Deadline), .. })));
        let mut file = OpenOptions::new().write(true).create_new(true).open(&path).unwrap();
        file.write_all(&bytes).unwrap(); file.sync_all().unwrap(); drop(file);
        let mut checks = 0;
        assert!(matches!(verify_final(0, &part, &path, || {
            checks += 1;
            if checks == 3 { Err(cancelled()) } else { Ok(()) }
        }), Err(NetworkPullError::Cancelled { .. })));
        assert_eq!(checks, 3, "cancellation is observed before hashing the second block");
        assert!(verify_final(0, &part, &path, || Ok(())).is_ok());
    }

    #[test]
    fn custom_mirror_redirects_keep_the_original_host_without_granting_github_authority() {
        let policy = MirrorRedirectPolicy::new("https://mirror.example/releases/download/v1/part00").unwrap();
        for target in [
            "https://mirror.example/download?sig=unchanged",
            "https://MIRROR.EXAMPLE/signed/part00",
            "https://mirror.example",
        ] {
            assert_eq!(policy.with_authorized_target(target, |url| url).unwrap(), target);
        }
        for target in [
            "https://other-mirror.example/part00",
            "https://child.mirror.example/part00",
            "https://mirror.example.attacker.invalid/part00",
            "https://github.com/owner/repo/releases/download/v1/part00",
            "https://release-assets.githubusercontent.com/part00",
        ] {
            assert!(policy.with_authorized_target(target, |_| {
                panic!("an unapproved redirect must not construct transport work");
            }).is_err());
        }
    }

    #[test]
    fn github_release_redirects_match_exact_hosts_and_remain_bound_across_hops() {
        let policy = MirrorRedirectPolicy::new("https://GitHub.com/owner/repo/releases/download/v1/part00").unwrap();
        let mut contacted = Vec::new();
        for target in [
            "https://github.com/owner/renamed/releases/download/v1/part00",
            "https://release-assets.githubusercontent.com/signed/part00?sig=one",
            "https://OBJECTS.GITHUBUSERCONTENT.COM/signed/part00?sig=two",
            "https://github-releases.githubusercontent.com/signed/part00?sig=three",
        ] {
            policy.with_authorized_target(target, |url| contacted.push(url.to_owned())).unwrap();
        }
        assert_eq!(contacted.len(), 4);
        for target in [
            "https://release-assets.githubusercontent.com.attacker.invalid/part00",
            "https://child.release-assets.githubusercontent.com/part00",
            "https://attacker-release-assets.githubusercontent.com/part00",
            "https://github.com.attacker.invalid/part00",
            "https://attacker.invalid/release-assets.githubusercontent.com/part00",
            "https://attacker.invalid/part00?next=https://github.com",
            "https://storage.blob.core.windows.net/part00",
            "https://bucket.s3.amazonaws.com/part00",
        ] {
            assert!(policy.with_authorized_target(target, |url| contacted.push(url.to_owned())).is_err());
        }
        assert_eq!(contacted.len(), 4, "an allowed CDN hop cannot broaden the original policy");

        // A custom mirror directly naming one asset host gains only that host,
        // not the github.com mirror's explicit set of cross-host permissions.
        let asset = MirrorRedirectPolicy::new("https://objects.githubusercontent.com/part00").unwrap();
        assert!(asset.with_authorized_target("https://release-assets.githubusercontent.com/part00", |_| {
            panic!("a redirected or independently named CDN cannot expand authority");
        }).is_err());
    }

    #[test]
    fn malformed_initial_and_redirect_targets_are_refused_before_transport() {
        let policy = MirrorRedirectPolicy::new("https://github.com/owner/repo/releases/download/v1/part00").unwrap();
        let attempts = Cell::new(0);
        for target in [
            "", "http://github.com/part00", "ftp://github.com/part00",
            "HTTPS://github.com/part00", "//github.com/part00", "/part00",
            "https://", "https:///part00",
            "https://user@github.com/part00", "https://user:secret@github.com/part00",
            "https://github.com@attacker.invalid/part00",
            "https://github.com:443/part00", "https://github.com:0443/part00",
            "https://github.com:0/part00", "https://github.com:65536/part00",
            "https://[::1]/part00", "https://github.com%2Eattacker.invalid/part00",
            "https://github.com%2f@attacker.invalid/part00",
            "https://github.com\\@attacker.invalid/part00",
            "https://github.com/part00\\next", "https://github.com/part00#fragment",
            "https://github.com /part00", "https://github.com/part 00",
            "https://github.com/part00?sig=one\r\nHost:attacker.invalid",
            "https://github.com/part00?sig=one\t", "https://github.com/part00?sig=one\x7f",
            "https://gıthub.com/part00", "https://github..com/part00",
            "https://github.com./part00", "https://-github.com/part00",
            "https://github-.com/part00",
        ] {
            assert!(MirrorRedirectPolicy::new(target).is_err(), "initial target: {target:?}");
            assert!(policy.with_authorized_target(target, |_| attempts.set(attempts.get() + 1)).is_err(),
                "redirect target: {target:?}");
        }
        assert_eq!(attempts.get(), 0, "invalid Location must be refused before invoking transport");
    }

    #[test]
    fn signed_redirect_paths_and_queries_are_forwarded_without_normalization() {
        let policy = MirrorRedirectPolicy::new("https://github.com/owner/repo/releases/download/v1/part00").unwrap();
        let target = concat!(
            "https://release-assets.githubusercontent.com/github-production-release-asset/123/part%2F00",
            "?sp=r&sv=2018-11-09&sig=a%2Fb+%3D&jwt=header.payload.signature",
            "&response-content-disposition=attachment%3B%20filename%3Dpart00",
            "&redirect_uri=https%3A%2F%2Fgithub.com%2Fasset&literal=one@two:three%23four",
        );
        let mut observed = String::new();
        policy.with_authorized_target(target, |url| observed.push_str(url)).unwrap();
        assert_eq!(observed, target);
        let parsed = ParsedUrl::parse(&observed).unwrap();
        assert_eq!(parsed.scheme, Scheme::Https);
        assert_eq!(parsed.port, 443);
        assert_eq!(parsed.host, "release-assets.githubusercontent.com");
        assert_eq!(parsed.path, target.strip_prefix("https://release-assets.githubusercontent.com").unwrap());
    }

    #[test]
    fn oversized_redirects_and_hostnames_are_rejected_before_request_allocation() {
        let policy = MirrorRedirectPolicy::new("https://github.com/owner/repo/releases/download/v1/part00").unwrap();
        let prefix = "https://release-assets.githubusercontent.com/part00?sig=";
        let exact = format!("{prefix}{}", "a".repeat(MAX_DOWNLOAD_URL_BYTES - prefix.len()));
        assert_eq!(policy.with_authorized_target(&exact, str::len).unwrap(), MAX_DOWNLOAD_URL_BYTES);
        let oversized = format!("{exact}a");
        let attempts = Cell::new(0);
        assert!(policy.with_authorized_target(&oversized, |_| attempts.set(1)).is_err());
        let long_label = format!("https://{}.example/part00", "a".repeat(64));
        assert!(MirrorRedirectPolicy::new(&long_label).is_err());
        let long_host = format!("https://{0}.{0}.{0}.{0}/part00", "a".repeat(63));
        assert!(MirrorRedirectPolicy::new(&long_host).is_err());
        assert_eq!(attempts.get(), 0);
    }

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
