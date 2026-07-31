//! Canonical release-manifest contract and pre-network compatibility gate.
//!
//! This module deliberately owns only small, authority-bearing metadata.  It
//! does not open files, create cache paths, or make HTTP requests: callers must
//! successfully parse and validate a [`ReleaseManifest`] before they are
//! allowed to hand any member to the streamed pull implementation.

use std::{collections::BTreeSet, error::Error, fmt, str};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::canonjson::{self, ParseLimits};

/// The only release manifest schema accepted by the v1 pull protocol.
pub const RELEASE_MANIFEST_SCHEMA_V1: &str = "fnlp-release-manifest-v1";
/// The manifest format that compatible v1 binaries understand.
pub const PULL_API_V1: &str = "fnlp-pull-v1";
/// The only current `.fnlpq` format a release manifest may name.
pub const FNLPQ_FORMAT_V1: &str = "fnlpq-v1";
/// Native packs are derived locally and never downloaded as an alternative
/// authority root.
pub const LOCAL_DERIVATION_PACKING_POLICY_V1: &str = "derive-local-v1";

/// The bounded v1 maximum for the complete canonical manifest.
pub const MAX_RELEASE_MANIFEST_BYTES: usize = 256 * 1024;
/// The v1 package protocol uses two decimal part digits.
pub const MAX_RELEASE_MANIFEST_PARTS: usize = 64;
/// A release manifest allows a small, deterministic ordered mirror list.
pub const MAX_PART_MIRRORS: usize = 4;
/// The pinned source-conversion closure currently has ten files; leave a
/// bounded allowance for a future versioned closure rather than accepting an
/// unbounded authority document.
pub const MAX_SOURCE_FILES: usize = 64;
/// No URL in the pre-network manifest may be larger than this bound.
pub const MAX_RELEASE_URL_BYTES: usize = 2048;
/// Limit identifiers separately so arbitrary long strings cannot move through
/// the manifest boundary as model or compatibility labels.
pub const MAX_IDENTIFIER_BYTES: usize = 160;

const RELEASE_MANIFEST_DIGEST_DOMAIN: &[u8] = b"franken_nlp/release-manifest/v1\0";
const RELEASE_PART_BYTES: u64 = 1_957_046_720;

/// Canonical, release-bound metadata for one immutable Generic artifact.
///
/// The field names are part of the published v1 JSON contract.  Dynamic maps
/// are intentionally absent: every authority-bearing field is typed and
/// unknown keys are rejected during parsing.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseManifest {
    /// Closed schema label for this manifest document.
    pub manifest_schema: String,
    /// The only model identity this manifest authorizes.
    pub model_id: String,
    /// Immutable Hugging Face source revision for the conversion closure.
    pub source_revision: String,
    /// Artifact-version identity independent of binary SemVer.
    pub artifact_id: String,
    /// Immutable GitHub release tag, never a mutable branch or alias.
    pub release_tag: String,
    /// Logical canonical Generic artifact identity.
    pub artifact: ReleaseArtifact,
    /// Ordered, fixed-size release members used for streamed reconstruction.
    pub parts: Vec<ReleasePart>,
    /// Ordered conversion-source closure and its root digest.
    pub source: ReleaseSourceClosure,
    /// Exact recipe, converter, container, and pull API compatibility rows.
    pub compatibility: ReleaseCompatibility,
    /// Runtime-visible materialized and logical identities.
    pub digests: ReleaseDigests,
    /// Digest of the three-file Apache attribution/notice bundle.
    pub license_bundle_sha256: String,
    /// Native-packing disposition for the downloaded Generic root.
    pub packing_policy: String,
    /// Explicit revocation or supersession state; revocation never rewrites
    /// bytes under the immutable release tag.
    pub lifecycle: ReleaseLifecycle,
}

/// Canonical reconstructed artifact name, length, and exact byte digest.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseArtifact {
    /// Portable basename of the Generic `.fnlpq` file.
    pub logical_name: String,
    /// Checked reconstructed byte length.
    pub bytes: u64,
    /// SHA-256 of the complete exact artifact byte stream.
    pub sha256: String,
}

/// One ordered `.partNN` object and its immutable mirror URLs.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReleasePart {
    /// Zero-based ordinal, required to equal this record's array position.
    pub id: u32,
    /// Exact `.partNN` basename.
    pub name: String,
    /// Exact member length.
    pub bytes: u64,
    /// SHA-256 of the stored member bytes.
    pub sha256: String,
    /// Deterministically ordered, immutable HTTPS release-tag URLs.
    pub mirrors: Vec<String>,
}

/// The ordered conversion source closure bound into the release.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseSourceClosure {
    /// Source-file records in the same order used for the source-root digest.
    pub files: Vec<ReleaseSourceFile>,
    /// Digest of the ordered source closure.
    pub source_root_sha256: String,
}

/// One pinned source member used to produce the artifact.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseSourceFile {
    /// Portable source-member basename.
    pub name: String,
    /// Exact source-member length.
    pub bytes: u64,
    /// SHA-256 of the exact source member.
    pub sha256: String,
}

/// Compatibility inputs that are rejected before any cache or network action.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseCompatibility {
    /// Immutable quantization/conversion recipe identifier.
    pub recipe_id: String,
    /// Versioned converter identity that produced the canonical artifact.
    pub converter_id: String,
    /// `.fnlpq` reader/writer format contract.
    pub fnlpq_format: String,
    /// Pull-manager protocol contract.
    pub pull_api: String,
}

/// Digests that make the model and its materialized bytes unambiguous.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseDigests {
    /// Embedded tokenizer-model bytes.
    pub tokenizer_sha256: String,
    /// Canonical template bytes.
    pub template_sha256: String,
    /// 201-name tensor census identity.
    pub census_sha256: String,
    /// Logical model identity, independent of physical packing placement.
    pub logical_model_sha256: String,
    /// Declared packing-set identity.
    pub packing_set_sha256: String,
}

/// Honest immutable-release status.  Old manifests remain readable after a
/// supersession or revocation because an offline binary may still embed one.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseLifecycle {
    /// `active`, `superseded`, or `revoked`.
    pub state: String,
    /// Required for `superseded`; optional for `revoked`; absent for `active`.
    pub superseded_by: Option<String>,
    /// Required for `revoked`; absent otherwise.
    pub revocation_reason: Option<String>,
}

/// A validated immutable embedded byte bundle supplied by a binary build.
///
/// The release workflow supplies these bytes from its exact model release;
/// this crate intentionally does not embed a fictional placeholder manifest.
/// A binary with no released model therefore exposes no default-pull authority
/// rather than silently trusting a mutable network catalog.
#[derive(Clone, Copy, Debug)]
pub struct EmbeddedReleaseManifest {
    bytes: &'static [u8],
    release_manifest_sha256: &'static str,
}

/// Typed, pointer-addressable release-manifest rejection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManifestError {
    /// Stable invariant label suitable for machine diagnostics.
    pub invariant: &'static str,
    /// JSON Pointer identifying the failed field, or `$` for the document.
    pub json_pointer: String,
    /// The expected contract value or relation.
    pub expected: String,
    /// The observed value or relation.
    pub observed: String,
}

impl ManifestError {
    fn new(
        invariant: &'static str,
        json_pointer: impl Into<String>,
        expected: impl Into<String>,
        observed: impl Into<String>,
    ) -> Self {
        Self {
            invariant,
            json_pointer: json_pointer.into(),
            expected: expected.into(),
            observed: observed.into(),
        }
    }
}

impl fmt::Display for ManifestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "release manifest invariant={} pointer={} expected={} observed={}",
            self.invariant, self.json_pointer, self.expected, self.observed
        )
    }
}

impl Error for ManifestError {}

impl ReleaseManifest {
    /// Parse one bounded canonical manifest and enforce all pre-network rules.
    pub fn parse(bytes: &[u8]) -> Result<Self, ManifestError> {
        if bytes.len() > MAX_RELEASE_MANIFEST_BYTES {
            return Err(ManifestError::new(
                "manifest-size-cap",
                "$",
                format!("at most {MAX_RELEASE_MANIFEST_BYTES} bytes"),
                format!("{} bytes", bytes.len()),
            ));
        }
        let text = str::from_utf8(bytes).map_err(|error| {
            ManifestError::new("manifest-utf8", "$", "valid UTF-8", error.to_string())
        })?;
        let limits = ParseLimits {
            max_depth: 16,
            max_string_bytes: MAX_RELEASE_URL_BYTES,
        };
        let canonical = canonjson::canonicalize_str(text, limits).map_err(|error| {
            ManifestError::new(
                "manifest-json",
                "$",
                "duplicate-key-free bounded JSON",
                error.to_string(),
            )
        })?;
        if canonical != bytes {
            return Err(ManifestError::new(
                "canonical-json",
                "$",
                "exact canonical JSON bytes",
                "different whitespace, key order, or escape spelling",
            ));
        }
        let value = canonjson::parse_str_with_limits(text, limits).map_err(|error| {
            ManifestError::new(
                "manifest-json",
                "$",
                "duplicate-key-free bounded JSON",
                error.to_string(),
            )
        })?;
        let manifest = serde_json::from_value(value).map_err(|error| {
            ManifestError::new(
                "manifest-schema",
                "$",
                "the fnlp release-manifest v1 field schema",
                error.to_string(),
            )
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Serialize this typed manifest as the sole accepted canonical byte form.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        canonjson::canonical_bytes(self).map_err(|error| {
            ManifestError::new(
                "canonical-json",
                "$",
                "serializable finite canonical JSON",
                error.to_string(),
            )
        })
    }

    /// Compute the exact domain-separated manifest identity used in receipts.
    pub fn release_manifest_sha256(&self) -> Result<String, ManifestError> {
        let bytes = self.canonical_bytes()?;
        Ok(release_manifest_sha256(&bytes))
    }

    /// Enforce the complete v1 pre-network contract on a typed manifest.
    pub fn validate(&self) -> Result<(), ManifestError> {
        expect_exact(
            "manifest-schema-version",
            "/manifest_schema",
            RELEASE_MANIFEST_SCHEMA_V1,
            &self.manifest_schema,
        )?;
        expect_identifier("model-id", "/model_id", &self.model_id)?;
        expect_lower_hex(
            "source-revision",
            "/source_revision",
            &self.source_revision,
            40,
        )?;
        expect_identifier("artifact-id", "/artifact_id", &self.artifact_id)?;
        expect_identifier("release-tag", "/release_tag", &self.release_tag)?;
        if matches!(self.release_tag.as_str(), "main" | "latest") {
            return Err(ManifestError::new(
                "immutable-release-tag",
                "/release_tag",
                "an exact immutable release tag, never main or latest",
                self.release_tag.clone(),
            ));
        }

        self.validate_artifact()?;
        self.validate_parts()?;
        self.validate_source()?;
        self.validate_compatibility()?;
        self.validate_digests()?;
        expect_lower_sha256(
            "license-bundle-digest",
            "/license_bundle_sha256",
            &self.license_bundle_sha256,
        )?;
        expect_exact(
            "packing-policy",
            "/packing_policy",
            LOCAL_DERIVATION_PACKING_POLICY_V1,
            &self.packing_policy,
        )?;
        self.validate_lifecycle()
    }

    fn validate_artifact(&self) -> Result<(), ManifestError> {
        expect_basename(
            "logical-artifact-name",
            "/artifact/logical_name",
            &self.artifact.logical_name,
        )?;
        if !self.artifact.logical_name.ends_with(".fnlpq") {
            return Err(ManifestError::new(
                "logical-artifact-format",
                "/artifact/logical_name",
                "a .fnlpq basename",
                self.artifact.logical_name.clone(),
            ));
        }
        if self.artifact.bytes == 0 {
            return Err(ManifestError::new(
                "artifact-length",
                "/artifact/bytes",
                "a non-zero canonical artifact length",
                "0",
            ));
        }
        expect_lower_sha256("artifact-digest", "/artifact/sha256", &self.artifact.sha256)
    }

    fn validate_parts(&self) -> Result<(), ManifestError> {
        if self.parts.is_empty() || self.parts.len() > MAX_RELEASE_MANIFEST_PARTS {
            return Err(ManifestError::new(
                "part-count-cap",
                "/parts",
                format!("1..={MAX_RELEASE_MANIFEST_PARTS} ordered parts"),
                self.parts.len().to_string(),
            ));
        }
        let mut names = BTreeSet::new();
        let mut total = 0_u64;
        for (index, part) in self.parts.iter().enumerate() {
            let pointer = format!("/parts/{index}");
            let expected_id = u32::try_from(index).map_err(|_| {
                ManifestError::new(
                    "part-id",
                    format!("{pointer}/id"),
                    "u32 ordinal",
                    index.to_string(),
                )
            })?;
            if part.id != expected_id {
                return Err(ManifestError::new(
                    "canonical-part-order",
                    format!("{pointer}/id"),
                    expected_id.to_string(),
                    part.id.to_string(),
                ));
            }
            let expected_name = format!("{}.part{index:02}", self.artifact.logical_name);
            if part.name != expected_name {
                return Err(ManifestError::new(
                    "canonical-part-name",
                    format!("{pointer}/name"),
                    expected_name,
                    part.name.clone(),
                ));
            }
            expect_basename("part-name", format!("{pointer}/name"), &part.name)?;
            let folded = part.name.to_ascii_lowercase();
            if !names.insert(folded) {
                return Err(ManifestError::new(
                    "unique-casefolded-part-name",
                    format!("{pointer}/name"),
                    "a unique ASCII case-folded part name",
                    part.name.clone(),
                ));
            }
            if part.bytes == 0 || part.bytes > RELEASE_PART_BYTES {
                return Err(ManifestError::new(
                    "part-size-cap",
                    format!("{pointer}/bytes"),
                    format!("1..={RELEASE_PART_BYTES}"),
                    part.bytes.to_string(),
                ));
            }
            if index + 1 != self.parts.len() && part.bytes != RELEASE_PART_BYTES {
                return Err(ManifestError::new(
                    "fixed-nonfinal-part-size",
                    format!("{pointer}/bytes"),
                    RELEASE_PART_BYTES.to_string(),
                    part.bytes.to_string(),
                ));
            }
            total = total.checked_add(part.bytes).ok_or_else(|| {
                ManifestError::new(
                    "part-size-sum-overflow",
                    "/parts",
                    "u64 checked sum",
                    "overflow",
                )
            })?;
            expect_lower_sha256("part-digest", format!("{pointer}/sha256"), &part.sha256)?;
            if part.mirrors.is_empty() || part.mirrors.len() > MAX_PART_MIRRORS {
                return Err(ManifestError::new(
                    "mirror-count-cap",
                    format!("{pointer}/mirrors"),
                    format!("1..={MAX_PART_MIRRORS} immutable HTTPS mirrors"),
                    part.mirrors.len().to_string(),
                ));
            }
            let mut urls = BTreeSet::new();
            for (mirror_index, url) in part.mirrors.iter().enumerate() {
                if !urls.insert(url) {
                    return Err(ManifestError::new(
                        "unique-mirror-url",
                        format!("{pointer}/mirrors/{mirror_index}"),
                        "a unique URL in this mirror list",
                        url.clone(),
                    ));
                }
                validate_release_url(
                    format!("{pointer}/mirrors/{mirror_index}"),
                    url,
                    &self.release_tag,
                    &part.name,
                )?;
            }
        }
        if total != self.artifact.bytes {
            return Err(ManifestError::new(
                "part-size-sum",
                "/artifact/bytes",
                total.to_string(),
                self.artifact.bytes.to_string(),
            ));
        }
        Ok(())
    }

    fn validate_source(&self) -> Result<(), ManifestError> {
        if self.source.files.is_empty() || self.source.files.len() > MAX_SOURCE_FILES {
            return Err(ManifestError::new(
                "source-file-count-cap",
                "/source/files",
                format!("1..={MAX_SOURCE_FILES} ordered source files"),
                self.source.files.len().to_string(),
            ));
        }
        let mut names = BTreeSet::new();
        for (index, source) in self.source.files.iter().enumerate() {
            let pointer = format!("/source/files/{index}");
            expect_basename("source-file-name", format!("{pointer}/name"), &source.name)?;
            if !names.insert(source.name.to_ascii_lowercase()) {
                return Err(ManifestError::new(
                    "unique-casefolded-source-name",
                    format!("{pointer}/name"),
                    "a unique ASCII case-folded source name",
                    source.name.clone(),
                ));
            }
            if source.bytes == 0 {
                return Err(ManifestError::new(
                    "source-file-length",
                    format!("{pointer}/bytes"),
                    "a non-zero source length",
                    "0",
                ));
            }
            expect_lower_sha256(
                "source-file-digest",
                format!("{pointer}/sha256"),
                &source.sha256,
            )?;
        }
        expect_lower_sha256(
            "source-root-digest",
            "/source/source_root_sha256",
            &self.source.source_root_sha256,
        )
    }

    fn validate_compatibility(&self) -> Result<(), ManifestError> {
        expect_identifier(
            "recipe-id",
            "/compatibility/recipe_id",
            &self.compatibility.recipe_id,
        )?;
        expect_identifier(
            "converter-id",
            "/compatibility/converter_id",
            &self.compatibility.converter_id,
        )?;
        expect_exact(
            "fnlpq-format-compatibility",
            "/compatibility/fnlpq_format",
            FNLPQ_FORMAT_V1,
            &self.compatibility.fnlpq_format,
        )?;
        expect_exact(
            "pull-api-compatibility",
            "/compatibility/pull_api",
            PULL_API_V1,
            &self.compatibility.pull_api,
        )
    }

    fn validate_digests(&self) -> Result<(), ManifestError> {
        for (name, value) in [
            ("tokenizer", &self.digests.tokenizer_sha256),
            ("template", &self.digests.template_sha256),
            ("census", &self.digests.census_sha256),
            ("logical-model", &self.digests.logical_model_sha256),
            ("packing-set", &self.digests.packing_set_sha256),
        ] {
            expect_lower_sha256(
                "required-identity-digest",
                format!("/digests/{name}_sha256"),
                value,
            )?;
        }
        Ok(())
    }

    fn validate_lifecycle(&self) -> Result<(), ManifestError> {
        let lifecycle = &self.lifecycle;
        match lifecycle.state.as_str() {
            "active" => {
                if lifecycle.superseded_by.is_some() || lifecycle.revocation_reason.is_some() {
                    return Err(ManifestError::new(
                        "active-lifecycle-fields",
                        "/lifecycle",
                        "active releases have no supersession or revocation detail",
                        "unexpected lifecycle detail",
                    ));
                }
            }
            "superseded" => {
                let replacement = lifecycle.superseded_by.as_deref().ok_or_else(|| {
                    ManifestError::new(
                        "supersession-target",
                        "/lifecycle/superseded_by",
                        "a replacement immutable artifact id",
                        "absent",
                    )
                })?;
                expect_identifier(
                    "supersession-target",
                    "/lifecycle/superseded_by",
                    replacement,
                )?;
                if lifecycle.revocation_reason.is_some() {
                    return Err(ManifestError::new(
                        "superseded-lifecycle-fields",
                        "/lifecycle/revocation_reason",
                        "absent for a non-revoked release",
                        "present",
                    ));
                }
            }
            "revoked" => {
                let reason = lifecycle.revocation_reason.as_deref().ok_or_else(|| {
                    ManifestError::new(
                        "revocation-reason",
                        "/lifecycle/revocation_reason",
                        "a non-empty bounded revocation reason",
                        "absent",
                    )
                })?;
                if reason.is_empty() || reason.len() > MAX_IDENTIFIER_BYTES || !reason.is_ascii() {
                    return Err(ManifestError::new(
                        "revocation-reason",
                        "/lifecycle/revocation_reason",
                        format!("1..={MAX_IDENTIFIER_BYTES} ASCII bytes"),
                        reason.to_owned(),
                    ));
                }
                if let Some(replacement) = &lifecycle.superseded_by {
                    expect_identifier(
                        "supersession-target",
                        "/lifecycle/superseded_by",
                        replacement,
                    )?;
                }
            }
            other => {
                return Err(ManifestError::new(
                    "lifecycle-state",
                    "/lifecycle/state",
                    "active, superseded, or revoked",
                    other,
                ));
            }
        }
        Ok(())
    }
}

impl EmbeddedReleaseManifest {
    /// Bind compile-time bytes to the independently recorded manifest digest.
    pub fn new(
        bytes: &'static [u8],
        expected_release_manifest_sha256: &'static str,
    ) -> Result<Self, ManifestError> {
        let actual = release_manifest_sha256(bytes);
        if actual != expected_release_manifest_sha256 {
            return Err(ManifestError::new(
                "embedded-manifest-digest",
                "$",
                expected_release_manifest_sha256,
                actual,
            ));
        }
        ReleaseManifest::parse(bytes)?;
        Ok(Self {
            bytes,
            release_manifest_sha256: expected_release_manifest_sha256,
        })
    }

    /// Return the exact bytes that the installed binary trusts by default.
    #[must_use]
    pub const fn bytes(self) -> &'static [u8] {
        self.bytes
    }

    /// Return the independently embedded domain-separated digest.
    #[must_use]
    pub const fn release_manifest_sha256(self) -> &'static str {
        self.release_manifest_sha256
    }

    /// Reparse the bytes at the pull boundary without granting I/O authority.
    pub fn parse(self) -> Result<ReleaseManifest, ManifestError> {
        ReleaseManifest::parse(self.bytes)
    }
}

/// Hash exact canonical release-manifest bytes in their dedicated domain.
#[must_use]
pub fn release_manifest_sha256(canonical_manifest_bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(RELEASE_MANIFEST_DIGEST_DOMAIN);
    hasher.update(canonical_manifest_bytes);
    hex_lower(&hasher.finalize())
}

fn expect_exact(
    invariant: &'static str,
    pointer: impl Into<String>,
    expected: &str,
    observed: &str,
) -> Result<(), ManifestError> {
    if observed == expected {
        Ok(())
    } else {
        Err(ManifestError::new(invariant, pointer, expected, observed))
    }
}

fn expect_identifier(
    invariant: &'static str,
    pointer: impl Into<String>,
    value: &str,
) -> Result<(), ManifestError> {
    if value.is_empty()
        || value.len() > MAX_IDENTIFIER_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(ManifestError::new(
            invariant,
            pointer,
            format!("1..={MAX_IDENTIFIER_BYTES} ASCII [A-Za-z0-9._-] bytes"),
            value,
        ));
    }
    Ok(())
}

fn expect_basename(
    invariant: &'static str,
    pointer: impl Into<String>,
    value: &str,
) -> Result<(), ManifestError> {
    let pointer = pointer.into();
    expect_identifier(invariant, &pointer, value)?;
    if value == "." || value == ".." || value.starts_with('.') {
        return Err(ManifestError::new(
            invariant,
            pointer,
            "a non-hidden single-component portable filename",
            value,
        ));
    }
    Ok(())
}

fn expect_lower_sha256(
    invariant: &'static str,
    pointer: impl Into<String>,
    value: &str,
) -> Result<(), ManifestError> {
    expect_lower_hex(invariant, pointer, value, 64)
}

fn expect_lower_hex(
    invariant: &'static str,
    pointer: impl Into<String>,
    value: &str,
    width: usize,
) -> Result<(), ManifestError> {
    if value.len() == width
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        Ok(())
    } else {
        Err(ManifestError::new(
            invariant,
            pointer,
            format!("{width} lowercase hexadecimal characters"),
            value,
        ))
    }
}

fn validate_release_url(
    pointer: impl Into<String>,
    url: &str,
    release_tag: &str,
    part_name: &str,
) -> Result<(), ManifestError> {
    let pointer = pointer.into();
    if url.is_empty() || url.len() > MAX_RELEASE_URL_BYTES || !url.is_ascii() {
        return Err(ManifestError::new(
            "mirror-url-cap",
            pointer,
            format!("1..={MAX_RELEASE_URL_BYTES} ASCII bytes"),
            url,
        ));
    }
    let remainder = url.strip_prefix("https://").ok_or_else(|| {
        ManifestError::new(
            "https-only-url",
            &pointer,
            "an https:// immutable release URL",
            url,
        )
    })?;
    let (authority, path) = remainder.split_once('/').ok_or_else(|| {
        ManifestError::new(
            "immutable-release-url",
            &pointer,
            "an HTTPS authority followed by releases/download/<tag>/<asset>",
            url,
        )
    })?;
    if authority.is_empty()
        || authority.contains('@')
        || !authority
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-'))
    {
        return Err(ManifestError::new(
            "mirror-url-authority",
            &pointer,
            "a hostname without credentials, ports, or escapes",
            authority,
        ));
    }
    let suffix = format!("/releases/download/{release_tag}/{part_name}");
    let observed_path = format!("/{path}");
    if !observed_path.ends_with(&suffix)
        || observed_path.len() == suffix.len()
        || observed_path.contains('?')
        || observed_path.contains('#')
    {
        return Err(ManifestError::new(
            "immutable-release-url",
            pointer,
            format!("HTTPS path ending exactly {suffix} with a repository prefix"),
            url,
        ));
    }
    Ok(())
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut value = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        value.push(HEX[usize::from(byte >> 4)] as char);
        value.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    value
}
