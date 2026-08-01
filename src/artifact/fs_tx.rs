//! Fail-closed model-root transactions and append-only activation journals.
//!
//! The mutable model-root OS surface is intentionally unavailable until
//! [`docs/PLATFORM_SURFACES.md`](../../docs/PLATFORM_SURFACES.md) ratifies an
//! owner-only, no-follow lock/rename/durability implementation.  The public
//! root opener therefore refuses rather than applying racy `stat`-then-open
//! checks.  The canonical journal core is fully implemented here so a future
//! ratified handle has one recovery authority to call.

use std::{
    cell::{Cell, RefCell},
    collections::BTreeSet,
    error::Error,
    fmt,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};

use sha2::{Digest, Sha256};

/// Domain framing prepended to every activation-record body before hashing.
pub const ACTIVATION_RECORD_DOMAIN: &[u8] = b"fnlp-activation-v1";
const BODY_FORMAT_VERSION: u8 = 1;
const ACTIVATION_DIGEST_BYTES: usize = 32;
const BODY_FIXED_PREFIX_BYTES: usize = 1 + 8 + (3 * ACTIVATION_DIGEST_BYTES);
const BODY_PREVIOUS_FLAG_OFFSET: usize = BODY_FIXED_PREFIX_BYTES;
const BODY_WITHOUT_PREVIOUS_BYTES: usize = BODY_FIXED_PREFIX_BYTES + 1;
const BODY_WITH_PREVIOUS_BYTES: usize = BODY_WITHOUT_PREVIOUS_BYTES + ACTIVATION_DIGEST_BYTES;
const ENVELOPE_WITHOUT_PREVIOUS_BYTES: usize =
    BODY_WITHOUT_PREVIOUS_BYTES + ACTIVATION_DIGEST_BYTES;
const ENVELOPE_WITH_PREVIOUS_BYTES: usize = BODY_WITH_PREVIOUS_BYTES + ACTIVATION_DIGEST_BYTES;
const FINAL_FILENAME_SEQUENCE_DIGITS: usize = 20;
const FINAL_FILENAME_DIGEST_HEX_DIGITS: usize = ACTIVATION_DIGEST_BYTES * 2;
const FINAL_FILENAME_SUFFIX: &str = ".fnlpaj";

/// Fixed-width SHA-256 identity used by activation records and filenames.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ActivationDigest([u8; 32]);

impl ActivationDigest {
    /// Wraps an already verified SHA-256 digest.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Parses one canonical lowercase SHA-256 hexadecimal content address.
    pub fn parse_hex(value: &str) -> Result<Self, FsTxError> {
        if value.len() != ACTIVATION_DIGEST_BYTES * 2
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(FsTxError::InvalidContentAddress);
        }
        let mut bytes = [0_u8; ACTIVATION_DIGEST_BYTES];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let high = hex_value(pair[0]).ok_or(FsTxError::InvalidContentAddress)?;
            let low = hex_value(pair[1]).ok_or(FsTxError::InvalidContentAddress)?;
            bytes[index] = (high << 4) | low;
        }
        Ok(Self(bytes))
    }

    /// Returns the fixed-width digest bytes.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    /// Lowercase hexadecimal suitable for immutable journal filenames.
    #[must_use]
    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut rendered = String::with_capacity(64);
        for byte in self.0 {
            rendered.push(char::from(HEX[usize::from(byte >> 4)]));
            rendered.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        rendered
    }
}

impl fmt::Display for ActivationDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

/// The canonical body of one immutable activation record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivationRecordBody {
    artifact_digest: ActivationDigest,
    config_digest: ActivationDigest,
    native_digest: ActivationDigest,
    previous_record_digest: Option<ActivationDigest>,
    sequence: u64,
}

impl ActivationRecordBody {
    /// Creates the unique sequence-zero genesis body.
    #[must_use]
    pub const fn genesis(
        artifact_digest: ActivationDigest,
        native_digest: ActivationDigest,
        config_digest: ActivationDigest,
    ) -> Self {
        Self {
            artifact_digest,
            config_digest,
            native_digest,
            previous_record_digest: None,
            sequence: 0,
        }
    }

    /// Reconstructs a retained body before its envelope digest is verified.
    /// This is intentionally explicit: discovery treats every body/digest pair
    /// as untrusted until [`ActivationRecord::digest_is_valid`] succeeds.
    #[must_use]
    pub const fn from_retained_parts(
        sequence: u64,
        artifact_digest: ActivationDigest,
        native_digest: ActivationDigest,
        config_digest: ActivationDigest,
        previous_record_digest: Option<ActivationDigest>,
    ) -> Self {
        Self {
            artifact_digest,
            config_digest,
            native_digest,
            previous_record_digest,
            sequence,
        }
    }

    /// Creates the next checked sequence body linked to an immutable record.
    pub fn successor(
        previous: &ActivationRecord,
        artifact_digest: ActivationDigest,
        native_digest: ActivationDigest,
        config_digest: ActivationDigest,
    ) -> Result<Self, FsTxError> {
        let sequence = previous
            .body
            .sequence
            .checked_add(1)
            .ok_or(FsTxError::SequenceOverflow)?;
        Ok(Self {
            artifact_digest,
            config_digest,
            native_digest,
            previous_record_digest: Some(previous.record_digest),
            sequence,
        })
    }

    /// Checked, monotonic sequence number; it never wraps.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Immutable artifact identity activated by this record.
    #[must_use]
    pub const fn artifact_digest(&self) -> ActivationDigest {
        self.artifact_digest
    }

    /// Native packing identity activated by this record.
    #[must_use]
    pub const fn native_digest(&self) -> ActivationDigest {
        self.native_digest
    }

    /// Execution/configuration identity activated by this record.
    #[must_use]
    pub const fn config_digest(&self) -> ActivationDigest {
        self.config_digest
    }

    /// The immediately previous immutable record, absent only for genesis.
    #[must_use]
    pub const fn previous_record_digest(&self) -> Option<ActivationDigest> {
        self.previous_record_digest
    }

    /// Canonical, length-free fixed-width body bytes.
    ///
    /// The format is `version || sequence_be || artifact || native || config
    /// || previous_present || previous?`, so it has exactly one spelling and
    /// cannot be confused by JSON escaping or map ordering.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(BODY_WITH_PREVIOUS_BYTES);
        bytes.push(BODY_FORMAT_VERSION);
        bytes.extend_from_slice(&self.sequence.to_be_bytes());
        bytes.extend_from_slice(&self.artifact_digest.0);
        bytes.extend_from_slice(&self.native_digest.0);
        bytes.extend_from_slice(&self.config_digest.0);
        match self.previous_record_digest {
            Some(previous) => {
                bytes.push(1);
                bytes.extend_from_slice(&previous.0);
            }
            None => bytes.push(0),
        }
        bytes
    }
}

/// A digest-bound record envelope retained under a unique immutable filename.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivationRecord {
    body: ActivationRecordBody,
    record_digest: ActivationDigest,
}

impl ActivationRecord {
    /// Creates a record whose digest is bound to this exact canonical body.
    #[must_use]
    pub fn new(body: ActivationRecordBody) -> Self {
        let record_digest = digest_record_body(&body);
        Self {
            body,
            record_digest,
        }
    }

    /// Returns the immutable body.
    #[must_use]
    pub const fn body(&self) -> &ActivationRecordBody {
        &self.body
    }

    /// Returns this record's domain-separated digest.
    #[must_use]
    pub const fn record_digest(&self) -> ActivationDigest {
        self.record_digest
    }

    /// Recomputes the domain-framed digest before any record can be adopted.
    #[must_use]
    pub fn digest_is_valid(&self) -> bool {
        self.record_digest == digest_record_body(&self.body)
    }

    /// Canonical staged-envelope bytes written before a future ratified rename.
    #[must_use]
    pub fn canonical_envelope_bytes(&self) -> Vec<u8> {
        let body = self.body.canonical_bytes();
        let mut envelope = Vec::with_capacity(body.len() + self.record_digest.0.len());
        envelope.extend_from_slice(&body);
        envelope.extend_from_slice(&self.record_digest.0);
        envelope
    }

    /// Parses one exact canonical envelope and authenticates its body digest.
    ///
    /// This is deliberately stricter than [`Self::from_retained_parts`]: a
    /// retained byte stream must use the current fixed-width format, have no
    /// trailing bytes, and carry the domain-separated digest for the decoded
    /// body before recovery may consider it. Chain continuity remains the
    /// separate authority of [`discover_activation`].
    pub fn parse_canonical_envelope(envelope: &[u8]) -> Result<Self, FsTxError> {
        let Some(&version) = envelope.first() else {
            return Err(FsTxError::EnvelopeLength {
                observed: envelope.len(),
            });
        };
        if version != BODY_FORMAT_VERSION {
            return Err(FsTxError::EnvelopeVersion { observed: version });
        }
        if envelope.len() < BODY_WITHOUT_PREVIOUS_BYTES {
            return Err(FsTxError::EnvelopeLength {
                observed: envelope.len(),
            });
        }

        let expected_length = match envelope[BODY_PREVIOUS_FLAG_OFFSET] {
            0 => ENVELOPE_WITHOUT_PREVIOUS_BYTES,
            1 => ENVELOPE_WITH_PREVIOUS_BYTES,
            observed => return Err(FsTxError::EnvelopePreviousFlag { observed }),
        };
        if envelope.len() != expected_length {
            return Err(FsTxError::EnvelopeLength {
                observed: envelope.len(),
            });
        }

        let mut offset = 1;
        let mut sequence_bytes = [0_u8; 8];
        let sequence_len = sequence_bytes.len();
        sequence_bytes.copy_from_slice(&envelope[offset..offset + sequence_len]);
        let sequence = u64::from_be_bytes(sequence_bytes);
        offset += sequence_len;

        let artifact_digest =
            digest_from_fixed_bytes(&envelope[offset..offset + ACTIVATION_DIGEST_BYTES]);
        offset += ACTIVATION_DIGEST_BYTES;
        let native_digest =
            digest_from_fixed_bytes(&envelope[offset..offset + ACTIVATION_DIGEST_BYTES]);
        offset += ACTIVATION_DIGEST_BYTES;
        let config_digest =
            digest_from_fixed_bytes(&envelope[offset..offset + ACTIVATION_DIGEST_BYTES]);
        offset += ACTIVATION_DIGEST_BYTES;

        let previous_record_digest = match envelope[offset] {
            0 => None,
            1 => {
                offset += 1;
                Some(digest_from_fixed_bytes(
                    &envelope[offset..offset + ACTIVATION_DIGEST_BYTES],
                ))
            }
            observed => return Err(FsTxError::EnvelopePreviousFlag { observed }),
        };
        let body = ActivationRecordBody::from_retained_parts(
            sequence,
            artifact_digest,
            native_digest,
            config_digest,
            previous_record_digest,
        );
        let record_digest = digest_from_fixed_bytes(
            &envelope[expected_length - ACTIVATION_DIGEST_BYTES..expected_length],
        );
        let record = Self::from_retained_parts(body, record_digest);
        if !record.digest_is_valid() {
            return Err(FsTxError::EnvelopeDigestMismatch);
        }
        Ok(record)
    }

    /// The destination basename for a `create_new` staged envelope after its
    /// same-filesystem immutable rename. A name is never overwritten.
    #[must_use]
    pub fn final_filename(&self) -> String {
        format!(
            "{:0width$}-{}{}",
            self.body.sequence,
            self.record_digest.to_hex(),
            FINAL_FILENAME_SUFFIX,
            width = FINAL_FILENAME_SEQUENCE_DIGITS,
        )
    }

    /// Refuses a retained filename unless it canonically binds this envelope's
    /// sequence and digest. A future directory reader must perform this check
    /// before it admits parsed bytes to chain discovery.
    pub fn validate_final_filename(&self, filename: &str) -> Result<(), FsTxError> {
        let Some(stem) = filename.strip_suffix(FINAL_FILENAME_SUFFIX) else {
            return Err(FsTxError::FinalFilenameInvalid {
                filename: filename.to_owned(),
            });
        };
        let Some((sequence_text, digest_text)) = stem.split_once('-') else {
            return Err(FsTxError::FinalFilenameInvalid {
                filename: filename.to_owned(),
            });
        };
        if sequence_text.len() != FINAL_FILENAME_SEQUENCE_DIGITS
            || !sequence_text.bytes().all(|byte| byte.is_ascii_digit())
            || digest_text.len() != FINAL_FILENAME_DIGEST_HEX_DIGITS
            || !digest_text
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(FsTxError::FinalFilenameInvalid {
                filename: filename.to_owned(),
            });
        }
        let sequence =
            sequence_text
                .parse::<u64>()
                .map_err(|_| FsTxError::FinalFilenameInvalid {
                    filename: filename.to_owned(),
                })?;
        if format!(
            "{:0width$}",
            sequence,
            width = FINAL_FILENAME_SEQUENCE_DIGITS
        ) != sequence_text
        {
            return Err(FsTxError::FinalFilenameInvalid {
                filename: filename.to_owned(),
            });
        }
        if sequence != self.body.sequence || digest_text != self.record_digest.to_hex() {
            return Err(FsTxError::FinalFilenameBindingMismatch {
                filename: filename.to_owned(),
            });
        }
        Ok(())
    }

    /// Constructs retained untrusted input for recovery tests/readers. Discovery
    /// still recomputes the digest and ignores this record if it is forged.
    #[must_use]
    pub const fn from_retained_parts(
        body: ActivationRecordBody,
        record_digest: ActivationDigest,
    ) -> Self {
        Self {
            body,
            record_digest,
        }
    }
}

/// An adopted unambiguous journal head.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivationHead {
    /// The retained record itself.
    pub record: ActivationRecord,
}

impl ActivationHead {
    /// Sequence of the adopted head.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.record.body.sequence
    }

    /// Digest of the adopted head.
    #[must_use]
    pub const fn digest(&self) -> ActivationDigest {
        self.record.record_digest
    }
}

/// One diagnostic row from a full activation-chain walk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainWalkEntry {
    /// Sequence advertised by this retained record.
    pub sequence: u64,
    /// Envelope digest as retained on disk.
    pub digest: ActivationDigest,
    /// Whether this row was adopted, ignored, or exposed a fork.
    pub verdict: ChainWalkVerdict,
}

/// A stable chain-walk diagnostic classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChainWalkVerdict {
    /// The record's recomputed digest mismatched and it was ignored.
    IgnoredDigestMismatch,
    /// The record was valid but disconnected from the unique genesis chain.
    IgnoredDisconnected,
    /// The record was adopted into the one contiguous chain.
    Adopted,
    /// The record is one of multiple valid successors of a retained head.
    ForkSuccessor,
}

/// Successful recovery output. Empty journals are an explicit state, not an
/// invented initial pointer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivationDiscovery {
    /// Last unambiguous record, or `None` if no unique valid genesis exists.
    pub head: Option<ActivationHead>,
    /// Complete deterministic walk, including every ignored retained record.
    pub walk: Vec<ChainWalkEntry>,
}

/// Typed durable mutation or recovery failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FsTxError {
    /// The platform registry has no ratified owner-only/no-follow root surface.
    PlatformSurfaceUnavailable { surface: &'static str },
    /// Opening or synchronizing a root-owned filesystem object failed.
    RootIo {
        operation: &'static str,
        detail: String,
    },
    /// The opened root was not an owner-only directory suitable for mutation.
    HostileRoot { reason: &'static str },
    /// A caller supplied a non-canonical content-address lock identity.
    InvalidContentAddress,
    /// One process attempted to take the same non-reentrant content lock twice.
    LockReentrant,
    /// A monotonic activation sequence would wrap at `u64::MAX`.
    SequenceOverflow,
    /// A final immutable filename was already retained and cannot be reopened.
    FinalNameExists { filename: String },
    /// A retained final filename has no exact canonical journal spelling.
    FinalFilenameInvalid { filename: String },
    /// A canonical-looking final filename names a different record body/digest.
    FinalFilenameBindingMismatch { filename: String },
    /// A retained envelope did not have one exact supported fixed width.
    EnvelopeLength { observed: usize },
    /// A retained envelope named an unsupported body format version.
    EnvelopeVersion { observed: u8 },
    /// A retained envelope used a non-canonical previous-record flag.
    EnvelopePreviousFlag { observed: u8 },
    /// A retained envelope body and its claimed record digest did not match.
    EnvelopeDigestMismatch,
    /// Two valid successors exist; recovery retains the previous head and never
    /// selects a winner by digest or filename order.
    ActivationFork {
        last_unambiguous: Option<ActivationHead>,
        successor_digests: Vec<ActivationDigest>,
        walk: Vec<ChainWalkEntry>,
    },
}

impl fmt::Display for FsTxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlatformSurfaceUnavailable { surface } => {
                write!(
                    formatter,
                    "FS_TXN refused: unratified platform surface {surface}"
                )
            }
            Self::RootIo { operation, detail } => {
                write!(formatter, "FS_TXN refused: {operation}: {detail}")
            }
            Self::HostileRoot { reason } => {
                write!(formatter, "FS_TXN refused: hostile model root: {reason}")
            }
            Self::InvalidContentAddress => {
                formatter.write_str("FS_TXN refused: content address is not lowercase SHA-256")
            }
            Self::LockReentrant => formatter.write_str("FS_TXN refused: content lock re-entry"),
            Self::SequenceOverflow => {
                formatter.write_str("FS_TXN refused: activation sequence overflow")
            }
            Self::FinalNameExists { filename } => {
                write!(
                    formatter,
                    "FS_TXN refused: immutable final already exists {filename}"
                )
            }
            Self::FinalFilenameInvalid { filename } => write!(
                formatter,
                "FS_TXN refused: activation final filename is not canonical {filename}"
            ),
            Self::FinalFilenameBindingMismatch { filename } => write!(
                formatter,
                "FS_TXN refused: activation final filename does not bind its envelope {filename}"
            ),
            Self::EnvelopeLength { observed } => write!(
                formatter,
                "FS_TXN refused: canonical activation envelope length {observed} is unsupported"
            ),
            Self::EnvelopeVersion { observed } => write!(
                formatter,
                "FS_TXN refused: canonical activation envelope version {observed} is unsupported"
            ),
            Self::EnvelopePreviousFlag { observed } => write!(
                formatter,
                "FS_TXN refused: canonical activation envelope previous flag {observed} is invalid"
            ),
            Self::EnvelopeDigestMismatch => {
                formatter.write_str("FS_TXN refused: canonical activation envelope digest mismatch")
            }
            Self::ActivationFork {
                last_unambiguous,
                successor_digests,
                ..
            } => write!(
                formatter,
                "FS_TXN activation fork last_unambiguous={:?} successors={successor_digests:?}",
                last_unambiguous.as_ref().map(ActivationHead::sequence)
            ),
        }
    }
}

impl Error for FsTxError {}

/// A checked, owner-only model-root directory handle.
///
/// The handle is deliberately opaque.  Callers must not re-resolve the caller
/// path after this constructor has admitted it; later staging and journal APIs
/// are added to this capability rather than granting raw-path authority.
#[derive(Debug)]
pub struct RatifiedModelRoot {
    directory: File,
    root: PathBuf,
    content_locks: ContentAddressLockSet,
}

/// A fully synced, not-yet-published sibling stage owned by one model root.
#[derive(Debug)]
pub struct StagedModelFile {
    final_path: PathBuf,
    stage_path: PathBuf,
}

impl RatifiedModelRoot {
    /// Returns the root spelling bound to the opened directory handle.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.root
    }

    /// Synchronizes the admitted directory before a later transaction reports
    /// a durable visibility point.
    pub fn sync_directory(&self) -> Result<(), FsTxError> {
        self.directory
            .sync_all()
            .map_err(|error| FsTxError::RootIo {
                operation: "sync model-root directory",
                detail: error.to_string(),
            })
    }

    /// Acquires the one process-local mutation scope for `content_digest`.
    ///
    /// A caller retains this guard until the staged file is published or the
    /// attempt refuses; a second scope for the same content address fails
    /// typed rather than silently sharing a staging path.
    pub fn try_lock_content(
        &self,
        content_digest: ActivationDigest,
    ) -> Result<ContentAddressLockGuard<'_>, FsTxError> {
        self.content_locks.try_lock(content_digest)
    }

    /// Derives a relative path only when it stays beneath this admitted root.
    pub fn relative_path(&self, candidate: &Path) -> Result<PathBuf, FsTxError> {
        let relative = candidate
            .strip_prefix(&self.root)
            .map_err(|_| FsTxError::HostileRoot {
                reason: "path is outside the admitted model root",
            })?;
        validate_relative_path(relative)?;
        Ok(relative.to_path_buf())
    }

    /// Reads a checked regular file beneath the admitted model root without
    /// following its terminal path component.
    pub fn read_regular_file(&self, relative: &Path) -> Result<Vec<u8>, FsTxError> {
        let path = self.absolute_relative_path(relative)?;
        let file = open_without_follow(&path, "open model-root regular file")?;
        let metadata = file.metadata().map_err(|error| FsTxError::RootIo {
            operation: "inspect opened model-root regular file",
            detail: error.to_string(),
        })?;
        if !metadata.file_type().is_file() {
            return Err(FsTxError::HostileRoot {
                reason: "model-root target is not a regular file",
            });
        }
        let mut bytes = Vec::new();
        let mut reader = file;
        reader.read_to_end(&mut bytes).map_err(|error| FsTxError::RootIo {
            operation: "read model-root regular file",
            detail: error.to_string(),
        })?;
        Ok(bytes)
    }

    /// Creates, writes, and syncs a private sibling stage for a previously
    /// absent immutable target.  The returned value can only be published by
    /// [`Self::publish_staged`].
    pub fn stage_bytes(&self, final_relative: &Path, bytes: &[u8]) -> Result<StagedModelFile, FsTxError> {
        let final_path = self.absolute_relative_path(final_relative)?;
        let parent = final_path.parent().ok_or(FsTxError::HostileRoot {
            reason: "model-root final target has no parent",
        })?;
        self.ensure_relative_directories(final_relative.parent())?;
        let filename = final_path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or(FsTxError::HostileRoot {
                reason: "model-root final target has no Unicode filename",
            })?;
        let stage_path = parent.join(format!(".{filename}.fnlp-stage"));
        let mut stage = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&stage_path)
            .map_err(|error| FsTxError::RootIo {
                operation: "create-new model-root staging file",
                detail: error.to_string(),
            })?;
        stage.write_all(bytes).map_err(|error| FsTxError::RootIo {
            operation: "write model-root staging file",
            detail: error.to_string(),
        })?;
        stage.sync_all().map_err(|error| FsTxError::RootIo {
            operation: "sync model-root staging file",
            detail: error.to_string(),
        })?;
        drop(stage);
        Ok(StagedModelFile {
            final_path,
            stage_path,
        })
    }

    /// Publishes a fully synced sibling stage at a previously absent immutable
    /// name and then syncs its parent directory.  Both names share one parent,
    /// so the rename is necessarily same-filesystem.
    pub fn publish_staged(&self, stage: StagedModelFile) -> Result<PathBuf, FsTxError> {
        match fs::symlink_metadata(&stage.final_path) {
            Ok(_) => {
                let filename = stage
                    .final_path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("<non-utf8>")
                    .to_owned();
                return Err(FsTxError::FinalNameExists { filename });
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(FsTxError::RootIo {
                    operation: "inspect immutable model-root final target",
                    detail: error.to_string(),
                });
            }
        }
        fs::rename(&stage.stage_path, &stage.final_path).map_err(|error| FsTxError::RootIo {
            operation: "rename synced model-root stage to immutable final",
            detail: error.to_string(),
        })?;
        let parent = stage.final_path.parent().ok_or(FsTxError::HostileRoot {
            reason: "model-root final target has no parent after staging",
        })?;
        sync_directory_at(parent)?;
        Ok(stage.final_path)
    }

    /// Opens the one append-only activation journal for this model root.
    pub fn activation_journal(&self) -> Result<FilesystemActivationJournal<'_>, FsTxError> {
        self.ensure_relative_directories(Some(Path::new(ACTIVATION_JOURNAL_DIRECTORY)))?;
        Ok(FilesystemActivationJournal { root: self })
    }

    fn absolute_relative_path(&self, relative: &Path) -> Result<PathBuf, FsTxError> {
        validate_relative_path(relative)?;
        Ok(self.root.join(relative))
    }

    fn ensure_relative_directories(&self, relative: Option<&Path>) -> Result<(), FsTxError> {
        let Some(relative) = relative else {
            return Ok(());
        };
        validate_relative_path(relative)?;
        let mut current = self.root.clone();
        for component in relative.components() {
            let Component::Normal(name) = component else {
                return Err(FsTxError::HostileRoot {
                    reason: "model-root directory component is not normal",
                });
            };
            current.push(name);
            match fs::create_dir(&current) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(FsTxError::RootIo {
                        operation: "create model-root directory",
                        detail: error.to_string(),
                    });
                }
            }
            let directory = open_without_follow(&current, "open model-root child directory")?;
            let metadata = directory.metadata().map_err(|error| FsTxError::RootIo {
                operation: "inspect model-root child directory",
                detail: error.to_string(),
            })?;
            if !metadata.file_type().is_dir() {
                return Err(FsTxError::HostileRoot {
                    reason: "model-root child target is not a directory",
                });
            }
            sync_directory_at(current.parent().ok_or(FsTxError::HostileRoot {
                reason: "model-root child directory has no parent",
            })?)?;
        }
        Ok(())
    }
}

/// Refuses mutable model-root access until a reviewed handle-relative,
/// no-replace transaction surface exists.
///
/// The terminal `O_NOFOLLOW` experiment is deliberately insufficient: it
/// cannot authenticate the effective owner/ACLs, anchor child operations to a
/// directory handle, or provide a no-replace final rename.  Returning this
/// capability would therefore overclaim platform authority.
pub fn open_ratified_model_root(_root: &Path) -> Result<RatifiedModelRoot, FsTxError> {
    Err(FsTxError::PlatformSurfaceUnavailable {
        surface: "model-root owner-only lock/no-follow/durability",
    })
}

const ACTIVATION_JOURNAL_DIRECTORY: &str = ".fnlp-activation";

fn validate_relative_path(relative: &Path) -> Result<(), FsTxError> {
    if relative.as_os_str().is_empty()
        || relative.components().any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(FsTxError::HostileRoot {
            reason: "model-root path is not a nonempty relative normal-component path",
        });
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn open_without_follow(path: &Path, operation: &'static str) -> Result<File, FsTxError> {
    use std::os::unix::fs::OpenOptionsExt;

    const O_NOFOLLOW: i32 = 0o400000;

    OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW)
        .open(path)
        .map_err(|error| FsTxError::RootIo {
            operation,
            detail: error.to_string(),
        })
}

#[cfg(not(target_os = "linux"))]
fn open_without_follow(_path: &Path, _operation: &'static str) -> Result<File, FsTxError> {
    Err(FsTxError::PlatformSurfaceUnavailable {
        surface: "model-root owner-only lock/no-follow/durability",
    })
}

fn sync_directory_at(path: &Path) -> Result<(), FsTxError> {
    open_without_follow(path, "open model-root directory for sync")?
        .sync_all()
        .map_err(|error| FsTxError::RootIo {
            operation: "sync model-root directory",
            detail: error.to_string(),
        })
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

/// A filesystem-backed append-only activation journal below one admitted root.
///
/// Its public operations preserve the same canonical envelopes and recovery
/// rules as [`SimulatedActivationJournal`], while the root capability owns the
/// `create_new`, sync, same-directory rename, and directory-sync sequence.
#[derive(Debug)]
pub struct FilesystemActivationJournal<'root> {
    root: &'root RatifiedModelRoot,
}

impl FilesystemActivationJournal<'_> {
    /// Recovers the one unambiguous chain retained by this root.
    pub fn discover(&self) -> Result<ActivationDiscovery, FsTxError> {
        let relative_directory = Path::new(ACTIVATION_JOURNAL_DIRECTORY);
        let directory = self.root.absolute_relative_path(relative_directory)?;
        let entries = fs::read_dir(&directory).map_err(|error| FsTxError::RootIo {
            operation: "enumerate activation journal directory",
            detail: error.to_string(),
        })?;
        let mut records = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| FsTxError::RootIo {
                operation: "read activation journal directory entry",
                detail: error.to_string(),
            })?;
            let filename = entry
                .file_name()
                .into_string()
                .map_err(|_| FsTxError::HostileRoot {
                    reason: "activation journal entry name is not Unicode",
                })?;
            if filename.starts_with('.') {
                // A synced but unpublished `create_new` stage is never a
                // recovery candidate.  It remains forensic evidence only.
                continue;
            }
            let relative = relative_directory.join(&filename);
            let envelope = self.root.read_regular_file(&relative)?;
            let record = ActivationRecord::parse_canonical_envelope(&envelope)?;
            record.validate_final_filename(&filename)?;
            records.push(record);
        }
        discover_activation(&records)
    }

    /// Appends one immutable activation or rollback record under the journal's
    /// dedicated non-reentrant mutation scope.
    pub fn append(
        &self,
        artifact_digest: ActivationDigest,
        native_digest: ActivationDigest,
        config_digest: ActivationDigest,
    ) -> Result<ActivationHead, FsTxError> {
        let _journal_lock = self.root.try_lock_content(ActivationDigest::from_bytes([0; 32]))?;
        let body = match self.discover()?.head {
            Some(previous) => ActivationRecordBody::successor(
                &previous.record,
                artifact_digest,
                native_digest,
                config_digest,
            )?,
            None => ActivationRecordBody::genesis(artifact_digest, native_digest, config_digest),
        };
        let record = ActivationRecord::new(body);
        let final_relative = Path::new(ACTIVATION_JOURNAL_DIRECTORY).join(record.final_filename());
        let stage = self
            .root
            .stage_bytes(&final_relative, &record.canonical_envelope_bytes())?;
        self.root.publish_staged(stage)?;
        Ok(ActivationHead { record })
    }
}

/// A safe in-memory stand-in for the create-new/sync/rename journal protocol.
///
/// It is used by the crash-state matrix and cannot mutate a user model root.
/// A future ratified filesystem adapter must preserve these append-only naming
/// and recovery rules exactly.
#[derive(Debug, Default)]
pub struct SimulatedActivationJournal {
    final_names: BTreeSet<String>,
    records: Vec<ActivationRecord>,
}

impl SimulatedActivationJournal {
    /// Creates an empty journal with no implicit genesis record.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            final_names: BTreeSet::new(),
            records: Vec::new(),
        }
    }

    /// Appends an activation or rollback record. Both operations use this same
    /// append-only mechanism; callers choose prior immutable digests to roll
    /// back rather than overwriting any active pointer.
    pub fn append(
        &mut self,
        artifact_digest: ActivationDigest,
        native_digest: ActivationDigest,
        config_digest: ActivationDigest,
    ) -> Result<ActivationHead, FsTxError> {
        let body = match discover_activation(&self.records)?.head {
            Some(previous) => ActivationRecordBody::successor(
                &previous.record,
                artifact_digest,
                native_digest,
                config_digest,
            )?,
            None => ActivationRecordBody::genesis(artifact_digest, native_digest, config_digest),
        };
        let record = ActivationRecord::new(body);
        self.retain_immutable(record.clone())?;
        Ok(ActivationHead { record })
    }

    /// Retains a staged/final envelope exactly as recovery would find it. This
    /// enables forged, torn, disconnected, and fork fixtures without granting
    /// any real filesystem write capability.
    pub fn retain_recovery_fixture(&mut self, record: ActivationRecord) -> Result<(), FsTxError> {
        self.retain_immutable(record)
    }

    /// Parses and retains one immutable final envelope through the same
    /// byte-and-name binding a future ratified directory reader must enforce.
    ///
    /// This remains an in-memory fixture ingress: it performs no filesystem
    /// access and does not weaken the fail-closed model-root policy. Tests that
    /// need forged or torn retained state must use [`Self::retain_recovery_fixture`]
    /// explicitly so their forensic-only status is visible at the call site.
    pub fn retain_canonical_final_envelope(
        &mut self,
        filename: &str,
        envelope: &[u8],
    ) -> Result<(), FsTxError> {
        let record = ActivationRecord::parse_canonical_envelope(envelope)?;
        record.validate_final_filename(filename)?;
        self.retain_immutable(record)
    }

    /// Discovers the unique contiguous chain from retained envelopes.
    pub fn discover(&self) -> Result<ActivationDiscovery, FsTxError> {
        discover_activation(&self.records)
    }

    /// Immutable retained records in creation/discovery input order.
    #[must_use]
    pub fn records(&self) -> &[ActivationRecord] {
        &self.records
    }

    fn retain_immutable(&mut self, record: ActivationRecord) -> Result<(), FsTxError> {
        let filename = record.final_filename();
        if !self.final_names.insert(filename.clone()) {
            return Err(FsTxError::FinalNameExists { filename });
        }
        self.records.push(record);
        Ok(())
    }
}

/// A process-local content-lock model used by the test matrix to prove that a
/// re-entrant caller is rejected rather than silently sharing a mutation scope.
#[derive(Debug, Default)]
pub struct NonReentrantContentLock {
    held: Cell<bool>,
}

impl NonReentrantContentLock {
    /// Acquires the model-root content lock once.
    pub fn try_lock(&self) -> Result<ContentLockGuard<'_>, FsTxError> {
        if self.held.replace(true) {
            return Err(FsTxError::LockReentrant);
        }
        Ok(ContentLockGuard { held: &self.held })
    }
}

/// RAII release for the simulated non-reentrant content lock.
#[derive(Debug)]
pub struct ContentLockGuard<'lock> {
    held: &'lock Cell<bool>,
}

impl Drop for ContentLockGuard<'_> {
    fn drop(&mut self) {
        self.held.set(false);
    }
}

/// A process-local lock table keyed by immutable content address.
///
/// This test-model surface makes the intended granularity explicit: separate
/// artifacts may make progress independently, while a second mutation scope
/// for the same digest fails typed rather than sharing an ambiguous lock.
#[derive(Debug, Default)]
pub struct ContentAddressLockSet {
    held: RefCell<BTreeSet<ActivationDigest>>,
}

impl ContentAddressLockSet {
    /// Acquires the one non-reentrant lock associated with `content_digest`.
    pub fn try_lock(
        &self,
        content_digest: ActivationDigest,
    ) -> Result<ContentAddressLockGuard<'_>, FsTxError> {
        if !self.held.borrow_mut().insert(content_digest) {
            return Err(FsTxError::LockReentrant);
        }
        Ok(ContentAddressLockGuard {
            held: &self.held,
            content_digest,
        })
    }
}

/// RAII release for one simulated content-address lock.
#[derive(Debug)]
pub struct ContentAddressLockGuard<'lock> {
    held: &'lock RefCell<BTreeSet<ActivationDigest>>,
    content_digest: ActivationDigest,
}

impl Drop for ContentAddressLockGuard<'_> {
    fn drop(&mut self) {
        let removed = self.held.borrow_mut().remove(&self.content_digest);
        debug_assert!(removed, "a content-address lock guard must release once");
    }
}

/// Recomputes every digest and follows only the one unique contiguous chain
/// from a valid sequence-zero genesis record. Invalid, staged/torn, gapped, and
/// disconnected records never become an active head.
pub fn discover_activation(records: &[ActivationRecord]) -> Result<ActivationDiscovery, FsTxError> {
    let mut valid = Vec::new();
    let mut walk = Vec::with_capacity(records.len());
    for record in records {
        if record.digest_is_valid() {
            valid.push(record);
        } else {
            walk.push(ChainWalkEntry {
                sequence: record.body.sequence,
                digest: record.record_digest,
                verdict: ChainWalkVerdict::IgnoredDigestMismatch,
            });
        }
    }

    let mut genesis = valid
        .iter()
        .copied()
        .filter(|record| record.body.sequence == 0 && record.body.previous_record_digest.is_none())
        .collect::<Vec<_>>();
    genesis.sort_by_key(|record| record.record_digest);
    if genesis.len() > 1 {
        let successors = genesis
            .iter()
            .map(|record| record.record_digest)
            .collect::<Vec<_>>();
        for record in genesis {
            walk.push(ChainWalkEntry {
                sequence: record.body.sequence,
                digest: record.record_digest,
                verdict: ChainWalkVerdict::ForkSuccessor,
            });
        }
        append_disconnected_walk_entries(&valid, &mut walk);
        sort_walk(&mut walk);
        return Err(FsTxError::ActivationFork {
            last_unambiguous: None,
            successor_digests: successors,
            walk,
        });
    }
    let Some(mut current) = genesis.pop() else {
        append_disconnected_walk_entries(&valid, &mut walk);
        sort_walk(&mut walk);
        return Ok(ActivationDiscovery { head: None, walk });
    };

    let mut adopted = BTreeSet::new();
    adopted.insert(current.record_digest);
    loop {
        let next_sequence = match current.body.sequence.checked_add(1) {
            Some(sequence) => sequence,
            None => {
                if valid
                    .iter()
                    .any(|record| record.body.previous_record_digest == Some(current.record_digest))
                {
                    return Err(FsTxError::SequenceOverflow);
                }
                break;
            }
        };
        let mut successors = valid
            .iter()
            .copied()
            .filter(|record| {
                record.body.sequence == next_sequence
                    && record.body.previous_record_digest == Some(current.record_digest)
            })
            .collect::<Vec<_>>();
        successors.sort_by_key(|record| record.record_digest);
        match successors.len() {
            0 => break,
            1 => {
                current = successors[0];
                adopted.insert(current.record_digest);
            }
            _ => {
                let successor_digests = successors
                    .iter()
                    .map(|record| record.record_digest)
                    .collect::<Vec<_>>();
                for record in successors {
                    walk.push(ChainWalkEntry {
                        sequence: record.body.sequence,
                        digest: record.record_digest,
                        verdict: ChainWalkVerdict::ForkSuccessor,
                    });
                }
                for record in &valid {
                    if adopted.contains(&record.record_digest) {
                        walk.push(ChainWalkEntry {
                            sequence: record.body.sequence,
                            digest: record.record_digest,
                            verdict: ChainWalkVerdict::Adopted,
                        });
                    }
                }
                append_disconnected_walk_entries(&valid, &mut walk);
                sort_walk(&mut walk);
                return Err(FsTxError::ActivationFork {
                    last_unambiguous: Some(ActivationHead {
                        record: current.clone(),
                    }),
                    successor_digests,
                    walk,
                });
            }
        }
    }

    for record in valid {
        walk.push(ChainWalkEntry {
            sequence: record.body.sequence,
            digest: record.record_digest,
            verdict: if adopted.contains(&record.record_digest) {
                ChainWalkVerdict::Adopted
            } else {
                ChainWalkVerdict::IgnoredDisconnected
            },
        });
    }
    sort_walk(&mut walk);
    Ok(ActivationDiscovery {
        head: Some(ActivationHead {
            record: current.clone(),
        }),
        walk,
    })
}

fn append_disconnected_walk_entries(valid: &[&ActivationRecord], walk: &mut Vec<ChainWalkEntry>) {
    let existing = walk
        .iter()
        .map(|entry| entry.digest)
        .collect::<BTreeSet<_>>();
    for record in valid {
        if !existing.contains(&record.record_digest) {
            walk.push(ChainWalkEntry {
                sequence: record.body.sequence,
                digest: record.record_digest,
                verdict: ChainWalkVerdict::IgnoredDisconnected,
            });
        }
    }
}

fn sort_walk(walk: &mut [ChainWalkEntry]) {
    walk.sort_by_key(|entry| (entry.sequence, entry.digest, entry.verdict as u8));
}

fn digest_record_body(body: &ActivationRecordBody) -> ActivationDigest {
    let mut hasher = Sha256::new();
    hasher.update(ACTIVATION_RECORD_DOMAIN);
    hasher.update(body.canonical_bytes());
    ActivationDigest(hasher.finalize().into())
}

fn digest_from_fixed_bytes(bytes: &[u8]) -> ActivationDigest {
    let mut digest = [0_u8; ACTIVATION_DIGEST_BYTES];
    digest.copy_from_slice(bytes);
    ActivationDigest::from_bytes(digest)
}
