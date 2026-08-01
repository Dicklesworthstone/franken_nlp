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
    path::Path,
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

/// An uninhabited placeholder for the future ratified model-root capability.
///
/// No target can construct this value: safe `std` cannot currently prove the
/// required owner, ACL, handle-relative, no-replace, locking, and durability
/// contract.  Keeping the type in the fallible opener signature preserves a
/// typed refusal seam without exposing a misleading partial capability.
#[derive(Debug)]
pub enum RatifiedModelRoot {}

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
