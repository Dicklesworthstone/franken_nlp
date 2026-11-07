//! Byte-preserving untrusted-document encoding.
//!
//! This proves **marker containment**, not a prompt-injection firewall.  The
//! resulting ids cannot be template-control ids, but instruction-shaped prose
//! remains document data that the model may react to.  All derived output is
//! likewise untrusted data.

use std::{collections::BTreeSet, error::Error, fmt};

use serde::{Deserialize, Serialize};

use super::{
    bpe::{DecodeBytesError, EncodeError, SpBpeTokenizer},
    specials::TemplateControlIds,
};

/// The existing one-byte/one-token policy remains the default. The compact
/// policy is a separately named semantic choice, not ordinary tokenizer parity
/// and not a claim about model quality. Bind the complete policy to the prompt
/// identity whenever it is selected.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "limits", rename_all = "snake_case", deny_unknown_fields)]
pub enum UntrustedEncoding {
    #[default]
    ByteFallback,
    SafeBpeV1(SafeBpeLimits),
}

impl UntrustedEncoding {
    pub const fn version(&self) -> &'static str {
        match self {
            Self::ByteFallback => "untrusted-byte-fallback-v1",
            Self::SafeBpeV1(_) => "untrusted-safe-bpe-v1",
        }
    }

    /// Validate without loading a model or allocating tokenizer workspace.
    pub fn validate(&self) -> Result<(), UntrustedDocumentError> {
        match self {
            Self::ByteFallback => Ok(()),
            Self::SafeBpeV1(limits) => limits.validate(),
        }
    }
}

/// Cumulative limits for every untrusted segment in one admitted request.
/// Pair checks bound the intentionally simple merge search; pair bytes also
/// bound comparisons involving long candidate pieces. These are work units,
/// not a measured latency or RSS claim.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SafeBpeLimits {
    pub max_input_bytes: usize,
    pub max_output_tokens: usize,
    pub max_pair_checks: u64,
    pub max_pair_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SafeBpeRequirements {
    /// Conservative source/token/symbol/verification payload allowance. The
    /// host separately admits the existing immutable tokenizer and its maps.
    pub workspace_bytes: u64,
}

impl SafeBpeLimits {
    pub const MAX_INPUT_BYTES: usize = 1 << 20;
    pub const MAX_OUTPUT_TOKENS: usize = 1 << 20;
    pub const MAX_PAIR_CHECKS: u64 = 1 << 26;
    pub const MAX_PAIR_BYTES: u64 = 1 << 29;

    pub fn validate(&self) -> Result<(), UntrustedDocumentError> {
        for (resource, value, maximum) in [
            ("input bytes", self.max_input_bytes as u64, Self::MAX_INPUT_BYTES as u64),
            ("output tokens", self.max_output_tokens as u64, Self::MAX_OUTPUT_TOKENS as u64),
            ("pair checks", self.max_pair_checks, Self::MAX_PAIR_CHECKS),
            ("pair bytes", self.max_pair_bytes, Self::MAX_PAIR_BYTES),
        ] {
            if value == 0 || value > maximum {
                return Err(UntrustedDocumentError::InvalidEncodingLimits { resource });
            }
        }
        Ok(())
    }

    /// The merge implementation stores spans into one bounded normalized
    /// string, so it never retains a new string for every merge. This loose
    /// bound also covers source copies and both independent decode checks.
    pub fn requirements(&self) -> Result<SafeBpeRequirements, UntrustedDocumentError> {
        self.validate()?;
        let workspace_bytes = (self.max_input_bytes as u64)
            .checked_mul(256)
            .and_then(|bytes| bytes.checked_add((self.max_output_tokens as u64) * 16))
            .and_then(|bytes| bytes.checked_add(4096))
            .ok_or(UntrustedDocumentError::InvalidEncodingLimits { resource: "workspace bytes" })?;
        Ok(SafeBpeRequirements { workspace_bytes })
    }
}

/// Reuse one counter across every message/segment of a request. Failed and
/// cancelled attempts retain consumed work; no implicit refund or reset occurs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct SafeBpeWork {
    input_bytes: usize,
    output_tokens: usize,
    pair_checks: u64,
    pair_bytes: u64,
}

impl SafeBpeWork {
    pub const fn input_bytes(&self) -> usize { self.input_bytes }
    pub const fn output_tokens(&self) -> usize { self.output_tokens }
    pub const fn pair_checks(&self) -> u64 { self.pair_checks }
    pub const fn pair_bytes(&self) -> u64 { self.pair_bytes }

    fn check(&self, limits: SafeBpeLimits) -> Result<(), UntrustedDocumentError> {
        for (resource, exceeded) in [
            ("input bytes", self.input_bytes > limits.max_input_bytes),
            ("output tokens", self.output_tokens > limits.max_output_tokens),
            ("pair checks", self.pair_checks > limits.max_pair_checks),
            ("pair bytes", self.pair_bytes > limits.max_pair_bytes),
        ] {
            if exceeded { return Err(UntrustedDocumentError::EncodingBudget { resource }); }
        }
        Ok(())
    }

    pub(super) fn input(&mut self, bytes: usize, limits: SafeBpeLimits) -> Result<(), UntrustedDocumentError> {
        let next = self.input_bytes.checked_add(bytes)
            .filter(|&value| value <= limits.max_input_bytes)
            .ok_or(UntrustedDocumentError::EncodingBudget { resource: "input bytes" })?;
        self.input_bytes = next;
        Ok(())
    }

    pub(super) fn token(&mut self, limits: SafeBpeLimits) -> Result<(), UntrustedDocumentError> {
        let next = self.output_tokens.checked_add(1)
            .filter(|&value| value <= limits.max_output_tokens)
            .ok_or(UntrustedDocumentError::EncodingBudget { resource: "output tokens" })?;
        self.output_tokens = next;
        Ok(())
    }

    pub(super) fn pair(&mut self, bytes: usize, limits: SafeBpeLimits) -> Result<(), UntrustedDocumentError> {
        let checks = self.pair_checks.checked_add(1)
            .filter(|&value| value <= limits.max_pair_checks)
            .ok_or(UntrustedDocumentError::EncodingBudget { resource: "pair checks" })?;
        let bytes = self.pair_bytes.checked_add(bytes as u64)
            .filter(|&value| value <= limits.max_pair_bytes)
            .ok_or(UntrustedDocumentError::EncodingBudget { resource: "pair bytes" })?;
        self.pair_checks = checks;
        self.pair_bytes = bytes;
        Ok(())
    }
}

/// A document segment admitted through the only untrusted-text token boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UntrustedDocument {
    bytes: Vec<u8>,
    ids: Vec<u32>,
}

impl UntrustedDocument {
    /// Exact source bytes retained for composition and receipt identity.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The complete token sequence, proven free of every template control id.
    #[must_use]
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
}

/// Encoder coupled to a parsed archived `TemplateControlIds` registry.
///
/// The byte-only path intentionally does not run injected-token matching, so
/// literal `<think>`/`<|im_start|>` source bytes stay source bytes rather than
/// becoming privileged control ids.
#[derive(Debug)]
pub struct UntrustedDocumentEncoder<'tokenizer, 'registry> {
    tokenizer: &'tokenizer SpBpeTokenizer,
    template_controls: &'registry TemplateControlIds,
}

impl<'tokenizer, 'registry> UntrustedDocumentEncoder<'tokenizer, 'registry> {
    #[must_use]
    pub const fn new(
        tokenizer: &'tokenizer SpBpeTokenizer,
        template_controls: &'registry TemplateControlIds,
    ) -> Self {
        Self {
            tokenizer,
            template_controls,
        }
    }

    /// The exact authoritative forbidden set, exposed for registry-drift tests
    /// and never augmented with a local hand-maintained list.
    #[must_use]
    pub fn forbidden_ids(&self) -> &BTreeSet<u32> {
        self.template_controls.ids()
    }

    /// Encode a document through byte fallback, reject any forbidden id, and
    /// verify the exact decode before returning a typed untrusted segment.
    pub fn encode(&self, bytes: &[u8]) -> Result<UntrustedDocument, UntrustedDocumentError> {
        let ids = self
            .tokenizer
            .encode_byte_fallback_only(bytes)
            .map_err(UntrustedDocumentError::Encode)?;

        for (offset, &id) in ids.iter().enumerate() {
            if let Some(control) = self.template_controls.entry(id) {
                // The reported offset is the position of the offending token in
                // the encoded id stream, NOT a byte offset into the source.
                // The current encode API does not surface per-token byte spans;
                // computing a true byte offset would require an
                // `encode_byte_fallback_with_offsets` helper. Until that is
                // added, callers must interpret the field as a token index and
                // recompute the byte range from the id stream if they need it.
                return Err(UntrustedDocumentError::ForbiddenControl {
                    token_offset: offset,
                    context: context_window(bytes, offset),
                    id,
                    piece_text: control.surface.clone(),
                });
            }
        }

        let decoded = self
            .tokenizer
            .decode_bytes(&ids)
            .map_err(UntrustedDocumentError::Decode)?;
        if decoded != bytes {
            let byte_offset = first_difference(bytes, &decoded);
            return Err(UntrustedDocumentError::BytePreservation {
                byte_offset,
                expected: bytes.get(byte_offset).copied(),
                observed: decoded.get(byte_offset).copied(),
            });
        }
        Ok(UntrustedDocument {
            bytes: bytes.to_vec(),
            ids,
        })
    }

    /// Explicit compact encoding with cumulative admission and cancellation.
    /// `checkpoint` returns false to cancel. It is called before/after the
    /// operation and at bounded symbol/pair/byte intervals during encoding.
    /// ByteFallback keeps the existing byte-only semantics and does not consume
    /// SafeBpeWork; safe-BPE must reuse that counter across all request segments.
    pub fn encode_with_policy<F: FnMut() -> bool>(
        &self,
        bytes: &[u8],
        policy: &UntrustedEncoding,
        work: &mut SafeBpeWork,
        checkpoint: &mut F,
    ) -> Result<UntrustedDocument, UntrustedDocumentError> {
        policy.validate()?;
        poll(checkpoint)?;
        let UntrustedEncoding::SafeBpeV1(limits) = policy else {
            let document = self.encode(bytes)?;
            poll(checkpoint)?;
            return Ok(document);
        };
        work.check(*limits)?;
        work.input(bytes.len(), *limits)?;
        let ids = self.tokenizer.encode_untrusted_bpe(bytes, self.forbidden_ids(), *limits, work, checkpoint)?;
        poll(checkpoint)?;
        // The tokenizer deliberately excludes every injected ID as well as the
        // authoritative control registry. Independently verify the resulting
        // bytes using the public decoder, both at stream start and after a
        // byte-emitting prefix. This catches dummy-prefix/whitespace drift.
        let decoded = self.tokenizer.decode_bytes(&ids).map_err(UntrustedDocumentError::Decode)?;
        if decoded != bytes {
            let byte_offset = first_difference(bytes, &decoded);
            return Err(UntrustedDocumentError::BytePreservation {
                byte_offset, expected: bytes.get(byte_offset).copied(), observed: decoded.get(byte_offset).copied(),
            });
        }
        poll(checkpoint)?;
        if let Some(&first) = ids.first() {
            let mut embedded = Vec::new();
            embedded.try_reserve_exact(ids.len() + 1).map_err(|_| UntrustedDocumentError::AllocationRefused)?;
            embedded.push(first);
            embedded.extend_from_slice(&ids);
            let combined = self.tokenizer.decode_bytes(&embedded).map_err(UntrustedDocumentError::Decode)?;
            if combined.first() != bytes.first() || combined.get(1..) != Some(bytes) {
                return Err(UntrustedDocumentError::EmbeddedBytePreservation);
            }
        }
        poll(checkpoint)?;
        let mut source = Vec::new();
        source.try_reserve_exact(bytes.len()).map_err(|_| UntrustedDocumentError::AllocationRefused)?;
        source.extend_from_slice(bytes);
        poll(checkpoint)?;
        Ok(UntrustedDocument { bytes: source, ids })
    }
}

pub(super) fn poll<F: FnMut() -> bool>(checkpoint: &mut F) -> Result<(), UntrustedDocumentError> {
    if checkpoint() { Ok(()) } else { Err(UntrustedDocumentError::Cancelled) }
}

/// Typed preflight rejection.  There is no lossy/drop/substitution branch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UntrustedDocumentError {
    Encode(EncodeError),
    Decode(DecodeBytesError),
    ForbiddenControl {
        /// Index of the offending token in the encoded id stream, NOT a byte
        /// offset into the source. The current encode API does not surface
        /// per-token byte spans; callers needing a byte range must recompute
        /// it from the id stream.
        token_offset: usize,
        context: Vec<u8>,
        id: u32,
        piece_text: String,
    },
    BytePreservation {
        byte_offset: usize,
        expected: Option<u8>,
        observed: Option<u8>,
    },
    InvalidEncodingLimits { resource: &'static str },
    EncodingBudget { resource: &'static str },
    UnsafeTokenId { id: u32 },
    EmbeddedBytePreservation,
    AllocationRefused,
    Cancelled,
}

impl fmt::Display for UntrustedDocumentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encode(error) => error.fmt(formatter),
            Self::Decode(error) => error.fmt(formatter),
            Self::ForbiddenControl {
                token_offset,
                context,
                id,
                piece_text,
            } => write!(
                formatter,
                "UNTRUSTED violation id={id} piece={piece_text:?} token_offset={token_offset} context_hex={}",
                hex(context)
            ),
            Self::BytePreservation {
                byte_offset,
                expected,
                observed,
            } => write!(
                formatter,
                "UNTRUSTED byte-preservation failure byte_offset={byte_offset} expected={expected:?} observed={observed:?}"
            ),
            Self::InvalidEncodingLimits { resource } => write!(formatter, "UNTRUSTED invalid safe-BPE limit resource={resource}"),
            Self::EncodingBudget { resource } => write!(formatter, "UNTRUSTED safe-BPE budget exceeded resource={resource}"),
            Self::UnsafeTokenId { id } => write!(formatter, "UNTRUSTED safe-BPE rejects control or injected token id={id}"),
            Self::EmbeddedBytePreservation => formatter.write_str("UNTRUSTED safe-BPE embedded bytes differ from source"),
            Self::AllocationRefused => formatter.write_str("UNTRUSTED safe-BPE allocation refused"),
            Self::Cancelled => formatter.write_str("UNTRUSTED safe-BPE encoding cancelled"),
        }
    }
}

impl Error for UntrustedDocumentError {}

fn first_difference(expected: &[u8], observed: &[u8]) -> usize {
    expected
        .iter()
        .zip(observed)
        .position(|(left, right)| left != right)
        .unwrap_or_else(|| expected.len().min(observed.len()))
}

fn context_window(bytes: &[u8], offset: usize) -> Vec<u8> {
    let start = offset.saturating_sub(32);
    let end = offset.saturating_add(33).min(bytes.len());
    bytes[start..end].to_vec()
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;

    let mut rendered = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        let _ = write!(rendered, "{byte:02x}");
    }
    rendered
}
