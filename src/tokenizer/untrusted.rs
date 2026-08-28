//! Byte-preserving untrusted-document encoding.
//!
//! This proves **marker containment**, not a prompt-injection firewall.  The
//! resulting ids cannot be template-control ids, but instruction-shaped prose
//! remains document data that the model may react to.  All derived output is
//! likewise untrusted data.

use std::{collections::BTreeSet, error::Error, fmt};

use super::{
    bpe::{DecodeBytesError, EncodeError, SpBpeTokenizer},
    specials::TemplateControlIds,
};

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
