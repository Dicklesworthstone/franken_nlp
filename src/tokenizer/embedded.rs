//! Embedded tokenizer construction and `.fnlpq` consistency checks.
//!
//! The product binary supplies the exact pinned `tokenizer.model` bytes through
//! [`EmbeddedTokenizer::from_bytes`].  The bytes and their SHA-256 are held in
//! one value so every later `.fnlpq` artifact check compares the same tokenizer
//! that produced token ids.  The actual pinned byte inclusion belongs to the
//! truth-pack fixture closure; this module deliberately never substitutes a
//! fallback tokenizer when those bytes are unavailable.

use std::{error::Error, fmt};

use sha2::{Digest, Sha256};

use crate::artifact::{
    format::SectionKind,
    reader::{FnlpqArtifact, FnlpqReadError},
};

use super::{
    bpe::{BpeBuildError, SpBpeTokenizer},
    sp_model::{SpmError, parse_spm_model},
};

/// Failures while turning the binary's embedded `tokenizer.model` bytes into
/// the exact SentencePiece BPE surface.
#[derive(Debug)]
pub enum EmbeddedTokenizerError {
    /// The owned minimal protobuf reader rejected the claimed tokenizer bytes.
    Model(SpmError),
    /// The parsed model could not provide a safe BPE surface.
    Bpe(BpeBuildError),
}

impl fmt::Display for EmbeddedTokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => write!(formatter, "embedded tokenizer model rejected: {error}"),
            Self::Bpe(error) => write!(
                formatter,
                "embedded tokenizer BPE surface rejected: {error}"
            ),
        }
    }
}

impl Error for EmbeddedTokenizerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Bpe(error) => Some(error),
        }
    }
}

/// Fail-closed errors while binding the binary tokenizer to one checked
/// `.fnlpq` artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenizerArtifactIntegrityError {
    /// A checked artifact unexpectedly has no `TOKENIZER_MODEL` section.
    MissingTokenizerModelSection,
    /// A checked directory entry could not be resolved back to its stored bytes.
    UnavailableTokenizerModelSection { ordinal: u64 },
    /// The binary and artifact carry different tokenizer byte strings.
    DigestMismatch {
        binary_sha256: [u8; 32],
        artifact_sha256: [u8; 32],
    },
}

impl fmt::Display for TokenizerArtifactIntegrityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTokenizerModelSection => {
                formatter.write_str("checked artifact has no TOKENIZER_MODEL section")
            }
            Self::UnavailableTokenizerModelSection { ordinal } => write!(
                formatter,
                "checked artifact TOKENIZER_MODEL section ordinal={ordinal} has no readable stored bytes"
            ),
            Self::DigestMismatch {
                binary_sha256,
                artifact_sha256,
            } => write!(
                formatter,
                "tokenizer digest mismatch binary_sha256={} artifact_sha256={}; next=use fnlp pull for a matching artifact or install a matching binary release",
                digest_hex(binary_sha256),
                digest_hex(artifact_sha256),
            ),
        }
    }
}

impl Error for TokenizerArtifactIntegrityError {}

/// The binary's exact tokenizer bytes, BPE engine, and immutable digest.
///
/// Construct this once from an `include_bytes!` static.  No user-controlled
/// path participates in this constructor, preventing accidental drift from the
/// binary's release-bound tokenizer authority.
#[derive(Debug)]
pub struct EmbeddedTokenizer {
    bytes: &'static [u8],
    sha256: [u8; 32],
    tokenizer: SpBpeTokenizer,
}

impl EmbeddedTokenizer {
    /// Parse and retain exactly the binary-embedded tokenizer bytes.
    pub fn from_bytes(bytes: &'static [u8]) -> Result<Self, EmbeddedTokenizerError> {
        let model = parse_spm_model(bytes).map_err(EmbeddedTokenizerError::Model)?;
        let tokenizer = SpBpeTokenizer::from_model(model).map_err(EmbeddedTokenizerError::Bpe)?;
        Ok(Self {
            bytes,
            sha256: Sha256::digest(bytes).into(),
            tokenizer,
        })
    }

    /// The immutable upstream `tokenizer.model` bytes compiled into this binary.
    #[must_use]
    pub const fn bytes(&self) -> &'static [u8] {
        self.bytes
    }

    /// SHA-256 of [`Self::bytes`].
    #[must_use]
    pub const fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    /// Lowercase hexadecimal SHA-256 of [`Self::bytes`].
    #[must_use]
    pub fn sha256_hex(&self) -> String {
        digest_hex(&self.sha256)
    }

    /// The L0 BPE engine built from exactly [`Self::bytes`].
    #[must_use]
    pub const fn tokenizer(&self) -> &SpBpeTokenizer {
        &self.tokenizer
    }

    /// Refuse an artifact whose checked `TOKENIZER_MODEL` section differs from
    /// the tokenizer embedded in this executable.
    pub fn verify_artifact(
        &self,
        artifact: &FnlpqArtifact,
    ) -> Result<(), TokenizerArtifactIntegrityError> {
        let section = artifact
            .sections()
            .iter()
            .find(|section| section.kind == SectionKind::TokenizerModel)
            .ok_or(TokenizerArtifactIntegrityError::MissingTokenizerModelSection)?;
        let bytes = artifact.section_bytes(section.ordinal).ok_or(
            TokenizerArtifactIntegrityError::UnavailableTokenizerModelSection {
                ordinal: section.ordinal,
            },
        )?;
        let artifact_sha256: [u8; 32] = Sha256::digest(bytes).into();
        if artifact_sha256 != self.sha256 {
            return Err(TokenizerArtifactIntegrityError::DigestMismatch {
                binary_sha256: self.sha256,
                artifact_sha256,
            });
        }
        Ok(())
    }

    /// Open and validate an artifact, then bind its tokenizer bytes to this
    /// executable's embedded tokenizer.
    pub fn verify_artifact_path(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), VerifyArtifactTokenizerError> {
        let artifact =
            FnlpqArtifact::open_owned(path).map_err(VerifyArtifactTokenizerError::Read)?;
        self.verify_artifact(&artifact)
            .map_err(VerifyArtifactTokenizerError::Integrity)
    }
}

/// Full error boundary for a requested artifact path.
#[derive(Debug)]
pub enum VerifyArtifactTokenizerError {
    /// The `.fnlpq` reader refused the supplied path before tokenizer use.
    Read(FnlpqReadError),
    /// A validated artifact carries a conflicting tokenizer payload.
    Integrity(TokenizerArtifactIntegrityError),
}

impl fmt::Display for VerifyArtifactTokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(error) => error.fmt(formatter),
            Self::Integrity(error) => error.fmt(formatter),
        }
    }
}

impl Error for VerifyArtifactTokenizerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Read(error) => Some(error),
            Self::Integrity(error) => Some(error),
        }
    }
}

fn digest_hex(digest: &[u8; 32]) -> String {
    use fmt::Write as _;

    let mut encoded = String::with_capacity(64);
    for byte in digest {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}
