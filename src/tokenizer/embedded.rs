//! Embedded tokenizer construction and `.fnlpq` consistency checks.
//!
//! The product binary supplies the exact pinned `tokenizer.model` bytes through
//! [`EmbeddedTokenizer::from_bytes`].  The bytes and their SHA-256 are held in
//! one value so every later `.fnlpq` artifact check compares the same tokenizer
//! that produced token ids.  The actual pinned byte inclusion belongs to the
//! truth-pack fixture closure; this module deliberately never substitutes a
//! fallback tokenizer when those bytes are unavailable. The embedded model
//! bytes are Apache-2.0 model material; the exact notice, Nanbeige attribution,
//! and modification notice remain in `docs/truth-pack/license/` and travel in
//! each checked artifact's license bundle.

use std::{collections::BTreeMap, error::Error, fmt};

use sha2::{Digest, Sha256};

use crate::artifact::{
    format::SectionKind,
    reader::{FnlpqArtifact, FnlpqRangeReader, FnlpqReadError},
};

use super::{
    bpe::{AddedToken, BpeBuildError, SpBpeTokenizer},
    sp_model::{SpmError, parse_spm_model},
};

/// Exact pinned SentencePiece bytes compiled into every product binary.
pub const PINNED_TOKENIZER_MODEL_BYTES: &[u8] =
    include_bytes!("../../assets/nanbeige4.2-3b/tokenizer.model");

/// Exact pinned post-SentencePiece token registry compiled into the binary.
pub const PINNED_ADDED_TOKENS_BYTES: &[u8] =
    include_bytes!("../../assets/nanbeige4.2-3b/added_tokens.json");

/// Exact tokenizer configuration bytes compiled with the tokenizer authority.
pub const PINNED_TOKENIZER_CONFIG_BYTES: &[u8] =
    include_bytes!("../../assets/nanbeige4.2-3b/tokenizer_config.json");

/// Exact special-token map bytes compiled with the tokenizer authority.
pub const PINNED_SPECIAL_TOKENS_MAP_BYTES: &[u8] =
    include_bytes!("../../assets/nanbeige4.2-3b/special_tokens_map.json");

/// Failures while turning the binary's embedded `tokenizer.model` bytes into
/// the exact SentencePiece BPE surface.
#[derive(Debug)]
pub enum EmbeddedTokenizerError {
    /// The owned minimal protobuf reader rejected the claimed tokenizer bytes.
    Model(SpmError),
    /// The exact added-token registry was not an object of token surface to ID.
    AddedTokenRegistry(serde_json::Error),
    /// The parsed model could not provide a safe BPE surface.
    Bpe(BpeBuildError),
    /// Pinned tokenizer_config/special-token assets disagree or are incomplete.
    Configuration(String),
}

impl fmt::Display for EmbeddedTokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => write!(formatter, "embedded tokenizer model rejected: {error}"),
            Self::AddedTokenRegistry(error) => {
                write!(formatter, "embedded added-token registry rejected: {error}")
            }
            Self::Bpe(error) => write!(
                formatter,
                "embedded tokenizer BPE surface rejected: {error}"
            ),
            Self::Configuration(detail) => write!(
                formatter,
                "embedded tokenizer configuration rejected: {detail}"
            ),
        }
    }
}

impl Error for EmbeddedTokenizerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::AddedTokenRegistry(error) => Some(error),
            Self::Bpe(error) => Some(error),
            Self::Configuration(_) => None,
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
    configured_bos_id: i32,
    configured_eos_id: i32,
}

impl EmbeddedTokenizer {
    /// Construct the product tokenizer from the pinned, compiled asset closure.
    pub fn pinned() -> Result<Self, EmbeddedTokenizerError> {
        let model = parse_spm_model(PINNED_TOKENIZER_MODEL_BYTES)
            .map_err(EmbeddedTokenizerError::Model)?;
        let added_by_surface: BTreeMap<String, u32> =
            serde_json::from_slice(PINNED_ADDED_TOKENS_BYTES)
                .map_err(EmbeddedTokenizerError::AddedTokenRegistry)?;
        let (configured_bos_id, configured_eos_id) =
            pinned_configured_ids(&added_by_surface)?;
        let tokenizer = SpBpeTokenizer::with_added_tokens_and_special_ids(
            model,
            added_by_surface
                .into_iter()
                .map(|(content, id)| AddedToken::new(content, id)),
            i32::try_from(configured_bos_id).map_err(|_| {
                EmbeddedTokenizerError::Configuration("BOS id exceeds i32".to_owned())
            })?,
            i32::try_from(configured_eos_id).map_err(|_| {
                EmbeddedTokenizerError::Configuration("EOS id exceeds i32".to_owned())
            })?,
        )
        .map_err(EmbeddedTokenizerError::Bpe)?;
        Ok(Self {
            bytes: PINNED_TOKENIZER_MODEL_BYTES,
            sha256: Sha256::digest(PINNED_TOKENIZER_MODEL_BYTES).into(),
            configured_bos_id: tokenizer.configured_bos_id(),
            configured_eos_id: tokenizer.configured_eos_id(),
            tokenizer,
        })
    }

    /// Parse and retain exactly the binary-embedded tokenizer bytes.
    ///
    /// This lower-level constructor is useful only for synthetic tests that
    /// have no post-SentencePiece token registry. Product call sites use
    /// [`Self::pinned`] so added-token precedence remains part of L0 behavior.
    pub fn from_bytes(bytes: &'static [u8]) -> Result<Self, EmbeddedTokenizerError> {
        Self::from_bytes_with_added_tokens(bytes, b"{}")
    }

    /// Parse immutable tokenizer bytes with their exact added-token registry.
    pub fn from_bytes_with_added_tokens(
        bytes: &'static [u8],
        added_tokens: &'static [u8],
    ) -> Result<Self, EmbeddedTokenizerError> {
        let model = parse_spm_model(bytes).map_err(EmbeddedTokenizerError::Model)?;
        let added_by_surface: BTreeMap<String, u32> = serde_json::from_slice(added_tokens)
            .map_err(EmbeddedTokenizerError::AddedTokenRegistry)?;
        let tokenizer = SpBpeTokenizer::with_added_tokens(
            model,
            added_by_surface
                .into_iter()
                .map(|(content, id)| AddedToken::new(content, id)),
        )
        .map_err(EmbeddedTokenizerError::Bpe)?;
        Ok(Self {
            bytes,
            sha256: Sha256::digest(bytes).into(),
            configured_bos_id: tokenizer.configured_bos_id(),
            configured_eos_id: tokenizer.configured_eos_id(),
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

    /// tokenizer_config/special-map BOS used by the product tokenizer.
    #[must_use]
    pub fn bos_token_id(&self) -> Option<u32> {
        u32::try_from(self.configured_bos_id).ok()
    }

    /// tokenizer_config/special-map EOS used by generation/scoring.
    #[must_use]
    pub fn eos_token_id(&self) -> Option<u32> {
        u32::try_from(self.configured_eos_id).ok()
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

    /// Open the bounded file-backed reader and bind the artifact tokenizer
    /// identity to this executable without retaining the complete model file.
    pub fn verify_artifact_path(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), VerifyArtifactTokenizerError> {
        let artifact =
            FnlpqRangeReader::open(path).map_err(VerifyArtifactTokenizerError::Read)?;
        let section = artifact
            .sections()
            .iter()
            .find(|section| section.kind == SectionKind::TokenizerModel)
            .ok_or(TokenizerArtifactIntegrityError::MissingTokenizerModelSection)
            .map_err(VerifyArtifactTokenizerError::Integrity)?;
        let artifact_sha256 = artifact
            .raw_section_sha256(section.ordinal)
            .map_err(VerifyArtifactTokenizerError::Read)?;
        if artifact_sha256 != self.sha256 {
            return Err(VerifyArtifactTokenizerError::Integrity(
                TokenizerArtifactIntegrityError::DigestMismatch {
                    binary_sha256: self.sha256,
                    artifact_sha256,
                },
            ));
        }
        Ok(())
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

fn pinned_configured_ids(
    added_by_surface: &BTreeMap<String, u32>,
) -> Result<(u32, u32), EmbeddedTokenizerError> {
    let config: serde_json::Value = serde_json::from_slice(PINNED_TOKENIZER_CONFIG_BYTES)
        .map_err(|error| EmbeddedTokenizerError::Configuration(format!("tokenizer_config JSON: {error}")))?;
    let special: serde_json::Value = serde_json::from_slice(PINNED_SPECIAL_TOKENS_MAP_BYTES)
        .map_err(|error| EmbeddedTokenizerError::Configuration(format!("special_tokens_map JSON: {error}")))?;
    let string = |root: &serde_json::Value, key: &str| -> Result<String, EmbeddedTokenizerError> {
        root.get(key).and_then(serde_json::Value::as_str).map(str::to_owned)
            .ok_or_else(|| EmbeddedTokenizerError::Configuration(format!("missing string {key}")))
    };
    let mapped_content = |key: &str| -> Result<String, EmbeddedTokenizerError> {
        special.get(key).and_then(serde_json::Value::as_object)
            .and_then(|object| object.get("content"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| EmbeddedTokenizerError::Configuration(format!("missing special_tokens_map {key}.content")))
    };
    if config.get("add_bos_token").and_then(serde_json::Value::as_bool) != Some(true)
        || config.get("add_eos_token").and_then(serde_json::Value::as_bool) != Some(false)
    {
        return Err(EmbeddedTokenizerError::Configuration(
            "EncodeOptions defaults disagree with tokenizer_config add_bos/add_eos".to_owned(),
        ));
    }
    let bos = string(&config, "bos_token")?;
    let eos = string(&config, "eos_token")?;
    if bos != mapped_content("bos_token")? || eos != mapped_content("eos_token")? {
        return Err(EmbeddedTokenizerError::Configuration(
            "tokenizer_config and special_tokens_map BOS/EOS surfaces disagree".to_owned(),
        ));
    }
    let bos_id = added_by_surface.get(&bos).copied().ok_or_else(|| {
        EmbeddedTokenizerError::Configuration(format!("configured BOS surface {bos:?} is absent from added-token registry"))
    })?;
    let eos_id = added_by_surface.get(&eos).copied().ok_or_else(|| {
        EmbeddedTokenizerError::Configuration(format!("configured EOS surface {eos:?} is absent from added-token registry"))
    })?;
    Ok((bos_id, eos_id))
}

fn digest_hex(digest: &[u8; 32]) -> String {
    use fmt::Write as _;

    let mut encoded = String::with_capacity(64);
    for byte in digest {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}
