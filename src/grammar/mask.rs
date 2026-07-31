//! Vocabulary-trie token masks over exact tokenizer-emitted bytes.
//!
//! The grammar compiler is intentionally tokenizer-independent.  This module
//! joins a compiled byte-state with the approved SentencePiece tokenizer only
//! at constrained-decoding time.  It never treats a raw SentencePiece piece
//! string as emitted text: every token is first decoded with a byte-fallback
//! probe already in the stream, preserving dummy-prefix whitespace behavior.
//!
//! The cache here is deliberately schema-only.  Its key has no document
//! component and it accepts only a [`SchemaMaskKey`], so source products must
//! remain in request-owned state rather than entering this cross-request map.

use std::{collections::BTreeMap, error::Error, fmt};

use crate::tokenizer::bpe::{DecodeBytesError, EncodeError, SpBpeTokenizer};

use super::compiler::MASK_BYTES_PER_STATE;

/// A fixed-width legal-token bitset.  The Nanbeige logit width is supplied by
/// the artifact/model authority, so the 37 padded non-token rows stay false.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseTokenMask {
    vocab_size: usize,
    bytes: Vec<u8>,
}

impl DenseTokenMask {
    /// Allocate an all-illegal vocabulary mask.
    #[must_use]
    pub fn empty(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            bytes: vec![0; vocab_size.div_ceil(8)],
        }
    }

    /// Width of the model-logit vector this mask applies to.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Number of bytes charged to cache admission.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Checked legal-bit update for one model-logit row.
    pub fn set_legal(&mut self, token_id: u32) -> Result<(), MaskOracleError> {
        let index = usize::try_from(token_id).map_err(|_| MaskOracleError::TokenOutOfRange {
            token_id,
            vocab_size: self.vocab_size,
        })?;
        if index >= self.vocab_size {
            return Err(MaskOracleError::TokenOutOfRange {
                token_id,
                vocab_size: self.vocab_size,
            });
        }
        self.bytes[index / 8] |= 1 << (index % 8);
        Ok(())
    }

    /// Whether this model-logit row is legal at the materialized grammar state.
    #[must_use]
    pub fn contains(&self, token_id: u32) -> bool {
        let Ok(index) = usize::try_from(token_id) else {
            return false;
        };
        index < self.vocab_size && (self.bytes[index / 8] & (1 << (index % 8))) != 0
    }

    /// Iterate legal model-logit row ids in ascending order.
    pub fn legal_ids(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.vocab_size).filter_map(|index| {
            self.contains(u32::try_from(index).ok()?)
                .then(|| u32::try_from(index).expect("vocab width fits u32"))
        })
    }
}

/// A one-token byte transition.  A state must reject a byte as soon as it
/// would make the token continuation impossible; it need not be accepting
/// after that token because later tokens may complete the output.
pub trait ByteState: Clone {
    /// Consume one emitted byte, returning false on an impossible transition.
    fn consume(&mut self, byte: u8) -> bool;
}

#[derive(Clone, Debug, Default)]
struct TrieNode {
    children: BTreeMap<u8, usize>,
    terminal_ids: Vec<u32>,
}

/// Trie of actual bytes emitted by every decodable model-logit row.
///
/// `token_bytes(id)` is `None` for padded model rows and for zero-byte control
/// pieces.  The latter cannot advance a grammar and are deliberately not legal
/// in a one-token constrained-decoding step.
#[derive(Clone, Debug)]
pub struct VocabTrie {
    nodes: Vec<TrieNode>,
    token_bytes: Vec<Option<Vec<u8>>>,
    indexed_token_count: usize,
}

impl VocabTrie {
    /// Build the full model-width trie from tokenizer decoding semantics.
    ///
    /// A decoded byte-fallback NUL precedes each candidate token.  This makes
    /// a leading SentencePiece `▁` emit a real ASCII space instead of being
    /// mistaken for the decoder's suppressed dummy prefix.
    pub fn from_tokenizer(
        tokenizer: &SpBpeTokenizer,
        model_vocab_size: usize,
    ) -> Result<Self, VocabTrieBuildError> {
        if model_vocab_size == 0 {
            return Err(VocabTrieBuildError::ZeroVocabSize);
        }
        if u32::try_from(model_vocab_size - 1).is_err() {
            return Err(VocabTrieBuildError::VocabSizeOutOfRange { model_vocab_size });
        }
        let probe_ids = tokenizer
            .encode_byte_fallback_only(&[0])
            .map_err(VocabTrieBuildError::ProbeEncode)?;
        let [probe_id] = probe_ids.as_slice() else {
            return Err(VocabTrieBuildError::ProbeNotOneToken {
                token_count: probe_ids.len(),
            });
        };
        if tokenizer.decode_bytes(&[*probe_id]).as_deref() != Ok(&[0][..]) {
            return Err(VocabTrieBuildError::ProbeRoundTrip);
        }

        let mut trie = Self {
            nodes: vec![TrieNode::default()],
            token_bytes: vec![None; model_vocab_size],
            indexed_token_count: 0,
        };
        for index in 0..model_vocab_size {
            let token_id = u32::try_from(index)
                .expect("model vocabulary width was checked to fit u32 before allocation");
            let decoded = match tokenizer.decode_bytes(&[*probe_id, token_id]) {
                Ok(decoded) => decoded,
                Err(DecodeBytesError::UnknownTokenId { .. }) => continue,
            };
            let Some(emitted) = decoded.strip_prefix(&[0]) else {
                return Err(VocabTrieBuildError::ProbePrefixLost { token_id });
            };
            if emitted.is_empty() {
                continue;
            }
            let bytes = emitted.to_vec();
            trie.insert(token_id, &bytes);
            trie.token_bytes[index] = Some(bytes);
            trie.indexed_token_count += 1;
        }
        Ok(trie)
    }

    /// The configured model-logit width, including any non-token padding rows.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len()
    }

    /// Number of model-logit rows that emit nonempty bytes and enter the trie.
    #[must_use]
    pub const fn indexed_token_count(&self) -> usize {
        self.indexed_token_count
    }

    /// Number of shared byte-prefix nodes retained by the vocabulary trie.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Exact emitted bytes for one candidate model-logit row.
    #[must_use]
    pub fn token_bytes(&self, token_id: u32) -> Option<&[u8]> {
        usize::try_from(token_id)
            .ok()
            .and_then(|index| self.token_bytes.get(index))
            .and_then(Option::as_deref)
    }

    fn insert(&mut self, token_id: u32, bytes: &[u8]) {
        let mut node = 0;
        for &byte in bytes {
            let next = match self.nodes[node].children.get(&byte).copied() {
                Some(existing) => existing,
                None => {
                    let created = self.nodes.len();
                    self.nodes.push(TrieNode::default());
                    self.nodes[node].children.insert(byte, created);
                    created
                }
            };
            node = next;
        }
        self.nodes[node].terminal_ids.push(token_id);
    }
}

/// Bounds one materialization and provides a cancellation checkpoint cadence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaskWorkLimits {
    /// Total trie nodes a materialization may visit before it fails closed.
    pub max_trie_node_visits: usize,
    /// Invoke the caller checkpoint after this many node visits.
    pub checkpoint_interval_nodes: usize,
}

impl Default for MaskWorkLimits {
    fn default() -> Self {
        Self {
            max_trie_node_visits: 1_000_000,
            checkpoint_interval_nodes: 1_024,
        }
    }
}

/// Aggregate, source-free work observation passed to cancellation checkpoints.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaskWorkProgress {
    pub trie_node_visits: usize,
}

/// A vocabulary-trie mask oracle over one exact tokenizer surface.
#[derive(Clone, Debug)]
pub struct VocabMaskOracle {
    trie: VocabTrie,
}

impl VocabMaskOracle {
    #[must_use]
    pub const fn new(trie: VocabTrie) -> Self {
        Self { trie }
    }

    #[must_use]
    pub const fn trie(&self) -> &VocabTrie {
        &self.trie
    }

    /// Materialize a legal-token bitset by walking shared byte prefixes once.
    ///
    /// The callback is the required cancellation checkpoint: returning false
    /// aborts without returning a partial success mask.
    pub fn materialize<S, F>(
        &self,
        initial_state: &S,
        limits: MaskWorkLimits,
        mut checkpoint: F,
    ) -> Result<DenseTokenMask, MaskOracleError>
    where
        S: ByteState,
        F: FnMut(MaskWorkProgress) -> bool,
    {
        if limits.max_trie_node_visits == 0 || limits.checkpoint_interval_nodes == 0 {
            return Err(MaskOracleError::InvalidWorkLimits);
        }
        let mut mask = DenseTokenMask::empty(self.trie.vocab_size());
        let mut visits = 0;
        self.walk(
            0,
            initial_state.clone(),
            &mut mask,
            limits,
            &mut visits,
            &mut checkpoint,
        )?;
        Ok(mask)
    }

    fn walk<S, F>(
        &self,
        node_index: usize,
        state: S,
        mask: &mut DenseTokenMask,
        limits: MaskWorkLimits,
        visits: &mut usize,
        checkpoint: &mut F,
    ) -> Result<(), MaskOracleError>
    where
        S: ByteState,
        F: FnMut(MaskWorkProgress) -> bool,
    {
        *visits = visits
            .checked_add(1)
            .ok_or(MaskOracleError::WorkCounterOverflow)?;
        if *visits > limits.max_trie_node_visits {
            return Err(MaskOracleError::WorkBudgetExceeded {
                max_trie_node_visits: limits.max_trie_node_visits,
            });
        }
        if *visits % limits.checkpoint_interval_nodes == 0
            && !checkpoint(MaskWorkProgress {
                trie_node_visits: *visits,
            })
        {
            return Err(MaskOracleError::CancelledAtCheckpoint {
                trie_node_visits: *visits,
            });
        }
        let node = &self.trie.nodes[node_index];
        for &token_id in &node.terminal_ids {
            mask.set_legal(token_id)?;
        }
        for (&byte, &child) in &node.children {
            let mut child_state = state.clone();
            if child_state.consume(byte) {
                self.walk(child, child_state, mask, limits, visits, checkpoint)?;
            }
        }
        Ok(())
    }
}

/// An immutable, source-free identity for schema×tokenizer mask reuse.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SchemaMaskKey {
    pub tokenizer_digest: [u8; 32],
    pub schema_digest: [u8; 32],
    pub grammar_version: u32,
}

/// Aggregate cache state safe to expose in health/eviction telemetry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaskCacheStats {
    pub schema_entries: usize,
    pub state_masks: usize,
    pub used_bytes: usize,
    pub capacity_bytes: usize,
}

/// Byte-budgeted cache of immutable schema masks only.
#[derive(Clone, Debug)]
pub struct SchemaMaskCache {
    capacity_bytes: usize,
    used_bytes: usize,
    entries: BTreeMap<SchemaMaskKey, BTreeMap<u32, DenseTokenMask>>,
}

impl SchemaMaskCache {
    #[must_use]
    pub const fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: BTreeMap::new(),
        }
    }

    /// Return a copied mask without exposing internal cache storage.
    #[must_use]
    pub fn get(&self, key: SchemaMaskKey, schema_state_id: u32) -> Option<DenseTokenMask> {
        self.entries
            .get(&key)
            .and_then(|states| states.get(&schema_state_id))
            .cloned()
    }

    /// Admit one schema-only mask, replacing an existing same-key state mask.
    pub fn insert(
        &mut self,
        key: SchemaMaskKey,
        schema_state_id: u32,
        mask: DenseTokenMask,
    ) -> Result<(), MaskCacheError> {
        let replacing = self
            .entries
            .get(&key)
            .and_then(|states| states.get(&schema_state_id))
            .map(DenseTokenMask::byte_len)
            .unwrap_or(0);
        let prospective = self
            .used_bytes
            .checked_sub(replacing)
            .and_then(|used| used.checked_add(mask.byte_len()))
            .ok_or(MaskCacheError::AccountingOverflow)?;
        if prospective > self.capacity_bytes {
            return Err(MaskCacheError::ByteBudgetExceeded {
                capacity_bytes: self.capacity_bytes,
                used_bytes: self.used_bytes,
                requested_bytes: mask.byte_len(),
                replacing_bytes: replacing,
            });
        }
        self.entries
            .entry(key)
            .or_default()
            .insert(schema_state_id, mask);
        self.used_bytes = prospective;
        Ok(())
    }

    /// Source-free aggregate telemetry; no document-derived keys or values are
    /// representable in this cache API.
    #[must_use]
    pub fn stats(&self) -> MaskCacheStats {
        MaskCacheStats {
            schema_entries: self.entries.len(),
            state_masks: self.entries.values().map(BTreeMap::len).sum(),
            used_bytes: self.used_bytes,
            capacity_bytes: self.capacity_bytes,
        }
    }
}

/// Vocab-trie construction refused the claimed tokenizer/model surface.
#[derive(Debug)]
pub enum VocabTrieBuildError {
    ZeroVocabSize,
    VocabSizeOutOfRange { model_vocab_size: usize },
    ProbeEncode(EncodeError),
    ProbeNotOneToken { token_count: usize },
    ProbeRoundTrip,
    ProbePrefixLost { token_id: u32 },
}

impl fmt::Display for VocabTrieBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroVocabSize => {
                formatter.write_str("vocab trie requires a nonzero model vocabulary size")
            }
            Self::VocabSizeOutOfRange { model_vocab_size } => {
                write!(
                    formatter,
                    "model vocabulary size {model_vocab_size} does not fit token ids"
                )
            }
            Self::ProbeEncode(error) => {
                write!(formatter, "vocab-trie byte probe could not encode: {error}")
            }
            Self::ProbeNotOneToken { token_count } => write!(
                formatter,
                "vocab-trie byte probe must produce one byte-piece id, observed {token_count}"
            ),
            Self::ProbeRoundTrip => {
                formatter.write_str("vocab-trie byte probe did not decode to its sentinel byte")
            }
            Self::ProbePrefixLost { token_id } => write!(
                formatter,
                "vocab-trie token id={token_id} lost the byte-probe prefix during exact decode"
            ),
        }
    }
}

impl Error for VocabTrieBuildError {}

/// Mask materialization refused an invalid state/budget rather than publishing
/// an incomplete legal set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaskOracleError {
    InvalidWorkLimits,
    WorkCounterOverflow,
    WorkBudgetExceeded { max_trie_node_visits: usize },
    CancelledAtCheckpoint { trie_node_visits: usize },
    TokenOutOfRange { token_id: u32, vocab_size: usize },
}

impl fmt::Display for MaskOracleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWorkLimits => formatter.write_str("mask work limits must both be nonzero"),
            Self::WorkCounterOverflow => {
                formatter.write_str("mask trie-node work counter overflowed")
            }
            Self::WorkBudgetExceeded {
                max_trie_node_visits,
            } => write!(
                formatter,
                "mask trie expansion exceeded node budget {max_trie_node_visits}"
            ),
            Self::CancelledAtCheckpoint { trie_node_visits } => write!(
                formatter,
                "mask trie expansion cancelled at node checkpoint {trie_node_visits}"
            ),
            Self::TokenOutOfRange {
                token_id,
                vocab_size,
            } => write!(
                formatter,
                "token id={token_id} is outside model vocabulary width {vocab_size}"
            ),
        }
    }
}

impl Error for MaskOracleError {}

/// Cache admission failures.  They contain only byte counts and immutable
/// schema-state ids, never source text or source-derived identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaskCacheError {
    AccountingOverflow,
    ByteBudgetExceeded {
        capacity_bytes: usize,
        used_bytes: usize,
        requested_bytes: usize,
        replacing_bytes: usize,
    },
}

impl fmt::Display for MaskCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AccountingOverflow => {
                formatter.write_str("mask-cache byte accounting overflowed")
            }
            Self::ByteBudgetExceeded {
                capacity_bytes,
                used_bytes,
                requested_bytes,
                replacing_bytes,
            } => write!(
                formatter,
                "mask-cache byte budget exceeded capacity={capacity_bytes} used={used_bytes} requested={requested_bytes} replacing={replacing_bytes}"
            ),
        }
    }
}

impl Error for MaskCacheError {}

/// The compiler's fixed Nanbeige dense-mask size is retained here as an
/// explicit cross-check for integrations that bind the standard model width.
pub const NANBEIGE_MASK_BYTES_PER_STATE: usize = MASK_BYTES_PER_STATE as usize;
