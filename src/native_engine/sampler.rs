//! Greedy and explicit seeded sampling.
//!
//! Sampling is addressed rather than streamed.  A draw is a pure function of
//! its effective seed and semantic position, so batching, reordering, and a
//! resume do not consume a shared RNG stream in a different order.  The
//! execution compiler owns construction of the private semantic request key;
//! this module deliberately never formats or logs that key.

use std::{cmp::Ordering, error::Error, fmt, str::FromStr};

use sha2::{Digest, Sha256};

use super::lmhead;

/// Version of the frozen draw, ranking, and interval rules in this module.
pub const SAMPLER_VERSION: &str = "fnlp-sampler-v1";
/// The Nanbeige sampling preset's bounded selection width.
pub const DEFAULT_TOP_K: usize = 20;
/// Largest top-k selection held entirely on the stack by [`TopK`].
pub const MAX_TOP_K: usize = DEFAULT_TOP_K;

/// A caller-provided or admission-generated 256-bit sampling seed.
///
/// Decimal CLI seeds are domain-separated before expansion so their byte
/// representation cannot be confused with a directly supplied 256-bit seed.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct Seed256([u8; 32]);

impl Seed256 {
    /// Expands a decimal `u64` as the frozen `fnlp-seed-u64-v1` construction.
    #[must_use]
    pub fn from_u64(value: u64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"fnlp-seed-u64-v1");
        hasher.update(value.to_be_bytes());
        Self(hasher.finalize().into())
    }

    /// Parses either 64 lowercase hexadecimal characters or a decimal `u64`.
    pub fn parse_cli(value: &str) -> Result<Self, SeedParseError> {
        if value.len() == 64 {
            let mut bytes = [0_u8; 32];
            for (index, destination) in bytes.iter_mut().enumerate() {
                let offset = index * 2;
                let high = hex_nibble(value.as_bytes()[offset])?;
                let low = hex_nibble(value.as_bytes()[offset + 1])?;
                *destination = (high << 4) | low;
            }
            return Ok(Self(bytes));
        }

        if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(SeedParseError::InvalidForm);
        }
        let parsed = value
            .parse::<u64>()
            .map_err(|_| SeedParseError::DecimalOutOfRange)?;
        Ok(Self::from_u64(parsed))
    }

    /// Returns the exact seed bytes used in addressed draws.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Encodes the effective seed for an explicitly opted-in result or receipt.
    #[must_use]
    pub fn to_lower_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut rendered = String::with_capacity(64);
        for byte in self.0 {
            rendered.push(char::from(HEX[usize::from(byte >> 4)]));
            rendered.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        rendered
    }
}

impl From<[u8; 32]> for Seed256 {
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl FromStr for Seed256 {
    type Err = SeedParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse_cli(value)
    }
}

impl fmt::Debug for Seed256 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Seed256(REDACTED)")
    }
}

/// Why a CLI seed was not a valid frozen seed representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SeedParseError {
    /// The input was neither lowercase hexadecimal nor unsigned decimal.
    InvalidForm,
    /// The decimal form was syntactically valid but larger than `u64`.
    DecimalOutOfRange,
}

impl fmt::Display for SeedParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidForm => formatter.write_str(
                "seed must be exactly 64 lowercase hexadecimal characters or a decimal u64",
            ),
            Self::DecimalOutOfRange => formatter.write_str("decimal seed is outside the u64 range"),
        }
    }
}

impl Error for SeedParseError {}

/// Canonical private digest of the complete semantic request.
///
/// This type intentionally implements neither `Debug` nor `Display`.  Its
/// producer belongs to the execution-identity/compiler boundary; it must not
/// be constructed from a caller-selected request id alone.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct StableRequestKey([u8; 32]);

impl StableRequestKey {
    /// Accepts the compiler's canonical internal request digest.
    #[must_use]
    pub const fn from_canonical_digest(digest: [u8; 32]) -> Self {
        Self(digest)
    }

    const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// The semantic coordinates of one random draw.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DrawAddress {
    /// Stable output-row address; this is not a physical batch index.
    pub sample_index: u64,
    /// Autoregressive position after the prompt.
    pub decode_step: u64,
    /// Draw within a decode step, for example after a rejection/resample.
    pub draw_index: u64,
}

impl DrawAddress {
    /// Creates a fully addressed draw coordinate.
    #[must_use]
    pub const fn new(sample_index: u64, decode_step: u64, draw_index: u64) -> Self {
        Self {
            sample_index,
            decode_step,
            draw_index,
        }
    }
}

/// Frozen SHA-256 digest for one addressable sampling draw.
#[must_use]
pub fn draw_digest(
    effective_seed: Seed256,
    stable_request_key: StableRequestKey,
    address: DrawAddress,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(SAMPLER_VERSION.as_bytes());
    update_length_framed(&mut hasher, effective_seed.as_bytes());
    update_length_framed(&mut hasher, stable_request_key.as_bytes());
    update_length_framed(&mut hasher, &address.sample_index.to_be_bytes());
    update_length_framed(&mut hasher, &address.decode_step.to_be_bytes());
    update_length_framed(&mut hasher, &address.draw_index.to_be_bytes());
    hasher.finalize().into()
}

/// Maps a frozen draw digest to the half-open interval `[0, 1)`.
#[must_use]
pub fn uniform_from_digest(digest: [u8; 32]) -> f64 {
    let prefix: [u8; 8] = digest[..8]
        .try_into()
        .expect("a SHA-256 digest always has an eight-byte prefix");
    let first_53_bits = u64::from_be_bytes(prefix) >> 11;
    (first_53_bits as f64) / ((1_u64 << 53) as f64)
}

/// Returns the deterministic uniform draw for the supplied semantic address.
#[must_use]
pub fn addressed_uniform(
    effective_seed: Seed256,
    stable_request_key: StableRequestKey,
    address: DrawAddress,
) -> f64 {
    uniform_from_digest(draw_digest(effective_seed, stable_request_key, address))
}

/// Full-logit greedy selection with no random authority or RNG state.
///
/// The NaN/Inf policy is exactly the existing `lmhead` total-order policy, so
/// this path remains bit-for-bit aligned with the native projection boundary.
#[must_use]
pub fn greedy_argmax(logits: &[f32]) -> Option<usize> {
    lmhead::greedy_argmax(logits)
}

/// A deterministically ranked logit entry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RankedToken {
    token_id: usize,
    logit: f32,
}

impl RankedToken {
    const PLACEHOLDER: Self = Self {
        token_id: 0,
        logit: 0.0,
    };

    /// Vocabulary position of this candidate.
    #[must_use]
    pub const fn token_id(self) -> usize {
        self.token_id
    }

    /// Pre-softmax f32 logit of this candidate.
    #[must_use]
    pub const fn logit(self) -> f32 {
        self.logit
    }
}

/// Fixed-capacity top-k result; the returned entries are best-to-worst.
#[derive(Clone, Debug, PartialEq)]
pub struct TopK {
    entries: [RankedToken; MAX_TOP_K],
    len: usize,
}

impl TopK {
    /// Performs a full-vocabulary scan using a fixed-size worst-first heap.
    ///
    /// Ranking is descending `f32` total order and then ascending token id.
    /// Sampling paths reject non-finite logits before ranking; greedy retains
    /// `lmhead`'s separately frozen total-order policy.
    pub fn select(logits: &[f32], k: usize) -> Result<Self, SamplerError> {
        validate_logits(logits)?;
        if !(1..=MAX_TOP_K).contains(&k) {
            return Err(SamplerError::InvalidTopK { requested: k });
        }

        let mut result = Self {
            entries: [RankedToken::PLACEHOLDER; MAX_TOP_K],
            len: 0,
        };
        for (token_id, logit) in logits.iter().copied().enumerate() {
            result.push_heap(RankedToken { token_id, logit }, k);
        }
        result.sort_descending();
        Ok(result)
    }

    /// Ranked candidates in best-to-worst order.
    #[must_use]
    pub fn as_slice(&self) -> &[RankedToken] {
        &self.entries[..self.len]
    }

    /// Samples the restricted top-k distribution with a half-open uniform.
    ///
    /// This path never allocates: the selected candidates, normalization, and
    /// cumulative selection all fit in the fixed [`MAX_TOP_K`] stack array.
    pub fn select_uniform(&self, uniform: f64) -> Result<usize, SamplerError> {
        validate_uniform(uniform)?;
        let entries = self.as_slice();
        let first = entries.first().ok_or(SamplerError::EmptyLogits)?;
        let normalizer = entries.iter().fold(0.0_f64, |sum, candidate| {
            sum + (f64::from(candidate.logit) - f64::from(first.logit)).exp()
        });
        let mut cumulative = 0.0;
        for candidate in entries {
            cumulative += (f64::from(candidate.logit) - f64::from(first.logit)).exp() / normalizer;
            if uniform < cumulative {
                return Ok(candidate.token_id);
            }
        }
        Ok(entries
            .last()
            .expect("a non-empty top-k result has a final candidate")
            .token_id)
    }

    fn push_heap(&mut self, candidate: RankedToken, capacity: usize) {
        if self.len < capacity {
            self.entries[self.len] = candidate;
            self.sift_up_worst(self.len);
            self.len += 1;
        } else if is_better(candidate, self.entries[0]) {
            self.entries[0] = candidate;
            self.sift_down_worst(0);
        }
    }

    fn sift_up_worst(&mut self, mut child: usize) {
        while child > 0 {
            let parent = (child - 1) / 2;
            if !is_worse(self.entries[child], self.entries[parent]) {
                break;
            }
            self.entries.swap(child, parent);
            child = parent;
        }
    }

    fn sift_down_worst(&mut self, mut parent: usize) {
        loop {
            let left = parent * 2 + 1;
            if left >= self.len {
                return;
            }
            let right = left + 1;
            let worst_child =
                if right < self.len && is_worse(self.entries[right], self.entries[left]) {
                    right
                } else {
                    left
                };
            if !is_worse(self.entries[worst_child], self.entries[parent]) {
                return;
            }
            self.entries.swap(parent, worst_child);
            parent = worst_child;
        }
    }

    fn sort_descending(&mut self) {
        for index in 1..self.len {
            let candidate = self.entries[index];
            let mut cursor = index;
            while cursor > 0 && is_better(candidate, self.entries[cursor - 1]) {
                self.entries[cursor] = self.entries[cursor - 1];
                cursor -= 1;
            }
            self.entries[cursor] = candidate;
        }
    }
}

/// An exact, full-normalizer nucleus distribution in deterministic rank order.
#[derive(Clone, Debug, PartialEq)]
pub struct Nucleus {
    entries: Vec<NucleusToken>,
    cutoff_mass: f64,
}

impl Nucleus {
    /// Full-vocabulary exact top-p, with no preceding top-k restriction.
    ///
    /// The maximum and normalizer are accumulated in increasing token-id order.
    /// Rank order is descending logit, then ascending token id.  The entry that
    /// crosses `top_p` is retained, and selection uses half-open intervals.
    pub fn exact_top_p(logits: &[f32], top_p: f64) -> Result<Self, SamplerError> {
        validate_logits(logits)?;
        if !(top_p.is_finite() && top_p > 0.0 && top_p <= 1.0) {
            return Err(SamplerError::InvalidTopP);
        }

        let maximum = logits
            .iter()
            .copied()
            .reduce(f32::max)
            .expect("non-empty logits were validated");
        let normalizer = logits.iter().fold(0.0_f64, |sum, logit| {
            sum + (f64::from(*logit) - f64::from(maximum)).exp()
        });
        debug_assert!(normalizer.is_finite() && normalizer > 0.0);

        let mut ranked: Vec<NucleusToken> = logits
            .iter()
            .copied()
            .enumerate()
            .map(|(token_id, logit)| NucleusToken {
                token: RankedToken { token_id, logit },
                probability: (f64::from(logit) - f64::from(maximum)).exp() / normalizer,
            })
            .collect();
        ranked.sort_unstable_by(|left, right| rank_order(left.token, right.token));

        let (cutoff_len, cutoff_mass) = if top_p == 1.0 {
            let full_mass = ranked
                .iter()
                .fold(0.0, |sum, candidate| sum + candidate.probability);
            (ranked.len(), full_mass)
        } else {
            let mut cutoff_len = 0;
            let mut cutoff_mass = 0.0;
            for candidate in &ranked {
                if cutoff_mass >= top_p {
                    break;
                }
                cutoff_mass += candidate.probability;
                cutoff_len += 1;
            }
            (cutoff_len, cutoff_mass)
        };
        ranked.truncate(cutoff_len);
        Ok(Self {
            entries: ranked,
            cutoff_mass,
        })
    }

    /// Entries retained by exact top-p, in frozen rank order.
    #[must_use]
    pub fn as_slice(&self) -> &[NucleusToken] {
        &self.entries
    }

    /// Probability mass retained before renormalization for selection.
    #[must_use]
    pub const fn cutoff_mass(&self) -> f64 {
        self.cutoff_mass
    }

    /// Selects using a uniform value in the half-open interval `[0, 1)`.
    pub fn select_uniform(&self, uniform: f64) -> Result<usize, SamplerError> {
        validate_uniform(uniform)?;
        let last = self.entries.last().ok_or(SamplerError::EmptyLogits)?;
        let mut cumulative = 0.0;
        for candidate in &self.entries {
            cumulative += candidate.probability / self.cutoff_mass;
            if uniform < cumulative {
                return Ok(candidate.token.token_id);
            }
        }
        // Rounded cumulative sums may end infinitesimally below one.  The
        // terminal fallback is therefore frozen rather than retrying a draw.
        Ok(last.token.token_id)
    }
}

/// A nucleus entry carrying its pre-renormalization full-vocabulary mass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NucleusToken {
    token: RankedToken,
    probability: f64,
}

impl NucleusToken {
    /// Candidate identity and original pre-softmax logit.
    #[must_use]
    pub const fn token(self) -> RankedToken {
        self.token
    }

    /// Exact-softmax probability before top-p renormalization.
    #[must_use]
    pub const fn probability(self) -> f64 {
        self.probability
    }
}

/// Sampling input rejected before a stochastic distribution is constructed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplerError {
    /// No vocabulary entries were supplied.
    EmptyLogits,
    /// A stochastic path refuses a NaN or infinity at the named vocabulary row.
    NonFiniteLogit { token_id: usize },
    /// The fixed stack-resident top-k heap has no such capacity.
    InvalidTopK { requested: usize },
    /// Top-p must be finite and in the closed interval `(0, 1]`.
    InvalidTopP,
    /// Nucleus selection requires a finite value in `[0, 1)`.
    InvalidUniform,
}

impl fmt::Display for SamplerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLogits => formatter.write_str("sampling requires at least one logit"),
            Self::NonFiniteLogit { token_id } => {
                write!(
                    formatter,
                    "sampling refuses a non-finite logit at token {token_id}"
                )
            }
            Self::InvalidTopK { requested } => write!(
                formatter,
                "top-k must be in 1..={MAX_TOP_K}; received {requested}"
            ),
            Self::InvalidTopP => formatter.write_str("top-p must be finite and in (0, 1]"),
            Self::InvalidUniform => formatter.write_str("uniform must be finite and in [0, 1)"),
        }
    }
}

impl Error for SamplerError {}

fn hex_nibble(value: u8) -> Result<u8, SeedParseError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(SeedParseError::InvalidForm),
    }
}

fn update_length_framed(hasher: &mut Sha256, value: &[u8]) {
    let length = u64::try_from(value.len()).expect("framed sampler inputs fit in u64");
    hasher.update(length.to_be_bytes());
    hasher.update(value);
}

fn validate_logits(logits: &[f32]) -> Result<(), SamplerError> {
    if logits.is_empty() {
        return Err(SamplerError::EmptyLogits);
    }
    if let Some((token_id, _)) = logits
        .iter()
        .enumerate()
        .find(|(_, logit)| !logit.is_finite())
    {
        return Err(SamplerError::NonFiniteLogit { token_id });
    }
    Ok(())
}

fn validate_uniform(uniform: f64) -> Result<(), SamplerError> {
    if !(uniform.is_finite() && (0.0..1.0).contains(&uniform)) {
        return Err(SamplerError::InvalidUniform);
    }
    Ok(())
}

fn rank_order(left: RankedToken, right: RankedToken) -> Ordering {
    right
        .logit
        .total_cmp(&left.logit)
        .then_with(|| left.token_id.cmp(&right.token_id))
}

fn is_better(left: RankedToken, right: RankedToken) -> bool {
    rank_order(left, right).is_lt()
}

fn is_worse(left: RankedToken, right: RankedToken) -> bool {
    rank_order(left, right).is_gt()
}
