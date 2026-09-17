//! Source-anchored corpus entity resolution (plan section 7.2).
//!
//! Lexical similarity ONLY generates candidates. Frozen bidirectional pair
//! scores authorize edges; deterministic complete-link clustering refuses a
//! merge unless EVERY cross-component pair is explicitly Same. Thus A=B and
//! B=C cannot silently override A!=C, an abstention, or an unexamined A/C pair.
//! No normalization rewrites source offsets; no public content digest or
//! cross-snapshot entity identity is minted by this module.

use std::{collections::{BTreeMap, BTreeSet}, error::Error, fmt, io};
use serde::{Deserialize, Serialize};
use crate::{canonjson, execution_identity::Sha256Digest,
    native_engine::decode::{DecodeCancellationKind, DecodeStepControl},
    tasks::ir::{DependencyScope, ScoreSpace},
    validation::{SourceSpan, validate_source_span, grounded_fields::VerifiedSourceSpan}};

pub const RESOLVE_VERSION: &str = "resolve-v1";
pub const BLOCKING_VERSION: &str = "exact-type-ascii-lexical-v1";
pub const CLUSTERING_VERSION: &str = "two-order-margin-complete-link-v1";

/// Original source is required: caller-provided offsets are never evidence by
/// themselves. Requests deliberately omit Debug to avoid accidental text logs.
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResolutionDocument {
    pub id: String,
    pub text: String,
    pub mentions: Vec<MentionInput>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MentionInput {
    pub entity_type: String,
    pub surface: String,
    pub span: VerifiedSourceSpan,
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockingPolicy {
    /// Byte-exact surface and type; no Unicode equivalence is implied.
    ExactSurface,
    /// ASCII letter folding, ASCII words, exact non-ASCII runs and initialisms.
    /// A recall heuristic, NOT NFKC, Unicode case-folding or a match decision.
    AsciiWordOverlap,
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResolveOptions {
    pub blocking: BlockingPolicy,
    pub context_scalars: usize,
    /// Each presentation order must independently exceed both alternatives.
    /// This is a caller-selected uncalibrated log-probability margin, not risk.
    pub minimum_margin_milli: u32,
}
impl Default for ResolveOptions {
    fn default() -> Self {
        Self { blocking: BlockingPolicy::AsciiWordOverlap, context_scalars: 128, minimum_margin_milli: 1000 }
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResolveLimits {
    pub max_documents: usize,
    pub max_mentions: usize,
    pub max_input_bytes: usize,
    pub max_surface_bytes: usize,
    pub max_keys_per_mention: usize,
    pub max_block_members: usize,
    pub max_candidate_pairs: usize,
    /// Includes duplicate pairs encountered through multiple lexical blocks.
    pub max_pair_visits: u64,
    pub max_cluster_checks: u64,
    pub max_scan_steps: u64,
    pub max_result_bytes: usize,
}
impl Default for ResolveLimits {
    fn default() -> Self {
        Self { max_documents: 4096, max_mentions: 16_384, max_input_bytes: 16 * 1024 * 1024,
            max_surface_bytes: 1024, max_keys_per_mention: 32, max_block_members: 512,
            max_candidate_pairs: 100_000, max_pair_visits: 1_000_000,
            max_cluster_checks: 10_000_000, max_scan_steps: 256 * 1024 * 1024,
            max_result_bytes: 16 * 1024 * 1024 }
    }
}
impl ResolveLimits {
    fn validate(self, options: ResolveOptions) -> Result<(), ResolveError> {
        if !(1..=65_536).contains(&self.max_documents) || !(1..=65_536).contains(&self.max_mentions)
            || !(1..=64 * 1024 * 1024).contains(&self.max_input_bytes)
            || !(1..=16_384).contains(&self.max_surface_bytes) || !(1..=128).contains(&self.max_keys_per_mention)
            || !(1..=4096).contains(&self.max_block_members) || self.max_candidate_pairs > 1_000_000
            || self.max_pair_visits == 0 || self.max_cluster_checks == 0 || self.max_scan_steps == 0
            || !(1..=64 * 1024 * 1024).contains(&self.max_result_bytes)
            || options.context_scalars > 2048 || !(1..=1_000_000).contains(&options.minimum_margin_milli)
        { return Err(ResolveError::InvalidLimits); }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolveError {
    InvalidLimits, InvalidInput, InputBudget, InvalidAnchor, BlockBudget, PairBudget,
    ScanBudget, IncompleteScores, ForeignScores, InvalidScores, ClusterBudget,
    OutputBudget, AllocationRefused, Serialization, Cancelled(DecodeCancellationKind),
}
impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLimits => "invalid resolution limits or margin policy",
            Self::InvalidInput => "resolution requires unique bounded document ids and anchored mentions",
            Self::InputBudget => "resolution input exceeds its aggregate budget",
            Self::InvalidAnchor => "resolution mention does not match the exact source coordinates",
            Self::BlockBudget => "resolution lexical block exceeds its complete candidate budget",
            Self::PairBudget => "resolution candidate enumeration exceeds its budget",
            Self::ScanBudget => "resolution source validation exceeds its scan budget",
            Self::IncompleteScores => "resolution requires each candidate assessment exactly once",
            Self::ForeignScores => "resolution assessments belong to another snapshot or policy",
            Self::InvalidScores => "resolution requires finite full-vocabulary sequence log probabilities",
            Self::ClusterBudget => "resolution complete-link checks exceed their budget",
            Self::OutputBudget => "complete resolution result exceeds its byte budget",
            Self::AllocationRefused => "resolution allocation refused",
            Self::Serialization => "resolution canonical serialization failed",
            Self::Cancelled(_) => "resolution cancelled",
        })
    }
}
impl Error for ResolveError {}

#[derive(Serialize)]
pub struct AnchoredMention<'a> {
    pub document_id: &'a str,
    pub entity_type: &'a str,
    pub surface: &'a str,
    pub span: VerifiedSourceSpan,
    #[serde(skip)]
    context: &'a str,
}
impl AnchoredMention<'_> {
    pub fn context(&self) -> &str { self.context }
}
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct CandidatePair { pub left: usize, pub right: usize }
impl CandidatePair {
    fn sorted(a: usize, b: usize) -> Self { Self { left: a.min(b), right: a.max(b) } }
}
/// A private snapshot-bound work ticket. Completion order is irrelevant. Its
/// binding is never serialized into public output or accepted from input JSON.
pub struct ResolutionPair<'p, 's> {
    pair: CandidatePair,
    binding: Sha256Digest,
    left: &'p AnchoredMention<'s>,
    right: &'p AnchoredMention<'s>,
}
impl<'p, 's> ResolutionPair<'p, 's> {
    pub fn indices(&self) -> CandidatePair { self.pair }
    pub fn left(&self) -> &AnchoredMention<'s> { self.left }
    pub fn right(&self) -> &AnchoredMention<'s> { self.right }
    /// Trusted scoring boundary, not proof of model execution. The native
    /// adapter computes these values and validates its complete score receipt.
    pub fn finish(self, scores: BidirectionalScores) -> FrozenPairScore {
        FrozenPairScore { pair: self.pair, binding: self.binding, scores }
    }
}
pub struct FrozenPairScore { pair: CandidatePair, binding: Sha256Digest, scores: BidirectionalScores }
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PairLogProbabilities { pub same: f64, pub different: f64, pub uncertain: f64 }
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct BidirectionalScores {
    pub forward: PairLogProbabilities,
    pub reverse: PairLogProbabilities,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PairDecision { Same, Different, Uncertain }
#[derive(Serialize)]
pub struct PairJudgment {
    pub pair: CandidatePair,
    pub scores: BidirectionalScores,
    pub decision: PairDecision,
    pub minimum_order_margin: f64,
}

pub struct ResolutionPlan<'s> {
    mentions: Vec<AnchoredMention<'s>>,
    pairs: Vec<CandidatePair>,
    binding: Sha256Digest,
    options: ResolveOptions,
    limits: ResolveLimits,
    document_count: usize,
    pair_visits: u64,
}
impl<'s> ResolutionPlan<'s> {
    pub fn prepare<C: DecodeStepControl>(documents: &'s [ResolutionDocument], options: ResolveOptions,
        limits: ResolveLimits, control: &mut C) -> Result<Self, ResolveError> {
        limits.validate(options)?; checkpoint(control)?;
        if documents.len() > limits.max_documents { return Err(ResolveError::InputBudget); }
        let mut ordered = reserved(documents.len())?; ordered.extend(documents);
        ordered.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        let (mut bytes, mut count) = (0_usize, 0_usize);
        for (i, document) in ordered.iter().enumerate() {
            if !identifier(&document.id, 256) || (i > 0 && ordered[i - 1].id == document.id) {
                return Err(ResolveError::InvalidInput);
            }
            bytes = add(bytes, add(document.id.len(), document.text.len())?)?;
            count = add(count, document.mentions.len())?;
            for mention in &document.mentions {
                bytes = add(bytes, add(mention.entity_type.len(), mention.surface.len())?)?;
            }
            if count > limits.max_mentions || bytes > limits.max_input_bytes { return Err(ResolveError::InputBudget); }
        }
        let mut mentions = reserved(count)?; let mut scan = limits.max_scan_steps;
        let mut manifest = reserved(ordered.len())?;
        for document in ordered {
            checkpoint(control)?;
            manifest.push((document.id.as_str(), Sha256Digest::of_bytes(document.text.as_bytes())));
            for mention in &document.mentions {
                checkpoint(control)?;
                if !identifier(&mention.entity_type, 64) || mention.surface.is_empty()
                    || mention.surface.len() > limits.max_surface_bytes || !mention.surface.chars().any(|c| !c.is_whitespace()) {
                    return Err(ResolveError::InvalidInput);
                }
                let s = mention.span;
                if s.byte_start >= s.byte_end || document.text.get(s.byte_start..s.byte_end) != Some(mention.surface.as_str()) {
                    return Err(ResolveError::InvalidAnchor);
                }
                let cost = (s.byte_end as u64).checked_mul(4).and_then(|n| n.checked_add(mention.surface.len() as u64))
                    .ok_or(ResolveError::ScanBudget)?;
                scan = scan.checked_sub(cost).ok_or(ResolveError::ScanBudget)?;
                validate_source_span(&document.text, &mention.surface,
                    SourceSpan::new(s.byte_start, s.byte_end, s.scalar_start, s.scalar_end)).map_err(|_| ResolveError::InvalidAnchor)?;
                let start = document.text[..s.byte_start].char_indices().rev().take(options.context_scalars)
                    .last().map_or(s.byte_start, |(i, _)| i);
                let end = document.text[s.byte_end..].char_indices().take(options.context_scalars)
                    .last().map_or(s.byte_end, |(i, c)| s.byte_end + i + c.len_utf8());
                mentions.push(AnchoredMention { document_id: &document.id, entity_type: &mention.entity_type,
                    surface: &mention.surface, span: s, context: &document.text[start..end] });
            }
        }
        mentions.sort_unstable_by(|a, b| mention_key(a).cmp(&mention_key(b)));
        if mentions.windows(2).any(|w| mention_key(&w[0]) == mention_key(&w[1])) { return Err(ResolveError::InvalidInput); }
        // Raw content hashes are strictly private: this binding is only used
        // to prevent score/snapshot substitution, not as a public receipt key.
        let binding = Sha256Digest::of_bytes(&canonjson::canonical_bytes(
            &(RESOLVE_VERSION, BLOCKING_VERSION, CLUSTERING_VERSION, options, limits, &manifest, &mentions)
        ).map_err(|_| ResolveError::Serialization)?);
        let mut blocks: BTreeMap<(&str, String), Vec<usize>> = BTreeMap::new();
        for (index, mention) in mentions.iter().enumerate() {
            checkpoint(control)?;
            for key in lexical_keys(mention.surface, options.blocking, limits.max_keys_per_mention)? {
                let block = blocks.entry((mention.entity_type, key)).or_default();
                if block.len() == limits.max_block_members { return Err(ResolveError::BlockBudget); }
                block.try_reserve(1).map_err(|_| ResolveError::AllocationRefused)?; block.push(index);
            }
        }
        let mut pairs = BTreeSet::new(); let mut visits = 0_u64;
        for members in blocks.values() {
            for (i, &a) in members.iter().enumerate() {
                checkpoint(control)?;
                for &b in &members[i + 1..] {
                    visits = visits.checked_add(1).filter(|&n| n <= limits.max_pair_visits).ok_or(ResolveError::PairBudget)?;
                    let pair = CandidatePair::sorted(a, b);
                    if !pairs.contains(&pair) {
                        if pairs.len() == limits.max_candidate_pairs { return Err(ResolveError::PairBudget); }
                        pairs.insert(pair);
                    }
                }
            }
        }
        let mut canonical_pairs = reserved(pairs.len())?; canonical_pairs.extend(pairs);
        checkpoint(control)?;
        Ok(Self { mentions, pairs: canonical_pairs, binding, options, limits,
            document_count: documents.len(), pair_visits: visits })
    }
    pub fn mentions(&self) -> &[AnchoredMention<'s>] { &self.mentions }
    pub fn candidate_count(&self) -> usize { self.pairs.len() }
    pub fn options(&self) -> ResolveOptions { self.options }
    pub fn limits(&self) -> ResolveLimits { self.limits }
    pub fn pairs(&self) -> impl ExactSizeIterator<Item = ResolutionPair<'_, 's>> {
        self.pairs.iter().map(|&pair| ResolutionPair { pair, binding: self.binding,
            left: &self.mentions[pair.left], right: &self.mentions[pair.right] })
    }
    /// Freeze ALL assessments before clustering. No partial-prefix success,
    /// greedy arrival-time union, unknown-pair inference or implicit retry.
    pub fn finalize<C: DecodeStepControl>(&self, mut scores: Vec<FrozenPairScore>, control: &mut C)
        -> Result<ResolutionResult, ResolveError> {
        checkpoint(control)?;
        if scores.len() != self.pairs.len() { return Err(ResolveError::IncompleteScores); }
        scores.sort_unstable_by_key(|s| s.pair);
        let mut judgments = reserved(scores.len())?;
        for (score, expected) in scores.into_iter().zip(&self.pairs) {
            checkpoint(control)?;
            if score.binding != self.binding { return Err(ResolveError::ForeignScores); }
            if score.pair != *expected { return Err(ResolveError::IncompleteScores); }
            let (decision, margin) = decide(score.scores, self.options.minimum_margin_milli)?;
            judgments.push(PairJudgment { pair: score.pair, scores: score.scores, decision, minimum_order_margin: margin });
        }
        let (clusters, blocked_merges, cluster_checks) = cluster(self.mentions.len(), &judgments,
            self.limits.max_cluster_checks, control)?;
        let mut mentions = reserved(self.mentions.len())?;
        for (index, m) in self.mentions.iter().enumerate() {
            mentions.push(ResolvedMention { index, document_id: copy(m.document_id)?, entity_type: copy(m.entity_type)?,
                surface: copy(m.surface)?, span: m.span });
        }
        let result = ResolutionResult { schema_version: 1, task_spec_version: RESOLVE_VERSION.to_owned(),
            blocking_version: BLOCKING_VERSION.to_owned(), clustering_version: CLUSTERING_VERSION.to_owned(),
            options: self.options, dependency_scope: DependencyScope::CorpusGlobal,
            score_space: ScoreSpace::FullVocabSequenceLogprob, calibration: "uncalibrated".to_owned(),
            document_count: self.document_count, mentions, clusters, judgments, blocked_merges,
            candidate_pair_visits: self.pair_visits, cluster_checks,
            warnings: ["lexical_blocking_may_miss_aliases", "pair_scores_are_not_identity_confidence",
                "complete_link_is_conservative", "cluster_ids_are_snapshot_local"], untrusted_fields: ["mentions"] };
        check_output(&result, self.limits.max_result_bytes)?; checkpoint(control)?; Ok(result)
    }
}
fn mention_key<'a>(m: &'a AnchoredMention<'_>) -> (&'a str, usize, &'a str, &'a str, usize, usize, usize) {
    (m.document_id, m.span.byte_start, m.entity_type, m.surface, m.span.byte_end, m.span.scalar_start, m.span.scalar_end)
}
fn identifier(s: &str, max: usize) -> bool { !s.is_empty() && s.len() <= max && !s.chars().any(char::is_control) }
fn lexical_keys(surface: &str, policy: BlockingPolicy, maximum: usize) -> Result<BTreeSet<String>, ResolveError> {
    let mut keys = BTreeSet::new();
    if policy == BlockingPolicy::ExactSurface { keys.insert(copy(surface)?); return Ok(keys); }
    // ASCII-only transformations are deliberately named. Non-ASCII scalar
    // values are retained exactly, not passed through platform Unicode tables.
    let normalized: String = surface.chars().map(|c| if c.is_ascii() { c.to_ascii_lowercase() } else { c }).collect();
    let words: Vec<_> = normalized.split(|c: char| c.is_ascii() && !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty()).collect();
    keys.insert(format!("s:{normalized}"));
    for word in &words {
        if word.len() >= 3 || !word.is_ascii() { keys.insert(format!("w:{word}")); }
        if keys.len() > maximum { return Err(ResolveError::BlockBudget); }
    }
    if (2..=8).contains(&words.len()) && words.iter().all(|w| w.bytes().all(|b| b.is_ascii_alphabetic())) {
        let acronym: String = words.iter().map(|w| char::from(w.as_bytes()[0])).collect();
        keys.insert(format!("w:{acronym}"));
    }
    if keys.len() > maximum { return Err(ResolveError::BlockBudget); }
    Ok(keys)
}
fn classify_order(p: PairLogProbabilities, threshold: f64) -> Result<(PairDecision, f64), ResolveError> {
    let values = [p.same, p.different, p.uncertain];
    if values.iter().any(|s| !s.is_finite() || *s > 0.0)
        || values.iter().map(|s| s.exp()).sum::<f64>() > 1.0 + 1e-9 { return Err(ResolveError::InvalidScores); }
    let same = p.same - p.different.max(p.uncertain);
    let different = p.different - p.same.max(p.uncertain);
    Ok(if same >= threshold { (PairDecision::Same, same) }
        else if different >= threshold { (PairDecision::Different, different) }
        else { (PairDecision::Uncertain, 0.0) })
}
fn decide(s: BidirectionalScores, threshold_milli: u32) -> Result<(PairDecision, f64), ResolveError> {
    let threshold = f64::from(threshold_milli) / 1000.0;
    let a = classify_order(s.forward, threshold)?; let b = classify_order(s.reverse, threshold)?;
    Ok(if a.0 == b.0 { (a.0, a.1.min(b.1)) } else { (PairDecision::Uncertain, 0.0) })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EntityCluster {
    /// Ordinal in this exact snapshot. Not an ID usable across snapshots.
    pub id: usize,
    /// Canonical mention indices, sorted. Every mention belongs to one cluster.
    pub mentions: Vec<usize>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ResolvedMention {
    pub index: usize, pub document_id: String, pub entity_type: String,
    pub surface: String, pub span: VerifiedSourceSpan,
}
#[derive(Serialize)]
pub struct ResolutionResult {
    pub schema_version: u32, pub task_spec_version: String,
    pub blocking_version: String, pub clustering_version: String, pub options: ResolveOptions,
    pub dependency_scope: DependencyScope, pub score_space: ScoreSpace, pub calibration: String,
    pub document_count: usize, pub mentions: Vec<ResolvedMention>, pub clusters: Vec<EntityCluster>,
    pub judgments: Vec<PairJudgment>, pub blocked_merges: usize,
    pub candidate_pair_visits: u64, pub cluster_checks: u64,
    pub warnings: [&'static str; 4], pub untrusted_fields: [&'static str; 1],
}
fn cluster<C: DecodeStepControl>(count: usize, judgments: &[PairJudgment], maximum: u64, control: &mut C)
    -> Result<(Vec<EntityCluster>, usize, u64), ResolveError> {
    let mut owners = reserved(count)?; owners.extend(0..count);
    let mut members = reserved(count)?;
    for index in 0..count { members.push(vec![index]); }
    let mut edges = reserved(judgments.len())?;
    edges.extend(judgments.iter().filter(|j| j.decision == PairDecision::Same));
    edges.sort_unstable_by(|a, b| b.minimum_order_margin.total_cmp(&a.minimum_order_margin).then_with(|| a.pair.cmp(&b.pair)));
    let (mut blocked, mut checks) = (0, 0_u64);
    for edge in edges {
        checkpoint(control)?;
        let (a, b) = (owners[edge.pair.left], owners[edge.pair.right]);
        if a == b { continue; }
        let mut compatible = true;
        'cross: for &left in &members[a] {
            checkpoint(control)?;
            for &right in &members[b] {
                checks = checks.checked_add(1).filter(|&n| n <= maximum).ok_or(ResolveError::ClusterBudget)?;
                let pair = CandidatePair::sorted(left, right);
                if judgments.binary_search_by_key(&pair, |j| j.pair).ok()
                    .is_none_or(|i| judgments[i].decision != PairDecision::Same) {
                    compatible = false; break 'cross;
                }
            }
        }
        if !compatible { blocked += 1; continue; }
        let (keep, lose) = (a.min(b), a.max(b));
        let additional = members[lose].len();
        members[keep].try_reserve(additional).map_err(|_| ResolveError::AllocationRefused)?;
        let moved = std::mem::take(&mut members[lose]);
        for &index in &moved { owners[index] = keep; }
        members[keep].extend(moved); members[keep].sort_unstable();
    }
    let mut clusters = reserved(count)?;
    for group in members { if !group.is_empty() { clusters.push(EntityCluster { id: clusters.len(), mentions: group }); } }
    Ok((clusters, blocked, checks))
}
pub(super) fn checkpoint<C: DecodeStepControl>(control: &mut C) -> Result<(), ResolveError> {
    control.prefill_checkpoint(0).map_or(Ok(()), |c| Err(ResolveError::Cancelled(c)))
}
pub(super) fn reserved<T>(n: usize) -> Result<Vec<T>, ResolveError> {
    let mut v = Vec::new(); v.try_reserve_exact(n).map_err(|_| ResolveError::AllocationRefused)?; Ok(v)
}
fn add(a: usize, b: usize) -> Result<usize, ResolveError> { a.checked_add(b).ok_or(ResolveError::InputBudget) }
fn copy(s: &str) -> Result<String, ResolveError> {
    let mut result = String::new(); result.try_reserve_exact(s.len()).map_err(|_| ResolveError::AllocationRefused)?;
    result.push_str(s); Ok(result)
}
pub(super) fn check_output<T: Serialize>(value: &T, cap: usize) -> Result<(), ResolveError> {
    struct Counter { left: usize, overflow: bool }
    impl io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > self.left { self.overflow = true; return Err(io::Error::other("resolution output budget")); }
            self.left -= bytes.len(); Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    let mut counter = Counter { left: cap, overflow: false };
    if serde_json::to_writer(&mut counter, value).is_err() {
        return Err(if counter.overflow { ResolveError::OutputBudget } else { ResolveError::Serialization });
    }
    if canonjson::canonical_bytes(value).map_err(|_| ResolveError::Serialization)?.len() > cap {
        return Err(ResolveError::OutputBudget);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
