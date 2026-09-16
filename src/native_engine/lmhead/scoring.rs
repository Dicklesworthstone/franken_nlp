//! Bounded, exact finite-continuation scoring (plan section 6.10).
//!
//! This is a no-spawn leaf. The caller owns model/KV admission and supplies
//! logits for the exact prompt plus the requested continuation prefix. No KV
//! snapshot or weight clone is made here. Prefix reuse in this trie does not
//! imply a qualified KV-fork implementation or a measured performance win.

use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};

use crate::tasks::ir::{Candidate, ScoreSpace};

use super::NANBEIGE_VOCAB_SIZE;

/// Terminal estimator used only by `sequence_score_softmax`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SequenceScoreRule {
    SumLogits,
    MeanLogits,
    TerminalLogit,
}

/// These are different score spaces, not interchangeable optimizations.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ScoringMode {
    FullVocabulary,
    TrieConditional,
    SequenceSoftmax { rule: SequenceScoreRule },
}

/// Explicit bounds checked before trie construction and before model calls.
#[derive(Clone, Copy, Debug)]
pub struct ScoringLimits {
    pub max_candidates: usize,
    /// Includes the additional EOS edge for every candidate.
    pub max_total_tokens: usize,
    pub max_nodes: usize,
    /// Includes EOS.
    pub max_depth: usize,
    pub max_candidate_id_bytes: usize,
    /// Full-vocabulary mode charges every denominator row, not selected rows.
    pub max_projected_logits: u64,
}

impl Default for ScoringLimits {
    fn default() -> Self {
        Self {
            max_candidates: 4096,
            max_total_tokens: 65_536,
            max_nodes: 65_537,
            max_depth: 1024,
            max_candidate_id_bytes: 1024,
            max_projected_logits: 100_000_000,
        }
    }
}

/// Rows the model must actually evaluate at a prefix.
#[derive(Clone, Copy, Debug)]
pub enum ProjectionRows<'a> {
    /// Return all raw logits in vocabulary-id order, including non-candidates.
    FullVocabulary { vocabulary_size: usize },
    /// Return raw logits in this exact, ascending token-id order.
    Selected(&'a [u32]),
}

/// Model-owned projection seam. `prefix` excludes the already-bound prompt.
///
/// Every call must refer to that same prompt/profile and exactly this prefix,
/// independent of previous calls. Implementations may use admitted KV reuse
/// or replay. They must not spawn, silently prune, or normalize the logits.
pub trait CandidateLogits {
    type Error: fmt::Display;

    fn project(
        &mut self,
        prefix: &[u32],
        rows: ProjectionRows<'_>,
    ) -> Result<Vec<f32>, Self::Error>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScoringError {
    InvalidVocabulary,
    LimitExceeded(&'static str),
    EmptyCandidateId,
    EmptyContinuation,
    DuplicateCandidateId,
    DuplicateContinuation,
    TokenOutOfRange(u32),
    EosInsideContinuation,
    AllocationRefused,
    ProjectionFailed,
    ProjectionLength { expected: usize, actual: usize },
    NonFiniteLogit { row: usize },
}

impl fmt::Display for ScoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never echo private candidate text, prefixes, or backend diagnostics.
        match self {
            Self::InvalidVocabulary => f.write_str("invalid scoring vocabulary or EOS id"),
            Self::LimitExceeded(axis) => write!(f, "candidate scoring limit exceeded: {axis}"),
            Self::EmptyCandidateId => f.write_str("candidate id must not be empty"),
            Self::EmptyContinuation => f.write_str("candidate continuation must not be empty"),
            Self::DuplicateCandidateId => f.write_str("duplicate candidate id"),
            Self::DuplicateContinuation => f.write_str("duplicate candidate token continuation"),
            Self::TokenOutOfRange(id) => write!(f, "candidate token {id} is out of range"),
            Self::EosInsideContinuation => f.write_str("EOS is reserved for explicit termination"),
            Self::AllocationRefused => f.write_str("candidate scoring allocation refused"),
            Self::ProjectionFailed => f.write_str("candidate model projection failed"),
            Self::ProjectionLength { expected, actual } => {
                write!(f, "candidate projection has {actual} rows; expected {expected}")
            }
            Self::NonFiniteLogit { row } => write!(f, "non-finite raw logit at row {row}"),
        }
    }
}

impl Error for ScoringError {}

/// A completed candidate, including its scored EOS edge.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateScore {
    pub id: String,
    pub scored_tokens: usize,
    /// Summed log probabilities in the two probability modes; the named raw
    /// terminal estimator in sequence-score mode. No length penalty is hidden.
    pub sequence_score: f64,
    /// Softmax of all completed `sequence_score` values. This is conditional
    /// on the complete candidate set, NEVER calibrated correctness confidence.
    pub candidate_weight: f64,
}

/// Deterministic work counts, separate from wall-clock telemetry.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScoringWork {
    pub prefix_evaluations: usize,
    pub scored_edges: usize,
    pub projected_logits: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateScores {
    pub score_space: ScoreSpace,
    pub normalization_scope: String,
    pub length_rule: String,
    pub eos_rule: String,
    pub eos_token_id: u32,
    pub full_vocab_denominators_computed: bool,
    /// Sorted by candidate id; ties can therefore be resolved independently
    /// of input order. No candidate is pruned, including underflowed weights.
    pub candidates: Vec<CandidateScore>,
    pub work: ScoringWork,
}

#[derive(Debug)]
struct Node {
    parent: Option<(usize, u32)>,
    edges: Vec<(u32, usize)>,
    terminal: Option<usize>,
}

/// A finite prefix trie whose terminals always follow a scored EOS edge.
#[derive(Debug)]
pub struct CandidateScorer {
    nodes: Vec<Node>,
    ids: Vec<String>,
    lengths: Vec<usize>,
    vocabulary_size: usize,
    eos: u32,
    max_depth: usize,
    max_projected_logits: u64,
}

fn reserved<T>(capacity: usize) -> Result<Vec<T>, ScoringError> {
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|_| ScoringError::AllocationRefused)?;
    Ok(values)
}

impl CandidateScorer {
    /// Compile a complete candidate language. Labels sharing a prefix are
    /// distinct because both the shorter label's EOS and the longer label's
    /// next token are real outgoing edges. Duplicate tokenizations are refused.
    pub fn compile(
        candidates: &[Candidate],
        vocabulary_size: usize,
        eos: u32,
        limits: ScoringLimits,
    ) -> Result<Self, ScoringError> {
        if vocabulary_size == 0 || vocabulary_size > NANBEIGE_VOCAB_SIZE
            || eos as usize >= vocabulary_size
        {
            return Err(ScoringError::InvalidVocabulary);
        }
        if candidates.is_empty() || candidates.len() > limits.max_candidates {
            return Err(ScoringError::LimitExceeded("candidate_count"));
        }
        let mut total = 0_usize;
        let mut max_depth = 0;
        for candidate in candidates {
            if candidate.id().is_empty() {
                return Err(ScoringError::EmptyCandidateId);
            }
            if candidate.id().len() > limits.max_candidate_id_bytes {
                return Err(ScoringError::LimitExceeded("candidate_id_bytes"));
            }
            let tokens = candidate.continuation().token_ids();
            if tokens.is_empty() {
                return Err(ScoringError::EmptyContinuation);
            }
            let depth = tokens.len().checked_add(1)
                .ok_or(ScoringError::LimitExceeded("depth"))?;
            if depth > limits.max_depth {
                return Err(ScoringError::LimitExceeded("depth"));
            }
            max_depth = max_depth.max(depth);
            total = total.checked_add(depth)
                .ok_or(ScoringError::LimitExceeded("total_tokens"))?;
            if total > limits.max_total_tokens {
                return Err(ScoringError::LimitExceeded("total_tokens"));
            }
            for &token in tokens {
                if token as usize >= vocabulary_size {
                    return Err(ScoringError::TokenOutOfRange(token));
                }
                if token == eos {
                    return Err(ScoringError::EosInsideContinuation);
                }
            }
        }
        // This is a conservative pre-allocation bound, not an allocation
        // followed by an after-the-fact admission check.
        let node_bound = total.checked_add(1)
            .ok_or(ScoringError::LimitExceeded("nodes"))?;
        if node_bound > limits.max_nodes {
            return Err(ScoringError::LimitExceeded("nodes"));
        }
        let mut order = reserved(candidates.len())?;
        order.extend(0..candidates.len());
        order.sort_unstable_by(|&a, &b| candidates[a].id().cmp(candidates[b].id()));
        if order.windows(2).any(|w| candidates[w[0]].id() == candidates[w[1]].id()) {
            return Err(ScoringError::DuplicateCandidateId);
        }
        let mut ids = reserved(candidates.len())?;
        let mut lengths = reserved(candidates.len())?;
        let mut ranks = reserved(candidates.len())?;
        ranks.resize(candidates.len(), 0);
        for (rank, &index) in order.iter().enumerate() {
            ids.push(candidates[index].id().to_owned());
            lengths.push(candidates[index].continuation().token_ids().len() + 1);
            ranks[index] = rank;
        }
        order.sort_unstable_by(|&a, &b| {
            candidates[a].continuation().token_ids().cmp(candidates[b].continuation().token_ids())
        });
        if order.windows(2).any(|w| {
            candidates[w[0]].continuation().token_ids() == candidates[w[1]].continuation().token_ids()
        }) {
            return Err(ScoringError::DuplicateContinuation);
        }
        let mut nodes = reserved(node_bound)?;
        nodes.push(Node { parent: None, edges: Vec::new(), terminal: None });
        for index in order {
            let mut node = 0;
            for token in candidates[index].continuation().token_ids().iter().copied()
                .chain(std::iter::once(eos))
            {
                node = match nodes[node].edges.binary_search_by_key(&token, |&(id, _)| id) {
                    Ok(edge) => nodes[node].edges[edge].1,
                    Err(edge) => {
                        let child = nodes.len();
                        nodes[node].edges.try_reserve(1).map_err(|_| ScoringError::AllocationRefused)?;
                        nodes.push(Node { parent: Some((node, token)), edges: Vec::new(), terminal: None });
                        nodes[node].edges.insert(edge, (token, child));
                        child
                    }
                };
            }
            nodes[node].terminal = Some(ranks[index]);
        }
        Ok(Self { nodes, ids, lengths, vocabulary_size, eos, max_depth,
            max_projected_logits: limits.max_projected_logits })
    }

    /// Score every terminal, evaluating each distinct nonterminal prefix once.
    /// Any failed projection rejects the whole result; no partial success.
    pub fn score<M: CandidateLogits>(
        &self,
        model: &mut M,
        mode: ScoringMode,
    ) -> Result<CandidateScores, ScoringError> {
        let prefix_evaluations = self.nodes.iter().filter(|n| !n.edges.is_empty()).count();
        let scored_edges = self.nodes.len() - 1;
        let projected_logits = if mode == ScoringMode::FullVocabulary {
            (prefix_evaluations as u64).checked_mul(self.vocabulary_size as u64)
                .ok_or(ScoringError::LimitExceeded("projected_logits"))?
        } else {
            scored_edges as u64
        };
        if projected_logits > self.max_projected_logits {
            return Err(ScoringError::LimitExceeded("projected_logits"));
        }
        let mut scores = reserved(self.nodes.len())?;
        scores.resize(self.nodes.len(), 0_f64);
        let mut terminals = reserved(self.ids.len())?;
        terminals.resize(self.ids.len(), 0_f64);
        let mut prefix = reserved(self.max_depth)?;
        let mut selected = reserved(self.ids.len())?;
        for (index, node) in self.nodes.iter().enumerate() {
            if let Some(rank) = node.terminal {
                terminals[rank] = match mode {
                    ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::MeanLogits } => {
                        scores[index] / self.lengths[rank] as f64
                    }
                    _ => scores[index],
                };
            }
            if node.edges.is_empty() {
                continue;
            }
            prefix.clear();
            let mut ancestor = index;
            while let Some((parent, token)) = self.nodes[ancestor].parent {
                prefix.push(token);
                ancestor = parent;
            }
            prefix.reverse();
            selected.clear();
            selected.extend(node.edges.iter().map(|&(token, _)| token));
            let rows = if mode == ScoringMode::FullVocabulary {
                ProjectionRows::FullVocabulary { vocabulary_size: self.vocabulary_size }
            } else {
                ProjectionRows::Selected(&selected)
            };
            let logits = model.project(&prefix, rows).map_err(|_| ScoringError::ProjectionFailed)?;
            let expected = if mode == ScoringMode::FullVocabulary {
                self.vocabulary_size
            } else {
                selected.len()
            };
            if logits.len() != expected {
                return Err(ScoringError::ProjectionLength { expected, actual: logits.len() });
            }
            for (row, value) in logits.iter().enumerate() {
                if !value.is_finite() {
                    return Err(ScoringError::NonFiniteLogit { row });
                }
            }
            let max = logits.iter().map(|&v| f64::from(v)).fold(f64::NEG_INFINITY, f64::max);
            let log_sum = logits.iter().map(|&v| (f64::from(v) - max).exp()).sum::<f64>().ln();
            for (row, &(token, child)) in node.edges.iter().enumerate() {
                let value = f64::from(logits[if mode == ScoringMode::FullVocabulary { token as usize } else { row }]);
                scores[child] = match mode {
                    ScoringMode::FullVocabulary | ScoringMode::TrieConditional => {
                        // Do not form max + log_sum: that loses log_sum for
                        // large finite equal logits, incorrectly returning 0.
                        scores[index] + ((value - max) - log_sum)
                    }
                    ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::TerminalLogit } => value,
                    ScoringMode::SequenceSoftmax { .. } => scores[index] + value,
                };
            }
        }
        let max = terminals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator = terminals.iter().map(|s| (s - max).exp()).sum::<f64>();
        let mut candidates = reserved(self.ids.len())?;
        for (rank, &sequence_score) in terminals.iter().enumerate() {
            candidates.push(CandidateScore {
                id: self.ids[rank].clone(),
                scored_tokens: self.lengths[rank],
                sequence_score,
                candidate_weight: (sequence_score - max).exp() / denominator,
            });
        }
        let (score_space, normalization_scope, length_rule) = match mode {
            ScoringMode::FullVocabulary => (ScoreSpace::FullVocabSequenceLogprob,
                "full_vocabulary_per_position; candidate_weights_over_completed_candidates", "sum_logprobs_including_eos"),
            ScoringMode::TrieConditional => (ScoreSpace::TrieLocalConditionalProbability,
                "legal_outgoing_edges_per_prefix; candidate_weights_over_completed_candidates", "sum_logprobs_including_eos"),
            ScoringMode::SequenceSoftmax { rule } => (ScoreSpace::SequenceScoreSoftmax,
                "softmax_over_completed_sequence_scores", match rule {
                    SequenceScoreRule::SumLogits => "sum_logits_including_eos",
                    SequenceScoreRule::MeanLogits => "mean_logits_including_eos",
                    SequenceScoreRule::TerminalLogit => "eos_logit_only",
                }),
        };
        Ok(CandidateScores {
            score_space,
            normalization_scope: normalization_scope.to_owned(),
            length_rule: length_rule.to_owned(),
            eos_rule: "append_and_score_exactly_one_eos".to_owned(),
            eos_token_id: self.eos,
            full_vocab_denominators_computed: mode == ScoringMode::FullVocabulary,
            candidates,
            work: ScoringWork { prefix_evaluations, scored_edges, projected_logits },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::ir::TokenSequence;

    fn candidate(id: &str, tokens: &[u32]) -> Candidate {
        Candidate::new(id, TokenSequence::new(tokens.to_vec()))
    }

    struct Toy { calls: Vec<Vec<u32>>, outside: f32 }
    impl CandidateLogits for Toy {
        type Error = &'static str;
        fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
            assert!(!self.calls.iter().any(|p| p == prefix), "prefix evaluated twice");
            self.calls.push(prefix.to_vec());
            let mut logits = vec![0.0, 0.0, 0.0, self.outside];
            if prefix == [1] { logits[0] = 2.0; }
            Ok(match rows {
                ProjectionRows::FullVocabulary { vocabulary_size } => {
                    assert_eq!(vocabulary_size, logits.len()); logits
                }
                ProjectionRows::Selected(ids) => ids.iter().map(|&id| logits[id as usize]).collect(),
            })
        }
    }
    fn toy(outside: f32) -> Toy { Toy { calls: Vec::new(), outside } }
    fn scorer(c: &[Candidate]) -> CandidateScorer {
        CandidateScorer::compile(c, 4, 0, ScoringLimits::default()).unwrap()
    }

    #[test]
    fn shared_prefix_and_eos_are_scored_without_pruning() {
        let c = scorer(&[candidate("short", &[1]), candidate("long", &[1, 2])]);
        let mut model = toy(0.0);
        let result = c.score(&mut model, ScoringMode::TrieConditional).unwrap();
        assert_eq!(model.calls, vec![vec![], vec![1], vec![1, 2]]);
        assert_eq!(result.work.scored_edges, 4);
        assert_eq!(result.work.projected_logits, 4);
        assert_eq!(result.candidates[0].id, "long");
        assert!((result.candidates[1].candidate_weight - 2_f64.exp() / (2_f64.exp() + 1.0)).abs() < 1e-14);
        assert!(!result.full_vocab_denominators_computed);
    }

    #[test]
    fn candidate_order_does_not_change_results_or_evaluation_order() {
        let a = candidate("a", &[1]); let b = candidate("b", &[1, 2]);
        for mode in [ScoringMode::FullVocabulary, ScoringMode::TrieConditional,
            ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::SumLogits }] {
            let mut left = toy(0.0); let mut right = toy(0.0);
            assert_eq!(scorer(&[a.clone(), b.clone()]).score(&mut left, mode).unwrap(),
                scorer(&[b.clone(), a.clone()]).score(&mut right, mode).unwrap());
            assert_eq!(left.calls, right.calls);
        }
    }

    #[test]
    fn full_vocab_denominators_include_non_candidates() {
        let c = scorer(&[candidate("a", &[1]), candidate("b", &[2])]);
        let full = c.score(&mut toy(20.0), ScoringMode::FullVocabulary).unwrap();
        let local = c.score(&mut toy(20.0), ScoringMode::TrieConditional).unwrap();
        assert_eq!(full.work.projected_logits, 12);
        assert!(full.candidates.iter().all(|s| s.sequence_score < -30.0));
        assert!(local.candidates.iter().all(|s| (s.sequence_score + 2_f64.ln()).abs() < 1e-14));
    }

    #[test]
    fn every_full_vocab_score_matches_naive_teacher_forcing() {
        let candidates = [candidate("a", &[1]), candidate("b", &[1, 2]), candidate("c", &[2, 1])];
        let result = scorer(&candidates).score(&mut toy(0.0), ScoringMode::FullVocabulary).unwrap();
        for (input, scored) in candidates.iter().zip(&result.candidates) {
            let mut prefix = Vec::new(); let mut score = 0.0;
            for token in input.continuation().token_ids().iter().copied().chain(std::iter::once(0)) {
                let logits = toy(0.0).project(&prefix, ProjectionRows::FullVocabulary { vocabulary_size: 4 }).unwrap();
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                score += (f64::from(logits[token as usize]) - max)
                    - logits.iter().map(|&v| (f64::from(v) - max).exp()).sum::<f64>().ln();
                prefix.push(token);
            }
            assert!((score - scored.sequence_score).abs() < 1e-14);
        }
    }

    #[test]
    fn duplicates_empty_and_embedded_eos_refuse() {
        for (input, expected) in [
            (vec![candidate("a", &[1]), candidate("a", &[2])], ScoringError::DuplicateCandidateId),
            (vec![candidate("a", &[1]), candidate("b", &[1])], ScoringError::DuplicateContinuation),
            (vec![candidate("a", &[])], ScoringError::EmptyContinuation),
            (vec![candidate("a", &[0])], ScoringError::EosInsideContinuation),
            (vec![candidate("a", &[4])], ScoringError::TokenOutOfRange(4)),
        ] {
            assert_eq!(CandidateScorer::compile(&input, 4, 0, ScoringLimits::default()).unwrap_err(), expected);
        }
    }

    #[test]
    fn projection_budget_refuses_before_any_model_call() {
        let c = CandidateScorer::compile(&[candidate("a", &[1])], 4, 0,
            ScoringLimits { max_projected_logits: 7, ..ScoringLimits::default() }).unwrap();
        let mut model = toy(0.0);
        assert_eq!(c.score(&mut model, ScoringMode::FullVocabulary).unwrap_err(),
            ScoringError::LimitExceeded("projected_logits"));
        assert!(model.calls.is_empty());
    }

    struct Constant(f32);
    impl CandidateLogits for Constant {
        type Error = &'static str;
        fn project(&mut self, _: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
            let count = match rows { ProjectionRows::FullVocabulary { vocabulary_size } => vocabulary_size,
                ProjectionRows::Selected(ids) => ids.len() };
            Ok(vec![self.0; count])
        }
    }

    #[test]
    fn huge_equal_finite_logits_keep_the_log_normalizer() {
        let c = scorer(&[candidate("a", &[1]), candidate("b", &[2])]);
        let result = c.score(&mut Constant(f32::MAX), ScoringMode::FullVocabulary).unwrap();
        for score in result.candidates {
            assert!((score.sequence_score + 2.0 * 4_f64.ln()).abs() < 1e-14);
            assert_eq!(score.candidate_weight, 0.5);
        }
    }

    #[test]
    fn nonfinite_raw_logits_never_become_successful_scores() {
        let c = scorer(&[candidate("a", &[1])]);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(c.score(&mut Constant(value), ScoringMode::FullVocabulary),
                Err(ScoringError::NonFiniteLogit { .. })));
        }
    }
}
