//! Split-phase finite-language scoring for interleaved native execution.
//!
//! A cursor owns only bounded scalar scores, a prefix and outgoing token IDs.
//! It never retains raw logits, an activation, a KV fork or a second scorer.
//! The serial callback and cohort drivers use this SAME probability program.
use super::*;

/// Borrowed next projection. The monotonically increasing ordinal detects
/// stale/replayed submissions within this cursor, not model/task authenticity.
/// A host must still bind the exact prompt and model to its admitted request.
#[derive(Clone, Copy, Debug)]
pub struct CandidateScoreRequest<'a> {
    pub ordinal: usize,
    pub prefix: &'a [u32],
    pub rows: ProjectionRows<'a>,
}

/// No Clone/Deserialize or successful partial-result operation. A malformed
/// projection permanently poisons this cursor, even if a retry would fit.
pub struct CandidateScoreCursor<'a> {
    scorer: &'a CandidateScorer,
    mode: ScoringMode,
    scores: Vec<f64>,
    terminals: Vec<f64>,
    prefix: Vec<u32>,
    selected: Vec<u32>,
    next: usize,
    expected: ScoringWork,
    completed: ScoringWork,
    poisoned: bool,
}
impl CandidateScorer {
    /// Compile the complete numerical work bound before requesting any logits.
    pub fn cursor(&self, mode: ScoringMode) -> Result<CandidateScoreCursor<'_>, ScoringError> {
        let prefix_evaluations = self.nodes.iter().filter(|n| !n.edges.is_empty()).count();
        let scored_edges = self.nodes.len() - 1;
        let projected_logits = if mode == ScoringMode::FullVocabulary {
            (prefix_evaluations as u64).checked_mul(self.vocabulary_size as u64)
                .ok_or(ScoringError::LimitExceeded("projected_logits"))?
        } else { scored_edges as u64 };
        if projected_logits > self.max_projected_logits {
            return Err(ScoringError::LimitExceeded("projected_logits"));
        }
        let mut scores = reserved(self.nodes.len())?; scores.resize(self.nodes.len(), 0.0);
        let mut terminals = reserved(self.ids.len())?; terminals.resize(self.ids.len(), 0.0);
        let mut cursor = CandidateScoreCursor { scorer: self, mode, scores, terminals,
            prefix: reserved(self.max_depth)?, selected: reserved(self.ids.len())?, next: 0,
            expected: ScoringWork { prefix_evaluations, scored_edges, projected_logits },
            completed: ScoringWork { prefix_evaluations: 0, scored_edges: 0, projected_logits: 0 }, poisoned: false };
        cursor.seek();
        Ok(cursor)
    }
}
impl CandidateScoreCursor<'_> {
    pub fn planned_work(&self) -> ScoringWork { self.expected }

    /// Repeated inspection is harmless; only accept advances the exact trie.
    pub fn request(&self) -> Result<Option<CandidateScoreRequest<'_>>, ScoringError> {
        if self.poisoned { return Err(ScoringError::CursorState); }
        if self.next == self.scorer.nodes.len() { return Ok(None); }
        Ok(Some(CandidateScoreRequest { ordinal: self.completed.prefix_evaluations,
            prefix: &self.prefix, rows: if self.mode == ScoringMode::FullVocabulary {
                ProjectionRows::FullVocabulary { vocabulary_size: self.scorer.vocabulary_size }
            } else { ProjectionRows::Selected(&self.selected) } }))
    }

    /// Consume one complete raw projection without taking or copying its buffer.
    /// Full-vocabulary denominators include ALL rows, including non-candidates.
    /// No normalization or score-space substitution is delegated to a driver.
    pub fn accept(&mut self, ordinal: usize, logits: &[f32]) -> Result<(), ScoringError> {
        if self.poisoned || self.next == self.scorer.nodes.len() || ordinal != self.completed.prefix_evaluations {
            self.poisoned = true;
            return Err(ScoringError::CursorState);
        }
        self.poisoned = true;
        let expected = if self.mode == ScoringMode::FullVocabulary { self.scorer.vocabulary_size } else { self.selected.len() };
        if logits.len() != expected {
            return Err(ScoringError::ProjectionLength { expected, actual: logits.len() });
        }
        for (row, value) in logits.iter().enumerate() {
            if !value.is_finite() { return Err(ScoringError::NonFiniteLogit { row }); }
        }
        let max = logits.iter().map(|&v| f64::from(v)).fold(f64::NEG_INFINITY, f64::max);
        let log_sum = logits.iter().map(|&v| (f64::from(v) - max).exp()).sum::<f64>().ln();
        let node = &self.scorer.nodes[self.next];
        for (row, &(token, child)) in node.edges.iter().enumerate() {
            let value = f64::from(logits[if self.mode == ScoringMode::FullVocabulary { token as usize } else { row }]);
            self.scores[child] = match self.mode {
                ScoringMode::FullVocabulary | ScoringMode::TrieConditional => {
                    // Keep the frozen operation order even for f32::MAX logits.
                    self.scores[self.next] + ((value - max) - log_sum)
                }
                ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::TerminalLogit } => value,
                ScoringMode::SequenceSoftmax { .. } => self.scores[self.next] + value,
            };
        }
        self.completed.prefix_evaluations += 1;
        self.completed.scored_edges += node.edges.len();
        self.completed.projected_logits += logits.len() as u64;
        self.next += 1;
        self.seek();
        self.poisoned = false;
        Ok(())
    }

    // Nodes are the compiler's original prefix-first order. Terminal leaves
    // consume a scored EOS edge but NEVER request another model forward.
    fn seek(&mut self) {
        self.prefix.clear(); self.selected.clear();
        while let Some(node) = self.scorer.nodes.get(self.next) {
            if let Some(rank) = node.terminal {
                self.terminals[rank] = match self.mode {
                    ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::MeanLogits } =>
                        self.scores[self.next] / self.scorer.lengths[rank] as f64,
                    _ => self.scores[self.next],
                };
            }
            if !node.edges.is_empty() {
                let mut ancestor = self.next;
                while let Some((parent, token)) = self.scorer.nodes[ancestor].parent {
                    self.prefix.push(token); ancestor = parent;
                }
                self.prefix.reverse();
                self.selected.extend(node.edges.iter().map(|&(token, _)| token));
                return;
            }
            self.next += 1;
        }
    }

    /// Only the complete candidate language can be normalized and published.
    pub fn finish(self) -> Result<CandidateScores, ScoringError> {
        if self.poisoned || self.next != self.scorer.nodes.len() || self.completed != self.expected {
            return Err(ScoringError::CursorState);
        }
        let max = self.terminals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator = self.terminals.iter().map(|s| (s - max).exp()).sum::<f64>();
        let mut candidates = reserved(self.scorer.ids.len())?;
        for (rank, &sequence_score) in self.terminals.iter().enumerate() {
            candidates.push(CandidateScore { id: self.scorer.ids[rank].clone(), scored_tokens: self.scorer.lengths[rank],
                sequence_score, candidate_weight: (sequence_score - max).exp() / denominator });
        }
        let (score_space, normalization_scope, length_rule) = match self.mode {
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
        Ok(CandidateScores { score_space, normalization_scope: normalization_scope.to_owned(),
            length_rule: length_rule.to_owned(), eos_rule: "append_and_score_exactly_one_eos".to_owned(),
            eos_token_id: self.scorer.eos, full_vocab_denominators_computed: self.mode == ScoringMode::FullVocabulary,
            candidates, work: self.completed })
    }
}

#[cfg(test)] mod tests;
