//! Private shared finite-head compiler, never a user-defined task interpreter.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{
    canonjson,
    execution_identity::Sha256Digest,
    native_engine::lmhead::{NANBEIGE_VOCAB_SIZE, scoring::{
        CandidateLogits, CandidateScorer, CandidateScores, ProjectionRows,
        ScoringError, ScoringLimits, ScoringMode, ScoringWork,
    }},
    tasks::ir::{DecodeStrategy, DependencyScope, FinitePostcondition, PromptSegment,
        PromptSegmentKind, ScoreSpace, TaskIR, TaskPlan},
    tokenizer::specials::TemplateControlIds,
};

/// All judge heads use full-vocabulary sequence log probabilities. No cheaper
/// estimator is silently promoted to the baseline or labeled a log probability.
pub const JUDGE_SCORER_VERSION: &str = "judge-full-vocab-scored-eos-v1";

#[derive(Clone, Copy, Debug)]
pub struct JudgeLimits {
    pub per_head: ScoringLimits,
    pub max_total_prompt_tokens: usize,
    pub max_total_candidate_tokens: usize,
    pub max_total_projected_logits: u64,
    pub max_output_bytes: u64,
}
impl Default for JudgeLimits {
    fn default() -> Self {
        Self { per_head: ScoringLimits::default(), max_total_prompt_tokens: 131_072,
            max_total_candidate_tokens: 65_536, max_total_projected_logits: 100_000_000,
            max_output_bytes: 4 * 1024 * 1024 }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JudgeError {
    Contract(&'static str),
    Limit(&'static str),
    Scoring(ScoringError),
    InvalidScores,
    AllocationRefused,
    Serialization,
}
impl fmt::Display for JudgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(reason) => write!(f, "judge contract refused: {reason}"),
            Self::Limit(axis) => write!(f, "judge budget exceeded: {axis}"),
            Self::Scoring(error) => write!(f, "judge scoring failed: {error}"),
            Self::InvalidScores => f.write_str("judge result failed complete-score validation"),
            Self::AllocationRefused => f.write_str("judge allocation refused"),
            Self::Serialization => f.write_str("judge serialization failed"),
        }
    }
}
impl Error for JudgeError {}
impl From<ScoringError> for JudgeError {
    fn from(error: ScoringError) -> Self { Self::Scoring(error) }
}

/// A projection provider receives the exact prompt on every call. `head` is
/// the canonical zero-based execution order, not permission to reuse another
/// prompt's KV. Return unnormalized logits for ALL requested vocabulary rows.
pub trait JudgeLogits {
    type Error: fmt::Display;
    fn project(&mut self, head: usize, prompt: &[PromptSegment], prefix: &[u32],
        rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error>;
}

pub(super) struct Head {
    pub ir: TaskIR,
    pub scorer: CandidateScorer,
    pub prompt_len: usize,
    pub max_prefix: usize,
    pub work: ScoringWork,
}

/// Private prompts and their private binding must not appear in Debug or JSON.
pub(super) struct Bundle {
    pub heads: Vec<Head>,
    pub eos: u32,
    pub work: ScoringWork,
    pub max_output_bytes: u64,
    pub binding: Sha256Digest,
}

impl Bundle {
    pub fn compile<P: Serialize>(tasks: &[&TaskPlan], eos: u32, controls: &TemplateControlIds,
        policy: &P, limits: JudgeLimits) -> Result<Self, JudgeError> {
        if tasks.is_empty() || tasks.len() > 32 {
            return Err(JudgeError::Limit("heads"));
        }
        if controls.ids().iter().any(|&id| id as usize >= NANBEIGE_VOCAB_SIZE)
            || !controls.entry(eos).is_some_and(|entry| entry.special) {
            return Err(JudgeError::Contract("EOS or control registry"));
        }
        // Check complete aggregate input bounds before cloning any task IR.
        let mut prompt_total = 0_usize;
        let mut candidate_total = 0_usize;
        let mut max_output_bytes = limits.max_output_bytes;
        for task in tasks {
            if task.task_spec_identity() != "judge-v1" {
                return Err(JudgeError::Contract("requires judge-v1 TaskPlan"));
            }
            let ir = task.ir();
            ir.validate().map_err(|_| JudgeError::Contract("invalid TaskIR"))?;
            if ir.dependency_scope() != DependencyScope::ItemLocal {
                return Err(JudgeError::Contract("judge must remain item-local"));
            }
            let wire = serde_json::to_value(ir).map_err(|_| JudgeError::Serialization)?;
            #[derive(Deserialize)]
            struct Conditions { postconditions: Vec<FinitePostcondition> }
            let conditions: Conditions = serde_json::from_value(wire.clone()).map_err(|_| JudgeError::Serialization)?;
            if conditions.postconditions.iter().any(|p| !matches!(p,
                FinitePostcondition::CandidateSetComplete | FinitePostcondition::OutputWithinBudget))
                || wire.get("continuation_trie").is_some_and(|v| !v.is_null()) {
                return Err(JudgeError::Contract("unsupported postcondition or external trie"));
            }
            for segment in ir.prompt_segments() {
                prompt_total = add(prompt_total, segment.token_ids().len(), "prompt_tokens")?;
                if segment.token_ids().iter().any(|&id| id as usize >= NANBEIGE_VOCAB_SIZE
                    || (segment.kind() == PromptSegmentKind::Document && controls.contains(id))) {
                    return Err(JudgeError::Contract("invalid token or document control"));
                }
            }
            let DecodeStrategy::PrefillOnly { candidates } = ir.decode_strategy() else {
                return Err(JudgeError::Contract("requires finite candidates"));
            };
            if candidates.len() < 2 || candidates.len() > 33 {
                return Err(JudgeError::Limit("candidates_per_head"));
            }
            for candidate in candidates {
                candidate_total = add(candidate_total,
                    add(candidate.continuation().token_ids().len(), 1, "candidate_tokens")?, "candidate_tokens")?;
                if candidate.continuation().token_ids().iter().any(|&id| controls.contains(id)) {
                    return Err(JudgeError::Contract("candidate contains control"));
                }
            }
            max_output_bytes = max_output_bytes.min(ir.budget().max_output_bytes);
        }
        if prompt_total > limits.max_total_prompt_tokens || candidate_total > limits.max_total_candidate_tokens
            || max_output_bytes == 0 {
            return Err(JudgeError::Limit("aggregate_input_or_output"));
        }
        let mut heads = reserved(tasks.len())?;
        let mut work = ScoringWork { prefix_evaluations: 0, scored_edges: 0, projected_logits: 0 };
        for task in tasks {
            let ir = task.ir();
            let DecodeStrategy::PrefillOnly { candidates } = ir.decode_strategy() else { unreachable!() };
            let mut bound = limits.per_head;
            bound.max_depth = bound.max_depth.min(ir.budget().max_output_tokens as usize);
            bound.max_nodes = bound.max_nodes.min(ir.budget().max_grammar_states as usize);
            let scorer = CandidateScorer::compile(candidates, NANBEIGE_VOCAB_SIZE, eos, bound)?;
            let mut sequences = reserved(candidates.len())?;
            sequences.extend(candidates.iter().map(|c| c.continuation().token_ids()));
            sequences.sort_unstable();
            let mut previous: &[u32] = &[];
            let mut prefixes = 1_usize;
            let mut max_prefix = 0;
            for sequence in sequences {
                let common = sequence.iter().zip(previous).take_while(|(a, b)| a == b).count();
                prefixes = add(prefixes, sequence.len() - common, "prefixes")?;
                max_prefix = max_prefix.max(sequence.len());
                previous = sequence;
            }
            let head_work = ScoringWork {
                prefix_evaluations: prefixes,
                scored_edges: add(prefixes - 1, candidates.len(), "edges")?,
                projected_logits: (prefixes as u64).checked_mul(NANBEIGE_VOCAB_SIZE as u64)
                    .ok_or(JudgeError::Limit("projected_logits"))?,
            };
            if head_work.projected_logits > bound.max_projected_logits {
                return Err(JudgeError::Limit("head_projected_logits"));
            }
            work = sum_work(work, head_work)?;
            if work.projected_logits > limits.max_total_projected_logits {
                return Err(JudgeError::Limit("projected_logits"));
            }
            let prompt_len = ir.prompt_segments().iter().try_fold(0_usize,
                |n, s| add(n, s.token_ids().len(), "prompt_tokens"))?;
            heads.push(Head { ir: ir.clone(), scorer, prompt_len, max_prefix, work: head_work });
        }
        #[derive(Serialize)]
        struct Binding<'a, P> { version: &'static str, plans: Vec<&'a TaskIR>, policy: &'a P,
            eos: u32, controls: Vec<(u32, bool, &'a str)>, max_output_bytes: u64 }
        let witness = Binding { version: JUDGE_SCORER_VERSION, plans: heads.iter().map(|h| &h.ir).collect(),
            policy, eos, controls: controls.entries().iter().map(|e| (e.id, e.special, e.surface.as_str())).collect(),
            max_output_bytes };
        let binding = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&witness).map_err(|_| JudgeError::Serialization)?);
        Ok(Self { heads, eos, work, max_output_bytes, binding })
    }

    pub fn score<M: JudgeLogits>(&self, provider: &mut M) -> Result<Vec<CandidateScores>, JudgeError> {
        self.score_with(|index, head| {
            struct Bound<'a, M> { provider: &'a mut M, index: usize, prompt: &'a [PromptSegment] }
            impl<M: JudgeLogits> CandidateLogits for Bound<'_, M> {
                type Error = M::Error;
                fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
                    self.provider.project(self.index, self.prompt, prefix, rows)
                }
            }
            head.scorer.score(&mut Bound { provider, index, prompt: head.ir.prompt_segments() },
                ScoringMode::FullVocabulary).map_err(JudgeError::from)
        })
    }

    pub fn score_with<E: From<JudgeError>, F>(&self, mut score: F) -> Result<Vec<CandidateScores>, E>
    where F: FnMut(usize, &Head) -> Result<CandidateScores, E> {
        let mut all = reserved(self.heads.len()).map_err(E::from)?;
        for (index, head) in self.heads.iter().enumerate() {
            let scores = score(index, head)?;
            head.validate_scores(&scores, self.eos).map_err(E::from)?;
            all.push(scores);
        }
        Ok(all)
    }

    pub fn check_output<T: Serialize>(&self, value: &T) -> Result<(), JudgeError> {
        let bytes = canonjson::canonical_bytes(value).map_err(|_| JudgeError::Serialization)?;
        if bytes.len() as u64 > self.max_output_bytes { return Err(JudgeError::Limit("complete_output_bytes")); }
        Ok(())
    }
}

impl Head {
    fn validate_scores(&self, scores: &CandidateScores, eos: u32) -> Result<(), JudgeError> {
        let DecodeStrategy::PrefillOnly { candidates } = self.ir.decode_strategy() else { unreachable!() };
        if scores.score_space != ScoreSpace::FullVocabSequenceLogprob || !scores.full_vocab_denominators_computed
            || scores.eos_token_id != eos || scores.work != self.work || scores.candidates.len() != candidates.len()
            || scores.candidates.windows(2).any(|w| w[0].id >= w[1].id) {
            return Err(JudgeError::InvalidScores);
        }
        let mut sum = 0.0;
        for expected in candidates {
            let actual = scores.candidates.iter().find(|c| c.id == expected.id()).ok_or(JudgeError::InvalidScores)?;
            if actual.scored_tokens != expected.continuation().token_ids().len() + 1
                || !actual.sequence_score.is_finite() || actual.sequence_score > 1e-10
                || !actual.candidate_weight.is_finite() || !(0.0..=1.0).contains(&actual.candidate_weight) {
                return Err(JudgeError::InvalidScores);
            }
            sum += actual.candidate_weight;
        }
        if (sum - 1.0_f64).abs() > 1e-10 { return Err(JudgeError::InvalidScores); }
        Ok(())
    }
}

pub(super) fn add(a: usize, b: usize, axis: &'static str) -> Result<usize, JudgeError> {
    a.checked_add(b).ok_or(JudgeError::Limit(axis))
}
pub(super) fn reserved<T>(count: usize) -> Result<Vec<T>, JudgeError> {
    let mut items = Vec::new();
    items.try_reserve_exact(count).map_err(|_| JudgeError::AllocationRefused)?;
    Ok(items)
}
fn sum_work(a: ScoringWork, b: ScoringWork) -> Result<ScoringWork, JudgeError> {
    Ok(ScoringWork { prefix_evaluations: add(a.prefix_evaluations, b.prefix_evaluations, "prefixes")?,
        scored_edges: add(a.scored_edges, b.scored_edges, "edges")?,
        projected_logits: a.projected_logits.checked_add(b.projected_logits).ok_or(JudgeError::Limit("projected_logits"))? })
}
