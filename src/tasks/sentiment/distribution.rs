//! Task-bound dimensional sentiment using the shared exact continuation scorer.

use std::{error::Error, fmt};
use serde::{Deserialize, Serialize};
use crate::{
    canonjson,
    execution_identity::Sha256Digest,
    native_engine::lmhead::{
        NANBEIGE_VOCAB_SIZE,
        scoring::{CandidateLogits, CandidateScorer, CandidateScores, ProjectionRows,
            ScoringError, ScoringLimits, ScoringMode},
    },
    tasks::ir::{DecodeStrategy, DependencyScope, PromptSegment, PromptSegmentKind,
        TaskIR, TaskPlan},
};

const SCALE: i32 = 1000;
const PPM: u32 = 1_000_000;
const MAX_BINS: usize = 33;

/// Closed interpreted axes. Negative coordinates mean the named low anchor,
/// not universally negative sentiment. The axis labels do not establish
/// measurement validity, calibration or access to an author's internal state.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SentimentAxis { Valence, Arousal, Dominance, Approach }

impl SentimentAxis {
    pub const ALL: [Self; 4] = [Self::Valence, Self::Arousal, Self::Dominance, Self::Approach];

    /// Version-one textual interpretation of the normalized coordinate.
    #[must_use]
    pub const fn anchors(self) -> (&'static str, &'static str) {
        match self {
            Self::Valence => ("negative", "positive"),
            Self::Arousal => ("calm", "activated"),
            Self::Dominance => ("powerless", "in_control"),
            Self::Approach => ("withdrawal", "approach"),
        }
    }
}

/// Exact candidate-to-coordinate mapping. A dimension covers a complete,
/// uniformly spaced, zero-containing grid from -1000 through +1000. Values
/// are integers in the plan; only result projections use floating point.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SentimentAnchor {
    pub candidate_id: String,
    pub value_milli: i32,
}

/// One independent dimension, already bound to the sentiment-v1 task registry.
/// All dimensions must refer to the same document and global-policy tokens,
/// but must have distinct exact prompts. Token meanings remain the trusted
/// task/template compiler's responsibility, not something inferred from IDs.
#[derive(Clone, Debug)]
pub struct SentimentAxisInput {
    pub axis: SentimentAxis,
    pub task: TaskPlan,
    pub anchors: Vec<SentimentAnchor>,
}

/// Explicit, uncalibrated abstention thresholds in millionths. A diffuse
/// distribution can abstain rather than masquerading as confidently neutral.
/// No task accuracy or psychological confidence is implied by these cutoffs.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SentimentPolicy {
    pub minimum_peak_weight_ppm: u32,
    pub maximum_normalized_entropy_ppm: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SentimentOptions {
    pub mode: ScoringMode,
    pub eos_token_id: u32,
    pub policy: SentimentPolicy,
}

/// Aggregate limits apply across ALL requested dimensions, not independently
/// renewable per-axis budgets. Per-axis scorer bounds may only be tightened by
/// the corresponding TaskIR. This structure does not grant model admission.
#[derive(Clone, Copy, Debug)]
pub struct SentimentLimits {
    pub per_axis: ScoringLimits,
    pub max_total_prompt_tokens: usize,
    pub max_total_candidate_tokens: usize,
    pub max_total_nodes: usize,
    pub max_total_projected_logits: u64,
    pub max_output_bytes: u64,
}

impl Default for SentimentLimits {
    fn default() -> Self {
        Self {
            per_axis: ScoringLimits::default(),
            max_total_prompt_tokens: 131_072,
            max_total_candidate_tokens: 65_536,
            max_total_nodes: 65_540,
            max_total_projected_logits: 100_000_000,
            max_output_bytes: 4 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SentimentDecision { Estimated, Abstained }

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SentimentWork {
    pub dimensions: usize,
    pub candidates: usize,
    pub prefix_evaluations: usize,
    pub scored_edges: usize,
    pub projected_logits: u64,
}

/// All candidates and score-space disclosures are retained. `estimate` is
/// absent on abstention. Distribution moments are descriptive projections of
/// conditional candidate weights, never calibrated posterior uncertainty.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SentimentDimension {
    pub axis: SentimentAxis,
    pub low_anchor: String,
    pub high_anchor: String,
    pub decision: SentimentDecision,
    pub estimate: Option<f64>,
    pub distribution_mean: f64,
    pub distribution_stddev: f64,
    pub normalized_entropy: f64,
    pub modal_value_milli: i32,
    pub anchors: Vec<SentimentAnchor>,
    pub scores: CandidateScores,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SentimentResult {
    pub schema_version: u32,
    pub task_spec: String,
    pub interpretation: String,
    pub calibration: String,
    pub policy: SentimentPolicy,
    pub dimensions: Vec<SentimentDimension>,
    pub work: SentimentWork,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SentimentError {
    WrongTask,
    InvalidPlan,
    InvalidPolicy,
    DuplicateAxis,
    InconsistentDocument,
    SharedAxisPrompt,
    InvalidAnchors,
    LimitExceeded(&'static str),
    Scoring(ScoringError),
    InvalidScores,
    AllocationRefused,
    Serialization,
}

impl fmt::Display for SentimentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongTask => f.write_str("sentiment requires sentiment-v1 task plans"),
            Self::InvalidPlan => f.write_str("sentiment requires bounded exact-token finite candidates"),
            Self::InvalidPolicy => f.write_str("sentiment policy thresholds must be in 0..=1000000"),
            Self::DuplicateAxis => f.write_str("sentiment dimension was specified more than once"),
            Self::InconsistentDocument => f.write_str("sentiment dimensions must share document and global policy"),
            Self::SharedAxisPrompt => f.write_str("independent sentiment dimensions require distinct prompts"),
            Self::InvalidAnchors => f.write_str("sentiment requires a complete centered coordinate grid"),
            Self::LimitExceeded(axis) => write!(f, "sentiment aggregate budget exceeded: {axis}"),
            Self::Scoring(error) => write!(f, "sentiment continuation scoring failed: {error}"),
            Self::InvalidScores => f.write_str("sentiment dimension scores failed independent finalization"),
            Self::AllocationRefused => f.write_str("sentiment allocation refused"),
            Self::Serialization => f.write_str("sentiment serialization failed"),
        }
    }
}
impl Error for SentimentError {}
impl From<ScoringError> for SentimentError {
    fn from(error: ScoringError) -> Self { Self::Scoring(error) }
}

/// Projection provider for exact per-axis prompts. The complete prompt is
/// supplied on every call; `prefix` excludes it. Implementations must not carry
/// another dimension's KV as though it belonged to this prompt, renormalize
/// logits, or replace a full-vocabulary denominator with selected rows.
pub trait SentimentLogits {
    type Error: fmt::Display;
    fn project(
        &mut self, axis: SentimentAxis, prompt: &[PromptSegment], prefix: &[u32],
        rows: ProjectionRows<'_>,
    ) -> Result<Vec<f32>, Self::Error>;
}

pub(super) struct HeadPlan {
    pub(super) axis: SentimentAxis,
    pub(super) ir: TaskIR,
    pub(super) anchors: Vec<SentimentAnchor>,
    pub(super) scorer: CandidateScorer,
    pub(super) work: SentimentWork,
    pub(super) prompt_len: usize,
    pub(super) prefix_count: usize,
    pub(super) max_prefix: usize,
}

/// A bounded collection of independent task heads, in canonical axis order.
/// Deliberately not Serialize or Debug: prompt tokens and their private binding
/// digest are not public result/telemetry fields.
pub struct SentimentPlan {
    pub(super) heads: Vec<HeadPlan>,
    pub(super) options: SentimentOptions,
    pub(super) max_output_bytes: u64,
    work: SentimentWork,
    binding_digest: Sha256Digest,
}

#[derive(Clone, Copy)]
struct InputIr<'a> {
    axis: SentimentAxis,
    ir: &'a TaskIR,
    anchors: &'a [SentimentAnchor],
}

impl SentimentPlan {
    /// Compile the complete bundle before model work. Registry identity and
    /// TaskIR bounds cannot be widened by a sentiment dimension.
    pub fn from_task_plans(
        inputs: &[SentimentAxisInput], options: SentimentOptions, limits: SentimentLimits,
    ) -> Result<Self, SentimentError> {
        if inputs.is_empty() || inputs.len() > SentimentAxis::ALL.len() {
            return Err(SentimentError::LimitExceeded("dimensions"));
        }
        let mut irs = reserved(inputs.len())?;
        for input in inputs {
            if input.task.task_spec_identity() != "sentiment-v1" {
                return Err(SentimentError::WrongTask);
            }
            irs.push(InputIr { axis: input.axis, ir: input.task.ir(), anchors: &input.anchors });
        }
        Self::compile_irs(&irs, options, limits)
    }

    fn compile_irs(
        inputs: &[InputIr<'_>], options: SentimentOptions, limits: SentimentLimits,
    ) -> Result<Self, SentimentError> {
        if inputs.is_empty() || inputs.len() > SentimentAxis::ALL.len() {
            return Err(SentimentError::LimitExceeded("dimensions"));
        }
        if options.policy.minimum_peak_weight_ppm > PPM
            || options.policy.maximum_normalized_entropy_ppm > PPM {
            return Err(SentimentError::InvalidPolicy);
        }
        let mut ordered = reserved(inputs.len())?;
        ordered.extend_from_slice(inputs);
        ordered.sort_unstable_by_key(|input| input.axis);
        if ordered.windows(2).any(|pair| pair[0].axis == pair[1].axis) {
            return Err(SentimentError::DuplicateAxis);
        }
        let mut heads = reserved(inputs.len())?;
        let mut prompt_total = 0_usize;
        let mut candidate_total = 0_usize;
        let mut node_total = 0_usize;
        let mut max_output_bytes = limits.max_output_bytes;
        let mut work = SentimentWork::default();
        for input in &ordered {
            let ir = input.ir;
            ir.validate().map_err(|_| SentimentError::InvalidPlan)?;
            if ir.dependency_scope() != DependencyScope::ItemLocal {
                return Err(SentimentError::InvalidPlan);
            }
            let DecodeStrategy::PrefillOnly { candidates } = ir.decode_strategy() else {
                return Err(SentimentError::InvalidPlan);
            };
            if candidates.len() < 3 || candidates.len() > MAX_BINS {
                return Err(SentimentError::InvalidAnchors);
            }
            if !segment_tokens(ir, PromptSegmentKind::Document).any(|_| true)
                || !segment_tokens(ir, PromptSegmentKind::TaskInstruction).any(|_| true)
                || ir.prompt_segments().iter().flat_map(|s| s.token_ids())
                    .any(|&id| id as usize >= NANBEIGE_VOCAB_SIZE) {
                return Err(SentimentError::InvalidPlan);
            }
            for previous in ordered.iter().take_while(|other| other.axis != input.axis) {
                for kind in [PromptSegmentKind::Document, PromptSegmentKind::GlobalPolicy] {
                    if !segment_tokens(previous.ir, kind).eq(segment_tokens(ir, kind)) {
                        return Err(SentimentError::InconsistentDocument);
                    }
                }
                if previous.ir.prompt_segments().iter().flat_map(|s| s.token_ids())
                    .eq(ir.prompt_segments().iter().flat_map(|s| s.token_ids())) {
                    return Err(SentimentError::SharedAxisPrompt);
                }
            }
            let prompt_len = ir.prompt_segments().iter().try_fold(0_usize, |n, s|
                checked_add(n, s.token_ids().len(), "prompt_tokens"))?;
            prompt_total = checked_add(prompt_total, prompt_len, "prompt_tokens")?;
            if prompt_total > limits.max_total_prompt_tokens {
                return Err(SentimentError::LimitExceeded("prompt_tokens"));
            }
            let tokens = candidates.iter().try_fold(0_usize, |n, c| {
                let length = checked_add(c.continuation().token_ids().len(), 1, "candidate_tokens")?;
                checked_add(n, length, "candidate_tokens")
            })?;
            candidate_total = checked_add(candidate_total, tokens, "candidate_tokens")?;
            node_total = checked_add(node_total, checked_add(tokens, 1, "nodes")?, "nodes")?;
            if candidate_total > limits.max_total_candidate_tokens || node_total > limits.max_total_nodes {
                return Err(SentimentError::LimitExceeded("candidate_tokens_or_nodes"));
            }
            let anchors = checked_anchors(input.anchors, candidates)?;
            let mut scorer_limits = limits.per_axis;
            scorer_limits.max_depth = scorer_limits.max_depth.min(ir.budget().max_output_tokens as usize);
            scorer_limits.max_nodes = scorer_limits.max_nodes.min(ir.budget().max_grammar_states as usize);
            let scorer = CandidateScorer::compile(candidates, NANBEIGE_VOCAB_SIZE, options.eos_token_id, scorer_limits)?;
            let (prefix_count, max_prefix) = prefix_counts(candidates)?;
            let edges = checked_add(prefix_count - 1, candidates.len(), "scored_edges")?;
            let projected = if options.mode == ScoringMode::FullVocabulary {
                (prefix_count as u64).checked_mul(NANBEIGE_VOCAB_SIZE as u64)
                    .ok_or(SentimentError::LimitExceeded("projected_logits"))?
            } else { edges as u64 };
            if projected > limits.per_axis.max_projected_logits {
                return Err(SentimentError::LimitExceeded("axis_projected_logits"));
            }
            let head_work = SentimentWork { dimensions: 1, candidates: candidates.len(),
                prefix_evaluations: prefix_count, scored_edges: edges, projected_logits: projected };
            add_work(&mut work, head_work)?;
            if work.projected_logits > limits.max_total_projected_logits {
                return Err(SentimentError::LimitExceeded("projected_logits"));
            }
            max_output_bytes = max_output_bytes.min(ir.budget().max_output_bytes);
            heads.push(HeadPlan { axis: input.axis, ir: ir.clone(), anchors, scorer,
                work: head_work, prompt_len, prefix_count, max_prefix });
        }
        if max_output_bytes == 0 { return Err(SentimentError::LimitExceeded("output_bytes")); }
        #[derive(Serialize)]
        struct AxisBinding<'a> { axis: SentimentAxis, ir: &'a TaskIR, anchors: &'a [SentimentAnchor] }
        #[derive(Serialize)]
        struct Binding<'a> { version: &'static str, axes: Vec<AxisBinding<'a>>,
            options: SentimentOptions, max_output_bytes: u64 }
        let binding = Binding { version: "dimensional-sentiment-execution-v1", options,
            max_output_bytes, axes: heads.iter().map(|h| AxisBinding {
                axis: h.axis, ir: &h.ir, anchors: &h.anchors,
            }).collect() };
        let bytes = canonjson::canonical_bytes(&binding).map_err(|_| SentimentError::Serialization)?;
        Ok(Self { heads, options, max_output_bytes, work,
            binding_digest: Sha256Digest::of_bytes(&bytes) })
    }

    /// Private semantic identity input: do not log this prompt-derived digest.
    #[must_use]
    pub const fn binding_digest(&self) -> &Sha256Digest { &self.binding_digest }

    #[must_use]
    pub const fn planned_work(&self) -> SentimentWork { self.work }

    /// Execute all independent dimensions, or return no result. Every axis gets
    /// a fresh scorer normalization over its OWN complete finite candidate set.
    pub fn execute<M: SentimentLogits>(&self, provider: &mut M) -> Result<SentimentResult, SentimentError> {
        self.execute_heads(|head, mode| {
            let mut bound = AxisProjection { provider, axis: head.axis, prompt: head.ir.prompt_segments() };
            head.scorer.score(&mut bound, mode).map_err(SentimentError::from)
        })
    }

    pub(super) fn execute_heads<E, F>(&self, mut score: F) -> Result<SentimentResult, E>
    where E: From<SentimentError>, F: FnMut(&HeadPlan, ScoringMode) -> Result<CandidateScores, E> {
        let mut dimensions = reserved(self.heads.len()).map_err(E::from)?;
        for head in &self.heads {
            let scores = score(head, self.options.mode)?;
            dimensions.push(finalize_dimension(head, scores, self.options).map_err(E::from)?);
        }
        let result = SentimentResult {
            schema_version: 1, task_spec: "sentiment-v1".to_owned(),
            interpretation: "independent_conditional_candidate_weight_projections".to_owned(),
            calibration: "uncalibrated_not_a_psychological_measurement".to_owned(),
            policy: self.options.policy, dimensions, work: self.work,
        };
        let bytes = canonjson::canonical_bytes(&result).map_err(|_| E::from(SentimentError::Serialization))?;
        if bytes.len() as u64 > self.max_output_bytes {
            return Err(SentimentError::LimitExceeded("output_bytes").into());
        }
        Ok(result)
    }
}

struct AxisProjection<'a, M> { provider: &'a mut M, axis: SentimentAxis, prompt: &'a [PromptSegment] }
impl<M: SentimentLogits> CandidateLogits for AxisProjection<'_, M> {
    type Error = M::Error;
    fn project(&mut self, prefix: &[u32], rows: ProjectionRows<'_>) -> Result<Vec<f32>, Self::Error> {
        self.provider.project(self.axis, self.prompt, prefix, rows)
    }
}

fn segment_tokens(ir: &TaskIR, kind: PromptSegmentKind) -> impl Iterator<Item = u32> + '_ {
    ir.prompt_segments().iter().filter(move |s| s.kind() == kind)
        .flat_map(|s| s.token_ids().iter().copied())
}

fn checked_anchors(
    anchors: &[SentimentAnchor], candidates: &[crate::tasks::ir::Candidate],
) -> Result<Vec<SentimentAnchor>, SentimentError> {
    let count = candidates.len();
    if anchors.len() != count || count < 3 || count > MAX_BINS || count % 2 == 0
        || (2 * SCALE) as usize % (count - 1) != 0 {
        return Err(SentimentError::InvalidAnchors);
    }
    let step = 2 * SCALE / (count - 1) as i32;
    let mut ordered = reserved(count)?;
    ordered.extend_from_slice(anchors);
    ordered.sort_unstable_by_key(|a| a.value_milli);
    for (index, anchor) in ordered.iter().enumerate() {
        if anchor.value_milli != -SCALE + index as i32 * step
            || anchor.candidate_id.len() > 64
            || candidates.iter().filter(|c| c.id() == anchor.candidate_id).count() != 1
            || ordered[..index].iter().any(|a| a.candidate_id == anchor.candidate_id) {
            return Err(SentimentError::InvalidAnchors);
        }
    }
    Ok(ordered)
}

fn prefix_counts(candidates: &[crate::tasks::ir::Candidate]) -> Result<(usize, usize), SentimentError> {
    let mut ordered = reserved(candidates.len())?;
    ordered.extend(candidates.iter().map(|c| c.continuation().token_ids()));
    ordered.sort_unstable();
    let mut previous: &[u32] = &[];
    let mut prefixes = 1_usize;
    let mut max_prefix = 0;
    for tokens in ordered {
        let common = previous.iter().zip(tokens).take_while(|(a, b)| a == b).count();
        prefixes = checked_add(prefixes, tokens.len() - common, "prefixes")?;
        max_prefix = max_prefix.max(tokens.len());
        previous = tokens;
    }
    Ok((prefixes, max_prefix))
}

fn finalize_dimension(head: &HeadPlan, scores: CandidateScores, options: SentimentOptions)
    -> Result<SentimentDimension, SentimentError> {
    use crate::tasks::ir::ScoreSpace;
    let expected_space = match options.mode {
        ScoringMode::FullVocabulary => ScoreSpace::FullVocabSequenceLogprob,
        ScoringMode::TrieConditional => ScoreSpace::TrieLocalConditionalProbability,
        ScoringMode::SequenceSoftmax { .. } => ScoreSpace::SequenceScoreSoftmax,
    };
    if scores.candidates.len() != head.anchors.len() || scores.score_space != expected_space
        || scores.eos_token_id != options.eos_token_id
        || scores.full_vocab_denominators_computed != (options.mode == ScoringMode::FullVocabulary)
        || scores.work.prefix_evaluations != head.work.prefix_evaluations
        || scores.work.scored_edges != head.work.scored_edges
        || scores.work.projected_logits != head.work.projected_logits
        || scores.candidates.windows(2).any(|w| w[0].id >= w[1].id) {
        return Err(SentimentError::InvalidScores);
    }
    let DecodeStrategy::PrefillOnly { candidates } = head.ir.decode_strategy() else {
        return Err(SentimentError::InvalidScores);
    };
    let mut mean = 0.0_f64;
    let mut entropy = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut peak = 0.0_f64;
    let mut modal = -SCALE;
    let mut modal_score = f64::NEG_INFINITY;
    for anchor in &head.anchors {
        let scored = scores.candidates.iter().find(|c| c.id == anchor.candidate_id)
            .ok_or(SentimentError::InvalidScores)?;
        let expected = candidates.iter().find(|c| c.id() == anchor.candidate_id)
            .ok_or(SentimentError::InvalidScores)?;
        if !scored.sequence_score.is_finite() || !scored.candidate_weight.is_finite()
            || !(0.0..=1.0).contains(&scored.candidate_weight)
            || scored.scored_tokens != expected.continuation().token_ids().len() + 1 {
            return Err(SentimentError::InvalidScores);
        }
        let weight = scored.candidate_weight;
        sum += weight;
        mean += weight * f64::from(anchor.value_milli) / f64::from(SCALE);
        if weight > 0.0 { entropy -= weight * weight.ln(); }
        peak = peak.max(weight);
        // Raw scores prevent exp-underflow from inventing ties. A genuine tie
        // uses the lowest coordinate, independent of arbitrary candidate IDs.
        if scored.sequence_score > modal_score {
            modal_score = scored.sequence_score;
            modal = anchor.value_milli;
        }
    }
    if (sum - 1.0).abs() > 1e-10 { return Err(SentimentError::InvalidScores); }
    let mean = mean.clamp(-1.0, 1.0);
    let entropy = (entropy / (head.anchors.len() as f64).ln()).clamp(0.0, 1.0);
    let variance = head.anchors.iter().map(|a| {
        let weight = scores.candidates.iter().find(|c| c.id == a.candidate_id)
            .expect("complete candidates checked above").candidate_weight;
        let residual = f64::from(a.value_milli) / f64::from(SCALE) - mean;
        weight * residual * residual
    }).sum::<f64>();
    let accepted = peak >= f64::from(options.policy.minimum_peak_weight_ppm) / f64::from(PPM)
        && entropy <= f64::from(options.policy.maximum_normalized_entropy_ppm) / f64::from(PPM);
    let (low, high) = head.axis.anchors();
    Ok(SentimentDimension {
        axis: head.axis, low_anchor: low.to_owned(), high_anchor: high.to_owned(),
        decision: if accepted { SentimentDecision::Estimated } else { SentimentDecision::Abstained },
        estimate: accepted.then_some(mean), distribution_mean: mean,
        distribution_stddev: variance.max(0.0).sqrt(), normalized_entropy: entropy,
        modal_value_milli: modal, anchors: head.anchors.clone(), scores,
    })
}

fn reserved<T>(count: usize) -> Result<Vec<T>, SentimentError> {
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| SentimentError::AllocationRefused)?;
    Ok(values)
}
fn checked_add(a: usize, b: usize, axis: &'static str) -> Result<usize, SentimentError> {
    a.checked_add(b).ok_or(SentimentError::LimitExceeded(axis))
}
fn add_work(total: &mut SentimentWork, next: SentimentWork) -> Result<(), SentimentError> {
    total.dimensions = checked_add(total.dimensions, next.dimensions, "dimensions")?;
    total.candidates = checked_add(total.candidates, next.candidates, "candidates")?;
    total.prefix_evaluations = checked_add(total.prefix_evaluations, next.prefix_evaluations, "prefixes")?;
    total.scored_edges = checked_add(total.scored_edges, next.scored_edges, "edges")?;
    total.projected_logits = total.projected_logits.checked_add(next.projected_logits)
        .ok_or(SentimentError::LimitExceeded("projected_logits"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::ir::{Candidate, FinitePostcondition, GrammarReference, TaskBudget, TokenSequence};
    use crate::native_engine::lmhead::scoring::SequenceScoreRule;

    fn ir(marker: u32, document: u32) -> TaskIR {
        TaskIR::new(vec![
            PromptSegment::new(PromptSegmentKind::GlobalPolicy, vec![7]),
            PromptSegment::new(PromptSegmentKind::TaskInstruction, vec![marker]),
            PromptSegment::new(PromptSegmentKind::Document, vec![document]),
            PromptSegment::new(PromptSegmentKind::AnswerScaffold, vec![9]),
        ], DecodeStrategy::PrefillOnly { candidates: [1, 2, 3].into_iter().map(|id|
            Candidate::new(format!("bin-{id}"), TokenSequence::new(vec![id]))).collect() },
            GrammarReference::none(), None,
            vec![FinitePostcondition::CandidateSetComplete, FinitePostcondition::OutputWithinBudget],
            TaskBudget { max_input_tokens: 64, max_output_tokens: 16, max_output_bytes: 100000,
                max_grammar_states: 64, max_kv_bytes: 10000000 }, DependencyScope::ItemLocal).unwrap()
    }
    fn anchors() -> Vec<SentimentAnchor> {
        [-1000, 0, 1000].into_iter().enumerate().map(|(index, value_milli)|
            SentimentAnchor { candidate_id: format!("bin-{}", index + 1), value_milli }).collect()
    }
    fn options() -> SentimentOptions {
        SentimentOptions { mode: ScoringMode::SequenceSoftmax { rule: SequenceScoreRule::TerminalLogit },
            eos_token_id: 0, policy: SentimentPolicy { minimum_peak_weight_ppm: 0,
                maximum_normalized_entropy_ppm: 900000 } }
    }
    fn plan() -> SentimentPlan {
        let low = ir(10, 8); let high = ir(11, 8); let anchors = anchors();
        SentimentPlan::compile_irs(&[
            InputIr { axis: SentimentAxis::Valence, ir: &low, anchors: &anchors },
            InputIr { axis: SentimentAxis::Arousal, ir: &high, anchors: &anchors },
        ], options(), SentimentLimits::default()).unwrap()
    }
    struct Model { flat: bool, calls: usize, fail_after: usize }
    impl SentimentLogits for Model {
        type Error = &'static str;
        fn project(&mut self, axis: SentimentAxis, prompt: &[PromptSegment], prefix: &[u32], rows: ProjectionRows<'_>)
            -> Result<Vec<f32>, Self::Error> {
            self.calls += 1;
            if self.calls > self.fail_after { return Err("fixture failure"); }
            assert_eq!(prompt[1].token_ids(), &[if axis == SentimentAxis::Valence { 10 } else { 11 }]);
            let preferred = if axis == SentimentAxis::Valence { 1 } else { 3 };
            let logit = if !self.flat && prefix == [preferred] { 8.0 } else { 0.0 };
            Ok(match rows {
                ProjectionRows::Selected(ids) => vec![logit; ids.len()],
                ProjectionRows::FullVocabulary { vocabulary_size } => vec![logit; vocabulary_size],
            })
        }
    }
    fn model(flat: bool) -> Model { Model { flat, calls: 0, fail_after: usize::MAX } }

    #[test]
    fn dimensions_are_independent_not_one_joint_softmax() {
        let plan = plan(); let result = plan.execute(&mut model(false)).unwrap();
        assert_eq!(result.work.dimensions, 2);
        assert_eq!(result.work.prefix_evaluations, 8);
        assert_eq!(result.work.projected_logits, 12);
        assert!(result.dimensions[0].estimate.unwrap() < -0.99);
        assert!(result.dimensions[1].estimate.unwrap() > 0.99);
        for dimension in result.dimensions {
            assert_eq!(dimension.scores.candidates.len(), 3);
            assert!((dimension.scores.candidates.iter().map(|c| c.candidate_weight).sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }
    #[test]
    fn diffuse_distribution_abstains_instead_of_claiming_neutrality() {
        let result = plan().execute(&mut model(true)).unwrap();
        for dimension in result.dimensions {
            assert_eq!(dimension.decision, SentimentDecision::Abstained);
            assert_eq!(dimension.estimate, None);
            assert!(dimension.distribution_mean.abs() < 1e-12);
            assert!((dimension.normalized_entropy - 1.0).abs() < 1e-12);
            assert!(dimension.distribution_stddev > 0.8);
        }
    }
    #[test]
    fn axis_and_anchor_input_order_does_not_change_binding_or_output() {
        let a = ir(10, 8); let b = ir(11, 8); let mut anchors = anchors(); anchors.reverse();
        let reverse = SentimentPlan::compile_irs(&[
            InputIr { axis: SentimentAxis::Arousal, ir: &b, anchors: &anchors },
            InputIr { axis: SentimentAxis::Valence, ir: &a, anchors: &anchors },
        ], options(), SentimentLimits::default()).unwrap();
        assert_eq!(plan().binding_digest(), reverse.binding_digest());
        assert_eq!(plan().execute(&mut model(false)).unwrap(), reverse.execute(&mut model(false)).unwrap());
    }
    #[test]
    fn mixed_documents_repeated_axes_and_reused_axis_prompts_refuse() {
        let a = ir(10, 8); let b = ir(11, 77); let points = anchors();
        let first = InputIr { axis: SentimentAxis::Valence, ir: &a, anchors: &points };
        for second in [
            InputIr { axis: SentimentAxis::Arousal, ir: &b, anchors: &points },
            InputIr { axis: SentimentAxis::Arousal, ir: &a, anchors: &points }, first,
        ] {
            assert!(SentimentPlan::compile_irs(&[first, second], options(), SentimentLimits::default()).is_err());
        }
    }
    #[test]
    fn missing_duplicate_or_nonuniform_anchor_grids_refuse() {
        let task = ir(10, 8); let DecodeStrategy::PrefillOnly { candidates } = task.decode_strategy() else { unreachable!() };
        let mut points = anchors(); points[1].value_milli = 1;
        assert!(checked_anchors(&points, candidates).is_err());
        points = anchors(); points[1].candidate_id = points[0].candidate_id.clone();
        assert!(checked_anchors(&points, candidates).is_err());
        assert!(checked_anchors(&anchors()[..2], candidates).is_err());
    }
    #[test]
    fn aggregate_work_and_complete_output_caps_are_not_reset_per_axis() {
        let a = ir(10, 8); let b = ir(11, 8); let points = anchors();
        let input = [InputIr { axis: SentimentAxis::Valence, ir: &a, anchors: &points },
            InputIr { axis: SentimentAxis::Arousal, ir: &b, anchors: &points }];
        let limits = SentimentLimits { max_total_projected_logits: 11, ..SentimentLimits::default() };
        assert!(matches!(SentimentPlan::compile_irs(&input, options(), limits), Err(SentimentError::LimitExceeded("projected_logits"))));
        let limits = SentimentLimits { max_output_bytes: 100, ..SentimentLimits::default() };
        let bounded = SentimentPlan::compile_irs(&input, options(), limits).unwrap();
        assert!(matches!(bounded.execute(&mut model(false)), Err(SentimentError::LimitExceeded("output_bytes"))));
    }
    #[test]
    fn failed_later_dimension_returns_no_partial_bundle() {
        let mut m = model(false); m.fail_after = 4;
        assert!(matches!(plan().execute(&mut m), Err(SentimentError::Scoring(ScoringError::ProjectionFailed))));
        assert_eq!(m.calls, 5);
    }
    #[test]
    fn scoring_semantics_are_bound_and_full_denominators_are_priced() {
        let a = ir(10, 8); let points = anchors(); let input = [InputIr { axis: SentimentAxis::Valence, ir: &a, anchors: &points }];
        let mut opts = options(); opts.mode = ScoringMode::FullVocabulary;
        let full = SentimentPlan::compile_irs(&input, opts, SentimentLimits::default()).unwrap();
        assert_eq!(full.planned_work().projected_logits, 4 * NANBEIGE_VOCAB_SIZE as u64);
        let selected = SentimentPlan::compile_irs(&input, options(), SentimentLimits::default()).unwrap();
        assert_ne!(full.binding_digest(), selected.binding_digest());
        let result = full.execute(&mut model(true)).unwrap();
        assert!(result.dimensions[0].scores.full_vocab_denominators_computed);
    }
}
