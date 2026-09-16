//! Full-source NLI plus exhaustive bounded evidence-window classification.
//! Byte-verified quotes prove membership only. Correlated model reads and
//! large candidate weights are not entailment or correctness certificates.

use serde::{Deserialize, Serialize};
use crate::{
    execution_identity::Sha256Digest,
    native_engine::{decode::DecodeStepControl, hf_bf16_eager::{HfBf16EagerEngine,
        candidate_scoring::PrefixBudget}, lmhead::scoring::{CandidateScores, ScoringWork}},
    tasks::{extract::SourceDocument, ir::{DecodeStrategy, PromptSegmentKind, TaskPlan}},
    tokenizer::specials::TemplateControlIds,
    validation::{SourceSpan, validate_source_span, grounded_fields::VerifiedSourceSpan},
};
use super::{common::{Bundle, reserved}, native, EagerJudgeRun, JudgeError, JudgeLimits, JudgeLogits, JudgeNativeError};

pub const FAITHFULNESS_VERSION: &str = "judge-full-source-and-evidence-nli-v1";
pub const EVIDENCE_PARTITION_VERSION: &str = "utf8-complete-delimiter-windows-v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FaithfulnessRelation { Entailed, Contradicted, Unsupported }
impl FaithfulnessRelation {
    pub(super) fn id(self) -> &'static str {
        match self { Self::Entailed => "entailed", Self::Contradicted => "contradicted", Self::Unsupported => "unsupported" }
    }
}

/// Explicit UNCALIBRATED policies. At most 31 windows plus the full-source
/// head fit the common judge bound. No tail or evidence match is truncated.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FaithfulnessPolicy {
    pub minimum_candidate_weight_ppm: u32,
    pub minimum_margin_milli: u32,
    pub evidence_window_bytes: u32,
    pub max_evidence_windows: u32,
    pub max_evidence_spans: u32,
}
impl FaithfulnessPolicy {
    pub fn validate(self) -> Result<(), JudgeError> {
        if self.minimum_candidate_weight_ppm > 1_000_000 || self.evidence_window_bytes == 0
            || self.max_evidence_windows == 0 || self.max_evidence_windows > 31
            || self.max_evidence_spans == 0 || self.max_evidence_spans > self.max_evidence_windows {
            return Err(JudgeError::Contract("faithfulness thresholds or evidence limits"));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FaithfulnessAbstention { AmbiguousDistribution, InsufficientEvidence, ConflictingEvidence }
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FaithfulnessAssessment {
    /// Descriptive argmax, retained even when the policy abstains.
    pub top_relation: FaithfulnessRelation,
    pub candidate_conditional_weight: f64,
    pub log_score_margin: f64,
    pub passes_policy: bool,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FaithfulnessWindow {
    pub span: VerifiedSourceSpan,
    pub score_head: usize,
    pub assessment: FaithfulnessAssessment,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FaithfulnessEvidence {
    pub span: VerifiedSourceSpan,
    pub quote: String,
    pub relation: FaithfulnessRelation,
    pub score_head: usize,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FaithfulnessResult {
    pub schema_version: u32,
    pub task_spec: String,
    pub algorithm: String,
    pub evidence_partition: String,
    pub calibration: String,
    /// Unsupported is a model relation to this source, not a timeout or error.
    pub relation: Option<FaithfulnessRelation>,
    pub abstention: Option<FaithfulnessAbstention>,
    pub global: FaithfulnessAssessment,
    /// Bounded source windows, not necessarily minimal supporting clauses.
    pub evidence: Vec<FaithfulnessEvidence>,
    /// Includes all uncertain and contrary windows; none disappear from work.
    pub windows: Vec<FaithfulnessWindow>,
    /// Head zero is the whole source; remaining heads are its windows. When
    /// the source fits in one window, head zero is reused rather than rerun.
    pub heads: Vec<CandidateScores>,
    pub policy: FaithfulnessPolicy,
    pub work: ScoringWork,
}

/// Private source binding, with no Deserialize, Serialize or Debug surface.
pub struct FaithfulnessPlan {
    pub(super) bundle: Bundle,
    source: String,
    spans: Vec<VerifiedSourceSpan>,
    policy: FaithfulnessPolicy,
}
impl FaithfulnessPlan {
    /// The pinned source encoder binds claim/source bytes to untrusted tokens.
    /// Exact ABI: global / instruction / source / instruction / claim / scaffold.
    /// Head zero sees ALL bytes. Subsequent heads see the complete deterministic
    /// partition, not a filtered retrieval result that could omit counterevidence.
    pub fn from_task_plans(tasks: &[TaskPlan], source: &SourceDocument, claim: &SourceDocument,
        eos: u32, controls: &TemplateControlIds, policy: FaithfulnessPolicy, limits: JudgeLimits)
        -> Result<Self, JudgeError> {
        policy.validate()?;
        if source.text().is_empty() || claim.text().is_empty()
            || source.token_ids().len() != source.text().len() || claim.token_ids().len() != claim.text().len() {
            return Err(JudgeError::Contract("faithfulness requires exact nonempty byte-token sources"));
        }
        let spans = partition_evidence(source.text(), policy)?;
        let count = if spans.len() == 1 { 1 } else { spans.len() + 1 };
        if tasks.len() != count { return Err(JudgeError::Contract("faithfulness requires every evidence head")); }
        for (index, task) in tasks.iter().enumerate() {
            let range = if index == 0 { 0..source.text().len() }
                else { spans[index - 1].byte_start..spans[index - 1].byte_end };
            validate_head(task, &tasks[0], &source.token_ids()[range], claim.token_ids())?;
        }
        #[derive(Serialize)]
        struct Policy<'a> { algorithm: &'static str, partition: &'static str,
            policy: FaithfulnessPolicy, spans: &'a [VerifiedSourceSpan] }
        let refs: Vec<_> = tasks.iter().collect();
        let bundle = Bundle::compile(&refs, eos, controls,
            &Policy { algorithm: FAITHFULNESS_VERSION, partition: EVIDENCE_PARTITION_VERSION, policy, spans: &spans }, limits)?;
        Ok(Self { bundle, source: source.text().to_owned(), spans, policy })
    }
    pub fn binding_digest(&self) -> &Sha256Digest { &self.bundle.binding }
    pub fn planned_work(&self) -> ScoringWork { self.bundle.work }
    pub fn execute<M: JudgeLogits>(&self, model: &mut M) -> Result<FaithfulnessResult, JudgeError> {
        self.finish(self.bundle.score(model)?)
    }
    pub fn execute_eager_with_control<C: DecodeStepControl>(&self, engine: &mut HfBf16EagerEngine,
        budget: PrefixBudget, control: &mut C) -> Result<EagerJudgeRun<FaithfulnessResult>, JudgeNativeError> {
        let (scores, work) = native::score_bundle(&self.bundle, engine, budget, control)?;
        native::wrap(&self.bundle, self.finish(scores)?, work)
    }
    pub(super) fn finish(&self, scores: Vec<CandidateScores>) -> Result<FaithfulnessResult, JudgeError> {
        if scores.len() != self.bundle.heads.len() { return Err(JudgeError::InvalidScores); }
        let global = assess(&scores[0], self.policy)?;
        let mut windows = reserved(self.spans.len())?;
        for (index, &span) in self.spans.iter().enumerate() {
            let head = if self.spans.len() == 1 { 0 } else { index + 1 };
            windows.push(FaithfulnessWindow { span, score_head: head, assessment: assess(&scores[head], self.policy)? });
        }
        let (relation, abstention) = decide(&global, &windows);
        let mut evidence = reserved(self.policy.max_evidence_spans as usize)?;
        if let Some(relation @ (FaithfulnessRelation::Entailed | FaithfulnessRelation::Contradicted)) = relation {
            for window in &windows {
                if !window.assessment.passes_policy || window.assessment.top_relation != relation { continue; }
                if evidence.len() == self.policy.max_evidence_spans as usize { return Err(JudgeError::Limit("evidence_spans")); }
                let span = window.span;
                let quote = self.source.get(span.byte_start..span.byte_end).ok_or(JudgeError::InvalidScores)?;
                validate_source_span(&self.source, quote, SourceSpan::new(span.byte_start, span.byte_end,
                    span.scalar_start, span.scalar_end)).map_err(|_| JudgeError::InvalidScores)?;
                evidence.push(FaithfulnessEvidence { span, quote: quote.to_owned(), relation, score_head: window.score_head });
            }
        }
        let result = FaithfulnessResult { schema_version: 1, task_spec: "judge-v1".to_owned(),
            algorithm: FAITHFULNESS_VERSION.to_owned(), evidence_partition: EVIDENCE_PARTITION_VERSION.to_owned(),
            calibration: "uncalibrated_correlated_reads_membership_not_entailment_proof".to_owned(),
            relation, abstention, global, evidence, windows, heads: scores, policy: self.policy, work: self.bundle.work };
        self.bundle.check_output(&result)?;
        Ok(result)
    }
}
fn validate_head(task: &TaskPlan, first: &TaskPlan, source: &[u32], claim: &[u32]) -> Result<(), JudgeError> {
    use PromptSegmentKind::{GlobalPolicy, TaskInstruction, Document, AnswerScaffold};
    let layout = [GlobalPolicy, TaskInstruction, Document, TaskInstruction, Document, AnswerScaffold];
    let segments = task.ir().prompt_segments(); let baseline = first.ir().prompt_segments();
    if segments.len() != layout.len() || baseline.len() != layout.len()
        || segments.iter().zip(layout).any(|(s, k)| s.kind() != k || s.token_ids().is_empty())
        || segments[2].token_ids() != source || segments[4].token_ids() != claim
        || (0..layout.len()).any(|i| i != 2 && segments[i] != baseline[i]) {
        return Err(JudgeError::Contract("faithfulness prompt, claim or source binding"));
    }
    let DecodeStrategy::PrefillOnly { candidates } = task.ir().decode_strategy() else {
        return Err(JudgeError::Contract("faithfulness requires finite NLI labels"));
    };
    let DecodeStrategy::PrefillOnly { candidates: expected } = first.ir().decode_strategy() else {
        return Err(JudgeError::Contract("faithfulness requires finite NLI labels"));
    };
    if candidates.len() != 3 || expected.len() != 3
        || [FaithfulnessRelation::Entailed, FaithfulnessRelation::Contradicted, FaithfulnessRelation::Unsupported].iter().any(|r| {
            let a = candidates.iter().find(|c| c.id() == r.id());
            let b = expected.iter().find(|c| c.id() == r.id());
            a.is_none() || b.is_none() || a != b
        }) { return Err(JudgeError::Contract("faithfulness complete shared label vocabulary")); }
    Ok(())
}
fn assess(scores: &CandidateScores, policy: FaithfulnessPolicy) -> Result<FaithfulnessAssessment, JudgeError> {
    let mut ordered = Vec::with_capacity(3);
    for relation in [FaithfulnessRelation::Unsupported, FaithfulnessRelation::Contradicted, FaithfulnessRelation::Entailed] {
        let score = scores.candidates.iter().find(|c| c.id == relation.id()).ok_or(JudgeError::InvalidScores)?;
        if !score.sequence_score.is_finite() || !score.candidate_weight.is_finite() { return Err(JudgeError::InvalidScores); }
        ordered.push((relation, score));
    }
    // Stable diagnostic tie order; exact ties abstain even at threshold zero.
    ordered.sort_by(|a, b| b.1.sequence_score.total_cmp(&a.1.sequence_score));
    let margin = ordered[0].1.sequence_score - ordered[1].1.sequence_score;
    if !margin.is_finite() { return Err(JudgeError::InvalidScores); }
    let weight = ordered[0].1.candidate_weight;
    Ok(FaithfulnessAssessment { top_relation: ordered[0].0, candidate_conditional_weight: weight,
        log_score_margin: margin, passes_policy: margin > 0.0
            && margin >= f64::from(policy.minimum_margin_milli) / 1000.0
            && weight >= f64::from(policy.minimum_candidate_weight_ppm) / 1_000_000.0 })
}
fn decide(global: &FaithfulnessAssessment, windows: &[FaithfulnessWindow])
    -> (Option<FaithfulnessRelation>, Option<FaithfulnessAbstention>) {
    use FaithfulnessRelation::*;
    use FaithfulnessAbstention::*;
    if !global.passes_policy { return (None, Some(AmbiguousDistribution)); }
    let has = |relation| windows.iter().any(|w| w.assessment.passes_policy && w.assessment.top_relation == relation);
    let entail = has(Entailed); let contradict = has(Contradicted);
    let conflict = match global.top_relation { Entailed => contradict, Contradicted => entail, Unsupported => entail || contradict };
    if conflict { return (None, Some(ConflictingEvidence)); }
    let missing = match global.top_relation { Entailed => !entail, Contradicted => !contradict, Unsupported => false };
    if missing { (None, Some(InsufficientEvidence)) } else { (Some(global.top_relation), None) }
}

/// Complete UTF-8 windows, preferring a newline/sentence-like delimiter in
/// each window's latter half, otherwise a scalar boundary. No normalization,
/// approximate offsets or dropped tail. Not a qualified sentence splitter.
pub fn partition_evidence(source: &str, policy: FaithfulnessPolicy) -> Result<Vec<VerifiedSourceSpan>, JudgeError> {
    policy.validate()?;
    if source.is_empty() { return Err(JudgeError::Contract("empty faithfulness source")); }
    let max_bytes = policy.evidence_window_bytes as usize;
    let upper = max_bytes.checked_mul(policy.max_evidence_windows as usize).ok_or(JudgeError::Limit("evidence_bytes"))?;
    if source.len() > upper { return Err(JudgeError::Limit("evidence_windows")); }
    let mut spans = reserved(policy.max_evidence_windows as usize)?;
    let mut start = 0; let mut scalar = 0;
    while start < source.len() {
        if spans.len() == policy.max_evidence_windows as usize { return Err(JudgeError::Limit("evidence_windows")); }
        let mut end = start.saturating_add(max_bytes).min(source.len());
        while end > start && !source.is_char_boundary(end) { end -= 1; }
        if end == start { return Err(JudgeError::Limit("evidence_window_utf8_width")); }
        if end < source.len() {
            let lower = start + (end - start) / 2;
            if let Some(boundary) = source[start..end].char_indices().filter_map(|(i, c)| {
                let next = start + i + c.len_utf8();
                (next >= lower && matches!(c, '\n' | '.' | '!' | '?')).then_some(next)
            }).last() { end = boundary; }
        }
        let next_scalar = scalar + source[start..end].chars().count();
        spans.push(VerifiedSourceSpan { byte_start: start, byte_end: end, scalar_start: scalar, scalar_end: next_scalar });
        start = end; scalar = next_scalar;
    }
    Ok(spans)
}

#[cfg(test)]
mod tests;
