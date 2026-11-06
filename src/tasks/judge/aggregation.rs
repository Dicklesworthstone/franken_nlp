//! Complete-source evidence over independent, exact faithfulness judgments.
//! No probability averaging, cross-chunk inference, or factuality certificate.
use super::{FaithfulnessAbstention, FaithfulnessPolicy, FaithfulnessRelation, FaithfulnessResult,
    faithfulness::{self, EVIDENCE_PARTITION_VERSION, FAITHFULNESS_VERSION}};
use crate::{native_engine::{decode::{DecodeCancellationKind, DecodeStepControl},
        lmhead::{NANBEIGE_VOCAB_SIZE, scoring::{CandidateScores, ScoringWork}}},
    tasks::ir::ScoreSpace, validation::grounded_fields::VerifiedSourceSpan};
use serde::{Deserialize, Serialize};

pub const FAITHFULNESS_AGGREGATION_PROFILE: &str = "faithfulness-complete-chunk-evidence-v1";
const SCORE_SCOPE: &str = "independent_chunk_judgments_no_cross_chunk_inference";
const CALIBRATION: &str = "uncalibrated_correlated_reads_membership_not_entailment_proof";
const SCALE: u32 = 1_000_000;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FaithfulnessAggregationPolicy {
    /// Original scalar coverage by non-abstaining complete chunk judgments.
    /// Contrary evidence always blocks acceptance, regardless of this floor.
    pub minimum_classified_coverage_ppm: u32,
}
impl Default for FaithfulnessAggregationPolicy {
    fn default() -> Self { Self { minimum_classified_coverage_ppm: SCALE } }
}
impl FaithfulnessAggregationPolicy {
    pub fn validate(self) -> Result<(), FaithfulnessAggregationError> {
        if self.minimum_classified_coverage_ppm > SCALE { return Err(FaithfulnessAggregationError::Policy); }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FaithfulnessAggregateAbstention {
    ConflictingEvidence, InsufficientClassifiedCoverage, UnresolvedEvidence,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct FaithfulnessCoverage {
    pub entailed_scalars: usize,
    pub contradicted_scalars: usize,
    pub unsupported_scalars: usize,
    pub abstained_scalars: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FaithfulnessAggregateEvidence {
    pub chunk_id: usize,
    /// Index into this chunk's unchanged native result heads.
    pub score_head: usize,
    pub span: VerifiedSourceSpan,
    pub quote: String,
    pub relation: FaithfulnessRelation,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FaithfulnessAggregate {
    pub schema_version: u32,
    pub profile: String,
    pub score_scope: String,
    pub calibration: String,
    pub policy: FaithfulnessAggregationPolicy,
    pub local_policy: FaithfulnessPolicy,
    pub chunk_count: usize,
    pub source_bytes: usize,
    pub source_scalars: usize,
    pub classified_scalars: usize,
    pub classified_coverage: f64,
    pub relation: Option<FaithfulnessRelation>,
    pub abstention: Option<FaithfulnessAggregateAbstention>,
    pub coverage: FaithfulnessCoverage,
    /// All decisive entailment/contradiction windows, including counterevidence
    /// when the document abstains. Original native chunk results retain every
    /// unsupported and uncertain window as well.
    pub evidence: Vec<FaithfulnessAggregateEvidence>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaithfulnessAggregationError {
    Policy, Coverage, Result, Evidence, Allocation, Cancelled(DecodeCancellationKind),
}
impl std::fmt::Display for FaithfulnessAggregationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Policy => "invalid faithfulness aggregation policy",
            Self::Coverage => "faithfulness chunks do not cover the complete original source",
            Self::Result => "faithfulness chunk result contract differs",
            Self::Evidence => "faithfulness evidence does not match the original source",
            Self::Allocation => "faithfulness aggregation allocation refused",
            Self::Cancelled(_) => "faithfulness aggregation cancelled",
        })
    }
}
impl std::error::Error for FaithfulnessAggregationError {}

pub(crate) struct FaithfulnessChunkView<'a> {
    pub chunk_id: usize,
    pub span: VerifiedSourceSpan,
    pub result: &'a FaithfulnessResult,
}
impl FaithfulnessAggregate {
    pub(crate) fn verify_metadata(&self, policy: FaithfulnessAggregationPolicy, local_policy: FaithfulnessPolicy,
        count: usize, bytes: usize, scalars: usize) -> Result<(), FaithfulnessAggregationError> {
        policy.validate()?;
        local_policy.validate().map_err(|_| FaithfulnessAggregationError::Policy)?;
        let classified = add(add(self.coverage.entailed_scalars, self.coverage.contradicted_scalars)?,
            self.coverage.unsupported_scalars)?;
        if self.schema_version != 1 || self.profile != FAITHFULNESS_AGGREGATION_PROFILE
            || self.score_scope != SCORE_SCOPE || self.calibration != CALIBRATION
            || self.policy != policy || self.local_policy != local_policy
            || self.chunk_count != count || count == 0 || self.source_bytes != bytes || bytes == 0
            || self.source_scalars != scalars || scalars == 0 || classified != self.classified_scalars
            || add(classified, self.coverage.abstained_scalars)? != scalars
            || self.classified_coverage != classified as f64 / scalars as f64
            || self.relation.is_some() == self.abstention.is_some() {
            return Err(FaithfulnessAggregationError::Result);
        }
        Ok(())
    }
}

pub(crate) fn aggregate_with_control<'a>(chunks: impl IntoIterator<Item = FaithfulnessChunkView<'a>>,
    count: usize, source: &str, local_policy: FaithfulnessPolicy, policy: FaithfulnessAggregationPolicy,
    control: &mut impl DecodeStepControl) -> Result<FaithfulnessAggregate, FaithfulnessAggregationError> {
    checkpoint(control)?;
    policy.validate()?;
    local_policy.validate().map_err(|_| FaithfulnessAggregationError::Policy)?;
    if source.is_empty() || count == 0 || count > 256 { return Err(FaithfulnessAggregationError::Coverage); }
    let mut evidence = Vec::new();
    evidence.try_reserve_exact(count.checked_mul(local_policy.max_evidence_windows as usize)
        .ok_or(FaithfulnessAggregationError::Allocation)?).map_err(|_| FaithfulnessAggregationError::Allocation)?;
    let mut coverage = FaithfulnessCoverage::default();
    let mut byte = 0; let mut scalar = 0; let mut seen = 0;
    let mut entail = false; let mut contradict = false; let mut local_conflict = false;
    let mut signature = None;
    for chunk in chunks {
        checkpoint(control)?;
        let span = chunk.span;
        if chunk.chunk_id != seen || seen >= count || span.byte_start != byte || span.scalar_start != scalar
            || span.byte_end <= byte || span.scalar_end <= scalar {
            return Err(FaithfulnessAggregationError::Coverage);
        }
        let text = source.get(byte..span.byte_end).ok_or(FaithfulnessAggregationError::Coverage)?;
        let length = text.chars().count();
        if add(scalar, length)? != span.scalar_end { return Err(FaithfulnessAggregationError::Coverage); }
        let result = chunk.result;
        validate_result(text, result, local_policy, &mut signature, control)?;
        let target = match result.relation {
            Some(FaithfulnessRelation::Entailed) => &mut coverage.entailed_scalars,
            Some(FaithfulnessRelation::Contradicted) => &mut coverage.contradicted_scalars,
            Some(FaithfulnessRelation::Unsupported) => &mut coverage.unsupported_scalars,
            None => &mut coverage.abstained_scalars,
        };
        *target = add(*target, length)?;
        local_conflict |= result.abstention == Some(FaithfulnessAbstention::ConflictingEvidence);
        for window in &result.windows {
            checkpoint(control)?;
            let relation = window.assessment.top_relation;
            if !window.assessment.passes_policy || relation == FaithfulnessRelation::Unsupported { continue; }
            entail |= relation == FaithfulnessRelation::Entailed;
            contradict |= relation == FaithfulnessRelation::Contradicted;
            let local = window.span;
            let lifted = VerifiedSourceSpan { byte_start: add(byte, local.byte_start)?, byte_end: add(byte, local.byte_end)?,
                scalar_start: add(scalar, local.scalar_start)?, scalar_end: add(scalar, local.scalar_end)? };
            let quote = source.get(lifted.byte_start..lifted.byte_end).ok_or(FaithfulnessAggregationError::Evidence)?;
            let mut owned = String::new();
            owned.try_reserve_exact(quote.len()).map_err(|_| FaithfulnessAggregationError::Allocation)?;
            owned.push_str(quote);
            evidence.push(FaithfulnessAggregateEvidence { chunk_id: seen, score_head: window.score_head,
                span: lifted, quote: owned, relation });
        }
        byte = span.byte_end; scalar = span.scalar_end; seen += 1;
    }
    if seen != count || byte != source.len() || scalar == 0 { return Err(FaithfulnessAggregationError::Coverage); }
    let classified = add(add(coverage.entailed_scalars, coverage.contradicted_scalars)?, coverage.unsupported_scalars)?;
    let covered = classified > 0 && (classified as u128) * u128::from(SCALE)
        >= (scalar as u128) * u128::from(policy.minimum_classified_coverage_ppm);
    use FaithfulnessAggregateAbstention::*;
    let (relation, abstention) = if local_conflict || (entail && contradict) {
        (None, Some(ConflictingEvidence))
    } else if !covered { (None, Some(InsufficientClassifiedCoverage))
    } else if coverage.entailed_scalars > 0 { (Some(FaithfulnessRelation::Entailed), None)
    } else if coverage.contradicted_scalars > 0 { (Some(FaithfulnessRelation::Contradicted), None)
    } else if entail || contradict { (None, Some(UnresolvedEvidence))
    } else { (Some(FaithfulnessRelation::Unsupported), None) };
    let aggregate = FaithfulnessAggregate { schema_version: 1, profile: FAITHFULNESS_AGGREGATION_PROFILE.to_owned(),
        score_scope: SCORE_SCOPE.to_owned(), calibration: CALIBRATION.to_owned(), policy, local_policy,
        chunk_count: count, source_bytes: byte, source_scalars: scalar, classified_scalars: classified,
        classified_coverage: classified as f64 / scalar as f64, relation, abstention, coverage, evidence };
    aggregate.verify_metadata(policy, local_policy, count, byte, scalar)?;
    checkpoint(control)?;
    Ok(aggregate)
}

fn validate_result(source: &str, result: &FaithfulnessResult, policy: FaithfulnessPolicy,
    signature: &mut Option<(u32, [usize; 3])>, control: &mut impl DecodeStepControl)
    -> Result<(), FaithfulnessAggregationError> {
    let windows = faithfulness::partition_evidence(source, policy).map_err(|_| FaithfulnessAggregationError::Result)?;
    let heads = if windows.len() == 1 { 1 } else { windows.len() + 1 };
    if result.schema_version != 1 || result.task_spec != "judge-v1" || result.algorithm != FAITHFULNESS_VERSION
        || result.evidence_partition != EVIDENCE_PARTITION_VERSION || result.calibration != CALIBRATION
        || result.policy != policy || result.windows.len() != windows.len() || result.heads.len() != heads {
        return Err(FaithfulnessAggregationError::Result);
    }
    let mut work = ScoringWork { prefix_evaluations: 0, scored_edges: 0, projected_logits: 0 };
    for head in &result.heads {
        checkpoint(control)?;
        validate_scores(head, signature)?;
        work.prefix_evaluations = add(work.prefix_evaluations, head.work.prefix_evaluations)?;
        work.scored_edges = add(work.scored_edges, head.work.scored_edges)?;
        work.projected_logits = work.projected_logits.checked_add(head.work.projected_logits)
            .ok_or(FaithfulnessAggregationError::Result)?;
    }
    if result.work != work || faithfulness::assess(&result.heads[0], policy)
        .map_err(|_| FaithfulnessAggregationError::Result)? != result.global {
        return Err(FaithfulnessAggregationError::Result);
    }
    for (index, (window, &span)) in result.windows.iter().zip(&windows).enumerate() {
        checkpoint(control)?;
        let head = if windows.len() == 1 { 0 } else { index + 1 };
        if window.span != span || window.score_head != head || faithfulness::assess(&result.heads[head], policy)
            .map_err(|_| FaithfulnessAggregationError::Result)? != window.assessment {
            return Err(FaithfulnessAggregationError::Result);
        }
    }
    if faithfulness::decide(&result.global, &result.windows) != (result.relation, result.abstention) {
        return Err(FaithfulnessAggregationError::Result);
    }
    let mut retained = result.evidence.iter();
    if let Some(relation @ (FaithfulnessRelation::Entailed | FaithfulnessRelation::Contradicted)) = result.relation {
        for window in &result.windows {
            if !window.assessment.passes_policy || window.assessment.top_relation != relation { continue; }
            let item = retained.next().ok_or(FaithfulnessAggregationError::Evidence)?;
            if item.span != window.span || item.score_head != window.score_head || item.relation != relation
                || source.get(item.span.byte_start..item.span.byte_end) != Some(item.quote.as_str()) {
                return Err(FaithfulnessAggregationError::Evidence);
            }
        }
    }
    if retained.next().is_some() || result.evidence.len() > policy.max_evidence_spans as usize {
        return Err(FaithfulnessAggregationError::Evidence);
    }
    Ok(())
}
fn validate_scores(scores: &CandidateScores, signature: &mut Option<(u32, [usize; 3])>)
    -> Result<(), FaithfulnessAggregationError> {
    if scores.score_space != ScoreSpace::FullVocabSequenceLogprob || !scores.full_vocab_denominators_computed
        || scores.normalization_scope != "full_vocabulary_per_position; candidate_weights_over_completed_candidates"
        || scores.length_rule != "sum_logprobs_including_eos" || scores.eos_rule != "append_and_score_exactly_one_eos"
        || scores.eos_token_id as usize >= NANBEIGE_VOCAB_SIZE || scores.candidates.len() != 3
        || scores.work.prefix_evaluations == 0 || scores.work.scored_edges < 3
        || (scores.work.prefix_evaluations as u64).checked_mul(NANBEIGE_VOCAB_SIZE as u64) != Some(scores.work.projected_logits) {
        return Err(FaithfulnessAggregationError::Result);
    }
    let mut depths = [0; 3];
    for (index, (candidate, id)) in scores.candidates.iter().zip(["contradicted", "entailed", "unsupported"]).enumerate() {
        if candidate.id != id || candidate.scored_tokens < 2 || !candidate.sequence_score.is_finite()
            || candidate.sequence_score > 1e-10
            || !candidate.candidate_weight.is_finite() || !(0.0..=1.0).contains(&candidate.candidate_weight) {
            return Err(FaithfulnessAggregationError::Result);
        }
        depths[index] = candidate.scored_tokens;
    }
    let current = (scores.eos_token_id, depths);
    if signature.is_some_and(|expected| expected != current) { return Err(FaithfulnessAggregationError::Result); }
    *signature = Some(current);
    let maximum = scores.candidates.iter().map(|c| c.sequence_score).fold(f64::NEG_INFINITY, f64::max);
    let mass: f64 = scores.candidates.iter().map(|c| (c.sequence_score - maximum).exp()).sum();
    if scores.candidates.iter().any(|candidate|
        ((candidate.sequence_score - maximum).exp() / mass - candidate.candidate_weight).abs() > 32.0 * f64::EPSILON) {
        return Err(FaithfulnessAggregationError::Result);
    }
    Ok(())
}
fn add(a: usize, b: usize) -> Result<usize, FaithfulnessAggregationError> {
    a.checked_add(b).ok_or(FaithfulnessAggregationError::Coverage)
}
fn checkpoint(control: &mut impl DecodeStepControl) -> Result<(), FaithfulnessAggregationError> {
    control.checkpoint(0).map_or(Ok(()), |cause| Err(FaithfulnessAggregationError::Cancelled(cause)))
}

#[cfg(test)] mod tests;
