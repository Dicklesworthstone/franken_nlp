//! Explicit finite-candidate judgment; user text never supplies execution authority.
use super::*;
use serde::Deserialize;
use crate::{native_engine::lmhead::scoring::ScoringLimits,
    tasks::{ir::TaskBudget, judge::{JudgeLimits, JudgeRequest, PairwisePolicy,
        RubricDefinition, RubricPolicy, FaithfulnessPolicy, partition_evidence}}};
use super::scored::ScoredArgs;

#[derive(Args)]
pub(crate) struct JudgeCommand {
    #[command(flatten)]
    pub(super) args: ScoredArgs,
}

pub(super) fn definition() -> clap::Command {
    JudgeCommand::augment_args(clap::Command::new("judge")
        .about("Score both comparison orders, every rubric criterion, or full-source faithfulness")
        .long_about("Judge one bounded JSON request with mode pairwise, rubric or faithfulness and an explicit uncalibrated policy. Scores use the complete finite candidate language and full-vocabulary denominators, including EOS. Faithfulness requires the whole source to fit and retains every evidence window; this is not a factuality certificate. No free generation, automatic retries, thinking or tools. One failed head fails the whole request."))
}
impl JudgeCommand {
    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        let (common, limits) = self.args.common()?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::judgment::execute(self, common, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, common, limits, input, output); Err(CandidateError::Unavailable) }
    }
}

/// No defaults silently choose a decision policy. The wire cannot supply a
/// TaskBudget, exact tokens, backend, prompt template, model or work receipt.
/// Rubric origin is a caller declaration, not publisher authentication.
#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum Input {
    Pairwise { criterion: String, a: String, b: String, policy: PairwisePolicy },
    Rubric { document: String, rubric: RubricDefinition, policy: RubricPolicy },
    Faithfulness { source: String, claim: String, policy: FaithfulnessPolicy },
}

pub(super) fn request(json: &str, budget: TaskBudget, cap: usize) -> Result<JudgeRequest, CandidateError> {
    if cap == 0 || cap > MAX_INPUT_BYTES || json.len() > cap { return Err(CandidateError::Input); }
    budget.validate().map_err(|_| CandidateError::Arguments)?;
    let value = canonjson::parse_str_with_limits(json, canonjson::ParseLimits {
        max_depth: 8, max_string_bytes: cap,
    }).map_err(|_| CandidateError::Input)?;
    let input: Input = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
    Ok(match input {
        Input::Pairwise { criterion, a, b, policy } => {
            if criterion.is_empty() || a.is_empty() || b.is_empty() { return Err(CandidateError::Input); }
            // Both thresholds are bounded u32 milli-log-odds, NOT ppm weights.
            // Retain their full supported range rather than changing semantics.
            JudgeRequest::Pairwise { criterion, a, b, policy, budget }
        }
        Input::Rubric { document, rubric, policy } => {
            if document.is_empty() || policy.minimum_peak_weight_ppm > 1_000_000
                || policy.maximum_normalized_entropy_ppm > 1_000_000 { return Err(CandidateError::Input); }
            rubric.validate().map_err(|_| CandidateError::Input)?;
            JudgeRequest::Rubric { document, rubric, policy, budget }
        }
        Input::Faithfulness { source, claim, policy } => {
            if source.is_empty() || claim.is_empty() { return Err(CandidateError::Input); }
            policy.validate().map_err(|_| CandidateError::Input)?;
            // Reject an impossible/excessive complete evidence partition before
            // opening model metadata. Never truncate the tail or invent quotes.
            partition_evidence(&source, policy).map_err(|_| CandidateError::Input)?;
            JudgeRequest::Faithfulness { source, claim, policy, budget }
        }
    })
}

/// Same host scoring resources as classify/sentiment, with the judge's closed
/// 33-candidate head ceiling. All aggregate counters cover the entire request.
/// This prices compiled language; it never changes its probability semantics.
pub(super) fn planning_limits(args: &ScoredArgs, budget: TaskBudget) -> JudgeLimits {
    let nodes = budget.max_grammar_states as usize;
    JudgeLimits {
        per_head: ScoringLimits { max_candidates: 33, max_total_tokens: nodes.saturating_sub(1),
            max_nodes: nodes, max_depth: args.max_candidate_tokens, max_candidate_id_bytes: 64,
            max_projected_logits: args.max_projected_logits },
        max_total_prompt_tokens: args.max_forward_positions as usize,
        max_total_candidate_tokens: nodes,
        max_total_projected_logits: args.max_projected_logits,
        max_output_bytes: args.max_result_bytes as u64,
    }
}

#[cfg(test)] pub(super) mod tests;
