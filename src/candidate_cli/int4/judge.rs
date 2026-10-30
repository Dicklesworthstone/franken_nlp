//! Complete native INT4 judgment through the existing typed request contract.
use super::*;
use crate::tasks::{ir::TaskBudget, judge::{JudgeRequest, partition_evidence}};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct JudgeArgs { scoring: classify::ScoringArgs }

pub(super) fn definition() -> clap::Command {
    classify::prefill_definition(classify::definition()).name("judge")
        .about("Run complete pairwise, rubric or faithfulness judgment with native grouped INT4")
        .long_about("Judge one bounded typed JSON request in pairwise, rubric or faithfulness mode with an explicit uncalibrated policy. Pairwise scores both presentation orders; rubric scores every criterion; faithfulness requires the full source to fit and retains every evidence window. Candidates include terminal EOS and full-vocabulary denominators. All heads, context, and five native work axes preflight before the first forward. No sampling, free text, thinking, tools, prompt/model overrides or silent backend fallback. Source conversion and independently digest-pinned snapshots share the same execution path. Only the completed judgment is published; one head failure publishes no partial result. Faithfulness is a model judgment, not a factuality certificate. Blocking IO observes the end-to-end deadline cooperatively.")
        .mut_arg("input", |arg| arg.help("Bounded pairwise/rubric/faithfulness JSON object; '-' reads stdin"))
        .mut_arg("max_result_bytes", |arg| arg.help("Complete native judgment result; candidate provenance is additional"))
        .mut_arg("max_forward_positions", |arg| arg.help("Whole-judgment native forward ceiling, shared by every order, criterion or evidence head"))
}
impl JudgeArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { scoring: classify::ScoringArgs::from_matches_with_prefill(matches)? })
    }
    pub(super) fn validate(self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        let mut limits = self.scoring.validate_with_prefill(args)?;
        // CLI input/planning/publication and hosted compiled input remain
        // separate charged owners during the complete native invocation.
        if limits.cohort.is_none() {
            limits.minimum_memory_bytes = sum(&[limits.minimum_memory_bytes, limits.preparation_bytes])?;
        }
        if limits.minimum_memory_bytes > limits.memory_bytes { return Err(CandidateError::Arguments); }
        Ok(limits)
    }
    pub(super) fn execute(self, args: Int4Args, input: &mut impl Read, output: &mut impl Write)
        -> Result<(), CandidateError> {
        let limits = self.validate(&args)?;
        #[cfg(feature = "asupersync-runtime")]
        { runtime::execute(args, self, limits, input, output) }
        #[cfg(not(feature = "asupersync-runtime"))]
        { let _ = (self, args, limits, input, output); Err(CandidateError::Unavailable) }
    }
}
fn parse_request(text: &str, budget: TaskBudget, cap: usize) -> Result<JudgeRequest, CandidateError> {
    crate::candidate_cli::judge::request(text, budget, cap)
}
// Independent complete-population check at the CLI boundary. A single
// evidence window already is the full source, so it uses the same one head.
fn request_shape(request: &JudgeRequest) -> Result<(&'static str, usize), CandidateError> {
    Ok(match request {
        JudgeRequest::Pairwise { .. } => ("pairwise", 2),
        JudgeRequest::Rubric { rubric, .. } => ("rubric", rubric.criteria.len()),
        JudgeRequest::Faithfulness { source, policy, .. } => {
            let windows = partition_evidence(source, *policy).map_err(|_| CandidateError::Input)?.len();
            ("faithfulness", if windows == 1 { 1 } else { windows.checked_add(1).ok_or(CandidateError::Input)? })
        }
    })
}
