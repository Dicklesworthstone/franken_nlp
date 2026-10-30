//! Independent affect heads through the typed finite-score request contract.
use super::*;
use crate::tasks::{ir::TaskBudget, sentiment::{SentimentPolicy, SentimentRequest}};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SentimentArgs { scoring: classify::ScoringArgs }

pub(super) fn definition() -> clap::Command {
    classify::prefill_definition(classify::definition()).name("sentiment")
        .about("Score independent affect dimensions with native grouped INT4")
        .long_about("Score the requested valence, arousal, dominance and approach axes as independent exact five-bin continuation languages. Input is the same bounded typed {document, axes?, policy?} JSON object as candidate sentiment. Each candidate includes terminal EOS and every probability uses the full vocabulary denominator. Axes retain separate distributions, means, dispersion, entropy and explicit uncalibrated abstention; no joint emotion simplex or psychological measurement is implied. All requested heads, context, and five native work axes preflight before the first forward. No sampling, free text, thinking, tools, prompt/model overrides or silent backend fallback. Source conversion and independently digest-pinned snapshots share the same path. Only the completed bundle is published; one head failure publishes no partial result. Blocking IO observes the end-to-end deadline cooperatively.")
        .mut_arg("input", |arg| arg.help("Bounded sentiment JSON object; '-' reads stdin"))
        .mut_arg("max_result_bytes", |arg| arg.help("Complete native sentiment result; candidate provenance is additional"))
        .mut_arg("max_forward_positions", |arg| arg.help("Whole-bundle native forward ceiling, shared by every requested axis"))
}
impl SentimentArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { scoring: classify::ScoringArgs::from_matches_with_prefill(matches)? })
    }
    pub(super) fn validate(self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        let mut limits = self.scoring.validate_with_prefill(args)?;
        // CLI preparation retains the typed input, planner and publication
        // staging while the host independently owns the compiled bundle.
        // Full context KV and native output are already priced by storage.
        limits.minimum_memory_bytes = sum(&[limits.minimum_memory_bytes, limits.preparation_bytes])?;
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

fn parse_request(text: &str, budget: TaskBudget, cap: usize)
    -> Result<(SentimentRequest, SentimentPolicy), CandidateError> {
    use crate::candidate_cli::scored::{self, Kind, Request};
    match scored::request(Kind::Sentiment, text, budget, cap)? {
        Request::Sentiment { request, policy } => Ok((request, policy)),
        Request::Classify(_) => Err(CandidateError::Input),
    }
}
