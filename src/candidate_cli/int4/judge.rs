//! Complete native INT4 judgment through the existing typed request contract.
use super::*;
use crate::tasks::{ir::TaskBudget, judge::{JudgeRequest, partition_evidence}};
use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct JudgeArgs { scoring: classify::ScoringArgs, encoding: EncodingArgs }

/// Repeated orders, criteria, evidence windows and document probes consume
/// tokenizer work independently of the original request and native context.
#[derive(Args, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct EncodingArgs {
    #[arg(long, value_name = "BYTES")]
    max_tokenizer_input_bytes: Option<usize>,
    #[arg(long, value_name = "TOKENS")]
    max_tokenizer_output_tokens: Option<usize>,
}
impl EncodingArgs {
    pub(super) fn policy(self, args: &Int4Args) -> Result<UntrustedEncoding, CandidateError> {
        self.policy_with_defaults(args, SafeBpeLimits::MAX_INPUT_BYTES, SafeBpeLimits::MAX_OUTPUT_TOKENS)
    }
    pub(super) fn policy_with_defaults(self, args: &Int4Args, input_bytes: usize, output_tokens: usize)
        -> Result<UntrustedEncoding, CandidateError> {
        if args.input_encoding.is_default() && self != Self::default() {
            return Err(CandidateError::Arguments);
        }
        args.input_encoding.policy(
            self.max_tokenizer_input_bytes.unwrap_or(input_bytes),
            self.max_tokenizer_output_tokens.unwrap_or(output_tokens))
    }
    pub(super) fn workspace_bytes(self, args: &Int4Args) -> Result<u64, CandidateError> {
        match self.policy(args)? {
            UntrustedEncoding::ByteFallback => Ok(0),
            UntrustedEncoding::SafeBpeV1(limits) => limits.requirements()
                .map(|required| required.workspace_bytes).map_err(|_| CandidateError::Arguments),
        }
    }
}
pub(super) fn encoding_definition(command: clap::Command) -> clap::Command {
    EncodingArgs::augment_args(input_encoding::definition(command))
        .mut_arg("max_tokenizer_input_bytes", |arg| arg.help("Cumulative untrusted bytes encoded across the complete judgment or document attempt; safe-bpe-v1 only, default 1048576"))
        .mut_arg("max_tokenizer_output_tokens", |arg| arg.help("Cumulative untrusted tokens produced across the complete judgment or document attempt; safe-bpe-v1 only, default 1048576"))
        .mut_arg("max_tokenizer_pair_checks", |arg| arg.help("Cumulative BPE pair checks across all judgment heads and document probes; safe-bpe-v1 only, default 4194304"))
        .mut_arg("max_tokenizer_pair_bytes", |arg| arg.help("Cumulative examined BPE pair bytes across all judgment heads and document probes; safe-bpe-v1 only, default 67108864"))
}

pub(super) fn definition() -> clap::Command {
    encoding_definition(classify::prefill_definition(classify::definition())).name("judge")
        .about("Run complete pairwise, rubric or faithfulness judgment with native grouped INT4")
        .long_about("Judge one bounded typed JSON request in pairwise, rubric or faithfulness mode with an explicit uncalibrated policy. Pairwise scores both presentation orders; rubric scores every criterion; faithfulness requires the full source to fit and retains every evidence window. Candidates include terminal EOS and full-vocabulary denominators. All heads, context, and five native work axes preflight before the first forward. No sampling, free text, thinking, tools, prompt/model overrides or silent backend fallback. Source conversion and independently digest-pinned snapshots share the same execution path. Only the completed judgment is published; one head failure publishes no partial result. Faithfulness is a model judgment, not a factuality certificate. Blocking IO observes the end-to-end deadline cooperatively.")
        .mut_arg("input", |arg| arg.help("Bounded pairwise/rubric/faithfulness JSON object; '-' reads stdin"))
        .mut_arg("max_result_bytes", |arg| arg.help("Complete native judgment result; candidate provenance is additional"))
        .mut_arg("max_forward_positions", |arg| arg.help("Whole-judgment native forward ceiling, shared by every order, criterion or evidence head"))
        .after_long_help("--input-encoding safe-bpe-v1 preserves exact request bytes and excludes template-control tokens. One bounded tokenizer ledger covers both pairwise orders, every rubric criterion, the complete source and every separately encoded evidence window. The four tokenizer ceilings bound cumulative planning work, independently of request size and native context. Evidence remains in original UTF-8 byte/scalar coordinates. One transient encoder workspace is admitted before input IO. Serial, layer-major prompts, native head cohorts and persistent CPU teams retain their existing scheduling rules. Compact input changes prompt tokens and makes no output-parity, task-quality or speed claim. Byte fallback remains the default.")
}
impl JudgeArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Ok(Self { scoring: classify::ScoringArgs::from_matches_with_prefill(matches)?,
            encoding: EncodingArgs::from_arg_matches(matches)? })
    }
    pub(super) fn validate(self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        let workspace = self.encoding.workspace_bytes(args)?;
        // The shared scoring quote prices retained plans. This route admits
        // its separately bounded transient encoder once, in the CLI owner;
        // the host receives the prepared plan and performs no tokenization.
        let mut native_args = args.clone();
        native_args.input_encoding = input_encoding::InputEncodingArgs::default();
        let mut limits = self.scoring.validate_one_off_with_prefill(&native_args)?;
        // CLI input/planning/publication and hosted compiled input remain
        // separate charged owners during the complete native invocation.
        if limits.cohort.is_none() {
            limits.minimum_memory_bytes = sum(&[limits.minimum_memory_bytes, limits.preparation_bytes])?;
        }
        limits.minimum_memory_bytes = sum(&[limits.minimum_memory_bytes, workspace])?;
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
