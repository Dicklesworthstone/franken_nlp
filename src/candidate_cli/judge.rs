//! Explicit finite-candidate judgment; user text never supplies execution authority.
use super::*;
use serde::Deserialize;
use crate::{native_engine::lmhead::scoring::ScoringLimits,
    tasks::{ir::TaskBudget, judge::{JudgeLimits, JudgeRequest, PairwisePolicy,
        RubricDefinition, RubricPolicy, FaithfulnessPolicy, partition_evidence}}};
use super::scored::{ScoredArgs, prefill::ScoringPrefillArgs};
use crate::tokenizer::untrusted::{SafeBpeLimits, UntrustedEncoding};

/// One judgment may encode a full source and all of its evidence windows.
/// Tokenizer work therefore has independent ceilings from wire bytes and the
/// largest native head context; it is never renewed between judgment heads.
#[derive(Args, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::candidate_cli) struct EncodingArgs {
    #[arg(long, value_name = "BYTES")]
    max_tokenizer_input_bytes: Option<usize>,
    #[arg(long, value_name = "TOKENS")]
    max_tokenizer_output_tokens: Option<usize>,
}
impl EncodingArgs {
    pub(in crate::candidate_cli) fn is_default(self) -> bool { self == Self::default() }
    pub(in crate::candidate_cli) fn policy(self, encoding: &input_encoding::InputEncodingArgs)
        -> Result<UntrustedEncoding, CandidateError> {
        if encoding.is_default() && !self.is_default() { return Err(CandidateError::Arguments); }
        encoding.policy(self.max_tokenizer_input_bytes.unwrap_or(SafeBpeLimits::MAX_INPUT_BYTES),
            self.max_tokenizer_output_tokens.unwrap_or(SafeBpeLimits::MAX_OUTPUT_TOKENS))
    }
    pub(in crate::candidate_cli) fn workspace_bytes(self, encoding: &input_encoding::InputEncodingArgs)
        -> Result<u64, CandidateError> {
        match self.policy(encoding)? {
            UntrustedEncoding::ByteFallback => Ok(0),
            UntrustedEncoding::SafeBpeV1(limits) => limits.requirements()
                .map(|quote| quote.workspace_bytes).map_err(|_| CandidateError::Arguments),
        }
    }
}
pub(in crate::candidate_cli) fn encoding_definition(command: clap::Command) -> clap::Command {
    EncodingArgs::augment_args(input_encoding::definition(command))
        .mut_arg("max_tokenizer_input_bytes", |arg| arg.help("Cumulative untrusted bytes encoded across every head of one complete judgment; safe-bpe-v1 only, default 1048576"))
        .mut_arg("max_tokenizer_output_tokens", |arg| arg.help("Cumulative untrusted tokens across every head of one complete judgment; separate from native context, safe-bpe-v1 only, default 1048576"))
        .mut_arg("max_tokenizer_pair_checks", |arg| arg.help("Cumulative BPE pair checks across the complete judgment; safe-bpe-v1 only, default 4194304"))
        .mut_arg("max_tokenizer_pair_bytes", |arg| arg.help("Cumulative examined BPE pair bytes across the complete judgment; safe-bpe-v1 only, default 67108864"))
}

#[derive(Args)]
pub(crate) struct JudgeCommand {
    #[command(flatten)]
    pub(super) args: ScoredArgs,
    #[command(flatten)]
    pub(super) prefill: ScoringPrefillArgs,
    #[arg(skip)]
    pub(super) compute: compute::Args,
    #[arg(skip)]
    pub(super) encoding: input_encoding::InputEncodingArgs,
    #[arg(skip)]
    pub(super) encoding_limits: EncodingArgs,
}

pub(super) fn definition() -> clap::Command {
    compute::definition(encoding_definition(JudgeCommand::augment_args(clap::Command::new("judge")
        .about("Score both comparison orders, every rubric criterion, or full-source faithfulness")
        .long_about("Judge one bounded JSON request with mode pairwise, rubric or faithfulness and an explicit uncalibrated policy. Scores use the complete finite candidate language and full-vocabulary denominators, including EOS. Faithfulness requires the whole source to fit and retains every evidence window; this is not a factuality certificate. No free generation, automatic retries, thinking or tools. One failed head fails the whole request."))))
        .after_long_help("--input-encoding safe-bpe-v1 preserves exact untrusted criterion, candidate, document and claim bytes while excluding template controls. One bounded tokenizer allowance covers the complete pairwise or rubric judgment, or the full faithfulness source plus every original-byte evidence window. --max-tokenizer-input-bytes and --max-tokenizer-output-tokens are cumulative planning-work ceilings, independent of wire size and native context; pair-work caps cover the same complete preparation. The encoder workspace is admitted before private IO and retained through completion. All four serial/prefill/team schedules preserve the selected semantic identity. Compact responses report the exact policy and actual preparation work. Byte-fallback remains the default; compact input changes prompt tokens and makes no tokenizer-parity, identical-output, quality or speed claim.")
}
impl JudgeCommand {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let mut command = Self::from_arg_matches(matches)?;
        command.compute = compute::Args::from_matches(matches)?;
        command.encoding = input_encoding::InputEncodingArgs::from_matches(matches)?;
        command.encoding_limits = EncodingArgs::from_arg_matches(matches)?;
        Ok(command)
    }
    pub(super) fn validate(&self) -> Result<(CandidateArgs, Limits), CandidateError> {
        let workspace = self.encoding_limits.workspace_bytes(&self.encoding)?;
        let prefill = self.prefill.limits()?;
        // The shared score quote prices retained prompts. This route owns a
        // distinct transient encoder whose cumulative bounds are independent
        // from the largest head, so add it once to the caller's held charge.
        let (common, mut limits) = self.args.common_with_compute(
            &input_encoding::InputEncodingArgs::default(), self.compute, prefill, false)?;
        limits.preparation_bytes = limits.preparation_bytes.checked_add(workspace)
            .ok_or(CandidateError::Arguments)?;
        let native = Int8MemoryRequirement::for_context(self.args.context_tokens)
            .map_err(|_| CandidateError::Arguments)?;
        let extra = prefill.map(|limits| limits.validate()).transpose()
            .map_err(|_| CandidateError::Arguments)?.unwrap_or(0);
        let peak = compute::request_peak_bytes(limits, native, self.args.max_result_bytes as u64,
            self.args.max_candidate_tokens as u64, 0, 0)?
            .checked_add(extra).ok_or(CandidateError::Arguments)?;
        if peak > limits.memory_bytes { return Err(CandidateError::Arguments); }
        Ok((common, limits))
    }
    pub(super) fn execute(self, input: &mut impl Read, output: &mut impl Write) -> Result<(), CandidateError> {
        let (common, limits) = self.validate()?;
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
