//! Explicit experimental extraction plus the same-model faithfulness reader.
//! Separate command selection cannot enable diagnostics on ordinary extraction.
use super::*;
use crate::{
    native_engine::lmhead::scoring::ScoringLimits,
    tasks::{extract::semantic::grouped_int4::Int4SemanticOptions,
        ir::TaskBudget, judge::JudgeLimits},
};

const OPTIONS_BYTES: usize = 1024 * 1024;

#[cfg(feature = "asupersync-runtime")]
mod runtime;
#[cfg(test)]
mod tests;

#[derive(Clone, Eq, PartialEq)]
pub(super) struct SemanticArgs {
    extraction: extract::ExtractionArgs,
    semantic_options: PathBuf,
}
impl std::fmt::Debug for SemanticArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("SemanticArgs") }
}

pub(super) fn definition() -> clap::Command {
    extract::definition().name("extract-semantic")
        .about("Experimentally review extracted scalar claims with the same INT4 model")
        .long_about("Extract schema-valid JSON from one exact source, then run the existing full-source faithfulness judge for every configured non-verbatim scalar claim. --semantic-options is a complete Int4SemanticOptions JSON object with experimental_opt_in=true, schema-aware claim rules, an explicit judge policy/budget, semantic limits and five-axis whole-request native work limits. The original extraction is retained unchanged alongside entailed/contradicted/unsupported/not_checked diagnostics. Null, verbatim, unmapped and empty values are not_checked; absent optional properties are not invented. Every dynamic field judge is planned and preflighted before the first second-reader forward. The maximum original extraction reservation plus all field judgments share one nonrenewable work allowance. Failure yields no combined result. This is an uncalibrated same-model-correlated diagnostic, not independent proof, automatic acceptance or field repair. No sampling, thinking, tools, retry, durable resume, inference network, release activation or quality claim. Source conversion and digest-pinned snapshots share the actual native path. Optional snapshot save occurs only after fixed-policy and memory admission; dynamic claims are available only after extraction.")
        .arg(clap::Arg::new("semantic_options").long("semantic-options").required(true)
            .value_name("FILE").value_parser(clap::value_parser!(PathBuf))
            .help("Complete explicit experimental Int4SemanticOptions JSON; local file, never stdin"))
        .mut_arg("max_result_bytes", |arg| arg.help("Original extraction result cap; combined output cap is semantic limits.max_output_bytes"))
}
impl SemanticArgs {
    pub(super) fn from_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let extraction = extract::ExtractionArgs::from_matches(matches)?;
        let semantic_options = matches.get_one::<PathBuf>("semantic_options").cloned().ok_or_else(||
            clap::Error::raw(clap::error::ErrorKind::MissingRequiredArgument, "semantic options file required"))?;
        Ok(Self { extraction, semantic_options })
    }
    pub(super) fn validate(&self, args: &Int4Args) -> Result<Int4Limits, CandidateError> {
        crate::candidate_cli::extract::check_schema_path(&self.semantic_options)?;
        let common = self.extraction.validate(args)?;
        self.storage(args, common, common.result_bytes)
    }
    fn storage(&self, args: &Int4Args, original: Int4Limits, result_bytes: usize)
        -> Result<Int4Limits, CandidateError> {
        let extraction = self.extraction.budget(args, original);
        let extra = sum(&[128 * MIB, crate::candidate_cli::extract::SCHEMA_BYTES as u64 * 16,
            extraction.max_output_bytes * 16, u64::from(extraction.max_grammar_states) * 32,
            OPTIONS_BYTES as u64 * 16])?;
        // Configuration/claim templates are caller-owned private data. The
        // complete combined output, not just the first-stage JSON, is priced.
        args.validate_storage(result_bytes, args.max_new_tokens, original.max_prompt_tokens, 0, extra)
    }
    fn admit_options(&self, args: &Int4Args, common: Int4Limits, options: &Int4SemanticOptions)
        -> Result<Int4Limits, CandidateError> {
        options.validate().map_err(|_| CandidateError::Arguments)?;
        let judge = options.specification.judge_budget;
        let extraction = self.extraction.budget(args, common);
        if judge.max_kv_bytes != common.native.kv_bytes
            || judge.max_input_tokens as usize > args.context_tokens
            || judge.max_output_tokens == 0 || judge.max_output_tokens > 1024
            || judge.max_input_tokens.checked_add(judge.max_output_tokens)
                .and_then(|n| n.checked_sub(1)).is_none_or(|n| n as usize > args.context_tokens)
            || !(1..=65_536).contains(&judge.max_grammar_states)
            || judge.max_output_bytes > options.limits.max_output_bytes
            || extraction.max_output_bytes > options.limits.max_output_bytes
            || extraction.max_output_bytes > options.limits.max_input_json_bytes as u64 {
            return Err(CandidateError::Arguments);
        }
        let bytes = usize::try_from(options.limits.max_output_bytes).map_err(|_| CandidateError::Arguments)?;
        self.storage(args, common, bytes)
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
fn parse_options(text: &str) -> Result<Int4SemanticOptions, CandidateError> {
    if text.len() > OPTIONS_BYTES { return Err(CandidateError::Input); }
    let value = crate::canonjson::parse_str(text).map_err(|_| CandidateError::Input)?;
    let options: Int4SemanticOptions = serde_json::from_value(value).map_err(|_| CandidateError::Input)?;
    options.validate().map_err(|_| CandidateError::Arguments)?;
    Ok(options)
}
fn judge_limits(options: &Int4SemanticOptions) -> JudgeLimits {
    let budget = options.specification.judge_budget;
    let nodes = budget.max_grammar_states as usize;
    let logits = options.limits.max_total_projected_logits.min(options.max_model_work.projected_logits);
    JudgeLimits {
        per_head: ScoringLimits { max_candidates: 3, max_total_tokens: nodes.saturating_sub(1),
            max_nodes: nodes, max_depth: budget.max_output_tokens as usize,
            max_candidate_id_bytes: 64, max_projected_logits: logits },
        max_total_prompt_tokens: options.limits.max_total_prompt_tokens,
        max_total_candidate_tokens: nodes, max_total_projected_logits: logits,
        max_output_bytes: budget.max_output_bytes,
    }
}
